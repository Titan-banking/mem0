"""Live ArcadeDB tests. Skipped unless MEM0_ARCADEDB_TEST_* variables are set (the smoke runner sets them).

Real components: Qdrant (local path), SQLite history, ArcadeDB over HTTP.
Fakes only at external boundaries: the LLM and the embedder (deterministic, hash-seeded).
"""

import asyncio
import hashlib
import json
import os
import re
import uuid


import pytest

from mem0.exceptions import DatabaseError, LLMError

np = pytest.importorskip("numpy")  # only the live-DB test needs it
URL = os.environ.get("MEM0_ARCADEDB_TEST_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set MEM0_ARCADEDB_TEST_URL/DATABASE/USERNAME/PASSWORD to run")

DIMS = 16
REL_RE = re.compile(r"^(?P<s>\w+) (?P<r>[\w ]+?) (?P<d>\w+)$")


class FakeLLM:
    """Memory-extraction prompt -> queued memories; relation prompt -> parse 'X works at Y' facts."""

    def __init__(self):
        self.queue = []
        self.relation_calls = 0
        self.raw_relation_reply = None

    def generate_response(self, messages, response_format=None, **kw):
        system = messages[0]["content"]
        if "graph relations" in system:
            self.relation_calls += 1
            if self.raw_relation_reply is not None:
                return self.raw_relation_reply
            items = json.loads(messages[1]["content"].split("memories:\n", 1)[1].split("\n", 1)[0])
            out = []
            for it in items:
                rels = []
                for m in re.finditer(r"(\w+) (works at|lives in) (\w+)", it["text"]):
                    rels.append({"source": m.group(1).lower(), "relationship": m.group(2).replace(" ", "_"),
                                 "destination": m.group(3).lower()})
                out.append({"memory_id": it["id"], "relations": rels})
            return json.dumps({"memories": out})
        return json.dumps({"memory": [{"text": t} for t in self.queue.pop(0)]})


class FakeEmbedder:
    def embed(self, text, memory_action=None):
        seed = int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")
        v = np.random.default_rng(seed).standard_normal(DIMS)
        return (v / np.linalg.norm(v)).tolist()

    def embed_batch(self, texts, memory_action=None):
        return [self.embed(t) for t in texts]


def _config(tmp_path, graph=True, namespace=None, password=None, collection=None):
    cfg = {
        "vector_store": {"provider": "qdrant", "config": {
            "collection_name": collection or f"c_{uuid.uuid4().hex[:8]}", "path": str(tmp_path / "qdrant"),
            "embedding_model_dims": DIMS, "on_disk": True}},
        "llm": {"provider": "openai", "config": {"api_key": "unused"}},
        "embedder": {"provider": "openai", "config": {"api_key": "unused", "embedding_dims": DIMS}},
        "history_db_path": str(tmp_path / "history.db"),
    }
    if graph:
        cfg["graph_store"] = {"provider": "arcadedb", "config": {
            "url": URL, "database": os.environ["MEM0_ARCADEDB_TEST_DATABASE"],
            "username": os.environ["MEM0_ARCADEDB_TEST_USERNAME"],
            "password": password or os.environ["MEM0_ARCADEDB_TEST_PASSWORD"],
            **({"namespace": namespace} if namespace else {})}}
    return cfg


def _memory(tmp_path, **kw):
    from mem0 import Memory
    m = Memory.from_config(_config(tmp_path, **kw))
    m.llm, m.embedding_model = FakeLLM(), FakeEmbedder()
    if m.graph is not None:
        m.graph.llm = m.llm
    return m


def _add(m, text, **scope):
    m.llm.queue.append([text])
    return m.add(text, **scope)


def _rels(resp):
    return {(r["source"], r["relationship"], r["destination"]): r["memory_ids"] for r in resp["relations"]}


@pytest.fixture(autouse=True)
def _isolation(monkeypatch, tmp_path):
    monkeypatch.setenv("MEM0_TELEMETRY", "False")
    monkeypatch.setenv("MEM0_DIR", str(tmp_path / "mem0dir"))


def test_tenants_with_same_endpoint_names_stay_isolated(tmp_path):
    m = _memory(tmp_path)
    try:
        a = _add(m, "Alice works at Acme", user_id="u1")
        _add(m, "Alice lives in Paris", user_id="u2")
        ra = m.search("Alice works at Acme", filters={"user_id": "u1"})
        rb = m.search("Alice lives in Paris", filters={"user_id": "u2"})
        assert _rels(a) == {("alice", "works_at", "acme"): [a["results"][0]["id"]]}  # add returns relations too
        assert set(_rels(ra)) == {("alice", "works_at", "acme")}
        assert set(_rels(rb)) == {("alice", "lives_in", "paris")}
        assert _rels(ra)[("alice", "works_at", "acme")] == [a["results"][0]["id"]]
        assert set(_rels(m.get_all(filters={"user_id": "u2"}))) == {("alice", "lives_in", "paris")}
        got = m.get(a["results"][0]["id"])
        assert got["relations"] == [{"source": "alice", "relationship": "works_at", "destination": "acme",
                                     "memory_ids": [a["results"][0]["id"]]}]
        assert m.get("does-not-exist") is None
    finally:
        m.graph.reset()
        m.close()


def test_shared_support_survives_delete_and_replace_removes_old_support(tmp_path):
    m = _memory(tmp_path)
    try:
        first = _add(m, "Alice works at Acme", user_id="u1")["results"][0]["id"]
        second = _add(m, "Alice works at Acme today", user_id="u1")["results"][0]["id"]
        both = _rels(m.get_all(filters={"user_id": "u1"}))
        assert sorted(both[("alice", "works_at", "acme")]) == sorted([first, second])
        m.delete(first)
        assert _rels(m.get_all(filters={"user_id": "u1"}))[("alice", "works_at", "acme")] == [second]
        m.update(second, "Alice works at Beta")
        after = _rels(m.get_all(filters={"user_id": "u1"}))
        assert ("alice", "works_at", "acme") not in after and after[("alice", "works_at", "beta")] == [second]
    finally:
        m.graph.reset()
        m.close()


def test_reopened_memory_cleans_up_scoped_and_other_scope_survives(tmp_path):
    ns = f"ns_{uuid.uuid4().hex[:6]}"
    m = _memory(tmp_path, namespace=ns)
    _add(m, "Alice works at Acme", user_id="u1")
    _add(m, "Bob works at Initech", user_id="u2")
    m.close()
    (tmp_path / "second").mkdir()
    m2 = _memory(tmp_path / "second", namespace=ns)  # fresh graph client, no cached schema/state
    try:
        m2.graph.delete_all({"user_id": "u1"})
        assert m2.graph.get_relations(["x"]) == []
        left = [r for r in m2.graph.client.query("SELECT source FROM Mem0Relation WHERE namespace = :n", {"n": ns})]
        assert [r["source"] for r in left] == ["bob"]
        m2.graph.reset()
        assert m2.graph.client.query("SELECT count(*) AS c FROM Mem0Relation WHERE namespace = :n", {"n": ns})[0]["c"] == 0
    finally:
        m2.close()


def test_two_namespaces_in_one_database_do_not_mix(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    # same vector collection name, different datasets: only the configured graph namespace keeps them apart
    m1 = _memory(tmp_path / "a", namespace="nsA", collection="shared")
    m2 = _memory(tmp_path / "b", namespace="nsB", collection="shared")
    try:
        a = _add(m1, "Alice works at Acme", user_id="u1")["results"][0]["id"]
        _add(m2, "Alice works at Zeta", user_id="u1")
        assert set(_rels(m1.get_all(filters={"user_id": "u1"}))) == {("alice", "works_at", "acme")}
        m1.graph.reset()
        assert set(_rels(m2.get_all(filters={"user_id": "u1"}))) == {("alice", "works_at", "zeta")}
        assert m1.graph.get_relations([a]) == []
    finally:
        m2.graph.reset()
        m1.close()
        m2.close()


def test_disabled_graph_keeps_exact_result_shape(tmp_path):
    m = _memory(tmp_path, graph=False)
    try:
        r = _add(m, "Alice works at Acme", user_id="u1")
        assert set(r) == {"results"} and "relations" not in r["results"][0]
        assert "relations" not in m.get_all(filters={"user_id": "u1"})
        assert "relations" not in m.get(r["results"][0]["id"])
    finally:
        m.close()


def test_infer_false_skips_extraction_and_metadata_update_does_not_reextract(tmp_path):
    m = _memory(tmp_path)
    try:
        raw = m.add("Alice works at Acme", user_id="u1", infer=False)
        assert m.llm.relation_calls == 0 and raw["relations"] == []
        mid = _add(m, "Carol works at Dune", user_id="u1")["results"][0]["id"]
        calls = m.llm.relation_calls
        m.update(mid, "Carol works at Dune", metadata={"k": "v"})
        assert m.llm.relation_calls == calls
    finally:
        m.graph.reset()
        m.close()


def test_expired_memory_relations_hidden_until_show_expired(tmp_path):
    m = _memory(tmp_path)
    try:
        m.llm.queue.append(["Alice works at Acme"])
        m.add("Alice works at Acme", user_id="u1", expiration_date="2000-01-01")
        assert _rels(m.get_all(filters={"user_id": "u1"})) == {}
        assert ("alice", "works_at", "acme") in _rels(m.get_all(filters={"user_id": "u1"}, show_expired=True))
    finally:
        m.graph.reset()
        m.close()


def test_names_with_quotes_unicode_and_injection_text_are_stored_verbatim(tmp_path):
    m = _memory(tmp_path)
    try:
        m.llm.queue.append(["x"])
        mid = m.add("x", user_id="u1", infer=False)["results"][0]["id"]
        weird = "o'brien\"; DROP TYPE Mem0Entity; --é日本"
        m.llm.raw_relation_reply = json.dumps({"memories": [{"memory_id": mid, "relations": [
            {"source": weird, "relationship": "knows", "destination": "ü"}]}]})
        m.graph.replace(mid, "whatever", {"user_id": "u1"})
        norm = weird.lower().replace(" ", "_")  # the shared normaliser lowercases and swaps spaces for underscores
        assert _rels(m.get_all(filters={"user_id": "u1"}))[(norm, "knows", "ü")] == [mid]
    finally:
        m.graph.reset()
        m.close()


@pytest.mark.parametrize("reply, match", [
    ("not json", "extract"),
    (json.dumps({"nope": []}), "Malformed"),
    (json.dumps({"memories": [{"memory_id": "unknown", "relations": []}]}), "memory"),
    (json.dumps({"memories": [{"memory_id": "<id>", "relations": [{"source": 1, "relationship": "r", "destination": "d"}]}]}), "relation"),
    (json.dumps({"memories": []}), "memory"),
])
def test_bad_extraction_is_an_error_not_an_empty_success(tmp_path, reply, match):
    m = _memory(tmp_path)
    try:
        m.llm.queue.append(["Alice works at Acme"])
        orig = m.llm.generate_response

        def patched(messages, **kw):
            if "graph relations" in messages[0]["content"]:
                items = json.loads(messages[1]["content"].split("memories:\n", 1)[1].split("\n", 1)[0])
                return reply.replace("<id>", items[0]["id"])
            return orig(messages, **kw)

        m.llm.generate_response = patched
        with pytest.raises(LLMError, match=match):
            m.add("Alice works at Acme", user_id="u1")
    finally:
        m.close()


def test_bad_credentials_and_missing_database_raise_without_leaking_password(tmp_path):
    m = _memory(tmp_path, password="definitely-wrong-pw")
    try:
        m.llm.queue.append(["Alice works at Acme"])
        with pytest.raises(DatabaseError) as ei:
            m.add("Alice works at Acme", user_id="u1")
        assert "definitely-wrong-pw" not in str(ei.value) and "definitely-wrong-pw" not in repr(ei.value.details)
    finally:
        m.close()


def test_mid_replacement_failure_rolls_back_to_previous_state(tmp_path):
    m = _memory(tmp_path)
    try:
        mid = _add(m, "Alice works at Acme", user_id="u1")["results"][0]["id"]
        real = m.graph.client.command
        n = {"c": 0}

        def flaky(sql, params=None, session_id=None, **kw):
            if session_id:
                n["c"] += 1
                if n["c"] == 3:  # after delete + first upsert inside the transaction
                    raise DatabaseError("injected failure", error_code="DB_001")
            return real(sql, params, session_id, **kw)

        m.graph.client.command = flaky
        with pytest.raises(DatabaseError):
            m.graph.replace(mid, "Alice works at Beta", {"user_id": "u1"})
        m.graph.client.command = real
        assert set(_rels(m.get_all(filters={"user_id": "u1"}))) == {("alice", "works_at", "acme")}
    finally:
        m.graph.reset()
        m.close()


def test_async_operations_keep_event_loop_responsive(tmp_path):
    from mem0 import AsyncMemory

    async def run():
        m = AsyncMemory.from_config(_config(tmp_path))
        m.llm, m.embedding_model = FakeLLM(), FakeEmbedder()
        m.graph.llm = m.llm
        beats = []

        async def heartbeat():
            while True:
                beats.append(1)
                await asyncio.sleep(0.005)

        hb = asyncio.create_task(heartbeat())
        try:
            m.llm.queue.append(["Alice works at Acme"])
            r = await m.add("Alice works at Acme", user_id="u1")
            assert _rels(r) == {("alice", "works_at", "acme"): [r["results"][0]["id"]]}
            s = await m.search("Alice works at Acme", filters={"user_id": "u1"})
            assert ("alice", "works_at", "acme") in _rels(s)
            assert len(beats) > 3
        finally:
            hb.cancel()
            await asyncio.to_thread(m.graph.reset)
            m.close()

    asyncio.run(run())


def test_get_relations_pages_through_more_than_one_page_without_loss(tmp_path):
    m = _memory(tmp_path)
    try:
        m.llm.queue.append(["x"])
        mid = m.add("x", user_id="u1", infer=False)["results"][0]["id"]
        n = 650  # > the 500-row page: a short page must not be mistaken for completion
        m.llm.raw_relation_reply = json.dumps({"memories": [{"memory_id": mid, "relations": [
            {"source": f"s{i:04d}", "relationship": "knows", "destination": f"d{i:04d}"} for i in range(n)]}]})
        m.graph.replace(mid, "many", {"user_id": "u1"})
        got = m.graph.get_relations([mid])
        assert len(got) == n and got == sorted(got, key=lambda r: (r["source"], r["relationship"], r["destination"]))
        assert all(r["memory_ids"] == [mid] for r in got)
    finally:
        m.graph.reset()
        m.close()
