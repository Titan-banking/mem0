"""Protocol-boundary tests for the ArcadeDB HTTP client and config. No database needed."""

import json

import httpx
import pytest
from pydantic import ValidationError

from mem0.exceptions import DatabaseError
from mem0.graphs.arcadedb import _ArcadeDBClient, _aggregate
from mem0.graphs.configs import ArcadeDBConfig, GraphStoreConfig

PASSWORD = "s3cret-pw"


def _client(handler):
    c = _ArcadeDBClient("http://db:2480/", "my db/1", "root", PASSWORD, 5)
    c._http = httpx.Client(transport=httpx.MockTransport(handler), auth=("root", PASSWORD))
    return c


def test_database_name_is_one_url_path_component_and_values_are_bound():
    seen = {}

    def handler(req):
        seen["path"], seen["body"] = req.url.raw_path.decode(), json.loads(req.content)
        return httpx.Response(200, json={"result": []})

    _client(handler).query("SELECT FROM T WHERE a = :a", {"a": "x'; DROP"})
    assert seen["path"] == "/api/v1/query/my%20db%2F1"
    assert seen["body"]["params"] == {"a": "x'; DROP"} and "DROP" not in seen["body"]["command"]


@pytest.mark.parametrize("response", [
    httpx.Response(401, json={"error": "Unauthorized"}),
    httpx.Response(200, json={"error": "boom", "detail": "bad"}),   # error envelope on an unexpected 2xx
    httpx.Response(200, text="<html>not json</html>"),
    httpx.Response(200, json={"result": "not-a-list"}),
    httpx.Response(200, json={"result": [], "truncated": True}),
    httpx.Response(200, json={"result": []}, headers={"arcadedb-session-expired": "true"}),
])
def test_bad_responses_are_errors_and_never_leak_credentials(response):
    c = _client(lambda req: response)
    with pytest.raises(DatabaseError) as ei:
        c.query("SELECT 1")
    assert ei.value.error_code == "DB_001" and ei.value.details["provider"] == "arcadedb"
    assert PASSWORD not in str(ei.value) and PASSWORD not in json.dumps(ei.value.details, default=str)


def test_transport_failure_keeps_cause_and_does_not_leak_password():
    def handler(req):
        raise httpx.ConnectError("connection refused", request=req)

    with pytest.raises(DatabaseError) as ei:
        _client(handler).query("SELECT 1")
    assert isinstance(ei.value.__cause__, httpx.ConnectError) and PASSWORD not in str(ei.value)


def _tx_handler(log, commit_status=204):
    def handler(req):
        kind = req.url.path.split("/")[3]
        log.append((kind, req.headers.get("arcadedb-session-id")))
        if kind == "begin":
            return httpx.Response(204, headers={"arcadedb-session-id": "AS-1"})
        if kind == "commit":
            return httpx.Response(commit_status)
        return httpx.Response(204 if kind == "rollback" else 200, json={"result": []})
    return handler


def test_session_id_is_sent_per_request_and_rolled_back_on_failure_without_masking_it():
    log = []
    c = _client(_tx_handler(log))
    with pytest.raises(RuntimeError, match="boom"):
        with c.transaction() as sid:
            c.command("DELETE FROM T", session_id=sid)
            raise RuntimeError("boom")
    assert [k for k, _ in log] == ["begin", "command", "rollback"]
    assert all(s == "AS-1" for k, s in log if k != "begin")
    assert "arcadedb-session-id" not in c._http.headers  # never stored on the shared client


def test_failed_commit_is_reported_as_unknown_outcome_and_is_not_replayed():
    log = []
    c = _client(_tx_handler(log, commit_status=500))
    with pytest.raises(DatabaseError, match="outcome unknown"):
        with c.transaction():
            pass
    assert [k for k, _ in log].count("commit") == 1 and "rollback" not in [k for k, _ in log]


def test_aggregate_merges_owners_and_sorts_deterministically():
    rows = [("b", "r", "c", "m2"), ("a", "r", "c", "m2"), ("a", "r", "c", "m1"), ("a", "r", "c", "m1")]
    assert _aggregate(rows) == [
        {"source": "a", "relationship": "r", "destination": "c", "memory_ids": ["m1", "m2"]},
        {"source": "b", "relationship": "r", "destination": "c", "memory_ids": ["m2"]},
    ]


@pytest.mark.parametrize("bad", [
    {"timeout": 0}, {"timeout": -1}, {"bogus": 1}, {"url": None},
])
def test_config_rejects_invalid_or_unknown_fields(bad):
    base = {"url": "http://x", "database": "d", "username": "u", "password": "p"}
    with pytest.raises(ValidationError):
        ArcadeDBConfig(**{**base, **bad})


def test_graph_store_config_rejects_unknown_provider_and_hides_password():
    with pytest.raises(ValidationError):
        GraphStoreConfig(provider="neo4j", config={"url": "u", "database": "d", "username": "u", "password": "p"})
    cfg = GraphStoreConfig(config={"url": "http://x", "database": "d", "username": "u", "password": PASSWORD})
    assert PASSWORD not in repr(cfg) and PASSWORD not in cfg.model_dump_json()
