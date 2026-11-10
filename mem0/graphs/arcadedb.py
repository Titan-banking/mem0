"""ArcadeDB graph enrichment provider.

Stores relations extracted from *persisted* memory text, owned per memory id, and reads them back for the memory
ids a result set actually contains. Uses SQL over ArcadeDB's HTTP API with bound parameters only.

Cross-store atomicity (vector store vs graph) is NOT provided: every failure surfaces as ``DatabaseError`` with the
affected memory ids and the operation stage so callers can retry explicitly.
"""

import json
import logging
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import httpx

from mem0.exceptions import DatabaseError, LLMError
from mem0.memory.utils import ensure_json_instruction, extract_json, remove_code_blocks, remove_spaces_from_entities

logger = logging.getLogger(__name__)

_IDENTITY = ("user_id", "agent_id", "run_id")
_ID_BATCH = 100
_PAGE = 500
_SESSION_EXPIRED = "arcadedb-session-expired"

_RELATION_EXTRACTION_PROMPT = """You extract graph relations from memory facts and reply with JSON only.

You receive a JSON list of memories, each {"id": ..., "text": ...}. For every memory, list the relations that its
text states explicitly as {"source": ..., "relationship": ..., "destination": ...}.

Rules:
- Use only the supplied text and only the supplied ids. Never infer relations that are not stated.
- Return exactly one entry per supplied id. A memory with no stated relation gets an empty "relations" array.
- Reply as JSON in exactly this shape:
{"memories": [{"memory_id": "<one supplied id>", "relations": [{"source": "alice", "relationship": "works_at", "destination": "acme"}]}]}
"""

# Fixed schema; names are never derived from LLM output.
_ENTITY_PROPS = ("namespace", "scope_id", "name", "user_id", "agent_id", "run_id")
_RELATION_PROPS = ("namespace", "scope_id", "memory_id", "relation_key", "source", "relationship", "destination",
                   "user_id", "agent_id", "run_id")
_RELATION_INDEXES = {
    ("scope_id", "memory_id", "relation_key"): True,
    ("namespace", "memory_id"): False,
    ("user_id",): False,
    ("agent_id",): False,
    ("run_id",): False,
}


def _db_error(message: str, operation: str, cause: Optional[BaseException] = None, **details) -> DatabaseError:
    err = DatabaseError(message, error_code="DB_001", details={"provider": "arcadedb", "operation": operation, **details})
    if cause is not None:
        err.__cause__ = cause
    return err


class _ArcadeDBClient:
    """Minimal ArcadeDB HTTP client. Credentials are only ever sent as a Basic auth header, never logged."""

    def __init__(self, url: str, database: str, username: str, password: str, timeout: float = 30.0):
        self._base = url.rstrip("/")
        self._db = quote(database, safe="")
        self._http = httpx.Client(auth=(username, password), timeout=timeout, verify=True, follow_redirects=False)

    def close(self) -> None:
        self._http.close()

    def _post(self, kind: str, body: Optional[dict], session_id: Optional[str], operation: str) -> httpx.Response:
        headers = {"arcadedb-session-id": session_id} if session_id else None
        try:
            resp = self._http.post(f"{self._base}/api/v1/{kind}/{self._db}", json=body, headers=headers)
        except httpx.HTTPError as e:
            raise _db_error(f"ArcadeDB request failed ({type(e).__name__})", operation, e) from e
        if _SESSION_EXPIRED in resp.headers:
            raise _db_error("ArcadeDB transaction session expired", operation, status=resp.status_code)
        return resp

    def _result(self, resp: httpx.Response, operation: str) -> List[Any]:
        try:
            data = resp.json()
        except ValueError as e:
            raise _db_error("ArcadeDB returned a non-JSON response", operation, e, status=resp.status_code) from e
        if resp.status_code != 200 or not isinstance(data, dict) or "error" in data:
            detail = data.get("detail") or data.get("error") if isinstance(data, dict) else None
            raise _db_error(f"ArcadeDB rejected the request (HTTP {resp.status_code}): {detail}", operation,
                            status=resp.status_code)
        result = data.get("result")
        if not isinstance(result, list):
            raise _db_error("ArcadeDB response is missing a result list", operation, status=resp.status_code)
        if data.get("truncated"):
            raise _db_error("ArcadeDB truncated the result set; refusing to return incomplete data", operation)
        return result

    def query(self, sql: str, params: Optional[dict] = None, session_id: Optional[str] = None, *,
              limit: int = _PAGE, operation: str = "query") -> List[Any]:
        body: Dict[str, Any] = {"language": "sql", "command": sql, "limit": limit}
        if params:
            body["params"] = params
        return self._result(self._post("query", body, session_id, operation), operation)

    def command(self, sql: str, params: Optional[dict] = None, session_id: Optional[str] = None, *,
                language: str = "sql", operation: str = "command") -> List[Any]:
        body: Dict[str, Any] = {"language": language, "command": sql}
        if params:
            body["params"] = params
        return self._result(self._post("command", body, session_id, operation), operation)

    @contextmanager
    def transaction(self, operation: str = "transaction"):
        resp = self._post("begin", None, None, operation)
        sid = resp.headers.get("arcadedb-session-id")
        if resp.status_code != 204 or not sid:
            raise _db_error(f"ArcadeDB begin failed (HTTP {resp.status_code})", operation)
        try:
            yield sid
        except BaseException:
            try:  # best effort; never mask the original failure
                self._post("rollback", None, sid, operation)
            except Exception:  # noqa: BLE001
                logger.debug("ArcadeDB rollback failed", exc_info=True)
            raise
        resp = self._post("commit", None, sid, operation)  # outcome of a failed commit is unknown: never replayed
        if resp.status_code != 204:
            raise _db_error(f"ArcadeDB commit failed (HTTP {resp.status_code}); outcome unknown", operation)


def _canon(*parts: Any) -> str:
    return json.dumps(list(parts), ensure_ascii=False, separators=(",", ":"))


def _aggregate(rows: Iterable[Tuple[str, str, str, str]]) -> List[dict]:
    """(source, relationship, destination, memory_id) -> aggregated, deterministically sorted relations."""
    owners: Dict[Tuple[str, str, str], set] = {}
    for s, r, d, mid in rows:
        owners.setdefault((s, r, d), set()).add(mid)
    return [{"source": s, "relationship": r, "destination": d, "memory_ids": sorted(ids)}
            for (s, r, d), ids in sorted(owners.items())]


class ArcadeDBGraph:
    def __init__(self, config, *, llm, namespace: str):
        self.config = config
        self.llm = llm
        self.namespace = namespace
        self.client = _ArcadeDBClient(config.url, config.database, config.username,
                                      config.password.get_secret_value(), config.timeout)
        self._schema_lock = threading.Lock()
        self._schema_ready = False

    def close(self) -> None:
        self.client.close()

    # ------------------------------------------------------------------ schema
    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            c = self.client.command
            op = "ensure_schema"
            c("CREATE VERTEX TYPE Mem0Entity IF NOT EXISTS", operation=op)
            for p in _ENTITY_PROPS:
                c(f"CREATE PROPERTY Mem0Entity.{p} IF NOT EXISTS STRING", operation=op)
            c("CREATE INDEX IF NOT EXISTS ON Mem0Entity (scope_id, name) UNIQUE", operation=op)
            c("CREATE EDGE TYPE Mem0Relation IF NOT EXISTS", operation=op)
            for p in _RELATION_PROPS:
                c(f"CREATE PROPERTY Mem0Relation.{p} IF NOT EXISTS STRING", operation=op)
            for cols, unique in _RELATION_INDEXES.items():
                c(f"CREATE INDEX IF NOT EXISTS ON Mem0Relation ({', '.join(cols)}) {'UNIQUE' if unique else 'NOTUNIQUE'}",
                  operation=op)
            self._validate_schema()
            self._schema_ready = True

    def _validate_schema(self) -> None:
        rows = self.client.query("SELECT name, type, properties, indexes FROM schema:types WHERE name IN :n",
                                 {"n": ["Mem0Entity", "Mem0Relation"]}, operation="validate_schema")
        types = {r["name"]: r for r in rows}

        def check(name, kind, props, indexes):
            t = types.get(name)
            if not t or t.get("type") != kind:
                raise _db_error(f"Incompatible ArcadeDB schema: {name} must be a {kind} type", "validate_schema")
            have = {p["name"]: p.get("type") for p in t.get("properties", [])}
            for p in props:
                if have.get(p) != "STRING":
                    raise _db_error(f"Incompatible ArcadeDB schema: {name}.{p} must be STRING (found {have.get(p)})",
                                    "validate_schema")
            idx = {tuple(i.get("properties", [])): bool(i.get("unique")) for i in t.get("indexes", [])}
            for cols, unique in indexes.items():
                if idx.get(cols) is not unique:
                    raise _db_error(
                        f"Incompatible ArcadeDB schema: {name} index {list(cols)} must be "
                        f"{'UNIQUE' if unique else 'NOTUNIQUE'}", "validate_schema")

        check("Mem0Entity", "vertex", _ENTITY_PROPS, {("scope_id", "name"): True})
        check("Mem0Relation", "edge", _RELATION_PROPS, _RELATION_INDEXES)

    # ------------------------------------------------------------------ scope helpers
    def _scope(self, scope: dict) -> Tuple[str, dict]:
        ids = {k: scope.get(k) for k in _IDENTITY}
        if not any(ids.values()):
            raise ValueError("graph operations require at least one of user_id, agent_id, run_id")
        return _canon(self.namespace, ids["user_id"], ids["agent_id"], ids["run_id"]), ids

    # ------------------------------------------------------------------ extraction
    def _extract(self, memories: List[dict]) -> Dict[str, List[dict]]:
        wanted = [m["id"] for m in memories]
        payload = json.dumps([{"id": m["id"], "text": m.get("memory", m.get("text", ""))} for m in memories],
                             ensure_ascii=False)
        system, user = ensure_json_instruction(_RELATION_EXTRACTION_PROMPT, f"Extract relations from these memories:\n{payload}\n")
        try:
            raw = self.llm.generate_response(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_object"})
            parsed = json.loads(extract_json(remove_code_blocks(raw)))
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"Failed to extract relations: LLM reply was not valid JSON ({type(e).__name__})") from e
        entries = parsed.get("memories") if isinstance(parsed, dict) else None
        if not isinstance(entries, list):
            raise LLMError("Malformed LLM extraction response: expected {'memories': [...]}")
        out: Dict[str, List[dict]] = {}
        for entry in entries:
            mid = entry.get("memory_id") if isinstance(entry, dict) else None
            if mid not in wanted:
                raise LLMError(f"Relation extraction returned an unknown memory_id: {mid!r}")
            if mid in out:
                raise LLMError(f"Relation extraction returned memory_id {mid!r} more than once")
            rels = entry.get("relations")
            if not isinstance(rels, list):
                raise LLMError(f"Relation extraction for memory_id {mid!r} has no relations list")
            for r in rels:
                if not (isinstance(r, dict) and all(isinstance(r.get(k), str) and r.get(k).strip()
                                                    for k in ("source", "relationship", "destination"))):
                    raise LLMError(f"Relation extraction returned an invalid relation for memory_id {mid!r}: {r!r}")
            seen, deduped = set(), []
            for r in remove_spaces_from_entities([dict(x) for x in rels], sanitize_relationship=False):
                key = (r["source"], r["relationship"], r["destination"])
                if key not in seen:
                    seen.add(key)
                    deduped.append({"source": key[0], "relationship": key[1], "destination": key[2]})
            out[mid] = deduped
        missing = [m for m in wanted if m not in out]
        if missing:
            raise LLMError(f"Relation extraction omitted memory_id(s): {missing}")
        return out

    # ------------------------------------------------------------------ writes
    def _store(self, sid: str, memory_id: str, relations: List[dict], scope_id: str, ids: dict) -> None:
        """Replace one memory's owned edges inside an open transaction."""
        self._remove_owned(sid, memory_id, scope_id)
        base = {"namespace": self.namespace, "scope_id": scope_id, **ids}
        for r in relations:
            for name in (r["source"], r["destination"]):
                self.client.command(
                    "UPDATE Mem0Entity SET namespace = :namespace, scope_id = :scope_id, name = :name, "
                    "user_id = :user_id, agent_id = :agent_id, run_id = :run_id "
                    "UPSERT WHERE scope_id = :scope_id AND name = :name", {**base, "name": name}, sid)
            # Never CREATE EDGE ... IF NOT EXISTS: it compares endpoints, not memory ownership or predicate.
            self.client.command(
                "CREATE EDGE Mem0Relation FROM (SELECT FROM Mem0Entity WHERE scope_id = :scope_id AND name = :src) "
                "TO (SELECT FROM Mem0Entity WHERE scope_id = :scope_id AND name = :dst) "
                "SET namespace = :namespace, scope_id = :scope_id, memory_id = :memory_id, relation_key = :rk, "
                "source = :src, relationship = :rel, destination = :dst, "
                "user_id = :user_id, agent_id = :agent_id, run_id = :run_id",
                {**base, "memory_id": memory_id, "src": r["source"], "dst": r["destination"], "rel": r["relationship"],
                 "rk": _canon(r["source"], r["relationship"], r["destination"])}, sid)

    def _remove_owned(self, sid: str, memory_id: str, scope_id: str) -> None:
        """Delete a memory's edges and prune only the endpoints those edges touched, if left without edges."""
        touched = set()
        for row in self.client.query(
                "SELECT source, destination FROM Mem0Relation WHERE scope_id = :s AND memory_id = :m",
                {"s": scope_id, "m": memory_id}, sid, operation="remove_owned"):
            touched.update((row["source"], row["destination"]))
        if not touched:
            return
        self.client.command("DELETE FROM Mem0Relation WHERE scope_id = :s AND memory_id = :m",
                            {"s": scope_id, "m": memory_id}, sid)
        for name in sorted(touched):
            self.client.command(
                "DELETE FROM Mem0Entity WHERE scope_id = :s AND name = :n AND bothE('Mem0Relation').size() = 0",
                {"s": scope_id, "n": name}, sid)

    def add(self, memories: List[dict], scope: dict) -> List[dict]:
        """Extract + store relations for already persisted memories. Returns aggregated relations for those ids."""
        if not memories:
            return []
        scope_id, ids = self._scope(scope)
        affected = [m["id"] for m in memories]
        extracted = self._extract(memories)  # before any transaction: sessions idle-expire after ~5s
        self._ensure_schema()
        rows = []
        try:
            for mid in affected:
                with self.client.transaction("add") as sid:
                    self._store(sid, mid, extracted[mid], scope_id, ids)
                rows += [(r["source"], r["relationship"], r["destination"], mid) for r in extracted[mid]]
        except DatabaseError as e:
            e.details.update({"memory_ids": affected, "stage": "add", "vector_persisted": True})
            raise
        return _aggregate(rows)

    def replace(self, memory_id: str, text: str, scope: dict) -> None:
        scope_id, ids = self._scope(scope)
        extracted = self._extract([{"id": memory_id, "memory": text}])[memory_id]  # validated before deleting anything
        self._ensure_schema()
        try:
            with self.client.transaction("replace") as sid:
                self._store(sid, memory_id, extracted, scope_id, ids)
        except DatabaseError as e:
            e.details.update({"memory_ids": [memory_id], "stage": "replace", "vector_persisted": True})
            raise

    def delete(self, memory_id: str, scope: dict) -> None:
        scope_id, _ = self._scope(scope)
        self._ensure_schema()
        try:
            with self.client.transaction("delete") as sid:
                self._remove_owned(sid, memory_id, scope_id)
        except DatabaseError as e:
            e.details.update({"memory_ids": [memory_id], "stage": "delete", "vector_persisted": True})
            raise

    def _bulk_delete(self, where: str, params: dict, stage: str) -> None:
        try:
            for table, extra in (("Mem0Relation", ""), ("Mem0Entity", " AND bothE('Mem0Relation').size() = 0")):
                while True:
                    res = self.client.command(f"DELETE FROM {table} WHERE {where}{extra} LIMIT {_PAGE}", params,
                                              operation=stage)
                    if not res or not res[0].get("count"):
                        break
        except DatabaseError as e:
            e.details["stage"] = stage
            raise

    def delete_all(self, scope: dict) -> None:
        """Remove every relation whose identity matches ALL supplied fields (unspecified fields are wildcards)."""
        self._scope(scope)  # requires at least one identity field
        self._ensure_schema()
        where, params = ["namespace = :namespace"], {"namespace": self.namespace}
        for k in _IDENTITY:
            if scope.get(k):
                where.append(f"{k} = :{k}")
                params[k] = scope[k]
        self._bulk_delete(" AND ".join(where), params, "delete_all")

    def reset(self) -> None:
        """Remove everything in this namespace only; never touches schema, other namespaces, or other types."""
        self._ensure_schema()
        self._bulk_delete("namespace = :namespace", {"namespace": self.namespace}, "reset")

    # ------------------------------------------------------------------ reads
    def get_relations(self, memory_ids: List[str]) -> List[dict]:
        ids = list(dict.fromkeys(memory_ids))
        if not ids:
            return []
        self._ensure_schema()
        rows: List[Tuple[str, str, str, str]] = []
        for i in range(0, len(ids), _ID_BATCH):
            batch, last = ids[i:i + _ID_BATCH], None
            while True:  # keyset pagination; completeness is "an empty page", never "a short page"
                sql = ("SELECT source, relationship, destination, memory_id, scope_id, relation_key FROM Mem0Relation "
                       "WHERE namespace = :ns AND memory_id IN :ids")
                params: Dict[str, Any] = {"ns": self.namespace, "ids": batch}
                if last:
                    sql += (" AND (scope_id > :ls OR (scope_id = :ls AND memory_id > :lm) OR "
                            "(scope_id = :ls AND memory_id = :lm AND relation_key > :lr))")
                    params.update({"ls": last[0], "lm": last[1], "lr": last[2]})
                sql += f" ORDER BY scope_id, memory_id, relation_key LIMIT {_PAGE}"
                # HTTP cap sits above the SQL page so a legitimately full page is never mistaken for truncation
                page = self.client.query(sql, params, limit=_PAGE + 1, operation="get_relations")
                if not page:
                    break
                rows += [(r["source"], r["relationship"], r["destination"], r["memory_id"]) for r in page]
                last = (page[-1]["scope_id"], page[-1]["memory_id"], page[-1]["relation_key"])
        return _aggregate(rows)
