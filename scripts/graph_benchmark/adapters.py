"""Benchmark adapters for ArcadeDB (HTTP SQL) and FalkorDB (RESP/Cypher).

Every query method returns plain Python data so the runner stays engine-agnostic:
  * vector queries -> list[(id: str, similarity: float)] ordered best-first
  * graph queries  -> list[str] of ids
Similarity is normalised as ``1 - distance`` for both engines.
"""

import time

from falkordb import FalkorDB

from mem0.graphs.arcadedb import _ArcadeDBClient

VECTOR_DIM = 384
INDEX_READY_DEADLINE_S = 1800


class ArcadeAdapter:
    name = "arcadedb"

    def __init__(self, url, database, username, password):
        self.client = _ArcadeDBClient(url, database, username, password, timeout=30)
        self.url, self.database = url, database
        self.ef_search, self.rerank = 100, None

    def close(self):
        self.client.close()

    def wait_ready(self, timeout=120):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self.client.query("SELECT 1")
                return True
            except Exception:
                time.sleep(1)
        return False

    # -- schema / load -------------------------------------------------
    def load_schema(self):
        c = self.client.command
        c("CREATE VERTEX TYPE BenchEntity IF NOT EXISTS")
        for p, t in (("id", "STRING"), ("tenant", "STRING"), ("embedding", "ARRAY_OF_FLOATS")):
            c(f"CREATE PROPERTY BenchEntity.{p} IF NOT EXISTS {t}")
        c("CREATE EDGE TYPE BenchRelation IF NOT EXISTS")
        for p in ("id", "tenant", "memory_id", "predicate"):
            c(f"CREATE PROPERTY BenchRelation.{p} IF NOT EXISTS STRING")
        c("CREATE INDEX IF NOT EXISTS ON BenchEntity (id) UNIQUE")
        c("CREATE INDEX IF NOT EXISTS ON BenchEntity (tenant) NOTUNIQUE")
        c("CREATE INDEX IF NOT EXISTS ON BenchRelation (id) UNIQUE")
        c("CREATE INDEX IF NOT EXISTS ON BenchRelation (memory_id) NOTUNIQUE")

    def load_vertices(self, entities, tenants, start):
        script, params = [], {}
        for j, emb in enumerate(entities):
            script.append(f"CREATE VERTEX BenchEntity CONTENT :d{j};")
            params[f"d{j}"] = {"id": str(start + j), "tenant": str(tenants[j]), "embedding": emb.tolist()}
        self.client.command("\n".join(script), params, language="sqlscript")

    def load_edges(self, edges):
        script, params = [], {}
        for j, e in enumerate(edges):
            script.append(
                f"CREATE EDGE BenchRelation FROM (SELECT FROM BenchEntity WHERE id = :s{j}) "
                f"TO (SELECT FROM BenchEntity WHERE id = :t{j}) "
                f"SET id = :i{j}, tenant = :n{j}, memory_id = :m{j}, predicate = :p{j};"
            )
            params.update({f"s{j}": e["source"], f"t{j}": e["destination"], f"i{j}": e["id"],
                           f"n{j}": e["tenant"], f"m{j}": e["memory_id"], f"p{j}": e["predicate"]})
        self.client.command("\n".join(script), params, language="sqlscript")

    def count(self):
        v = self.client.query("SELECT count(*) AS c FROM BenchEntity")[0]["c"]
        e = self.client.query("SELECT count(*) AS c FROM BenchRelation")[0]["c"]
        return v, e

    def build_vector_index(self, config):
        meta = {"dimensions": VECTOR_DIM, "similarity": "COSINE", "encoding": "FLOAT32", "quantization": "NONE",
                "maxConnections": 32, "beamWidth": 200, "efSearch": 100, "addHierarchy": False,
                "storeVectorsInGraph": False, "buildGraphNow": True, **config}
        import json

        import httpx
        http = self.client._http
        prev = http.timeout
        http.timeout = httpx.Timeout(INDEX_READY_DEADLINE_S)  # CREATE INDEX is synchronous and builds the graph
        try:
            self.client.command(f"CREATE INDEX ON BenchEntity (embedding) LSM_VECTOR METADATA {json.dumps(meta)}")
        finally:
            http.timeout = prev
        self.ef_search = int(meta.get("efSearch", 100))
        self.rerank = 10 if meta.get("quantization") == "BINARY" else None
        # create-index is synchronous; confirm with a real probe query
        deadline = time.time() + INDEX_READY_DEADLINE_S
        probe = [0.1] * VECTOR_DIM
        while time.time() < deadline:
            try:
                self.client.query(
                    "SELECT id FROM (SELECT expand(`vector.neighbors`('BenchEntity[embedding]', :q, 1)))", {"q": probe})
                return True
            except Exception:
                time.sleep(2)
        return False

    def index_metadata(self):
        return self.client.query("SELECT FROM schema:indexes")

    # -- queries -------------------------------------------------------
    def _opts(self, extra=""):
        o = f"efSearch:{self.ef_search}"
        if self.rerank:
            o += f", rerank:{self.rerank}"
        return "{" + o + (", " + extra if extra else "") + "}"

    def set_ef(self, ef):
        self.ef_search = ef

    @staticmethod
    def _sim(rows, key):
        out = []
        for r in rows:
            d = r[key]
            out.append((str(r["id"]), (1.0 - d) if key == "distance" else d))
        return out

    def one_hop(self, src):
        return [str(r["id"]) for r in self.client.query(
            "SELECT DISTINCT(id) AS id FROM (SELECT expand(out('BenchRelation')) FROM BenchEntity WHERE id = :s)", {"s": src})]

    def two_hop(self, src):
        return [str(r["id"]) for r in self.client.query(
            "SELECT DISTINCT(id) AS id FROM (SELECT expand(out('BenchRelation').out('BenchRelation')) FROM BenchEntity WHERE id = :s)",
            {"s": src})]

    def memory_enrichment(self, mem_ids):
        return [str(r["id"]) for r in self.client.query(
            "SELECT id FROM BenchRelation WHERE memory_id IN :ids", {"ids": mem_ids})]

    def exact_filtered(self, q, tenant, k):
        rows = self.client.query(
            "SELECT id, `vector.cosineSimilarity`(embedding, :q) AS similarity FROM BenchEntity "
            "WHERE tenant = :t ORDER BY similarity DESC, id ASC LIMIT :k", {"q": q, "t": tenant, "k": k})
        return self._sim(rows, "similarity")

    def ann(self, q, k):
        rows = self.client.query(
            "SELECT id, distance FROM (SELECT expand(`vector.neighbors`('BenchEntity[embedding]', :q, :k, %s)))"
            % self._opts(), {"q": q, "k": k})
        return self._sim(rows, "distance")

    def postfiltered_ann(self, q, tenant, k, fetch_k):
        rows = self.client.query(
            "SELECT id, distance FROM (SELECT expand(`vector.neighbors`('BenchEntity[embedding]', :q, :f, %s))) "
            "WHERE tenant = :t ORDER BY distance ASC LIMIT :k" % self._opts(),
            {"q": q, "f": fetch_k, "t": tenant, "k": k})
        return self._sim(rows, "distance")

    def native_filtered_ann(self, q, tenant, k):
        rows = self.client.query(
            "SELECT id, distance FROM (SELECT expand(`vector.neighbors`('BenchEntity[embedding]', :q, :k, "
            "%s)))" % self._opts("where: {tenant: :t}"), {"q": q, "k": k, "t": tenant})
        return self._sim(rows, "distance")

    def out_edges_batched(self, ids):
        # drive from the indexed BenchEntity.id lookup; filtering on the unindexed edge endpoint scans every edge
        return [str(r["id"]) for r in self.client.query(
            "SELECT id FROM (SELECT expand(outE('BenchRelation')) FROM BenchEntity WHERE id IN :ids)", {"ids": ids})]

    def vector_plus_hop(self, q, tenant, k, fetch_k):
        hits = self.postfiltered_ann(q, tenant, k, fetch_k)
        return hits, self.out_edges_batched([h[0] for h in hits]) if hits else []

    def vector_plus_hop_native(self, q, tenant, k):
        hits = self.native_filtered_ann(q, tenant, k)
        return hits, self.out_edges_batched([h[0] for h in hits]) if hits else []


class FalkorAdapter:
    name = "falkordb"

    def __init__(self, host, port, password):
        self.db = FalkorDB(host=host, port=port, password=password, socket_timeout=30)
        self.g = self.db.select_graph("bench")

    def close(self):
        self.db.close()

    def wait_ready(self, timeout=120):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self.db.connection.ping()
                return True
            except Exception:
                time.sleep(1)
        return False

    def load_schema(self):
        q = self.g.query
        q("CREATE INDEX FOR (n:BenchEntity) ON (n.id)")
        q("CREATE INDEX FOR (n:BenchEntity) ON (n.tenant)")
        q("CREATE INDEX FOR ()-[r:BenchRelation]-() ON (r.id)")
        q("CREATE INDEX FOR ()-[r:BenchRelation]-() ON (r.memory_id)")

    def load_vertices(self, entities, tenants, start):
        rows = [{"id": str(start + j), "t": str(tenants[j]), "e": emb.tolist()} for j, emb in enumerate(entities)]
        self.g.query("UNWIND $rows AS r CREATE (:BenchEntity {id: r.id, tenant: r.t, embedding: vecf32(r.e)})",
                     {"rows": rows})

    def load_edges(self, edges):
        self.g.query(
            "UNWIND $rows AS r MATCH (s:BenchEntity {id: r.source}), (d:BenchEntity {id: r.destination}) "
            "CREATE (s)-[:BenchRelation {id: r.id, tenant: r.tenant, memory_id: r.memory_id, predicate: r.predicate}]->(d)",
            {"rows": edges})

    def count(self):
        v = self.g.query("MATCH (n:BenchEntity) RETURN count(n)").result_set[0][0]
        e = self.g.query("MATCH ()-[r:BenchRelation]->() RETURN count(r)").result_set[0][0]
        return v, e

    def build_vector_index(self, config):
        opts = {"dimension": VECTOR_DIM, "similarityFunction": "cosine", "M": 16, "efConstruction": 200,
                "efRuntime": 100, **config}
        self.ef_runtime = opts["efRuntime"]
        body = ", ".join(f"{k}: {v!r}" if isinstance(v, str) else f"{k}: {v}" for k, v in opts.items())
        self.g.query(f"CREATE VECTOR INDEX FOR (n:BenchEntity) ON (n.embedding) OPTIONS {{{body}}}")
        deadline = time.time() + INDEX_READY_DEADLINE_S
        while time.time() < deadline:
            rows = self.index_metadata()
            vec = [r for r in rows if "VECTOR" in str(r)]
            if vec and all("UNDER CONSTRUCTION" not in str(r) and "Indexing" not in str(r) for r in vec):
                return True
            time.sleep(2)
        return False

    def set_ef(self, ef):
        """Falkor's efRuntime is index metadata: drop and recreate the vector index on the same data."""
        self.g.query("DROP INDEX FOR (n:BenchEntity) ON (n.embedding)")
        return self.build_vector_index({"efRuntime": ef})

    def index_metadata(self):
        return [[str(c) for c in r] for r in self.g.query("CALL db.indexes()").result_set]

    @staticmethod
    def _sim(rows):
        return [(str(r[0]), 1.0 - r[1]) for r in rows]

    def one_hop(self, src):
        return [str(r[0]) for r in self.g.query(
            "MATCH (:BenchEntity {id: $s})-[:BenchRelation]->(m) RETURN DISTINCT m.id", {"s": src}).result_set]

    def two_hop(self, src):
        return [str(r[0]) for r in self.g.query(
            "MATCH (:BenchEntity {id: $s})-[:BenchRelation]->()-[:BenchRelation]->(m) RETURN DISTINCT m.id",
            {"s": src}).result_set]

    def memory_enrichment(self, mem_ids):
        return [str(r[0]) for r in self.g.query(
            "MATCH ()-[r:BenchRelation]->() WHERE r.memory_id IN $ids RETURN r.id", {"ids": mem_ids}).result_set]

    def exact_filtered(self, q, tenant, k):
        return self._sim(self.g.query(
            "MATCH (n:BenchEntity) WHERE n.tenant = $t WITH n, vec.cosineDistance(n.embedding, vecf32($q)) AS d "
            "RETURN n.id, d ORDER BY d ASC, n.id ASC LIMIT $k", {"q": q, "t": tenant, "k": k}).result_set)

    def ann(self, q, k):
        return self._sim(self.g.query(
            "CALL db.idx.vector.queryNodes('BenchEntity','embedding',$k,vecf32($q)) YIELD node, score "
            "RETURN node.id, score ORDER BY score ASC", {"q": q, "k": k}).result_set)

    def postfiltered_ann(self, q, tenant, k, fetch_k):
        return self._sim(self.g.query(
            "CALL db.idx.vector.queryNodes('BenchEntity','embedding',$f,vecf32($q)) YIELD node, score "
            "WITH node, score WHERE node.tenant = $t RETURN node.id, score ORDER BY score ASC LIMIT $k",
            {"q": q, "f": fetch_k, "t": tenant, "k": k}).result_set)

    def native_filtered_ann(self, q, tenant, k):
        raise NotImplementedError("FalkorDB vector procedure has no native predicate filter")

    def out_edges_batched(self, ids):
        return [str(r[0]) for r in self.g.query(
            "MATCH (n:BenchEntity)-[r:BenchRelation]->() WHERE n.id IN $ids RETURN r.id", {"ids": ids}).result_set]

    def vector_plus_hop(self, q, tenant, k, fetch_k):
        hits = self.postfiltered_ann(q, tenant, k, fetch_k)
        return hits, self.out_edges_batched([h[0] for h in hits]) if hits else []

    def vector_plus_hop_native(self, q, tenant, k):
        raise NotImplementedError("FalkorDB vector procedure has no native predicate filter")
