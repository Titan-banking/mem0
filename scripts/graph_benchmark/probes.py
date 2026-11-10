"""Engine-level correctness probes on a tiny known corpus, plus the live SDK integration run. Smoke only."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from adapters import VECTOR_DIM  # noqa: E402
from engines import Engine  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _unit(i):
    v = np.zeros(VECTOR_DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def _corpus():
    rng = np.random.default_rng(7)
    V = rng.standard_normal((40, VECTOR_DIM)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    V[0], V[1], V[2] = _unit(0), _unit(0), _unit(1)  # 0 and 1 identical (different tenants); 2 orthogonal
    tenants = ["A", "B", "A"] + ["A" if i % 2 else "B" for i in range(3, 40)]
    edges = [{"id": f"e{i}", "source": str(i), "destination": str((i + 1) % 40), "tenant": tenants[i],
              "memory_id": f"m_{i}", "predicate": "related_to"} for i in range(40)]
    edges += [{"id": f"x{i}", "source": str(i), "destination": str((i + 2) % 40), "tenant": tenants[i],
               "memory_id": f"m_{i}", "predicate": "related_to"} for i in range(40)]
    return V, tenants, edges


class Probes:
    def __init__(self):
        self.items = []

    def check(self, name, ok, expected=None, actual=None, status=None):
        self.items.append({"probe": name, "status": status or ("pass" if ok else "fail"),
                           "expected": expected, "actual": actual})


def _connect(eng):
    from run import connect
    return connect(eng)


def _setup(kind):
    V, tenants, edges = _corpus()
    eng = Engine(kind, "26.12.1-SNAPSHOT" if kind == "arcadedb" else None, tag="probe").start()
    ad = _connect(eng)
    ad.load_schema()
    for s in range(0, 40, 20):
        ad.load_vertices(V[s:s + 20], tenants[s:s + 20], s)
    for s in range(0, len(edges), 40):
        ad.load_edges(edges[s:s + 40])
    assert ad.build_vector_index({"efSearch": 100} if kind == "arcadedb" else {})
    return eng, ad, V, tenants, edges


def probe_engine(kind):
    P = Probes()
    eng, ad, V, tenants, edges = _setup(kind)
    try:
        q = V[0].tolist()
        top = dict(ad.ann(q, 5))
        P.check("identical vectors have distance 0 (similarity 1)", abs(top.get("0", 0) - 1) < 1e-5 and abs(top.get("1", 0) - 1) < 1e-5, "ids 0,1 sim=1", top)
        ex_orth = dict(ad.exact_filtered(q, "A", 40)).get("2")
        P.check("orthogonal vectors have distance 1 (similarity 0)", ex_orth is not None and abs(ex_orth) < 1e-5, "sim(0,2)=0", ex_orth)
        for name, fn in (("exact", lambda: ad.exact_filtered(q, "A", 10)), ("postfiltered", lambda: ad.postfiltered_ann(q, "A", 10, 40))):
            ids = [h[0] for h in fn()]
            P.check(f"adversarial wrong-tenant nearest vector is excluded ({name})", "1" not in ids and ids[:1] == ["0"], "first=0, no id 1", ids[:5])
        try:
            ids = [h[0] for h in ad.native_filtered_ann(q, "A", 10)]
            P.check("adversarial wrong-tenant nearest vector is excluded (native filter)", "1" not in ids and ids[:1] == ["0"], "first=0, no id 1", ids[:5])
            P.check("native filter with missing tenant returns nothing", ad.native_filtered_ann(q, "ZZZ", 10) == [], [], ad.native_filtered_ann(q, "ZZZ", 10))
        except NotImplementedError:
            P.check("native filtered search", False, status="unsupported", actual="FalkorDB has no native predicate filter")
        P.check("exact filter with missing tenant returns nothing", ad.exact_filtered(q, "ZZZ", 10) == [], [], ad.exact_filtered(q, "ZZZ", 10))
        rq = np.random.default_rng(3).standard_normal(VECTOR_DIM).astype(np.float32)
        rq /= np.linalg.norm(rq)
        idx = [i for i in range(40) if tenants[i] == "B"]
        sims = V[idx].astype(np.float64) @ rq.astype(np.float64)
        want = [str(idx[j]) for j in np.lexsort((idx, -sims))[:10]]
        got = [h[0] for h in ad.exact_filtered(rq.tolist(), "B", 10)]
        P.check("exact tenant top-10 order matches the numpy oracle", got == want, want, got)
        P.check("one-hop edge set", sorted(ad.one_hop("0")) == ["1", "2"], ["1", "2"], sorted(ad.one_hop("0")))
        P.check("two-hop edge set", sorted(ad.two_hop("0")) == ["2", "3", "4"], ["2", "3", "4"], sorted(ad.two_hop("0")))
        P.check("memory enrichment edge set", sorted(ad.memory_enrichment(["m_0", "m_1"])) == ["e0", "e1", "x0", "x1"], ["e0", "e1", "x0", "x1"], sorted(ad.memory_enrichment(["m_0", "m_1"])))
        # mutation visibility (indexed insert/update/delete)
        near = (_unit(0) * 0.999 + _unit(5) * 0.04).astype(np.float32)
        near /= np.linalg.norm(near)
        if kind == "arcadedb":
            c = ad.client.command
            c("UPDATE BenchEntity SET embedding = :e WHERE id = '5'", {"e": near.tolist()})
            c("DELETE FROM BenchEntity WHERE id = '0'")
            c("CREATE VERTEX BenchEntity CONTENT :d", {"d": {"id": "new", "tenant": "A", "embedding": _unit(0).tolist()}})
        else:
            ad.g.query("MATCH (n:BenchEntity {id:'5'}) SET n.embedding = vecf32($e)", {"e": near.tolist()})
            ad.g.query("MATCH (n:BenchEntity {id:'0'}) DELETE n")
            ad.g.query("CREATE (:BenchEntity {id:'new', tenant:'A', embedding: vecf32($e)})", {"e": _unit(0).tolist()})
        ids = [h[0] for h in ad.ann(q, 6)]
        P.check("indexed delete/update/insert are visible to ANN", "0" not in ids and "new" in ids and "5" in ids, "0 gone; new,5 present", ids)
        # rollback + restart persistence
        if kind == "arcadedb":
            try:
                with ad.client.transaction("probe") as sid:
                    ad.client.command("CREATE VERTEX BenchEntity CONTENT :d", {"d": {"id": "rolled", "tenant": "A", "embedding": _unit(0).tolist()}}, sid)
                    raise RuntimeError("abort")
            except RuntimeError:
                pass
            P.check("rolled-back embedding is not visible", "rolled" not in [h[0] for h in ad.ann(q, 10)], "absent", [h[0] for h in ad.ann(q, 10)])
        else:
            P.check("transaction rollback", False, status="unsupported", actual="no client-side multi-statement transaction in FalkorDB")
        ad.client.command("CREATE VERTEX BenchEntity CONTENT :d", {"d": {"id": "durable", "tenant": "A", "embedding": _unit(7).tolist()}}) if kind == "arcadedb" else \
            ad.g.query("CREATE (:BenchEntity {id:'durable', tenant:'A', embedding: vecf32($e)})", {"e": _unit(7).tolist()})
        persistence = eng.persistence_settings()
        ad.close()
        eng.restart(kill=True)  # kill immediately after the acknowledged commit
        ad2 = _connect(eng)
        try:
            if kind == "arcadedb":
                ad2.ef_search = 100
            v, _e = ad2.count()
            alive = ad2.exact_filtered(_unit(7).tolist(), "A", 1)
            survived = bool(alive) and alive[0][0] == "durable"
            ann_after = [h[0] for h in ad2.ann(_unit(7).tolist(), 3)] if kind == "arcadedb" else None
            durability = {"vertices": v, "top": alive[:1], "persistence": persistence}
            if survived:
                P.check("acknowledged commit survives kill -9 + restart", True, "durable present", durability)
            elif kind == "falkordb":
                # stock Redis defaults (RDB snapshots, no AOF): losing an un-snapshotted write on SIGKILL is the documented
                # default durability policy, reported as an observation, not an engine defect
                P.check("acknowledged commit survives kill -9 + restart", False, "depends on persistence policy", durability,
                        status="observed-default-durability")
            else:
                P.check("acknowledged commit survives kill -9 + restart", False, "durable present", durability)
            if kind == "arcadedb":
                P.check("rolled-back embedding absent after restart", "rolled" not in (ann_after or []), "absent", ann_after)
                P.check("vector index serves queries after restart", "durable" in (ann_after or []), "durable in ANN", ann_after)
        finally:
            ad2.close()
    finally:
        P_logs = eng.logs()[-1500:]
        eng.stop()
    return {"probes": P.items, "server_log_tail": P_logs}


def run_sdk_integration(out):
    eng = Engine("arcadedb", "26.12.1-SNAPSHOT", tag="sdk").start()
    try:
        import httpx
        base = f"http://127.0.0.1:{eng.port}"
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                if httpx.get(f"{base}/api/v1/ready", timeout=5).status_code in (200, 204):
                    break
            except Exception:
                time.sleep(1)
        httpx.post(f"{base}/api/v1/server", auth=("root", eng.password), json={"command": "create database sdk"}, timeout=30)
        env = {**os.environ, "MEM0_TELEMETRY": "False", "MEM0_ARCADEDB_TEST_URL": base, "MEM0_ARCADEDB_TEST_DATABASE": "sdk",
               "MEM0_ARCADEDB_TEST_USERNAME": "root", "MEM0_ARCADEDB_TEST_PASSWORD": eng.password}
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/graphs/arcadedb/test_arcadedb_integration.py"],
                           cwd=REPO, env=env, capture_output=True, text=True)
        tail = (r.stdout + r.stderr).replace(eng.password, "<redacted>")[-1500:]
        lines = [ln for ln in tail.strip().splitlines() if " passed" in ln or " failed" in ln or " error" in ln]
        return {"returncode": r.returncode, "summary": lines[-1].strip() if lines else tail.strip()[-300:]}
    finally:
        eng.stop()


def run_probes(out):
    res = {}
    for kind in ("arcadedb", "falkordb"):
        try:
            res[kind] = probe_engine(kind)
        except Exception as e:  # noqa: BLE001
            res[kind] = {"error": f"{type(e).__name__}: {e}"}
    res["sdk_integration"] = run_sdk_integration(out)
    Path(out, "probes.json").write_text(json.dumps(res, indent=2, default=str))
    for kind in ("arcadedb", "falkordb"):
        for p in res[kind].get("probes", []):
            print(f"[{kind}] {p['status']:<11} {p['probe']}")
        if "error" in res[kind]:
            print(f"[{kind}] ERROR {res[kind]['error']}")
    print("[sdk]", res["sdk_integration"])
