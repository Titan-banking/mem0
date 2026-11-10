"""ArcadeDB vs FalkorDB benchmark runner (smoke + full). See reports/arcadedb-vs-falkordb/."""

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from adapters import ArcadeAdapter, FalkorAdapter  # noqa: E402
from dataset import generate_dataset  # noqa: E402
from engines import ARCADE_IMAGES, FALKOR_IMAGE, Engine, sh  # noqa: E402

K = 10
SEED = 20261110
ARCADE_VERSION = "26.12.1-SNAPSHOT"
WARMUP, TIMED = 100, 1000
RECALL_FLOOR = 0.95
VECTOR_CASES = ["ann", "exact_filtered", "postfiltered", "native_filtered", "vector_plus_hop", "vector_plus_hop_native"]
GRAPH_CASES = ["one_hop", "two_hop", "memory_enrichment"]
# Arcade quantization variants are supplementary and only exercise the lossy/new paths
QUANT_CASES = ["ann", "postfiltered", "native_filtered", "vector_plus_hop_native"]


# ---------------------------------------------------------------- fixtures / oracle
class Fixture:
    def __init__(self, out, size, tenants):
        self.size, self.tenants = size, tenants
        tag = f"{size}_{tenants}"
        self.entities = np.load(out / f"entities_{tag}.npy")
        self.queries = np.load(out / f"queries_{tag}.npy")
        self.qtenants = np.load(out / f"query_tenants_{tag}.npy")
        self.edges = json.load(open(out / f"edges_{tag}.json"))
        self.tenant_of = np.arange(size) % tenants
        self.out_by_id = {}
        for e in self.edges:
            self.out_by_id.setdefault(e["source"], []).append(e)
        self._oracle = {}
        self.by_mem = {}
        for e in self.edges:
            self.by_mem.setdefault(e["memory_id"], []).append(e["id"])

    def oracle_vec(self, i, tenant=None):
        """Exact top-K via float64 cosine over stored float32 vectors (vectors are unit norm)."""
        key = (i, tenant)
        if key in self._oracle:
            return self._oracle[key]
        q = self.queries[i].astype(np.float64)
        idx = np.arange(self.size) if tenant is None else np.where(self.tenant_of == int(tenant))[0]
        sims = self.entities[idx].astype(np.float64) @ q
        order = np.lexsort((idx, -sims))
        top = order[:K]
        self._oracle[key] = {"ids": [str(idx[j]) for j in top], "sims": [float(sims[j]) for j in top],
                             "kth": float(sims[order[min(K, len(order)) - 1]]), "eligible": int(len(idx))}
        return self._oracle[key]

    def mem_ids(self, i):
        return [f"m_{(i * K + j) % self.size}" for j in range(K)]

    def expected_graph(self, case, i):
        src = str(i % self.size)
        if case == "one_hop":
            return sorted({e["destination"] for e in self.out_by_id.get(src, [])})
        if case == "two_hop":
            return sorted({e2["destination"] for e1 in self.out_by_id.get(src, [])
                           for e2 in self.out_by_id.get(e1["destination"], [])})
        return sorted(eid for m in self.mem_ids(i) for eid in self.by_mem.get(m, []))


def score_vector(fx, i, hits, filtered, tenant):
    """Tie-aware recall, shortfall, duplicates and tenant leakage for one query."""
    o = fx.oracle_vec(i, tenant if filtered else None)
    ids = [h[0] for h in hits]
    q = fx.queries[i].astype(np.float64)
    ok = 0
    for rid in set(ids):
        sim = float(fx.entities[int(rid)].astype(np.float64) @ q)
        in_scope = (not filtered) or fx.tenant_of[int(rid)] == int(tenant)
        if in_scope and sim >= o["kth"] - 1e-5:
            ok += 1
    want = min(K, o["eligible"])
    return {"recall": ok / want if want else 1.0, "returned": len(ids), "shortfall": max(0, want - len(ids)),
            "duplicates": len(ids) - len(set(ids)),
            "leakage": sum(1 for r in set(ids) if filtered and fx.tenant_of[int(r)] != int(tenant)),
            "exact_match": ids == o["ids"] and all(abs(h[1] - s) < 1e-5 for h, s in zip(hits, o["sims"]))}


# ---------------------------------------------------------------- engine plumbing
def connect(engine):
    if engine.kind == "arcadedb":
        import httpx
        deadline = time.time() + 120
        base = f"http://127.0.0.1:{engine.port}"
        while time.time() < deadline:
            try:
                r = httpx.post(f"{base}/api/v1/server", auth=("root", engine.password),
                               json={"command": "create database bench"}, timeout=10)
                if r.status_code in (200, 400, 403):
                    break
            except Exception:
                time.sleep(1)
        ad = ArcadeAdapter(base, "bench", "root", engine.password)
    else:
        # the client constructor talks to the server immediately; Docker's proxy resets connections until Redis listens
        deadline, last = time.time() + 120, None
        ad = None
        while ad is None and time.time() < deadline:
            try:
                ad = FalkorAdapter("127.0.0.1", engine.port, engine.password)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(1)
        if ad is None:
            raise RuntimeError(f"falkordb not accepting connections within 120s: {last}\n{engine.logs()[-2000:]}")
    if not ad.wait_ready():
        raise RuntimeError(f"{engine.kind} not ready within 120s:\n{engine.logs()[-2000:]}")
    return ad


def load_dataset(ad, fx):
    t0 = time.perf_counter()
    ad.load_schema()
    for s in range(0, fx.size, 100):
        ad.load_vertices(fx.entities[s:s + 100], fx.tenant_of[s:s + 100], s)
    t_v = time.perf_counter()
    for s in range(0, len(fx.edges), 100):
        ad.load_edges(fx.edges[s:s + 100])
    t_e = time.perf_counter()
    v, e = ad.count()
    if (v, e) != (fx.size, len(fx.edges)):
        raise RuntimeError(f"row-count validation failed: got v={v} e={e}, want v={fx.size} e={len(fx.edges)}")
    return {"vertex_ingest_s": t_v - t0, "edge_ingest_s": t_e - t_v, "vertices": v, "edges": e}


def call(ad, case, fx, i):
    q = fx.queries[i].tolist()
    t = str(fx.qtenants[i])
    fetch_k = min(fx.size, 20 * fx.tenants)
    if case == "one_hop":
        return ad.one_hop(str(i % fx.size))
    if case == "two_hop":
        return ad.two_hop(str(i % fx.size))
    if case == "memory_enrichment":
        return ad.memory_enrichment(fx.mem_ids(i))
    if case == "ann":
        return ad.ann(q, K)
    if case == "exact_filtered":
        return ad.exact_filtered(q, t, K)
    if case == "postfiltered":
        return ad.postfiltered_ann(q, t, K, fetch_k)
    if case == "native_filtered":
        return ad.native_filtered_ann(q, t, K)
    if case == "vector_plus_hop":
        return ad.vector_plus_hop(q, t, K, fetch_k)
    if case == "vector_plus_hop_native":
        return ad.vector_plus_hop_native(q, t, K)
    raise KeyError(case)


def timed_batch(ad, case, fx, concurrency, deadline_s=30):
    """Time each query from request build through full decode. Failures keep error text, no zero latency."""
    idx = list(range(WARMUP, WARMUP + TIMED))

    def one(i):
        t0 = time.perf_counter_ns()
        try:
            res = call(ad, case, fx, i)
            return i, time.perf_counter_ns() - t0, res, None
        except NotImplementedError as e:
            return i, None, None, "unsupported: " + str(e)
        except Exception as e:  # noqa: BLE001 - recorded as an explicit failed sample
            return i, None, None, f"{type(e).__name__}: {str(e)[:300]}"

    t0 = time.perf_counter()
    if concurrency == 1:
        out = [one(i) for i in idx]
    else:
        with ThreadPoolExecutor(concurrency) as pool:
            out = list(pool.map(one, idx))
    return out, time.perf_counter() - t0


def evaluate(case, fx, i, res):
    if case in GRAPH_CASES:
        got = sorted(set(res))
        want = fx.expected_graph(case, i)
        return {"returned": len(got), "exact_match": got == want}
    hits = res[0] if case.startswith("vector_plus") else res
    filtered = case != "ann"
    m = score_vector(fx, i, hits, filtered, fx.qtenants[i])
    if case.startswith("vector_plus"):
        want = sorted({e["id"] for h in hits for e in fx.out_by_id.get(h[0], [])})
        m["hop_exact"] = sorted(set(res[1])) == want
    return m


def run_cases(ad, fx, cases, meta, samples_f, plans):
    for case in cases:
        for i in range(WARMUP):  # warmup, untimed and unrecorded
            try:
                call(ad, case, fx, i)
            except Exception:  # noqa: BLE001
                break
        for conc in (1, 8):
            out, wall = timed_batch(ad, case, fx, conc)
            fails = 0
            for i, ns, res, err in out:
                rec = {**meta, "case": case, "concurrency": conc, "query_id": i, "elapsed_ns": ns, "error": err}
                if err is None:
                    rec.update(evaluate(case, fx, i, res))
                    ids = ([h[0] for h in res[0]] if case.startswith("vector_plus") else
                           [h[0] for h in res] if case in VECTOR_CASES else res)
                    rec["result_ids"] = ids[:50]
                else:
                    fails += 1
                samples_f.write(json.dumps(rec) + "\n")
            samples_f.write(json.dumps({**meta, "case": case, "concurrency": conc, "wall_s": wall,
                                        "batch": True, "n": len(out), "failures": fails}) + "\n")
        samples_f.flush()


def capture_plans(ad, fx, plans):
    q = fx.queries[0].tolist()
    try:
        if isinstance(ad, ArcadeAdapter):
            plans["ann"] = str(ad.client.query("EXPLAIN SELECT id FROM (SELECT expand(`vector.neighbors`("
                                               "'BenchEntity[embedding]', :q, 10, {efSearch:100})))", {"q": q}))[:3000]
            plans["native_filtered"] = str(ad.client.query(
                "EXPLAIN SELECT id FROM (SELECT expand(`vector.neighbors`('BenchEntity[embedding]', :q, 10, "
                "{efSearch:100, where:{tenant:'0'}})))", {"q": q}))[:3000]
        else:
            plans["ann"] = str(ad.g.explain("CALL db.idx.vector.queryNodes('BenchEntity','embedding',10,vecf32($q)) "
                                            "YIELD node RETURN node.id", {"q": q}))[:3000]
    except Exception as e:  # noqa: BLE001
        plans["error"] = f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------- one measured run
def one_run(kind, size, tenants, rep, config, out, fx, samples_f, runs_f, cases):
    label, index_cfg, ef = config
    version = ARCADE_VERSION if kind == "arcadedb" else None
    eng = Engine(kind, version, tag=f"{size}_{tenants}_r{rep}").start()
    run = {"engine": kind, "version": version or "6.0.4", "size": size, "tenants": tenants, "repetition": rep,
           "config": label, "index_config": index_cfg, "started": time.time(), "status": "ok"}
    try:
        ad = connect(eng)
        eng.start_sampler()
        run["load"] = load_dataset(ad, fx)
        run["index_before"] = ad.index_metadata()
        t0 = time.perf_counter()
        ready = ad.build_vector_index({**index_cfg, **({"efSearch": ef} if kind == "arcadedb" and ef else {}),
                                       **({"efRuntime": ef} if kind == "falkordb" and ef else {})})
        run["index_build_s"] = time.perf_counter() - t0
        run["index_ready"] = bool(ready)
        if not ready:
            raise RuntimeError("vector index not ready within 30 minutes")
        run["index_after"] = ad.index_metadata()
        run["total_usable_s"] = run["load"]["vertex_ingest_s"] + run["load"]["edge_ingest_s"] + run["index_build_s"]
        plans = {}
        capture_plans(ad, fx, plans)
        run["plans"] = plans
        meta = {k: run[k] for k in ("engine", "version", "size", "tenants", "repetition", "config")}
        run_cases(ad, fx, cases, meta, samples_f, plans)
        run["resources"] = eng.resource_summary()
        run["persistence"] = eng.persistence_settings()
        run["server_log_tail"] = eng.logs()[-3000:]
        ad.close()
    except Exception as e:  # noqa: BLE001
        run["status"] = "failed"
        run["error"] = f"{type(e).__name__}: {e}"
        run["server_log_tail"] = eng.logs()[-3000:]
    finally:
        eng.stop()
    runs_f.write(json.dumps(run) + "\n")
    runs_f.flush()
    return run


# ---------------------------------------------------------------- manifest
def pkg_versions():
    names = ["falkordb", "redis", "numpy", "httpx", "mem0ai", "pydantic"]
    return {n: metadata.version(n) for n in names if n}


def build_manifest(out, mode, extra=None):
    info = json.loads(sh("docker", "info", "--format", "{{json .}}").stdout)
    img_ids = {}
    for name, ref in {**{f"arcadedb-{v}": r for v, r in ARCADE_IMAGES.items()}, "falkordb-6.0.4": FALKOR_IMAGE}.items():
        sh("docker", "pull", "--platform", "linux/amd64", ref, check=False)
        r = sh("docker", "image", "inspect", ref, "--format", "{{json .Config.Labels}}", check=False)
        img_ids[name] = {"ref": ref, "labels": json.loads(r.stdout) if r.stdout.strip() not in ("", "null") else None}
    m = {"mode": mode, "seed": SEED, "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "images": img_ids, "comparison_engine": "falkordb", "arcadedb_version_under_test": ARCADE_VERSION,
         "observed_servers": observe_identity(),
         "client_packages": pkg_versions(), "python": platform.python_version(),
         "host": {"cpus": info.get("NCPU"), "mem_bytes": info.get("MemTotal"), "docker": info.get("ServerVersion"),
                  "kernel": info.get("KernelVersion"), "os": platform.platform()},
         "container_limits": {"cpus": 4, "memory": "16g", "memory_swap": "16g"},
         "k": K, "warmup": WARMUP, "timed": TIMED, "concurrency": [1, 8], "recall_floor": RECALL_FLOOR,
         "ingest_note": "default-configuration ingest; NOT equal durable-commit throughput",
         "protocol_note": "ArcadeDB over HTTP/JSON, FalkorDB over RESP: latency is application-facing"}
    m.update(extra or {})
    (out / "manifest.json").write_text(json.dumps(m, indent=2))
    return m


def observe_identity():
    """Record what the pinned servers report about themselves (version, build revision, JDK / Redis + module versions)."""
    seen = {}
    eng = Engine("arcadedb", ARCADE_VERSION, tag="ident").start()
    try:
        ad = connect(eng)
        import httpx
        info = httpx.get(f"http://127.0.0.1:{eng.port}/api/v1/server?mode=basic", auth=("root", eng.password), timeout=30).json()
        log = eng.logs()
        jdk = next((ln.split("Running on ", 1)[1] for ln in log.splitlines() if "Running on " in ln), None)
        seen["arcadedb"] = {"server_version": info.get("version"), "jdk": jdk,
                            "source_revision": (info.get("version") or "").split("build ")[-1].split("/")[0].rstrip(")")}
        ad.close()
    finally:
        eng.stop()
    eng = Engine("falkordb", None, tag="ident").start()
    try:
        ad = connect(eng)
        srv = ad.db.connection.info("server")
        mods = ad.db.connection.execute_command("MODULE", "LIST")
        seen["falkordb"] = {"redis_version": srv.get("redis_version"), "modules": [[x.decode() if isinstance(x, bytes) else x for x in m] for m in mods]}
        ad.close()
    finally:
        eng.stop()
    return seen


# ---------------------------------------------------------------- modes
def smoke(out):
    out.mkdir(parents=True, exist_ok=True)
    h = generate_dataset(1000, 10, out)
    build_manifest(out, "smoke", {"dataset_hash": h})
    fx = Fixture(out, 1000, 10)
    sf, rf = open(out / "samples.jsonl", "w"), open(out / "runs.jsonl", "w")
    # smoke uses 30 timed queries and never produces headline numbers
    global TIMED
    TIMED = 30
    cfg = ("baseline", {"quantization": "NONE", "efSearch": 100}, None)
    for kind in ("arcadedb", "falkordb"):
        r = one_run(kind, 1000, 10, 0, cfg, out, fx, sf, rf, GRAPH_CASES + VECTOR_CASES)
        print(kind, r["status"], r.get("error", ""))
    sf.close()
    rf.close()
    from probes import run_probes
    run_probes(out)
    from report import summarize
    summarize(out)


def full(out, manifest_path):
    out.mkdir(parents=True, exist_ok=True)
    if manifest_path and Path(manifest_path).exists():
        smoke_manifest = json.loads(Path(manifest_path).read_text())
        assert smoke_manifest["images"], "smoke manifest has no pinned images"
    hashes = {}
    for size, tl in ((10_000, (10, 100)), (100_000, (10, 100))):
        for t in tl:
            hashes[f"{size}_{t}"] = generate_dataset(size, t, out)
    build_manifest(out, "full", {"dataset_hashes": hashes})
    from report import _jl
    done = {(r["engine"], r["size"], r["tenants"], r["config"], r["repetition"])
            for r in _jl(out / "runs.jsonl") if r["status"] == "ok"}
    sf, rf = open(out / "samples.jsonl", "a"), open(out / "runs.jsonl", "a")

    def one_run_once(kind, size, t, rep, cfg, out, fx, sf, rf, cases):
        if (kind, size, t, cfg[0], rep) in done:  # resume: completed runs are never repeated
            return None
        return one_run(kind, size, t, rep, cfg, out, fx, sf, rf, cases)

    base = ("baseline", {"quantization": "NONE", "efSearch": 100}, None)
    for size, tl in ((10_000, (10, 100)), (100_000, (10, 100))):
        for t in tl:
            fx = Fixture(out, size, t)
            for rep in range(3):
                order = ("arcadedb", "falkordb") if rep % 2 == 0 else ("falkordb", "arcadedb")
                for kind in order:
                    one_run_once(kind, size, t, rep, base, out, fx, sf, rf, GRAPH_CASES + VECTOR_CASES)
            # width-400 rerun of both vector workloads if either unquantized engine misses the recall floor
            from report import mean_recall
            def _low(k):
                vals = [mean_recall(out, k, size, t, c) for c in ("ann", "postfiltered")]
                return any(v is not None and v < RECALL_FLOOR for v in vals)

            low = [k for k in ("arcadedb", "falkordb") if _low(k)]
            if low:
                wide = ("ef400", {"quantization": "NONE"}, 400)
                for rep in range(3):
                    for kind in (("arcadedb", "falkordb") if rep % 2 == 0 else ("falkordb", "arcadedb")):
                        one_run_once(kind, size, t, rep, wide, out, fx, sf, rf, ["ann", "postfiltered", "vector_plus_hop"])
            if size == 100_000:
                for q in ("INT8", "BINARY"):
                    cfg = (f"arcade-{q}", {"quantization": q, **({"rerankOversample": 10} if q == "BINARY" else {})}, None)
                    for rep in range(3):
                        one_run_once("arcadedb", size, t, rep, cfg, out, fx, sf, rf, QUANT_CASES)
    sf.close()
    rf.close()
    from report import summarize, render
    summarize(out)
    render(out, out.parent / "arcadedb-vs-falkordb.html")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--manifest")
    a = ap.parse_args()
    os.environ.setdefault("MEM0_TELEMETRY", "False")
    (smoke if a.mode == "smoke" else lambda o: full(o, a.manifest))(Path(a.output))
