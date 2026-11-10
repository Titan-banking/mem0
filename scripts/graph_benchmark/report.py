"""Summarise raw benchmark artifacts and render the offline HTML report. Reads files only; never touches a database."""

import argparse
import json
import math
import re
from collections import defaultdict
from html import escape
from pathlib import Path

RECALL_FLOOR = 0.95
# cases with identical semantics on both engines (ratios allowed); others are single-engine or supplementary
COMPARABLE = ["one_hop", "two_hop", "memory_enrichment", "exact_filtered", "ann", "postfiltered", "vector_plus_hop"]
RECALL_CASES = {"ann", "postfiltered"}
CASE_LABEL = {
    "one_hop": "One-hop graph", "two_hop": "Two-hop graph", "memory_enrichment": "Memory enrichment (10 ids)",
    "ann": "Unfiltered ANN top-10", "exact_filtered": "Exact tenant-filtered top-10",
    "postfiltered": "Post-filtered ANN (overfetch)", "native_filtered": "Native adaptive filtered search (Arcade only)",
    "vector_plus_hop": "Vector (post-filtered) + one hop", "vector_plus_hop_native": "Vector (native filter) + one hop (Arcade only)",
}


def _jl(path):
    p = Path(path)
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def nearest_rank(xs, p):
    xs = sorted(xs)
    return xs[max(1, math.ceil(p / 100 * len(xs))) - 1] if xs else None


def _key(r):
    return (r["engine"], r["size"], r["tenants"], r["config"])


def summarize(out):
    out = Path(out)
    rows = _jl(out / "samples.jsonl")
    runs = _jl(out / "runs.jsonl")
    cells = defaultdict(lambda: defaultdict(lambda: {"ns": [], "fail": [], "wall": None, "n": 0, "m": defaultdict(list)}))
    for r in rows:
        c = cells[(*_key(r), r["case"], r["concurrency"])][r["repetition"]]
        if r.get("batch"):
            c["wall"], c["n"] = r["wall_s"], r["n"]
        elif r["error"] is None:
            c["ns"].append(r["elapsed_ns"])
            for k in ("recall", "shortfall", "duplicates", "leakage", "returned"):
                if k in r:
                    c["m"][k].append(r[k])
            for k in ("exact_match", "hop_exact"):
                if k in r:
                    c["m"][k].append(1.0 if r[k] else 0.0)
        else:
            c["fail"].append(r["error"])
    summary = []
    for (eng, size, ten, cfg, case, conc), reps in sorted(cells.items()):
        per = []
        for rep, c in sorted(reps.items()):
            per.append({"rep": rep, "ok": len(c["ns"]), "failed": len(c["fail"]),
                        "p50_ms": (nearest_rank(c["ns"], 50) or 0) / 1e6 if c["ns"] else None,
                        "p95_ms": (nearest_rank(c["ns"], 95) or 0) / 1e6 if c["ns"] else None,
                        "p99_ms": (nearest_rank(c["ns"], 99) or 0) / 1e6 if c["ns"] else None,
                        "qps": (c["n"] / c["wall"]) if c["wall"] and c["ns"] else None,
                        "first_error": c["fail"][0] if c["fail"] else None})
        allm = defaultdict(list)
        for c in reps.values():
            for k, v in c["m"].items():
                allm[k] += v
        unsupported = all((p["first_error"] or "").startswith("unsupported") and p["ok"] == 0 for p in per) and per
        mean = lambda k: (sum(allm[k]) / len(allm[k])) if allm.get(k) else None  # noqa: E731
        p50s = [p["p50_ms"] for p in per if p["p50_ms"] is not None]
        qps = [p["qps"] for p in per if p["qps"] is not None]
        summary.append({
            "engine": eng, "size": size, "tenants": ten, "config": cfg, "case": case, "concurrency": conc,
            "status": "unsupported" if unsupported else ("failed" if not p50s else "ok"),
            "reps": per, "p50_ms_mean": sum(p50s) / len(p50s) if p50s else None,
            "p50_ms_range": [min(p50s), max(p50s)] if p50s else None,
            "p95_ms_mean": _avg([p["p95_ms"] for p in per]), "p99_ms_mean": _avg([p["p99_ms"] for p in per]),
            "qps_mean": sum(qps) / len(qps) if qps else None, "qps_range": [min(qps), max(qps)] if qps else None,
            "failures": sum(p["failed"] for p in per), "successes": sum(p["ok"] for p in per),
            "mean_recall": mean("recall"), "shortfall": sum(allm.get("shortfall", [])),
            "duplicates": sum(allm.get("duplicates", [])), "leakage": sum(allm.get("leakage", [])),
            "exact_match_rate": mean("exact_match"), "hop_exact_rate": mean("hop_exact")})
    (out / "summary.json").write_text(json.dumps({"cells": summary, "runs": [_slim(r) for r in runs]}, indent=2))
    corr = [{k: c[k] for k in ("engine", "size", "tenants", "config", "case", "concurrency", "mean_recall", "shortfall",
                               "duplicates", "leakage", "exact_match_rate", "hop_exact_rate", "failures")}
            for c in summary if c["concurrency"] == 1]
    (out / "correctness.json").write_text(json.dumps(corr, indent=2))
    return summary


def _avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _slim(r):
    return {k: v for k, v in r.items() if k not in ("server_log_tail", "plans", "index_before", "index_after")}


def mean_recall(out, engine, size, tenants, case, config="baseline"):
    vals = []
    for r in _jl(Path(out) / "samples.jsonl"):
        if (r["engine"], r["size"], r["tenants"], r["config"], r["case"]) == (engine, size, tenants, config, case) \
                and r.get("error") is None and "recall" in r and not r.get("batch") and r["concurrency"] == 1:
            vals.append(r["recall"])
    return sum(vals) / len(vals) if vals else None


# ------------------------------------------------------------------ HTML
CSS = """body{font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1180px;padding:16px;color:#1b1f23}
h1{font-size:1.7rem}h2{margin-top:2rem;border-bottom:2px solid #d0d7de;padding-bottom:4px}h3{margin-top:1.4rem}
table{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:.85rem}th,td{border:1px solid #d0d7de;padding:4px 7px;text-align:left}td.n{text-align:right}
th{background:#f3f5f7}th.n{text-align:right}.wrap{overflow-x:auto}
.na{color:#8a5a00;background:#fff6e0}.fail{color:#a40e26;background:#ffebe9}.ok{color:#0a5f2a}.warn{background:#fff6e0;padding:8px 12px;border-left:4px solid #bf8700}
code,pre{background:#f3f5f7;padding:1px 4px;border-radius:3px;font-size:.85em}pre{padding:8px;white-space:pre-wrap;word-break:break-word}small{color:#57606a}
@media(max-width:700px){body{padding:8px}table{font-size:.75rem}}"""


def _f(x, d=2):
    return "–" if x is None else f"{x:,.{d}f}"


def _cell(c):
    if c is None:
        return '<td class="na">not run</td>'
    if c["status"] == "unsupported":
        return '<td class="na">unsupported</td>'
    if c["status"] == "failed":
        return f'<td class="fail" title="{escape(str(c["reps"][0]["first_error"]))}">failed ({c["failures"]})</td>'
    return f"<td>{_f(c['p50_ms_mean'])}</td>"


def _numeric(x):
    return bool(re.match(r"^[\d,.\-–]+\s*(×|ms|s|GiB|MiB)?(\s|$|<small)", x))


def _table(head, body, cls=""):
    return (f'<div class="wrap"><table class="{cls}"><tr>' + "".join(f"<th>{escape(h)}</th>" for h in head) + "</tr>"
            + "".join("<tr>" + "".join(x if x.startswith("<td") else f'<td class="{"n" if _numeric(x) else ""}">{x}</td>' for x in r) + "</tr>" for r in body)
            + "</table></div>")


def render(out, html_path):
    out = Path(out)
    S = json.loads((out / "summary.json").read_text())
    cells, runs = S["cells"], S["runs"]
    manifest = json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").exists() else {}
    pr = json.loads((out / "pr-review.json").read_text()) if (out / "pr-review.json").exists() else {}
    bugs = json.loads((out / "bugs.json").read_text()) if (out / "bugs.json").exists() else None
    idx = {(c["engine"], c["size"], c["tenants"], c["config"], c["case"], c["concurrency"]): c for c in cells}
    scales = sorted({(c["size"], c["tenants"]) for c in cells})
    configs = sorted({c["config"] for c in cells})
    h = ["<h1>ArcadeDB vs FalkorDB: graph + vector measurements</h1>",
         f"<small>Generated from raw artifacts in <code>{escape(str(out))}</code>. Seed {manifest.get('seed')}. "
         f"Mode: {escape(str(manifest.get('mode')))}.</small>"]

    # --- conclusions
    wins = {"arcadedb": 0, "falkordb": 0}
    ratio_rows = []
    for size, ten in scales:
        for case in COMPARABLE:
            a, f = (idx.get((e, size, ten, "baseline", case, 1)) for e in ("arcadedb", "falkordb"))
            ok = a and f and a["status"] == "ok" and f["status"] == "ok"
            gate = ok
            reason = ""
            if ok and case in RECALL_CASES:
                gate = (a["mean_recall"] or 0) >= RECALL_FLOOR and (f["mean_recall"] or 0) >= RECALL_FLOOR
                reason = "" if gate else f"recall below {RECALL_FLOOR} (A {_f(a['mean_recall'],3)}, F {_f(f['mean_recall'],3)}); no ratio"
            if ok and case == "exact_filtered":
                gate = (a["exact_match_rate"] == 1.0 and f["exact_match_rate"] == 1.0)
                reason = "" if gate else "exact mismatch; no ratio"
            if gate:
                ratio = f["p50_ms_mean"] / a["p50_ms_mean"] if a["p50_ms_mean"] else None
                wins["arcadedb" if ratio and ratio > 1 else "falkordb"] += 1
                ratio_rows.append([f"{size:,}/{ten}", CASE_LABEL[case], _f(a["p50_ms_mean"]), _f(f["p50_ms_mean"]),
                                   f"{_f(ratio)}×" if ratio else "–", "ArcadeDB faster" if ratio and ratio > 1 else "FalkorDB faster"])
            else:
                ratio_rows.append([f"{size:,}/{ten}", CASE_LABEL[case], _cell(a), _cell(f), "–", escape(reason or "not comparable / not run")])
    h += ["<h2>1. Observed conclusions</h2>",
          '<div class="warn"><b>Scope.</b> Microbenchmarks on one 8-CPU host, containers capped at 4 CPUs / 16 GiB, single-threaded Python '
          "client, ArcadeDB over HTTP/JSON vs FalkorDB over RESP (latency is application-facing, not engine-internal). Ingest is "
          "default-configuration, <b>not</b> equal durable-commit throughput. HNSW (FalkorDB) and Vamana-style (ArcadeDB) knobs are "
          "named configurations, not equivalent settings. Ratios are shown only where semantics match and recall ≥ "
          f"{RECALL_FLOOR} (exact cases must match the oracle). This is not a mem0 end-to-end or semantic-quality benchmark.</div>",
          f"<p>Gated p50 comparisons at concurrency 1, baseline config: ArcadeDB faster in <b>{wins['arcadedb']}</b>, "
          f"FalkorDB faster in <b>{wins['falkordb']}</b> cells. Per-cell values and ranges are in section 5.</p>",
          _table(["size/tenants", "case", "ArcadeDB p50 ms", "FalkorDB p50 ms", "FalkorDB÷Arcade", "note"], ratio_rows)]

    # --- concurrency scaling
    scale_rows = []
    for size, ten in scales:
        for case in COMPARABLE + ["native_filtered"]:
            vals = []
            for e in ("arcadedb", "falkordb"):
                a, b = idx.get((e, size, ten, "baseline", case, 1)), idx.get((e, size, ten, "baseline", case, 8))
                ok = a and b and a["status"] == "ok" and b["status"] == "ok" and a["qps_mean"] and b["qps_mean"]
                vals.append(f"{b['qps_mean'] / a['qps_mean']:.2f}×" if ok else "–")
            if any(v != "–" for v in vals):
                scale_rows.append([f"{size:,}/{ten}", CASE_LABEL[case], vals[0], vals[1]])
    h += ["<h3>Concurrency scaling (QPS at c=8 ÷ QPS at c=1, baseline)</h3>",
          '<p class="warn">The client is one Python process driving threads (GIL). For sub-millisecond cases (graph queries) the c=8 figure is '
          "bounded by the client, so values near 1× say little about either engine. Only server-heavy cases (e.g. exact filtered scoring, ~10 ms) "
          "reveal engine-side concurrency behaviour. Causes were not investigated.</p>",
          _table(["size/tenants", "case", "ArcadeDB", "FalkorDB"], scale_rows)]

    # --- manifest
    h += ["<h2>2. Reproducibility and resources</h2>"]
    imgs = [[escape(k), f"<code>{escape(v['ref'])}</code>"] for k, v in manifest.get("images", {}).items()]
    h += [_table(["image", "pinned reference"], imgs),
          "<p>Observed from the running servers (startup log / <code>/api/v1/server</code> / <code>INFO</code> / <code>MODULE LIST</code>), "
          "not inferred from image tags:</p>",
          f"<pre>{escape(json.dumps(manifest.get('observed_servers'), indent=1))}</pre>",
          f"<pre>{escape(json.dumps({k: manifest.get(k) for k in ('host', 'container_limits', 'client_packages', 'python', 'k', 'warmup', 'timed', 'concurrency', 'ingest_note', 'protocol_note')}, indent=1))}</pre>"]

    # --- PR review
    h += ["<h2>3. Last three months of ArcadeDB PRs (targeted review)</h2>",
          f"<p>Window: {escape(str(pr.get('window')))}. Query: <code>{escape(str(pr.get('query')))}</code>. Source-backed targeted "
          "review, not a claim that every PR was read. Upstream benchmark figures are citations, not local measurements.</p>",
          _table(["PR", "merged", "consequence for this benchmark"],
                 [[f'<a href="{escape(f["url"])}">#{f["pr"]}</a>', escape(f["date"]), escape(f["note"])] for f in pr.get("findings", [])])]

    # --- correctness
    rows = []
    for c in sorted((c for c in cells if c["concurrency"] == 1), key=lambda c: (c["size"], c["tenants"], c["config"], c["case"], c["engine"])):
        if c["status"] != "ok":
            rows.append([c["engine"], f"{c['size']:,}/{c['tenants']}", c["config"], CASE_LABEL.get(c["case"], c["case"]), c["status"], "", "", "", "", ""])
            continue
        low = c["mean_recall"] is not None and c["case"] in RECALL_CASES and c["mean_recall"] < RECALL_FLOOR
        rows.append([c["engine"], f"{c['size']:,}/{c['tenants']}", c["config"], CASE_LABEL.get(c["case"], c["case"]), "ok",
                     f'<td class="{"fail" if low else ""}">{_f(c["mean_recall"], 4)}</td>' if c["mean_recall"] is not None else "–",
                     str(c["shortfall"]), str(c["duplicates"]),
                     f'<td class="{"fail" if c["leakage"] else "ok"}">{c["leakage"]}</td>',
                     _f(c["exact_match_rate"], 3) if c["exact_match_rate"] is not None else "–"])
    h += ["<h2>4. Correctness and recall</h2>", f"<p>Oracle: exhaustive float64 cosine on stored float32 vectors, tie-aware at the kth boundary, tolerance 1e-5. Recall floor for a headline ratio: {RECALL_FLOOR}.</p>",
          _table(["engine", "size/tenants", "config", "case", "status", "mean recall@10", "shortfall", "duplicates", "tenant leakage", "exact-match rate"], rows)]

    # --- workloads
    h += ["<h2>5. Latency and throughput</h2>", "<p>p50/p95/p99 are nearest-rank, successful queries only, mean across fresh-data repetitions (range of p50 shown). "
          "QPS = queries ÷ batch wall time. Failed, unsupported and not-run cells are explicit.</p>"]
    for size, ten in scales:
        for cfg in configs:
            part = [c for c in cells if c["size"] == size and c["tenants"] == ten and c["config"] == cfg]
            if not part:
                continue
            body = []
            for case in sorted({c["case"] for c in part}, key=lambda x: list(CASE_LABEL).index(x) if x in CASE_LABEL else 99):
                for eng in ("arcadedb", "falkordb"):
                    r = []
                    for conc in (1, 8):
                        c = idx.get((eng, size, ten, cfg, case, conc))
                        if c is None:
                            r += ['<td class="na">not run</td>'] * 4
                        elif c["status"] != "ok":
                            r += [f'<td class="{"na" if c["status"] == "unsupported" else "fail"}">{c["status"]}</td>'] + ["<td>–</td>"] * 3
                        else:
                            rng = c["p50_ms_range"]
                            r += [f"<td>{_f(c['p50_ms_mean'])} <small>[{_f(rng[0])}–{_f(rng[1])}]</small></td>", _f(c["p95_ms_mean"]), _f(c["p99_ms_mean"]), _f(c["qps_mean"], 1)]
                    if any(x for x in r):
                        body.append([CASE_LABEL.get(case, case), eng] + r)
            h += [f"<h3>{size:,} entities, {ten} tenants — config <code>{escape(cfg)}</code>"
                  + (" <small>(supplementary, lossy; compare against FalkorDB float32 only with the recall column)</small>" if cfg.startswith("arcade-") else "") + "</h3>",
                  _table(["case", "engine", "c=1 p50 ms", "p95", "p99", "QPS", "c=8 p50 ms", "p95", "p99", "QPS"], body)]

    # --- ingestion / readiness / resources
    ing = []
    for r in runs:
        if r.get("status") != "ok":
            ing.append([r["engine"], f"{r['size']:,}/{r['tenants']}", r["config"], str(r["repetition"]), f'<td class="fail">failed: {escape(str(r.get("error"))[:140])}</td>', "", "", "", "", ""])
            continue
        res = r.get("resources", {})
        ing.append([r["engine"], f"{r['size']:,}/{r['tenants']}", r["config"], str(r["repetition"]),
                    _f(r["load"]["vertex_ingest_s"], 1), _f(r["load"]["edge_ingest_s"], 1), _f(r["index_build_s"], 1),
                    _f(r["total_usable_s"], 1), "ready" if r["index_ready"] else "NOT READY",
                    f"{_f((res.get('peak_container_mem_bytes') or 0) / 2**30, 2)} GiB / {_f((res.get('volume_bytes') or 0) / 2**20, 0)} MiB"])
    h += ["<h2>6. Ingestion, index readiness and resources</h2>",
          "<p>Default persistence settings preserved per engine; ingest numbers are default-configuration, not equal durability. "
          "Memory is sampled once per second from the container cgroup (<code>docker stats</code>), which includes page cache and is not pure RSS. "
          "ArcadeDB runs with the container-default JVM heap sizing (the server log reports a ~12 GiB max heap under the 16 GiB cap), so its peak memory "
          "reflects heap growth and is not a minimum requirement. FalkorDB ran stock Redis persistence (RDB schedule <code>3600 1 300 100 60 10000</code>, AOF off): "
          "its snapshot had not fired when the volume was sampled, so <b>0 MiB is not a data size</b>. Index readiness is polled every 2 s, which quantises "
          "FalkorDB's build times (26.0 s, 28.0 s).</p>",
          _table(["engine", "size/tenants", "config", "rep", "vertex ingest s", "edge ingest s", "index build s", "total usable s", "index", "peak mem / volume"], ing)]

    # --- regression matrix
    h += ["<h2>7. Regression matrix, reproductions, issues</h2>"]
    if not bugs:
        h += ["<p>No engine defect was investigated or filed from this run's artifacts (<code>bugs.json</code> absent).</p>"]
    else:
        def _ev(e):
            return escape(json.dumps(e, ensure_ascii=False)) if not isinstance(e, str) else escape(e)
        h += [f"<p>{escape(bugs.get('summary', ''))}</p>",
              f"<p><b>Upstream issues filed:</b> {escape(', '.join(map(str, bugs.get('filed_issues', []))) or 'none')}. "
              f"<b>Cross-version matrix:</b> {escape(str(bugs.get('regression_matrix', '')))}.</p>",
              "<h3>Investigated</h3>",
              _table(["item", "classification", "evidence"],
                     [[escape(i["item"]), escape(i["classification"]),
                       "<br>".join(f"<code>{escape(k)}</code>: {_ev(v)}" for k, v in i.get("evidence", {}).items()
                                   if k not in ("self_query_not_found_before_restart",)) if isinstance(i.get("evidence"), dict) else _ev(i.get("evidence", ""))]
                      for i in bugs.get("investigated", [])]),
              "<h3>Excluded: harness defects found and fixed during this run (not engine defects)</h3>",
              _table(["what happened", "cause", "evidence"],
                     [[escape(x["what"]), escape(x["cause"]), escape(x["evidence"])] for x in bugs.get("excluded_non_defects", [])])]
    Path(html_path).write_text("<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
                               f"<title>ArcadeDB vs FalkorDB</title><style>{CSS}</style><body>" + "\n".join(h) + "</body></html>")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    summarize(a.input)
    render(a.input, a.output)
