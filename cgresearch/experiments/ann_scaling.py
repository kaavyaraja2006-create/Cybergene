"""Scalable similarity search benchmark (experiment 9).

Methods: sqlite_json_scan (the prototype's ImmuneMemory.similarity_search), numpy_exact (in-memory
exact cosine), hnswlib (ANN; several ef_search), faiss_hnsw (if installed).
All use the same legacy 28-d vectors and the same queries. Quality metrics are vs the exact top-10.
`recall_vs_exact@10` is tie-tolerant (a returned item counts if its cosine >= exact 10th-best - 1e-9);
`overlap@10` is strict id-set overlap. Both are reported because synthetic vectors contain ties."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from collections import defaultdict
from typing import Any, Dict

import numpy as np

from .._backend import BACKEND  # noqa: F401
from memory import ImmuneMemory  # type: ignore

from ..runlog import ExperimentRun
from .corpus import make_corpus
from .retrieval import l2norm


def _rss():
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except Exception:
        return 0


def _lat_stats(lat):
    a = np.asarray(lat) * 1e3
    return {"latency_ms_median": float(np.median(a)), "latency_ms_p95": float(np.percentile(a, 95)),
            "latency_ms_p99": float(np.percentile(a, 99)), "latency_ms_mean": float(a.mean()),
            "qps_single_thread": float(1e3 / a.mean())}


def run_ann(cfg: Dict[str, Any], out_root="results", argv=None) -> ExperimentRun:
    run = ExperimentRun(cfg.get("name", "ann_scaling"), cfg, out_root, argv, {"synthetic": True}, cfg["seeds"])
    sizes = cfg.get("sizes", [1000, 3000, 10000, 50000, 100000])
    nq = cfg.get("n_queries", 200)
    sql_q = cfg.get("sql_max_queries_large", 20)
    hn_cfg = cfg.get("hnsw", {"M": 16, "ef_construction": 200, "ef_search": [20, 50, 200]})
    try:
        import hnswlib
    except Exception:
        hnswlib = None
    try:
        import faiss
    except Exception:
        faiss = None
    run.manifest["ann_backends"] = {"hnswlib": hnswlib is not None, "faiss": faiss is not None}
    for seed in cfg["seeds"]:
        t0 = time.perf_counter()
        store, _ = make_corpus(max(sizes), seed)
        queries, _ = make_corpus(nq, seed + 10_000)
        V = np.vstack([d.vector() for d in store]); Vn = l2norm(V)
        Qv = np.vstack([q.vector() for q in queries]); Qn = l2norm(Qv)
        gen_s = time.perf_counter() - t0
        for n in sizes:
            Mn = Vn[:n]
            exact_top, exact_kth = [], []
            sims_all = Qn @ Mn.T
            for i in range(nq):
                top = np.argsort(-sims_all[i], kind="stable")[:10]
                exact_top.append(top); exact_kth.append(sims_all[i][top[-1]])
            base = {"seed": seed, "n_records": n, "dim": int(V.shape[1])}

            def quality(ids_list, qn=nq):
                ov, rc = [], []
                for i, ids in enumerate(ids_list):
                    ids = np.asarray(ids)[:10]
                    ov.append(len(set(ids.tolist()) & set(exact_top[i].tolist())) / 10.0)
                    rc.append(float((sims_all[i][ids] >= exact_kth[i] - 1e-9).sum() / 10.0))
                return {"overlap_at_10": float(np.mean(ov)), "recall_vs_exact_at_10": float(np.mean(rc))}

            # --- numpy exact
            rss0 = _rss(); lat = []; ids = []
            for i in range(nq):
                t0 = time.perf_counter()
                s = Mn @ Qn[i]; top = np.argpartition(-s, 10)[:10]; top = top[np.argsort(-s[top], kind="stable")]
                lat.append(time.perf_counter() - t0); ids.append(top)
            run.add_row("per_seed_method", {**base, "method": "numpy_exact", "n_queries": nq, "build_seconds": 0.0,
                                            "index_memory_bytes": int(Mn.nbytes), **_lat_stats(lat), **quality(ids)})
            run.add_rows("per_query_latency", [{**base, "method": "numpy_exact", "q": i, "latency_s": l} for i, l in enumerate(lat)])
            # --- sqlite/json linear scan (prototype implementation)
            qs = nq if n < 50000 else min(nq, sql_q)
            path = run.artifact_path("sql_tmp.db").with_suffix(f".{seed}.{n}.db")
            if path.exists():
                os.remove(path)
            mem = ImmuneMemory(str(path))
            t0 = time.perf_counter()
            mem.conn.executemany("INSERT INTO attack_dna (dna_id, attack_type, payload_json, vector_json, created_at) VALUES (?,?,?,?,?)",
                                 [(store[j].dna_id, store[j].attack_type, "{}", json.dumps(V[j].tolist()), 0.0) for j in range(n)])
            mem.conn.commit(); build = time.perf_counter() - t0
            id_index = {store[j].dna_id: j for j in range(n)}
            lat = []; ids = []
            for i in range(qs):
                t0 = time.perf_counter(); res = mem.similarity_search(queries[i], 10); lat.append(time.perf_counter() - t0)
                ids.append([id_index[r[0]] for r in res])
            ov = np.mean([len(set(ids[i]) & set(exact_top[i].tolist())) / 10.0 for i in range(qs)])
            rc = np.mean([(sims_all[i][np.asarray(ids[i])] >= exact_kth[i] - 1e-9).sum() / 10.0 for i in range(qs)])
            run.add_row("per_seed_method", {**base, "method": "sqlite_json_scan", "n_queries": qs, "build_seconds": build,
                                            "index_memory_bytes": int(os.path.getsize(path)), **_lat_stats(lat),
                                            "overlap_at_10": float(ov), "recall_vs_exact_at_10": float(rc)})
            run.add_rows("per_query_latency", [{**base, "method": "sqlite_json_scan", "q": i, "latency_s": l} for i, l in enumerate(lat)])
            mem.conn.close(); os.remove(path)
            # --- hnswlib
            if hnswlib is not None:
                rss0 = _rss(); t0 = time.perf_counter()
                idx = hnswlib.Index(space="cosine", dim=V.shape[1])
                idx.init_index(max_elements=n, M=hn_cfg["M"], ef_construction=hn_cfg["ef_construction"], random_seed=seed)
                idx.add_items(Mn.astype(np.float32), np.arange(n)); build = time.perf_counter() - t0
                rss_delta = _rss() - rss0
                for ef in hn_cfg["ef_search"]:
                    idx.set_ef(max(ef, 10)); lat = []; ids = []
                    for i in range(nq):
                        t0 = time.perf_counter(); lab, _ = idx.knn_query(Qn[i].astype(np.float32), k=10)
                        lat.append(time.perf_counter() - t0); ids.append(lab[0])
                    run.add_row("per_seed_method", {**base, "method": f"hnswlib_ef{ef}", "n_queries": nq, "build_seconds": build,
                                                    "index_memory_bytes": int(rss_delta), "memory_note": "RSS delta around build",
                                                    **_lat_stats(lat), **quality(ids)})
                    run.add_rows("per_query_latency", [{**base, "method": f"hnswlib_ef{ef}", "q": i, "latency_s": l} for i, l in enumerate(lat)])
            if faiss is not None:
                t0 = time.perf_counter()
                fi = faiss.IndexHNSWFlat(V.shape[1], hn_cfg["M"], faiss.METRIC_INNER_PRODUCT)
                fi.hnsw.efConstruction = hn_cfg["ef_construction"]; fi.add(Mn.astype(np.float32)); build = time.perf_counter() - t0
                for ef in hn_cfg["ef_search"]:
                    fi.hnsw.efSearch = ef; lat = []; ids = []
                    for i in range(nq):
                        t0 = time.perf_counter(); _, lab = fi.search(Qn[i:i + 1].astype(np.float32), 10)
                        lat.append(time.perf_counter() - t0); ids.append(lab[0])
                    run.add_row("per_seed_method", {**base, "method": f"faiss_hnsw_ef{ef}", "n_queries": nq, "build_seconds": build,
                                                    "index_memory_bytes": None, **_lat_stats(lat), **quality(ids)})
        run.add_row("corpus_generation", {"seed": seed, "seconds": gen_s, "max_records": max(sizes)})
        run._flush()
    # aggregate across seeds
    agg = defaultdict(lambda: defaultdict(list))
    for r in run.tables["per_seed_method"]:
        for k in ("latency_ms_median", "latency_ms_p95", "latency_ms_p99", "qps_single_thread", "build_seconds",
                  "overlap_at_10", "recall_vs_exact_at_10"):
            agg[(r["method"], r["n_records"])][k].append(r[k])
    rows = [{"method": m, "n_records": n, **{f"{k}_mean": float(np.mean(v)) for k, v in d.items()},
             **{f"{k}_std": float(np.std(v, ddof=1)) if len(v) > 1 else None for k, v in d.items()}}
            for (m, n), d in sorted(agg.items())]
    run.save_table("summary_by_method_size", rows)
    run.finish()
    return run
