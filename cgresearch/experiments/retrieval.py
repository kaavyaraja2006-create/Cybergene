"""Attack DNA retrieval ablation (experiment 8): which genomes help similarity search?

Identical stored corpus, query set, metric (cosine), seeds across all genome configurations.
Queries are held-out records (never in the corpus)."""
from __future__ import annotations

import time
import tracemalloc
from collections import defaultdict
from typing import Any, Dict

import numpy as np

from ..runlog import ExperimentRun
from ..stats import paired_comparisons, summarize
from .corpus import make_corpus, relevance

GENOME_CONFIGS = {
    "network": ("network",),
    "network+behavior": ("network", "behavior"),
    "network+behavior+resource": ("network", "behavior", "resource"),
    "all_five": ("network", "behavior", "resource", "context", "evolution"),
}


def l2norm(M):
    return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)


def rank_metrics(rel_row: np.ndarray, order: np.ndarray, n_rel_total: int, ks=(1, 5, 10)) -> Dict[str, float]:
    r = rel_row[order[:10]]
    out = {f"P@{k}": float((r[:k] >= 1).mean()) for k in ks}
    for k in (5, 10):
        out[f"R@{k}"] = float((r[:k] >= 1).sum() / max(n_rel_total, 1))
    hit = np.nonzero(rel_row[order] >= 1)[0]
    out["MRR"] = float(1.0 / (hit[0] + 1)) if len(hit) else 0.0
    disc = 1.0 / np.log2(np.arange(2, 12))
    gains = 2.0 ** r - 1
    ideal = np.sort(2.0 ** rel_row - 1)[::-1][:10]
    out["nDCG@10"] = float((gains * disc[: len(r)]).sum() / max((ideal * disc[: len(ideal)]).sum(), 1e-12))
    return out


def run_retrieval(cfg: Dict[str, Any], out_root="results", argv=None) -> ExperimentRun:
    run = ExperimentRun(cfg.get("name", "retrieval_ablation"), cfg, out_root, argv,
                        {"synthetic": True, "note": "generator-labelled Attack DNA corpus"}, cfg["seeds"])
    n_store, n_q = cfg.get("n_store", 3000), cfg.get("n_queries", 200)
    per = defaultdict(lambda: defaultdict(dict))
    for seed in cfg["seeds"]:
        store, st = make_corpus(n_store, seed)
        queries, qt = make_corpus(n_q, seed + 10_000)
        rel = relevance(qt, st)
        for cname, genomes in GENOME_CONFIGS.items():
            tracemalloc.start()
            M = l2norm(np.vstack([d.vector(genomes) for d in store]))
            Q = l2norm(np.vstack([q.vector(genomes) for q in queries]))
            mem_now, _ = tracemalloc.get_traced_memory(); tracemalloc.stop()
            agg = defaultdict(list); lat = []
            for i in range(n_q):
                t0 = time.perf_counter()
                sims = M @ Q[i]
                top = np.argpartition(-sims, 10)[:10]
                top = top[np.argsort(-sims[top], kind="stable")]
                lat.append(time.perf_counter() - t0)
                # full ordering only needed for MRR beyond top-10: use full sort
                full = np.argsort(-sims, kind="stable")
                m = rank_metrics(rel[i], full, int((rel[i] >= 1).sum()))
                for k, v in m.items():
                    agg[k].append(v)
            row = {"seed": seed, "config": cname, "dim": int(M.shape[1]), "n_store": n_store, "n_queries": n_q,
                   **{k: float(np.mean(v)) for k, v in agg.items()},
                   "latency_ms_median": 1e3 * float(np.median(lat)), "latency_ms_p95": 1e3 * float(np.percentile(lat, 95)),
                   "index_memory_bytes": int(M.nbytes), "traced_alloc_bytes": int(mem_now)}
            run.add_row("per_seed", row)
            for k in ("P@1", "P@5", "P@10", "R@5", "R@10", "MRR", "nDCG@10"):
                per[k][cname][seed] = row[k]
    nb = cfg.get("n_boot", 10_000)
    summ, comps = [], []
    for k, byc in per.items():
        for c, d in byc.items():
            summ.append({"metric": k, "config": c, **summarize([d[s] for s in sorted(d)], nb), "per_seed": [d[s] for s in sorted(d)]})
        comps += paired_comparisons(byc, k, [(b, "network") for b in byc if b != "network"] +
                                    [("network+behavior+resource", "network+behavior"), ("all_five", "network+behavior+resource")])
    run.save_table("summary", summ); run.save_table("comparisons", comps)
    run.finish()
    return run
