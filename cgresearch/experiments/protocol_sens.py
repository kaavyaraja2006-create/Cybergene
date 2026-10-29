"""RQ5 -- Do evaluation-protocol choices change which detector 'wins'?

Factors (crossed):  data protocol (dedup before vs after sampling)  x  tuning protocol  x  scenario.
Tuning protocols: v1 'first_k' grid selected on LABELLED PR-AUC (the criticised v1 behaviour), the same with a full random grid,
and label-free pseudo-anomaly tuning.  Metric: PR-AUC and recall at the conformal alpha-threshold, per (scenario, window, seed) cell.
Output: per-cell metrics, per-protocol Friedman/Nemenyi summaries, and rank stability (Kendall tau, winner flips) vs the v1 protocol.
"""
from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Dict

import numpy as np
from sklearn.metrics import average_precision_score

from ..calibration import THRESHOLD_METHODS, alert_metrics
from ..detectors import REGISTRY
from ..foldscore import score_fold
from ..runlog import ExperimentRun
from ..scenarios import make_folds
from ..stats_cluster import rank_stability, rank_summary
from .common import prepare

DEFAULT_PROTOCOLS = {
    "v1_firstk_labeled": {"grid": "first_k", "n": 12, "criterion": "labeled_prauc"},
    "random_labeled": {"grid": "random", "n": 12, "criterion": "labeled_prauc"},
    "random_labelfree": {"grid": "random", "n": 12, "criterion": "pseudo_anomaly"},
}
REFERENCE = "v1_firstk_labeled"


def _matrix(cells, models, key):
    """cells: {cell_id: {model: value}} -> (ids, matrix) with only complete rows."""
    ids = [c for c, v in cells.items() if all(m in v and np.isfinite(v[m]) for m in models)]
    return ids, np.array([[cells[c][m] for m in models] for c in ids])


_CACHE: dict = {}      # per-process cache shared by the folds of a run (see foldscore.score_fold)


def run_protocol_sens(cfg: Dict[str, Any], out_root="results", argv=None) -> ExperimentRun:
    dedup_variants = cfg.get("dedup_variants", [{"dedup_first": False}])
    data0, dsinfo, dd0 = prepare({**cfg, **dedup_variants[0]})
    run = ExperimentRun(cfg.get("name", "protocol_sens"), cfg, out_root, argv, dsinfo, cfg["seeds"])
    run.manifest["dedup"] = dd0
    models = [m for m in cfg.get("models", ["isolation_forest", "pca", "hbos", "ecod", "autoencoder", "lof"]) if not REGISTRY[m].supervised]
    protocols = cfg.get("protocols", DEFAULT_PROTOCOLS)
    scenarios = cfg.get("scenarios", [{"kind": "random"}, {"kind": "lofo", "min_support": 50, "max_families": 6}, {"kind": "temporal"}])
    alpha = cfg.get("alpha", 0.05)
    rows = []
    for dv in dedup_variants:
        dname = "dedup_first" if dv.get("dedup_first") else "dedup_after"
        data = data0 if dv is dedup_variants[0] else prepare({**cfg, **dv})[0]
        for sc in scenarios:
            folds = make_folds(data, sc["kind"], cfg["seeds"], sc)
            for k, fold in enumerate(folds, 1):
                for pname, tun in protocols.items():
                    scored = score_fold(data, fold, models, tun, cfg.get("port_mode", "onehot"), cfg.get("standardize", True), cfg.get("max_fit_rows"), keep_trials=False, cache=_CACHE)
                    for m, res in scored.items():
                        for wid, (s, y) in res["windows"].items():
                            if wid == "control" or len(np.unique(y)) < 2:
                                continue
                            tau = THRESHOLD_METHODS["conformal"](res["cal"], alpha)
                            met = alert_metrics(s, y, tau)
                            rows.append({"data_protocol": dname, "scenario": sc["kind"], "protocol": pname, "fold": fold.name, "seed": fold.seed,
                                         "window": wid, "model": m, "pr_auc": float(average_precision_score(y, s)), "recall": met["recall"], "fpr": met["fpr"],
                                         "params": str(res["params"])})
                print(f"[{run.experiment}] {dname}/{sc['kind']} fold {k}/{len(folds)}", flush=True)
    run.add_rows("per_cell", rows)
    # ---- rankings per (data_protocol, scenario) over cells = (fold, window, seed); compare tuning protocols with the reference
    summ, stab = [], []
    groups = defaultdict(list)
    for r in rows:
        groups[(r["data_protocol"], r["scenario"])].append(r)
    for (dp, sc), rr in sorted(groups.items()):
        mats = {}
        for p in protocols:
            cells = defaultdict(dict)
            for r in rr:
                if r["protocol"] == p:
                    cells[(r["fold"], r["window"], r["seed"])][r["model"]] = r["pr_auc"]
            mats[p] = _matrix(cells, models, "pr_auc")
        common = None
        for p, (ids, _) in mats.items():
            common = set(ids) if common is None else common & set(ids)
        common = sorted(common or [])
        if len(common) < 3:
            continue
        aligned = {}
        for p, (ids, M) in mats.items():
            pos = {c: i for i, c in enumerate(ids)}
            aligned[p] = M[[pos[c] for c in common]]
        for p, M in aligned.items():
            s = rank_summary(M, models)
            best = int(np.argmin(s["avg_rank"]))
            summ.append({"data_protocol": dp, "scenario": sc, "protocol": p, "winner": models[best], **{k: v for k, v in s.items() if k not in ("methods", "avg_rank")},
                         **{f"rank_{m}": float(r) for m, r in zip(models, s["avg_rank"])}})
        if REFERENCE in aligned:
            for p, M in aligned.items():
                if p != REFERENCE:
                    stab.append({"data_protocol": dp, "scenario": sc, "protocol": p, "vs": REFERENCE, **rank_stability(aligned[REFERENCE], M)})
    run.save_table("summary_rankings", summ)
    run.save_table("rank_stability", stab)
    # ---- accuracy effect of the protocol on the SAME detector (mean PR-AUC by protocol)
    eff = defaultdict(list)
    for r in rows:
        eff[(r["data_protocol"], r["scenario"], r["model"], r["protocol"])].append(r["pr_auc"])
    run.save_table("pr_auc_by_protocol", [{"data_protocol": a, "scenario": b, "model": c, "protocol": d, "mean_pr_auc": float(np.mean(v)), "n_cells": len(v)}
                                          for (a, b, c, d), v in sorted(eff.items())])
    run.finish()
    return run
