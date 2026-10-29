"""RQ3 -- Can a label-free monitor detect calibration failure, and does acting on it help?

Temporal scenario: calibrate on the first block, deploy on later blocks (days).  Five policies are compared
(monitor.POLICIES).  Only 'oracle_periodic' uses benign labels (an upper bound).  Also reported:
  * false_alarm_control : the martingale on an EXCHANGEABLE stream (shuffled benign test data) -- Ville bound check
  * predictive validity : does the unlabelled one-sided KS gap D+ predict the FPR inflation of the static detector?
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict

import numpy as np
from scipy.stats import spearmanr

from ..foldscore import score_fold
from ..monitor import POLICIES, monitor_stream, run_policy
from ..runlog import ExperimentRun
from ..scenarios import make_folds
from ..stats_cluster import cluster_bootstrap_ci
from .common import prepare


def run_shift_monitor(cfg: Dict[str, Any], out_root="results", argv=None) -> ExperimentRun:
    data, dsinfo, dd = prepare(cfg)
    run = ExperimentRun(cfg.get("name", "shift_monitor"), cfg, out_root, argv, dsinfo, cfg["seeds"])
    run.manifest["dedup"] = dd
    models = cfg.get("models", ["isolation_forest", "pca", "hbos", "ecod", "autoencoder"])
    alphas = cfg.get("alphas", [0.05])
    method = cfg.get("method", "conformal")
    scen = {"kind": "temporal", **cfg.get("temporal", {})}
    folds = make_folds(data, "temporal", cfg["seeds"], scen)
    rows, fa_rows = [], []
    for k, fold in enumerate(folds, 1):
        scored = score_fold(data, fold, models, cfg.get("tuning", {}), cfg.get("port_mode", "onehot"), cfg.get("standardize", True), cfg.get("max_fit_rows"))
        for m, res in scored.items():
            wins = [{"id": wid, "scores": s, "y": y} for wid, (s, y) in res["windows"].items()]
            # false-alarm control: exchangeable stream = shuffled benign scores of ALL windows' benign part is NOT exchangeable,
            # so use the control window (held-out benign from the calibration block) repeated by resampling with the cal pool
            ctrl = res["windows"].get("control")
            if ctrl is not None:
                s_ctrl = ctrl[0][ctrl[1] == 0]
                rng = np.random.default_rng(fold.seed)
                pool = np.concatenate([res["cal"], s_ctrl])
                for rep in range(cfg.get("fa_reps", 20)):
                    perm = rng.permutation(len(pool))
                    init, stream = pool[perm[: len(res["cal"])]], pool[perm[len(res["cal"]):]]
                    al = monitor_stream(stream, init, cfg.get("mart_delta", 0.01), seed=rep)
                    fa_rows.append({"model": m, "seed": fold.seed, "rep": rep, "stream_len": int(len(stream)), "false_alarm": al is not None})
            for a in alphas:
                for pol in cfg.get("policies", POLICIES):
                    rr = run_policy(pol, res["cal"], [dict(w) for w in wins], a, method, cfg.get("recal_size", 2000),
                                    cfg.get("mart_delta", 0.01), cfg.get("ks_gamma", 0.02), seed=fold.seed, bar=cfg.get("bar"))
                    for r in rr:
                        rows.append({**r, "model": m, "seed": fold.seed, "supervised": res["supervised"]})
        print(f"[{run.experiment}] temporal fold {k}/{len(folds)} done", flush=True)
    run.add_rows("per_window_policy", rows)
    run.add_rows("false_alarm_control", fa_rows)
    # ---- summaries (cluster = window id)
    n_boot = cfg.get("n_boot", 10_000)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["window"] == "control":
            continue
        by[(r["model"], r["alpha"], r["policy"])][r["window"]].append(r)
    summ = []
    for (m, a, pol), cl in sorted(by.items()):
        for name, f in (("fpr", lambda x: x["fpr"]), ("recall", lambda x: x["recall"]), ("violation_rate(fpr>alpha)", lambda x: float(x["fpr"] > x["alpha"])),
                        ("recalibrations_per_window", lambda x: float(x["recalibrated_after"]))):
            vals = {c: [f(x) for x in v if not (isinstance(f(x), float) and np.isnan(f(x)))] for c, v in cl.items()}
            summ.append({"model": m, "alpha": a, "policy": pol, "metric": name, **cluster_bootstrap_ci(vals, n_boot)})
    run.save_table("summary_policies", summ)
    fa = defaultdict(list)
    for r in fa_rows:
        fa[r["model"]].append(float(r["false_alarm"]))
    run.save_table("summary_false_alarm", [{"model": m, "delta": cfg.get("mart_delta", 0.01), "false_alarm_rate": float(np.mean(v)), "n_streams": len(v)} for m, v in fa.items()])
    # ---- predictive validity of the unlabelled KS gap
    pv = []
    for m in models:
        st = [r for r in rows if r["model"] == m and r["policy"] == "static" and r["window"] != "control" and np.isfinite(r["d_plus_unlabeled"])]
        if len(st) >= 4:
            infl = [r["fpr"] - r["alpha"] for r in st]
            rho_u = spearmanr([r["d_plus_unlabeled"] for r in st], infl).statistic
            rho_o = spearmanr([r["d_plus_oracle"] for r in st], infl).statistic
            pv.append({"model": m, "n_windows": len(st), "spearman_dplus_unlabeled_vs_inflation": float(rho_u),
                       "spearman_dplus_oracle_vs_inflation": float(rho_o)})
    run.save_table("predictive_validity", pv)
    run.finish()
    return run
