"""RQ1/RQ2 -- Does a benign-calibrated threshold keep its false-positive rate when the deployment data shifts?

For every scenario (random / leave-one-family-out / temporal), detector, target alpha and threshold method
(empirical v1 quantile, split-conformal, PAC-conformal) we record the REALISED benign FPR in every evaluation
window together with the exact decomposition  realised = alpha_cal + shift_term  and the one-sided KS gap D+
(calibration.py, R3).  Inference is over windows/families (clusters), never over seeds.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ..calibration import THRESHOLD_METHODS, alert_metrics, fpr_decomposition
from ..foldscore import score_fold
from ..runlog import ExperimentRun
from ..scenarios import make_folds
from ..stats_cluster import cluster_bootstrap_ci
from .common import prepare

DEFAULT_MODELS = ["isolation_forest", "ocsvm", "lof", "autoencoder", "pca", "hbos", "ecod", "kitnet_style", "logreg", "lightgbm"]


def window_rows(fold, scored, alphas, methods, scen_name):
    rows = []
    for m, res in scored.items():
        cal = res["cal"]
        for wid, (s, yw) in res["windows"].items():
            both = len(np.unique(yw)) == 2
            auc = float(roc_auc_score(yw, s)) if both else np.nan
            pr = float(average_precision_score(yw, s)) if both else np.nan
            ben = s[yw == 0]
            for a in alphas:
                for meth in methods:
                    tau = THRESHOLD_METHODS[meth](cal, a)
                    met = alert_metrics(s, yw, tau)
                    dec = fpr_decomposition(cal, ben, tau) if np.isfinite(tau) and len(ben) else {"alpha_cal": np.nan, "shift_term": np.nan, "realised_fpr": np.nan, "d_plus": np.nan}
                    rows.append({"scenario": scen_name, "kind": fold.kind, "fold": fold.name, "seed": fold.seed, "window": wid, "model": m,
                                 "supervised": res["supervised"], "alpha": a, "method": meth, "n_cal": int(len(cal)), "tau": tau,
                                 "tau_infinite": not np.isfinite(tau), "fpr": met["fpr"], "recall": met["recall"], "precision": met["precision"],
                                 "alerts": met["alerts"], "n_benign": met["n_benign"], "n_attack": met["n_attack"],
                                 "alpha_cal": dec["alpha_cal"], "shift_term": dec["shift_term"], "d_plus": dec["d_plus"],
                                 "roc_auc": auc, "pr_auc": pr})
    return rows


def summarise(rows, n_boot):
    """Cluster = (dataset implied) scenario/window id; values = per-seed observations."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["tau_infinite"] and r["method"] != "empirical":
            continue
        key = (r["kind"], r["model"], r["alpha"], r["method"])
        cl = f'{r["window"]}'
        by[key][cl].append(r)
    summ = []
    for (kind, model, a, meth), cl in sorted(by.items()):
        def agg(f):
            return {c: [f(x) for x in v if not (isinstance(f(x), float) and np.isnan(f(x)))] for c, v in cl.items()}
        fpr = agg(lambda x: x["fpr"]); rec = agg(lambda x: x["recall"])
        over = agg(lambda x: float(x["fpr"] > x["alpha"]))
        material = agg(lambda x: float(x["fpr"] > 1.5 * x["alpha"]))
        base = {"kind": kind, "model": model, "alpha": a, "method": meth}
        summ.append({**base, "metric": "fpr", **cluster_bootstrap_ci(fpr, n_boot)})
        summ.append({**base, "metric": "recall", **cluster_bootstrap_ci(rec, n_boot)})
        summ.append({**base, "metric": "violation_rate(fpr>alpha)", **cluster_bootstrap_ci(over, n_boot)})
        summ.append({**base, "metric": "material_violation_rate(fpr>1.5alpha)", **cluster_bootstrap_ci(material, n_boot)})
    return summ


_CACHE: dict = {}      # per-process cache shared by the folds of a run (see foldscore.score_fold)


def run_calib_transfer(cfg: Dict[str, Any], out_root="results", argv=None) -> ExperimentRun:
    data, dsinfo, dd = prepare(cfg)
    run = ExperimentRun(cfg.get("name", "calib_transfer"), cfg, out_root, argv, dsinfo, cfg["seeds"])
    run.manifest["dedup"] = dd
    models = cfg.get("models", DEFAULT_MODELS)
    alphas = cfg.get("alphas", [0.01, 0.05])
    methods = cfg.get("methods", ["empirical", "conformal", "conformal_pac"])
    tuning = cfg.get("tuning", {})
    rows, trials = [], []
    for scen in cfg["scenarios"]:
        kind = scen["kind"]
        folds = make_folds(data, kind, cfg["seeds"], scen)
        for k, fold in enumerate(folds, 1):
            scored = score_fold(data, fold, models, tuning, cfg.get("port_mode", "onehot"), cfg.get("standardize", True), cfg.get("max_fit_rows"), cache=_CACHE)
            rows += window_rows(fold, scored, alphas, methods, scen.get("name", kind))
            for m, res in scored.items():
                trials += [{"fold": fold.name, "seed": fold.seed, **t, "params": str(t["params"])} for t in res.get("trials", [])]
            print(f"[{run.experiment}] {kind} fold {k}/{len(folds)} ({fold.name}) done", flush=True)
    run.add_rows("per_window", rows)
    run.add_rows("tuning_trials", trials)
    run.save_table("summary", summarise(rows, cfg.get("n_boot", 10_000)))
    ok = [r for r in rows if np.isfinite(r["alpha_cal"]) and not r["tau_infinite"]]
    ident = max((abs(r["fpr"] - (r["alpha_cal"] + r["shift_term"])) for r in ok), default=0.0)
    bound = float(np.mean([r["fpr"] <= r["alpha_cal"] + r["d_plus"] + 1e-12 for r in ok])) if ok else None
    run.manifest["identity_max_abs_error"] = ident
    run.manifest["ks_bound_holds_fraction"] = bound
    run.finish()
    return run
