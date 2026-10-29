"""RQ4 -- What does attack contamination of the calibration set do to the FPR guarantee?

R4 (calibration.py): FPR <= alpha + eps * D_dom, with D_dom = sup_t (F_attack(t) - F_benign(t))_+ .
We contaminate the benign calibration scores with a fraction eps of attack scores (drawn from attack rows not in any
test window), apply the threshold method, and measure the realised benign FPR and recall on the test window.
`dominance` per attack family shows WHERE the bound degrades (families that score below benign traffic).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict

import numpy as np

from ..calibration import (THRESHOLD_METHODS, alert_metrics, contaminate, dkw_slack, dominance_gap, robust_alpha)
from ..foldscore import score_fold
from ..runlog import ExperimentRun
from ..scenarios import make_folds
from ..stats_cluster import cluster_bootstrap_ci
from ..foldscore import DEFAULT_TUNING
from .common import prepare
from ..detectors import REGISTRY


def run_contamination(cfg: Dict[str, Any], out_root="results", argv=None) -> ExperimentRun:
    data, dsinfo, dd = prepare(cfg)
    run = ExperimentRun(cfg.get("name", "contamination"), cfg, out_root, argv, dsinfo, cfg["seeds"])
    run.manifest["dedup"] = dd
    models = [m for m in cfg.get("models", ["isolation_forest", "pca", "hbos", "ecod", "autoencoder"]) if not REGISTRY[m].supervised]
    alphas = cfg.get("alphas", [0.05])
    eps_grid = cfg.get("eps", [0.0, 0.005, 0.01, 0.02, 0.05, 0.10])
    methods = cfg.get("methods", ["empirical", "conformal"])
    folds = make_folds(data, "random", cfg["seeds"], cfg.get("scenario", {}))
    rows, dom_rows = [], []
    for k, fold in enumerate(folds, 1):
        # attack scores for the pool need the fitted detector: score the pool as an extra window
        from ..scenarios import Window
        fold.windows.append(Window("__pool__", fold.attack_pool))
        scored = score_fold(data, fold, models, cfg.get("tuning", {}), cfg.get("port_mode", "onehot"), cfg.get("standardize", True), cfg.get("max_fit_rows"))
        fams = data.attack_cat[fold.attack_pool]
        for m, res in scored.items():
            cal = res["cal"]
            s_te, y_te = res["windows"]["test"]
            s_pool, _ = res["windows"]["__pool__"]
            d_all = dominance_gap(cal, s_pool)
            dom_rows.append({"model": m, "seed": fold.seed, "family": "ALL", "d_dom": d_all, "n_attack": int(len(s_pool))})
            for f in sorted(set(fams)):
                dom_rows.append({"model": m, "seed": fold.seed, "family": f, "d_dom": dominance_gap(cal, s_pool[fams == f]), "n_attack": int((fams == f).sum())})
            rng = np.random.default_rng(np.random.SeedSequence([fold.seed, 0xC0]))
            slack = dkw_slack(len(cal), max(len(cal), 1), 0.05)
            for a in alphas:
                for eps in eps_grid:
                    for meth in methods + ["robust_conformal"]:
                        cc = contaminate(cal, s_pool, eps, rng)
                        if meth == "robust_conformal":
                            ra = robust_alpha(a, eps)
                            tau = THRESHOLD_METHODS["conformal"](cc, ra) if ra > 0 else float("inf")
                        else:
                            tau = THRESHOLD_METHODS[meth](cc, a)
                        met = alert_metrics(s_te, y_te, tau)
                        bound = a + eps * d_all
                        rows.append({"model": m, "seed": fold.seed, "alpha": a, "eps": eps, "method": meth, "fpr": met["fpr"], "recall": met["recall"],
                                     "d_dom": d_all, "bound": bound, "dkw_slack": slack, "bound_holds": met["fpr"] <= bound + slack,
                                     "dominates": d_all <= 0.01, "tau_infinite": not np.isfinite(tau)})
        print(f"[{run.experiment}] fold {k}/{len(folds)} done", flush=True)
    run.add_rows("per_cell", rows)
    run.add_rows("dominance", dom_rows)
    n_boot = cfg.get("n_boot", 10_000)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[(r["model"], r["alpha"], r["eps"], r["method"])][r["seed"]].append(r)
    summ = []
    for (m, a, e, meth), cl in sorted(by.items()):
        for name, f in (("fpr", lambda x: x["fpr"]), ("recall", lambda x: x["recall"]), ("bound_holds", lambda x: float(x["bound_holds"]))):
            vals = {c: [f(x) for x in v if not np.isnan(f(x))] for c, v in cl.items()}
            summ.append({"model": m, "alpha": a, "eps": e, "method": meth, "metric": name, **cluster_bootstrap_ci(vals, n_boot)})
    run.save_table("summary", summ)
    run.finish()
    return run
