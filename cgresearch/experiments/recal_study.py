"""RQ6 -- Controlled study of label-free recalibration under shift and contamination (score level, ground truth known).

Scores are drawn directly so that the TRUE deployment benign distribution F_dep is known and the realised FPR can be computed to
~1e-3 precision.  Factors: base score law x drift type x attack type x contamination eps.  Methods:
  static          : keep the calibration threshold                 (no adaptation)
  naive           : conformal threshold of the raw window         (contaminated recalibration; the v2-smoke failure mode)
  trim_known_eps  : drop the top eps of the window (needs eps known exactly)
  trim_2x_eps     : same with a 2x over-estimate of eps               (realistic operator error)
  robust_alpha    : conformal at alpha - eps on the raw window     (R4 worst-case-safe; infeasible when eps >= alpha)
  oracle          : conformal on the benign part of the window     (needs labels; upper bound)
  bar_location / bar_affine / bar_loglinear / bar_local / bar_auto (label-free extrapolation-validated choice) / bar_auto_hint : bulk-anchored transport (calibration.py R5), label-free
Inference clusters = scenarios (base x drift x attack), never replicates.
"""
from __future__ import annotations

import zlib
from collections import defaultdict
from typing import Any, Dict

import numpy as np

from ..calibration import (alert_metrics, bar_recalibrate, bulk_anchored_transport, conformal_threshold, robust_alpha, trimmed_threshold)
from ..runlog import ExperimentRun
from ..stats_cluster import cluster_bootstrap_ci

BASES = {
    "gamma": lambda r, n: r.gamma(3.0, 1.0, n),
    "lognormal": lambda r, n: r.lognormal(0.0, 0.6, n),
    "bimodal": lambda r, n: np.where(r.random(n) < 0.7, r.gamma(3.0, 1.0, n), r.gamma(3.0, 1.0, n) + 4.0),
}
DRIFTS = {                                   # monotone maps applied to benign scores at deployment
    "none": lambda x: x,
    "location": lambda x: x + 1.5,
    "scale": lambda x: 1.6 * x,
    "affine": lambda x: 1.4 * x + 1.0,
    "power": lambda x: 0.35 * x ** 1.35,      # non-affine
    "tail": None,                              # only the upper tail drifts (bulk unchanged); defined inside the loop from a reference quantile
}
ATTACKS = {                                   # attack scores relative to the DEPLOYED benign law (callable of benign deployment scores)
    "strong": lambda r, xb, n: r.choice(xb, n) * 1.8 + 6.0,
    "moderate": lambda r, xb, n: r.choice(xb, n) * 1.3 + 2.0,
    "overlap": lambda r, xb, n: r.choice(xb, n) * 0.8 + 0.5,     # many attacks score BELOW typical benign: D_dom > 0
}
METHODS = ["static", "naive", "trim_known_eps", "trim_2x_eps", "robust_alpha", "oracle",
           "bar_location", "bar_affine", "bar_loglinear", "bar_local", "bar_auto", "bar_auto_hint", "bar_auto_guard"]


def tau_for(method, cal, win, win_benign, alpha, eps):
    if method == "static":
        return conformal_threshold(cal, alpha)
    if method == "naive":
        return conformal_threshold(win, alpha)
    if method == "trim_known_eps":
        return trimmed_threshold(win, alpha, eps)
    if method == "trim_2x_eps":
        return trimmed_threshold(win, alpha, min(2 * eps, 0.9))
    if method == "robust_alpha":
        ra = robust_alpha(alpha, eps)
        return conformal_threshold(win, ra) if ra > 0 else float("inf")
    if method == "oracle":
        return conformal_threshold(win_benign, alpha)
    kw = {"bar_location": {"mode": "location"}, "bar_affine": {"mode": "affine"}, "bar_loglinear": {"mode": "loglinear"},
          "bar_local": {"mode": "local"}, "bar_auto": {"mode": "auto"}, "bar_auto_guard": {}, "bar_auto_hint": {"mode": "auto", "eps_hint": eps / 2.0}}[method]
    if method == "bar_auto_guard":                # operator supplies an upper bound on contamination (here: true eps + 0.02)
        new, _ = bar_recalibrate(cal, win, alpha, guard_eps_max=eps + 0.02, mode="auto")
    else:
        new, _ = bulk_anchored_transport(cal, win, **kw)
    return conformal_threshold(new, alpha)


def run_recal_study(cfg: Dict[str, Any], out_root="results", argv=None) -> ExperimentRun:
    run = ExperimentRun(cfg.get("name", "recal_study"), cfg, out_root, argv, {"synthetic": True, "source": "score-level simulation"}, cfg["seeds"])
    alpha = cfg.get("alpha", 0.05); n_cal = cfg.get("n_cal", 2000); n_win = cfg.get("n_win", 3000); reps = cfg.get("reps", 50)
    eps_grid = cfg.get("eps", [0.0, 0.02, 0.05, 0.10, 0.20])
    rows = []
    for si, seed in enumerate(cfg["seeds"]):
        for bname, bf in BASES.items():
            for dname, df in DRIFTS.items():
                for aname, af in ATTACKS.items():
                    for eps in eps_grid:
                        for rep in range(reps):
                            rng = np.random.default_rng(np.random.SeedSequence([seed, si, rep, zlib.crc32((bname + dname + aname).encode()), int(eps * 1000)]))
                            cal = bf(rng, n_cal)
                            ref = df(bf(rng, 50_000)) if dname != "tail" else None
                            if dname == "tail":                         # tail map is defined by a fixed reference quantile
                                q80 = np.quantile(bf(rng, 50_000), 0.8)
                                f = lambda x, q=q80: x + 0.9 * np.maximum(x - q, 0.0)
                                ref = f(bf(rng, 50_000)); dep_b = f(bf(rng, n_win))
                            else:
                                dep_b = df(bf(rng, n_win))
                            k = int(round(eps * n_win))
                            att = af(rng, ref, k) if k else np.zeros(0)
                            win_b = dep_b[: n_win - k]
                            win = np.concatenate([win_b, att]); rng.shuffle(win)
                            test_att = af(rng, ref, 4000)
                            for m in METHODS:
                                tau = tau_for(m, cal, win, win_b, alpha, eps)
                                fpr = float((ref > tau).mean())
                                rec = float((test_att > tau).mean())
                                rows.append({"scenario": f"{bname}|{dname}|{aname}", "base": bname, "drift": dname, "attack": aname, "eps": eps, "seed": seed, "rep": rep,
                                             "method": m, "fpr": fpr, "recall": rec, "tau_inf": not np.isfinite(tau)})
        print(f"[{run.experiment}] seed {seed} done", flush=True)
    run.add_rows("per_rep", rows)
    # summaries: cluster = scenario (replicates and seeds averaged inside)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[(r["method"], r["eps"], r["drift"] != "none")][r["scenario"]].append(r)
    summ = []
    for (m, eps, drifted), cl in sorted(by.items()):
        for name, f in (("fpr_over_alpha", lambda x: x["fpr"] / alpha), ("abs_fpr_error_over_alpha", lambda x: abs(x["fpr"] - alpha) / alpha),
                        ("violation(fpr>alpha)", lambda x: float(x["fpr"] > alpha)), ("material_violation(fpr>1.5alpha)", lambda x: float(x["fpr"] > 1.5 * alpha)),
                        ("recall", lambda x: x["recall"])):
            summ.append({"method": m, "eps": eps, "drifted": drifted, "metric": name,
                         **cluster_bootstrap_ci({c: [f(x) for x in v] for c, v in cl.items()}, cfg.get("n_boot", 5000))})
    run.save_table("summary", summ)
    # attack-type split (D_dom > 0 case) for the headline method comparison
    by2 = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["drift"] != "none":
            by2[(r["method"], r["attack"], r["eps"])][f'{r["base"]}|{r["drift"]}'].append(r)
    s2 = []
    for (m, a, eps), cl in sorted(by2.items()):
        for name, f in (("fpr_over_alpha", lambda x: x["fpr"] / alpha), ("recall", lambda x: x["recall"])):
            s2.append({"method": m, "attack": a, "eps": eps, "metric": name, **cluster_bootstrap_ci({c: [f(x) for x in v] for c, v in cl.items()}, cfg.get("n_boot", 5000))})
    run.save_table("summary_by_attack", s2)
    run.finish()
    return run
