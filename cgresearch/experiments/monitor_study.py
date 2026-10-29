"""RQ7 -- Drift triggers and certificates, controlled score-level study (ground truth known).

A. Trigger comparison.  Reference sample from a base law; a stream of deployment scores that drifts at t0.  Triggers: our conformal power
   martingale (Ville false-alarm bound), our windowed certified-KS monitor, and the baselines two-sided windowed KS test, simplified ADWIN,
   Page-Hinkley, DDM on the alert-rate stream.  Baselines are TUNED so that their false-alarm rate on drift-free streams is <= `far_target`
   (a fair operating point); the martingale uses its guaranteed delta.  We report
     * false-alarm rate on drift-free streams,
     * detection rate and delay after HARMFUL drift (scores go up: FPR inflates),
     * alarm rate under HARMLESS drift (scores go down: FPR only falls) -- an alarm there is a wasted recalibration,
   against the true one-sided gap D+.
B. Delay prediction.  predicted_delay (label-free, from the reference and a post-drift window) vs the martingale's realised delay.
C. Certificate coverage.  fpr_certificate (calibration.py R6) vs the true deployed benign FPR: coverage (target >= 1-delta) and tightness.
"""
from __future__ import annotations

import zlib
from collections import defaultdict
from typing import Any, Dict

import numpy as np

from ..calibration import conformal_threshold, fpr_certificate, ks_plus
from ..monitor import (ADWINLite, ConformalMartingale, DDMAlertRate, KSMonitor, KSWindowTrigger, PageHinkley, first_alarm, predicted_delay)
from ..runlog import ExperimentRun
from ..stats_cluster import cluster_bootstrap_ci
from .recal_study import ATTACKS, BASES

HARMFUL = {"scale1.1": lambda x: 1.1 * x, "scale1.25": lambda x: 1.25 * x, "scale1.5": lambda x: 1.5 * x, "scale2.0": lambda x: 2.0 * x,
           "loc0.5": lambda x: x + 0.5, "loc1.0": lambda x: x + 1.0}
HARMLESS = {"scale0.8": lambda x: 0.8 * x, "scale0.6": lambda x: 0.6 * x}


class CertKS:
    """Our KSMonitor (R3 certificate) run on a sliding window, every `step` points."""
    def __init__(self, ref, window=500, step=100, gamma=0.02, delta=0.05):
        self.ref, self.window, self.step, self.mon = np.asarray(ref, float), window, step, KSMonitor(gamma, delta)
        self.buf, self.t, self.alarm_t = [], 0, None
    def update(self, x):
        self.t += 1; self.buf.append(float(x)); self.buf = self.buf[-self.window:]
        if self.alarm_t is None and len(self.buf) >= self.window and self.t % self.step == 0 and self.mon.check(self.ref, self.buf)["alarm"]:
            self.alarm_t = self.t
        return self.alarm_t is not None


def _make(name, ref, tau, par, delta):
    return {"martingale": lambda: ConformalMartingale(ref, delta=delta, seed=0), "cert_ks": lambda: CertKS(ref),
            "ks_test": lambda: KSWindowTrigger(ref, p_thr=par), "adwin": lambda: ADWINLite(ref, delta=par),
            "page_hinkley": lambda: PageHinkley(ref, lam=par), "ddm_alert": lambda: DDMAlertRate(ref, tau, k=par)}[name]()


# parameter grids ordered from LEAST to MOST sensitive
GRIDS = {"ks_test": [1e-9, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2], "adwin": [1e-9, 1e-7, 1e-5, 1e-3, 1e-2],
         "page_hinkley": [60.0, 40.0, 25.0, 15.0, 10.0, 6.0, 4.0], "ddm_alert": [20.0, 14.0, 10.0, 8.0, 6.0, 5.0, 4.0, 3.5, 3.0, 2.5]}


def _tune(name, ref, tau, null_streams, target, delta):
    chosen = GRIDS[name][0]
    for par in GRIDS[name]:
        fa = np.mean([first_alarm(_make(name, ref, tau, par, delta), st) is not None for st in null_streams])
        if fa <= target:
            chosen = par
        else:
            break
    return chosen


def run_monitor_study(cfg: Dict[str, Any], out_root="results", argv=None) -> ExperimentRun:
    run = ExperimentRun(cfg.get("name", "monitor_study"), cfg, out_root, argv, {"synthetic": True, "source": "score-level simulation"}, cfg["seeds"])
    alpha, delta = cfg.get("alpha", 0.05), cfg.get("mart_delta", 0.01)
    n_cal, L, t0 = cfg.get("n_cal", 2000), cfg.get("stream_len", 4000), cfg.get("t0", 1000)
    reps, n_tune, far = cfg.get("reps", 20), cfg.get("n_tune", 40), cfg.get("far_target", 0.01)
    eps_grid = cfg.get("eps", [0.0, 0.10])
    bases = {k: BASES[k] for k in cfg.get("bases", ["gamma", "lognormal"])}
    triggers = ["martingale", "cert_ks", "ks_test", "adwin", "page_hinkley", "ddm_alert"]
    rows_tr, rows_far, rows_delay = [], [], []
    for seed in cfg["seeds"]:
        for bname, bf in bases.items():
            rng0 = np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(bname.encode()), 1]))
            ref = bf(rng0, n_cal); tau = conformal_threshold(ref, alpha)
            tuned_streams = [bf(rng0, L) for _ in range(n_tune)]                       # drift-free tuning streams
            par = {nm: _tune(nm, ref, tau, tuned_streams, far, delta) for nm in GRIDS}
            par["martingale"] = par["cert_ks"] = None
            # ---- false-alarm rate on fresh drift-free streams
            null = [bf(rng0, L) for _ in range(reps * 2)]
            for nm in triggers:
                fa = [first_alarm(_make(nm, ref, tau, par[nm], delta), st) is not None for st in null]
                rows_far.append({"base": bname, "seed": seed, "trigger": nm, "param": par[nm], "false_alarm_rate": float(np.mean(fa)), "n_streams": len(fa)})
            # ---- drift scenarios
            for kind, table in (("harmful", HARMFUL), ("harmless", HARMLESS)):
                for dname, dmap in table.items():
                    big = bf(rng0, 100_000); dplus = ks_plus(ref, dmap(big))
                    dminus = ks_plus(dmap(big), ref)          # the opposite direction: how far scores moved DOWN
                    for eps in eps_grid:
                        scen = f"{bname}|{dname}|eps{eps}"
                        for rep in range(reps):
                            rng = np.random.default_rng(np.random.SeedSequence([seed, rep, zlib.crc32(scen.encode())]))
                            pre = bf(rng, t0); post_b = dmap(bf(rng, L - t0))
                            m = rng.random(L - t0) < eps
                            att = ATTACKS["strong"](rng, dmap(big), int(m.sum()))
                            post = post_b.copy(); post[m] = att
                            stream = np.concatenate([pre, post])
                            for nm in triggers:
                                tr = _make(nm, ref, tau, par[nm], delta)
                                a = first_alarm(tr, stream)
                                rows_tr.append({"scenario": scen, "base": bname, "drift": dname, "kind": kind, "eps": eps, "seed": seed, "rep": rep, "trigger": nm,
                                                "d_plus": dplus, "d_minus": dminus, "alarm_before_onset": a is not None and a <= t0,
                                                "detected": a is not None and a > t0, "delay": (a - t0) if (a is not None and a > t0) else np.nan,
                                                "alarm_any": a is not None})
                            if kind == "harmful" and eps == 0.0:
                                # ---- B: label-free delay prediction from a 1000-point post-drift window
                                tr = ConformalMartingale(ref, delta=delta, seed=rep)
                                a = first_alarm(tr, post)
                                pd_ = predicted_delay(ref, post[:1000], delta)
                                rows_delay.append({"scenario": scen, "drift": dname, "d_plus": dplus, "seed": seed, "rep": rep,
                                                   "realised_delay": float(a) if a is not None else np.nan, "predicted_delay": pd_})
        print(f"[{run.experiment}] seed {seed}: triggers done", flush=True)
    run.add_rows("trigger_runs", rows_tr); run.add_rows("false_alarm_runs", rows_far); run.add_rows("delay_prediction_runs", rows_delay)
    nb = cfg.get("n_boot", 5000)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows_tr:
        by[(r["trigger"], r["kind"], r["eps"])][r["scenario"]].append(r)
    summ = []
    for (tg, kind, eps), cl in sorted(by.items()):
        if kind == "harmful":
            mets = (("detection_rate", lambda x: float(x["detected"])), ("alarm_before_onset", lambda x: float(x["alarm_before_onset"])),
                    ("delay_detected", lambda x: x["delay"]))
        else:
            mets = (("alarm_rate_under_harmless_drift", lambda x: float(x["alarm_any"])),)
        for name, f in mets:
            vals = {c: [f(x) for x in v if not (isinstance(f(x), float) and np.isnan(f(x)))] for c, v in cl.items()}
            vals = {c: v for c, v in vals.items() if v}
            if vals:
                summ.append({"trigger": tg, "kind": kind, "eps": eps, "metric": name, **cluster_bootstrap_ci(vals, nb)})
    run.save_table("summary_triggers", summ)
    far_by = defaultdict(list)
    for r in rows_far:
        far_by[r["trigger"]].append(r["false_alarm_rate"])
    run.save_table("summary_false_alarm", [{"trigger": k, "mean_false_alarm_rate": float(np.mean(v)), "target": far if k not in ("martingale", "cert_ks") else (delta if k == "martingale" else None),
                                            "n_cells": len(v)} for k, v in far_by.items()])
    # B summary
    dd = defaultdict(list)
    for r in rows_delay:
        if np.isfinite(r["realised_delay"]) and np.isfinite(r["predicted_delay"]):
            dd[r["scenario"]].append(r["realised_delay"] / r["predicted_delay"])
    if dd:
        run.save_table("summary_delay_prediction", [{"scenario": k, "median_ratio_realised_over_predicted": float(np.median(v)), "n": len(v)} for k, v in sorted(dd.items())])
    # ---- C: certificate coverage
    cert_rows = []
    n_win = cfg.get("cert_window", 1000); cdelta = cfg.get("cert_delta", 0.05)
    for seed in cfg["seeds"]:
        for bname, bf in bases.items():
            for dname, dmap in {"none": lambda x: x, **HARMFUL, **HARMLESS}.items():
                for aname, af in ATTACKS.items():
                    for eps in (0.0, 0.02, 0.05, 0.10, 0.20):
                        for rep in range(cfg.get("cert_reps", 30)):
                            rng = np.random.default_rng(np.random.SeedSequence([seed, rep, zlib.crc32((bname + dname + aname).encode()), int(eps * 1000), 9]))
                            cal = bf(rng, n_cal); tau = conformal_threshold(cal, alpha)
                            ref = dmap(bf(rng, 100_000)); k = int(round(eps * n_win))
                            win = np.concatenate([dmap(bf(rng, n_win - k)), af(rng, ref, k) if k else np.zeros(0)])
                            true = float((ref > tau).mean())
                            for em in ("exact", "loose"):                 # operator's bound on the attack share: exact, or 2*eps+0.02
                                eps_max = eps if em == "exact" else 2 * eps + 0.02
                                for dom in (True, False):
                                    c = fpr_certificate(cal, win, tau, eps_max=eps_max, delta=cdelta, attacks_dominate=dom)
                                    cert_rows.append({"base": bname, "drift": dname, "attack": aname, "eps": eps, "eps_max_mode": em, "seed": seed, "rep": rep,
                                                      "assume_dominance": dom, "true_fpr": true, "bound": c["bound"], "alert_rate_bound": c["alert_rate_bound"],
                                                      "shift_bound": c["shift_bound"], "covered": true <= c["bound"] + 1e-12,
                                                      "shift_covered": true <= c["shift_bound"] + 1e-12, "alert_covered": true <= c["alert_rate_bound"] + 1e-12,
                                                      "slack_to_true": c["bound"] - true, "shift_slack": c["shift_bound"] - true, "alert_slack": c["alert_rate_bound"] - true})
    run.add_rows("certificate_runs", cert_rows)
    cb = defaultdict(lambda: defaultdict(list))
    for r in cert_rows:
        cb[(r["eps_max_mode"], r["assume_dominance"], r["attack"], r["eps"])][f'{r["base"]}|{r["drift"]}'].append(r)
    cs = []
    for (em, dom, at, eps), cl in sorted(cb.items()):
        for name, f in (("coverage", lambda x: float(x["covered"])), ("bound_minus_true", lambda x: x["slack_to_true"]),
                        ("shift_bound_coverage", lambda x: float(x["shift_covered"])), ("shift_bound_minus_true", lambda x: x["shift_slack"]),
                        ("alert_bound_coverage", lambda x: float(x["alert_covered"])), ("alert_bound_minus_true", lambda x: x["alert_slack"])):
            cs.append({"eps_max_mode": em, "assume_dominance": dom, "attack": at, "eps": eps, "metric": name, **cluster_bootstrap_ci({c: [f(x) for x in v] for c, v in cl.items()}, nb)})
    run.save_table("summary_certificate", cs)
    run.finish()
    return run
