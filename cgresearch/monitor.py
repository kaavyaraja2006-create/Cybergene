"""Label-free drift monitors for a benign-calibrated detector, and recalibration policies built on them.

ConformalMartingale
    Online smoothed conformal p-values against the growing pool (calibration scores + stream so far) are i.i.d.
    Uniform(0,1) when the stream is exchangeable with the calibration data (Vovk et al., 2005).  A simple-mixture
    power martingale  M_t = mean_eps prod_i eps * p_i^(eps-1)  therefore satisfies Ville's inequality:
        P( sup_t M_t >= 1/delta ) <= delta   under exchangeability,
    i.e. the alarm has a *guaranteed* false-alarm probability.  Small p-values (scores drifting UP) feed the
    martingale, which is the direction that inflates the false-positive rate.

KSMonitor (certified inflation)
    Uses R3 of calibration.py: realised FPR - alpha_cal <= D+.  Alarm when the *lower* confidence bound on D+
    computed from an UNLABELLED window exceeds a tolerance gamma.

Policies (what to do with the alarm) are evaluated in `run_policy`.  All policies see only scores for decisions;
labels are used only to *evaluate* (and by the oracle policy, which is an upper bound, not deployable).
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.special import logsumexp

from .calibration import (empirical_threshold, weighted_conformal_threshold, THRESH_FALLBACK, THRESHOLD_METHODS, alert_metrics, bar_recalibrate, bulk_anchored_transport, dkw_slack, ks_plus)


class ConformalMartingale:
    def __init__(self, init_scores: Sequence[float], delta: float = 0.01, eps_grid: Optional[Sequence[float]] = None,
                 seed: int = 0):
        self.delta = delta
        self.eps = np.asarray(eps_grid if eps_grid is not None else np.linspace(0.1, 0.9, 9), float)
        self.rng = np.random.default_rng(seed)
        self.reset(init_scores)

    def reset(self, init_scores: Sequence[float]):
        self.pool: List[float] = sorted(float(x) for x in init_scores)
        self.log_m = np.zeros(len(self.eps))
        self.t = 0
        self.alarm_t: Optional[int] = None
        self.log_mix = 0.0

    @property
    def alarmed(self) -> bool:
        return self.alarm_t is not None

    def pvalue(self, s: float) -> float:
        n = len(self.pool)
        gt = n - bisect_right(self.pool, s)
        eq = bisect_right(self.pool, s) - bisect_left(self.pool, s)
        theta = self.rng.random()
        return (gt + theta * (eq + 1)) / (n + 1)

    def update(self, s: float) -> bool:
        p = max(self.pvalue(s), 1e-12)
        insort(self.pool, float(s))
        self.log_m += np.log(self.eps) + (self.eps - 1.0) * np.log(p)
        self.log_mix = float(logsumexp(self.log_m) - np.log(len(self.eps)))
        self.t += 1
        if self.alarm_t is None and self.log_mix >= np.log(1.0 / self.delta):
            self.alarm_t = self.t
        return self.alarmed


def monitor_stream(stream: Sequence[float], init_scores: Sequence[float], delta: float = 0.01, seed: int = 0) -> Optional[int]:
    """Index (1-based) of the first alarm in `stream`, or None."""
    m = ConformalMartingale(init_scores, delta=delta, seed=seed)
    for s in stream:
        if m.update(float(s)):
            return m.alarm_t
    return None


class KSMonitor:
    """Certified FPR-inflation test on an unlabelled window."""
    def __init__(self, gamma: float = 0.02, delta: float = 0.05):
        self.gamma, self.delta = gamma, delta

    def check(self, cal_scores: Sequence[float], window_scores: Sequence[float]) -> Dict[str, float]:
        d = ks_plus(cal_scores, window_scores)
        slack = dkw_slack(len(cal_scores), len(window_scores), self.delta)
        return {"d_plus": d, "slack": slack, "lower": d - slack, "alarm": bool(d - slack > self.gamma)}


# ----------------------------------------------------------------------------- baseline drift triggers (all label-free, same update() API)
class _Ecdf:
    def __init__(self, ref): self.r = np.sort(np.asarray(ref, float))
    def __call__(self, x): return float(np.searchsorted(self.r, x, side="right")) / (len(self.r) + 1)


class KSWindowTrigger:
    """Two-sided two-sample KS test of the last `window` scores against the reference, evaluated every `step` points; alarm if p < p_thr."""
    def __init__(self, ref, window=500, step=100, p_thr=0.01):
        self.ref, self.window, self.step, self.p_thr = np.asarray(ref, float), window, step, p_thr
        self.buf, self.t, self.alarm_t = [], 0, None
    def update(self, x):
        from scipy.stats import ks_2samp
        self.t += 1; self.buf.append(float(x)); self.buf = self.buf[-self.window:]
        if self.alarm_t is None and len(self.buf) >= self.window and self.t % self.step == 0:
            if ks_2samp(self.ref, self.buf).pvalue < self.p_thr:
                self.alarm_t = self.t
        return self.alarm_t is not None


class ADWINLite:
    """Simplified ADWIN (Bifet & Gavalda 2007, Hoeffding cut bound) on rank-transformed scores in [0,1]; two-sided."""
    def __init__(self, ref, delta=0.002, max_len=3000, check_every=20, min_len=30):
        self.f, self.delta, self.max_len, self.every, self.min_len = _Ecdf(ref), delta, max_len, check_every, min_len
        self.w, self.t, self.alarm_t = [], 0, None
    def update(self, x):
        self.t += 1; self.w.append(self.f(x)); self.w = self.w[-self.max_len:]
        n = len(self.w)
        if self.alarm_t is None and n >= 2 * self.min_len and self.t % self.every == 0:
            cs = np.concatenate([[0.0], np.cumsum(self.w)])
            cuts = np.unique(np.linspace(self.min_len, n - self.min_len, 40).astype(int))
            n0, n1 = cuts, n - cuts
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            eps = np.sqrt(np.log(4.0 * n / self.delta) / (2.0 * m))
            if (np.abs(cs[cuts] / n0 - (cs[n] - cs[cuts]) / n1) > eps).any():
                self.alarm_t = self.t
        return self.alarm_t is not None


class PageHinkley:
    """Page-Hinkley (upward) on rank-transformed scores."""
    def __init__(self, ref, delta=0.005, lam=5.0):
        self.f, self.delta, self.lam = _Ecdf(ref), delta, lam
        self.t, self.mean, self.m, self.M, self.alarm_t = 0, 0.0, 0.0, 0.0, None
    def update(self, x):
        v = self.f(x); self.t += 1; self.mean += (v - self.mean) / self.t
        self.m += v - self.mean - self.delta; self.M = min(self.M, self.m)
        if self.alarm_t is None and self.m - self.M > self.lam:
            self.alarm_t = self.t
        return self.alarm_t is not None


class DDMAlertRate:
    """DDM (Gama et al. 2004) on the *alert* stream 1[score > tau] -- the label-free analogue of the error stream.
    The rate is Laplace-smoothed, p = (alerts+1)/(t+2): raw DDM has a degenerate minimum (p=0, s=0) whenever the first
    alerts arrive late, which at a 5% alert rate happens ~20% of the time and makes it fire on every drift-free stream."""
    def __init__(self, ref, tau, k=3.0, min_t=100):
        self.tau, self.k, self.min_t = float(tau), k, min_t
        self.t, self.a, self.pmin, self.smin, self.alarm_t = 0, 0, np.inf, np.inf, None
    def update(self, x):
        self.t += 1; self.a += int(x > self.tau)
        p = (self.a + 1.0) / (self.t + 2.0); s = np.sqrt(p * (1 - p) / self.t)
        if self.t >= self.min_t:
            if p + s < self.pmin + self.smin:
                self.pmin, self.smin = p, s
            if self.alarm_t is None and p + s > self.pmin + self.k * self.smin:
                self.alarm_t = self.t
        return self.alarm_t is not None


def first_alarm(trigger, stream) -> Optional[int]:
    for v in stream:
        if trigger.update(float(v)):
            return trigger.alarm_t
    return None


def martingale_growth_rate(cal_scores, window_scores, eps_grid=None, seed=0) -> float:
    """Label-free estimate of the per-sample log-growth of the conformal power martingale after drift:
    lambda = max_eps mean_i [ log eps + (eps-1) log p_i ] with p_i the conformal p-values of the window against the calibration sample.
    Predicted detection delay  ~ log(1/delta) / lambda  (Wald-type; validated empirically in monitor-study, no tight proof claimed)."""
    eps = np.asarray(eps_grid if eps_grid is not None else np.linspace(0.1, 0.9, 9), float)
    c = np.sort(np.asarray(cal_scores, float)); w = np.asarray(window_scores, float)
    rng = np.random.default_rng(seed)
    gt = len(c) - np.searchsorted(c, w, side="right"); eq = np.searchsorted(c, w, side="right") - np.searchsorted(c, w, side="left")
    p = np.maximum((gt + rng.random(len(w)) * (eq + 1)) / (len(c) + 1), 1e-12)
    lam = np.array([np.mean(np.log(e) + (e - 1.0) * np.log(p)) for e in eps])
    return float(lam.max())


def predicted_delay(cal_scores, window_scores, delta=0.01) -> float:
    lam = martingale_growth_rate(cal_scores, window_scores)
    return float(np.log(1.0 / delta) / lam) if lam > 1e-9 else float("inf")


# ----------------------------------------------------------------------------- recalibration policies
# trigger x recalibration rule.  "naive" = recompute the threshold from the (possibly attack-contaminated) window scores;
# "bar" = bulk-anchored transport of the ORIGINAL calibration sample (calibration.bulk_anchored_transport), label-free.
POLICIES = ["static", "oracle_periodic", "unlabeled_periodic", "martingale", "ks_certified",
            "periodic_bar", "martingale_bar", "ks_bar", "periodic_wcp", "aci_oracle"]
_SPEC = {"static": (None, None), "oracle_periodic": ("always", "oracle"), "unlabeled_periodic": ("always", "naive"),
         "martingale": ("martingale", "naive"), "ks_certified": ("ks", "naive"),
         "periodic_bar": ("always", "bar"), "martingale_bar": ("martingale", "bar"), "ks_bar": ("ks", "bar"),
         "periodic_wcp": ("always", "wcp"), "aci_oracle": ("always", "aci"),
         # diagnostics: reference sample drawn at random from the WHOLE window instead of its last `recal_size` points
         "oracle_random": ("always", "oracle_rand"), "naive_random": ("always", "naive_rand")}


def _new_threshold(scores: np.ndarray, alpha: float, method: str):
    tau = THRESHOLD_METHODS[method](scores, alpha)
    if not np.isfinite(tau):                       # too few calibration points for the guarantee
        tau = THRESH_FALLBACK(scores, alpha)
        return tau, True
    return tau, False


def run_policy(policy: str, cal_scores: np.ndarray, windows: List[Dict], alpha: float, method: str = "conformal",
               recal_size: int = 2000, mart_delta: float = 0.01, ks_gamma: float = 0.02, seed: int = 0,
               bar: Optional[Dict] = None) -> List[Dict]:
    """windows: [{'id':..., 'scores': time-ordered scores, 'y': 0/1 labels (evaluation only)}].
    Returns one row per window. Decisions for window w use only information from windows < w (and, for the
    martingale, the unlabelled scores seen so far in window w to time its alarm for the *next* window).
    `bar` = kwargs for bulk_anchored_transport (mode, bulk, lo, eps_hint ...)."""
    assert policy in _SPEC, policy
    trig, rule = _SPEC[policy]
    bar = {"mode": "auto", **{k: v for k, v in (bar or {}).items() if k != "aci_gamma"}}
    cal = np.asarray(cal_scores, float)
    cal0, alpha_t, aci_gamma = cal.copy(), alpha, (bar or {}).get("aci_gamma", 0.5)
    tau, fallback = _new_threshold(cal, alpha, method)
    mart = ConformalMartingale(cal, delta=mart_delta, seed=seed) if trig == "martingale" else None
    ksm = KSMonitor(gamma=ks_gamma) if trig == "ks" else None
    rows = []
    recal_flag = False                      # was the threshold refreshed *before* this window?
    for w in windows:
        s = np.asarray(w["scores"], float); y = np.asarray(w["y"], int)
        met = alert_metrics(s, y, tau)
        benign = s[y == 0]
        row = {"policy": policy, "window": w["id"], "tau": tau, "alpha": alpha, "method": method,
               "recalibrated_before": recal_flag,
               "d_plus_oracle": ks_plus(cal, benign) if len(benign) else float("nan"),
               "d_plus_unlabeled": ks_plus(cal, s), "n_cal": int(len(cal)), "fallback_empirical": fallback, **met}
        # ---- decide for the next window (UNLABELLED information only, except the oracle policy: an upper bound, not deployable)
        fire = False
        if trig == "always":
            fire = rule != "oracle" or len(benign) > 0
        elif trig == "martingale":
            alarm_at = None
            for t, v in enumerate(s, 1):
                if mart.update(float(v)) and alarm_at is None:
                    alarm_at = t
            row["alarm_at"] = alarm_at
            fire = alarm_at is not None
        elif trig == "ks":
            chk = ksm.check(cal, s)
            row["ks_lower"] = chk["lower"]
            fire = chk["alarm"]
        recal_flag = False
        if fire and rule == "wcp" and len(s) >= 20:         # weighted-conformal baseline: reference stays the ORIGINAL calibration sample
            tau = weighted_conformal_threshold(cal0, s, alpha)
            if not np.isfinite(tau):
                tau = empirical_threshold(cal0, alpha)
            recal_flag = True
        elif fire and rule == "aci" and len(benign) >= 20:   # Adaptive Conformal Inference, per-window step; USES benign labels (reference only)
            alpha_t = float(np.clip(alpha_t + aci_gamma * (alpha - met["fpr"]), 0.001, 0.5))
            tau = empirical_threshold(cal0, alpha_t)
            recal_flag = True
        elif fire and len(s) >= 20 and rule in ("oracle", "naive", "bar", "oracle_rand", "naive_rand"):
            if rule in ("oracle_rand", "naive_rand"):
                rng_ = np.random.default_rng(seed * 1009 + len(rows))
                pool = np.asarray(benign if rule == "oracle_rand" else s, float)
                new_cal = pool[rng_.permutation(len(pool))[:recal_size]] if len(pool) else pool
            elif rule == "oracle":
                new_cal = np.asarray(benign, float)[-recal_size:]
            elif rule == "naive":
                new_cal = s[-recal_size:]
            else:                                           # bar: transport the current reference sample onto the new regime
                bk = dict(bar); gk = {k: bk.pop(k) for k in ("guard_eps_max", "guard_slack") if k in bk}
                new_cal, info = bar_recalibrate(cal, s, alpha, lambda sc, a: _new_threshold(sc, a, method)[0], **gk, **bk)
                row["bar_slope"] = info["slope"]; row["bar_intercept"] = info["intercept"]; row["bar_mode"] = info["mode"]
                row["bar_guard_triggered"] = info["guard_triggered"]
            if len(new_cal) >= 20:
                cal = new_cal
                tau, fallback = _new_threshold(cal, alpha, method)
                if mart is not None:
                    mart.reset(cal)
                recal_flag = True
        row["recalibrated_after"] = recal_flag
        rows.append(row)
    return rows
