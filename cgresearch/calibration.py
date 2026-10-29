"""Benign-calibrated thresholds with finite-sample guarantees, and what happens to them under shift.

Everything here is about ONE decision rule:  alert  <=>  score(x) > tau,  with higher = more anomalous,
and tau chosen from *benign calibration scores only* so that the false-positive rate (FPR) is ~alpha.

Results used by the paper (each is verified by a unit test in tests/test_calibration.py):

R1  Split-conformal threshold (marginal).  Let s_(1) <= ... <= s_(n) be benign calibration scores and
    k = ceil((n+1)(1-alpha)).  If the calibration scores and a new benign score are exchangeable then
    tau = s_(k) gives  E[FPR] <= alpha  (and >= alpha - 1/(n+1) when scores are continuous).
    If k > n, no finite threshold has that guarantee and tau = +inf (never alert).

R2  Training-conditional (PAC) threshold.  For continuous exchangeable scores the realised FPR of tau = s_(k)
    is  1 - U_(k)  with  U_(k) ~ Beta(k, n+1-k).  Choosing the smallest k with
    P(1 - U_(k) > alpha) = BetaCDF(1-alpha; k, n+1-k) <= delta gives  P_cal( FPR > alpha ) <= delta.

R3  Exact shift decomposition.  For any threshold tau, with F_cal and F_dep the benign score CDFs at
    calibration and deployment time,
        FPR_dep(tau) = [1 - F_cal(tau)] + [F_cal(tau) - F_dep(tau)]  <=  alpha_cal + D+,
    where D+ = sup_t (F_cal(t) - F_dep(t)) is the one-sided Kolmogorov-Smirnov gap.  It is the exact quantity
    that inflates FPR, and it is estimable from scores alone.

R4  Contamination safety.  If the calibration set is a mixture (1-eps) F_b + eps F_a of benign and attack scores,
    and tau is the (1-alpha) quantile of the mixture, then
        FPR_b(tau) <= alpha + eps * D_dom,    D_dom = sup_t (F_a(t) - F_b(t))_+ .
    If attacks are stochastically larger than benign (D_dom = 0) a contaminated calibration set can only make
    the detector *more* conservative (FPR <= alpha), at a cost in recall.  When a detector scores some attacks as
    *more normal than benign* (D_dom > 0, e.g. the Isolation Forest behaviour on UNSW-NB15) the guarantee degrades
    by exactly eps * D_dom.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy.stats import beta as _beta


# ----------------------------------------------------------------------------- thresholds
def empirical_threshold(cal_scores: Sequence[float], alpha: float) -> float:
    """Plain empirical (1-alpha) quantile ('higher'): the v1 protocol. No finite-sample correction."""
    s = np.sort(np.asarray(cal_scores, float))
    if len(s) == 0:
        return float("inf")
    k = int(np.ceil((1.0 - alpha) * len(s))) - 1
    return float(s[min(max(k, 0), len(s) - 1)])


def conformal_rank(n: int, alpha: float) -> int:
    """k = ceil((n+1)(1-alpha)) (1-based rank). k > n means 'no finite threshold'."""
    return int(np.ceil((n + 1) * (1.0 - alpha) - 1e-12))


def pac_rank(n: int, alpha: float, delta: float) -> Optional[int]:
    """Smallest 1-based rank k with P(realised FPR > alpha) <= delta (R2). None if no k <= n qualifies."""
    if n <= 0:
        return None
    ks = np.arange(1, n + 1)
    cdf = _beta.cdf(1.0 - alpha, ks, n + 1 - ks)
    ok = np.nonzero(cdf <= delta)[0]
    return int(ks[ok[0]]) if len(ok) else None


def conformal_threshold(cal_scores: Sequence[float], alpha: float, delta: Optional[float] = None) -> float:
    """R1 (delta=None) or R2 (delta given). Returns +inf when the guarantee needs more calibration data."""
    s = np.sort(np.asarray(cal_scores, float))
    n = len(s)
    k = conformal_rank(n, alpha) if delta is None else pac_rank(n, alpha, delta)
    if k is None or k > n or k < 1:
        return float("inf")
    return float(s[k - 1])


def min_calibration_size(alpha: float, delta: Optional[float] = None) -> int:
    """Smallest n for which the guarantee is attainable with a finite threshold."""
    n = 1
    while n < 10_000_000:
        k = conformal_rank(n, alpha) if delta is None else pac_rank(n, alpha, delta)
        if k is not None and k <= n:
            return n
        n = n + 1 if n < 200 else int(n * 1.05) + 1
    return n


THRESH_FALLBACK = empirical_threshold   # used only when too few calibration points for a finite conformal threshold

THRESHOLD_METHODS = {
    "empirical": lambda s, a: empirical_threshold(s, a),
    "conformal": lambda s, a: conformal_threshold(s, a),
    "conformal_pac": lambda s, a: conformal_threshold(s, a, delta=0.05),
}


# ----------------------------------------------------------------------------- shift (R3)
def cdf_at(sorted_scores: np.ndarray, t) -> np.ndarray:
    """Empirical CDF P(S <= t) of an already-sorted sample."""
    return np.searchsorted(sorted_scores, t, side="right") / max(len(sorted_scores), 1)


def ks_plus(cal_scores: Sequence[float], dep_scores: Sequence[float]) -> float:
    """One-sided KS gap D+ = sup_t (F_cal(t) - F_dep(t)) >= 0 (positive = deployment scores drifted UP)."""
    a = np.sort(np.asarray(cal_scores, float))
    b = np.sort(np.asarray(dep_scores, float))
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    grid = np.unique(np.concatenate([a, b]))
    return float(max(0.0, np.max(cdf_at(a, grid) - cdf_at(b, grid))))


def dkw_slack(n: int, m: int, delta: float = 0.05) -> float:
    """High-probability slack on D+: with prob >= 1-delta,  D+_pop <= D+_hat + slack.
    Two-sided DKW (Massart) at level delta/2 for EACH sample: P(sup|F_n - F| > e) <= 2 exp(-2 n e^2), so e = sqrt(ln(4/delta)/(2n));
    the two failure events union to delta.  (v2 pre-release used ln(2/delta), which only gave 1-2*delta; fixed.)"""
    return float(np.sqrt(np.log(4.0 / delta) / (2.0 * n)) + np.sqrt(np.log(4.0 / delta) / (2.0 * m)))


def fpr_decomposition(cal_benign: Sequence[float], dep_benign: Sequence[float], tau: float) -> Dict[str, float]:
    """R3 with oracle benign deployment scores: realised FPR = alpha_cal + shift_term exactly."""
    a = np.sort(np.asarray(cal_benign, float))
    b = np.sort(np.asarray(dep_benign, float))
    f_cal = float(cdf_at(a, tau))
    f_dep = float(cdf_at(b, tau))
    return {"alpha_cal": 1.0 - f_cal, "shift_term": f_cal - f_dep, "realised_fpr": 1.0 - f_dep,
            "d_plus": ks_plus(a, b)}


# ----------------------------------------------------------------------------- contamination (R4)
def dominance_gap(benign_scores: Sequence[float], attack_scores: Sequence[float]) -> float:
    """D_dom = sup_t (F_attack(t) - F_benign(t))_+ : how far attacks score *below* benign. 0 = attacks dominate."""
    b = np.sort(np.asarray(benign_scores, float))
    a = np.sort(np.asarray(attack_scores, float))
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    grid = np.unique(np.concatenate([a, b]))
    return float(max(0.0, np.max(cdf_at(a, grid) - cdf_at(b, grid))))


def contaminate(benign_cal: np.ndarray, attack_pool: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    """Calibration sample of the same size whose fraction `eps` are attack scores (drawn with replacement
    from the pool) and the rest benign (R4 setting)."""
    n = len(benign_cal)
    if eps <= 0 or len(attack_pool) == 0:
        return np.asarray(benign_cal, float).copy()
    k = int(round(eps * n))
    keep = rng.choice(n, n - k, replace=False)
    att = rng.choice(len(attack_pool), k, replace=True)
    return np.concatenate([np.asarray(benign_cal, float)[keep], np.asarray(attack_pool, float)[att]])


def robust_alpha(alpha: float, eps_max: float) -> float:
    """Worst-case-safe working level when only an upper bound eps_max on contamination is known (D_dom <= 1):
    calibrating the mixture at alpha - eps_max keeps FPR <= alpha. Returns 0 when alpha <= eps_max (infeasible)."""
    return max(alpha - eps_max, 0.0)


# ----------------------------------------------------------------------------- evaluation helper
def alert_metrics(scores: np.ndarray, y: np.ndarray, tau: float) -> Dict[str, float]:
    pred = scores > tau
    pos = y == 1
    neg = ~pos
    tp = int((pred & pos).sum()); fp = int((pred & neg).sum())
    return {"recall": tp / pos.sum() if pos.any() else float("nan"),
            "fpr": fp / neg.sum() if neg.any() else float("nan"),
            "alerts": int(pred.sum()), "n": int(len(y)), "n_attack": int(pos.sum()), "n_benign": int(neg.sum()),
            "precision": tp / max(int(pred.sum()), 1)}


# ----------------------------------------------------------------------------- bulk-anchored recalibration (R5)
BAR_MODES = ("location", "affine", "loglinear", "local", "auto")


def _quantiles(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.quantile(np.asarray(x, float), u, method="linear")


def _fit_pairs(x: np.ndarray, y: np.ndarray, mode: str, c: np.ndarray, w: np.ndarray, local_span: float, u: np.ndarray):
    """Fit g on quantile pairs (x_u -> y_u). Returns (g, slope, intercept)."""
    if mode == "location":
        b = float(np.median(y - x))
        return (lambda z: np.asarray(z, float) + b), 1.0, b
    if mode == "affine":
        xs = x - x.mean()
        a = float(max((xs * (y - y.mean())).sum() / max((xs ** 2).sum(), 1e-300), 0.0)); b = float(y.mean() - a * x.mean())
        return (lambda z: a * np.asarray(z, float) + b), a, b
    if mode == "loglinear":                       # affine in log space: multiplicative / power-law drift
        m = float(min(c.min(), w.min())); sft = float(max(np.subtract(*np.quantile(c, [0.75, 0.25])), 1e-12)) * 0.1
        T = lambda z: np.log(np.maximum(np.asarray(z, float) - m, 0.0) + sft)
        tx, ty = T(x), T(y)
        xs = tx - tx.mean()
        a = float(max((xs * (ty - ty.mean())).sum() / max((xs ** 2).sum(), 1e-300), 0.0)); b = float(ty.mean() - a * tx.mean())
        return (lambda z: np.exp(a * T(z) + b) + m - sft), a, b
    ux = np.maximum.accumulate(x); uy = np.maximum.accumulate(y)       # local: piecewise-linear + linear extrapolation
    msk = u >= u[-1] - local_span
    dx, dy = ux[msk][-1] - ux[msk][0], uy[msk][-1] - uy[msk][0]
    a = float(max(dy / dx, 0.0)) if dx > 1e-300 else 1.0
    x_edge, y_edge = ux[-1], uy[-1]
    def g(z):
        z = np.asarray(z, float)
        return np.where(z <= x_edge, np.interp(z, ux, uy), y_edge + a * (z - x_edge))
    return g, a, float(y_edge - a * x_edge)


def bulk_anchored_map(cal_scores: Sequence[float], window_scores: Sequence[float], bulk: float = 0.7, lo: float = 0.05,
                      n_grid: int = 40, mode: str = "auto", local_span: float = 0.25, eps_hint: float = 0.0,
                      holdout: float = 0.15):
    """Estimate a monotone map g with  g(old benign score) ~ new benign score  from the LOWER `bulk` fraction of an
    UNLABELLED deployment window, where attacks (which sit in the upper tail) are rare, and extrapolate it into the tail.

    Modes: location (x+b) | affine (ax+b) | loglinear (affine in log space: multiplicative / power drift) |
           local (piecewise-linear in the bulk, linear extrapolation) |
           auto = pick among the four by LABEL-FREE extrapolation validation: fit on quantile levels [lo, bulk-holdout], score the
           prediction error on the held-out top `holdout` slice of the bulk (the closest thing to the tail we can check without labels),
           then refit the winner on the whole bulk.
    eps_hint > 0 undoes the upward bias attacks put on bulk quantiles, Q_dep(u) = Q_W(u (1-eps)). Joint estimation of eps from the
    bulk alone was tried and is NOT identifiable (eps is absorbed by the shift parameters), so it is an operator input (default 0 = conservative).
    """
    assert mode in BAR_MODES, mode
    c = np.asarray(cal_scores, float); w = np.asarray(window_scores, float)
    u = np.linspace(lo, bulk, n_grid)
    x, y = _quantiles(c, u), _quantiles(w, u * (1.0 - eps_hint))
    info_extra = {}
    if mode == "auto":
        k = int(np.searchsorted(u, bulk - holdout))
        scale = max(float(np.subtract(*np.quantile(w, [0.75, 0.25]))), 1e-12)
        errs = {}
        for m in ("location", "affine", "loglinear", "local"):
            g, _, _ = _fit_pairs(x[:k], y[:k], m, c, w, local_span, u[:k])
            errs[m] = float(np.sqrt(np.mean((g(x[k:]) - y[k:]) ** 2)) / scale)
        mode = min(errs, key=errs.get)
        info_extra = {"auto_errors": errs}
    g, a, b = _fit_pairs(x, y, mode, c, w, local_span, u)
    return g, {"mode": mode, "slope": a, "intercept": b, "bulk": bulk, **info_extra}


def bulk_anchored_transport(cal_scores: Sequence[float], window_scores: Sequence[float], **kw):
    """Transported calibration set g(C).  Its conformal threshold equals g(old threshold) (g is non-decreasing) and it can
    be used as the new reference sample for monitors.  Never uses labels.  The threshold is floored at the window's bulk edge."""
    g, info = bulk_anchored_map(cal_scores, window_scores, **kw)
    new = g(np.asarray(cal_scores, float))
    floor = float(np.quantile(np.asarray(window_scores, float), kw.get("bulk", 0.7)))
    new = np.maximum(new, floor)
    return new, info


def trimmed_threshold(window_scores: Sequence[float], alpha: float, eps_max: float) -> float:
    """Baseline needing a KNOWN contamination bound: drop the top eps_max fraction of the window, take the (1-alpha) quantile of the rest."""
    s = np.sort(np.asarray(window_scores, float))
    k = max(int(np.floor(len(s) * (1.0 - eps_max))), 1)
    return empirical_threshold(s[:k], alpha)


def bar_recalibrate(cal_scores, window_scores, alpha: float, threshold_fn=None, guard_eps_max=None, guard_slack: float = 0.01, **bar_kw):
    """Bulk-anchored recalibration with an optional label-free CONSISTENCY GUARD.

    The bulk cannot see a change that is confined to the upper tail (two deployment laws that agree on the bulk give identical bulk
    quantiles), so a transported threshold can silently under-shoot.  The guard uses an operator upper bound `guard_eps_max` on the
    attack share of the window: if BAR is right, at most alpha*(1-eps) + eps of the window can exceed the new threshold, so a window
    alert rate above alpha + guard_eps_max + guard_slack rejects the transport and falls back to the (conservative) raw-window
    reference, which sees the tail directly and is safe when attacks outscore benign traffic (R4).
    Returns (new_reference_sample, info)."""
    threshold_fn = threshold_fn or (lambda sc, a: conformal_threshold(sc, a))
    w = np.asarray(window_scores, float)
    new, info = bulk_anchored_transport(cal_scores, w, **bar_kw)
    info = dict(info); info["guard_triggered"] = False
    if guard_eps_max is not None:
        tau = threshold_fn(new, alpha)
        rate = float((w > tau).mean()) if np.isfinite(tau) else 0.0
        info["window_alert_rate"] = rate
        if rate > alpha + guard_eps_max + guard_slack:
            info["guard_triggered"] = True
            return w.copy(), info
    return new, info


# ----------------------------------------------------------------------------- weighted conformal baseline (Tibshirani et al. 2019, score-level)
def weighted_conformal_threshold(cal_scores: Sequence[float], window_scores: Sequence[float], alpha: float) -> float:
    """Baseline: covariate-shift-weighted conformal quantile. Density ratio dep/cal estimated by a logistic classifier on the score
    (features s, s^2) trained to separate calibration from UNLABELLED window scores; weighted (1-alpha) quantile of the calibration
    scores with 1/(n+1) mass at +inf.  Applied label-free, so the window's attacks contaminate the ratio (that is the point of the baseline)."""
    from sklearn.linear_model import LogisticRegression
    c = np.asarray(cal_scores, float); w = np.asarray(window_scores, float)
    mu, sd = np.concatenate([c, w]).mean(), max(np.concatenate([c, w]).std(), 1e-12)
    f = lambda z: np.c_[(z - mu) / sd, ((z - mu) / sd) ** 2]
    clf = LogisticRegression(C=10.0, max_iter=500).fit(np.vstack([f(c), f(w)]), np.r_[np.zeros(len(c)), np.ones(len(w))])
    p = np.clip(clf.predict_proba(f(c))[:, 1], 1e-6, 1 - 1e-6)
    wt = (len(c) / len(w)) * p / (1 - p)
    wt = np.minimum(wt, np.quantile(wt, 0.999))
    order = np.argsort(c); cs, ws = c[order], wt[order]
    mass = ws / (ws.sum() + ws.mean())                      # remaining mass sits at +inf (test point)
    cum = np.cumsum(mass)
    idx = int(np.searchsorted(cum, 1.0 - alpha, side="left"))
    return float(cs[idx]) if idx < len(cs) else float("inf")


# ----------------------------------------------------------------------------- label-free FPR certificate (R6)
def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    return 1.0 if k >= n else float(_beta.ppf(1.0 - delta, k + 1, n - k))


def fpr_certificate(cal_scores, window_scores, tau: float, eps_max: float, delta: float = 0.05, attacks_dominate: bool = True) -> Dict[str, float]:
    """High-probability UPPER bound (>= 1-delta) on the deployed BENIGN false-positive rate of threshold `tau` from UNLABELLED data only.
    Needs: tau was set from the calibration sample `cal_scores` at conformal rank k; the window holds at most a fraction eps_max of attacks.

    (a) alert-rate bound   : benign alerts <= all alerts, benign count >= (1-eps_max) n   =>  FPR <= CP_upper(alerts, n) / (1 - eps_max)
                             valid with NO assumption beyond eps_max.
    (b) shift bound (R3/R4): FPR_dep = alpha_cal + [F_cal - F_dep](tau) <= alpha_cal_ub + D+_benign,
                             D+_benign <= (D+_mix + eps_max * D_dom) / (1 - eps_max); D_dom = 0 if attacks dominate benign scores (assumed
                             when attacks_dominate=True), else <= 1.  alpha_cal_ub is the exact Beta upper quantile of 1-F_cal(tau); the
                             D+_mix estimation error uses the DKW slack.
    The two are combined by min; the confidence budget delta is split 1/2 (a) + 1/4 (Beta) + 1/4 (DKW) so the combined bound holds w.p. >= 1-delta."""
    c = np.sort(np.asarray(cal_scores, float)); w = np.asarray(window_scores, float)
    n, m = len(c), len(w)
    alerts = int((w > tau).sum())
    a_ub = clopper_pearson_upper(alerts, m, delta / 2) / max(1.0 - eps_max, 1e-9)
    k = int(np.searchsorted(c, tau, side="right"))          # number of calibration scores <= tau
    cal_alpha_ub = float(_beta.ppf(1.0 - delta / 4, max(n - k + 1, 1), max(k, 1))) if k >= 1 else 1.0
    d_mix = ks_plus(c, w)
    slack = dkw_slack(n, m, delta / 4)
    d_dom = 0.0 if attacks_dominate else 1.0
    s_ub = cal_alpha_ub + (d_mix + slack + eps_max * d_dom) / max(1.0 - eps_max, 1e-9)
    return {"alert_rate_bound": min(a_ub, 1.0), "shift_bound": min(s_ub, 1.0), "bound": min(a_ub, s_ub, 1.0),
            "alerts": alerts, "d_plus_mix": d_mix, "cal_alpha_ub": cal_alpha_ub, "slack": slack}
