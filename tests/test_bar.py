"""Bulk-anchored recalibration (calibration.py R5): what is guaranteed, what is conservative, and what is provably invisible."""
import numpy as np

from cgresearch.calibration import (bar_recalibrate, bulk_anchored_map, bulk_anchored_transport, conformal_threshold)
from cgresearch.monitor import run_policy

A, B, ALPHA = 1.6, 2.0, 0.05


def _data(seed, eps=0.0, n_cal=4000, n_win=4000, drift=lambda x: A * x + B):
    r = np.random.default_rng(seed)
    cal = r.gamma(3, 1, n_cal)
    ref = drift(r.gamma(3, 1, 200_000))                     # deployed benign law
    k = int(eps * n_win)
    win = np.r_[drift(r.gamma(3, 1, n_win - k)), drift(r.gamma(3, 1, k)) * 2 + 8]   # attacks dominate benign
    return cal, ref, win


def test_exact_recovery_under_affine_drift_without_contamination():
    errs = []
    for s in range(5):
        cal, ref, win = _data(s)
        for mode in ("affine", "local", "auto"):
            new, _ = bulk_anchored_transport(cal, win, mode=mode)
            errs.append(abs((ref > conformal_threshold(new, ALPHA)).mean() - ALPHA))
    assert max(errs) < 0.012


def test_transported_threshold_is_the_mapped_threshold():
    cal, _, win = _data(0)
    g, _ = bulk_anchored_map(cal, win, mode="affine")
    new, _ = bulk_anchored_transport(cal, win, mode="affine")
    assert abs(conformal_threshold(new, ALPHA) - g(conformal_threshold(cal, ALPHA))) < 1e-6
    assert (np.diff(g(np.sort(cal))) >= 0).all()             # monotone map


def test_conservative_under_dominating_contamination():
    for eps in (0.05, 0.10, 0.20):
        f = []
        for s in range(4):
            cal, ref, win = _data(s, eps)
            new, _ = bulk_anchored_transport(cal, win, mode="affine")
            f.append((ref > conformal_threshold(new, ALPHA)).mean())
        assert np.mean(f) <= ALPHA + 0.005, (eps, np.mean(f))


def test_far_better_than_static_and_raw_window_recall():
    cal, ref, win = _data(1, eps=0.2)
    tau_s = conformal_threshold(cal, ALPHA)
    new, _ = bulk_anchored_transport(cal, win, mode="affine")
    tau_b = conformal_threshold(new, ALPHA); tau_n = conformal_threshold(win, ALPHA)
    assert (ref > tau_s).mean() > 0.3                        # static is broken by the drift
    assert abs((ref > tau_b).mean() - ALPHA) < abs((ref > tau_s).mean() - ALPHA)
    assert tau_b < tau_n                                     # lower threshold than the contaminated-window quantile => more recall


def test_tail_only_drift_is_invisible_to_the_bulk_and_the_guard_catches_it():
    r = np.random.default_rng(3)
    cal = r.gamma(3, 1, 5000)
    base = r.gamma(3, 1, 5000)
    q = np.quantile(base, 0.8)
    tail = base + 0.9 * np.maximum(base - q, 0)               # same lowest 80%, heavier tail
    g1, i1 = bulk_anchored_map(cal, base, mode="affine"); g2, i2 = bulk_anchored_map(cal, tail, mode="affine")
    assert abs(i1["slope"] - i2["slope"]) < 1e-9 and abs(i1["intercept"] - i2["intercept"]) < 1e-9   # identical bulk => identical map
    _, info_ok = bar_recalibrate(cal, base, ALPHA, guard_eps_max=0.0, mode="affine")
    _, info_bad = bar_recalibrate(cal, tail, ALPHA, guard_eps_max=0.0, mode="affine")
    assert not info_ok["guard_triggered"] and info_bad["guard_triggered"]


def test_guard_quiet_when_nothing_changed():
    r = np.random.default_rng(5)
    for s in range(10):
        cal, win = r.gamma(3, 1, 3000), r.gamma(3, 1, 3000)
        _, info = bar_recalibrate(cal, win, ALPHA, guard_eps_max=0.0, mode="auto")
        assert not info["guard_triggered"]


def test_bar_policy_beats_static_under_drift():
    r = np.random.default_rng(0)
    cal = r.gamma(3, 1, 3000)
    wins = []
    for i, f in enumerate([1.0, 1.4, 1.9, 2.5]):
        b = f * r.gamma(3, 1, 3000)
        att = f * r.gamma(3, 1, 300) * 2 + 8
        s = np.r_[b, att]; y = np.r_[np.zeros(len(b), int), np.ones(len(att), int)]
        p = r.permutation(len(s)); wins.append({"id": f"w{i}", "scores": s[p], "y": y[p]})
    fpr = lambda rows: np.mean([x["fpr"] for x in rows[1:]])
    st = run_policy("static", cal, wins, ALPHA)
    mb = run_policy("martingale_bar", cal, wins, ALPHA, mart_delta=0.01)
    pb = run_policy("periodic_bar", cal, wins, ALPHA)
    assert fpr(pb) < fpr(st) and fpr(mb) < fpr(st)
    assert fpr(pb) < 0.08
