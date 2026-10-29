"""Monte-Carlo and exact checks of the guarantees stated in cgresearch/calibration.py and monitor.py."""
import numpy as np
from scipy.stats import norm

from cgresearch import calibration as C
from cgresearch.monitor import ConformalMartingale, KSMonitor, monitor_stream, run_policy


def test_r1_conformal_marginal_fpr_is_valid_and_tight():
    rng = np.random.default_rng(0)
    n, alpha, trials = 200, 0.05, 4000
    fprs = np.array([1 - norm.cdf(C.conformal_threshold(rng.normal(size=n), alpha)) for _ in range(trials)])
    assert fprs.mean() <= alpha + 0.002                      # valid
    assert fprs.mean() >= alpha - 1 / (n + 1) - 0.002        # and not wastefully conservative
    emp = np.array([1 - norm.cdf(C.empirical_threshold(rng.normal(size=n), alpha)) for _ in range(trials)])
    assert emp.mean() > fprs.mean() - 1e-9                   # v1 empirical quantile is never more conservative


def test_r2_pac_threshold_controls_violation_probability():
    rng = np.random.default_rng(1)
    n, alpha, delta, trials = 300, 0.05, 0.10, 3000
    viol_pac = np.mean([(1 - norm.cdf(C.conformal_threshold(rng.normal(size=n), alpha, delta))) > alpha for _ in range(trials)])
    viol_emp = np.mean([(1 - norm.cdf(C.empirical_threshold(rng.normal(size=n), alpha))) > alpha for _ in range(trials)])
    assert viol_pac <= delta + 0.03
    assert viol_emp > 0.3            # the naive quantile violates the target in roughly half of the calibration draws


def test_infinite_threshold_when_calibration_too_small():
    assert C.conformal_threshold(np.arange(10.0), 0.05) == float("inf")
    assert C.min_calibration_size(0.05) == 19 and C.min_calibration_size(0.05, 0.05) == 59
    assert np.isfinite(C.conformal_threshold(np.arange(19.0), 0.05))


def test_r3_decomposition_is_exact_and_bounded_by_ks():
    rng = np.random.default_rng(2)
    cal, dep = rng.normal(0, 1, 5000), rng.normal(0.4, 1.2, 7000)
    for alpha in (0.01, 0.05, 0.1):
        tau = C.conformal_threshold(cal, alpha)
        d = C.fpr_decomposition(cal, dep, tau)
        assert abs(d["realised_fpr"] - (d["alpha_cal"] + d["shift_term"])) < 1e-12
        assert d["realised_fpr"] <= d["alpha_cal"] + d["d_plus"] + 1e-12
    assert C.ks_plus(cal, cal) == 0.0
    assert C.ks_plus(dep, cal) == 0.0 or C.ks_plus(dep, cal) < C.ks_plus(cal, dep)   # one-sided: drift UP only counts


def test_r4_contamination_safe_when_attacks_dominate_and_degrades_by_eps_dgap_otherwise():
    rng = np.random.default_rng(3)
    n, alpha = 400_000, 0.05
    ben = rng.normal(0, 1, n)
    for mu, eps in ((2.0, 0.10), (-1.5, 0.10), (-1.5, 0.05)):
        att = rng.normal(mu, 1, n)
        cal = C.contaminate(ben, att, eps, rng)
        tau = C.empirical_threshold(cal, alpha)
        fpr = 1 - norm.cdf(tau)
        dgap = C.dominance_gap(ben, att)
        if mu > 0:
            assert dgap == 0.0 or dgap < 0.01
            assert fpr <= alpha + 2e-3                       # contamination only makes the detector more conservative
        else:
            assert dgap > 0.5
            assert fpr <= alpha + eps * dgap + 2e-3          # R4 bound
            assert fpr > alpha + 1e-3                        # and it is violated: the bound is not vacuous
    assert abs(C.robust_alpha(0.05, 0.02) - 0.03) < 1e-12 and C.robust_alpha(0.05, 0.08) == 0.0


def test_martingale_false_alarm_probability_is_below_delta_and_power_is_high():
    rng = np.random.default_rng(4)
    delta, trials = 0.05, 250
    fa = np.mean([monitor_stream(rng.normal(size=400), rng.normal(size=300), delta, seed=i) is not None for i in range(trials)])
    assert fa <= delta + 0.04                                # Ville: <= delta under exchangeability
    delays = [monitor_stream(rng.normal(1.0, 1, 400), rng.normal(size=300), delta, seed=i) for i in range(60)]
    assert all(d is not None for d in delays) and np.median(delays) < 100


def test_martingale_ignores_downward_drift_and_handles_ties():
    rng = np.random.default_rng(5)
    assert monitor_stream(rng.normal(-1.0, 1, 300), rng.normal(size=300), 0.01, seed=1) is None
    m = ConformalMartingale(np.zeros(50), delta=0.01)       # all ties: p-values still valid (smoothed)
    for _ in range(100):
        m.update(0.0)
    assert not m.alarmed


def test_ks_monitor_certifies_only_real_inflation():
    rng = np.random.default_rng(6)
    mon = KSMonitor(gamma=0.02, delta=0.05)
    assert not mon.check(rng.normal(size=3000), rng.normal(size=3000))["alarm"]
    assert mon.check(rng.normal(size=3000), rng.normal(0.5, 1, 3000))["alarm"]


def test_policies_static_inflates_oracle_restores():
    rng = np.random.default_rng(7)
    cal = rng.normal(0, 1, 3000)
    wins = []
    for i, shift in enumerate([0.0, 0.3, 0.8, 1.2]):
        s = np.r_[rng.normal(shift, 1, 3000), rng.normal(shift + 4, 1, 300)]
        y = np.r_[np.zeros(3000, int), np.ones(300, int)]
        order = rng.permutation(len(s))
        wins.append({"id": i, "scores": s[order], "y": y[order]})
    st = run_policy("static", cal, [dict(w) for w in wins], 0.05)
    orc = run_policy("oracle_periodic", cal, [dict(w) for w in wins], 0.05)
    mar = run_policy("martingale", cal, [dict(w) for w in wins], 0.05)
    assert st[3]["fpr"] > 0.2 and st[0]["fpr"] < 0.08
    assert orc[3]["fpr"] < st[3]["fpr"] - 0.1
    assert mar[3]["fpr"] < st[3]["fpr"] - 0.05 and any(r["recalibrated_after"] for r in mar)
