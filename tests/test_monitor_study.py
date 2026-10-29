import numpy as np

from cgresearch.calibration import conformal_threshold, fpr_certificate, weighted_conformal_threshold
from cgresearch.monitor import (ADWINLite, DDMAlertRate, KSWindowTrigger, PageHinkley, first_alarm, martingale_growth_rate, predicted_delay,
                                ConformalMartingale, run_policy)


def test_baseline_triggers_quiet_without_drift_and_fire_with_drift():
    r = np.random.default_rng(0)
    ref = r.gamma(3, 1, 3000); tau = conformal_threshold(ref, 0.05)
    mk = lambda: [KSWindowTrigger(ref, p_thr=1e-5), ADWINLite(ref, delta=1e-5), PageHinkley(ref, lam=40.0), DDMAlertRate(ref, tau, k=10.0)]
    null = [first_alarm(t, r.gamma(3, 1, 3000)) for t in mk()]
    assert all(a is None for a in null), null
    drift = np.r_[r.gamma(3, 1, 500), 1.6 * r.gamma(3, 1, 3000)]
    got = [first_alarm(t, drift) for t in mk()]
    assert all(a is not None for a in got), got


def test_delay_prediction_is_within_a_small_factor():
    ratios = []
    for s in range(12):
        r = np.random.default_rng(s)
        ref = r.gamma(3, 1, 2000); post = 1.5 * r.gamma(3, 1, 3000)
        a = first_alarm(ConformalMartingale(ref, delta=0.01, seed=s), post)
        ratios.append(a / predicted_delay(ref, post[:1000], 0.01))
    assert 0.4 < np.median(ratios) < 2.5, np.median(ratios)
    assert martingale_growth_rate(ref, r.gamma(3, 1, 1000)) < 0.02          # no drift -> no growth


def test_certificate_coverage_and_validity():
    """The unlabelled certificate must cover the true deployed benign FPR in >= 1-delta of replicates (both bounds, with and without drift)."""
    cov, n = 0, 0
    for s in range(60):
        r = np.random.default_rng(s)
        for drift in (lambda x: x, lambda x: 1.4 * x + 0.5):
            cal = r.gamma(3, 1, 2000); tau = conformal_threshold(cal, 0.05)
            ref = drift(r.gamma(3, 1, 100_000)); eps = 0.1; k = 100
            win = np.r_[drift(r.gamma(3, 1, 900)), ref[:k] * 2 + 8]
            c = fpr_certificate(cal, win, tau, eps_max=eps, delta=0.05)
            cov += (ref > tau).mean() <= c["bound"]; n += 1
    assert cov / n >= 0.95, cov / n


def test_certificate_alert_bound_needs_no_dominance():
    r = np.random.default_rng(1)
    cal = r.gamma(3, 1, 2000); tau = conformal_threshold(cal, 0.05)
    ref = 1.5 * r.gamma(3, 1, 100_000)
    win = np.r_[1.5 * r.gamma(3, 1, 900), 0.3 * r.gamma(3, 1, 100)]         # attacks score BELOW benign (dominance violated)
    c = fpr_certificate(cal, win, tau, eps_max=0.1, delta=0.05, attacks_dominate=False)
    assert (ref > tau).mean() <= c["alert_rate_bound"]


def test_wcp_and_aci_policies_run():
    r = np.random.default_rng(0)
    cal = r.gamma(3, 1, 2000)
    wins = []
    for i, f in enumerate([1.0, 1.4, 1.9]):
        b = f * r.gamma(3, 1, 2000); a = f * r.gamma(3, 1, 150) * 2 + 8
        s = np.r_[b, a]; y = np.r_[np.zeros(len(b), int), np.ones(len(a), int)]
        p = r.permutation(len(s)); wins.append({"id": f"w{i}", "scores": s[p], "y": y[p]})
    for pol in ("periodic_wcp", "aci_oracle"):
        rows = run_policy(pol, cal, wins, 0.05)
        assert len(rows) == 3 and all(np.isfinite(x["tau"]) for x in rows)
    st = run_policy("static", cal, wins, 0.05)
    ac = run_policy("aci_oracle", cal, wins, 0.05)
    assert ac[-1]["fpr"] < st[-1]["fpr"]
    assert np.isfinite(weighted_conformal_threshold(cal, wins[1]["scores"], 0.05))
