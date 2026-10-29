"""Fail if score orientation or thresholding is inconsistent (plan item 2)."""
import numpy as np
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from cgresearch.detectors import REGISTRY
from cgresearch.metrics import calibrate_threshold, threshold_metrics


def _toy(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    Xb = rng.normal(0, 1, (n, 6)); Xa = rng.normal(0, 1, (300, 6)) + 4.0   # obvious, far-away attacks
    return Xb[:1000], Xb[1000:], Xa


@pytest.mark.parametrize("name", ["isolation_forest", "ocsvm", "lof", "autoencoder", "rule_based"])
def test_unsupervised_higher_means_more_anomalous(name):
    tr, te, att = _toy()
    d = REGISTRY[name](seed=0)
    d.fit(tr)
    sb, sa = d.score(te), d.score(att)
    assert sa.mean() > sb.mean(), f"{name}: attacks must score HIGHER"
    y = np.r_[np.zeros(len(sb)), np.ones(len(sa))]
    assert roc_auc_score(y, np.r_[sb, sa]) > 0.9


def test_sklearn_isolation_forest_orientation_facts():
    """Documents sklearn's convention: score_samples/decision_function are HIGHER = MORE NORMAL."""
    tr, te, att = _toy()
    m = IsolationForest(random_state=0).fit(tr)
    assert m.score_samples(att).mean() < m.score_samples(te).mean()
    assert m.decision_function(att).mean() < m.decision_function(te).mean()
    assert np.allclose(m.decision_function(te), m.score_samples(te) - m.offset_)
    # our contract negates it
    d = REGISTRY["isolation_forest"](seed=0).fit(tr)
    assert np.allclose(d.score(te), -IsolationForest(n_estimators=200, random_state=0).fit(tr).score_samples(te))


def test_wrong_orientation_is_caught():
    tr, te, att = _toy()
    m = IsolationForest(random_state=0).fit(tr)
    y = np.r_[np.zeros(len(te)), np.ones(len(att))]
    s = np.r_[m.score_samples(te), m.score_samples(att)]          # un-negated = BUG
    assert roc_auc_score(y, s) < 0.1                               # AUC far below 0.5 flags the bug
    thr = calibrate_threshold(s[: len(te)], 0.05)
    assert threshold_metrics(y, s, thr)["recall"] < 0.05           # and recall collapses


def test_threshold_calibration_measured_fpr_close_to_target():
    rng = np.random.default_rng(1)
    bv, bt = rng.random(5000), rng.random(20000)
    for f in (0.01, 0.02, 0.05, 0.10):
        thr = calibrate_threshold(bv, f)
        assert (bv > thr).mean() <= f + 1e-12
        y = np.zeros(len(bt)); m = threshold_metrics(y, bt, thr)
        assert abs(m["measured_fpr"] - f) < 0.01


def test_alert_rule_is_strictly_greater():
    s = np.array([0.1, 0.5, 0.5, 0.9]); y = np.array([0, 0, 1, 1])
    m = threshold_metrics(y, s, 0.5)
    assert m["alerts"] == 1 and m["tp"] == 1 and m["fp"] == 0


def test_ties_reported():
    from cgresearch.metrics import tie_fraction_at
    assert tie_fraction_at(np.array([1, 1, 1, 2.0]), 1.0) == 0.75


def test_end_to_end_detector_orientation_through_pipeline():
    """Full pipeline on synthetic data with strong separation: calibrated recall must be high."""
    from cgresearch.data.synthetic import make_synthetic_flows
    from cgresearch.data.splits import make_split
    from cgresearch.preprocess import FlowPreprocessor
    d = make_synthetic_flows(3000, seed=2, separation=3.0)
    sp = make_split(d.y, 0)
    pre = FlowPreprocessor("drop").fit(d, sp["benign_train"]); X = pre.transform(d)
    det = REGISTRY["lof"](seed=0).fit(X[sp["benign_train"]])
    thr = calibrate_threshold(det.score(X[sp["benign_val"]]), 0.05)
    m = threshold_metrics(d.y[sp["test"]], det.score(X[sp["test"]]), thr)
    assert m["recall"] > 0.3 and m["mcc"] > 0
