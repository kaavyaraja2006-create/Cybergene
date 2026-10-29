"""Validation-only hyperparameter selection. Test indices are never passed in here."""
from __future__ import annotations

import numpy as np

from ..metrics import threshold_free


def tune(det_cls, name, seed, Xtr, ytr, Xval, yval, feature_names=None, max_configs=6, **kw):
    """Select by validation PR-AUC. Returns (best_params, trials[list of dicts])."""
    probe = det_cls(seed=seed, **kw)
    grid = probe.param_grid()[:max_configs]
    trials, best, best_s = [], {}, -np.inf
    for p in grid:
        d = det_cls(seed=seed, **({"feature_names": feature_names} if feature_names is not None else {}))
        d.params = p
        d.fit(Xtr, ytr if d.supervised else None)
        s = threshold_free(yval, d.score(Xval))["pr_auc"]
        trials.append({"model": name, "params": p, "val_pr_auc": s})
        if s > best_s:
            best, best_s = p, s
    return best, trials


# =====================================================================================================
# v2 protocol: explicit grid mode and a LABEL-FREE selection criterion for unsupervised detectors.
# =====================================================================================================
from sklearn.metrics import roc_auc_score  # noqa: E402


def pseudo_anomalies(Xb: np.ndarray, seed: int) -> dict:
    """Synthetic anomalies built from BENIGN tuning data only (no attack labels):
    'perm'  : every column permuted independently (keeps marginals, destroys dependence between features)
    'shift' : a random 30% of columns shifted by +3 column standard deviations."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xA110]))
    perm = np.column_stack([rng.permutation(Xb[:, j]) for j in range(Xb.shape[1])])
    sd = Xb.std(0)
    cols = rng.random(Xb.shape[1]) < 0.3
    if not cols.any():
        cols[rng.integers(0, Xb.shape[1])] = True
    shift = Xb.copy()
    shift[:, cols] += 3.0 * np.maximum(sd[cols], 1e-6)
    return {"perm": perm, "shift": shift}


def pseudo_anomaly_score(det, Xb_tune: np.ndarray, seed: int) -> float:
    """Mean ROC-AUC of separating held-out benign data from pseudo anomalies (higher is better)."""
    sb = det.score(Xb_tune)
    aucs = []
    for P in pseudo_anomalies(Xb_tune, seed).values():
        sp = det.score(P)
        aucs.append(roc_auc_score(np.r_[np.zeros(len(sb)), np.ones(len(sp))], np.r_[sb, sp]))
    return float(np.mean(aucs))


def tune_v2(det_cls, name, seed, Xfit, yfit, Xtune_benign, Xtune_lab, ytune_lab, feature_names=None,
            mode: str = "random", n: int = 12, criterion: str = "pseudo_anomaly"):
    """Select hyper-parameters.
    criterion='labeled_prauc'  : validation PR-AUC on labelled tuning data (needs attack labels -> NOT unsupervised).
    criterion='pseudo_anomaly' : AUC against pseudo anomalies made from benign tuning data only (label-free).
    Supervised detectors always use 'labeled_prauc'. Returns (best_params, trials)."""
    configs = det_cls.config_list(mode, n, seed)
    sup = det_cls.supervised
    crit = "labeled_prauc" if sup else criterion
    trials, best, best_s = [], configs[0] if configs else {}, -np.inf
    for p in configs:
        kw = {"feature_names": feature_names} if feature_names is not None and name == "rule_based" else {}
        d = det_cls(seed=seed, **kw)
        d.params = p
        d.fit(Xfit, yfit if sup else None)
        if crit == "labeled_prauc":
            s = threshold_free(ytune_lab, d.score(Xtune_lab))["pr_auc"]
        elif crit == "pseudo_anomaly":
            s = pseudo_anomaly_score(d, Xtune_benign, seed)
        else:
            raise ValueError(crit)
        trials.append({"model": name, "params": p, "criterion": crit, "score": s})
        if s > best_s:
            best, best_s = p, s
    return best, trials
