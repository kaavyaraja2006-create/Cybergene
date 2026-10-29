"""Shared engine: fit -> tune -> score one Fold for a list of detectors.

Roles (see scenarios.py): unsupervised detectors are fitted on fold.fit (benign), supervised ones on fold.sup;
hyper-parameters are chosen on fold.tune_* ; scores are produced for fold.cal and every evaluation window.
Thresholds are NOT chosen here: each experiment applies the threshold method(s) it studies to the returned scores.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .detectors import REGISTRY
from .detectors.tuning import tune_v2
from .preprocess import FlowPreprocessor
from .scenarios import Fold
from .data.unsw import FlowData

DEFAULT_TUNING = {"grid": "random", "n": 12, "criterion": "pseudo_anomaly"}


def score_fold(data: FlowData, fold: Fold, models: List[str], tuning: Optional[Dict[str, Any]] = None,
               port_mode: str = "onehot", standardize: bool = True, max_fit_rows: Optional[int] = None,
               keep_trials: bool = True, cache: Optional[dict] = None) -> Dict[str, Dict[str, Any]]:
    """cache: optional dict shared across folds of ONE run.  Unsupervised detectors are fitted on fold.fit only, so in leave-one-family-out
    (same benign split for every family) the fitted detector, its preprocessor and its label-free tuning are IDENTICAL across families:
    they are computed once per (seed, model) and reused (exact, not an approximation; tests/test_fast_cache.py)."""
    tuning = {**DEFAULT_TUNING, **(tuning or {})}
    y = data.y
    rng = np.random.default_rng(np.random.SeedSequence([fold.seed, 0xF17]))
    fit_idx = fold.fit
    if max_fit_rows and len(fit_idx) > max_fit_rows:
        fit_idx = np.sort(rng.choice(fit_idx, max_fit_rows, replace=False))
    def _h(a):
        return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
    if cache is not None and (cache.get("_seed") != fold.seed or cache.get("_data") is not data):   # one seed, one dataset object at a time
        cache.clear(); cache["_seed"] = fold.seed; cache["_data"] = data
    pkey = ("pre", fold.seed, _h(fit_idx), port_mode, standardize)
    if cache is not None and pkey in cache:
        pre = cache[pkey]
    else:
        pre = FlowPreprocessor(port_mode, standardize=standardize).fit(data, fit_idx)
        if cache is not None:
            cache[pkey] = pre
    need = np.unique(np.concatenate([fold.fit, fold.tune_benign, fold.tune_labeled, fold.cal, fold.sup] + [w.idx for w in fold.windows]))
    X = np.zeros((len(y), 0)); Xn = pre.transform(data.subset(need))
    X = np.zeros((len(y), Xn.shape[1])); X[need] = Xn
    out: Dict[str, Dict[str, Any]] = {}
    for m in models:
        cls = REGISTRY[m]
        sup = cls.supervised
        if sup and len(fold.sup) == 0:
            continue
        fn = pre.num_cols if m == "rule_based" else None
        t0 = time.perf_counter()
        ckey = None
        if cache is not None and not sup:
            ckey = ("det", m, fold.seed, _h(fit_idx), _h(fold.tune_benign), tuple(sorted(tuning.items())), max_fit_rows,
                    _h(fold.tune_labeled) if tuning["criterion"] != "pseudo_anomaly" else "", port_mode, standardize)
        if ckey is not None and ckey in cache:
            det, best, trials = cache[ckey]
        else:
            tr_idx = fold.sup if sup else fit_idx
            if sup and max_fit_rows and len(tr_idx) > 4 * max_fit_rows:
                tr_idx = np.sort(rng.choice(tr_idx, 4 * max_fit_rows, replace=False))
            if cls.grid or cls.ext_grid:
                best, trials = tune_v2(cls, m, fold.seed, X[tr_idx], y[tr_idx] if sup else None, X[fold.tune_benign], X[fold.tune_labeled],
                                       y[fold.tune_labeled], fn, tuning["grid"], int(tuning["n"]), tuning["criterion"])
            else:
                best, trials = {}, []
            det = cls(seed=fold.seed, **({"feature_names": fn} if fn is not None else {}))
            det.params = best
            det.fit(X[tr_idx], y[tr_idx] if sup else None)
            if ckey is not None:
                cache[ckey] = (det, best, trials)
        res = {"params": best, "supervised": sup, "cal": det.score(X[fold.cal]),
               "windows": {w.id: (det.score(X[w.idx]), y[w.idx]) for w in fold.windows},
               "fit_seconds": time.perf_counter() - t0}
        if keep_trials:
            res["trials"] = trials
        out[m] = res
    return out
