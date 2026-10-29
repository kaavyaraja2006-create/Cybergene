"""CalibratedSentinelAgent -- the prototype Sentinel with a finite-sample-calibrated alert threshold and a drift monitor.

Differences from backend/sentinel_agent.SentinelAgent:
  * the alert rule is `score > tau`, tau = split-conformal (1-alpha) rank of BENIGN calibration scores held out from fitting
    (the original uses IsolationForest(contamination=0.15).predict, i.e. an assumed 15% anomaly rate);
  * a ConformalMartingale watches the unlabelled session-score stream; on alarm the threshold is re-calibrated from the most
    recent unlabelled scores (attacks in that pool can only make tau more conservative when attacks outscore benign, R4).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import IsolationForest

from ._backend import BACKEND  # noqa: F401
from sentinel_agent import FEATURE_NAMES, SentinelAgent, session_features  # noqa: E402
from .calibration import bar_recalibrate, conformal_threshold, empirical_threshold
from .monitor import ConformalMartingale


class CalibratedSentinelAgent(SentinelAgent):
    def __init__(self, alpha: float = 0.05, monitor: bool = True, mart_delta: float = 0.01, recal_size: int = 500,
                 cal_frac: float = 0.4, seed: int = 42, recal_mode: str = "naive", guard_eps_max=None, eps_hint: float = 0.0):
        super().__init__(contamination=0.15)
        self.alpha, self.use_monitor, self.mart_delta, self.recal_size, self.cal_frac, self.seed = alpha, monitor, mart_delta, recal_size, cal_frac, seed
        assert recal_mode in ("naive", "bar"), recal_mode
        self.recal_mode, self.guard_eps_max, self.eps_hint = recal_mode, guard_eps_max, eps_hint
        self.tau: Optional[float] = None
        self.mart: Optional[ConformalMartingale] = None
        self.n_recal = 0
        self.alarms: List[int] = []
        self._cal_scores = np.zeros(0)
        self._step = 0

    # score: higher = more anomalous
    def _raw(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(X)

    def warm_up(self, normal_events):
        feats = list(session_features(normal_events).values())
        if len(feats) < 40:
            raise ValueError("need >= 40 benign warm-up sessions")
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(feats))
        n_cal = int(len(feats) * self.cal_frac)
        cal = np.vstack([feats[i] for i in idx[:n_cal]]); fit = np.vstack([feats[i] for i in idx[n_cal:]])
        self.model = IsolationForest(n_estimators=150, random_state=self.seed).fit(fit)
        self._cal_scores = self._raw(cal)
        self._ref = self._cal_scores.copy()                 # current reference sample (transported when recal_mode='bar')
        self.tau = conformal_threshold(self._cal_scores, self.alpha)
        if not np.isfinite(self.tau):
            self.tau = empirical_threshold(self._cal_scores, self.alpha)
        if self.use_monitor:
            self.mart = ConformalMartingale(self._cal_scores, delta=self.mart_delta, seed=self.seed)

    def score_session(self, events) -> Dict[str, dict]:
        feats = session_features(events)
        sids = list(feats)
        X = np.vstack([feats[s] for s in sids])
        sc = self._raw(X)
        out = {s: {"anomaly_score": round(float(v), 4), "is_anomaly": bool(v > self.tau), "features": dict(zip(FEATURE_NAMES, feats[s].tolist()))}
               for s, v in zip(sids, sc)}
        if self.mart is not None:                       # decisions for NEXT batch use only these unlabelled scores
            fired = False
            for v in sc:
                if self.mart.update(float(v)):
                    fired = True
            if fired:
                self._step += 1
                self.alarms.append(self._step)
                if len(sc) >= 20:
                    if self.recal_mode == "bar":              # transport the reference sample; the window's attacks never set the threshold
                        self._ref, _ = bar_recalibrate(self._ref, sc, self.alpha, guard_eps_max=self.guard_eps_max, mode="auto", eps_hint=self.eps_hint)
                    else:
                        self._ref = sc[-self.recal_size:].copy()
                    self.tau = conformal_threshold(self._ref, self.alpha)
                    if not np.isfinite(self.tau):
                        self.tau = empirical_threshold(self._ref, self.alpha)
                    self.mart.reset(self._ref)
                    self.n_recal += 1
            else:
                self._step += 1
        return out
