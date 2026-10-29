"""
Agent 1 -- Sentinel Agent.

Responsible for telemetry monitoring, baseline creation and anomaly
scoring. Uses an Isolation Forest over a small handcrafted feature set
computed on a sliding session window, which is enough to separate the
synthetic normal traffic from the synthetic attack bursts and is cheap
to retrain online as more data arrives.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np
from sklearn.ensemble import IsolationForest

from simulator import Event


FEATURE_NAMES = [
    "requests_per_window", "failed_logins", "unique_resources",
    "unique_actions", "avg_cpu", "avg_egress",
]


def session_features(events: List[Event]) -> Dict[str, np.ndarray]:
    """Group events by session_id and compute a feature vector per session."""
    sessions: Dict[str, List[Event]] = defaultdict(list)
    for e in events:
        sessions[e.session_id].append(e)

    out = {}
    for sid, evs in sessions.items():
        failed = sum(1 for e in evs if e.status == "failed")
        resources = len({e.resource for e in evs})
        actions = len({e.action for e in evs})
        avg_cpu = float(np.mean([e.cpu_pct for e in evs]))
        avg_egress = float(np.mean([e.network_egress_kbps for e in evs]))
        out[sid] = np.array([len(evs), failed, resources, actions, avg_cpu, avg_egress])
    return out


class SentinelAgent:
    """Baseline anomaly detector. Retrains on demand from a pool of
    normal-labeled sessions so the frontier adapts as the environment's
    "normal" shifts."""

    def __init__(self, contamination: float = 0.15):
        self.contamination = contamination
        self.model: IsolationForest | None = None
        self._fit_pool: List[np.ndarray] = []

    def warm_up(self, normal_events: List[Event]):
        feats = session_features(normal_events)
        self._fit_pool.extend(feats.values())
        if len(self._fit_pool) >= 8:
            self._refit()

    def _refit(self):
        X = np.vstack(self._fit_pool[-500:])
        self.model = IsolationForest(
            n_estimators=150, contamination=self.contamination, random_state=42
        )
        self.model.fit(X)

    def score_session(self, events: List[Event]) -> Dict[str, dict]:
        """Return {session_id: {anomaly_score, is_anomaly, features}} for
        every session present in `events`."""
        feats = session_features(events)
        results = {}
        for sid, vec in feats.items():
            if self.model is None:
                # No baseline yet -- fall back to a simple heuristic so the
                # system is still useful cold.
                score = min(1.0, (vec[1] * 0.15) + (vec[3] / 10.0) + (vec[5] / 5000.0))
                is_anom = score > 0.6
            else:
                # IsolationForest: lower decision_function = more anomalous.
                raw = self.model.decision_function(vec.reshape(1, -1))[0]
                score = float(np.clip(0.5 - raw, 0.0, 1.0))
                is_anom = self.model.predict(vec.reshape(1, -1))[0] == -1
            results[sid] = {
                "anomaly_score": round(float(score), 3),
                "is_anomaly": bool(is_anom),
                "features": dict(zip(FEATURE_NAMES, vec.tolist())),
            }
        return results
