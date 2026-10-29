"""Case study: the calibrated Sentinel vs the original Sentinel on the prototype's simulated cloud traffic with injected benign drift.

Drift = multiplicative growth of benign CPU / egress per phase (seasonal load growth).  Compared agents:
  original      : SentinelAgent (IsolationForest, contamination=0.15 predict rule)
  cal_static    : conformal threshold, no monitor
  cal_monitored : conformal threshold + conformal-martingale drift alarm + naive unlabelled recalibration (raw window)
  cal_monitored_bar : same alarm, bulk-anchored recalibration (attacks in the window cannot raise the threshold)
The simulator is synthetic and the attack templates are separable by construction, so this is a SYSTEMS demonstration of the mechanism
(FPR control and recalibration under drift), NOT evidence about real-world detection accuracy.  Report it as such.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict

import numpy as np

from .._backend import BACKEND  # noqa: F401
from simulator import CloudSimulator  # noqa: E402
from sentinel_agent import SentinelAgent  # noqa: E402
from ..runlog import ExperimentRun
from ..sentinel_calibrated import CalibratedSentinelAgent
from ..stats_cluster import cluster_bootstrap_ci


def _drift(events, factor):
    for e in events:
        e.cpu_pct = round(min(e.cpu_pct * factor, 100.0), 1)
        e.network_egress_kbps = round(e.network_egress_kbps * factor, 1)
    return events


def run_sentinel_case(cfg: Dict[str, Any], out_root="results", argv=None) -> ExperimentRun:
    run = ExperimentRun(cfg.get("name", "sentinel_case"), cfg, out_root, argv, {"synthetic": True, "source": "backend/simulator.py"}, cfg["seeds"])
    n = int(cfg.get("n_sessions", 2000)); phases = int(cfg.get("phases", 6)); growth = float(cfg.get("drift_per_phase", 0.15))
    n_att = int(cfg.get("attacks_per_phase", 60)); alpha = cfg.get("alpha", 0.05)
    rows = []
    for seed in cfg["seeds"]:
        sim = CloudSimulator(seed=seed)
        warm = sim.emit_normal(n)
        agents = {"original": SentinelAgent(), "cal_static": CalibratedSentinelAgent(alpha, monitor=False, seed=seed),
                  "cal_monitored": CalibratedSentinelAgent(alpha, monitor=True, seed=seed),
                  "cal_monitored_bar": CalibratedSentinelAgent(alpha, monitor=True, seed=seed, recal_mode="bar"),
                  "cal_monitored_bar_hint": CalibratedSentinelAgent(alpha, monitor=True, seed=seed, recal_mode="bar", eps_hint=cfg.get("eps_hint", 0.05))}
        for a in agents.values():
            a.warm_up(warm)
        for ph in range(phases):
            f = (1 + growth) ** ph
            ben = _drift(sim.emit_normal(max(n // 2, 100)), f)
            att = [e for _ in range(n_att) for e in _drift(sim.emit_attack(), f)]
            batch = ben + att
            sim.rng.shuffle(batch)
            truth = {e.session_id: e.label for e in batch}
            for name, ag in agents.items():
                res = ag.score_session(batch)
                nb = sum(1 for s, l in truth.items() if l == "normal"); na = len(truth) - nb
                fp = sum(1 for s, r in res.items() if truth[s] == "normal" and r["is_anomaly"])
                tp = sum(1 for s, r in res.items() if truth[s] == "attack" and r["is_anomaly"])
                rows.append({"seed": seed, "phase": ph, "drift_factor": f, "agent": name, "fpr": fp / max(nb, 1), "recall": tp / max(na, 1),
                             "recalibrations_so_far": getattr(ag, "n_recal", 0), "tau": getattr(ag, "tau", float("nan"))})
        print(f"[{run.experiment}] seed {seed} done", flush=True)
    run.add_rows("per_phase", rows)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["agent"]][r["phase"]].append(r)
    summ = []
    for ag, cl in by.items():
        for metric in ("fpr", "recall"):
            summ.append({"agent": ag, "metric": metric, **cluster_bootstrap_ci({c: [x[metric] for x in v] for c, v in cl.items()}, cfg.get("n_boot", 10_000))})
        summ.append({"agent": ag, "metric": "violation_rate(fpr>alpha)", **cluster_bootstrap_ci({c: [float(x["fpr"] > alpha) for x in v] for c, v in cl.items()}, cfg.get("n_boot", 10_000))})
    run.save_table("summary", summ)
    run.finish()
    return run
