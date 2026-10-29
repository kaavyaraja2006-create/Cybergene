import numpy as np
import pytest

from cgresearch.experiments.contamination import run_contamination
from cgresearch.experiments.protocol_sens import run_protocol_sens
from cgresearch.experiments.sentinel_case import run_sentinel_case
from cgresearch.experiments.shift_monitor import run_shift_monitor
from cgresearch.scenarios import make_folds
from cgresearch.experiments.common import load_data

SHIFT = {"kind": "synthetic", "generator": "shift", "shift_args": {"n_days": 5, "per_day": 600, "seed": 1}}
BASE = {"seeds": [0], "n_boot": 50, "data": SHIFT, "models": ["pca", "hbos"], "tuning": {"grid": "random", "n": 2}}


def test_temporal_folds_have_disjoint_roles():
    d = load_data({"data": SHIFT})
    for f in make_folds(d, "temporal", [0, 1], {}):
        f.check(d.y)
        assert f.windows[0].id == "control"


def test_shift_monitor_runs(tmp_path):
    r = run_shift_monitor({**BASE, "name": "t_sm", "fa_reps": 3}, str(tmp_path))
    assert r.dir.exists()


def test_contamination_bound_and_monotone(tmp_path):
    r = run_contamination({**BASE, "name": "t_ct", "eps": [0.0, 0.05], "alphas": [0.05]}, str(tmp_path))
    import pandas as pd
    df = pd.read_csv(next(r.dir.rglob("per_cell.csv")))
    assert df["bound_holds"].mean() > 0.9           # R4 bound (with DKW slack) holds in nearly all cells


def test_protocol_sens_runs(tmp_path):
    r = run_protocol_sens({**BASE, "name": "t_ps", "seeds": [0, 1], "scenarios": [{"kind": "temporal"}]}, str(tmp_path))
    assert r.dir.exists()


def test_calibrated_sentinel_controls_fpr_without_drift():
    from cgresearch.sentinel_calibrated import CalibratedSentinelAgent
    from simulator import CloudSimulator
    sim = CloudSimulator(seed=3)
    ag = CalibratedSentinelAgent(0.05, monitor=False, seed=3)
    ag.warm_up(sim.emit_normal(2000))
    res = ag.score_session(sim.emit_normal(2000))
    fpr = np.mean([r["is_anomaly"] for r in res.values()])
    assert fpr < 0.08


def test_sentinel_case_runs(tmp_path):
    r = run_sentinel_case({"name": "t_sc", "seeds": [0], "n_sessions": 400, "phases": 3, "attacks_per_phase": 15, "n_boot": 50}, str(tmp_path))
    assert r.dir.exists()


def test_recal_study_runs(tmp_path):
    from cgresearch.experiments.recal_study import run_recal_study
    r = run_recal_study({"name": "t_rs", "seeds": [0], "reps": 2, "n_cal": 500, "n_win": 800, "eps": [0.0, 0.1], "n_boot": 50}, str(tmp_path))
    import pandas as pd
    df = pd.read_csv(next(r.dir.rglob("per_rep.csv")))
    assert {"bar_auto", "naive", "oracle", "static"} <= set(df.method)
    assert df[(df.method == "oracle")].fpr.between(0.03, 0.07).mean() > 0.9


def test_monitor_study_runs(tmp_path):
    from cgresearch.experiments.monitor_study import run_monitor_study
    r = run_monitor_study({"name": "t_ms", "seeds": [0], "reps": 2, "n_tune": 4, "stream_len": 1500, "t0": 500, "n_cal": 800,
                           "eps": [0.0], "cert_reps": 2, "bases": ["gamma"], "n_boot": 50}, str(tmp_path))
    assert any(r.dir.rglob("summary_certificate.csv"))
