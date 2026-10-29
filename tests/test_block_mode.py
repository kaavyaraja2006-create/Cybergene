import numpy as np
from cgresearch.scenarios import make_folds
from cgresearch.data.synthetic import make_shift_flows


def test_block_mode_stime_gives_finer_blocks():
    d = make_shift_flows(n_days=5, per_day=1500, seed=1)
    coarse = make_folds(d, "temporal", [0], {"n_blocks": 15})
    fine = make_folds(d, "temporal", [0], {"n_blocks": 15, "block_mode": "stime", "fit_blocks": [0, 1, 2], "label_blocks": [3]})
    assert len(fine[0].windows) - 1 == 11               # 15 blocks - 3 fit - 1 label (+control excluded)
    assert len(coarse[0].windows) < len(fine[0].windows)
