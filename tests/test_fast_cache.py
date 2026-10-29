import numpy as np

from cgresearch.experiments.common import load_data
from cgresearch.foldscore import score_fold
from cgresearch.scenarios import make_folds

CFG = {"data": {"kind": "synthetic", "synthetic_args": {"n": 4000, "seed": 2}}}


def test_cache_is_exact_across_lofo_families():
    data = load_data(CFG)
    folds = make_folds(data, "lofo", [0], {"min_support": 30, "max_families": 3})
    tun = {"grid": "random", "n": 3}
    cache = {}
    for f in folds:
        a = score_fold(data, f, ["pca", "hbos", "isolation_forest"], tun, cache=cache)
        b = score_fold(data, f, ["pca", "hbos", "isolation_forest"], tun, cache=None)
        for m in a:
            assert a[m]["params"] == b[m]["params"]
            assert np.allclose(a[m]["cal"], b[m]["cal"])
            for w in a[m]["windows"]:
                assert np.allclose(a[m]["windows"][w][0], b[m]["windows"][w][0])


def test_cache_does_not_leak_between_datasets_or_seeds():
    d1 = load_data(CFG); d2 = load_data({"data": {"kind": "synthetic", "synthetic_args": {"n": 4000, "seed": 3}}})
    cache = {}
    f1 = make_folds(d1, "random", [0], {})[0]; f2 = make_folds(d2, "random", [0], {})[0]
    a = score_fold(d1, f1, ["pca"], {"grid": "random", "n": 2}, cache=cache)
    b = score_fold(d2, f2, ["pca"], {"grid": "random", "n": 2}, cache=cache)
    c = score_fold(d2, f2, ["pca"], {"grid": "random", "n": 2}, cache=None)
    assert np.allclose(b["pca"]["cal"], c["pca"]["cal"]) and not np.allclose(a["pca"]["cal"][:5], b["pca"]["cal"][:5])


def test_prepare_disk_cache_roundtrip(tmp_path, monkeypatch):
    import pandas as pd
    from cgresearch.experiments.common import prepare
    from cgresearch.data.synthetic import make_synthetic_flows
    rng = np.random.default_rng(0)
    d = make_synthetic_flows(n=1500, seed=1)
    # write a CIC-like CSV directory so the cached (real-data) path is exercised
    df = pd.DataFrame({" Flow Duration": rng.gamma(2, 1e4, 1500), " Total Fwd Packets": rng.integers(1, 30, 1500), " Total Backward Packets": rng.integers(1, 30, 1500),
                       " Protocol": rng.choice([6, 17], 1500), " Label": np.where(rng.random(1500) < 0.2, "DoS Hulk", "BENIGN")})
    (tmp_path / "data").mkdir(); df.to_csv(tmp_path / "data" / "Monday-WorkingHours.csv", index=False)
    monkeypatch.chdir(tmp_path)
    cfg = {"data": {"kind": "cic2017", "path": str(tmp_path / "data"), "version": "t"}, "dedup": "exact"}
    a = prepare(cfg); b = prepare(cfg)
    assert (tmp_path / ".cache").exists() and len(list((tmp_path / ".cache").glob("prep_*.pkl"))) == 1
    assert np.array_equal(a[0].y, b[0].y) and a[2] == b[2]
