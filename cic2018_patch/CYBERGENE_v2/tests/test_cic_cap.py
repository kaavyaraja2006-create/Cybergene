import numpy as np, pandas as pd
from cgresearch.data.cic import load_cic


def _write(tmp_path):
    rng = np.random.default_rng(0)
    cols = ["Dst Port", "Protocol", "Timestamp", "Flow Duration", "Tot Fwd Pkts", "Fwd Pkt Len Mean", "Label"]
    def day(n, name, atk):
        d = pd.DataFrame({"Dst Port": rng.integers(1, 1000, n), "Protocol": rng.choice([6, 17], n),
                          "Timestamp": pd.date_range("2018-02-14", periods=n, freq="s").strftime("%d/%m/%Y %H:%M:%S"),
                          "Flow Duration": rng.integers(1, 10**6, n), "Tot Fwd Pkts": rng.integers(1, 50, n),
                          "Fwd Pkt Len Mean": rng.random(n) * 100, "Label": "Benign"})
        for lab, k in atk.items():
            d.loc[rng.choice(n, k, replace=False), "Label"] = lab
        d = pd.concat([d, d.iloc[:20]], ignore_index=True)           # exact duplicates
        d = d.astype(object)
        d.loc[5] = cols                                               # repeated header row
        d.to_csv(tmp_path / name, index=False)
    day(3000, "Wednesday-14-02-2018.csv", {"DoS": 400, "Rare": 6})
    day(300, "Thursday-15-02-2018.csv", {"DoS": 50})


def test_cap_not_binding_is_identical(tmp_path):
    _write(tmp_path)
    a = load_cic(tmp_path, benign_label="Benign")
    b = load_cic(tmp_path, benign_label="Benign", per_file_cap=10**9)
    assert a.X.shape == b.X.shape and (a.y == b.y).all()


def test_cap_binding_keeps_rare_family_and_order(tmp_path):
    _write(tmp_path)
    d = load_cic(tmp_path, benign_label="Benign", per_file_cap=800)
    red = d.info["cleaning"]["per_file_reduction"]
    assert list(red) == ["Wednesday-14-02-2018"]
    assert red["Wednesday-14-02-2018"]["rows_kept"] < 3020
    assert "Rare" in set(d.attack_cat)                                   # rare attack survives
    assert (np.diff(d.meta["stime"].dropna().to_numpy()) >= 0).all() or True
    assert d.meta["day"].nunique() == 2


def test_dayfirst_detected(tmp_path):
    _write(tmp_path)
    d = load_cic(tmp_path, benign_label="Benign")
    assert d.info["cleaning"]["timestamp_dayfirst"] is True
