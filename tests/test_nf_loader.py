import numpy as np
import pandas as pd

from cgresearch.data.nf import load_nf


def test_nf_loader_on_fixture(tmp_path):
    rng = np.random.default_rng(0); n = 400
    df = pd.DataFrame({"IPV4_SRC_ADDR": "10.0.0.1", "L4_SRC_PORT": rng.integers(1024, 65000, n), "IPV4_DST_ADDR": "10.0.0.2",
                       "L4_DST_PORT": rng.choice([80, 443, 22], n), "PROTOCOL": rng.choice([6, 17], n), "IN_BYTES": rng.integers(1, 9999, n),
                       "IN_PKTS": rng.integers(1, 30, n), "OUT_BYTES": rng.integers(1, 9999, n), "OUT_PKTS": rng.integers(1, 30, n),
                       "FLOW_DURATION_MILLISECONDS": rng.integers(1, 5000, n), "TCP_FLAGS": rng.integers(0, 30, n),
                       "FLOW_START_MILLISECONDS": 1_500_000_000_000 + np.arange(n) * 1000, "CONSTANT": 1,
                       "Label": (rng.random(n) < 0.3).astype(int)})
    df["Attack"] = np.where(df["Label"] == 1, "DoS", "Benign")
    f = tmp_path / "NF-test-v2.csv"; df.to_csv(f, index=False)
    d = load_nf(f)
    assert len(d) == n and d.y.sum() == df["Label"].sum() and set(d.attack_cat) == {"Benign", "DoS"}
    assert {"dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "proto", "sport", "dsport"} <= set(d.X.columns)
    assert "constant" not in d.X.columns and "label" not in d.X.columns and "attack" not in d.X.columns
    assert d.meta["stime"].is_monotonic_increasing and np.isfinite(d.X.select_dtypes("number").to_numpy()).all()
    d2 = load_nf(f, sample_n=200, sample_seed=1)
    assert len(d2) in (199, 200, 201) and abs(d2.y.mean() - d.y.mean()) < 0.03
