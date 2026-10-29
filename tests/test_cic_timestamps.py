import numpy as np, pandas as pd
from cgresearch.data.cic import load_cic


def test_mixed_timestamp_formats_are_parsed(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    ts = np.where(np.arange(n) % 2 == 0, "3/7/2017 8:55", "3/7/2017 08:55:12")     # two formats in one file
    d = pd.DataFrame({"Destination Port": rng.integers(1, 1000, n), "Protocol": rng.choice([6, 17], n), "Timestamp": ts,
                      "Flow Duration": rng.integers(1, 10**6, n), "Total Fwd Packets": rng.integers(1, 50, n),
                      "Label": np.where(np.arange(n) % 5 == 0, "DoS", "BENIGN")})
    d.to_csv(tmp_path / "Monday-WorkingHours.pcap_ISCX.csv", index=False)
    fd = load_cic(tmp_path, benign_label="BENIGN")
    assert fd.meta["stime"].isna().mean() < 0.01
