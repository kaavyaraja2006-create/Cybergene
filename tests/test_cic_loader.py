import numpy as np
import pandas as pd

from cgresearch.data.cic import load_cic, norm
from cgresearch.data.audit import deduplicate
from cgresearch.detectors import REGISTRY
from cgresearch.experiments.detection import run_detection


def _write_cic(path, n=600, seed=0, style="2017"):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.3
    cols17 = {" Destination Port": rng.choice([80, 443, 22, 8080], n), " Flow Duration": rng.integers(1, 5_000_000, n),
              " Total Fwd Packets": rng.integers(1, 50, n), " Total Backward Packets": rng.integers(0, 40, n),
              "Total Length of Fwd Packets": rng.integers(0, 9000, n), " Total Length of Bwd Packets": rng.integers(0, 20000, n),
              " Flow Packets/s": rng.random(n) * 1e3, " Protocol": rng.choice([6, 17], n), " Constant": 1}
    df = pd.DataFrame(cols17)
    df.loc[y, " Total Fwd Packets"] += 80
    df.loc[0, " Flow Packets/s"] = np.inf
    df[" Label"] = np.where(y, "DoS Hulk", "BENIGN")
    if style == "2018":
        df = df.rename(columns={" Destination Port": "Dst Port", " Total Fwd Packets": "Tot Fwd Pkts"})
        df.loc[len(df)] = [("Label" if c.strip() == "Label" else c) for c in df.columns]   # repeated header row
    df.to_csv(path, index=False)
    return y


def test_norm_and_alignment_2017_and_2018(tmp_path):
    for style in ("2017", "2018"):
        f = tmp_path / f"{style}.csv"; _write_cic(f, style=style)
        d = load_cic(f)
        assert {"dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "dsport", "proto"} <= set(d.X.columns), style
        assert d.info["alignment"]["dsport"] in ("destination_port", "dst_port")
        assert d.X["dur"].max() <= 5.0 + 1e-9                    # microseconds -> seconds
        assert np.isfinite(d.X.select_dtypes("number").to_numpy()).all()
        assert d.info["cleaning"]["inf_or_nan_cells"] >= 1 and "constant" in d.info["cleaning"]["constant_columns_dropped"]
        assert set(d.X["proto"].unique()) <= {"tcp", "udp", "icmp", "other", "hopopt"}
        if style == "2018":
            assert d.info["cleaning"]["repeated_header_rows_dropped"] == 1


def test_labels_binary_and_categories_kept(tmp_path):
    f = tmp_path / "c.csv"; y = _write_cic(f)
    d = load_cic(f)
    assert d.y.sum() == y.sum() and set(d.attack_cat) == {"BENIGN", "DoS Hulk"}


def test_full_pipeline_runs_on_cic_format(tmp_path):
    f = tmp_path / "c.csv"; _write_cic(f, n=900)
    cfg = {"name": "cic_t", "data": {"kind": "cic2017", "path": str(f), "version": "fixture"}, "dedup": "exact",
           "port_mode": "onehot", "seeds": [0, 1], "models": ["isolation_forest", "rule_based", "logreg"], "tune": False, "n_boot": 100}
    r = run_detection(cfg, str(tmp_path / "out"))
    assert r.manifest["dataset"]["sha256"] and r.manifest["dataset"]["synthetic"] is False
    assert r.manifest["dataset"]["alignment"]["dur"] == "flow_duration"
    rb = [x for x in r.tables["per_seed_threshold_free"] if x["model"] == "rule_based"]
    assert len(rb) == 2


def test_non_ascii_labels_cleaned_and_written_as_utf8(tmp_path):
    """Regression: 'Web Attack \\x96 Brute Force' crashed CSV writing under Windows cp1252."""
    from cgresearch.runlog import write_csv
    f = tmp_path / "c.csv"; _write_cic(f)
    raw = f.read_bytes().replace(b"DoS Hulk", b"Web Attack \x96 Brute Force")   # cp1252 en-dash byte
    f.write_bytes(raw)
    d = load_cic(f)
    assert "Web Attack - Brute Force" in set(d.attack_cat) and all(c.isascii() for c in d.attack_cat)
    write_csv(tmp_path / "t.csv", [{"label": "Web Attack \u2013 Brute Force", "x": "caf\u00e9"}])
    assert "caf\u00e9" in (tmp_path / "t.csv").read_text(encoding="utf-8")


def test_blank_label_rows_are_dropped_not_counted_as_attacks(tmp_path):
    f = tmp_path / "blank.csv"; y = _write_cic(f)
    raw = pd.read_csv(f)
    blanks = pd.DataFrame([[np.nan] * raw.shape[1]] * 40, columns=raw.columns)   # blank lines as in GeneratedLabelledFlows
    pd.concat([raw, blanks], ignore_index=True).to_csv(f, index=False)
    d = load_cic(f)
    assert d.info["cleaning"]["blank_label_rows_dropped"] == 40
    assert len(d.y) == len(y) and d.y.sum() == y.sum()
    assert "nan" not in set(d.attack_cat)
    d2 = load_cic(f, sample_n=300, sample_seed=0)
    assert "nan" not in set(d2.attack_cat)
