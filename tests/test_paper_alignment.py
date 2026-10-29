"""Tests for the features added to match the draft paper: UNSW raw-file loading with sampling,
legacy-vs-calibrated protocol, standardisation ablation, weak-label retrieval, report export."""
import json

import numpy as np
import pandas as pd

from cgresearch.data.unsw import RAW_COLUMNS, load_unsw
from cgresearch.experiments.detection import run_detection
from cgresearch.experiments.if_diag import run_if_diagnostics, verdict
from cgresearch.experiments.retrieval_real import assign_genomes, run_retrieval_real
from cgresearch.experiments.variants import run_variants
from cgresearch.metrics import implied_fpr
from cgresearch.preprocess import FlowPreprocessor
from cgresearch.data.synthetic import make_synthetic_flows

SYN = {"kind": "synthetic", "synthetic_args": {"n": 2000, "seed": 1, "attack_frac": 0.4}}


def _raw_unsw(path, n=800, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(int)
    rows = []
    for i in range(n):
        r = {c: 0 for c in RAW_COLUMNS}
        r.update(srcip=f"59.166.0.{i % 9}", sport=int(rng.integers(1024, 65000)), dstip="149.171.126.1", dsport=str(rng.choice([80, "0x000b", 53])),
                 proto="tcp", state="FIN", service="-", dur=float(rng.random()), sbytes=int(rng.integers(1, 9999)), dbytes=int(rng.integers(1, 9999)),
                 spkts=int(rng.integers(1, 20)), dpkts=int(rng.integers(1, 20)), stime=1421927414 + i, ltime=1421927415 + i,
                 attack_cat="Exploits" if y[i] else " ", label=int(y[i]))
        rows.append(r)
    pd.DataFrame(rows)[RAW_COLUMNS].to_csv(path, header=False, index=False)
    return y


def test_unsw_headerless_raw_file_with_hex_ports_and_stratified_sample(tmp_path):
    f = tmp_path / "UNSW-NB15_1.csv"; y = _raw_unsw(f)
    d = load_unsw(f, sample_n=400, sample_seed=3)
    assert len(d) == 400 and abs(d.y.mean() - y.mean()) < 0.02            # stratified: prevalence preserved
    assert d.meta["dsport"].notna().all()                                  # '0x000b' parsed
    d2 = load_unsw(f, sample_n=400, sample_seed=3)
    assert (d.y == d2.y).all() and d.X.equals(d2.X)                         # sample is seeded/reproducible
    assert {"sport", "dsport"} <= set(d.X.columns) and "label" not in d.X.columns and "srcip" not in d.X.columns
    assert d.info["sha256"] and d.info["sample"]["n"] == 400


def test_port_modes_and_standardize_toggle():
    d = make_synthetic_flows(600, seed=2); idx = np.arange(400)
    for m in ("raw", "separate_std", "drop", "onehot", "onehot_scaled"):
        p = FlowPreprocessor(m).fit(d, idx); X = p.transform(d)
        assert X.shape[1] == len(p.feature_sources) and np.isfinite(X).all()
    assert FlowPreprocessor("drop").fit(d, idx).transform(d).shape[1] < FlowPreprocessor("raw").fit(d, idx).transform(d).shape[1]
    a = FlowPreprocessor("drop", standardize=True).fit(d, idx).transform(d)
    b = FlowPreprocessor("drop", standardize=False).fit(d, idx).transform(d)
    assert abs(a[idx, :22].mean()) < 0.2 and not np.allclose(a, b)


def test_implied_fpr_matches_measured_when_pr_are_from_same_counts():
    tp, fp, fn, tn = 30, 20, 10, 940
    p, r, pi = tp / (tp + fp), tp / (tp + fn), (tp + fn) / (tp + fp + fn + tn)
    assert abs(implied_fpr(p, r, pi) - fp / (fp + tn)) < 1e-12


def test_legacy_protocol_trivial_baselines_and_paired_table(tmp_path):
    cfg = {"name": "t", "data": SYN, "dedup": "exact", "seeds": [0, 1, 2], "models": ["isolation_forest", "lof", "logreg"], "tune": False,
           "n_boot": 100, "legacy_contamination": [0.15], "target_fprs": [0.05]}
    r = run_detection(cfg, str(tmp_path))
    leg = r.tables["per_seed_legacy_contamination"]
    assert {x["model"] for x in leg} == {"isolation_forest", "lof"}          # supervised models have no contamination protocol
    cal = [x for x in r.tables["per_seed_fpr_sweep"] if x["model"] == "lof"]
    # finite-sample noise: ~240 benign val/test rows give SE ~2 pp per seed, so check the seed mean
    assert abs(np.mean([x["measured_fpr"] for x in cal]) - 0.05) < 0.04
    # legacy threshold comes from TRAIN scores, calibrated from VAL: they differ
    l0 = next(x for x in leg if x["model"] == "lof" and x["seed"] == 0); c0 = next(x for x in cal if x["seed"] == 0)
    assert l0["threshold"] != c0["threshold"]
    flag = [x for x in r.tables["trivial_baselines"] if x["model"] == "flag_all" and x["seed"] == 0][0]
    assert flag["recall"] == 1.0 and abs(flag["precision"] - flag["test_prevalence"]) < 1e-12
    assert r.tables["legacy_vs_calibrated_paired"]


def test_variants_standardization_ablation_is_paired(tmp_path):
    cfg = {"name": "v", "data": SYN, "dedup": "exact", "seeds": [0, 1, 2], "models": ["isolation_forest", "ocsvm"], "tune": False, "n_boot": 100,
           "target_fprs": [0.05], "variants": [{"name": "standardized", "standardize": True}, {"name": "unstandardized", "standardize": False}]}
    r = run_variants(cfg, str(tmp_path))
    assert r.tables["variant_comparisons_vs_first"] and all(c["n_pairs"] == 3 for c in r.tables["variant_comparisons_vs_first"])
    subs = list(r.manifest["sub_runs"].values())
    s0 = [json.load(open(f"{d}/artifacts/../manifest.json")) for d in subs]
    assert len({m["seeds"].__str__() for m in s0}) == 1


def test_isolation_forest_scale_invariance_documented():
    """IF is invariant to per-feature affine scaling: onehot vs onehot_scaled must agree (else bug)."""
    d = make_synthetic_flows(1500, seed=5)
    from cgresearch.data.splits import make_split
    from cgresearch.detectors import REGISTRY
    sp = make_split(d.y, 0)
    s = []
    for m in ("onehot", "onehot_scaled"):
        p = FlowPreprocessor(m).fit(d, sp["benign_train"]); X = p.transform(d)
        s.append(REGISTRY["isolation_forest"](seed=0).fit(X[sp["benign_train"]]).score(X[sp["test"]]))
    assert abs(np.corrcoef(s[0], s[1])[0, 1]) > 0.99


def test_if_verdict_flags_orientation_and_code_defects(tmp_path):
    r = run_if_diagnostics({"name": "d", "data": SYN, "dedup": "exact", "seeds": [0, 1]}, str(tmp_path))
    rows = [dict(x) for x in r.tables["if_diagnostics"]]
    assert len({x["port_mode"] for x in rows}) == 5 and r.tables["top_scoring_benign_raw"].__len__() == 50
    bad = [dict(x, orientation_pair_sums_to_1=False) for x in rows]
    assert "code_defect" in verdict(bad)["classification"]
    inv = [dict(x, roc_auc=0.3, auc_score_samples=0.7) for x in rows]
    assert "score_orientation" in verdict(inv)["classification"]


def test_genome_assignment_is_explicit_and_reports_unmatched():
    g, un = assign_genomes(["proto", "dsport", "spkts", "ct_srv_src", "dur", "sbytes", "service", "weird_col"])
    assert set(g) == {"network", "behavior", "resource", "context"} and un == ["weird_col"]
    assert len(g["context"]) == 1


def test_retrieval_real_uses_weak_labels_and_disjoint_queries(tmp_path):
    cfg = {"name": "rr", "data": SYN, "dedup": "exact", "seeds": [0, 1], "n_store": 300, "n_queries": 40, "min_category_support": 20, "n_boot": 100}
    r = run_retrieval_real(cfg, str(tmp_path))
    assert r.manifest["categories"]["label_type"].startswith("WEAK") and r.manifest["genome_columns"]
    cfgs = {x["config"] for x in r.tables["per_seed"]}
    assert "network" in cfgs and not any("evolution" in c for c in cfgs)     # no evolution genome on public data
    for x in r.tables["per_seed"]:
        assert 0 <= x["P@5"] <= 1 and x["n_queries"] <= 40


def test_report_export_labels_traceability_and_smoke_filter(tmp_path):
    from cgresearch.report import build
    run_detection({"name": "SMOKE_rep", "data": SYN, "dedup": "exact", "seeds": [0, 1], "models": ["isolation_forest", "logreg"], "tune": False,
                   "n_boot": 100}, str(tmp_path / "res"))
    assert build(tmp_path / "res", tmp_path / "paper") == []                  # smoke/synthetic excluded by default
    idx = build(tmp_path / "res", tmp_path / "paper2", include_smoke=True)
    assert idx and all(i["experiment_id"] and i["config_sha256"] for i in idx)
    tex = (tmp_path / "paper2" / "tables" / "smoke_rep_detection.tex").read_text()
    assert "\\label{tab:smoke_rep_detection}" in tex and "NOT A RESULT" in tex and "% experiment_id:" in tex
    assert (tmp_path / "paper2" / "figures" / "smoke_rep_roc_pr.pdf").exists()


def test_nested_run_paths_stay_short_for_windows(tmp_path):
    """Windows MAX_PATH is 260 chars: sub-runs must not repeat the long experiment name."""
    from cgresearch.experiments.dedup import run_dedup_comparison
    cfg = {"name": "unsw_sampleA_dedup_comparison", "data": SYN, "seeds": [0, 1], "models": ["isolation_forest"], "tune": False,
           "n_boot": 50, "modes": ["none", "exact"], "legacy_contamination": [0.15]}
    r = run_dedup_comparison(cfg, str(tmp_path))
    longest = max(len(str(p.relative_to(tmp_path))) for p in r.dir.rglob("*"))
    assert longest < 170, longest
    assert (r.dir / "sub_runs" / "none" / "manifest.json").exists()


def test_parallel_seeds_give_identical_results(tmp_path):
    """Runs in a subprocess: multiprocessing 'spawn' re-imports __main__, which hangs under pytest."""
    import subprocess, sys, textwrap
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / "par.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(root)!r})
        from cgresearch.experiments.detection import run_detection
        if __name__ == "__main__":
            SYN = {{"kind": "synthetic", "synthetic_args": {{"n": 1500, "seed": 1, "attack_frac": 0.4}}}}
            base = {{"data": SYN, "dedup": "exact", "seeds": [0, 1, 2], "models": ["isolation_forest", "logreg"], "tune": False, "n_boot": 50}}
            a = run_detection({{**base, "name": "p1"}}, {str(tmp_path / "o")!r})
            b = run_detection({{**base, "name": "p2", "jobs": 2}}, {str(tmp_path / "o")!r})
            k = lambda r: [(x["seed"], x["model"], round(x["roc_auc"], 12), x["split_hash"]) for x in r.tables["per_seed_threshold_free"]]
            print("IDENTICAL" if k(a) == k(b) else "DIFFERENT")
    """))
    out = subprocess.run([sys.executable, "-W", "ignore", str(script)], capture_output=True, text=True, timeout=240)
    assert "IDENTICAL" in out.stdout, out.stdout + out.stderr
