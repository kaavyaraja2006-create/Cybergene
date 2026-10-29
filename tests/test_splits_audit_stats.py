import numpy as np

from cgresearch.data.audit import deduplicate, exact_duplicate_groups, leakage_report, near_duplicate_pairs
from cgresearch.data.splits import assert_disjoint, make_split, split_hash
from cgresearch.data.synthetic import make_synthetic_flows
from cgresearch.stats import bootstrap_ci, cliffs_delta, holm_bonferroni, paired_comparisons, wilcoxon_paired


def test_split_is_deterministic_disjoint_and_model_independent():
    y = (np.random.default_rng(0).random(2000) < 0.2).astype(int)
    a, b = make_split(y, 5), make_split(y, 5)
    assert split_hash(a) == split_hash(b)
    assert split_hash(a) != split_hash(make_split(y, 6))
    assert_disjoint(a)
    assert y[a["benign_train"]].sum() == 0 and y[a["benign_val"]].sum() == 0   # unsupervised never sees attacks
    assert len(np.concatenate([a["train"], a["val"], a["test"]])) == len(y)


def test_dedup_removes_exact_and_near_before_split():
    d = make_synthetic_flows(2000, seed=3, dup_frac=0.05, near_dup_frac=0.05)
    g = exact_duplicate_groups(d)
    assert len(np.unique(g)) < len(d)
    e, _ = deduplicate(d, "exact")
    assert len(np.unique(exact_duplicate_groups(e))) == len(e)
    n, _ = deduplicate(d, "near")
    assert len(n) < len(e)
    assert len(near_duplicate_pairs(n)["pairs"]) < len(near_duplicate_pairs(e)["pairs"])


def test_leakage_report_counts_cross_split_copies():
    d = make_synthetic_flows(1500, seed=4, dup_frac=0.1)
    sp = make_split(d.y, 0)
    r = leakage_report(d, {k: sp[k] for k in ("train", "val", "test")})
    assert r["exact_duplicates"]["redundant_rows"] > 0
    assert r["cross_split_exact"]["train__test"]["rows_in_b_with_copy_in_a"] > 0
    ddp, _ = deduplicate(d, "exact"); sp2 = make_split(ddp.y, 0)
    r2 = leakage_report(ddp, {k: sp2[k] for k in ("train", "val", "test")})
    assert r2["cross_split_exact"]["train__test"]["rows_in_b_with_copy_in_a"] == 0


def test_holm_and_cliffs_known_values():
    assert np.allclose(holm_bonferroni([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert cliffs_delta([3, 4, 5], [0, 1, 2]) == 1.0 and cliffs_delta([1, 2], [1, 2]) == 0.0


def test_bootstrap_ci_contains_mean_and_is_seeded():
    v = np.random.default_rng(0).normal(5, 1, 30)
    c1, c2 = bootstrap_ci(v, 2000, seed=1), bootstrap_ci(v, 2000, seed=1)
    assert c1 == c2 and c1["ci_low"] < v.mean() < c1["ci_high"]


def test_wilcoxon_degenerate_does_not_fabricate():
    w = wilcoxon_paired([1, 1, 1], [1, 1, 1])
    assert w["p_value"] == 1.0 and "zero" in w["note"]


def test_paired_comparisons_use_intersection_of_seeds():
    per = {"a": {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0}, "b": {0: 0.0, 1: 1.0, 2: 2.5, 3: 3.0, 4: 4.5, 6: 9.0}}
    r = paired_comparisons(per, "m")[0]
    assert r["n_pairs"] == 5 and r["mean_diff_a_minus_b"] > 0
