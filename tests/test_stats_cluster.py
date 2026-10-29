import numpy as np

from cgresearch.stats_cluster import (cluster_bootstrap_ci, paired_cluster_test, rank_summary, rank_stability,
                                      nemenyi_cd, kendall_tau)


def test_cluster_bootstrap_ignores_within_cluster_seed_count():
    a = {"d1": [0.2] * 30, "d2": [0.4] * 30, "d3": [0.6]}          # 30 seeds in a cluster must not outweigh 1 seed in another
    r = cluster_bootstrap_ci(a, seed=1)
    assert abs(r["mean"] - 0.4) < 1e-12 and r["n_clusters"] == 3 and r["ci_low"] < 0.4 < r["ci_high"]


def test_paired_test_is_exact_and_honest_about_few_clusters():
    a = {i: [0.9] for i in range(5)}; b = {i: [0.1] for i in range(5)}
    r = paired_cluster_test(a, b)
    assert abs(r["p_value"] - 2 / 32) < 1e-12                       # 5 clusters can never give p < 0.0625 (two-sided)
    assert r["n_clusters"] == 5
    rng = np.random.default_rng(0)
    a = {i: [rng.normal()] for i in range(40)}; b = {i: [rng.normal()] for i in range(40)}
    assert paired_cluster_test(a, b)["p_value"] > 0.05              # null: no false positive from 40 random clusters (seeded)


def test_friedman_nemenyi_and_rank_stability():
    rng = np.random.default_rng(2)
    base = np.array([0.9, 0.7, 0.5, 0.3])
    M = base + rng.normal(0, 0.05, (12, 4))
    s = rank_summary(M, list("ABCD"))
    assert s["p_value"] < 0.001 and s["avg_rank"][0] < s["avg_rank"][3] and 0.5 < s["nemenyi_cd"] < 3
    assert abs(nemenyi_cd(2, 20) - 1.96 * np.sqrt(6 / 120)) < 0.01  # k=2 reduces to the normal critical value
    same = rank_stability(M, M + rng.normal(0, 0.001, M.shape))
    assert same["mean_cell_tau"] > 0.9 and same["winner_flip_rate"] < 0.2
    rev = rank_stability(M, -M)
    assert rev["mean_cell_tau"] < -0.9 and rev["winner_flip_rate"] > 0.8
    assert kendall_tau([1, 2, 3], [1, 2, 3]) == 1.0
