"""Cluster-level inference. The independent units are CLUSTERS (attack families, days, datasets) -- not random seeds.

v1 reported Wilcoxon p-values over 30 seeds that re-split ONE sample; those seeds are pseudo-replicates, so the
p-values (and 'Cliff's delta = 1') overstated certainty. Here:
  * cluster_bootstrap_ci : resample clusters with replacement (within-cluster seeds are averaged first);
  * paired_cluster_test  : exact/Monte-Carlo sign-flip permutation test on the per-cluster mean differences;
  * rank_summary         : Friedman / Iman-Davenport test + Nemenyi critical difference over (dataset x scenario) cells
                           (Demsar 2006), and rank-stability measures between evaluation protocols.
"""
from __future__ import annotations

from itertools import product
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from scipy import stats as sps


def _cluster_means(cv: Mapping[Any, Sequence[float]]) -> np.ndarray:
    m = [np.nanmean(np.asarray(v, float)) for v in cv.values() if len(v) and not np.all(np.isnan(np.asarray(v, float)))]
    return np.asarray(m, float)


def cluster_bootstrap_ci(cv: Mapping[Any, Sequence[float]], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    m = _cluster_means(cv)
    if len(m) == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_clusters": 0}
    rng = np.random.default_rng(seed)
    boots = m[rng.integers(0, len(m), size=(n_boot, len(m)))].mean(axis=1)
    return {"mean": float(m.mean()), "ci_low": float(np.percentile(boots, 100 * alpha / 2)),
            "ci_high": float(np.percentile(boots, 100 * (1 - alpha / 2))), "n_clusters": int(len(m))}


def paired_cluster_test(a: Mapping[Any, Sequence[float]], b: Mapping[Any, Sequence[float]], n_perm: int = 20_000,
                        n_boot: int = 10_000, seed: int = 0) -> Dict[str, float]:
    keys = [k for k in a if k in b]
    d = np.array([np.nanmean(np.asarray(a[k], float)) - np.nanmean(np.asarray(b[k], float)) for k in keys])
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0:
        return {"mean_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan, "n_clusters": 0, "min_p": np.nan}
    obs = abs(d.mean())
    if n <= 16:                                              # exact: enumerate all 2^n sign patterns
        signs = np.array(list(product([-1, 1], repeat=n)))
        null = np.abs((signs * d).mean(axis=1))
        p = float((null >= obs - 1e-15).mean())
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1, 1], size=(n_perm, n))
        null = np.abs((signs * d).mean(axis=1))
        p = float((1 + (null >= obs - 1e-15).sum()) / (n_perm + 1))
    ci = cluster_bootstrap_ci({i: [x] for i, x in enumerate(d)}, n_boot, 0.05, seed)
    return {"mean_diff": float(d.mean()), "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "p_value": p,
            "n_clusters": int(n), "min_p": float(2.0 ** (1 - n)) if n <= 60 else 0.0}   # smallest attainable p (honest about few clusters)


def proportion_ci(cv: Mapping[Any, Sequence[float]], **kw) -> Dict[str, float]:
    """Cluster bootstrap CI of a mean of 0/1 indicators (e.g. 'FPR exceeded the target')."""
    return cluster_bootstrap_ci(cv, **kw)


# ----------------------------------------------------------------------------- ranks
def average_ranks(M: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """M: cells x methods. Returns the per-cell rank matrix (1 = best, ties averaged)."""
    X = -M if higher_is_better else M
    return np.vstack([sps.rankdata(row, method="average") for row in X])


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    q = sps.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


def rank_summary(M: np.ndarray, methods: Sequence[str], higher_is_better: bool = True) -> Dict[str, Any]:
    n, k = M.shape
    R = average_ranks(M, higher_is_better)
    avg = R.mean(axis=0)
    chi2 = 12.0 * n / (k * (k + 1)) * (np.sum(avg ** 2) - k * (k + 1) ** 2 / 4.0)
    denom = n * (k - 1) - chi2
    f = (n - 1) * chi2 / denom if denom > 0 else np.inf
    p = float(sps.f.sf(f, k - 1, (k - 1) * (n - 1))) if np.isfinite(f) else 0.0
    return {"methods": list(methods), "avg_rank": avg.tolist(), "friedman_chi2": float(chi2), "iman_davenport_F": float(f),
            "p_value": p, "nemenyi_cd": nemenyi_cd(k, n), "n_cells": int(n)}


def kendall_tau(rank_a: Sequence[float], rank_b: Sequence[float]) -> float:
    t = sps.kendalltau(rank_a, rank_b).statistic
    return float(t) if t == t else float("nan")


def rank_stability(Ma: np.ndarray, Mb: np.ndarray, higher_is_better: bool = True) -> Dict[str, float]:
    """Compare detector rankings of two protocols over the same cells x methods: mean per-cell Kendall tau, share of
    cells where the winner changes, and Kendall tau of the average ranks."""
    Ra, Rb = average_ranks(Ma, higher_is_better), average_ranks(Mb, higher_is_better)
    taus = [kendall_tau(a, b) for a, b in zip(Ra, Rb)]
    flips = float(np.mean(np.argmin(Ra, axis=1) != np.argmin(Rb, axis=1)))
    return {"mean_cell_tau": float(np.nanmean(taus)), "winner_flip_rate": flips,
            "tau_of_avg_ranks": kendall_tau(Ra.mean(0), Rb.mean(0)), "n_cells": int(len(Ra))}
