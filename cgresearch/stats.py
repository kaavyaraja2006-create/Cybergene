"""Statistics: bootstrap CI, paired Wilcoxon + Holm-Bonferroni, Cliff's delta. No fabricated values:
degenerate cases return NaN with a note."""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats as sps


def bootstrap_ci(values: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0,
                 stat=np.mean) -> Dict[str, float]:
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = stat(v[idx], axis=1)
    return {"ci_low": float(np.percentile(boots, 100 * alpha / 2)),
            "ci_high": float(np.percentile(boots, 100 * (1 - alpha / 2)))}


def summarize(values: Sequence[float], n_boot: int = 10_000, seed: int = 0) -> Dict[str, float]:
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    out = {"n": int(len(v)), "mean": float(v.mean()) if len(v) else np.nan,
           "std": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
           "median": float(np.median(v)) if len(v) else np.nan}
    out.update(bootstrap_ci(v, n_boot, seed=seed))
    return out


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """delta = P(a>b) - P(a<b) over all cross pairs; in [-1, 1]."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a[:, None] - b[None, :]
    return float(((diff > 0).sum() - (diff < 0).sum()) / diff.size)


def cliffs_label(d: float) -> str:
    x = abs(d)
    return "negligible" if x < 0.147 else "small" if x < 0.33 else "medium" if x < 0.474 else "large"


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return {"statistic": np.nan, "p_value": 1.0, "note": "all paired differences are zero"}
    try:
        r = sps.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", method="auto")
        return {"statistic": float(r.statistic), "p_value": float(r.pvalue), "note": ""}
    except ValueError as e:
        return {"statistic": np.nan, "p_value": np.nan, "note": str(e)}


def holm_bonferroni(p: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (NaN-safe: NaNs are left NaN and excluded from m)."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    order = np.argsort(pv)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pv[i])
        adj[i] = min(1.0, running)
    out[ok] = adj
    return out


def paired_comparisons(per_seed: Dict[str, Dict[int, float]], metric: str, pairs: List[tuple] | None = None) -> List[Dict]:
    """per_seed[model][seed] -> value. Seeds are intersected so pairs are strictly paired."""
    models = sorted(per_seed)
    pairs = pairs or list(combinations(models, 2))
    rows = []
    for a, b in pairs:
        seeds = sorted(set(per_seed[a]) & set(per_seed[b]))
        va = np.array([per_seed[a][s] for s in seeds]); vb = np.array([per_seed[b][s] for s in seeds])
        w = wilcoxon_paired(va, vb)
        d = cliffs_delta(va, vb)
        diff = va - vb
        rows.append({"metric": metric, "model_a": a, "model_b": b, "n_pairs": len(seeds),
                     "mean_diff_a_minus_b": float(diff.mean()), "median_diff": float(np.median(diff)),
                     "diff_ci_low": bootstrap_ci(diff)["ci_low"], "diff_ci_high": bootstrap_ci(diff)["ci_high"],
                     "wilcoxon_stat": w["statistic"], "p_value": w["p_value"], "note": w["note"],
                     "cliffs_delta": d, "cliffs_magnitude": cliffs_label(d)})
    adj = holm_bonferroni([r["p_value"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_holm"] = float(a) if not np.isnan(a) else np.nan
    return rows
