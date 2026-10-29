"""Leakage & duplicate audit.

Exact duplicates: identical feature rows (optionally including endpoint metadata).
Near duplicates: same (srcip,dstip,proto,dsport) block, start times within `time_window` seconds,
and cosine similarity of standardized feature vectors >= `cos_threshold`.
Without endpoint/time metadata the near-duplicate check degrades to feature-only blocking
(proto/service) over a sorted window; the report says which mode was used.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .unsw import FlowData, CATEGORICAL


def row_hashes(data: FlowData, include_meta: bool = False) -> np.ndarray:
    df = data.X.copy()
    if include_meta:
        df = pd.concat([df, data.meta.drop(columns=[c for c in ("stime", "ltime") if c in data.meta])], axis=1)
    return pd.util.hash_pandas_object(df, index=False).to_numpy()


def exact_duplicate_groups(data: FlowData, include_meta: bool = False) -> np.ndarray:
    """Return group id per row (rows sharing an id are exact duplicates)."""
    h = row_hashes(data, include_meta)
    _, inv = np.unique(h, return_inverse=True)
    return inv


def standardized_matrix(data: FlowData) -> np.ndarray:
    num = data.X[data.numeric_cols + data.port_cols].to_numpy(float)
    num = np.sign(num) * np.log1p(np.abs(num))
    mu, sd = num.mean(0), num.std(0)
    sd[sd == 0] = 1.0
    return (num - mu) / sd


def near_duplicate_pairs(data: FlowData, time_window: float = 5.0, cos_threshold: float = 0.999,
                         window: int = 50, max_pairs: int = 5_000_000) -> Dict:
    Z = standardized_matrix(data)
    Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    m = data.meta
    has_endpoints = all(c in m for c in ("srcip", "dstip", "stime"))
    if has_endpoints:
        key_cols = [c for c in ("srcip", "dstip", "dsport") if c in m] + (["proto"] if "proto" in data.X else [])
        kdf = pd.DataFrame({c: (m[c] if c in m else data.X[c]) for c in key_cols})
        kid = pd.util.hash_pandas_object(kdf, index=False).to_numpy()
        tvals = m["stime"].to_numpy(float)
        mode = "endpoints+time+cosine"
    else:
        key_cols = [c for c in ("proto", "service", "state") if c in data.X]
        kid = pd.util.hash_pandas_object(data.X[key_cols], index=False).to_numpy()
        tvals = np.arange(len(data), dtype=float)
        time_window = float(window)
        mode = "feature-block+order-window+cosine (no endpoint/time metadata)"
    order = np.lexsort((tvals, kid))
    pairs: List = []
    ks, ts = kid[order], tvals[order]
    n = len(order)
    for off in range(1, window + 1):
        if off >= n:
            break
        same = (ks[:-off] == ks[off:]) & ((ts[off:] - ts[:-off]) <= time_window)
        i = np.nonzero(same)[0]
        if not len(i):
            continue
        a, b = order[i], order[i + off]
        cs = np.einsum("ij,ij->i", Zn[a], Zn[b])
        ok = cs >= cos_threshold
        pairs.extend(zip(a[ok].tolist(), b[ok].tolist(), cs[ok].tolist()))
        if len(pairs) > max_pairs:
            break
    return {"pairs": pairs, "mode": mode, "time_window": time_window, "cos_threshold": cos_threshold}


def _components(n: int, pairs) -> np.ndarray:
    parent = np.arange(n)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b, *_ in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])


def deduplicate(data: FlowData, mode: str = "exact", **near_kw):
    """Return (deduplicated FlowData, kept_index_into_original). First occurrence is kept.
    mode: none | exact | near (near implies exact first)."""
    n = len(data)
    if mode == "none":
        return data, np.arange(n)
    gid = exact_duplicate_groups(data)
    keep = np.zeros(n, bool)
    _, first = np.unique(gid, return_index=True)
    keep[first] = True
    idx = np.nonzero(keep)[0]
    out = data.subset(idx)
    if mode == "near":
        res = near_duplicate_pairs(out, **near_kw)
        comp = _components(len(out), res["pairs"])
        k2 = np.zeros(len(out), bool)
        _, f2 = np.unique(comp, return_index=True)
        k2[f2] = True
        idx = idx[np.nonzero(k2)[0]]
        out = data.subset(idx)
    return out, idx


def leakage_report(data: FlowData, splits: Optional[Dict[str, np.ndarray]] = None,
                   n_examples: int = 5, **near_kw) -> Dict:
    n = len(data)
    gid = exact_duplicate_groups(data)
    counts = np.bincount(gid)
    dup_mask = counts[gid] > 1
    n_redundant = int((counts - 1).clip(min=0).sum())
    # label conflicts: identical features, different labels
    df = pd.DataFrame({"g": gid, "y": data.y})
    conflict_groups = df.groupby("g")["y"].nunique()
    conflict_groups = conflict_groups[conflict_groups > 1].index.to_numpy()
    first = np.zeros(n, bool)
    first[np.unique(gid, return_index=True)[1]] = True
    redundant_row = ~first
    by_label = {int(l): {"rows_in_duplicate_groups": int((dup_mask & (data.y == l)).sum()),
                         "redundant_copies": int((redundant_row & (data.y == l)).sum())} for l in (0, 1)}
    by_cat = pd.Series(data.attack_cat[dup_mask]).value_counts().to_dict()
    examples = []
    for g in np.nonzero(counts > 1)[0][:n_examples]:
        rows = np.nonzero(gid == g)[0]
        examples.append({"row_indices": rows[:5].tolist(), "multiplicity": int(counts[g]),
                         "label": int(data.y[rows[0]]), "features": data.X.iloc[rows[0]].to_dict()})
    near = near_duplicate_pairs(data, **near_kw)
    pairs = near["pairs"]
    in_near = np.zeros(n, bool)
    for a, b, _ in pairs:
        in_near[a] = in_near[b] = True
    near_examples = [{"i": a, "j": b, "cosine": round(c, 6), "label_i": int(data.y[a]), "label_j": int(data.y[b]),
                      "meta_i": data.meta.iloc[a].to_dict(), "meta_j": data.meta.iloc[b].to_dict()}
                     for a, b, c in pairs[:n_examples]]
    rep = {
        "n_rows": n,
        "exact_duplicates": {"redundant_rows": n_redundant, "pct": 100 * n_redundant / max(n, 1),
                             "rows_in_dup_groups": int(dup_mask.sum()), "by_label": by_label,
                             "by_attack_cat": by_cat, "label_conflict_groups": int(len(conflict_groups)),
                             "examples": examples},
        "near_duplicates": {"mode": near["mode"], "time_window": near["time_window"],
                            "cos_threshold": near["cos_threshold"], "pairs_found": len(pairs),
                            "rows_involved": int(in_near.sum()), "pct_rows_involved": 100 * in_near.mean(),
                            "label_mismatch_pairs": int(sum(data.y[a] != data.y[b] for a, b, _ in pairs)),
                            "examples": near_examples},
    }
    if splits:
        names = list(splits)
        h = {k: set(gid[v].tolist()) for k, v in splits.items()}
        cross = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                shared_groups = h[a] & h[b]
                cross[f"{a}__{b}"] = {
                    "shared_exact_groups": len(shared_groups),
                    "rows_in_b_with_copy_in_a": int(np.isin(gid[splits[b]], list(h[a])).sum()),
                    "pct_of_b": 100 * float(np.isin(gid[splits[b]], list(h[a])).mean()) if len(splits[b]) else 0.0}
        rep["cross_split_exact"] = cross
        within = {k: int((np.bincount(gid[v]) - 1).clip(min=0).sum()) for k, v in splits.items()}
        rep["within_split_redundant_rows"] = within
        near_cross = {}
        sid = np.full(n, -1)
        for k, (name, idx) in enumerate(splits.items()):
            sid[idx] = k
        cross_pairs = [(a, b) for a, b, _ in pairs if sid[a] != sid[b] and sid[a] >= 0 and sid[b] >= 0]
        rep["cross_split_near_pairs"] = len(cross_pairs)
    return rep
