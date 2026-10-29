"""NetFlow-v2 loader (NF-UNSW-NB15-v2, NF-CSE-CIC-IDS2018-v2; Sarhan, Layeghy & Portmann 2022).

STATUS: written from the dataset documentation's column list; exercised by a fixture CSV in tests/test_nf_loader.py
but NOT yet run against the real files. Check `info['alignment']` / `info['cleaning']` on the first real run.
All columns not mapped below are kept as numeric features under their lower-case names.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .. import provenance as prov
from .unsw import FlowData, stratified_sample

ID_COLS = {"IPV4_SRC_ADDR": "srcip", "IPV4_DST_ADDR": "dstip", "L4_SRC_PORT": "sport", "L4_DST_PORT": "dsport"}
DROP = {"LABEL", "ATTACK", "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"}
PROTO_NAMES = {6: "tcp", 17: "udp", 1: "icmp"}


def load_nf(path: str | Path, version: Optional[str] = None, sample_n: Optional[int] = None, sample_seed: int = 0,
            max_rows: Optional[int] = None) -> FlowData:
    p = Path(path)
    files = sorted(p.glob("*.csv")) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"no CSV files under {p}")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    up = {c.upper(): c for c in df.columns}
    if "LABEL" not in up:
        raise ValueError("no 'Label' column found")
    df["__y"] = pd.to_numeric(df[up["LABEL"]], errors="coerce")
    blank = int(df["__y"].isna().sum())
    df = df[df["__y"].notna()].reset_index(drop=True)
    if sample_n:
        df = stratified_sample(df, "__y", sample_n, sample_seed)
    if max_rows:
        df = df.iloc[:max_rows].reset_index(drop=True)
    y = df["__y"].astype(int).to_numpy()
    cat = df[up["ATTACK"]].fillna("Benign").astype(str).str.strip().to_numpy() if "ATTACK" in up else np.where(y == 1, "Attack", "Benign")
    meta = pd.DataFrame({v: df[up[k]] for k, v in ID_COLS.items() if k in up})
    if "FLOW_START_MILLISECONDS" in up:
        meta["stime"] = pd.to_numeric(df[up["FLOW_START_MILLISECONDS"]], errors="coerce") / 1000.0
    for c in ("sport", "dsport"):
        if c in meta:
            meta[c] = pd.to_numeric(meta[c], errors="coerce")
    X = pd.DataFrame(index=df.index)
    if "PROTOCOL" in up:
        pn = pd.to_numeric(df[up["PROTOCOL"]], errors="coerce")
        X["proto"] = pn.map(lambda v: PROTO_NAMES.get(int(v), "other") if pd.notna(v) else "other")
    for k, v in ID_COLS.items():
        if v in ("sport", "dsport") and k in up:
            X[v] = pd.to_numeric(df[up[k]], errors="coerce")
    used = set(DROP) | {"PROTOCOL", "__Y", "L4_SRC_PORT", "L4_DST_PORT"}
    for c in df.columns:
        if c.upper() in used:
            continue
        X[c.lower()] = pd.to_numeric(df[c], errors="coerce")
    alignment = {}
    def pick(*names):
        for n in names:
            if n in up:
                return pd.to_numeric(df[up[n]], errors="coerce")
        return None
    dur = pick("FLOW_DURATION_MILLISECONDS")
    X["dur"] = (dur / 1000.0) if dur is not None else 0.0
    for tgt, srcs in (("spkts", ("IN_PKTS",)), ("dpkts", ("OUT_PKTS",)), ("sbytes", ("IN_BYTES",)), ("dbytes", ("OUT_BYTES",))):
        v = pick(*srcs)
        X[tgt] = v if v is not None else 0.0
        alignment[tgt] = srcs[0] if v is not None else None
    X["rate"] = (X["spkts"] + X["dpkts"]) / np.maximum(X["dur"], 1e-3)
    X["sload"] = X["sbytes"] * 8 / np.maximum(X["dur"], 1e-3)
    X["dload"] = X["dbytes"] * 8 / np.maximum(X["dur"], 1e-3)
    num = [c for c in X.columns if c != "proto"]
    arr = X[num].replace([np.inf, -np.inf], np.nan)
    cleaning = {"unlabelled_rows_dropped": blank, "inf_or_nan_cells": int(arr.isna().sum().sum())}
    X[num] = arr.fillna(arr.median(numeric_only=True)).fillna(0.0)
    const = [c for c in num if X[c].nunique() <= 1]
    cleaning["constant_columns_dropped"] = const
    X = X.drop(columns=const)
    info = prov.dataset_info(path, version)
    info.update({"n_rows": int(len(df)), "alignment": alignment, "cleaning": cleaning,
                 "sample": {"n": sample_n, "seed": sample_seed} if sample_n else None, "attack_labels": pd.Series(cat[y == 1]).value_counts().to_dict()})
    return FlowData(X.reset_index(drop=True), y, meta.reset_index(drop=True), cat, info)
