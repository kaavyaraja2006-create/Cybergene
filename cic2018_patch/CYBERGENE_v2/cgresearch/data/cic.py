"""CIC-IDS2017 / CSE-CIC-IDS2018 loader with EXPLICIT feature alignment.

Column names differ across releases ('Flow Duration' vs 'Flow Duration', 'Tot Fwd Pkts' vs 'Total Fwd
Packets', 'Dst Port' vs 'Destination Port', ...). Names are normalised (lower-case, non-alphanumerics
-> '_') and mapped through ALIASES to the common schema used by the pipeline and by the rule-based
detector. The mapping is saved in info['alignment'] so it is auditable. Results must be reported per
dataset; nothing here merges datasets.

Units: CIC 'Flow Duration' is in microseconds -> converted to seconds ('dur').
Known data issues handled and counted (recorded in info['cleaning']): inf/NaN values, header rows
repeated inside 2018 files, negative durations/lengths (kept but counted), exact duplicates
(handled by the audit, not here).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .. import provenance as prov
from .unsw import FlowData, stratified_sample

ALIASES: Dict[str, List[str]] = {
    "dur": ["flow_duration"],
    "spkts": ["total_fwd_packets", "tot_fwd_pkts"],
    "dpkts": ["total_backward_packets", "tot_bwd_pkts"],
    "sbytes": ["total_length_of_fwd_packets", "totlen_fwd_pkts"],
    "dbytes": ["total_length_of_bwd_packets", "totlen_bwd_pkts"],
    "rate": ["flow_packets_s", "flow_pkts_s"],
    "sport": ["source_port", "src_port"],
    "dsport": ["destination_port", "dst_port"],
    "proto": ["protocol"],
    "srcip": ["source_ip", "src_ip"],
    "dstip": ["destination_ip", "dst_ip"],
    "stime": ["timestamp"],
    "label": ["label"],
    "flow_id": ["flow_id"],
}
PROTO_NAMES = {6: "tcp", 17: "udp", 1: "icmp", 0: "hopopt"}


_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def day_label(stem: str) -> str:
    """Sortable day label from a CIC file name: '0_monday' ... (2017 files carry the weekday) or the raw stem
    (2018 files are named by date, which sorts chronologically)."""
    low = str(stem).lower()
    for i, d in enumerate(_DAYS):
        if d in low:
            return f"{i}_{d}"
    return str(stem)


def norm(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


_KEEP_TEXT = {"label", "timestamp", "flow_id", "src_ip", "dst_ip", "source_ip", "destination_ip", "zz_src"}


def _reduce_file(f: Path, cap: int, seed: int, benign_label: str, log: dict) -> pd.DataFrame:
    """Memory-safe per-file reduction for very large days (CSE-CIC-IDS2018 is ~16M rows).

    Only applied when the file has more than `cap` rows; smaller files are returned untouched, so with a large cap the
    result is identical to the plain loader. Steps: drop repeated-header / blank-label rows, convert feature columns to
    numeric, drop exact duplicate rows, then keep ALL attack labels up to cap//8 rows each (rare families must survive
    for leave-one-family-out) and benign up to cap rows, original order preserved. This changes the class prevalence;
    FPR/recall are unaffected, precision is not comparable to the uncapped file. Everything is logged in `info`."""
    df = pd.read_csv(f, low_memory=False, encoding="latin-1", skipinitialspace=True)
    df.columns = [norm(c) for c in df.columns]
    n_in = len(df)
    if n_in <= cap:
        return df.assign(zz_src=f.stem)
    lab = df["label"].astype(str).str.strip()
    bad = lab.str.lower().isin(["label", "", "nan", "none"]) | df["label"].isna()
    df = df[~bad]
    for c in df.columns:
        if c not in _KEEP_TEXT:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    n_clean = len(df)
    df = df.drop_duplicates()
    n_dedup = len(df)
    lab = df["label"].astype(str).str.strip()
    rng = np.random.default_rng(seed)
    keep = []
    for name, idx in lab.groupby(lab).indices.items():
        k = cap if name == benign_label else max(1, cap // 8)
        keep.append(idx if len(idx) <= k else rng.choice(idx, k, replace=False))
    sel = np.sort(np.concatenate(keep))
    df = df.iloc[sel]
    log[f.stem] = {"rows_in": int(n_in), "after_header_blank_drop": int(n_clean), "after_exact_dedup": int(n_dedup),
                   "rows_kept": int(len(df)), "cap": int(cap)}
    return df.assign(zz_src=f.stem)


def load_cic(path: str | Path, version: Optional[str] = None, sample_n: Optional[int] = None,
             sample_seed: int = 0, max_rows: Optional[int] = None, benign_label: str = "BENIGN",
             per_file_cap: Optional[int] = None) -> FlowData:
    p = Path(path)
    files = sorted(p.glob("*.csv")) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"no CSV files under {p}")
    reduction: dict = {}
    if per_file_cap:
        df = pd.concat([_reduce_file(f, int(per_file_cap), sample_seed, benign_label, reduction) for f in files], ignore_index=True)
    else:
        df = pd.concat([pd.read_csv(f, low_memory=False, encoding="latin-1", skipinitialspace=True).assign(zz_src=f.stem) for f in files],
                       ignore_index=True)
        df.columns = [norm(c) for c in df.columns]
    if per_file_cap:
        df.columns = [norm(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    alignment, rename = {}, {}
    for common, names in ALIASES.items():
        hit = next((n for n in names if n in df.columns), None)
        if hit:
            rename[hit] = common; alignment[common] = hit
    cleaning = {"rows_raw": int(len(df))}
    if reduction:
        cleaning["per_file_reduction"] = reduction
    df = df.rename(columns=rename)
    if "label" not in df:
        raise ValueError("no label column found (expected 'Label')")
    rep_hdr = df["label"].astype(str).str.strip().str.lower() == "label"   # 2018 files repeat the header
    cleaning["repeated_header_rows_dropped"] = int(rep_hdr.sum())
    df = df[~rep_hdr].reset_index(drop=True)
    # CIC-IDS2017 CSVs contain rows whose cells are all blank (",,,,"). Their label is NaN, which
    # str() turns into "nan" != "BENIGN", so they were silently counted as attacks (v8 defect).
    lab_raw = df["label"]
    blank = lab_raw.isna() | (lab_raw.astype(str).str.strip().str.lower().isin(["", "nan", "none"]))
    cleaning["blank_label_rows_dropped"] = int(blank.sum())
    df = df[~blank].reset_index(drop=True)
    if sample_n:
        df["_y"] = (df["label"].astype(str).str.strip() != benign_label).astype(int)
        df = stratified_sample(df, "_y", sample_n, sample_seed).drop(columns="_y")
    if max_rows:
        df = df.iloc[:max_rows].reset_index(drop=True)
    # labels like "Web Attack \x96 Brute Force" carry a cp1252 en-dash: map any non-ASCII run to "-"
    lab = df["label"].astype(str).str.strip().str.replace(r"[^\x20-\x7e]+", "-", regex=True)
    y = (lab != benign_label).astype(int).to_numpy()
    meta_cols = [c for c in ("srcip", "dstip", "sport", "dsport", "stime", "flow_id") if c in df]
    meta = df[meta_cols].copy()
    meta["source_file"] = df["zz_src"].to_numpy()
    meta["day"] = [day_label(x) for x in meta["source_file"]]
    if "stime" in meta:
        first = pd.to_numeric(meta["stime"].astype(str).str.extract(r"^\s*(\d{1,2})[/-]")[0], errors="coerce")
        dayfirst = bool((first > 12).any())      # 2018 files are d/m/Y, 2017 files m/d/Y: decide from the data, never assume
        cleaning["timestamp_dayfirst"] = dayfirst
        ts = pd.to_datetime(meta["stime"], errors="coerce", dayfirst=dayfirst)
        meta["stime"] = (ts - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)      # NaT -> NaN (not a huge negative int)
        cleaning["unparseable_timestamps"] = int(meta["stime"].isna().sum())
    for c in ("sport", "dsport"):
        if c in meta:
            meta[c] = pd.to_numeric(meta[c], errors="coerce")
    X = df.drop(columns=[c for c in ("label", "srcip", "dstip", "stime", "flow_id", "zz_src") if c in df]).copy()
    for c in X.columns:
        if c != "proto":
            X[c] = pd.to_numeric(X[c], errors="coerce")
    if "proto" in X:
        pn = pd.to_numeric(X["proto"], errors="coerce")
        X["proto"] = pn.map(lambda v: PROTO_NAMES.get(int(v), "other") if pd.notna(v) else "other")
    for c in ("sport", "dsport"):
        if c in X:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    if "dur" in X:
        X["dur"] = X["dur"] / 1e6
    num = [c for c in X.columns if c != "proto"]
    arr = X[num].replace([np.inf, -np.inf], np.nan)
    cleaning["inf_or_nan_cells"] = int(arr.isna().sum().sum())
    cleaning["negative_cells"] = int((arr < 0).sum().sum())
    X[num] = arr.fillna(arr.median(numeric_only=True)).fillna(0.0)
    const = [c for c in num if X[c].nunique() <= 1]
    cleaning["constant_columns_dropped"] = const
    X = X.drop(columns=const)
    info = prov.dataset_info(path, version)
    info.update({"n_rows": int(len(df)), "alignment": alignment, "cleaning": cleaning,
                 "sample": {"n": sample_n, "seed": sample_seed} if sample_n else None,
                 "attack_labels": lab[y == 1].value_counts().to_dict()})
    return FlowData(X.reset_index(drop=True), y, meta.reset_index(drop=True), lab.to_numpy(), info)
