"""UNSW-NB15 loader. Accepts a CSV file or a directory of CSVs.

Supports (a) the 4 raw headerless files UNSW-NB15_1..4.csv (49 columns, includes IPs, ports,
timestamps) and (b) headered files such as the partitioned training/testing sets (no IPs/ports;
the audit then falls back to feature-only duplicate detection).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .. import provenance as prov

RAW_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl",
    "sloss", "dloss", "service", "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "sjit", "djit", "stime", "ltime", "sintpkt",
    "dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd",
    "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label",
]
META_COLS = ["srcip", "dstip", "sport", "dsport", "stime", "ltime", "id"]
LABEL_COLS = ["label", "attack_cat"]
CATEGORICAL = ["proto", "state", "service"]
PORT_COLS = ["sport", "dsport"]


@dataclass
class FlowData:
    """Rows are flows. `X` = modelling features (incl. proto/state/service and ports as numeric)."""
    X: pd.DataFrame
    y: np.ndarray                      # 1 = attack, 0 = benign
    meta: pd.DataFrame                 # srcip,dstip,sport,dsport,stime when available
    attack_cat: np.ndarray
    info: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.y)

    def subset(self, idx) -> "FlowData":
        idx = np.asarray(idx)
        return FlowData(self.X.iloc[idx].reset_index(drop=True), self.y[idx],
                        self.meta.iloc[idx].reset_index(drop=True), self.attack_cat[idx], dict(self.info))

    @property
    def numeric_cols(self) -> List[str]:
        return [c for c in self.X.columns if c not in CATEGORICAL and c not in PORT_COLS]

    @property
    def categorical_cols(self) -> List[str]:
        return [c for c in CATEGORICAL if c in self.X.columns]

    @property
    def port_cols(self) -> List[str]:
        return [c for c in PORT_COLS if c in self.X.columns]


def _to_port(s: pd.Series) -> pd.Series:
    def conv(v):
        if isinstance(v, str):
            v = v.strip()
            try:
                return float(int(v, 16)) if v.lower().startswith("0x") else float(v)
            except ValueError:
                return np.nan
        return v
    return pd.to_numeric(s.map(conv), errors="coerce")


def feature_names_from_file(path: str | Path) -> List[str]:
    """Official NUSW-NB15_features.csv -> lower-case column names (49), in file order."""
    f = pd.read_csv(path, encoding="latin-1")
    names = [str(n).strip().lower() for n in f.iloc[:, 1].tolist()]
    fix = {"sintpkt": "sintpkt", "dintpkt": "dintpkt", "sjit": "sjit", "djit": "djit", "ct_src_ ltm": "ct_src_ltm",
           "ct_src_dport_ltm": "ct_src_dport_ltm", "ct_dst_sport_ltm": "ct_dst_sport_ltm", "label": "label"}
    return [fix.get(n, n) for n in names]


def stratified_sample(df: pd.DataFrame, label_col: str, n: int, seed: int) -> pd.DataFrame:
    """Stratified random sample of n rows preserving the label proportions (documented, seeded)."""
    y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    rng = np.random.default_rng(seed)
    parts = []
    for lab in sorted(y.unique()):
        idx = np.nonzero(y.to_numpy() == lab)[0]
        k = int(round(n * len(idx) / len(df)))
        parts.append(rng.choice(idx, min(k, len(idx)), replace=False))
    sel = np.sort(np.concatenate(parts))
    return df.iloc[sel].reset_index(drop=True)


def stratified_subset(data: "FlowData", n: int, seed: int) -> "FlowData":
    """Stratified (by label) random subset of an already-loaded FlowData, original row order preserved.
    Used AFTER deduplication so that duplicate rates describe the full file, not the subsample."""
    if n >= len(data):
        return data
    rng = np.random.default_rng(seed)
    parts = []
    for lab in (0, 1):
        idx = np.nonzero(data.y == lab)[0]
        k = int(round(n * len(idx) / len(data)))
        parts.append(rng.choice(idx, min(k, len(idx)), replace=False))
    return data.subset(np.sort(np.concatenate(parts)))


def load_unsw(path: str | Path, version: Optional[str] = None, max_rows: Optional[int] = None,
              sample_n: Optional[int] = None, sample_seed: int = 0, features_file: Optional[str] = None) -> FlowData:
    """sample_n/sample_seed reproduce a 'Sample A' style stratified subsample (record both in the config).
    features_file: official NUSW-NB15_features.csv, used to name headerless raw files."""
    p = Path(path)
    files = sorted(p.glob("*.csv")) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"no CSV files under {p}")
    frames = []
    for f in files:
        head = pd.read_csv(f, nrows=1, header=None, low_memory=False)
        headerless = head.shape[1] == len(RAW_COLUMNS) and not str(head.iloc[0, 0]).lower().startswith("srcip")
        if headerless:
            names = feature_names_from_file(features_file) if features_file else RAW_COLUMNS
            assert len(names) == len(RAW_COLUMNS), "feature-name file must list 49 columns"
            df = pd.read_csv(f, header=None, names=names, low_memory=False)
        else:
            df = pd.read_csv(f, low_memory=False)
            df.columns = [c.strip().lower() for c in df.columns]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if sample_n:
        df = stratified_sample(df, "label", sample_n, sample_seed)
    if max_rows:
        df = df.iloc[:max_rows].reset_index(drop=True)
    if "label" not in df:
        raise ValueError("expected a 'Label' column")
    y = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).to_numpy()
    cat = df["attack_cat"].fillna("Normal").astype(str).str.strip().to_numpy() if "attack_cat" in df else np.where(y == 1, "Attack", "Normal")
    for c in PORT_COLS:
        if c in df:
            df[c] = _to_port(df[c])
    for c in ("stime", "ltime"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    meta = df[[c for c in META_COLS if c in df]].copy()
    drop = set(META_COLS) | set(LABEL_COLS)
    X = df[[c for c in df.columns if c not in drop or c in PORT_COLS]].copy()
    for c in CATEGORICAL:
        if c in X:
            X[c] = X[c].fillna("-").astype(str).str.strip().str.lower()
    for c in X.columns:
        if c not in CATEGORICAL:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    num = [c for c in X.columns if c not in CATEGORICAL]
    X[num] = X[num].fillna(X[num].median(numeric_only=True)).fillna(0.0)
    info = prov.dataset_info(path, version)
    info["n_rows"] = int(len(df))
    info["sample"] = {"n": sample_n, "seed": sample_seed, "prevalence": float(y.mean())} if sample_n else None
    return FlowData(X.reset_index(drop=True), y, meta.reset_index(drop=True), cat, info)
