"""Identical preprocessing for every model. Fit on benign TRAIN only.

port_mode:
  raw            ports as numeric columns, treated exactly like every other numeric column
                 (signed log1p + z-score); suspected artefact: ports are identifiers, not magnitudes
  separate_std   ports kept numeric but LINEARLY z-scored with their own scaler (no log), i.e.
                 "ports standardized separately" from the log-transformed flow features
  drop           ports removed
  onehot         top-K ports one-hot (+ 'other'); one-hot block NOT rescaled
  onehot_scaled  as onehot, then standardized (inflates rare categories) - for diagnosis only
Heavy-tailed numeric columns get signed log1p, then robust-clipped z-scores (numeric block
scaled separately from the categorical/one-hot block).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .data.unsw import FlowData


class FlowPreprocessor:
    def __init__(self, port_mode: str = "onehot", top_ports: int = 20, clip: float = 10.0,
                 standardize: bool = True):
        """standardize=False skips z-scoring/clipping of the numeric block (log1p is kept), for the
        feature-standardisation ablation."""
        assert port_mode in ("raw", "separate_std", "drop", "onehot", "onehot_scaled")
        self.port_mode, self.top_ports, self.clip, self.standardize = port_mode, top_ports, clip, standardize

    def fit(self, data: FlowData, idx: np.ndarray) -> "FlowPreprocessor":
        X = data.X.iloc[idx]
        self.num_cols = list(data.numeric_cols)
        self.cat_cols = list(data.categorical_cols)
        self.port_cols = list(data.port_cols)
        num = self._num(X, self.num_cols + (self.port_cols if self.port_mode == "raw" else []))
        self.mu, self.sd = num.mean(0), num.std(0)
        self.sd[self.sd == 0] = 1.0
        if self.port_mode == "separate_std":
            pv = X[self.port_cols].to_numpy(float) if self.port_cols else np.zeros((len(X), 0))
            self.pmu, self.psd = pv.mean(0), pv.std(0)
            self.psd[self.psd == 0] = 1.0
        self.cat_levels = {c: sorted(X[c].unique().tolist()) for c in self.cat_cols}
        self.port_levels = {}
        if self.port_mode in ("onehot", "onehot_scaled"):
            for c in self.port_cols:
                self.port_levels[c] = X[c].value_counts().index[: self.top_ports].tolist()
        blk = self._onehot(X)
        self.feature_sources = (list(self.num_cols) + (self.port_cols if self.port_mode in ("raw", "separate_std") else [])
                                + self._onehot_sources())
        self.oh_mu = blk.mean(0) if blk.size else np.zeros(0)
        self.oh_sd = blk.std(0) if blk.size else np.zeros(0)
        self.oh_sd[self.oh_sd == 0] = 1.0
        return self

    def _onehot_sources(self):
        src = []
        for c in self.cat_cols:
            src += [c] * len(self.cat_levels[c])
        for c, lv in self.port_levels.items():
            src += [c] * (len(lv) + 1)
        return src

    @staticmethod
    def _num(X: pd.DataFrame, cols: List[str]) -> np.ndarray:
        a = X[cols].to_numpy(float)
        return np.sign(a) * np.log1p(np.abs(a))

    def _onehot(self, X: pd.DataFrame) -> np.ndarray:
        blocks = []
        for c in self.cat_cols:
            for lv in self.cat_levels[c]:
                blocks.append((X[c].to_numpy() == lv).astype(float))
        for c, lvls in self.port_levels.items():
            v = X[c].to_numpy()
            seen = np.zeros(len(v), bool)
            for lv in lvls:
                m = v == lv
                blocks.append(m.astype(float))
                seen |= m
            blocks.append((~seen).astype(float))
        return np.column_stack(blocks) if blocks else np.zeros((len(X), 0))

    def transform(self, data: FlowData, idx: Optional[np.ndarray] = None) -> np.ndarray:
        X = data.X if idx is None else data.X.iloc[idx]
        num = self._num(X, self.num_cols + (self.port_cols if self.port_mode == "raw" else []))
        if self.standardize:
            num = np.clip((num - self.mu) / self.sd, -self.clip, self.clip)
        if self.port_mode == "separate_std" and self.port_cols:
            pv = np.clip((X[self.port_cols].to_numpy(float) - self.pmu) / self.psd, -self.clip, self.clip)
            num = np.hstack([num, pv])
        oh = self._onehot(X)
        if self.port_mode == "onehot_scaled":
            oh = np.clip((oh - self.oh_mu) / self.oh_sd, -self.clip, self.clip)
        return np.hstack([num, oh]).astype(np.float64)

    @property
    def name(self) -> str:
        return f"ports={self.port_mode},std={self.standardize}"
