"""Evaluation scenarios ("folds") for a benign-calibrated detector.

A Fold separates FOUR roles of benign data, which v1 conflated:
    fit   : the detector is trained here
    tune  : hyper-parameters are selected here (benign-only for label-free tuning; benign+attack for labelled tuning)
    cal   : the alert threshold is calibrated here  -- never used for fitting or selection, so the
            exchangeability assumption behind the conformal guarantees (calibration.py) is not broken
    test  : evaluation windows (what the deployed detector sees)

Three scenario kinds:
    random   : i.i.d. split (the v1 protocol). Calibration and test are exchangeable -> guarantees should hold.
    lofo     : leave-one-attack-family-out. Supervised models never see family f; the window holds f + benign.
    temporal : fit/tune/cal on the earliest block(s); evaluation windows are later blocks (days). Benign traffic
               itself drifts -> the guarantee's exchangeability assumption is violated; this is the stress test.
               A 'control' window (held-out benign from the calibration block) stays exchangeable.
The unit of statistical analysis for lofo/temporal is the fold/window (family, day), NOT the seed.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .data.unsw import FlowData


@dataclass
class Window:
    id: str
    idx: np.ndarray                      # row indices (time-ordered for temporal scenarios)


@dataclass
class Fold:
    kind: str
    name: str
    seed: int
    fit: np.ndarray
    tune_benign: np.ndarray
    tune_labeled: np.ndarray
    cal: np.ndarray
    sup: np.ndarray                      # labelled training rows for supervised baselines (may be empty)
    windows: List[Window] = field(default_factory=list)
    attack_pool: np.ndarray = field(default_factory=lambda: np.zeros(0, int))   # attack rows NOT in any test window (contamination studies)

    def check(self, y: np.ndarray):
        """Hard invariants: roles are disjoint where they must be; benign-only roles contain no attacks."""
        for nm in ("fit", "tune_benign", "cal"):
            assert (y[getattr(self, nm)] == 0).all(), f"{nm} contains attack rows"
        s = {nm: set(getattr(self, nm).tolist()) for nm in ("fit", "tune_benign", "cal")}
        assert not (s["fit"] & s["tune_benign"]) and not (s["fit"] & s["cal"]) and not (s["tune_benign"] & s["cal"]), "fit/tune/cal overlap"
        used = s["fit"] | s["tune_benign"] | s["cal"] | set(self.sup.tolist()) | set(self.tune_labeled.tolist())
        for w in self.windows:
            if self.kind == "temporal":
                continue                                  # later blocks never overlap by construction (checked in tests)
            assert not (set(w.idx.tolist()) & used), f"window {w.id} overlaps training/selection/calibration data"


def _split(idx: np.ndarray, fracs: Sequence[float], rng: np.random.Generator) -> List[np.ndarray]:
    idx = idx[rng.permutation(len(idx))]
    cuts = np.round(np.cumsum(fracs)[:-1] * len(idx)).astype(int)
    return [np.sort(p) for p in np.split(idx, cuts)]


def _rng(seed: int, salt: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), salt]))


def random_fold(y: np.ndarray, seed: int, benign_frac=(0.5, 0.15, 0.15, 0.2), attack_frac=(0.4, 0.2, 0.4)) -> Fold:
    rng = _rng(seed, 0x51)
    ben, att = np.nonzero(y == 0)[0], np.nonzero(y == 1)[0]
    bf, bt, bc, bte = _split(ben, benign_frac, rng)
    at, atu, ate = _split(att, attack_frac, rng)
    f = Fold("random", f"seed{seed}", seed, fit=bf, tune_benign=bt, tune_labeled=np.sort(np.r_[bt, atu]), cal=bc,
             sup=np.sort(np.r_[bf, at]), windows=[Window("test", np.sort(np.r_[bte, ate]))], attack_pool=np.sort(np.r_[at, atu]))
    f.check(y)
    return f


def lofo_folds(y: np.ndarray, families: np.ndarray, seed: int, min_support: int = 50,
               benign_frac=(0.5, 0.15, 0.15, 0.2), max_families: Optional[int] = None) -> List[Fold]:
    """One fold per attack family f: supervised data and tuning attacks exclude f; the window = held-out benign + ALL of f."""
    rng = _rng(seed, 0x10F0)
    ben = np.nonzero(y == 0)[0]
    bf, bt, bc, bte = _split(ben, benign_frac, rng)
    fams = [f for f in sorted(set(families[y == 1])) if (families[y == 1] == f).sum() >= min_support]
    if max_families:
        fams = fams[:max_families]
    out = []
    for f in fams:
        att_other = np.nonzero((y == 1) & (families != f))[0]
        att_f = np.nonzero((y == 1) & (families == f))[0]
        at, atu = _split(att_other, (0.7, 0.3), _rng(seed, zlib.crc32(str(f).encode())))
        fold = Fold("lofo", f"lofo:{f}", seed, fit=bf, tune_benign=bt, tune_labeled=np.sort(np.r_[bt, atu]), cal=bc,
                    sup=np.sort(np.r_[bf, at]), windows=[Window(f"lofo:{f}", np.sort(np.r_[bte, att_f]))],
                    attack_pool=np.sort(att_other))
        fold.check(y)
        out.append(fold)
    return out


def block_ids(stime: Optional[np.ndarray], n_blocks: int = 5, day: Optional[np.ndarray] = None, n: Optional[int] = None) -> np.ndarray:
    """Integer block id per row. Natural 'day' labels win; otherwise equal-count quantile bins of start time;
    otherwise equal-count bins of row order."""
    if day is not None:
        u = {d: i for i, d in enumerate(sorted(set(day.tolist())))}
        return np.array([u[d] for d in day.tolist()])
    if stime is not None and np.isfinite(stime).mean() > 0.9:
        t = np.where(np.isfinite(stime), stime, np.nanmedian(stime))
        order = np.argsort(t, kind="stable")
    else:
        order = np.arange(len(stime) if stime is not None else int(n))
    b = np.empty(len(order), int)
    b[order] = np.minimum((np.arange(len(order)) * n_blocks) // len(order), n_blocks - 1)
    return b


def temporal_fold(y: np.ndarray, blocks: np.ndarray, seed: int, fit_blocks: Sequence[int] = (0,), label_blocks: Sequence[int] = (1,),
                  eval_blocks: Optional[Sequence[int]] = None, benign_frac=(0.6, 0.15, 0.15, 0.1), order: Optional[np.ndarray] = None) -> Fold:
    """fit/tune/cal come from benign rows of `fit_blocks`; supervised/labelled-tuning data from `label_blocks`
    (may be empty -> supervised baselines are skipped); windows are the remaining blocks, time-ordered.
    A 'control' window = the held-out 10% benign of the fit blocks (exchangeable with cal)."""
    rng = _rng(seed, 0x7E3)
    fit_blocks, label_blocks = list(fit_blocks), list(label_blocks)
    order = np.arange(len(y)) if order is None else order            # time order of rows (stable)
    in_fit = np.isin(blocks, fit_blocks)
    ben = np.nonzero(in_fit & (y == 0))[0]
    bf, bt, bc, bte = _split(ben, benign_frac, rng)
    lab = np.nonzero(np.isin(blocks, label_blocks))[0]
    if len(lab):
        sup_idx, tune_idx = _split(lab, (0.7, 0.3), rng)
        sup = np.sort(np.r_[bf, sup_idx])
        tune_lab = np.sort(np.r_[bt, tune_idx])
    else:
        sup, tune_lab = np.zeros(0, int), np.sort(bt)
    all_blocks = sorted(set(blocks.tolist()))
    ev = [b for b in all_blocks if b not in fit_blocks and b not in label_blocks] if eval_blocks is None else list(eval_blocks)
    rank = np.empty(len(order), int); rank[order] = np.arange(len(order))
    wins = [Window("control", bte)]
    for b in ev:
        idx = np.nonzero(blocks == b)[0]
        wins.append(Window(f"block{b}", idx[np.argsort(rank[idx], kind="stable")]))
    pool = np.nonzero((y == 1) & np.isin(blocks, fit_blocks + label_blocks))[0]
    f = Fold("temporal", f"seed{seed}", seed, fit=bf, tune_benign=bt, tune_labeled=tune_lab, cal=bc, sup=sup, windows=wins, attack_pool=pool)
    f.check(y)
    return f


def make_folds(data: FlowData, kind: str, seeds: Sequence[int], cfg: Optional[dict] = None) -> List[Fold]:
    cfg = cfg or {}
    y = data.y
    if kind == "random":
        return [random_fold(y, s, tuple(cfg.get("benign_frac", (0.5, 0.15, 0.15, 0.2))), tuple(cfg.get("attack_frac", (0.4, 0.2, 0.4)))) for s in seeds]
    if kind == "lofo":
        out = []
        for s in seeds:
            out += lofo_folds(y, data.attack_cat, s, cfg.get("min_support", 50), max_families=cfg.get("max_families"))
        return out
    if kind == "temporal":
        st = data.meta["stime"].to_numpy(float) if "stime" in data.meta else None
        day = data.meta["day"].to_numpy() if "day" in data.meta else None
        if cfg.get("block_mode") == "stime" and st is not None and np.isfinite(st).mean() > 0.9:
            day = None                                      # ignore natural day labels: equal-count time bins (finer blocks than days)
        blocks = block_ids(st, cfg.get("n_blocks", 5), day, n=len(y))
        order = np.argsort(st, kind="stable") if st is not None and np.isfinite(st).all() else None
        return [temporal_fold(y, blocks, s, cfg.get("fit_blocks", (0,)), cfg.get("label_blocks", (1,)), cfg.get("eval_blocks"),
                              tuple(cfg.get("benign_frac", (0.6, 0.15, 0.15, 0.1))), order) for s in seeds]
    raise ValueError(kind)
