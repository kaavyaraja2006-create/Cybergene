"""Paired, seed-determined splits. The split depends ONLY on (labels, seed, fractions): every
detector/baseline sees identical train/val/test indices for a given seed.

Protocol
  benign : train / val / test   (default 0.6 / 0.2 / 0.2)
  attack : train / val / test   (default 0.4 / 0.2 / 0.4)
Unsupervised models fit on benign_train ONLY. Supervised models fit on benign_train+attack_train.
Thresholds are calibrated on benign VAL only. Hyperparameters are tuned on VAL (benign+attack).
Test is touched once, for final scoring.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .. import provenance as prov


def make_split(y: np.ndarray, seed: int, benign_frac=(0.6, 0.2, 0.2), attack_frac=(0.4, 0.2, 0.4)) -> Dict[str, np.ndarray]:
    assert abs(sum(benign_frac) - 1) < 1e-9 and abs(sum(attack_frac) - 1) < 1e-9
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xC0FFEE]))
    out = {}
    parts = {}
    for lab, fr in ((0, benign_frac), (1, attack_frac)):
        idx = np.nonzero(y == lab)[0]
        idx = idx[rng.permutation(len(idx))]
        c1 = int(round(fr[0] * len(idx)))
        c2 = c1 + int(round(fr[1] * len(idx)))
        parts[lab] = (idx[:c1], idx[c1:c2], idx[c2:])
    for i, name in enumerate(("train", "val", "test")):
        out[name] = np.sort(np.r_[parts[0][i], parts[1][i]])
    out["benign_train"] = np.sort(parts[0][0])
    out["attack_train"] = np.sort(parts[1][0])
    out["benign_val"] = np.sort(parts[0][1])
    out["attack_val"] = np.sort(parts[1][1])
    out["benign_test"] = np.sort(parts[0][2])
    out["attack_test"] = np.sort(parts[1][2])
    return out


def split_hash(split: Dict[str, np.ndarray]) -> str:
    return prov.config_hash({k: prov.array_hash(v) for k, v in sorted(split.items())})


def assert_disjoint(split: Dict[str, np.ndarray]):
    a, b, c = (set(split[k].tolist()) for k in ("train", "val", "test"))
    assert not (a & b) and not (a & c) and not (b & c), "split partitions overlap"
