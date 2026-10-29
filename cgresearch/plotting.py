from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def curve_band(path, grid, curves_by_model, xlabel, ylabel, title, logx=False):
    """curves_by_model[model] -> array (n_seeds, len(grid)); plot mean with 5-95 percentile band."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for m, arr in sorted(curves_by_model.items()):
        arr = np.asarray(arr)
        ax.plot(grid, arr.mean(0), label=m)
        ax.fill_between(grid, np.percentile(arr, 5, 0), np.percentile(arr, 95, 0), alpha=0.15)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def score_hist(path, benign, attack, title, bins=60):
    fig, ax = plt.subplots(figsize=(6, 4))
    lo, hi = np.percentile(np.r_[benign, attack], [0.5, 99.5])
    ax.hist(np.clip(benign, lo, hi), bins=bins, alpha=0.55, density=True, label="benign")
    ax.hist(np.clip(attack, lo, hi), bins=bins, alpha=0.55, density=True, label="attack")
    ax.set_xlabel("score (higher = more anomalous)"); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
