"""Metrics with an enforced orientation contract: HIGHER SCORE == MORE ANOMALOUS / MORE ATTACK-LIKE.
Alert rule everywhere: alert iff score > threshold (strictly), threshold from benign VALIDATION.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import (average_precision_score, matthews_corrcoef, precision_recall_curve,
                             roc_auc_score, roc_curve)

TARGET_FPRS = (0.01, 0.02, 0.05, 0.10)


def calibrate_threshold(benign_val_scores: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold t such that the fraction of benign-val scores > t is <= target_fpr.
    Uses the (1-fpr) empirical quantile ('higher' interpolation => never exceeds target on val)."""
    s = np.sort(np.asarray(benign_val_scores, float))
    k = int(np.ceil((1.0 - target_fpr) * len(s))) - 1
    k = min(max(k, 0), len(s) - 1)
    return float(s[k])


def tie_fraction_at(scores: np.ndarray, thr: float) -> float:
    return float(np.mean(np.asarray(scores) == thr))


def threshold_metrics(y: np.ndarray, scores: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (scores > thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    mcc = float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 or len(np.unique(y)) > 1 else 0.0
    return {"threshold": float(thr), "precision": prec, "recall": rec, "f1": f1, "mcc": mcc,
            "measured_fpr": fp / (fp + tn) if fp + tn else 0.0, "alerts": int(pred.sum()),
            "alert_rate": float(pred.mean()), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def threshold_free(y: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    return {"roc_auc": float(roc_auc_score(y, scores)), "pr_auc": float(average_precision_score(y, scores))}


def curves(y: np.ndarray, scores: np.ndarray, grid: int = 501) -> Dict[str, np.ndarray]:
    """Full ROC and PR curves + metric-vs-FPR curves on a common FPR grid (for averaging across seeds)."""
    fpr, tpr, thr = roc_curve(y, scores)
    p, r, _ = precision_recall_curve(y, scores)
    g = np.linspace(0, 1, grid)
    rec_g = np.interp(g, fpr, tpr)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    tp = rec_g * pos
    fp = g * neg
    prec_g = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-12), 1.0)
    f1_g = np.where(prec_g + rec_g > 0, 2 * prec_g * rec_g / np.maximum(prec_g + rec_g, 1e-12), 0.0)
    return {"roc_fpr": fpr, "roc_tpr": tpr, "pr_precision": p, "pr_recall": r,
            "fpr_grid": g, "recall_at_fpr": rec_g, "precision_at_fpr": prec_g, "f1_at_fpr": f1_g}


def recall_at_fpr(y: np.ndarray, scores: np.ndarray, fpr: float) -> float:
    f, t, _ = roc_curve(y, scores)
    return float(np.interp(fpr, f, t))


def implied_fpr(precision: float, recall: float, prevalence: float) -> float:
    """FPR implied by reported precision p, recall r and prevalence pi (paper Eq. 5):
    FPR = r*pi*(1-p) / (p*(1-pi)). NaN if p==0. Used only to compare against MEASURED FPR."""
    if precision <= 0 or prevalence >= 1:
        return float("nan")
    return float(recall * prevalence * (1 - precision) / (precision * (1 - prevalence)))
