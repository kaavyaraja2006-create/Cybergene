"""Diagnostic: is the benign drift between calibration and deployment confined to the UPPER TAIL (invisible to bulk-anchored recalibration)?

For every (model, later window) of the temporal scenario we use ground-truth benign scores (evaluation only) to measure
  * quantile shifts   Delta(u) = (Q_dep(u) - Q_cal(u)) / IQR_cal   at bulk levels (0.5, 0.7, 0.9) and tail levels (0.95, 0.99)
  * 'tail excess'     = mean(Delta(.95), Delta(.99)) - Delta(.7): how much more the tail moved than the bulk
and compare thresholds that are all FIT on the first half of the window and EVALUATED on its second half (no in-sample bias):
  static | oracle (clean benign half) | BAR on the CLEAN benign half (pure extrapolation error) | BAR(bulk=0.9) on the clean half |
  BAR on the real (attack-contaminated) unlabelled half | naive on the unlabelled half.
BAR-clean separates the two possible causes of BAR's failure: extrapolation error from the bulk (tail drift) vs window contamination.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.stats import spearmanr

from ..calibration import alert_metrics, bar_recalibrate, conformal_threshold
from ..foldscore import score_fold
from ..runlog import ExperimentRun
from ..scenarios import make_folds
from .common import prepare

LEVELS = (0.5, 0.7, 0.9, 0.95, 0.99)


def _thr(sc, a):
    t = conformal_threshold(sc, a)
    return t if np.isfinite(t) else float(np.quantile(sc, 1 - a))


def run_tail_diag(cfg: Dict[str, Any], out_root="results", argv=None) -> ExperimentRun:
    data, dsinfo, dd = prepare(cfg)
    run = ExperimentRun(cfg.get("name", "tail_diag"), cfg, out_root, argv, dsinfo, cfg["seeds"])
    run.manifest["dedup"] = dd
    models = cfg.get("models", ["isolation_forest", "pca", "hbos", "ecod"])
    alphas = cfg.get("alphas", [0.05])
    scen = {"kind": "temporal", **cfg.get("temporal", {})}
    folds = make_folds(data, "temporal", cfg["seeds"], scen)
    rows = []
    for k, fold in enumerate(folds, 1):
        scored = score_fold(data, fold, models, cfg.get("tuning", {}), cfg.get("port_mode", "onehot"), cfg.get("standardize", True), cfg.get("max_fit_rows"))
        for m, res in scored.items():
            c = np.asarray(res["cal"], float)
            iqr = max(float(np.subtract(*np.quantile(c, [0.75, 0.25]))), 1e-12)
            for wid, (s, y) in res["windows"].items():
                s = np.asarray(s, float); y = np.asarray(y, int)
                h = len(s) // 2
                s1, s2, y1, y2 = s[:h], s[h:], y[:h], y[h:]
                b1, b2 = s1[y1 == 0], s2[y2 == 0]
                if len(b1) < 300 or len(b2) < 300:
                    continue
                allb = s[y == 0]
                dl = {u: float((np.quantile(allb, u) - np.quantile(c, u)) / iqr) for u in LEVELS}
                for a in alphas:
                    def ev(tau):
                        return float((b2 > tau).mean()) / a
                    row = {"model": m, "seed": fold.seed, "window": wid, "alpha": a, "n_b1": len(b1), "n_b2": len(b2),
                           "attack_share_w": float(y.mean()),
                           **{f"shift_q{int(u * 100)}": dl[u] for u in LEVELS},
                           "tail_excess": 0.5 * (dl[0.95] + dl[0.99]) - dl[0.7]}
                    row["ratio_static"] = ev(_thr(c, a))
                    row["ratio_oracle"] = ev(_thr(b1, a))
                    for tag, bulk, win in (("bar_clean", 0.7, b1), ("bar_clean_b90", 0.9, b1), ("bar_window", 0.7, s1)):
                        try:
                            new, info = bar_recalibrate(c, win, a, _thr, mode="auto", bulk=bulk)
                            tau = _thr(new, a)
                            row[f"ratio_{tag}"] = ev(tau); row[f"mode_{tag}"] = info["mode"]
                            row[f"recall_{tag}"] = alert_metrics(s2, y2, tau)["recall"]
                        except Exception as e:  # noqa
                            row[f"ratio_{tag}"] = float("nan"); row[f"mode_{tag}"] = repr(e)[:40]
                    tn = _thr(s1, a)
                    row["ratio_naive_window"] = ev(tn); row["recall_naive_window"] = alert_metrics(s2, y2, tn)["recall"]
                    row["recall_static"] = alert_metrics(s2, y2, _thr(c, a))["recall"]
                    rows.append(row)
        print(f"[{run.experiment}] temporal fold {k}/{len(folds)} done", flush=True)
    run.add_rows("per_window", rows)
    # ---- summaries
    summ = []
    for m in sorted({r["model"] for r in rows}) + ["ALL"]:
        rr = [r for r in rows if (m == "ALL" or r["model"] == m) and r["window"] != "control" and r["alpha"] == alphas[0]]
        if not rr:
            continue
        mean = lambda key: float(np.nanmean([r[key] for r in rr]))
        te = [r["tail_excess"] for r in rr]
        ext_err = [r["ratio_bar_clean"] - r["ratio_oracle"] for r in rr]
        ok = np.isfinite(ext_err)
        rho = spearmanr(np.array(te)[ok], np.array(ext_err)[ok]).statistic if ok.sum() > 4 else float("nan")
        summ.append({"model": m, "n": len(rr), "mean_tail_excess": float(np.mean(te)),
                     **{f"mean_shift_q{int(u * 100)}": mean(f"shift_q{int(u * 100)}") for u in LEVELS},
                     **{k: mean(k) for k in ("ratio_static", "ratio_oracle", "ratio_bar_clean", "ratio_bar_clean_b90", "ratio_bar_window", "ratio_naive_window",
                                             "recall_static", "recall_bar_window", "recall_naive_window")},
                     "spearman_tail_excess_vs_extrapolation_error": float(rho)})
    run.save_table("summary", summ)
    run.finish()
    return run
