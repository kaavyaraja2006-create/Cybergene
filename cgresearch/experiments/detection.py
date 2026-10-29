"""Detection + FPR-sweep experiment (experiments 4,5,6 of the plan).

For every seed: ONE split shared by all models (paired); preprocessing fit on benign train;
hyperparameters tuned on validation only; thresholds calibrated on benign validation only;
test used once. Saves raw per-seed rows, curves, splits, summary statistics, paired tests.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict

import numpy as np

from ..data.splits import make_split, split_hash, assert_disjoint
from ..detectors import REGISTRY
from ..detectors.tuning import tune
from ..metrics import TARGET_FPRS, implied_fpr, calibrate_threshold, curves, threshold_free, threshold_metrics, tie_fraction_at
from ..plotting import curve_band
from ..preprocess import FlowPreprocessor
from ..runlog import ExperimentRun
from ..stats import paired_comparisons, summarize
from .common import prepare

THRESH_METRICS = ["precision", "recall", "f1", "mcc", "measured_fpr", "alerts", "alert_rate"]


def run_seed(data, seed, cfg, run: ExperimentRun | None = None):
    sp = make_split(data.y, seed, tuple(cfg.get("benign_frac", (0.6, 0.2, 0.2))), tuple(cfg.get("attack_frac", (0.4, 0.2, 0.4))))
    assert_disjoint(sp)
    pre = FlowPreprocessor(cfg.get("port_mode", "onehot"), standardize=cfg.get("standardize", True)).fit(data, sp["benign_train"])
    Xall = pre.transform(data)
    y = data.y
    tf_rows, fpr_rows, trials, curve_store, legacy_rows, trivial_rows = [], [], [], {}, [], []
    sh = split_hash(sp)
    for mname in cfg["models"]:
        cls = REGISTRY[mname]
        kw = {"feature_names": pre.num_cols} if mname == "rule_based" else {}
        sup = cls.supervised
        tr_idx = sp["train"] if sup else sp["benign_train"]
        Xtr, ytr = Xall[tr_idx], y[tr_idx]
        Xval, yval = Xall[sp["val"]], y[sp["val"]]
        if cfg.get("tune", True) and cls.grid:
            best, tr = tune(cls, mname, seed, Xtr, ytr, Xval, yval,
                            feature_names=pre.num_cols if mname == "rule_based" else None,
                            max_configs=cfg.get("max_configs", 6))
            trials += [{"seed": seed, **t} for t in tr]
        else:
            best = {}
        det = cls(seed=seed, **kw)
        det.params = best
        t0 = time.perf_counter(); det.fit(Xtr, ytr if sup else None); fit_s = time.perf_counter() - t0
        s_bv = det.score(Xall[sp["benign_val"]])
        Xte = Xall[sp["test"]]; yte = y[sp["test"]]
        t0 = time.perf_counter(); s_te = det.score(Xte); score_s = time.perf_counter() - t0
        # single-record latency on a small sample
        lat = []
        for i in range(min(50, len(Xte))):
            t0 = time.perf_counter(); det.score(Xte[i:i + 1]); lat.append(time.perf_counter() - t0)
        base = {"seed": seed, "model": mname, "split_hash": sh, "params": best, "supervised": sup}
        tf = threshold_free(yte, s_te)
        tf_rows.append({**base, **tf, "fit_seconds": fit_s, "batch_score_us_per_record": 1e6 * score_s / len(Xte),
                        "single_record_latency_us_median": 1e6 * float(np.median(lat)),
                        "single_record_latency_us_p95": 1e6 * float(np.percentile(lat, 95)),
                        "n_test": int(len(yte)), "n_test_attack": int(yte.sum())})
        for f in cfg.get("target_fprs", TARGET_FPRS):
            thr = calibrate_threshold(s_bv, f)
            m = threshold_metrics(yte, s_te, thr)
            fpr_rows.append({**base, "target_fpr": f, **m, **tf, "implied_fpr": implied_fpr(m["precision"], m["recall"], float(yte.mean())),
                             "tie_fraction_at_threshold_test": tie_fraction_at(s_te, thr),
                             "latency_us_per_record": 1e6 * score_s / len(Xte)})
        if not sup:
            # LEGACY protocol (paper Eq. 4): threshold = (1-c) quantile of benign TRAIN scores, c = contamination
            s_tr = det.score(Xtr)
            for c in cfg.get("legacy_contamination", [0.15]):
                thr = calibrate_threshold(s_tr, c)
                m = threshold_metrics(yte, s_te, thr)
                legacy_rows.append({**base, "contamination": c, **m, **tf,
                                    "implied_fpr": implied_fpr(m["precision"], m["recall"], float(yte.mean())),
                                    "tie_fraction_at_threshold_test": tie_fraction_at(s_te, thr)})
        if cfg.get("save_curves", True):
            curve_store[mname] = curves(yte, s_te)
    pi = float(y[sp["test"]].mean()); n_te = len(sp["test"])
    for tname, pred1 in (("flag_all", True), ("flag_none", False)):
        prec = pi if pred1 else 0.0; rec = 1.0 if pred1 else 0.0
        trivial_rows.append({"seed": seed, "model": tname, "split_hash": sh, "precision": prec, "recall": rec,
                             "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "mcc": 0.0,
                             "measured_fpr": 1.0 if pred1 else 0.0, "alerts": n_te if pred1 else 0,
                             "alert_rate": 1.0 if pred1 else 0.0, "test_prevalence": pi})
    split_art = {k: v for k, v in sp.items()}
    return tf_rows, fpr_rows, trials, curve_store, split_art, legacy_rows, trivial_rows


def summarize_rows(rows, models, key_metrics, group_key=None, n_boot=10_000):
    out, comps = [], []
    groups = sorted({r[group_key] for r in rows}) if group_key else [None]
    for g in groups:
        sub = [r for r in rows if group_key is None or r[group_key] == g]
        for metric in key_metrics:
            per = defaultdict(dict)
            for r in sub:
                per[r["model"]][r["seed"]] = r[metric]
            for m in models:
                vals = [per[m][s] for s in sorted(per[m])]
                out.append({"metric": metric, group_key or "scope": g, "model": m,
                            **summarize(vals, n_boot), "per_seed": vals, "seeds": sorted(per[m])})
            for c in paired_comparisons({m: per[m] for m in models}, metric):
                c[group_key or "scope"] = g
                comps.append(c)
    return out, comps


def run_detection(cfg: Dict[str, Any], out_root="results", argv=None, run_dir=None) -> ExperimentRun:
    data, dsinfo, dd = prepare(cfg)
    seeds = list(cfg["seeds"])
    run = ExperimentRun(cfg.get("name", "detection"), cfg, out_root, argv, dsinfo, seeds, run_dir)
    run.manifest["dedup"] = dd
    run.manifest["detector_hyperparameter_grids"] = {m: REGISTRY[m].grid for m in cfg["models"]}
    curve_acc = {k: defaultdict(list) for k in ("roc", "recall_at_fpr", "precision_at_fpr", "f1_at_fpr", "pr")}
    grid = None
    jobs = int(cfg.get("jobs", 1)); t_start = time.perf_counter()
    def consume(s, res, k):
        nonlocal grid
        tf, fr, tr, cv, sp, lg, tv = res
        run.add_rows("per_seed_legacy_contamination", lg); run.add_rows("trivial_baselines", tv)
        run.add_rows("per_seed_threshold_free", tf); run.add_rows("per_seed_fpr_sweep", fr)
        run.add_rows("tuning_trials", tr)
        run.save_npz(f"split_seed{s}", **sp)  # train/val/test indices (into the deduplicated frame)
        flat = {f"{m}__{kk}": v for m, c in cv.items() for kk, v in c.items()}
        if flat:
            run.save_npz(f"curves_seed{s}", **flat)
        for m, c in cv.items():
            grid = c["fpr_grid"]
            for key in ("recall_at_fpr", "precision_at_fpr", "f1_at_fpr"):
                curve_acc[key][m].append(c[key])
        run._flush()
        el = time.perf_counter() - t_start
        print(f"[{run.experiment}] seed {s} done ({k}/{len(seeds)}), elapsed {el / 60:.1f} min, "
              f"ETA {el / k * (len(seeds) - k) / 60:.1f} min", flush=True)
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [(s, ex.submit(run_seed, data, s, cfg)) for s in seeds]
            for k, (s, f) in enumerate(futs, 1):   # consumed in seed order => identical output to jobs=1
                consume(s, f.result(), k)
    else:
        for k, s in enumerate(seeds, 1):
            consume(s, run_seed(data, s, cfg), k)
    nb = cfg.get("n_boot", 10_000)
    models = cfg["models"]
    s1, c1 = summarize_rows(run.tables["per_seed_threshold_free"], models, ["roc_auc", "pr_auc"], None, nb)
    s2, c2 = summarize_rows(run.tables["per_seed_fpr_sweep"], models, THRESH_METRICS, "target_fpr", nb)
    run.save_table("summary_threshold_free", s1); run.save_table("comparisons_threshold_free", c1)
    run.save_table("summary_fpr_sweep", s2); run.save_table("comparisons_fpr_sweep", c2)
    if run.tables.get("per_seed_legacy_contamination"):
        sl, cl = summarize_rows(run.tables["per_seed_legacy_contamination"], [m for m in models if not REGISTRY[m].supervised],
                                ["precision", "recall", "f1", "mcc", "measured_fpr", "implied_fpr", "alerts"], "contamination", nb)
        run.save_table("summary_legacy_contamination", sl); run.save_table("comparisons_legacy_contamination", cl)
        # paired: legacy(c) vs calibrated(5%) on the same seeds (RQ1 operating-point decomposition)
        cal5 = {(r["model"], r["seed"]): r for r in run.tables["per_seed_fpr_sweep"] if r["target_fpr"] == 0.05}
        rows = []
        for c in cfg.get("legacy_contamination", [0.15]):
            for m in [m for m in models if not REGISTRY[m].supervised]:
                for met in ("recall", "f1", "mcc", "precision", "roc_auc", "pr_auc"):
                    per = {"legacy": {r["seed"]: r[met] for r in run.tables["per_seed_legacy_contamination"] if r["model"] == m and r["contamination"] == c},
                           "calibrated_5pct": {s: cal5[(m, s)][met] for (mm, s) in cal5 if mm == m}}
                    cc = paired_comparisons(per, met, [("legacy", "calibrated_5pct")])[0]
                    rows.append({"model": m, "contamination": c, **cc})
        run.save_table("legacy_vs_calibrated_paired", rows)
    tr_rows = []
    for t in ("flag_all", "flag_none"):
        for met in ("precision", "recall", "f1", "mcc", "measured_fpr"):
            tr_rows.append({"model": t, "metric": met, **summarize([r[met] for r in run.tables["trivial_baselines"] if r["model"] == t], nb)})
    run.save_table("summary_trivial_baselines", tr_rows)
    lat = [{"model": m, **summarize([r["single_record_latency_us_median"] for r in run.tables["per_seed_threshold_free"] if r["model"] == m], nb)} for m in models]
    run.save_table("summary_latency_us", lat)
    if grid is not None:
        for k, yl in (("recall_at_fpr", "recall"), ("precision_at_fpr", "precision"), ("f1_at_fpr", "F1")):
            curve_band(run.artifact_path(f"{k}.png"), grid, curve_acc[k], "FPR", yl, f"{yl} vs FPR (mean, 5-95% over seeds)", logx=False)
    return_run = run
    run.finish()
    return return_run
