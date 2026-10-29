"""Isolation Forest diagnosis (experiment 2).

Questions answered with RAW evidence per seed and preprocessing variant:
  Q1 orientation: AUC of each sklearn score variant (score_samples, -score_samples, decision_function,
     -decision_function). Exactly one orientation family should give AUC>0.5; the pair sums to 1.
  Q2 threshold: does sklearn predict()==-1 (contamination threshold) disagree with a threshold
     calibrated on benign validation? measured FPR / recall / MCC under each rule.
  Q3 ties: fraction of test scores exactly at the calibrated threshold.
  Q4 distributions: class score histograms (data saved), mean/quantiles per class.
  Q5 top-scoring benign and attack records dumped to CSV for manual inspection.
  Q6 preprocessing variants: ports raw / dropped / one-hot / one-hot-scaled.
It reports measurements only; it does NOT assert a cause.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from ..data.splits import make_split
from ..metrics import calibrate_threshold, curves, recall_at_fpr, threshold_free, threshold_metrics, tie_fraction_at
from ..plotting import score_hist
from ..preprocess import FlowPreprocessor
from ..runlog import ExperimentRun
from .common import prepare


def run_if_diagnostics(cfg: Dict[str, Any], out_root="results", argv=None) -> ExperimentRun:
    data, dsinfo, dd = prepare(cfg)
    seeds = list(cfg["seeds"])
    run = ExperimentRun(cfg.get("name", "if_diagnostics"), cfg, out_root, argv, dsinfo, seeds)
    run.manifest["dedup"] = dd
    contamination = cfg.get("contamination", 0.05)
    for variant in cfg.get("port_modes", ["raw", "separate_std", "drop", "onehot", "onehot_scaled"]):
        for seed in seeds:
            sp = make_split(data.y, seed)
            pre = FlowPreprocessor(variant).fit(data, sp["benign_train"])
            X = pre.transform(data)
            m = IsolationForest(n_estimators=cfg.get("n_estimators", 200), contamination=contamination,
                                random_state=seed).fit(X[sp["benign_train"]])
            te, bv = sp["test"], sp["benign_val"]
            y = data.y[te]
            raw_ss, raw_df = m.score_samples(X[te]), m.decision_function(X[te])
            variants = {"score_samples": raw_ss, "neg_score_samples": -raw_ss,
                        "decision_function": raw_df, "neg_decision_function": -raw_df}
            aucs = {k: float(roc_auc_score(y, v)) for k, v in variants.items()}
            # contract: score = -score_samples (higher = more anomalous)
            s_te = -raw_ss
            s_bv = -m.score_samples(X[bv])
            row = {"port_mode": variant, "seed": seed, **{f"auc_{k}": v for k, v in aucs.items()},
                   "orientation_pair_sums_to_1": bool(abs(aucs["score_samples"] + aucs["neg_score_samples"] - 1) < 1e-9),
                   "decision_vs_score_same_order": bool(np.allclose(np.argsort(raw_ss, kind="stable"), np.argsort(raw_df, kind="stable"))),
                   "mean_score_benign": float(s_te[y == 0].mean()), "mean_score_attack": float(s_te[y == 1].mean()),
                   "attack_score_quantiles_pct_of_benign_val": [float((s_bv < np.quantile(s_te[y == 1], q)).mean()) for q in (0.1, 0.5, 0.9)],
                   **threshold_free(y, s_te)}
            # rule A: sklearn predict() (contamination-based threshold, learned on benign TRAIN)
            predA = (m.predict(X[te]) == -1).astype(int)
            row.update({"predict_alert_rate": float(predA.mean()),
                        "predict_fpr": float(predA[y == 0].mean()), "predict_recall": float(predA[y == 1].mean()),
                        "predict_mcc": float(matthews_corrcoef(y, predA))})
            for f in (0.05,):
                thr = calibrate_threshold(s_bv, f)
                tm = threshold_metrics(y, s_te, thr)
                row.update({f"cal{int(f*100)}_{k}": tm[k] for k in ("measured_fpr", "recall", "precision", "mcc", "alerts")})
                row[f"cal{int(f*100)}_tie_fraction"] = tie_fraction_at(s_te, thr)
            # inverted-orientation counterfactual (what a sign bug would produce)
            thr_inv = calibrate_threshold(-s_bv, 0.05)
            ti = threshold_metrics(y, -s_te, thr_inv)
            row.update({"INVERTED_recall_at5": ti["recall"], "INVERTED_mcc_at5": ti["mcc"]})
            run.add_row("if_diagnostics", row)
            if seed == seeds[0]:
                score_hist(run.artifact_path(f"score_hist_{variant}.png"), s_te[y == 0], s_te[y == 1],
                           f"IF scores, ports={variant}, seed={seed}")
                c = curves(y, s_te)
                run.save_npz(f"curves_{variant}_seed{seed}", **c)
                run.save_npz(f"scores_{variant}_seed{seed}", scores=s_te, y=y, test_idx=te)
                order = np.argsort(-s_te)
                for lab, nm in ((0, "benign"), (1, "attack")):
                    idx = [i for i in order if y[i] == lab][: cfg.get("top_k", 50)]
                    for rank, i in enumerate(idx):
                        run.add_row(f"top_scoring_{nm}_{variant}", {"rank": rank, "score": float(s_te[i]),
                                    "row_index": int(te[i]), "attack_cat": str(data.attack_cat[te[i]]),
                                    **data.X.iloc[te[i]].to_dict()})
    run.save_json("verdict", verdict(run.tables["if_diagnostics"]))
    run.finish()
    return run


def verdict(rows, port_delta=0.10, near_chance=0.02):
    """Rule-based classification of the IF anomaly from RECORDED evidence. Heuristic thresholds are
    arguments; the evidence per rule is returned so a human can overrule it.
      code_defect        : sklearn orientation pair does not sum to 1, or decision_function and
                           score_samples disagree in rank (implementation inconsistency)
      score_orientation  : contract-oriented AUC < 0.5 while the opposite orientation is > 0.5
      port_shortcut      : calibrated recall@5%FPR or ROC-AUC differs by >= port_delta between
                           ports kept as numbers (raw/separate_std) and ports dropped/one-hot
      threshold_protocol : sklearn predict() (contamination) recall/MCC differ materially from the
                           calibrated operating point (operating-point effect, not a model defect)
      genuine_model_behaviour : none of the above fired
    """
    import numpy as np
    by = {}
    for r in rows:
        by.setdefault(r["port_mode"], []).append(r)
    mean = lambda k, rs: float(np.mean([x[k] for x in rs]))
    ev = {}
    ev["code_defect"] = not all(r["orientation_pair_sums_to_1"] and r["decision_vs_score_same_order"] for r in rows)
    ev["score_orientation"] = any(r["roc_auc"] < 0.5 and r["auc_score_samples"] > 0.5 for r in rows)
    num = [m for m in ("raw", "separate_std") if m in by]; non = [m for m in ("drop", "onehot") if m in by]
    deltas = {}
    for a in num:
        for b in non:
            deltas[f"{a}_vs_{b}"] = {"d_recall_at_5pct": mean("cal5_recall", by[a]) - mean("cal5_recall", by[b]),
                                     "d_roc_auc": mean("roc_auc", by[a]) - mean("roc_auc", by[b])}
    ev["port_shortcut"] = any(abs(d["d_recall_at_5pct"]) >= port_delta or abs(d["d_roc_auc"]) >= port_delta for d in deltas.values())
    ev["threshold_protocol"] = any(abs(mean("predict_recall", rs) - mean("cal5_recall", rs)) >= port_delta for rs in by.values())
    order = ["code_defect", "score_orientation", "port_shortcut", "threshold_protocol"]
    hits = [k for k in order if ev[k]]
    return {"classification": hits or ["genuine_model_behaviour"], "evidence_flags": ev, "port_deltas": deltas,
            "per_variant_means": {m: {k: mean(k, rs) for k in ("roc_auc", "pr_auc", "cal5_recall", "cal5_mcc", "cal5_measured_fpr",
                                                                 "predict_recall", "predict_mcc", "cal5_tie_fraction")} for m, rs in by.items()},
            "thresholds": {"port_delta": port_delta},
            "note": "Heuristic, evidence-based label; multiple labels may apply. Not a proof of cause."}
