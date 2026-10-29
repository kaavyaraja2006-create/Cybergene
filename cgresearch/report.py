"""Result artifacts -> LaTeX tables + PDF/PNG figures with auto captions/labels.

Every number comes from the run's tables/*.json (never typed in). Each output file starts with a
comment naming the experiment_id, config hash and source tables; `paper/tables/INDEX.csv` lists all
outputs. SMOKE_/synthetic runs are skipped unless include_smoke=True, and are captioned as NOT RESULTS.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "savefig.bbox": "tight", "axes.prop_cycle": matplotlib.cycler(color=OKABE)})


def esc(s: Any) -> str:
    return str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


def num(x, d=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{d}f}"


def pm(r: Dict[str, Any], d=3, ci=True):
    s = f"{num(r['mean'], d)} $\\pm$ {num(r['std'], d)}"
    return s + (f" [{num(r['ci_low'], d)}, {num(r['ci_high'], d)}]" if ci else "")


def pval(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


class Run:
    def __init__(self, d: Path):
        self.dir = d
        self.m = json.load(open(d / "manifest.json", encoding="utf-8"))
        self.name = self.m["experiment"]
        self.id = self.m["experiment_id"]
        self.synthetic = bool(self.m.get("dataset", {}).get("synthetic", False))
        self.cfg = json.load(open(d / "config.json", encoding="utf-8"))

    def t(self, name):
        p = self.dir / "tables" / f"{name}.json"
        return json.load(open(p, encoding="utf-8")) if p.exists() else None

    def has(self, name):
        return (self.dir / "tables" / f"{name}.json").exists()

    @property
    def tag(self):
        return self.name.replace("SMOKE_", "smoke_")


class Writer:
    def __init__(self, paper: Path):
        self.tdir, self.fdir = paper / "tables", paper / "figures"
        self.tdir.mkdir(parents=True, exist_ok=True); self.fdir.mkdir(parents=True, exist_ok=True)
        self.index: List[Dict[str, str]] = []

    def caption(self, run: Run, text: str, n: Optional[int] = None, ci: bool = False):
        nb = run.cfg.get("n_boot", 10000)
        extra = (f" Mean $\\pm$ SD over {n} paired seeds" + (f"; brackets are 95\\% percentile-bootstrap CIs ({nb} resamples)" if ci else "") + ".") if n else ""
        warn = " \\textbf{SYNTHETIC SMOKE DATA --- NOT A RESULT.}" if run.synthetic else ""
        return f"{text}.{extra}{warn}"

    def table(self, run: Run, kind: str, header: List[str], rows: List[List[str]], caption: str, sources: List[str],
              colspec: Optional[str] = None, n: Optional[int] = None, ci: bool = False):
        label = f"tab:{run.tag}_{kind}"
        colspec = colspec or "l" + "r" * (len(header) - 1)
        lines = [f"% experiment_id: {run.id}", f"% config_sha256: {run.m['config_sha256']}", f"% sources: {', '.join(sources)}",
                 "\\begin{table}[t]", "\\centering", "\\small", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
                 " & ".join(header) + " \\\\", "\\midrule"]
        lines += [" & ".join(r) + " \\\\" for r in rows]
        lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{self.caption(run, caption, n, ci)}}}", f"\\label{{{label}}}", "\\end{table}"]
        f = self.tdir / f"{run.tag}_{kind}.tex"
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.index.append({"file": str(f), "type": "table", "label": label, "experiment_id": run.id,
                           "config_sha256": run.m["config_sha256"], "sources": ";".join(sources)})

    def figure(self, run: Run, kind: str, fig, caption: str, sources: List[str]):
        label = f"fig:{run.tag}_{kind}"
        for ext in ("pdf", "png"):
            fig.savefig(self.fdir / f"{run.tag}_{kind}.{ext}", dpi=300)
        plt.close(fig)
        cap = self.caption(run, caption)
        (self.fdir / f"{run.tag}_{kind}.tex").write_text(
            encoding="utf-8", data=f"% experiment_id: {run.id}\n% sources: {', '.join(sources)}\n\\begin{{figure}}[t]\n\\centering\n"
            f"\\includegraphics[width=\\linewidth]{{figures/{run.tag}_{kind}.pdf}}\n\\caption{{{cap}}}\n\\label{{{label}}}\n\\end{{figure}}\n")
        self.index.append({"file": str(self.fdir / f"{run.tag}_{kind}.pdf"), "type": "figure", "label": label,
                           "experiment_id": run.id, "config_sha256": run.m["config_sha256"], "sources": ";".join(sources)})


# ------------------------------------------------------------------ detection
def report_detection(run: Run, w: Writer):
    models = run.cfg["models"]
    tf = run.t("summary_threshold_free"); n = tf[0]["n"] if tf else None
    rows = []
    for mname in models:
        r = {x["metric"]: x for x in tf if x["model"] == mname}
        lat = next((x for x in run.t("summary_latency_us") if x["model"] == mname), None)
        rows.append([esc(mname), pm(r["roc_auc"]), pm(r["pr_auc"]), num(lat["mean"], 1) if lat else "--"])
    w.table(run, "detection", ["Model", "ROC-AUC", "PR-AUC", "Latency ($\\mu$s/record)"], rows,
            "Threshold-free detection performance on the held-out test split", ["summary_threshold_free", "summary_latency_us"], n=n, ci=True)
    fs = run.t("summary_fpr_sweep"); imp = {}
    for f in run.cfg.get("target_fprs", [0.01, 0.02, 0.05, 0.10]):
        rows = []
        for mname in models:
            g = {x["metric"]: x for x in fs if x["model"] == mname and abs(x["target_fpr"] - f) < 1e-12}
            rows.append([esc(mname)] + [pm(g[k], 3, ci=False) for k in ("measured_fpr", "precision", "recall", "f1", "mcc")] + [num(g["alerts"]["mean"], 0)])
        w.table(run, f"fpr_{int(round(f * 100)):02d}", ["Model", "Measured FPR", "Precision", "Recall", "F1", "MCC", "Alerts"], rows,
                f"Operating point calibrated on benign validation data for target FPR {f * 100:g}\\%", ["summary_fpr_sweep"], n=n)
    # baseline tests at 5% (baselines vs unsupervised detectors)
    comps = run.t("comparisons_fpr_sweep") or []
    unsup = [m for m in models if m in ("isolation_forest", "ocsvm", "lof")]
    base = [m for m in models if m not in unsup]
    rows = []
    for c in comps:
        if abs(c["target_fpr"] - 0.05) > 1e-12 or c["metric"] not in ("recall", "f1", "mcc"):
            continue
        a, b = c["model_a"], c["model_b"]
        if (a in base and b in unsup) or (b in base and a in unsup):
            if a in unsup:  # orient as baseline minus unsupervised
                a, b = b, a; sign = -1
            else:
                sign = 1
            rows.append([esc(c["metric"]), esc(a), esc(b), num(sign * c["mean_diff_a_minus_b"]), pval(c["p_value"]), pval(c.get("p_holm")), num(sign * c["cliffs_delta"], 2), esc(c["cliffs_magnitude"])])
    if rows:
        w.table(run, "baseline_tests_fpr05", ["Metric", "Baseline", "Unsupervised", "Mean diff", "p", "p (Holm)", "Cliff's $\\delta$", "Magnitude"], rows,
                "Paired Wilcoxon signed-rank tests at the 5\\% FPR operating point (baseline minus unsupervised; Holm correction within metric and FPR)", ["comparisons_fpr_sweep"], "lllrrrrl")
    if run.has("summary_legacy_contamination"):
        sl = run.t("summary_legacy_contamination"); rows = []
        for mname in [m for m in models if m in unsup]:
            for c in sorted({x["contamination"] for x in sl}):
                g = {x["metric"]: x for x in sl if x["model"] == mname and x["contamination"] == c}
                rows.append([esc(mname), num(c, 2)] + [pm(g[k], 3, ci=False) for k in ("recall", "f1", "measured_fpr", "implied_fpr")])
        w.table(run, "legacy_contamination", ["Model", "$c$", "Recall", "F1", "Measured FPR", "Implied FPR"], rows,
                "Legacy fixed-contamination threshold (quantile of benign training scores); measured and implied FPR", ["summary_legacy_contamination"], n=n)
        lp = [x for x in run.t("legacy_vs_calibrated_paired") if x["metric"] in ("recall", "f1", "mcc")]
        w.table(run, "legacy_vs_calibrated", ["Model", "Metric", "Legacy $-$ calibrated", "p (Holm)", "Cliff's $\\delta$"],
                [[esc(x["model"]), esc(x["metric"]), num(x["mean_diff_a_minus_b"]), pval(x["p_holm"]), num(x["cliffs_delta"], 2)] for x in lp],
                "Paired comparison of the legacy contamination threshold with the 5\\% validation-calibrated threshold", ["legacy_vs_calibrated_paired"], "llrrr")
    if run.has("summary_trivial_baselines"):
        st = run.t("summary_trivial_baselines")
        w.table(run, "trivial", ["Rule", "Precision", "Recall", "F1", "MCC"],
                [[esc(m)] + [num(next(x for x in st if x["model"] == m and x["metric"] == k)["mean"]) for k in ("precision", "recall", "f1", "mcc")] for m in ("flag_all", "flag_none")],
                "Trivial reference rules (flag every flow / flag none); precision of flag-all equals test prevalence", ["summary_trivial_baselines"])
    # --- curves
    seeds = run.m["seeds"]; npz = {s: np.load(run.dir / "artifacts" / f"curves_seed{s}.npz") for s in seeds if (run.dir / "artifacts" / f"curves_seed{s}.npz").exists()}
    if not npz:
        return
    g = next(iter(npz.values()))[f"{models[0]}__fpr_grid"]
    def stack(key):
        return {m: np.array([npz[s][f"{m}__{key}"] for s in npz]) for m in models}
    def band(ax, grid, arr, label, color):
        ax.plot(grid, arr.mean(0), label=label, color=color, lw=1.4)
        ax.fill_between(grid, np.percentile(arr, 2.5, 0), np.percentile(arr, 97.5, 0), color=color, alpha=0.12, lw=0)
    for key, yl in (("recall_at_fpr", "Recall (TPR)"), ("precision_at_fpr", "Precision"), ("f1_at_fpr", "F1")):
        fig, ax = plt.subplots(figsize=(5.2, 3.6)); st = stack(key)
        for i, m in enumerate(models):
            band(ax, g, st[m], m, OKABE[i % 8])
        for f in run.cfg.get("target_fprs", []):
            ax.axvline(f, color="grey", lw=0.5, ls=":")
        ax.set_xscale("symlog", linthresh=0.01); ax.set_xlim(0, 1); ax.set_xlabel("False-positive rate"); ax.set_ylabel(yl); ax.legend(fontsize=7, frameon=False)
        w.figure(run, key.replace("_at_fpr", "_vs_fpr"), fig, f"{yl} versus FPR. Lines are means over seeds, bands the 2.5--97.5 percentile range; dotted lines mark the target FPRs", ["artifacts/curves_seed*.npz"])
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.6)); gp = np.linspace(0, 1, 501)
    for i, m in enumerate(models):
        tpr = np.array([np.interp(gp, npz[s][f"{m}__roc_fpr"], npz[s][f"{m}__roc_tpr"]) for s in npz])
        pr = np.array([np.interp(gp, npz[s][f"{m}__pr_recall"][::-1], npz[s][f"{m}__pr_precision"][::-1]) for s in npz])
        band(axs[0], gp, tpr, m, OKABE[i % 8]); band(axs[1], gp, pr, m, OKABE[i % 8])
    axs[0].plot([0, 1], [0, 1], color="grey", lw=0.6, ls="--"); axs[0].set_xlabel("FPR"); axs[0].set_ylabel("TPR"); axs[0].set_title("ROC")
    axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision"); axs[1].set_title("Precision--recall"); axs[0].legend(fontsize=6, frameon=False, loc="lower right")
    w.figure(run, "roc_pr", fig, "ROC and precision--recall curves (mean over seeds, 2.5--97.5 percentile band)", ["artifacts/curves_seed*.npz"])
    fig, axs = plt.subplots(2, len(models), figsize=(2.2 * len(models), 4.2), squeeze=False, sharey="row")
    for i, m in enumerate(models):
        for s in npz:
            axs[0, i].plot(npz[s][f"{m}__roc_fpr"], npz[s][f"{m}__roc_tpr"], lw=0.4, alpha=0.5, color=OKABE[i % 8])
            axs[1, i].plot(npz[s][f"{m}__pr_recall"], npz[s][f"{m}__pr_precision"], lw=0.4, alpha=0.5, color=OKABE[i % 8])
        axs[0, i].set_title(m, fontsize=7)
    axs[0, 0].set_ylabel("TPR"); axs[1, 0].set_ylabel("Precision")
    w.figure(run, "roc_pr_per_seed", fig, "Per-seed ROC (top) and precision--recall (bottom) curves for every detector", ["artifacts/curves_seed*.npz"])


# ------------------------------------------------------------------ IF diagnosis / dedup / variants
def report_if(run: Run, w: Writer):
    rows_all = run.t("if_diagnostics"); by = {}
    for r in rows_all:
        by.setdefault(r["port_mode"], []).append(r)
    mean = lambda k, rs: float(np.mean([x[k] for x in rs]))
    rows = [[esc(m), num(mean("auc_score_samples", rs)), num(mean("auc_neg_score_samples", rs)), num(mean("roc_auc", rs)), num(mean("pr_auc", rs)),
             num(mean("cal5_recall", rs)), num(mean("cal5_measured_fpr", rs)), num(mean("cal5_mcc", rs)), num(mean("predict_recall", rs)), num(mean("predict_mcc", rs)),
             num(mean("cal5_tie_fraction", rs), 4)] for m, rs in by.items()]
    w.table(run, "diagnosis", ["Ports", "AUC raw", "AUC neg.", "ROC-AUC", "PR-AUC", "Rec@5", "FPR@5", "MCC@5", "Rec(pred)", "MCC(pred)", "Ties"], rows,
            "Isolation Forest diagnosis (means over seeds). AUC raw/neg.\\ = AUC of sklearn \\texttt{score\\_samples} and its negation; Rec@5 = recall at the 5\\% validation-calibrated threshold; Rec(pred) = recall of sklearn \\texttt{predict}; Ties = test-score mass exactly at the threshold",
            ["if_diagnostics"], n=len(run.m["seeds"]))
    v = run.t("verdict")
    if v:
        w.table(run, "verdict", ["Classification", "Evidence flags"], [[esc(", ".join(v["classification"])), esc(json.dumps(v["evidence_flags"]))]],
                "Rule-based classification of the Isolation Forest behaviour from recorded evidence (heuristic; not a proof of cause)", ["verdict"], "ll")
    for variant in by:
        f = next(iter((run.dir / "artifacts").glob(f"scores_{variant}_seed*.npz")), None)
        if f is None:
            continue
        z = np.load(f); s, y = z["scores"], z["y"]
        fig, ax = plt.subplots(figsize=(5, 3.2)); lo, hi = np.percentile(s, [0.5, 99.5])
        ax.hist(np.clip(s[y == 0], lo, hi), bins=60, alpha=0.6, density=True, label="benign", color=OKABE[0])
        ax.hist(np.clip(s[y == 1], lo, hi), bins=60, alpha=0.6, density=True, label="attack", color=OKABE[1])
        ax.set_xlabel("Anomaly score (higher = more anomalous)"); ax.set_ylabel("Density"); ax.legend(frameon=False)
        w.figure(run, f"score_hist_{variant}", fig, f"Isolation Forest score distributions on the test split, ports = {esc(variant)} (first seed)", [f.name])


def report_dedup(run: Run, w: Writer):
    lr = json.load(open(run.dir / "tables" / "leakage_report.json", encoding="utf-8")); e, ne = lr["exact_duplicates"], lr["near_duplicates"]
    rows = [["Rows", str(lr["n_rows"]), "--"],
            ["Exact duplicates (redundant rows)", str(e["redundant_rows"]), f"{e['pct']:.2f}\\%"],
            ["Benign redundant copies", str(e["by_label"]["0"]["redundant_copies"]), "--"],
            ["Attack redundant copies", str(e["by_label"]["1"]["redundant_copies"]), "--"],
            ["Label-conflict groups", str(e["label_conflict_groups"]), "--"],
            ["Near-duplicate pairs", str(ne["pairs_found"]), "--"],
            ["Rows involved in near-duplicates", str(ne["rows_involved"]), f"{ne['pct_rows_involved']:.2f}\\%"]]
    for k, v in (lr.get("cross_split_exact") or {}).items():
        rows.append([f"Rows in {esc(k.split('__')[1])} with an exact copy in {esc(k.split('__')[0])}", str(v["rows_in_b_with_copy_in_a"]), f"{v['pct_of_b']:.2f}\\%"])
    w.table(run, "leakage", ["Quantity", "Count", "Share"], rows, f"Leakage and duplicate audit ({esc(ne['mode'])})", ["leakage_report"], "lrr")
    if run.has("original_vs_deduplicated"):
        rows = []
        for r in run.t("original_vs_deduplicated"):
            if "dedup_mode" in r and r["metric"] in ("roc_auc", "recall", "f1", "mcc"):
                rows.append([esc(r["model"]), esc(r["metric"]), esc(r["dedup_mode"]), pm(r, 3, ci=False), str(r["n_rows_after_dedup"])])
        w.table(run, "dedup_effect", ["Model", "Metric", "Dedup", "Mean $\\pm$ SD", "Rows"], rows,
                "Detection metrics at the 5\\% FPR operating point before and after deduplication", ["original_vs_deduplicated"], "lllrr")


def report_variants(run: Run, w: Writer):
    vs = run.t("variant_summary"); names = [v["name"] for v in run.cfg["variants"]]; rows = []
    for model in sorted({x["model"] for x in vs}):
        for met in ("roc_auc", "pr_auc", "recall", "f1", "mcc"):
            rows.append([esc(model), esc(met)] + [pm(next(x for x in vs if x["model"] == model and x["metric"] == met and x["variant"] == n), 3, ci=False) for n in names])
    w.table(run, "variants", ["Model", "Metric"] + [esc(n) for n in names], rows, "Preprocessing-variant ablation at the 5\\% FPR operating point",
            ["variant_summary"], "ll" + "r" * len(names), n=vs[0]["n"])
    cs = run.t("variant_comparisons_vs_first")
    w.table(run, "variant_tests", ["Model", "Metric", "Variant", "Mean diff", "p (Holm)", "Cliff's $\\delta$"],
            [[esc(c["model"]), esc(c["metric"]), esc(c["model_a"]), num(c["mean_diff_a_minus_b"]), pval(c["p_holm"]), num(c["cliffs_delta"], 2)] for c in cs],
            f"Paired tests of each variant against {esc(names[0])}", ["variant_comparisons_vs_first"], "lllrrr")


# ------------------------------------------------------------------ mutation / retrieval / ann / robustness
def report_mutation(run: Run, w: Writer):
    s = run.t("summary"); pols = run.cfg["policies"]
    for prior in sorted({x["direction_prior"] for x in s}):
        rows = []
        for p in pols:
            g = {x["metric"]: x for x in s if x["strategy"] == p and x["direction_prior"] == prior}
            rows.append([esc(p), pm(g["success_rate"], 3, False), pm(g["generations_censored"], 2, False), pm(g["failed_rate"], 3, False), pm(g["cost"], 2, False),
                         pm(g["false_containment_rate"], 3, False), num(g["fitness_final_with_birth_penalty"]["mean"]), num(g["fitness_final_without_birth_penalty"]["mean"])])
        w.table(run, f"policies_prior{str(prior).replace('.', '')}", ["Policy", "Success", "Generations$^\\dagger$", "Failed", "Cost", "False contain.", "Fitness (pen.)", "Fitness (no pen.)"], rows,
                f"Simulated failure-driven mutation, direction prior {prior}. $^\\dagger$Generations censored at cap+1 for failures. SIMULATION, not real containment", ["summary"], n=s[0]["n"])
    cs = run.t("comparisons_vs_retry")
    w.table(run, "tests_vs_retry", ["Prior", "Metric", "Policy", "Mean diff", "p (Holm)", "Cliff's $\\delta$"],
            [[str(c["direction_prior"]), esc(c["metric"]), esc(c["model_a"]), num(c["mean_diff_a_minus_b"]), pval(c["p_holm"]), num(c["cliffs_delta"], 2)] for c in cs if c["metric"] in ("success_rate", "generations_censored", "cost")],
            "Paired Wilcoxon tests of each policy against unmodified retry", ["comparisons_vs_retry"], "llrrrr")
    priors = sorted({x["direction_prior"] for x in s}); fig, axs = plt.subplots(1, len(priors), figsize=(3.2 * len(priors), 3.2), sharey=True, squeeze=False)
    for j, pr in enumerate(priors):
        for i, p in enumerate(pols):
            r = next(x for x in s if x["strategy"] == p and x["direction_prior"] == pr and x["metric"] == "success_rate")
            axs[0, j].bar(i, r["mean"], yerr=[[r["mean"] - r["ci_low"]], [r["ci_high"] - r["mean"]]], color=OKABE[i % 8], capsize=2)
        axs[0, j].set_xticks(range(len(pols))); axs[0, j].set_xticklabels(pols, rotation=60, ha="right", fontsize=7); axs[0, j].set_title(f"prior {pr}", fontsize=8)
    axs[0, 0].set_ylabel("Containment success rate")
    w.figure(run, "success", fig, "Simulated containment success by policy (bars: mean, whiskers: 95\\% bootstrap CI over seeds)", ["summary"])


def report_retrieval(run: Run, w: Writer):
    s = run.t("summary"); cfgs = list(dict.fromkeys(x["config"] for x in s)); mets = ["P@1", "P@5", "P@10", "R@5", "R@10", "MRR", "nDCG@10"]
    rows = [[esc(c)] + [pm(next(x for x in s if x["config"] == c and x["metric"] == m), 3, False) for m in mets] for c in cfgs]
    weak = " Ground truth: WEAK dataset attack-category labels; context = service only; no evolution genome." if run.name.startswith("retrieval_real") or "real" in run.name else " Ground truth derived from synthetic generator labels."
    w.table(run, "ablation", ["Genomes"] + [esc(m) for m in mets], rows, "Attack-DNA retrieval genome ablation." + weak, ["summary"], n=s[0]["n"])
    cs = run.t("comparisons_adjacent") or run.t("comparisons") or []
    rows = [[esc(c["metric"]), esc(c["model_a"]), esc(c["model_b"]), num(c["mean_diff_a_minus_b"]), pval(c["p_holm"]), num(c["cliffs_delta"], 2)] for c in cs if c["metric"] in ("P@5", "nDCG@10")]
    if rows:
        w.table(run, "ablation_tests", ["Metric", "Config A", "Config B", "A$-$B", "p (Holm)", "Cliff's $\\delta$"], rows, "Paired tests between genome configurations", ["comparisons"], "lllrrr")
    fig, ax = plt.subplots(figsize=(5.4, 3.4)); width = 0.8 / len(cfgs)
    for i, c in enumerate(cfgs):
        vals = [next(x for x in s if x["config"] == c and x["metric"] == m) for m in ("P@1", "P@5", "P@10", "MRR", "nDCG@10")]
        ax.bar(np.arange(5) + i * width, [v["mean"] for v in vals], width, yerr=[[v["mean"] - v["ci_low"] for v in vals], [v["ci_high"] - v["mean"] for v in vals]], label=c, color=OKABE[i % 8], capsize=1.5)
    ax.set_xticks(np.arange(5) + width * (len(cfgs) - 1) / 2); ax.set_xticklabels(["P@1", "P@5", "P@10", "MRR", "nDCG@10"]); ax.legend(fontsize=6, frameon=False); ax.set_ylim(0, 1.02)
    w.figure(run, "ablation", fig, "Retrieval quality by genome configuration (mean, 95\\% bootstrap CI)", ["summary"])


def report_ann(run: Run, w: Writer):
    s = run.t("summary_by_method_size"); sizes = sorted({x["n_records"] for x in s}); methods = list(dict.fromkeys(x["method"] for x in s))
    rows = [[esc(x["method"]), str(x["n_records"]), num(x["latency_ms_median_mean"], 3), num(x["latency_ms_p95_mean"], 3), num(x["latency_ms_p99_mean"], 3),
             num(x["qps_single_thread_mean"], 0), num(x["build_seconds_mean"], 2), num(x["recall_vs_exact_at_10_mean"], 3), num(x["overlap_at_10_mean"], 3)] for x in s]
    hw = run.m["environment"]["cpu"]
    w.table(run, "scaling", ["Method", "N", "Median ms", "p95 ms", "p99 ms", "QPS", "Build s", "Recall@10", "Overlap@10"], rows,
            f"Similarity-search scaling (means over seeds; single thread; {esc(hw.get('model'))}, {hw.get('logical_cores')} logical cores; recall/overlap versus exact search)", ["summary_by_method_size"], "lrrrrrrrr")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for i, m in enumerate(methods):
        pts = sorted((x["n_records"], x["latency_ms_median_mean"]) for x in s if x["method"] == m)
        ax.plot(*zip(*pts), marker="o", ms=3, label=m, color=OKABE[i % 8])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Stored records"); ax.set_ylabel("Median query latency (ms)"); ax.legend(fontsize=6, frameon=False)
    w.figure(run, "latency", fig, "Median query latency versus number of stored records", ["summary_by_method_size"])


def report_robustness(run: Run, w: Writer):
    if run.has("summary_drift"):
        d = run.t("summary_drift"); fig, ax = plt.subplots(figsize=(5.2, 3.4)); i = 0
        for model in sorted({x["model"] for x in d}):
            for pol, ls in (("static", "-"), ("retrain_previous_window", "--")):
                pts = sorted((x["window"], x["mean"]) for x in d if x["model"] == model and x["policy"] == pol and x["metric"] == "recall")
                ax.plot(*zip(*pts), ls=ls, marker="o", ms=3, label=f"{model} / {pol}", color=OKABE[i % 8]); i += 1 if pol != "static" else 0
        ax.set_xlabel("Time window"); ax.set_ylabel("Recall at 5\\% FPR (calibrated per policy)"); ax.legend(fontsize=5, frameon=False)
        w.figure(run, "drift", fig, "Concept drift: recall by window for a static model and for retraining on the previous window (SIMULATED drift unless real timestamps were used)", ["summary_drift"])
        w.table(run, "drift", ["Model", "Policy", "Window", "ROC-AUC", "Recall", "Meas. FPR"],
                [[esc(m), esc(p), str(wi)] + [pm(next(x for x in d if x["model"] == m and x["policy"] == p and x["window"] == wi and x["metric"] == k), 3, False) for k in ("roc_auc", "recall", "measured_fpr")]
                 for m in sorted({x["model"] for x in d}) for p in ("static", "retrain_previous_window") for wi in sorted({x["window"] for x in d})],
                "Concept-drift results", ["summary_drift"], "llrrrr", n=d[0]["n"])
    if run.has("summary_poisoning"):
        p = run.t("summary_poisoning"); rows = []
        for pt in sorted({x["poison_type"] for x in p}):
            for rate in sorted({x["poison_rate"] for x in p}):
                for mit in ("none", "knn_vetting", "duplicate_cap"):
                    g = {x["metric"]: x for x in p if x["poison_type"] == pt and x["poison_rate"] == rate and x["mitigation"] == mit}
                    rows.append([esc(pt), num(rate, 2), esc(mit), pm(g["retrieval_pollution_at_k"], 3, False), pm(g["wrong_response_rate"], 3, False)])
        w.table(run, "poisoning", ["Poison", "Rate", "Mitigation", "Pollution@k", "Wrong response"], rows, "Memory poisoning (SIMULATED corpus)", ["summary_poisoning"], "lrlrr", n=p[0]["n"])
    if run.has("summary_adversarial"):
        a = run.t("summary_adversarial"); fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for i, sc in enumerate(sorted({x["scorer"] for x in a})):
            pts = sorted((x["epsilon"], x["mean"]) for x in a if x["scorer"] == sc and x["metric"] == "recall_at_5pct_fpr")
            ax.plot(*zip(*pts), marker="o", ms=3, label=sc, color=OKABE[i % 8])
        ax.set_xlabel("Perturbation budget $\\epsilon$"); ax.set_ylabel("Recall at 5\\% FPR"); ax.legend(fontsize=6, frameon=False)
        w.figure(run, "adversarial", fig, "Black-box feature-perturbation evasion (SIMULATED attacker with a query budget)", ["summary_adversarial"])
        w.table(run, "adversarial", ["Scorer", "$\\epsilon$", "Recall@5\\%FPR", "Evasion rate"],
                [[esc(sc), num(e, 2), pm(next(x for x in a if x["scorer"] == sc and x["epsilon"] == e and x["metric"] == "recall_at_5pct_fpr"), 3, False),
                  pm(next(x for x in a if x["scorer"] == sc and x["epsilon"] == e and x["metric"] == "evasion_rate_among_detected"), 3, False)]
                 for sc in sorted({x["scorer"] for x in a}) for e in sorted({x["epsilon"] for x in a})],
                "Adversarial-telemetry robustness", ["summary_adversarial"], "lrrr", n=a[0]["n"])


DISPATCH = [("summary_fpr_sweep", report_detection), ("if_diagnostics", report_if), ("leakage_report", report_dedup),
            ("variant_summary", report_variants), ("generation_log", report_mutation), ("summary_by_method_size", report_ann),
            ("summary_drift", report_robustness), ("summary_poisoning", report_robustness), ("summary_adversarial", report_robustness)]


def build(results: str | Path, paper: str | Path, include_smoke: bool = False) -> List[Dict[str, str]]:
    runs = {}
    for m in sorted(Path(results).glob("*/*/manifest.json")):
        r = Run(m.parent)
        if r.m["status"] != "completed" or ((r.synthetic or r.name.startswith("SMOKE_")) and not include_smoke):
            continue
        key = r.name
        if key not in runs or r.m["timestamp_utc"] > runs[key].m["timestamp_utc"]:
            runs[key] = r
    w = Writer(Path(paper)); done = set()
    for r in runs.values():
        if r.has("summary") and r.has("comparisons_adjacent") or (r.has("summary") and r.has("comparisons") and r.has("per_seed")):
            report_retrieval(r, w); continue
        for marker, fn in DISPATCH:
            if (r.has(marker) or (marker == "leakage_report" and (r.dir / "tables/leakage_report.json").exists())) and (r.id, fn) not in done:
                if fn is report_dedup and r.has("per_seed_fpr_sweep"):
                    continue
                fn(r, w); done.add((r.id, fn))
    with open(Path(paper) / "tables" / "INDEX.csv", "w", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=["file", "type", "label", "experiment_id", "config_sha256", "sources"]); cw.writeheader(); cw.writerows(w.index)
    return w.index
