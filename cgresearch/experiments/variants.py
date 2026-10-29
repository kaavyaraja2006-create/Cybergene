"""Preprocessing-variant ablation (paper RQ2: does feature standardisation change conclusions, and
does the answer depend on the metric?). Also used for port treatments on Sample A.

cfg['variants'] = [{name: 'standardized', standardize: true}, {name: 'unstandardized', standardize: false}, ...]
Each variant is a full paired detection run (same seeds => same splits). Output: per model/metric
variant summaries plus paired Wilcoxon(+Holm)/Cliff's delta of every variant against the first one."""
from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Dict

from ..runlog import ExperimentRun
from ..stats import paired_comparisons, summarize
from .common import load_data
from .detection import run_detection

METRICS_AT_5 = ("roc_auc", "pr_auc", "recall", "precision", "f1", "mcc", "measured_fpr")


def run_variants(cfg: Dict[str, Any], out_root="results", argv=None) -> ExperimentRun:
    raw = load_data(cfg)
    run = ExperimentRun(cfg.get("name", "variants"), cfg, out_root, argv, dict(raw.info), cfg["seeds"])
    subs = {}
    for v in cfg["variants"]:
        c = copy.deepcopy(cfg); c.pop("variants")
        c.update({k: val for k, val in v.items() if k != "name"})
        c["name"] = f"{cfg.get('name', 'variants')}__{v['name']}"
        subs[v["name"]] = run_detection(c, str(run.dir / "sub_runs"), argv, run_dir=run.dir / "sub_runs" / v["name"])
    names = [v["name"] for v in cfg["variants"]]
    per = defaultdict(lambda: defaultdict(dict))
    for vn, r in subs.items():
        for row in r.tables["per_seed_fpr_sweep"]:
            if row["target_fpr"] == cfg.get("operating_fpr", 0.05):
                for m in METRICS_AT_5:
                    per[(row["model"], m)][vn][row["seed"]] = row[m]
    summ, comps = [], []
    for (model, m), byv in sorted(per.items()):
        for vn in names:
            vals = [byv[vn][s] for s in sorted(byv[vn])]
            summ.append({"model": model, "metric": m, "variant": vn, **summarize(vals, cfg.get("n_boot", 10_000)), "per_seed": vals})
        for vn in names[1:]:
            c = paired_comparisons({names[0]: byv[names[0]], vn: byv[vn]}, m, [(vn, names[0])])[0]
            c["model"] = model; comps.append(c)
    run.save_table("variant_summary", summ)
    run.save_table("variant_comparisons_vs_first", comps)
    run.manifest["sub_runs"] = {k: str(r.dir) for k, r in subs.items()}
    run.finish()
    return run
