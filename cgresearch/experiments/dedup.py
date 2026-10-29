"""Leakage audit + original-vs-deduplicated comparison (experiment 3)."""
from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Dict

import numpy as np

from ..data.audit import leakage_report
from ..data.splits import make_split
from ..runlog import ExperimentRun
from ..stats import paired_comparisons, summarize
from .common import load_data
from .detection import run_detection


def run_dedup_comparison(cfg: Dict[str, Any], out_root="results", argv=None) -> ExperimentRun:
    raw = load_data(cfg)
    run = ExperimentRun(cfg.get("name", "dedup_comparison"), cfg, out_root, argv, dict(raw.info), cfg["seeds"])
    # audit on the ORIGINAL data with the seed-0 split to measure cross-split leakage
    sp = make_split(raw.y, cfg["seeds"][0])
    rep = leakage_report(raw, {k: sp[k] for k in ("train", "val", "test")}, **cfg.get("near_dup", {}))
    run.save_json("leakage_report", rep)
    sub_runs = {}
    for mode in cfg.get("modes", ["none", "exact", "near"]):
        c = copy.deepcopy(cfg); c["dedup"] = mode; c["name"] = f"{cfg.get('name', 'dedup')}_{mode}"
        sub_runs[mode] = run_detection(c, str(run.dir / "sub_runs"), argv, run_dir=run.dir / "sub_runs" / mode)
    rows = []
    paired = defaultdict(dict)
    for mode, r in sub_runs.items():
        for row in r.tables["per_seed_fpr_sweep"]:
            if row["target_fpr"] == 0.05:
                for met in ("roc_auc", "pr_auc", "recall", "f1", "mcc", "measured_fpr"):
                    paired[(row["model"], met)].setdefault(mode, {})[row["seed"]] = row[met]
    base = cfg.get("modes", ["none", "exact", "near"])[0]
    for (model, met), by_mode in sorted(paired.items()):
        for mode, per in by_mode.items():
            rows.append({"model": model, "metric": met, "dedup_mode": mode, **summarize(list(per.values()), cfg.get("n_boot", 10_000)),
                         "n_rows_after_dedup": sub_runs[mode].manifest["dedup"]["rows_after"]})
        for mode in by_mode:
            if mode != base:
                c = paired_comparisons({f"{base}": by_mode[base], f"{mode}": by_mode[mode]}, met)[0]
                c.update({"model": model, "note_pairing": "paired by seed; splits differ because data differ"})
                rows.append({"comparison": f"{base}_vs_{mode}", **c})
    run.save_table("original_vs_deduplicated", rows)
    run.manifest["sub_runs"] = {m: str(r.dir) for m, r in sub_runs.items()}
    run.finish()
    return run
