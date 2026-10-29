"""CLI: python -m cgresearch.cli <command> --config configs/<file>.yaml [--set a.b=1] [--out results] [--smoke]

Commands (v1): detection | variants | if-diagnostics | dedup | audit | mutation | retrieval | retrieval-real | ann | robustness | report | index
Commands (v2, calibration under shift): calib-transfer | shift-monitor | contamination | protocol-sens | sentinel-case | recal-study | monitor-study
`--smoke` forces tiny SYNTHETIC settings (never use its output as a result)."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml


def parse_seeds(v):
    if isinstance(v, list):
        return [int(x) for x in v]
    if isinstance(v, int):
        return list(range(v))
    a, b = str(v).split("-")
    return list(range(int(a), int(b) + 1))


def set_dotted(cfg, key, value):
    cur = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = yaml.safe_load(value)


SMOKE = {
    "common": {"seeds": [0, 1], "n_boot": 200, "data": {"kind": "synthetic", "synthetic_args": {"n": 2500, "seed": 1, "dup_frac": 0.03, "near_dup_frac": 0.02}}},
    "detection": {"models": ["isolation_forest", "ocsvm", "lof", "autoencoder", "rule_based", "logreg", "random_forest", "lightgbm"], "max_configs": 2},
    "retrieval": {"n_store": 800, "n_queries": 40},
    "retrieval-real": {"n_store": 600, "n_queries": 40, "min_category_support": 20},
    "ann": {"sizes": [1000, 3000], "n_queries": 30},
    "mutation": {"episodes_per_reason": 5},
    "calib-transfer": {"models": ["isolation_forest", "pca", "hbos", "ecod", "logreg"], "tuning": {"grid": "random", "n": 3},
                       "scenarios": [{"kind": "random"}, {"kind": "lofo", "min_support": 30}, {"kind": "temporal", "fit_blocks": [0], "label_blocks": [1]}]},
    "tail-diag": {"models": ["isolation_forest", "pca", "hbos"], "tuning": {"grid": "random", "n": 3}},
    "shift-monitor": {"models": ["isolation_forest", "pca", "hbos"], "tuning": {"grid": "random", "n": 3}},
    "contamination": {"models": ["isolation_forest", "pca", "hbos"], "tuning": {"grid": "random", "n": 3}},
    "protocol-sens": {"models": ["isolation_forest", "pca", "hbos", "ecod"], "tuning": {"grid": "random", "n": 3}},
    "recal-study": {"reps": 6, "n_cal": 1000, "n_win": 1500, "eps": [0.0, 0.05, 0.2]},
    "monitor-study": {"reps": 3, "n_tune": 10, "stream_len": 2500, "t0": 800, "n_cal": 1000, "eps": [0.0, 0.1], "cert_reps": 3, "bases": ["gamma"]},
    "sentinel-case": {"n_sessions": 600, "phases": 4, "attacks_per_phase": 30},
    "robustness": {"poisoning": {"n_store": 600, "n_queries": 30, "poison_rates": [0.0, 0.1]},
                   "adversarial": {"query_budget": 5}},
}


def deep_update(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            deep_update(a[k], v)
        else:
            a[k] = v
    return a


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cgresearch")
    ap.add_argument("command", choices=["detection", "if-diagnostics", "dedup", "audit", "mutation", "retrieval", "retrieval-real", "variants", "ann", "robustness", "report", "index",
                                            "calib-transfer", "shift-monitor", "contamination", "protocol-sens", "sentinel-case", "recal-study", "monitor-study", "tail-diag"])
    ap.add_argument("--config", "-c")
    ap.add_argument("--out", default="results")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL")
    ap.add_argument("--seeds", help="e.g. 0-29 or 30 (=0..29)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, help="parallel worker processes over seeds (detection/variants/dedup); results are identical to --jobs 1")
    ap.add_argument("--paper", default="paper", help="output dir for `report` (tables/ and figures/)")
    ap.add_argument("--include-smoke", action="store_true", help="report: also render SMOKE/synthetic runs (captioned as not results)")
    a = ap.parse_args(argv)
    raw_argv = list(sys.argv if argv is None else argv)
    if a.command == "report":
        from .report import build
        idx = build(a.out, a.paper, a.include_smoke)
        print(f"wrote {len(idx)} artifacts under {a.paper}/ (INDEX.csv lists sources)")
        return 0
    if a.command == "index":
        rows = []
        for m in sorted(Path(a.out).glob("*/*/manifest.json")):
            d = json.load(open(m))
            rows.append({k: d.get(k) for k in ("experiment_id", "experiment", "timestamp_utc", "status", "config_sha256", "runtime_seconds")} | {"dir": str(m.parent), "git_commit": d["git"]["commit"], "synthetic": d.get("dataset", {}).get("synthetic")})
        json.dump(rows, open(Path(a.out) / "INDEX.json", "w"), indent=2)
        print(f"indexed {len(rows)} runs -> {Path(a.out) / 'INDEX.json'}")
        return 0
    cfg = yaml.safe_load(open(a.config)) if a.config else {}
    if a.smoke:
        deep_update(cfg, copy.deepcopy(SMOKE["common"]))
        if a.command in ("calib-transfer", "shift-monitor", "contamination", "protocol-sens", "tail-diag"):
            cfg["data"] = {"kind": "synthetic", "generator": "shift", "shift_args": {"n_days": 5, "per_day": 1500, "seed": 1}}
        key = {"if-diagnostics": "detection", "dedup": "detection", "audit": "detection", "variants": "detection"}.get(a.command, a.command)
        deep_update(cfg, copy.deepcopy(SMOKE.get(key, {})))
        cfg.setdefault("data", {})
    for kv in a.set:
        k, v = kv.split("=", 1)
        set_dotted(cfg, k, v)
    if a.smoke and not str(cfg.get("name", a.command)).startswith("SMOKE_"):
        cfg["name"] = "SMOKE_" + str(cfg.get("name", a.command))   # applied AFTER --set so it can never be lost
    if a.jobs:
        cfg["jobs"] = a.jobs
    if a.seeds:
        cfg["seeds"] = parse_seeds(a.seeds)
    elif "seeds" in cfg:
        cfg["seeds"] = parse_seeds(cfg["seeds"])
    for sub in ("drift", "poisoning", "adversarial"):  # propagate seeds into robustness sub-configs
        if sub in cfg and isinstance(cfg[sub], dict):
            cfg[sub].setdefault("seeds", cfg["seeds"])
    if a.command == "retrieval":
        from .experiments.retrieval import run_retrieval as fn
    elif a.command == "retrieval-real":
        from .experiments.retrieval_real import run_retrieval_real as fn
    elif a.command == "variants":
        from .experiments.variants import run_variants as fn
    elif a.command == "ann":
        from .experiments.ann_scaling import run_ann as fn
    elif a.command == "mutation":
        from .experiments.mutation import run_mutation as fn
    elif a.command == "robustness":
        from .experiments.robustness import run_robustness as fn
    elif a.command == "calib-transfer":
        from .experiments.calib_transfer import run_calib_transfer as fn
    elif a.command == "shift-monitor":
        from .experiments.shift_monitor import run_shift_monitor as fn
    elif a.command == "contamination":
        from .experiments.contamination import run_contamination as fn
    elif a.command == "protocol-sens":
        from .experiments.protocol_sens import run_protocol_sens as fn
    elif a.command == "tail-diag":
        from .experiments.tail_diag import run_tail_diag as fn
    elif a.command == "recal-study":
        from .experiments.recal_study import run_recal_study as fn
    elif a.command == "monitor-study":
        from .experiments.monitor_study import run_monitor_study as fn
    elif a.command == "sentinel-case":
        from .experiments.sentinel_case import run_sentinel_case as fn
    elif a.command == "detection":
        from .experiments.detection import run_detection as fn
    elif a.command == "if-diagnostics":
        from .experiments.if_diag import run_if_diagnostics as fn
    elif a.command == "dedup":
        from .experiments.dedup import run_dedup_comparison as fn
    else:  # audit
        from .experiments.common import load_data
        from .data.audit import leakage_report
        from .runlog import ExperimentRun
        data = load_data(cfg)
        run = ExperimentRun(cfg.get("name", "audit"), cfg, a.out, raw_argv, dict(data.info), cfg.get("seeds", []))
        run.save_json("leakage_report", leakage_report(data, None, **cfg.get("near_dup", {})))
        run.finish()
        print(run.dir)
        return 0
    run = fn(cfg, a.out, raw_argv)
    print(run.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
