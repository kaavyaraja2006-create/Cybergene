"""ExperimentRun: one directory per run holding manifest + raw artifacts.

Layout: <out_root>/<experiment>/<experiment_id>/
    manifest.json   provenance + config + timing (written at start, finalised at end)
    config.json     resolved configuration
    tables/*.csv|json   raw per-seed / per-episode tables
    artifacts/*     curves (.npz), splits (.npz), plots
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from . import provenance as prov


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_jsonable)


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, default=_jsonable) if isinstance(v, (list, dict)) else v)
                        for k, v in r.items()})


class ExperimentRun:
    def __init__(self, experiment: str, config: Dict[str, Any], out_root: str | Path = "results",
                 argv: Optional[List[str]] = None, dataset: Optional[Dict[str, Any]] = None,
                 seeds: Optional[Iterable[int]] = None, run_dir: Optional[str | Path] = None):
        """run_dir: use this exact directory (used for nested sub-runs to keep Windows paths < 260 chars)."""
        self.experiment = experiment
        self.config = config
        self.config_hash = prov.config_hash(config)
        self.started = datetime.now(timezone.utc)
        self.id = f"{experiment}-{self.started.strftime('%Y%m%dT%H%M%SZ')}-{self.config_hash[:8]}"
        self.dir = Path(run_dir) if run_dir else Path(out_root) / experiment / self.id
        (self.dir / "tables").mkdir(parents=True, exist_ok=True)
        (self.dir / "artifacts").mkdir(parents=True, exist_ok=True)
        self._t0 = time.perf_counter()
        self.tables: Dict[str, List[Dict[str, Any]]] = {}
        self.manifest: Dict[str, Any] = {
            "experiment_id": self.id, "experiment": experiment,
            "timestamp_utc": self.started.isoformat(),
            "git": prov.git_info(),
            "environment": prov.environment(),
            "dataset": dataset or {},
            "seeds": sorted(int(s) for s in seeds) if seeds is not None else None,
            "cli_args": argv if argv is not None else sys.argv,
            "config_sha256": self.config_hash,
            "status": "running",
            "result_files": [],
        }
        write_json(self.dir / "config.json", config)
        self._flush()

    # --- recording ---------------------------------------------------------
    def add_row(self, table: str, row: Dict[str, Any]):
        self.tables.setdefault(table, []).append(row)

    def add_rows(self, table: str, rows: Iterable[Dict[str, Any]]):
        self.tables.setdefault(table, []).extend(rows)

    def save_table(self, table: str, rows: Optional[List[Dict[str, Any]]] = None):
        if rows is not None:
            self.tables[table] = rows
        rows = self.tables.get(table, [])
        write_csv(self.dir / "tables" / f"{table}.csv", rows)
        write_json(self.dir / "tables" / f"{table}.json", rows)
        self._record(f"tables/{table}.csv", f"tables/{table}.json")

    def save_json(self, name: str, obj: Any, sub: str = "tables"):
        write_json(self.dir / sub / f"{name}.json", obj)
        self._record(f"{sub}/{name}.json")

    def save_npz(self, name: str, **arrays):
        np.savez_compressed(self.dir / "artifacts" / f"{name}.npz", **arrays)
        self._record(f"artifacts/{name}.npz")

    def artifact_path(self, name: str) -> Path:
        p = self.dir / "artifacts" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        self._record(f"artifacts/{name}")
        return p

    def _record(self, *files):
        for f in files:
            if f not in self.manifest["result_files"]:
                self.manifest["result_files"].append(f)

    def _flush(self):
        write_json(self.dir / "manifest.json", self.manifest)

    def finish(self, status: str = "completed", extra: Optional[Dict[str, Any]] = None):
        for t in list(self.tables):
            if f"tables/{t}.csv" not in self.manifest["result_files"]:
                self.save_table(t)
        self.manifest["status"] = status
        self.manifest["runtime_seconds"] = round(time.perf_counter() - self._t0, 3)
        if extra:
            self.manifest.update(extra)
        self._flush()
        return self.dir
