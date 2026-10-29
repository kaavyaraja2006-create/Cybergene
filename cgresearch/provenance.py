"""Environment / dataset / config provenance capture (no network access needed)."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

TRACKED_PACKAGES = ["numpy", "scipy", "pandas", "scikit-learn", "lightgbm", "xgboost",
                    "hnswlib", "faiss-cpu", "matplotlib", "PyYAML", "psutil", "pytest"]


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()


def array_hash(arr) -> str:
    import numpy as np
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.tobytes()).hexdigest()


def git_info(repo: Optional[Path] = None) -> Dict[str, Any]:
    here = Path(__file__).resolve()
    # flat layout: <repo>/cgresearch/provenance.py -> parents[1] is the repo root. (The v1 code used
    # parents[2], which pointed at the repo's PARENT directory and recorded the wrong commit/dirty flag.)
    repo = Path(repo) if repo else next((p for p in (here.parents[1], here.parents[2]) if (p / ".git").exists()), here.parents[1])
    def run(*args):
        try:
            return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL,
                                           text=True, timeout=10).strip()
        except Exception:
            return None
    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if commit else None,
            "note": None if commit else "not a git repository: commit hash unavailable"}


def cpu_info() -> Dict[str, Any]:
    model = platform.processor() or None
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    mem = None
    try:
        import psutil
        mem = psutil.virtual_memory().total
    except Exception:
        pass
    return {"model": model, "logical_cores": os.cpu_count(), "total_memory_bytes": mem}


def gpu_info() -> Dict[str, Any]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True, timeout=10)
        return {"gpus": [l.strip() for l in out.splitlines() if l.strip()]}
    except Exception:
        return {"gpus": []}


def package_versions() -> Dict[str, Optional[str]]:
    out = {}
    for p in TRACKED_PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def environment() -> Dict[str, Any]:
    return {
        "python": sys.version, "python_executable": sys.executable,
        "os": platform.platform(), "machine": platform.machine(),
        "cpu": cpu_info(), "gpu": gpu_info(), "packages": package_versions(),
        "env_threads": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }


def dataset_info(path: Optional[str | Path], version: Optional[str] = None,
                 synthetic: bool = False) -> Dict[str, Any]:
    if synthetic or path is None:
        return {"path": None, "version": version or "synthetic", "sha256": None, "synthetic": True}
    p = Path(path)
    files = sorted(p.glob("*.csv")) if p.is_dir() else [p]
    per_file = {str(f): sha256_file(f) for f in files}
    combined = hashlib.sha256(canonical_json(per_file).encode()).hexdigest()
    return {"path": str(p), "version": version, "sha256": combined, "files": per_file, "synthetic": False}
