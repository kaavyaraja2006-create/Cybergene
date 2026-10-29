from __future__ import annotations

from typing import Any, Dict

from ..data.synthetic import make_synthetic_flows
from ..data.unsw import FlowData, load_unsw, stratified_subset
from ..data.nf import load_nf
from ..data.cic import load_cic
from ..data.audit import deduplicate
from .. import provenance as prov


def load_data(cfg: Dict[str, Any]) -> FlowData:
    d = cfg["data"]
    if d.get("kind", "synthetic") == "synthetic":
        if d.get("generator") == "shift":
            from ..data.synthetic import make_shift_flows
            data = make_shift_flows(**d.get("shift_args", {}))
        else:
            data = make_synthetic_flows(**d.get("synthetic_args", {}))
    elif d["kind"] in ("nf_unsw_v2", "nf_cic2018_v2", "netflow"):
        data = load_nf(d["path"], d.get("version"), d.get("sample_n"), d.get("sample_seed", 0), d.get("max_rows"))
    elif d["kind"] in ("cic2017", "cic2018"):
        data = load_cic(d["path"], d.get("version"), d.get("sample_n"), d.get("sample_seed", 0), d.get("max_rows"), per_file_cap=d.get("per_file_cap"), include=d.get("include"))
    else:
        data = load_unsw(d["path"], d.get("version"), d.get("max_rows"), d.get("sample_n"),
                         d.get("sample_seed", 0), d.get("features_file"))
    return data


def _prepare_uncached(cfg: Dict[str, Any]):
    """Load, then deduplicate BEFORE any splitting. Returns (data, dataset_info, dedup_info).
    cfg['dedup_first']=true (v2 protocol): deduplicate the FULL file first and sample afterwards, so the reported
    duplicate rate describes the real dataset. false (v1 behaviour): sample, then deduplicate the subsample."""
    mode = cfg.get("dedup", "none")
    dcfg = cfg["data"]
    first = bool(cfg.get("dedup_first")) and dcfg.get("sample_n") and mode != "none"
    if first:
        n = int(dcfg["sample_n"]); seed = dcfg.get("sample_seed", 0)
        full_cfg = {**cfg, "data": {k: v for k, v in dcfg.items() if k != "sample_n"}}
        raw_full = load_data(full_cfg)
        deduped, kept = deduplicate(raw_full, mode, **cfg.get("near_dup", {}))
        data = stratified_subset(deduped, n, seed)
        raw = raw_full
        extra = {"dedup_first": True, "rows_full_before": len(raw_full), "rows_full_after": len(deduped),
                 "attack_full_before": int(raw_full.y.sum()), "attack_full_after": int(deduped.y.sum()), "sample_n": n}
    else:
        raw = load_data(cfg)
        data, kept = deduplicate(raw, mode, **cfg.get("near_dup", {}))
        extra = {"dedup_first": False}
    info = dict(raw.info)
    if not info.get("synthetic"):
        info = {**prov.dataset_info(dcfg["path"], dcfg.get("version")),
                **{k: v for k, v in raw.info.items() if k in ("sample", "alignment", "cleaning", "attack_labels", "n_rows")}}
    if first:      # dedup statistics describe the FULL file; sampling is reported separately
        dd = {"mode": mode, "rows_before": extra["rows_full_before"], "rows_after": extra["rows_full_after"],
              "removed": extra["rows_full_before"] - extra["rows_full_after"], "attack_before": extra["attack_full_before"],
              "attack_after": extra["attack_full_after"], "rows_sampled": len(data), "attack_sampled": int(data.y.sum()), **extra}
    else:
        dd = {"mode": mode, "rows_before": len(raw), "rows_after": len(data),
              "removed": len(raw) - len(data), "attack_before": int(raw.y.sum()), "attack_after": int(data.y.sum()), **extra}
    return data, info, dd


def _cache_key(cfg: Dict[str, Any]) -> str:
    import hashlib, json
    from pathlib import Path
    d = cfg["data"]
    stats = []
    p = Path(d.get("path", ""))
    for f in ([p] if p.is_file() else sorted(p.glob("*.csv")) if p.is_dir() else []):
        st = f.stat(); stats.append([f.name, st.st_size, int(st.st_mtime)])
    key = {"data": d, "dedup": cfg.get("dedup", "none"), "near_dup": cfg.get("near_dup"), "dedup_first": cfg.get("dedup_first"), "files": stats}
    if d.get("kind") == "cic2017":
        key["loader"] = "cic2017_v3"      # v3: mixed-format timestamp parsing (v2 left ~19% of 2017 timestamps unparsed)
    if d.get("kind") == "cic2018":
        key["loader"] = "cic2018_v2"      # v1 mislabelled every 2018 row as an attack ("Benign" != "BENIGN"); never reuse that cache
    blob = json.dumps(key, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def prepare(cfg: Dict[str, Any]):
    """Load + dedup (+ sample), cached on disk for REAL datasets: the first experiment pays the cost of reading/deduplicating the
    full file, the next experiments with identical data settings reuse it (.cache/prep_<hash>.pkl). Disable: set CG_CACHE=0.
    The key includes data settings, dedup settings and the data files' sizes/mtimes, so a changed file is never served stale."""
    import os, pickle
    from pathlib import Path
    if cfg["data"].get("kind", "synthetic") == "synthetic" or os.environ.get("CG_CACHE", "1") == "0":
        return _prepare_uncached(cfg)
    f = Path(".cache") / f"prep_{_cache_key(cfg)}.pkl"
    if f.exists():
        try:
            with open(f, "rb") as fh:
                out = pickle.load(fh)
            print(f"[cache] reused prepared data {f.name}", flush=True)
            return out
        except Exception:
            pass
    out = _prepare_uncached(cfg)
    try:
        f.parent.mkdir(exist_ok=True)
        with open(f, "wb") as fh:
            pickle.dump(out, fh, protocol=4)
    except Exception as e:      # caching is an optimisation only
        print(f"[cache] not saved: {e}", flush=True)
    return out
