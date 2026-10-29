"""Sanity-check real runs BEFORE sending results or writing numbers.   python tools/check_run.py [results_dir]
Reads every results/*/*/manifest.json and reports PASS / WARN / FAIL per run.  Exit code 1 if any FAIL."""
import json, sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
fails = warns = 0


def say(level, run, msg):
    global fails, warns
    fails += level == "FAIL"; warns += level == "WARN"
    print(f"[{level}] {run}: {msg}")


for mf in sorted(root.glob("*/*/manifest.json")):
    m = json.load(open(mf)); name = mf.parent.name; ds = m.get("dataset") or {}
    if m.get("status") != "completed":
        say("FAIL", name, f"status = {m.get('status')}"); continue
    if m["experiment"].startswith("SMOKE_") or ds.get("synthetic"):
        print(f"[skip] {name}: synthetic/smoke run (not a result)"); continue
    ok = True
    if not ds.get("sha256"):
        say("FAIL", name, "no dataset sha256 recorded"); ok = False
    v = str(ds.get("version") or "")
    if not v or "state variant" in v or "record which" in v:
        say("WARN", name, "dataset version string is still the placeholder; write the exact release (e.g. CIC-IDS2017 GeneratedLabelledFlows)")
    cl = ds.get("cleaning") or {}
    if "blank_label_rows_dropped" in cl:
        print(f"[info] {name}: blank-label rows dropped = {cl['blank_label_rows_dropped']}, inf/NaN cells = {cl.get('inf_or_nan_cells')}")
    g = m.get("git") or {}
    if not g.get("commit") or g.get("dirty"):
        say("WARN", name, "git commit missing or working tree dirty: commit before the final run so results map to a commit")
    if "identity_max_abs_error" in m and m["identity_max_abs_error"] > 1e-9:
        say("FAIL", name, f"shift-decomposition identity error {m['identity_max_abs_error']:.2e} (expected ~1e-16) -> bug"); ok = False
    if "ks_bound_holds_fraction" in m and m["ks_bound_holds_fraction"] < 0.95:
        say("WARN", name, f"KS/DKW bound holds in only {m['ks_bound_holds_fraction']:.1%} of cells (expected >= ~99%)")
    dd = m.get("dedup") or {}
    if dd.get("rows_before"):
        print(f"[info] {name}: dedup {dd.get('mode')} removed {dd.get('removed')}/{dd.get('rows_before')} ({dd.get('removed', 0) / dd['rows_before']:.1%}); attacks {dd.get('attack_before')} -> {dd.get('attack_after')}; dedup_first={dd.get('dedup_first')}")
    if len(m.get("seeds") or []) < 3:
        say("WARN", name, f"only {len(m.get('seeds') or [])} seeds (pilot?). Preregistered runs use 10")
    if ok:
        print(f"[PASS] {name}  ({m.get('runtime_seconds', 0):.0f}s)")
print(f"\n{fails} FAIL, {warns} WARN")
sys.exit(1 if fails else 0)
