"""Pack ONLY the small result tables (+ manifests/configs) into one zip to send back.  python tools/pack_results.py [results_dir] [out.zip]
Excludes artifacts/ and raw per-row dumps larger than 25 MB.  Never includes data/."""
import sys, zipfile
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results"); out = Path(sys.argv[2] if len(sys.argv) > 2 else "cybergene_results_to_send.zip")
LIMIT = 25 * 1024 * 1024; n = skipped = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(root.rglob("*")):
        if not f.is_file() or "artifacts" in f.parts or f.suffix not in (".csv", ".json", ".yaml", ".yml", ".md", ".txt"):
            continue
        if f.stat().st_size > LIMIT:
            skipped += 1; print("skipped (too large):", f); continue
        z.write(f, f.relative_to(root.parent)); n += 1
print(f"wrote {out} with {n} files ({out.stat().st_size / 1e6:.1f} MB), skipped {skipped}")
