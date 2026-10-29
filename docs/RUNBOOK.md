# Runbook

## 1. Setup
```
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt            # + requirements-optional.txt for lightgbm/hnswlib
python -m pytest -q tests                  # expect 62 passed
```
## 2. Data (not redistributed) — see docs/DATA_GUIDE.md
```
data/UNSW-NB15/{UNSW-NB15_1.csv,NUSW-NB15_features.csv,UNSW_NB15_training-set.csv}
data/CIC-IDS2017/*.csv      data/CIC-IDS2018/*.csv (optional)      data/NF-UNSW-NB15-v2.csv (optional)
```
## 2b. Windows PowerShell (your machine)
```
.\.venv\Scripts\Activate.ps1
.\reproduce_v2.ps1 -Smoke                      # plumbing check, synthetic
.\reproduce_v2.ps1 -Pilot -Datasets cic2017    # small real pass (3 seeds, 4 detectors); then:
python tools\check_run.py results              # PASS/WARN/FAIL sanity checks on manifests
.\reproduce_v2.ps1 -Datasets cic2017,unsw_raw  # full preregistered run (10 seeds)
python tools\pack_results.py results cybergene_results_to_send.zip   # tables only -> send this zip
```
## 3. Plumbing check (minutes)
```
./reproduce_v2.sh --smoke          # Windows: run the same `python -m cgresearch.cli <cmd> --smoke` lines
```
Outputs are `SMOKE_*` and synthetic. **Never** cite them.
## 4. Real runs
Freeze docs/PREREGISTRATION.md first (`git tag prereg-v1`). Then, one dataset at a time (cheapest first):
```
./reproduce_v2.sh cic2017
./reproduce_v2.sh unsw_raw
./reproduce_v2.sh cic2018 nf_unsw      # optional
```
Cost guide: reduce `seeds` (e.g. `--seeds 0-2`) and `max_fit_rows` first; OCSVM/LOF dominate runtime — drop them with `--set models=[...]` for a first pass. Every run folder holds `config.json`, `manifest.json` (git commit, data hashes, dedup info), and `tables/*.csv|json`.
## 5. What to check before writing numbers
- `manifest.json -> dataset.cleaning` (blank-label rows dropped; CIC variant string filled in the config).
- `calib_transfer manifest: identity_max_abs_error` ≈ 1e-16 and `ks_bound_holds_fraction` ≈ 1.
- temporal scenario on UNSW train-partition has no timestamps → blocks are row order: do not call it "temporal" in the paper; use `unsw_raw` (has `stime`) or CIC (`day`).
- `tau_infinite` cells: n_cal too small for the guarantee at that α.
## 6. After running
Commit results summaries (not raw data), push to GitHub, re-archive on Zenodo, and send the tables to the manuscript step (docs/MANUSCRIPT_V2_OUTLINE.md).
