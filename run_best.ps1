# CYBERGENE best-evidence run. Safe to stop (Ctrl+C) and restart: finished experiments are skipped.
# Usage (inside the project folder, venv active):  .\run_best.ps1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -ErrorAction SilentlyContinue
$env:PYTHONHASHSEED = "0"; $env:OMP_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
New-Item -ItemType Directory -Force logs | Out-Null

$models4 = "models=[isolation_forest,pca,hbos,ecod]"
$models3 = "models=[lof,autoencoder,kitnet_style]"

function Step($label, $cmd, $config, $name, $seeds, $out, $models) {
    if (Test-Path "$out\$name") { "SKIP $label (already in $out\$name)" ; return }
    if ($config -match "cic2018" -and -not (Test-Path "data\CIC-IDS2018\*.csv")) { "SKIP $label (no data\CIC-IDS2018 yet)"; return }
    "=== $label  $(Get-Date -Format HH:mm)"
    $t = Get-Date
    $a = @($cmd, "-c", $config, "--seeds", $seeds, "--out", $out, "--set", $models, "--set", "max_fit_rows=30000", "--set", "n_boot=2000")
    python -W ignore -m cgresearch.cli @a 2>&1 | Tee-Object -FilePath "logs\$label.log"
    $m = [math]::Round(((Get-Date) - $t).TotalMinutes, 1)
    "$label : $m min" | Add-Content logs\timings.txt
    "--- $label finished in $m min"
}

# A. UNSW-NB15 (fast), 10 seeds
Step "A1_unsw_calib"  "calib-transfer" "configs\v2_unsw_raw_calib_transfer.yaml" "unsw_raw_calib_transfer" "0-9" "results10" $models4
Step "A2_unsw_shift"  "shift-monitor"  "configs\v2_unsw_raw_shift_monitor.yaml"  "unsw_raw_shift_monitor"  "0-9" "results10" $models4
Step "A3_unsw_contam" "contamination"  "configs\v2_unsw_raw_contamination.yaml"  "unsw_raw_contamination"  "0-9" "results10" $models4

# B. Simulation studies (fast)
"=== B simulations"
if (-not (Test-Path "results10\recal_study"))   { python -W ignore -m cgresearch.cli recal-study   -c configs\v2_recal_study.yaml   --seeds 0-9 --out results10 --set reps=20 2>&1 | Tee-Object "logs\B1_recal.log" }
if (-not (Test-Path "results10\monitor_study")) { python -W ignore -m cgresearch.cli monitor-study -c configs\v2_monitor_study.yaml --seeds 0-4 --out results10 --set reps=8 --set n_tune=20 --set cert_reps=10 2>&1 | Tee-Object "logs\B2_monitor.log" }

# C. CIC-IDS2017 main time-shift result, 10 seeds
Step "C1_cic_shift"  "shift-monitor" "configs\v2_cic2017_shift_monitor.yaml" "cic2017_shift_monitor" "0-9" "results10" $models4
Step "C2_cic_contam" "contamination" "configs\v2_cic2017_contamination.yaml" "cic2017_contamination" "0-9" "results10" $models4

# D. Pre-registered protocol sensitivity (H5), 3 seeds
Step "D1_cic_protocol" "protocol-sens" "configs\v2_cic2017_protocol_sens.yaml" "cic2017_protocol_sens" "0-2" "results10" $models4

# E. Extra detectors on CIC-IDS2017, 3 seeds (separate folder)
Step "E1_cic_shift_extra" "shift-monitor"  "configs\v2_cic2017_shift_monitor.yaml"  "cic2017_shift_monitor"  "0-2" "results10det" $models3
Step "E2_cic_calib_extra" "calib-transfer" "configs\v2_cic2017_calib_transfer.yaml" "cic2017_calib_transfer" "0-2" "results10det" $models3

# F. Third dataset: full CIC-IDS2018 (only runs if data\CIC-IDS2018\*.csv exists), 5 seeds
Step "F1_cic18_shift"  "shift-monitor" "configs\v2_cic2018_shift_monitor.yaml" "cic2018_shift_monitor" "0-4" "results10" $models4
Step "F2_cic18_contam" "contamination" "configs\v2_cic2018_contamination.yaml" "cic2018_contamination" "0-4" "results10" $models4
Step "F3_cic18_calib"  "calib-transfer" "configs\v2_cic2018_calib_transfer.yaml" "cic2018_calib_transfer" "0-2" "results10" $models4

# G. Check and pack
python tools\check_run.py results10
if (Test-Path results10det) { python tools\check_run.py results10det }
python tools\pack_results.py results10 cybergene_results10.zip
if (Test-Path results10det) { python tools\pack_results.py results10det cybergene_results10det.zip }
"ALL DONE. Send cybergene_results10.zip (+ det zip) and logs\timings.txt"
