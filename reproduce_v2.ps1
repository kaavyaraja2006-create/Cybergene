# CYBERGENE v2 on Windows PowerShell.   Usage:  .\reproduce_v2.ps1 [-Smoke] [-Pilot] [-Datasets cic2017,unsw_raw]
# -Smoke : tiny SYNTHETIC data (plumbing check; outputs are SMOKE_*, never cite).
# -Pilot : small real-data pass (3 seeds, 4 fast detectors, 30k fit rows) -> check numbers before the full run.
param([switch]$Smoke, [switch]$Pilot, [string[]]$Datasets = @("cic2017","unsw_raw"))
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONHASHSEED = "0"; $env:OMP_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
python -m pytest -q tests
if ($LASTEXITCODE -ne 0) { throw "tests failed - fix before running experiments" }
function CG { python -W ignore -m cgresearch.cli @args; if ($LASTEXITCODE -ne 0) { throw "cgresearch.cli failed: $args" } }
if ($Smoke) {
  foreach ($c in "calib-transfer","shift-monitor","contamination","protocol-sens","sentinel-case","recal-study","monitor-study") { CG $c --smoke }
  exit 0
}
$paths = @{ cic2017 = "data\CIC-IDS2017"; cic2018 = "data\CIC-IDS2018"; unsw_raw = "data\UNSW-NB15\UNSW-NB15_1.csv"; unsw_train = "data\UNSW-NB15\UNSW_NB15_training-set.csv"; nf_unsw = "data\NF-UNSW-NB15-v2.csv" }
$extra = @()
if ($Pilot) { $extra = @("--seeds","0-2","--set","models=[isolation_forest,pca,hbos,ecod]","--set","max_fit_rows=30000","--set","n_boot=2000") }
foreach ($d in $Datasets) {
  if (-not (Test-Path $paths[$d])) { Write-Host "skip $d (data missing: $($paths[$d]))"; continue }
  foreach ($e in "calib_transfer","shift_monitor","contamination","protocol_sens") {
    $cmd = $e.Replace("_","-")
    CG $cmd -c "configs\v2_${d}_$e.yaml" @extra
  }
}
if (-not $Pilot) {
  CG recal-study -c configs\v2_recal_study.yaml
  CG monitor-study -c configs\v2_monitor_study.yaml
  CG sentinel-case -c configs\v2_sentinel_case.yaml
}
python tools\check_run.py results
python tools\pack_results.py results cybergene_results_to_send.zip
Write-Host "`nSend cybergene_results_to_send.zip (tables only, no raw data)."
