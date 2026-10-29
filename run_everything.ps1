<#
 CYBERGENE - ONE FILE FOR EVERYTHING (CIC-IDS2017, UNSW-NB15, CIC-IDS2018).
 Put this file in the project root (next to configs\, tools\, cgresearch\), activate the venv, then run:
     .\run_everything.ps1
 Safe to stop (Ctrl+C) and restart: finished experiments are skipped, the download resumes/skips finished files.
 It never needs you: failures are logged and the next step starts. Expect many hours.
#>
$ErrorActionPreference = "Continue"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -ErrorAction SilentlyContinue
$env:PYTHONHASHSEED = "0"; $env:OMP_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
$ProgressPreference = "SilentlyContinue"
New-Item -ItemType Directory -Force logs | Out-Null
function Log($m) { $l = "$(Get-Date -Format 'yyyy-MM-dd HH:mm')  $m"; Write-Host $l; Add-Content logs\run_everything.log $l }

# ---------- 0. keep the laptop awake for the whole run ----------
try {
    powercfg /change standby-timeout-ac 0 | Out-Null; powercfg /change hibernate-timeout-ac 0 | Out-Null
    powercfg /change monitor-timeout-ac 0 | Out-Null
    Add-Type -Namespace Win -Name Pwr -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
    [void][Win.Pwr]::SetThreadExecutionState(0x80000003)      # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    Log "keep-awake enabled (keep the laptop plugged in)"
} catch { Log "could not set keep-awake: $_" }

# ---------- 1. sanity checks ----------
if (-not (Test-Path "cgresearch") -or -not (Test-Path "configs")) { Log "ERROR: run this from the project root (the folder with cgresearch and configs)"; return }
$free = [math]::Round((Get-PSDrive -Name ((Get-Location).Drive.Name)).Free / 1GB, 1)
Log "free disk: $free GB (CIC-2018 needs about 8 GB)"

# ---------- 2. CIC-2018 patch (memory-safe loader), applied automatically if found ----------
$patched = Select-String -Path cgresearch\data\cic.py -Pattern "per_file_cap" -Quiet
if (-not $patched) {
    $zip = @("cic2018_patch.zip", "$env:USERPROFILE\Downloads\cic2018_patch.zip") | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($zip) {
        Expand-Archive -Force $zip "$env:TEMP\cg_patch"; Copy-Item "$env:TEMP\cg_patch\CYBERGENE_v2\*" . -Recurse -Force
        Log "applied $zip"; $patched = $true
    } else { Log "WARNING: cic2018_patch.zip not found -> CIC-2018 stages will be skipped (put the zip in this folder or Downloads and rerun)" }
}

# ---------- 3. CIC-2018 download in the background (resumable / skips finished files) ----------
$dl = $null
$dest = Join-Path (Get-Location) "data\CIC-IDS2018"
if ($patched -and -not (Test-Path "$dest\.download_complete")) {
    New-Item -ItemType Directory -Force $dest | Out-Null
    $dl = Start-Job -Name cic18 -ArgumentList $dest -ScriptBlock {
        param($dest)
        $ProgressPreference = "SilentlyContinue"
        $log = Join-Path (Split-Path (Split-Path $dest)) "logs\download_cic2018.log"
        function L($m) { Add-Content $log "$(Get-Date -Format HH:mm)  $m" }
        $prefix = "Processed Traffic Data for ML Algorithms/"
        $ok = $false
        if (Get-Command aws -ErrorAction SilentlyContinue) {
            L "using aws cli"
            aws s3 sync --no-sign-request --region ca-central-1 "s3://cse-cic-ids2018/$prefix" $dest 2>&1 | ForEach-Object { L $_ }
            $ok = ($LASTEXITCODE -eq 0)
        }
        if (-not $ok) {
            L "using HTTPS listing"
            try {
                $base = "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com"
                $xml = [xml](Invoke-WebRequest -UseBasicParsing "$base/?list-type=2&prefix=$([uri]::EscapeDataString($prefix))").Content
                $items = @($xml.ListBucketResult.Contents | Where-Object { $_.Key -like "*.csv" })
                L "found $($items.Count) csv files"
                $ok = ($items.Count -gt 0)
                foreach ($it in $items) {
                    $name = Split-Path $it.Key -Leaf; $out = Join-Path $dest $name
                    if ((Test-Path $out) -and ((Get-Item $out).Length -eq [int64]$it.Size)) { L "have $name"; continue }
                    L "downloading $name ($([math]::Round([int64]$it.Size/1MB)) MB)"
                    $tmp = "$out.part"; $wc = New-Object System.Net.WebClient
                    try { $wc.DownloadFile("$base/$([uri]::EscapeUriString($it.Key))", $tmp); Move-Item -Force $tmp $out } catch { L "FAILED $name : $_"; $ok = $false }
                }
            } catch { L "listing failed: $_"; $ok = $false }
        }
        if ($ok) { Set-Content (Join-Path $dest ".download_complete") "ok"; L "download complete" } else { L "download NOT complete" }
    }
    Log "CIC-2018 download started in background (see logs\download_cic2018.log)"
}

# ---------- 4. experiment runner (skips finished, retries once, logs time) ----------
$models4 = "models=[isolation_forest,pca,hbos,ecod]"
$modelsX = "models=[lof,autoencoder,kitnet_style]"
$modelsS = "models=[logreg,lightgbm]"
function Done($out, $name) { return (@(Get-ChildItem "$out\$name\*\tables" -ErrorAction SilentlyContinue).Count -gt 0) }
function Step($label, $cmd, $config, $name, $seeds, $out, $models) {
    if (Done $out $name) { Log "SKIP $label (finished in $out\$name)"; return }
    if ($config -match "cic2018") {
        if (-not $patched) { Log "SKIP $label (patch missing)"; return }
        if ($dl) { Wait-Job $dl | Out-Null; Receive-Job $dl | Out-Null }
        if (-not (Test-Path "$dest\.download_complete")) { Log "SKIP $label (CIC-2018 download not complete, see logs\download_cic2018.log)"; return }
    }
    if ($config -match "cic2017" -and -not (Test-Path "data\CIC-IDS2017\*.csv")) { Log "SKIP $label (data\CIC-IDS2017 missing)"; return }
    if ($config -match "unsw" -and -not (Test-Path "data\UNSW-NB15")) { Log "SKIP $label (data\UNSW-NB15 missing)"; return }
    $a = @($cmd, "-c", $config, "--seeds", $seeds, "--out", $out, "--set", $models, "--set", "max_fit_rows=30000", "--set", "n_boot=2000")
    for ($try = 1; $try -le 2; $try++) {
        Log "START $label (try $try)"
        $t = Get-Date
        python -W ignore -m cgresearch.cli @a 2>&1 | Tee-Object -FilePath "logs\$label.log" -Append
        $m = [math]::Round(((Get-Date) - $t).TotalMinutes, 1)
        if (Done $out $name) { Log "DONE  $label in $m min"; Add-Content logs\timings.txt "$label : $m min"; return }
        Log "FAILED $label after $m min (see logs\$label.log)"
    }
}
function Raw($label, $name, $out, [string[]]$a) {
    if (Done $out $name) { Log "SKIP $label (finished)"; return }
    Log "START $label"; $t = Get-Date
    python -W ignore -m cgresearch.cli @a 2>&1 | Tee-Object -FilePath "logs\$label.log" -Append
    $m = [math]::Round(((Get-Date) - $t).TotalMinutes, 1)
    if (Done $out $name) { Log "DONE  $label in $m min"; Add-Content logs\timings.txt "$label : $m min" } else { Log "FAILED $label (see logs\$label.log)" }
}

# ---------- 5. experiments, ordered by value ----------
# A. UNSW-NB15, 10 seeds
Step "A1_unsw_calib"  "calib-transfer" "configs\v2_unsw_raw_calib_transfer.yaml" "unsw_raw_calib_transfer" "0-9" "results10" $models4
Step "A2_unsw_shift"  "shift-monitor"  "configs\v2_unsw_raw_shift_monitor.yaml"  "unsw_raw_shift_monitor"  "0-9" "results10" $models4
Step "A3_unsw_contam" "contamination"  "configs\v2_unsw_raw_contamination.yaml"  "unsw_raw_contamination"  "0-9" "results10" $models4
# B. Simulation studies
Raw "B1_recal"   "recal_study"   "results10" @("recal-study",   "-c", "configs\v2_recal_study.yaml",   "--seeds", "0-9", "--out", "results10", "--set", "reps=20")
Raw "B2_monitor" "monitor_study" "results10" @("monitor-study", "-c", "configs\v2_monitor_study.yaml", "--seeds", "0-4", "--out", "results10", "--set", "reps=8", "--set", "n_tune=20", "--set", "cert_reps=10")
Raw "B3_sentinel" "sentinel_case" "results10" @("sentinel-case", "-c", "configs\v2_sentinel_case.yaml", "--out", "results10")
# C. CIC-IDS2017 main time-shift result, 10 seeds
Step "C1_cic_shift"  "shift-monitor" "configs\v2_cic2017_shift_monitor.yaml" "cic2017_shift_monitor" "0-9" "results10" $models4
Step "C2_cic_contam" "contamination" "configs\v2_cic2017_contamination.yaml" "cic2017_contamination" "0-9" "results10" $models4
# D. Pre-registered protocol sensitivity (H5), 3 seeds, both labelled datasets
Step "D1_cic_protocol"  "protocol-sens" "configs\v2_cic2017_protocol_sens.yaml"    "cic2017_protocol_sens"   "0-2" "results10" $models4
Step "D2_unsw_protocol" "protocol-sens" "configs\v2_unsw_raw_protocol_sens.yaml"   "unsw_raw_protocol_sens"  "0-2" "results10" $models4
# E. CIC-IDS2018 (third dataset): waits for the download, 5 seeds (calibration transfer 3)
Step "E1_cic18_shift"  "shift-monitor"  "configs\v2_cic2018_shift_monitor.yaml"  "cic2018_shift_monitor"  "0-4" "results10" $models4
Step "E2_cic18_contam" "contamination"  "configs\v2_cic2018_contamination.yaml"  "cic2018_contamination"  "0-4" "results10" $models4
Step "E3_cic18_calib"  "calib-transfer" "configs\v2_cic2018_calib_transfer.yaml" "cic2018_calib_transfer" "0-2" "results10" $models4
Step "E4_cic18_protocol" "protocol-sens" "configs\v2_cic2018_protocol_sens.yaml"  "cic2018_protocol_sens"  "0-2" "results10" $models4
# F. Extra detectors and supervised baselines on CIC-2017 / UNSW, 3 seeds (separate folder)
Step "F1_cic_shift_extra"  "shift-monitor"  "configs\v2_cic2017_shift_monitor.yaml"  "cic2017_shift_monitor"  "0-2" "results10det" $modelsX
Step "F2_unsw_shift_extra" "shift-monitor"  "configs\v2_unsw_raw_shift_monitor.yaml"  "unsw_raw_shift_monitor" "0-2" "results10det" $modelsX
Step "F3_cic_calib_extra"  "calib-transfer" "configs\v2_cic2017_calib_transfer.yaml" "cic2017_calib_transfer" "0-2" "results10det" $modelsX
Step "F4_cic_supervised"   "calib-transfer" "configs\v2_cic2017_calib_transfer.yaml" "cic2017_calib_transfer" "0-2" "results10sup" $modelsS
# G. The longest one last: CIC-2017 calibration transfer, 5 seeds
Step "G1_cic_calib5" "calib-transfer" "configs\v2_cic2017_calib_transfer.yaml" "cic2017_calib_transfer" "0-4" "results10" $models4

# ---------- 6. check and pack ----------
foreach ($d in "results10", "results10det", "results10sup") {
    if (Test-Path $d) { python tools\check_run.py $d 2>&1 | Tee-Object -FilePath "logs\check_$d.log"; python tools\pack_results.py $d "cybergene_$d.zip" }
}
Compress-Archive -Force -Path logs\*.log, logs\timings.txt -DestinationPath cybergene_logs.zip -ErrorAction SilentlyContinue
Log "ALL DONE. Send me: cybergene_results10.zip, cybergene_results10det.zip, cybergene_results10sup.zip (if present) and cybergene_logs.zip"
[void][Win.Pwr]::SetThreadExecutionState(0x80000000)
