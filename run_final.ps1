# Final evidence runs for the manuscript. Run from the project folder (after run_item3.ps1 has finished):
#   .\run_final.ps1                 (all stages)     .\run_final.ps1 -Stages 1,3   (only some)
#   .\run_final.ps1 -Stages 1 -Only cic2018   (resume: only these datasets)
# Stage 1: policies, 10 seeds, alpha 0.01/0.05/0.10, FIXED CIC-IDS2017 timestamp parser     (long: ~2-4 h)
# Stage 2: policies on the 11 finer time blocks, 5 seeds                                    (~1-2 h)
# Stage 3: calibration transfer at alpha 0.01/0.05/0.10, random + time blocks, 10 seeds    (~1-2 h)
# Stage 4: remaining planned detectors (OCSVM, Kitsune-style), 5 seeds                      (~1-2 h)
# Stage 6: oracle diagnostic (reference sample drawn from the whole window vs its last 2000 points), 5 seeds  (~1-2 h; needs oracle_patch.zip)
# Stage 5: NF-UNSW-NB15-v2 as a fourth dataset (needs data\NF-UNSW-NB15-v2.csv)             (~1 h)
param([int[]]$Stages = @(1,2,3,4,5,6), [string[]]$Only = @())
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force logs | Out-Null
$m4 = "models=[isolation_forest,pca,hbos,ecod]"
$a3 = "alphas=[0.01,0.05,0.10]"
$ds = @(@("unsw_raw","unsw_raw"), @("cic2017","cic2017"), @("cic2018","cic2018"))
function Sel($n) { return ($Only.Count -eq 0) -or ($Only -contains $n) }
$made = @()
function Run($cmd, $cfg, $seeds, $name, $extra) {
  $t = Get-Date; Write-Host "START $name $t"
  $args = @("-W","ignore","-m","cgresearch.cli",$cmd,"-c","configs\$cfg.yaml","--seeds",$seeds,"--out","results10","--set","name=$name") + $extra
  & python @args 2>&1 | Tee-Object -FilePath "logs\$name.log"
  Write-Host "END $name after $([int]((Get-Date)-$t).TotalMinutes) min"
  $script:made += $name
}
if ($Stages -contains 1) { foreach ($j in $ds) { if (-not (Sel $j[0])) { continue }; Run "shift-monitor" "v2_$($j[1])_shift_monitor" "0-9" "$($j[0])_policy10" @("--set",$m4,"--set",$a3) } }
if ($Stages -contains 2) {
  $fine = "temporal={block_mode: stime, n_blocks: 15, fit_blocks: [0,1,2], label_blocks: [3]}"
  foreach ($j in $ds) { if (-not (Sel $j[0])) { continue }; Run "shift-monitor" "v2_$($j[1])_shift_monitor" "0-4" "$($j[0])_finepolicy" @("--set",$m4,"--set",$fine) } }
if ($Stages -contains 3) {
  $sc = "scenarios=[{kind: random},{kind: temporal, n_blocks: 5, fit_blocks: [0], label_blocks: [1]}]"
  foreach ($j in $ds) { if (-not (Sel $j[0])) { continue }; Run "calib-transfer" "v2_$($j[1])_calib_transfer" "0-9" "$($j[0])_alpha_calib" @("--set",$m4,"--set",$a3,"--set",$sc,"--set","methods=[conformal]") } }
if ($Stages -contains 4) {
  $sc = "scenarios=[{kind: random},{kind: temporal, n_blocks: 5, fit_blocks: [0], label_blocks: [1]}]"
  foreach ($j in $ds) { if (-not (Sel $j[0])) { continue }; Run "calib-transfer" "v2_$($j[1])_calib_transfer" "0-4" "$($j[0])_extra2_calib" @("--set","models=[ocsvm,kitnet_style]","--set",$sc,"--set","max_fit_rows=20000") } }
if ($Stages -contains 6) {
  $pol = "policies=[static,oracle_periodic,oracle_random,unlabeled_periodic,naive_random,periodic_bar]"
  foreach ($j in $ds) { if (-not (Sel $j[0])) { continue }; Run "shift-monitor" "v2_$($j[1])_shift_monitor" "0-4" "$($j[0])_oracle_diag" @("--set",$m4,"--set",$pol) } }
if ($Stages -contains 5) {
  if (Test-Path "data\NF-UNSW-NB15-v2.csv") {
    $sc = "scenarios=[{kind: random},{kind: lofo, min_support: 100},{kind: temporal, n_blocks: 5, fit_blocks: [0], label_blocks: [1]}]"
    Run "calib-transfer" "v2_nf_unsw_calib_transfer" "0-4" "nf_unsw_calib" @("--set",$m4,"--set",$sc)
    $fine = "scenarios=[{kind: temporal, block_mode: stime, n_blocks: 15, fit_blocks: [0,1,2], label_blocks: [3]}]"
    Run "calib-transfer" "v2_nf_unsw_calib_transfer" "0-4" "nf_unsw_fine_calib" @("--set",$m4,"--set",$fine)
  } else { Write-Host "Stage 5 skipped: data\NF-UNSW-NB15-v2.csv not found" } }
$d = "final_zip"; Remove-Item $d -Recurse -Force -ErrorAction SilentlyContinue
foreach ($n in $made) {
  $s = Get-ChildItem "results10\$n" -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
  if ($s) { $t = "$d\$n\$($s.Name)"; New-Item -ItemType Directory $t -Force | Out-Null; Copy-Item "$($s.FullName)\manifest.json" $t; Copy-Item "$($s.FullName)\tables" "$t\tables" -Recurse
            Get-ChildItem "$t\tables" -Recurse -File | Where-Object Length -gt 25MB | Remove-Item } }
Compress-Archive $d final_results.zip -Force
Write-Host ("zip MB: " + [math]::Round((Get-Item final_results.zip).Length/1MB,1))
