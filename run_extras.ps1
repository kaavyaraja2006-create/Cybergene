# Two extra real-data experiments. Run from the project folder AFTER unzipping extras_patch.zip over it:
#   Expand-Archive "$env:USERPROFILE\Downloads\extras_patch.zip" -DestinationPath . -Force
#   .\run_extras.ps1            (everything, ~1-2 h)      .\run_extras.ps1 -Parts 1   (only part 1)
# Part 1: benign-only day. CIC-IDS2017 Monday has no attacks at all, so any rise of the false-positive rate over time there is time drift, not attack composition.
# Part 2: retraining. Refit the detector and recalibrate on block k, test on block k+1 (k = 1,2,3), on all three datasets.
# Part 3: oracle diagnostic (reference drawn from the whole window), 5 seeds.
param([int[]]$Parts = @(1,2,3))
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force logs | Out-Null
$m4 = "models=[isolation_forest,pca,hbos,ecod]"
$made = @()
function Run($cmd, $cfg, $seeds, $name, $extra) {
  $t = Get-Date; Write-Host "START $name $t"
  $a = @("-W","ignore","-m","cgresearch.cli",$cmd,"-c","configs\$cfg.yaml","--seeds",$seeds,"--out","results10","--set","name=$name") + $extra
  & python @a 2>&1 | Tee-Object -FilePath "logs\$name.log"
  Write-Host "END $name after $([int]((Get-Date)-$t).TotalMinutes) min"
  $script:made += $name
}
if ($Parts -contains 1) {
  $fine = "scenarios=[{kind: temporal, block_mode: stime, n_blocks: 15, fit_blocks: [0,1,2], label_blocks: [3]}]"
  Run "calib-transfer" "v2_cic2017_calib_transfer" "0-4" "cic2017_monday_calib" @("--set","data.include=monday","--set",$m4,"--set",$fine,"--set","alphas=[0.05]","--set","methods=[conformal]")
}
if ($Parts -contains 2) {
  $rt = "scenarios=[{kind: temporal, name: rt_1to2, n_blocks: 5, fit_blocks: [1], label_blocks: [], eval_blocks: [2]}, {kind: temporal, name: rt_2to3, n_blocks: 5, fit_blocks: [2], label_blocks: [], eval_blocks: [3]}, {kind: temporal, name: rt_3to4, n_blocks: 5, fit_blocks: [3], label_blocks: [], eval_blocks: [4]}]"
  $rt18 = "scenarios=[{kind: temporal, name: rt_1to2, n_blocks: 5, fit_blocks: [1], label_blocks: [], eval_blocks: [2]}, {kind: temporal, name: rt_2to3, n_blocks: 5, fit_blocks: [2], label_blocks: [], eval_blocks: [3]}]"
  foreach ($d in "unsw_raw","cic2017","cic2018") {
    $sc = if ($d -eq "cic2018") { $rt18 } else { $rt }     # CIC-IDS2018 has only blocks 0-3
    Run "calib-transfer" "v2_${d}_calib_transfer" "0-4" "${d}_retrain" @("--set",$m4,"--set",$sc,"--set","alphas=[0.05]","--set","methods=[conformal]")
  }
}
if ($Parts -contains 3) {
  $pol = "policies=[static,oracle_periodic,oracle_random,unlabeled_periodic,naive_random,periodic_bar]"
  foreach ($d in "unsw_raw","cic2017","cic2018") {
    Run "shift-monitor" "v2_${d}_shift_monitor" "0-4" "${d}_oracle_diag" @("--set",$m4,"--set",$pol)
  }
}
$z = "extras_zip"; Remove-Item $z -Recurse -Force -ErrorAction SilentlyContinue
foreach ($n in $made) {
  $s = Get-ChildItem "results10\$n" -Directory -ErrorAction SilentlyContinue | Where-Object { Test-Path "$($_.FullName)\manifest.json" } | Sort-Object LastWriteTime | Select-Object -Last 1
  if ($s) { $t = "$z\$n\$($s.Name)"; New-Item -ItemType Directory $t -Force | Out-Null; Copy-Item "$($s.FullName)\manifest.json" $t; Copy-Item "$($s.FullName)\tables" "$t\tables" -Recurse
            Get-ChildItem "$t\tables" -Recurse -File | Where-Object Length -gt 25MB | Remove-Item } }
Compress-Archive $z extras_results.zip -Force
Write-Host ("zip MB: " + [math]::Round((Get-Item extras_results.zip).Length/1MB,1))
