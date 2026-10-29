# (A) finer time blocks for the headline time-shift claim, (B) BAR with bulk edge 0.90 on all datasets. Run from the project folder.
$ErrorActionPreference = "Continue"
$m4 = "models=[isolation_forest,pca,hbos,ecod]"
$fine = "scenarios=[{kind: temporal, block_mode: stime, n_blocks: 15, fit_blocks: [0,1,2], label_blocks: [3]}]"
New-Item -ItemType Directory -Force logs | Out-Null
$ds = @(@("unsw_raw","unsw_raw"), @("cic2017","cic2017"), @("cic2018","cic2018"))
foreach ($j in $ds) {
  $t = Get-Date; Write-Host "START fine-blocks $($j[0]) $t"
  python -W ignore -m cgresearch.cli calib-transfer -c "configs\v2_$($j[1])_calib_transfer.yaml" --seeds 0-4 --out results10 --set "name=$($j[0])_fine_calib" --set $m4 --set $fine 2>&1 | Tee-Object -FilePath "logs\fine_$($j[0]).log"
  Write-Host "END fine $($j[0]) after $([int]((Get-Date)-$t).TotalMinutes) min"
}
foreach ($j in $ds) {
  $t = Get-Date; Write-Host "START bar90 $($j[0]) $t"
  python -W ignore -m cgresearch.cli shift-monitor -c "configs\v2_$($j[1])_shift_monitor.yaml" --seeds 0-4 --out results10 --set "name=$($j[0])_bar90" --set $m4 --set bar.bulk=0.9 2>&1 | Tee-Object -FilePath "logs\bar90_$($j[0]).log"
  Write-Host "END bar90 $($j[0]) after $([int]((Get-Date)-$t).TotalMinutes) min"
}
$d = "confirm_zip"; Remove-Item $d -Recurse -Force -ErrorAction SilentlyContinue
foreach ($j in $ds) { foreach ($tag in "fine_calib","bar90") { $n = "$($j[0])_$tag"
  $s = Get-ChildItem "results10\$n" -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
  if ($s) { $t = "$d\$n\$($s.Name)"; New-Item -ItemType Directory $t -Force | Out-Null; Copy-Item "$($s.FullName)\manifest.json" $t; Copy-Item "$($s.FullName)\tables" "$t\tables" -Recurse
            Get-ChildItem "$t\tables" -Recurse -File | Where-Object Length -gt 25MB | Remove-Item } } }
Compress-Archive $d confirm_results.zip -Force
Write-Host ("zip MB: " + [math]::Round((Get-Item confirm_results.zip).Length/1MB,1))
