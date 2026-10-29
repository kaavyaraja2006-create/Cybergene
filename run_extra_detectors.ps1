# Extra detectors (autoencoder, LOF) for the calibration-under-shift claim. Run from the project folder:  .\run_extra_detectors.ps1
$ErrorActionPreference = "Continue"
$m = "models=[autoencoder,lof]"
$sc = "scenarios=[{kind: random},{kind: temporal, n_blocks: 5, fit_blocks: [0], label_blocks: [1]}]"
New-Item -ItemType Directory -Force logs | Out-Null
$jobs = @(@("unsw_raw","v2_unsw_raw_calib_transfer"), @("cic2017","v2_cic2017_calib_transfer"), @("cic2018","v2_cic2018_calib_transfer"))
foreach ($j in $jobs) {
  $t = Get-Date; Write-Host "START $($j[0]) $t"
  python -W ignore -m cgresearch.cli calib-transfer -c "configs\$($j[1]).yaml" --seeds 0-4 --out results10 --set "name=$($j[0])_extra_calib" --set $m --set $sc --set max_fit_rows=20000 2>&1 | Tee-Object -FilePath "logs\extra_$($j[0]).log"
  Write-Host "END $($j[0]) after $([int]((Get-Date)-$t).TotalMinutes) min"
}
$d = "extra_zip"; Remove-Item $d -Recurse -Force -ErrorAction SilentlyContinue
foreach ($j in $jobs) { $n = "$($j[0])_extra_calib"
  $s = Get-ChildItem "results10\$n" -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
  if ($s) { $t = "$d\$n\$($s.Name)"; New-Item -ItemType Directory $t -Force | Out-Null; Copy-Item "$($s.FullName)\manifest.json" $t; Copy-Item "$($s.FullName)\tables" "$t\tables" -Recurse
            Get-ChildItem "$t\tables" -Recurse -File | Where-Object Length -gt 25MB | Remove-Item } }
Compress-Archive $d extra_detectors_results.zip -Force
Write-Host ("zip MB: " + [math]::Round((Get-Item extra_detectors_results.zip).Length/1MB,1))
