# Item 3: calibration transfer with 10 seeds for CIC-IDS2017 and CIC-IDS2018 (same settings as the earlier 3-seed runs). Run from the project folder.
$ErrorActionPreference = "Continue"
$m4 = "models=[isolation_forest,pca,hbos,ecod]"
New-Item -ItemType Directory -Force logs | Out-Null
$ds = @("cic2017","cic2018")
foreach ($d0 in $ds) {
  $t = Get-Date; Write-Host "START calib10 $d0 $t"
  python -W ignore -m cgresearch.cli calib-transfer -c "configs\v2_${d0}_calib_transfer.yaml" --seeds 0-9 --out results10 --set $m4 2>&1 | Tee-Object -FilePath "logs\calib10_$d0.log"
  Write-Host "END calib10 $d0 after $([int]((Get-Date)-$t).TotalMinutes) min"
}
$d = "item3_zip"; Remove-Item $d -Recurse -Force -ErrorAction SilentlyContinue
foreach ($n in "cic2017_calib_transfer","cic2018_calib_transfer") {
  $s = Get-ChildItem "results10\$n" -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
  if ($s) { $t = "$d\$n\$($s.Name)"; New-Item -ItemType Directory $t -Force | Out-Null; Copy-Item "$($s.FullName)\manifest.json" $t; Copy-Item "$($s.FullName)\tables" "$t\tables" -Recurse
            Get-ChildItem "$t\tables" -Recurse -File | Where-Object Length -gt 25MB | Remove-Item } }
Compress-Archive $d item3_results.zip -Force
Write-Host ("zip MB: " + [math]::Round((Get-Item item3_results.zip).Length/1MB,1))
