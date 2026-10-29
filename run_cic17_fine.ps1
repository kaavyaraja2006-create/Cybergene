# Re-run the finer-time-block measurement for CIC-IDS2017 with the fixed timestamp parser. Run from the project folder.
$m4 = "models=[isolation_forest,pca,hbos,ecod]"
$fine = "scenarios=[{kind: temporal, block_mode: stime, n_blocks: 15, fit_blocks: [0,1,2], label_blocks: [3]}]"
New-Item -ItemType Directory -Force logs | Out-Null
$t = Get-Date
python -W ignore -m cgresearch.cli calib-transfer -c "configs\v2_cic2017_calib_transfer.yaml" --seeds 0-4 --out results10 --set "name=cic2017_fine2_calib" --set $m4 --set $fine 2>&1 | Tee-Object -FilePath "logs\fine2_cic2017.log"
Write-Host "END after $([int]((Get-Date)-$t).TotalMinutes) min"
$d = "fine2_zip"; Remove-Item $d -Recurse -Force -ErrorAction SilentlyContinue
$s = Get-ChildItem "results10\cic2017_fine2_calib" -Directory | Sort-Object LastWriteTime | Select-Object -Last 1
$t2 = "$d\cic2017_fine2_calib\$($s.Name)"; New-Item -ItemType Directory $t2 -Force | Out-Null
Copy-Item "$($s.FullName)\manifest.json" $t2; Copy-Item "$($s.FullName)\tables" "$t2\tables" -Recurse
Compress-Archive $d fine2_results.zip -Force
Write-Host ("zip MB: " + [math]::Round((Get-Item fine2_results.zip).Length/1MB,1))
