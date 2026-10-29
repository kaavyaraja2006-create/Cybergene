# Windows PowerShell version of reproduce.sh.  Usage:  .\reproduce.ps1            (real data)
#                                                        .\reproduce.ps1 -Smoke     (tiny synthetic plumbing check)
param([switch]$Smoke, [switch]$Resume)   # -Resume skips steps whose run already completed
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONHASHSEED = "0"; $env:OMP_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
function StepDone($argv) {
    # a step is done if results\<name>\*\manifest.json exists with status "completed"
    $name = $null
    for ($i = 0; $i -lt $argv.Count; $i++) {
        if ($argv[$i] -eq "-c") { $m = Select-String -Path $argv[$i + 1] -Pattern '^name:\s*(\S+)' | Select-Object -First 1; if ($m) { $name = $m.Matches[0].Groups[1].Value } }
        if ($argv[$i] -eq "--set" -and $argv[$i + 1] -like "name=*") { $name = $argv[$i + 1].Substring(5) }
    }
    if (-not $name) { return $false }
    if ($Smoke) { $name = "SMOKE_$name" }
    foreach ($f in Get-ChildItem -Path "results\$name" -Recurse -Depth 1 -Filter manifest.json -ErrorAction SilentlyContinue) {
        if ((Get-Content $f.FullName -Raw | ConvertFrom-Json).status -eq "completed") { return $true }
    }
    return $false
}
function CG {
    if ($Resume -and (StepDone $args)) { Write-Host "skip (already completed): $($args -join ' ')"; return }
    $a = @("-W", "ignore", "-m", "cgresearch.cli") + $args
    if ($Smoke) { $a += "--smoke" }
    & python @a
    if ($LASTEXITCODE -ne 0) { throw "command failed: $($args -join ' ')" }
}
python -m pytest -q tests
if ($LASTEXITCODE -ne 0) { throw "tests failed" }
CG mutation -c configs/mutation.yaml
CG retrieval -c configs/retrieval.yaml
CG ann -c configs/ann.yaml
function RunDataset($p) {
    CG if-diagnostics -c "configs/${p}_if_diagnostics.yaml"
    CG if-diagnostics -c "configs/${p}_if_diagnostics.yaml" --set dedup=exact --set "name=${p}_if_diagnostics_dedup"
    CG dedup -c "configs/${p}_dedup.yaml"
    CG detection -c "configs/${p}_detection.yaml"
    if (Test-Path "configs/${p}_standardization.yaml") { CG variants -c "configs/${p}_standardization.yaml" }
    if (Test-Path "configs/${p}_retrieval_real.yaml") { CG retrieval-real -c "configs/${p}_retrieval_real.yaml" }
}
if ($Smoke -or (Test-Path "data/UNSW-NB15")) {
    RunDataset "unsw_sampleA"; RunDataset "unsw_sampleB"
    CG robustness -c configs/robustness.yaml
}
if ($Smoke -or (Test-Path "data/CIC-IDS2017")) {
    CG if-diagnostics -c configs/cic2017_if_diagnostics.yaml
    CG detection -c configs/cic2017_detection.yaml
}
python -m cgresearch.cli index --out results
if ($Smoke) { python -m cgresearch.cli report --out results --paper paper --include-smoke } else { python -m cgresearch.cli report --out results --paper paper }
