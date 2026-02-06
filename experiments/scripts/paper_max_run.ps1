# Paper MAX Run Script - PowerShell
# Runs MPNet encoder ablation with seeds 1,2 (seed 0 already done)
#
# Usage: ./paper_max_run.ps1 [-OutputDir <path>]

param(
    [string]$OutputDir = "experiments/reports/encoder_ablation"
)

$ErrorActionPreference = "Continue"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "$OutputDir/logs"
$runDir = "$OutputDir/runs"

# Create directories
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

Write-Host "=== Paper MAX Run Script ==="
Write-Host "Timestamp: $timestamp"
Write-Host "Log directory: $logDir"
Write-Host ""

# Save environment info
$envInfo = @{
    timestamp = $timestamp
    python_version = (python --version 2>&1)
    git_commit = (git rev-parse HEAD 2>&1)
    hostname = $env:COMPUTERNAME
}
$envInfo | ConvertTo-Json | Out-File "$runDir/env_info.json" -Encoding UTF8

# MPNet Seed 1
Write-Host "[1/2] Running MPNet seed 1..."
$seed1Log = "$logDir/mpnet_seed1.log"
python -m experiments.run dataset=unified_v3 embedding=sbert_mpnet lenses=identity seed=1 "outputs.run_id=encoder_mpnet_s1" "outputs.base_dir=experiments/reports/encoder_ablation" 2>&1 | Tee-Object -FilePath $seed1Log

if (Test-Path "experiments/reports/encoder_ablation/encoder_mpnet_s1/results.json") {
    Write-Host "[OK] MPNet seed 1 completed"
} else {
    Write-Host "[WARN] MPNet seed 1 may have issues - check log"
}

# MPNet Seed 2
Write-Host "[2/2] Running MPNet seed 2..."
$seed2Log = "$logDir/mpnet_seed2.log"
python -m experiments.run dataset=unified_v3 embedding=sbert_mpnet lenses=identity seed=2 "outputs.run_id=encoder_mpnet_s2" "outputs.base_dir=experiments/reports/encoder_ablation" 2>&1 | Tee-Object -FilePath $seed2Log

if (Test-Path "experiments/reports/encoder_ablation/encoder_mpnet_s2/results.json") {
    Write-Host "[OK] MPNet seed 2 completed"
} else {
    Write-Host "[WARN] MPNet seed 2 may have issues - check log"
}

Write-Host ""
Write-Host "=== MPNet Ablation Complete ==="
Write-Host "Check logs in: $logDir"

# Aggregate results if all seeds available
Write-Host ""
Write-Host "Aggregating encoder ablation results..."

$mpnetResults = @()
foreach ($seed in 0, 1, 2) {
    $resultsPath = "experiments/reports/encoder_ablation/encoder_mpnet"
    if ($seed -gt 0) {
        $resultsPath = "experiments/reports/encoder_ablation/encoder_mpnet_s$seed"
    }
    $resultsFile = "$resultsPath/results.json"
    if (Test-Path $resultsFile) {
        $data = Get-Content $resultsFile | ConvertFrom-Json
        $mpnetResults += @{
            seed = $seed
            pareto_count = $data.pareto_count
        }
        Write-Host "  Seed $seed: $($data.pareto_count) Pareto"
    } else {
        Write-Host "  Seed $seed: NOT FOUND"
    }
}

if ($mpnetResults.Count -eq 3) {
    $counts = $mpnetResults | ForEach-Object { $_.pareto_count }
    $mean = ($counts | Measure-Object -Average).Average
    $std = [math]::Sqrt(($counts | ForEach-Object { [math]::Pow($_ - $mean, 2) } | Measure-Object -Sum).Sum / $counts.Count)
    Write-Host ""
    Write-Host "MPNet (3 seeds): $mean +/- $std"
} else {
    Write-Host ""
    Write-Host "WARN: Not all 3 seeds available"
}

Write-Host ""
Write-Host "Done."
