# Run pytest with PYTHONPATH set to packages directory
# Usage: .\scripts\test.ps1 [pytest args...]

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

$env:PYTHONPATH = "$ProjectRoot\packages;$env:PYTHONPATH"

Set-Location $ProjectRoot
pytest -q @args
