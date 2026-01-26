# BURN Script Wrapper - RopeCap Multi-Pose Processor
# Single camera + multiple poses -> single COLMAP DB
#
# Usage:
#   .\BURN-Scripts\BURN_RopeCapMultiPose.ps1 -ConfigFile configs\ropecap_multipose.yaml
#   .\BURN-Scripts\BURN_RopeCapMultiPose.ps1 -ConfigFile configs\ropecap_multipose.yaml -DryRun

param(
    [Parameter(Mandatory=$true)]
    [string]$ConfigFile,
    
    [Parameter(Mandatory=$false)]
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# Resolve paths
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$TemplateScript = Join-Path $ScriptDir "BURN_template_RopeCapMultiPose.py"

# Resolve config path (relative to repo root if not absolute)
if (-not [System.IO.Path]::IsPathRooted($ConfigFile)) {
    $ConfigFile = Join-Path $RepoRoot $ConfigFile
}

# Validate
if (-not (Test-Path $ConfigFile)) {
    Write-Error "Config file not found: $ConfigFile"
    exit 1
}

if (-not (Test-Path $TemplateScript)) {
    Write-Error "Template script not found: $TemplateScript"
    exit 1
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "BURN - RopeCap Multi-Pose Processor" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Config: $ConfigFile"
Write-Host "Template: $TemplateScript"
Write-Host ""

if ($DryRun) {
    Write-Host "[DRY RUN] Would execute:" -ForegroundColor Yellow
    Write-Host "  python `"$TemplateScript`" --config `"$ConfigFile`""
    exit 0
}

# Activate conda and run
Push-Location $RepoRoot
try {
    Write-Host "Starting processing..." -ForegroundColor Green
    Write-Host ""
    
    # Run Python script
    & python $TemplateScript --config $ConfigFile
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Processing failed with exit code: $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Processing completed successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
}
finally {
    Pop-Location
}
