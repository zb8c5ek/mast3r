# =============================================================================
# BURN Script Template - RopeCap Group Processor (PowerShell)
# =============================================================================
# This is a template .ps1 file for running the RopeCap group processor.
#
# Usage:
#   .\BURN-Scripts\BURN_RopeCapGroup.ps1 -ConfigFile configs\ropecap_20260119_163000.yaml
#
# Or run directly:
#   python BURN-Scripts\BURN_template_RopeCapGroup.py --config configs\ropecap_20260119_163000.yaml
# =============================================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$ConfigFile,
    
    [Parameter(Mandatory=$false)]
    [switch]$DryRun
)

# Script settings
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir

# Change to root directory
Push-Location $RootDir

try {
    # Validate config file exists
    if (-not (Test-Path $ConfigFile)) {
        Write-Error "Config file not found: $ConfigFile"
        exit 1
    }
    
    $ConfigPath = Resolve-Path $ConfigFile
    
    Write-Host "========================================"
    Write-Host "BURN Script - RopeCap Group Processor"
    Write-Host "========================================"
    Write-Host "Config file: $ConfigPath"
    Write-Host "Root dir: $RootDir"
    Write-Host "========================================"
    Write-Host ""
    
    if ($DryRun) {
        Write-Host "[DRY RUN] Would execute:"
        Write-Host "  python BURN-Scripts\BURN_template_RopeCapGroup.py --config `"$ConfigPath`""
    } else {
        # Run the Python script
        $pythonCmd = "python"
        $scriptPath = Join-Path $ScriptDir "BURN_template_RopeCapGroup.py"
        
        Write-Host "Starting processing..."
        Write-Host ""
        
        & $pythonCmd $scriptPath --config $ConfigPath
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Processing failed with exit code: $LASTEXITCODE"
            exit $LASTEXITCODE
        }
        
        Write-Host ""
        Write-Host "========================================"
        Write-Host "Processing completed successfully!"
        Write-Host "========================================"
    }
    
} finally {
    Pop-Location
}
