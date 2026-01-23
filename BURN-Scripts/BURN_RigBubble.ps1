# =============================================================================
# BURN Script Template - RopeCap Rig-Bubble Processor (PowerShell)
# =============================================================================
# Processes groups with rig-bubbles (multiple cameras combined per bubble).
#
# Usage:
#   .\BURN-Scripts\BURN_RigBubble.ps1 -ConfigFile configs\ropecap_rigbubble_20260121.yaml
#
# Or run directly:
#   python BURN-Scripts\BURN_template_RigBubble.py --config configs\ropecap_rigbubble_20260121.yaml
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
    Write-Host "BURN Script - RopeCap Rig-Bubble Processor"
    Write-Host "========================================"
    Write-Host "Config file: $ConfigPath"
    Write-Host "Root dir: $RootDir"
    Write-Host "========================================"
    Write-Host ""
    
    if ($DryRun) {
        Write-Host "[DRY RUN] Would execute:"
        Write-Host "  python BURN-Scripts\BURN_template_RigBubble.py --config `"$ConfigPath`""
    } else {
        # Run the Python script
        $pythonCmd = "python"
        $scriptPath = Join-Path $ScriptDir "BURN_template_RigBubble.py"
        
        Write-Host "Starting rig-bubble processing..."
        Write-Host ""
        
        & $pythonCmd $scriptPath --config $ConfigPath
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Processing failed with exit code: $LASTEXITCODE"
            exit $LASTEXITCODE
        }
        
        Write-Host ""
        Write-Host "========================================"
        Write-Host "Rig-bubble processing completed successfully!"
        Write-Host "========================================"
    }
    
} finally {
    Pop-Location
}
