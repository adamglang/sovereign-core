# Hard reset script for Sovereign Core venv
# Nukes venv, recreates, installs everything, fixes onnxruntime conflict

$ErrorActionPreference = "Stop"

Write-Host "=== HARD RESET: Sovereign Core venv ===" -ForegroundColor Red
Write-Host "This will DELETE and recreate the virtual environment." -ForegroundColor Yellow
Write-Host ""

# Deactivate venv if active
Write-Host "Deactivating any active venv..." -ForegroundColor Cyan
try { deactivate } catch {}

# Kill any Python processes that might lock venv
Write-Host "Checking for Python processes..." -ForegroundColor Cyan
$pythonProcs = Get-Process python -ErrorAction SilentlyContinue
if ($pythonProcs) {
    Write-Host "WARNING: Found running Python processes. Close VSCode/terminals first!" -ForegroundColor Red
    $pythonProcs | Format-Table Id, ProcessName, Path
    $response = Read-Host "Continue anyway? This may fail if venv is locked (y/N)"
    if ($response -ne 'y' -and $response -ne 'Y') {
        Write-Host "Aborted. Close VSCode and terminals, then try again." -ForegroundColor Yellow
        exit 1
    }
}

# Delete venv directory
if (Test-Path ".\venv") {
    Write-Host "Deleting venv directory..." -ForegroundColor Yellow
    try {
        Remove-Item -Recurse -Force .\venv -ErrorAction Stop
        Write-Host "Done: venv deleted" -ForegroundColor Green
    } catch {
        Write-Host "FAILED to delete venv: $_" -ForegroundColor Red
        Write-Host "" -ForegroundColor Yellow
        Write-Host "To fix:" -ForegroundColor Yellow
        Write-Host "  1. Close VSCode completely" -ForegroundColor Gray
        Write-Host "  2. Manually delete the venv folder" -ForegroundColor Gray
        Write-Host "  3. Reopen VSCode and run this script again" -ForegroundColor Gray
        exit 1
    }
} else {
    Write-Host "Done: No existing venv found" -ForegroundColor Green
}

# Create fresh venv
Write-Host "" 
Write-Host "Creating fresh virtual environment..." -ForegroundColor Cyan
python -m venv venv
Write-Host "Done: venv created" -ForegroundColor Green

# Activate venv
Write-Host ""
Write-Host "Activating venv..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip | Out-Null
Write-Host "Done: pip upgraded" -ForegroundColor Green

# Install dependencies
Write-Host ""
Write-Host "Installing project dependencies..." -ForegroundColor Cyan
Write-Host "(This will take a few minutes...)" -ForegroundColor Gray
pip install -e . --no-warn-script-location | Out-Null
Write-Host "Done: Dependencies installed" -ForegroundColor Green

# Fix onnxruntime conflict by reinstalling GPU version
Write-Host ""
Write-Host "Fixing onnxruntime/onnxruntime-gpu conflict..." -ForegroundColor Yellow
pip uninstall -y onnxruntime onnxruntime-gpu | Out-Null
pip install onnxruntime-gpu>=1.16.0 | Out-Null
Write-Host "Done: onnxruntime-gpu reinstalled" -ForegroundColor Green

# Verify GPU support
Write-Host ""
Write-Host "Verifying GPU support..." -ForegroundColor Cyan
$providers = python -c "import onnxruntime; print(','.join(onnxruntime.get_available_providers()))"

if ($providers -match "CUDA") {
    Write-Host "SUCCESS: CUDA ENABLED - GPU acceleration working!" -ForegroundColor Green
    Write-Host "  Providers: $providers" -ForegroundColor Gray
} else {
    Write-Host "WARNING: CUDA NOT AVAILABLE" -ForegroundColor Red
    Write-Host "  Providers: $providers" -ForegroundColor Gray
    Write-Host ""
    Write-Host "CUDA Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Verify CUDA 12.6 bin directory is in PATH" -ForegroundColor Gray
    Write-Host "  2. Check DLL exists: Get-Command cublasLt64_12.dll" -ForegroundColor Gray
    Write-Host "  3. Restart terminal to pick up PATH changes" -ForegroundColor Gray
}

Write-Host ""
Write-Host "=== RESET COMPLETE ===" -ForegroundColor Green
Write-Host "venv is activated and ready to use!" -ForegroundColor Cyan
Write-Host ""
Write-Host "To test: python -m sovereign_core.main" -ForegroundColor Cyan
