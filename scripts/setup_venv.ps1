# Setup script for Sovereign Core (PowerShell)
# Handles the onnxruntime/onnxruntime-gpu conflict automatically

param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

Write-Host "=== Sovereign Core Environment Setup ===" -ForegroundColor Cyan

# Check if venv exists
if (Test-Path ".\venv") {
    if ($Force) {
        Write-Host "Removing existing venv..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force .\venv -ErrorAction SilentlyContinue
    } else {
        Write-Host "Virtual environment already exists. Use -Force to recreate." -ForegroundColor Yellow
        Write-Host "To activate: .\venv\Scripts\Activate.ps1" -ForegroundColor Green
        exit 0
    }
}

# Create venv
Write-Host "Creating virtual environment..." -ForegroundColor Cyan
python -m venv venv

# Activate venv
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install dependencies
Write-Host "Installing project dependencies..." -ForegroundColor Cyan
pip install -e .

# Fix onnxruntime conflict
Write-Host "`nFixing onnxruntime/onnxruntime-gpu conflict..." -ForegroundColor Yellow
Write-Host "Uninstalling CPU-only onnxruntime..." -ForegroundColor Cyan
pip uninstall -y onnxruntime

# Verify GPU support
Write-Host "`nVerifying CUDA support..." -ForegroundColor Cyan
$providers = python -c "import onnxruntime; print(' '.join(onnxruntime.get_available_providers()))"

if ($providers -match "CUDA") {
    Write-Host "✓ CUDA support enabled!" -ForegroundColor Green
    Write-Host "  Available providers: $providers" -ForegroundColor Gray
} else {
    Write-Host "✗ WARNING: CUDA not available" -ForegroundColor Red
    Write-Host "  Available providers: $providers" -ForegroundColor Gray
    Write-Host "`nTroubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Ensure CUDA 12.6 bin directory is in PATH" -ForegroundColor Gray
    Write-Host "  2. Restart terminal after PATH changes" -ForegroundColor Gray
    Write-Host "  3. Verify: where cublasLt64_12.dll" -ForegroundColor Gray
}

Write-Host "`n=== Setup Complete ===" -ForegroundColor Green
Write-Host "To activate venv: .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "To run Sovereign: python -m sovereign_core.main" -ForegroundColor Cyan
