# Installs OpenJet without a system Python or pipx (Windows PowerShell):
#
#   irm https://github.com/L-Forster/open-jet/releases/latest/download/install.ps1 | iex
#
# uv downloads its own Python for OpenJet, separate from the system's. $env:OPENJET_VERSION pins a
# release (for example $env:OPENJET_VERSION = "0.5.4"). Errors throw instead of exit, so the
# PowerShell window that ran it stays open.
$ErrorActionPreference = "Stop"

$PythonVersion = "3.12"
$BinDir = Join-Path $env:USERPROFILE ".local\bin"

$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    $uv = Join-Path $BinDir "uv.exe"
    if (-not (Test-Path $uv)) {
        Write-Host "Installing uv, which provides OpenJet's Python..."
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    }
}

$spec = "open-jet"
if ($env:OPENJET_VERSION) { $spec = "open-jet==$($env:OPENJET_VERSION)" }

Write-Host "Installing $spec with Python $PythonVersion..."
& $uv tool install --force --python $PythonVersion $spec
if ($LASTEXITCODE -ne 0) { throw "uv tool install failed (exit code $LASTEXITCODE)" }
& $uv tool update-shell *> $null

Write-Host ""
Write-Host "OpenJet is installed. Open a new terminal, then run:"
Write-Host ""
Write-Host "  openjet setup"
Write-Host ""
Write-Host "If openjet is not found, run it as $BinDir\openjet.exe."
