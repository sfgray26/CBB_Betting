#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Ensure the Python virtual environment matches requirements.txt for local testing.
.DESCRIPTION
    Creates or refreshes the venv at ./venv, installs all packages from requirements.txt,
    and runs a quick smoke test to confirm key test dependencies are importable.

    Run this once after pulling the repo, or after any change to requirements.txt.

    Usage:
        .\scripts\setup_test_env.ps1          # normal setup
        .\scripts\setup_test_env.ps1 -Force   # delete and recreate the venv from scratch
.NOTES
    File: scripts/setup_test_env.ps1
    Requires: Python 3.11+ on PATH (or venv already at ./venv)
#>
[CmdletBinding()]
param(
    [switch]$Force   # Wipe and recreate the venv before installing
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

Push-Location $RepoRoot
try {
    # ── 1. Optionally nuke existing venv ─────────────────────────────────────
    if ($Force -and (Test-Path "venv")) {
        Write-Host "[setup_test_env] -Force: removing existing venv..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force "venv"
    }

    # ── 2. Create venv if it doesn't exist ───────────────────────────────────
    if (-not (Test-Path "venv\Scripts\python.exe")) {
        Write-Host "[setup_test_env] Creating virtual environment..." -ForegroundColor Cyan
        python -m venv venv
        if ($LASTEXITCODE -ne 0) { throw "python -m venv failed" }
    } else {
        Write-Host "[setup_test_env] Existing venv found at ./venv" -ForegroundColor Green
    }

    # ── 3. Upgrade pip silently ───────────────────────────────────────────────
    Write-Host "[setup_test_env] Upgrading pip..." -ForegroundColor Cyan
    & "venv\Scripts\python.exe" -m pip install --upgrade pip --quiet

    # ── 4. Install requirements.txt ──────────────────────────────────────────
    Write-Host "[setup_test_env] Installing requirements.txt..." -ForegroundColor Cyan
    & "venv\Scripts\python.exe" -m pip install -r requirements.txt --quiet
    if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt failed" }
    Write-Host "[setup_test_env] requirements.txt installed OK" -ForegroundColor Green

    # ── 5. Smoke-test critical test dependencies ──────────────────────────────
    $CriticalPackages = @("requests", "apscheduler", "pytest", "sqlalchemy", "pydantic", "fastapi")
    $Missing = @()
    foreach ($pkg in $CriticalPackages) {
        $check = & "venv\Scripts\python.exe" -c "import $pkg" 2>&1
        if ($LASTEXITCODE -ne 0) {
            $Missing += $pkg
            Write-Host "  MISSING: $pkg" -ForegroundColor Red
        } else {
            Write-Host "  OK:      $pkg" -ForegroundColor Green
        }
    }

    if ($Missing.Count -gt 0) {
        throw "Missing packages after install: $($Missing -join ', '). Check requirements.txt."
    }

    # ── 6. Quick pytest smoke test ────────────────────────────────────────────
    Write-Host "[setup_test_env] Running smoke tests..." -ForegroundColor Cyan
    & "venv\Scripts\python.exe" -m pytest tests\test_injury_overlay.py tests\test_fantasy_budget.py -q --tb=short
    if ($LASTEXITCODE -ne 0) { throw "Smoke tests failed — check output above." }

    Write-Host ""
    Write-Host "[setup_test_env] Environment is ready. Run tests with:" -ForegroundColor Green
    Write-Host "  .\venv\Scripts\python -m pytest tests\ -q --tb=short" -ForegroundColor White
} finally {
    Pop-Location
}
