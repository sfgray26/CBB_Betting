#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Load environment variables from the project's .env file into the current PowerShell session.
.DESCRIPTION
    Dot-source this script to load .env variables into your current shell so agents (Claude, Gemini, Kimi) inherit them.
    Usage: . .\scripts\load-env.ps1
    Or add to your PowerShell profile for automatic loading.
.NOTES
    This only modifies the current process environment, not system/user env vars.
    Safe to run multiple times — existing env vars are NOT overwritten by default.
#>

param(
    [switch]$Force,  # Overwrite existing env vars
    [string]$EnvFile = ".env"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")
$envPath = Join-Path $projectRoot $EnvFile

if (-not (Test-Path $envPath)) {
    Write-Error ".env file not found at: $envPath"
    return
}

$loaded = 0
$skipped = 0

Get-Content $envPath | ForEach-Object {
    $line = $_.Trim()
    # Skip comments and empty lines
    if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith("#")) {
        return
    }
    # Match KEY=VALUE (handle = in value)
    if ($line -match '^([^=]+)=(.*)$') {
        $name = $matches[1].Trim()
        $value = $matches[2].Trim()

        # Remove surrounding quotes if present
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        if ([string]::IsNullOrEmpty($value)) {
            return  # Skip empty values
        }

        $existing = [Environment]::GetEnvironmentVariable($name, "Process")
        if ($Force -or [string]::IsNullOrEmpty($existing)) {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
            $loaded++
        } else {
            $skipped++
        }
    }
}

Write-Host "Loaded $loaded env var(s) from $EnvFile" -ForegroundColor Green
if ($skipped -gt 0) {
    Write-Host "Skipped $skipped already-set var(s). Use -Force to overwrite." -ForegroundColor DarkGray
}
