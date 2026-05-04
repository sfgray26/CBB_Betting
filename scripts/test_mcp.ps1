#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Test MCP server infrastructure for Gemini CLI and Kimi CLI.
.DESCRIPTION
    Verifies env vars, MCP configs, package availability, and basic connectivity.
    Run after setup_mcp_agents.ps1 or after modifying .env.
.NOTES
    File: test_mcp.ps1
    Date: 2026-04-28
#>

$ErrorActionPreference = "Stop"
Write-Host "=== MCP Infrastructure Test ===" -ForegroundColor Cyan
Write-Host ""

# -----------------------------------------------------------------------------
# 1. Load .env
# -----------------------------------------------------------------------------
$loadEnvScript = ".\scripts\load-env.ps1"
if (Test-Path $loadEnvScript) {
    . $loadEnvScript
} else {
    Write-Warning "load-env.ps1 not found. Ensure .env vars are loaded manually."
}

# Set DATABASE_URI alias if needed
if ($env:DATABASE_URL -and -not $env:DATABASE_URI) {
    [Environment]::SetEnvironmentVariable("DATABASE_URI", $env:DATABASE_URL, "Process")
    Write-Host "Set DATABASE_URI from DATABASE_URL" -ForegroundColor DarkGray
}

# -----------------------------------------------------------------------------
# 2. Check required env vars
# -----------------------------------------------------------------------------
Write-Host "--- Environment Variables ---" -ForegroundColor Cyan
$vars = @("DATABASE_URI", "BALLDONTLIE_API_KEY", "RAILWAY_API_TOKEN", "GITHUB_PERSONAL_ACCESS_TOKEN", "CONTEXT7_API_KEY")
foreach ($v in $vars) {
    $val = [Environment]::GetEnvironmentVariable($v, "Process")
    if ($val) {
        Write-Host "  $v : SET" -ForegroundColor Green
    } else {
        Write-Host "  $v : NOT SET (optional unless noted)" -ForegroundColor Yellow
    }
}
Write-Host ""

# -----------------------------------------------------------------------------
# 3. Check Kimi MCP servers
# -----------------------------------------------------------------------------
Write-Host "--- Kimi MCP Servers ---" -ForegroundColor Cyan
kimi mcp list 2>&1
Write-Host ""

# -----------------------------------------------------------------------------
# 4. Check Gemini settings.json
# -----------------------------------------------------------------------------
Write-Host "--- Gemini MCP Config ---" -ForegroundColor Cyan
$geminiSettings = ".gemini/settings.json"
if (Test-Path $geminiSettings) {
    try {
        $json = Get-Content $geminiSettings | ConvertFrom-Json
        if ($json.mcpServers) {
            Write-Host "  mcpServers found:" -ForegroundColor Green
            $json.mcpServers.PSObject.Properties | ForEach-Object {
                Write-Host "    $($_.Name) : configured" -ForegroundColor Green
            }
        } else {
            Write-Host "  WARNING: no mcpServers section" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  ERROR: Invalid JSON in $geminiSettings" -ForegroundColor Red
    }
} else {
    Write-Host "  ERROR: $geminiSettings not found" -ForegroundColor Red
}
Write-Host ""

# -----------------------------------------------------------------------------
# 5. Check Docker availability
# -----------------------------------------------------------------------------
Write-Host "--- Docker ---" -ForegroundColor Cyan
try {
    $dockerInfo = docker info --format "{{.ServerVersion}}" 2>$null
    if ($dockerInfo) {
        Write-Host "  Docker: $dockerInfo" -ForegroundColor Green
    } else {
        Write-Host "  Docker: not running" -ForegroundColor Red
    }
} catch {
    Write-Host "  Docker: not available" -ForegroundColor Red
}
Write-Host ""

# -----------------------------------------------------------------------------
# 6. Check Context7 reachability
# -----------------------------------------------------------------------------
Write-Host "--- Context7 Endpoint ---" -ForegroundColor Cyan
try {
    $resp = Invoke-WebRequest -Uri "https://mcp.context7.com/mcp" -Method POST -Body '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' -ContentType "application/json" -TimeoutSec 15 -UseBasicParsing
    Write-Host "  Context7: HTTP $($resp.StatusCode) OK" -ForegroundColor Green
} catch {
    $status = if ($_.Exception.Response) { $_.Exception.Response.StatusCode.Value__ } else { "N/A" }
    if ($status -eq 406 -or $status -eq 401 -or $status -eq 405) {
        Write-Host "  Context7: HTTP $status (endpoint reachable, auth may be required)" -ForegroundColor Green
    } else {
        Write-Host "  Context7: HTTP $status" -ForegroundColor Yellow
    }
}
Write-Host ""

# -----------------------------------------------------------------------------
# 7. Check npm packages
# -----------------------------------------------------------------------------
Write-Host "--- npm Package Cache ---" -ForegroundColor Cyan
$packages = @(
    "@modelcontextprotocol/server-sequential-thinking",
    "@balldontlie/mcp-server",
    "@railway/mcp-server"
)
foreach ($pkg in $packages) {
    $cached = npm cache ls $pkg 2>$null
    if ($LASTEXITCODE -eq 0 -and $cached) {
        Write-Host "  $pkg : cached" -ForegroundColor Green
    } else {
        Write-Host "  $pkg : not cached (will download on first use)" -ForegroundColor Yellow
    }
}
Write-Host ""

# -----------------------------------------------------------------------------
# 8. Interactive test commands
# -----------------------------------------------------------------------------
Write-Host "=== Interactive Test Commands ===" -ForegroundColor Cyan
Write-Host "Run these manually to verify end-to-end MCP functionality:" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  # Kimi CLI"
Write-Host "  kimi"
Write-Host "  @postgres-audit List tables"
Write-Host "  @balldontlie Get today's MLB games"
Write-Host "  @sequential-thinking Think through a complex problem"
Write-Host ""
Write-Host "  # Gemini CLI"
Write-Host "  gemini"
Write-Host "  @railway List services"
Write-Host "  @postgres-readonly List tables in public schema"
Write-Host ""
Write-Host "=== Test Complete ===" -ForegroundColor Cyan
