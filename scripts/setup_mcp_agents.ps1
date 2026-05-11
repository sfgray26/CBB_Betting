#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Setup MCP servers and skills for Gemini CLI and Kimi CLI.
.DESCRIPTION
    Configures MCP servers for both agents and verifies the setup.
    Run this after pulling the repo on a new machine or after updating MCP configs.
    Requires: Node.js, Docker, npx, railway CLI (authenticated), kimi CLI, gemini CLI
.NOTES
    File: setup_mcp_agents.ps1
    Date: 2026-04-28
#>

$ErrorActionPreference = "Stop"
Write-Host "=== CBB Edge MCP Agent Setup ===" -ForegroundColor Cyan
Write-Host ""

# -----------------------------------------------------------------------------
# 1. Load environment variables from .env if present
# -----------------------------------------------------------------------------
$loadEnvScript = ".\scripts\load-env.ps1"
if (Test-Path $loadEnvScript) {
    Write-Host "Loading .env via load-env.ps1..." -ForegroundColor DarkGray
    . $loadEnvScript
} elseif (Test-Path .env) {
    Write-Host "Loading .env (inline fallback)..." -ForegroundColor DarkGray
    Get-Content .env | ForEach-Object {
        if ($_ -match '^([^#][^=]*)=(.*)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

# -----------------------------------------------------------------------------
# 2. Check prerequisites
# -----------------------------------------------------------------------------
Write-Host "Checking prerequisites..." -ForegroundColor DarkGray
$tools = @("npx", "docker", "railway", "kimi", "gemini")
$missing = @()
foreach ($tool in $tools) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        $missing += $tool
    }
}
if ($missing.Count -gt 0) {
    Write-Host "MISSING TOOLS: $($missing -join ', ')" -ForegroundColor Red
    Write-Host "Please install missing tools before continuing." -ForegroundColor Red
    exit 1
}
Write-Host "All prerequisites found." -ForegroundColor Green
Write-Host ""

# -----------------------------------------------------------------------------
# 3. Setup Kimi CLI MCP servers
# -----------------------------------------------------------------------------
Write-Host "--- Kimi CLI MCP Setup ---" -ForegroundColor Cyan

# Railway MCP (uses Railway CLI auth)
Write-Host "Adding Railway MCP..." -ForegroundColor DarkGray
kimi mcp add --transport stdio railway -- npx -y @railway/mcp-server 2>$null | Out-Null

# PostgreSQL MCP (read-only)
$localDbUri = "postgresql://postgres:zinedinezidane1%24@127.0.0.1:5432/cbb_edge"
if ($env:DATABASE_URI) { $localDbUri = $env:DATABASE_URI }
elseif ($env:DATABASE_URL) { $localDbUri = $env:DATABASE_URL }
Write-Host "Adding PostgreSQL MCP (read-only)..." -ForegroundColor DarkGray
$env:DATABASE_URI = $localDbUri
kimi mcp add --transport stdio postgres-audit -- docker run -i --rm -e DATABASE_URI crystaldba/postgres-mcp --access-mode=restricted 2>$null | Out-Null

# BallDontLie MCP (hosted HTTP endpoint; research/ad-hoc only, not production ingestion)
if ($env:BALLDONTLIE_API_KEY) {
    Write-Host "Adding BallDontLie MCP..." -ForegroundColor DarkGray
    kimi mcp add --transport http balldontlie https://mcp.balldontlie.io/mcp --header "Authorization: $($env:BALLDONTLIE_API_KEY)" 2>$null | Out-Null
} else {
    Write-Host "Skipping BallDontLie MCP: BALLDONTLIE_API_KEY not set" -ForegroundColor Yellow
}

# Sequential Thinking MCP
Write-Host "Adding Sequential Thinking MCP..." -ForegroundColor DarkGray
kimi mcp add --transport stdio sequential-thinking -- npx -y @modelcontextprotocol/server-sequential-thinking 2>$null | Out-Null

# Context7 MCP
if ($env:CONTEXT7_API_KEY) {
    Write-Host "Adding Context7 MCP..." -ForegroundColor DarkGray
    kimi mcp add --transport http context7 https://mcp.context7.com/mcp --header "CONTEXT7_API_KEY: $($env:CONTEXT7_API_KEY)" 2>$null | Out-Null
} else {
    Write-Host "Adding Context7 MCP (free tier, no API key)..." -ForegroundColor DarkGray
    kimi mcp add --transport http context7 https://mcp.context7.com/mcp 2>$null | Out-Null
}

Write-Host "Kimi MCP servers configured." -ForegroundColor Green
Write-Host ""

# -----------------------------------------------------------------------------
# 4. Verify Kimi CLI setup
# -----------------------------------------------------------------------------
Write-Host "Verifying Kimi MCP servers..." -ForegroundColor DarkGray
kimi mcp list
Write-Host ""

# -----------------------------------------------------------------------------
# 5. Setup Gemini CLI MCP servers
# -----------------------------------------------------------------------------
Write-Host "--- Gemini CLI MCP Setup ---" -ForegroundColor Cyan
Write-Host "Gemini CLI uses .gemini/settings.json (already updated in repo)." -ForegroundColor DarkGray
Write-Host "Ensure RAILWAY_API_TOKEN and DATABASE_URI are set in your shell." -ForegroundColor DarkGray
Write-Host ""

# -----------------------------------------------------------------------------
# 6. Verify Gemini CLI setup
# -----------------------------------------------------------------------------
Write-Host "Verifying Gemini CLI auth..." -ForegroundColor DarkGray
$railwayAuth = railway whoami 2>&1
if ($railwayAuth -match "Logged in") {
    Write-Host "Railway CLI: AUTHENTICATED" -ForegroundColor Green
} else {
    Write-Host "Railway CLI: NOT AUTHENTICATED — run 'railway login'" -ForegroundColor Red
}
Write-Host ""

# -----------------------------------------------------------------------------
# 7. Check skills directories
# -----------------------------------------------------------------------------
Write-Host "--- Skills Inventory ---" -ForegroundColor Cyan
$geminiSkills = Get-ChildItem .gemini/skills -Directory -ErrorAction SilentlyContinue | ForEach-Object { $_.Name }
$kimiSkills = Get-ChildItem .kimi/skills -Directory -ErrorAction SilentlyContinue | ForEach-Object { $_.Name }

Write-Host "Gemini skills: $($geminiSkills -join ', ')" -ForegroundColor Green
Write-Host "Kimi skills: $($kimiSkills -join ', ')" -ForegroundColor Green
Write-Host ""

# -----------------------------------------------------------------------------
# 8. Remaining manual steps
# -----------------------------------------------------------------------------
Write-Host "=== Remaining Manual Steps ===" -ForegroundColor Cyan
$remaining = @()

if (-not $env:RAILWAY_API_TOKEN) {
    $remaining += "Set RAILWAY_API_TOKEN environment variable (get from Railway dashboard -> Account Settings -> Tokens)"
}
if (-not $env:GITHUB_PERSONAL_ACCESS_TOKEN) {
    $remaining += "Set GITHUB_PERSONAL_ACCESS_TOKEN for GitHub MCP (optional, create at https://github.com/settings/tokens)"
}
if (-not $env:CONTEXT7_API_KEY) {
    $remaining += "Set CONTEXT7_API_KEY for higher rate limits (optional, get free key at https://context7.com/dashboard)"
}
if (-not $env:BALLDONTLIE_API_KEY) {
    $remaining += "Set BALLDONTLIE_API_KEY for @balldontlie MCP research tools"
}

if ($remaining.Count -eq 0) {
    Write-Host "All required configuration complete!" -ForegroundColor Green
} else {
    foreach ($item in $remaining) {
        Write-Host "  - $item" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Cyan
Write-Host "Test Gemini:  gemini -> '@railway List services'" -ForegroundColor DarkGray
Write-Host "Test Kimi:    kimi -> '@postgres-audit List tables'" -ForegroundColor DarkGray
Write-Host "Test BDL:     @balldontlie Get today's MLB games" -ForegroundColor DarkGray
