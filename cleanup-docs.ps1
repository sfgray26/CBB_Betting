# Documentation Cleanup Script
# Generated: April 22, 2026
# Total files to remove: 97 (29% reduction from 333 to 236 files)
# See DOCUMENTATION_CLEANUP_RECOMMENDATION.md for full analysis

param(
    [string]$Phase = "all",
    [switch]$DryRun = $false,
    [switch]$Verbose = $false
)

$ErrorActionPreference = "Stop"
$removed = 0
$errors = 0

function Remove-SafeFile {
    param([string]$Path)
    
    if (-not (Test-Path $Path)) {
        if ($Verbose) { Write-Host "⚠️  SKIP: $Path (not found)" -ForegroundColor Yellow }
        return $false
    }
    
    if ($DryRun) {
        Write-Host "🔍 DRY RUN: Would remove $Path" -ForegroundColor Cyan
        return $true
    }
    
    try {
        Remove-Item -Path $Path -Force -ErrorAction Stop
        Write-Host "✅ Removed: $Path" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "❌ ERROR: $Path - $_" -ForegroundColor Red
        $script:errors++
        return $false
    }
}

function Remove-SafeDirectory {
    param([string]$Path)
    
    if (-not (Test-Path $Path)) {
        if ($Verbose) { Write-Host "⚠️  SKIP: $Path (not found)" -ForegroundColor Yellow }
        return $false
    }
    
    if ($DryRun) {
        $count = (Get-ChildItem -Path $Path -Recurse | Measure-Object).Count
        Write-Host "🔍 DRY RUN: Would remove $Path ($count items)" -ForegroundColor Cyan
        return $true
    }
    
    try {
        Remove-Item -Path $Path -Recurse -Force -ErrorAction Stop
        Write-Host "✅ Removed directory: $Path" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "❌ ERROR: $Path - $_" -ForegroundColor Red
        $script:errors++
        return $false
    }
}

Write-Host "`n=== DOCUMENTATION CLEANUP SCRIPT ===" -ForegroundColor Magenta
Write-Host "Phase: $Phase | Dry Run: $DryRun | Verbose: $Verbose`n" -ForegroundColor Magenta

# Phase 1: Root Duplicate Prompts (14 files)
if ($Phase -eq "all" -or $Phase -eq "1") {
    Write-Host "`n--- PHASE 1: Root Duplicate Prompts (14 files) ---" -ForegroundColor Yellow
    
    $files = @(
        "CLAUDE_ARCHITECT_PROMPT_MARCH28.md",
        "CLAUDE_FANTASY_ROADMAP_PROMPT.md",
        "CLAUDE_GEMINI_SKILLS_PROMPT.md",
        "CLAUDE_K33_MCP_DELEGATION_PROMPT.md",
        "CLAUDE_K34_K38_KIMI_DELEGATION.md",
        "CLAUDE_KIMI_DELEGATION.md",
        "CLAUDE_LOCAL_LLM_PROMPT.md",
        "CLAUDE_OPENCLAW_ORCHESTRATION_PROMPT.md",
        "CLAUDE_PHASE0_IMPLEMENTATION_PROMPT.md",
        "CLAUDE_PHASE1_IMPLEMENTATION_PROMPT.md",
        "CLAUDE_PHASE2_IMPLEMENTATION_PROMPT.md",
        "CLAUDE_PHASE3_IMPLEMENTATION_PROMPT.md",
        "CLAUDE_RETURN_PROMPT.md",
        "CLAUDE_TEAM_COORDINATION_PROMPT.md",
        "CLAUDE_UAT_FIXES_PROMPT.md",
        "CLAUDE_UI_CONTRACT_REORIENTATION_PROMPT.md",
        "CLAUDE_UI_UX_ARCHITECT_PROMPT.md"
    )
    
    foreach ($file in $files) {
        if (Remove-SafeFile $file) { $script:removed++ }
    }
}

# Phase 2: CBB Task Directories (12 files)
if ($Phase -eq "all" -or $Phase -eq "2") {
    Write-Host "`n--- PHASE 2: CBB Task Directories (12 files) ---" -ForegroundColor Yellow
    
    $dirs = @(".agent_tasks", ".openclaw")
    
    foreach ($dir in $dirs) {
        if (Remove-SafeDirectory $dir) { $script:removed++ }
    }
}

# Phase 3: Old UAT Findings (10 files)
if ($Phase -eq "all" -or $Phase -eq "3") {
    Write-Host "`n--- PHASE 3: Old UAT Findings (10 files) ---" -ForegroundColor Yellow
    
    $files = @(
        "tasks\uat_findings.md",
        "tasks\uat_findings_fresh.md",
        "tasks\uat_findings_post_deploy.md",
        "tasks\uat_findings_post_deploy_v2.md",
        "tasks\uat_findings_post_deploy_v3.md",
        "tasks\uat_findings_post_deploy_v4.md",
        "tasks\uat_findings_post_deploy_v5.md",
        "tasks\uat_findings_post_deploy_v6.md",
        "tasks\uat_findings_post_deploy_v7.md",
        "tasks\validation-post-fix-report.md"
    )
    
    foreach ($file in $files) {
        if (Remove-SafeFile $file) { $script:removed++ }
    }
}

# Phase 4: Old Delegation Logs (8 files)
if ($Phase -eq "all" -or $Phase -eq "4") {
    Write-Host "`n--- PHASE 4: Old Delegation Logs (8 files) ---" -ForegroundColor Yellow
    
    $files = @(
        "memory\delegation_k-33_railway_mcp_devops_tooling.md",
        "memory\delegation_k-31_railway_redis_optimization.md",
        "memory\delegation_k-30_odds_api_comparison.md",
        "memory\delegation_g-32_player_id_mapping_migration.md",
        "memory\delegation_g-29_redis_deployment.md",
        "memory\2026-03-25-il-roster-support.md",
        "memory\2026-03-27.md",
        "memory\calibration.md"
    )
    
    foreach ($file in $files) {
        if (Remove-SafeFile $file) { $script:removed++ }
    }
}

# Phase 5: CBB Reports (43 files) - VERIFY FIRST
if ($Phase -eq "all" -or $Phase -eq "5") {
    Write-Host "`n--- PHASE 5: CBB Reports (43 files - VERIFY FIRST) ---" -ForegroundColor Yellow
    
    $files = @(
        # March CBB reports
        "reports\2026-03-06-balldontlie-api-research.md",
        "reports\2026-03-06-seed-data-research.md",
        "reports\2026-03-06-tournament-intelligence.md",
        "reports\2026-03-07-health-check-thresholds.md",
        "reports\2026-03-07-k6-o8-baseline-spec.md",
        "reports\2026-03-07-model-quality-audit.md",
        "reports\2026-03-12-api-ground-truth.md",
        "reports\2026-03-12-possession-sim-audit.md",
        "reports\2026-03-12-recalibration-v92-spec.md",
        "reports\2026-03-13-clv-attribution.md",
        "reports\2026-03-16-a26t2-implementation-spec.md",
        "reports\2026-03-16-project-state-assessment.md",
        "reports\2026-03-23-oracle-validation-spec.md",
        "reports\2026-03-24-design-phase2-openclaw.md",
        "reports\2026-03-24-mlb-openclaw-patterns.md",
        "reports\2026-03-24-openclaw-quickref.md",
        # CBB audit reports
        "reports\BETTING_HISTORY_AUDIT_MARCH_2026.md",
        "reports\AUDIT_VERIFICATION_KIMI_MARCH_2026.md",
        "reports\DISCORD_TOURNAMENT_AUDIT_MARCH_2026.md",
        # CBB validation reports
        "reports\validation\VALIDATION_REPORT_2026_03_20.md",
        "reports\validation\VALIDATION_REPORT_ALERTS.md",
        "reports\validation\VALIDATION_REPORT_BET_HISTORY.md",
        "reports\validation\VALIDATION_REPORT_CALIBRATION.md",
        "reports\validation\VALIDATION_REPORT_CLV.md",
        "reports\validation\VALIDATION_REPORT_LIVE_SLATE.md",
        "reports\validation\VALIDATION_REPORT_ODDS_MONITOR.md",
        "reports\validation\VALIDATION_REPORT_TODAY.md",
        "reports\validation\REVALIDATION_REPORT_ALERTS.md",
        "reports\validation\REVALIDATION_REPORT_CLV.md",
        "reports\validation\PHASE_1_VALIDATION_SUMMARY.md",
        "reports\validation\PHASE_1_PLUS_VALIDATION.md",
        "reports\validation\PHASE_1_FINAL_VALIDATION.md",
        "reports\validation\CLAUDE_HANDOFF_VALIDATION.md",
        # OpenClaw CBB specs
        "reports\OPENCLAW_AUTONOMY_SPEC_v4.md",
        "reports\OPENCLAW_AUTONOMY_SPEC_v4_MLB_ADDENDUM.md",
        "reports\OPENCLAW_AUTONOMY_SPEC_v4_PHASE1_NOW.md",
        "reports\openclaw-issue-analysis.md"
    )
    
    foreach ($file in $files) {
        if (Remove-SafeFile $file) { $script:removed++ }
    }
}

# Phase 6: Obsolete Docs (18 files)
if ($Phase -eq "all" -or $Phase -eq "6") {
    Write-Host "`n--- PHASE 6: Obsolete Docs (18 files) ---" -ForegroundColor Yellow
    
    $files = @(
        # Root
        "KIMI_K39_K43_DELEGATION.md",
        "KIMI_RESEARCH_BRIEF.md",
        "DOCUMENTATION_CLEANUP_PLAN.md",
        "DOCUMENTATION_CLEANUP_SUMMARY.md",
        # Tasks
        "tasks\cbb_enhancement_plan.md",
        "tasks\CLV_VALIDATION_PICKUP.md",
        "tasks\OPENCLAW_ORCHESTRATOR_SPEC.md",
        # Docs
        "docs\BRACKET_PROJECTION_PLAN.md",
        "docs\THIRD_RATING_SOURCE.md",
        # Archive execution plans
        "docs\archive\plans\EXECUTION_PLAN_K34.md",
        "docs\archive\plans\EXECUTION_PLAN_K35.md",
        "docs\archive\plans\EXECUTION_PLAN_K36.md",
        "docs\archive\plans\EXECUTION_PLAN_K37.md",
        "docs\archive\plans\EXECUTION_PLAN_K38.md",
        # Archive incidents (duplicates)
        "docs\archive\incidents\ops_whip_root_cause_analysis.md",
        "docs\archive\incidents\ops_whip_root_cause_analysis_CORRECTED.md",
        "docs\archive\incidents\ops_whip_root_cause_analysis_FINAL.md",
        "docs\archive\incidents\ops_whip_root_cause_RAILWAY_INVESTIGATION.md"
    )
    
    foreach ($file in $files) {
        if (Remove-SafeFile $file) { $script:removed++ }
    }
}

# Summary
Write-Host "`n=== CLEANUP SUMMARY ===" -ForegroundColor Magenta
Write-Host "Phase: $Phase" -ForegroundColor Cyan
Write-Host "Removed: $removed items" -ForegroundColor Green
Write-Host "Errors: $errors" -ForegroundColor $(if ($errors -gt 0) { "Red" } else { "Green" })
Write-Host "Dry Run: $DryRun`n" -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "💡 Run without -DryRun to perform actual deletion" -ForegroundColor Yellow
}

if ($errors -gt 0) {
    Write-Host "⚠️  Some files could not be removed. Check errors above." -ForegroundColor Red
    exit 1
}

Write-Host "✅ Cleanup complete!" -ForegroundColor Green

# Post-cleanup verification
if (-not $DryRun -and ($Phase -eq "all" -or $Phase -eq "6")) {
    Write-Host "`n--- POST-CLEANUP VERIFICATION ---" -ForegroundColor Yellow
    $mdCount = (Get-ChildItem -Path . -Filter "*.md" -Recurse | Measure-Object).Count
    Write-Host "Total .md files remaining: $mdCount (expected: ~236)" -ForegroundColor Cyan
    
    if ($mdCount -lt 200 -or $mdCount -gt 250) {
        Write-Host "⚠️  WARNING: File count ($mdCount) outside expected range (200-250)" -ForegroundColor Yellow
    }
}
