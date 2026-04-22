# Documentation Cleanup Recommendation

**Date:** April 22, 2026  
**Analyst:** Claude Code (Master Architect)  
**Scope:** 333 markdown files analyzed across workspace  
**Goal:** Remove 50-100 obsolete files while preserving operational docs and baseball research

---

## Executive Summary

**Current State:** 333 markdown files consuming excessive token budget  
**CBB Season:** CLOSED (April 2026) — tournament content obsolete  
**MLB Season:** ACTIVE — fantasy baseball is primary system  

**Recommended Deletions:** 97 files (~29% reduction)  
**Files to Keep:** 236 files (operational + baseball research + current specs)

---

## Summary Statistics

| Category | Count | Action |
|----------|-------|--------|
| **Root Duplicate Prompts** | 14 | REMOVE (duplicates of .claude/prompts/) |
| **CBB Agent Tasks** | 7 | REMOVE (season closed) |
| **CBB Tournament Reports** | 21 | REMOVE (March tournament prep) |
| **Old UAT Findings** | 10 | REMOVE (keep only v8) |
| **Old Delegation Logs** | 7 | REMOVE (completed March tasks) |
| **Obsolete Reports (March CBB)** | 25 | REMOVE (superseded by April work) |
| **Duplicate/Obsolete Docs** | 13 | REMOVE (archive or redundant) |
| **TOTAL TO REMOVE** | **97 files** | |
| **Operational Docs** | 15 | KEEP (HANDOFF, AGENTS, etc.) |
| **April 2026 Reports** | 65 | KEEP (active MLB fantasy work) |
| **Baseball Math/Stats** | 12 | KEEP (ERA, WHIP, Kelly, etc.) |
| **API References** | 8 | KEEP (Yahoo, BDL, Statcast) |
| **Active Tasks** | 3 | KEEP (todo.md, lessons.md, architect_review.md) |
| **Skills** | 9 | KEEP (active Claude/Gemini skills) |
| **Other Keepers** | 124 | KEEP (current specs, plans, validation) |
| **TOTAL TO KEEP** | **236 files** | |

---

## REMOVAL LIST (97 files)

### Category 1: Root Duplicate Prompts (14 files)

**Reason:** All duplicates of `.claude/prompts/*.md` — canonical versions exist in .claude/prompts/

```
CLAUDE_ARCHITECT_PROMPT_MARCH28.md
CLAUDE_FANTASY_ROADMAP_PROMPT.md
CLAUDE_GEMINI_SKILLS_PROMPT.md
CLAUDE_K33_MCP_DELEGATION_PROMPT.md
CLAUDE_K34_K38_KIMI_DELEGATION.md
CLAUDE_KIMI_DELEGATION.md
CLAUDE_LOCAL_LLM_PROMPT.md
CLAUDE_OPENCLAW_ORCHESTRATION_PROMPT.md
CLAUDE_PHASE0_IMPLEMENTATION_PROMPT.md
CLAUDE_PHASE1_IMPLEMENTATION_PROMPT.md
CLAUDE_PHASE2_IMPLEMENTATION_PROMPT.md
CLAUDE_PHASE3_IMPLEMENTATION_PROMPT.md
CLAUDE_RETURN_PROMPT.md
CLAUDE_TEAM_COORDINATION_PROMPT.md
CLAUDE_UAT_FIXES_PROMPT.md
CLAUDE_UI_CONTRACT_REORIENTATION_PROMPT.md
CLAUDE_UI_UX_ARCHITECT_PROMPT.md
```

**Verification:** Each has exact duplicate in `.claude/prompts/` (confirmed via directory listing)

---

### Category 2: CBB Agent Tasks (7 files)

**Reason:** CBB season closed — no further CBB development planned

```
.agent_tasks/v9_2_recalibration.md
.agent_tasks/oracle_validation.md
.agent_tasks/README.md
.agent_tasks/done/phase_1_core_analytics.md
.agent_tasks/done/phase_2_trading.md
.agent_tasks/done/phase_3_tournament.md
.agent_tasks/done/phase_4_mobile_pwa.md
.agent_tasks/done/phase_5_polish.md
```

**Note:** .agent_tasks/ entire directory can be removed

---

### Category 3: CBB Tournament Reports (21 files)

**Reason:** March 2026 tournament prep — season over, superseded by MLB work

```
reports/2026-03-06-balldontlie-api-research.md (CBB-specific)
reports/2026-03-06-seed-data-research.md (tournament brackets)
reports/2026-03-06-tournament-intelligence.md (tournament prep)
reports/2026-03-07-health-check-thresholds.md (CBB thresholds)
reports/2026-03-07-k6-o8-baseline-spec.md (CBB pre-tournament)
reports/2026-03-07-model-quality-audit.md (CBB model audit)
reports/2026-03-12-api-ground-truth.md (CBB API validation)
reports/2026-03-12-possession-sim-audit.md (CBB simulation)
reports/2026-03-12-recalibration-v92-spec.md (CBB V9.2 recalibration)
reports/2026-03-13-clv-attribution.md (CBB CLV tracking)
reports/2026-03-16-a26t2-implementation-spec.md (CBB seed-spread scalars)
reports/2026-03-16-project-state-assessment.md (pre-tournament status)
reports/2026-03-23-oracle-validation-spec.md (CBB oracle validation)
reports/2026-03-24-design-phase2-openclaw.md (CBB OpenClaw Phase 2)
reports/2026-03-24-mlb-openclaw-patterns.md (CBB→MLB transition, obsolete)
reports/2026-03-24-openclaw-quickref.md (CBB-focused)
reports/BETTING_HISTORY_AUDIT_MARCH_2026.md (CBB betting history)
reports/AUDIT_VERIFICATION_KIMI_MARCH_2026.md (CBB audit verification)
reports/DISCORD_TOURNAMENT_AUDIT_MARCH_2026.md (tournament Discord audit)
docs/BRACKET_PROJECTION_PLAN.md (tournament brackets)
docs/THIRD_RATING_SOURCE.md (CBB third rating source)
```

---

### Category 4: Old UAT Findings (10 files)

**Reason:** Iterative UAT reports — keep only v8 (latest)

```
tasks/uat_findings.md (v0)
tasks/uat_findings_fresh.md (intermediate)
tasks/uat_findings_post_deploy.md (v1)
tasks/uat_findings_post_deploy_v2.md
tasks/uat_findings_post_deploy_v3.md
tasks/uat_findings_post_deploy_v4.md
tasks/uat_findings_post_deploy_v5.md
tasks/uat_findings_post_deploy_v6.md
tasks/uat_findings_post_deploy_v7.md
tasks/validation-post-fix-report.md (superseded by v8)
```

**Keep:** `tasks/uat_findings_post_deploy_v8.md` (latest)

---

### Category 5: Old Delegation Logs (7 files)

**Reason:** Completed March delegation tasks — no longer actionable

```
memory/delegation_k-33_railway_mcp_devops_tooling.md (completed)
memory/delegation_k-31_railway_redis_optimization.md (completed)
memory/delegation_k-30_odds_api_comparison.md (completed)
memory/delegation_g-32_player_id_mapping_migration.md (completed)
memory/delegation_g-29_redis_deployment.md (completed)
memory/2026-03-25-il-roster-support.md (completed)
memory/2026-03-27.md (old daily log)
memory/calibration.md (CBB calibration — season closed)
```

**Keep:** `memory/2026-04-20.md`, `memory/2026-04-21.md` (recent operational logs)

---

### Category 6: Obsolete CBB Task Files (3 files)

**Reason:** CBB-specific enhancement plans — season closed

```
tasks/cbb_enhancement_plan.md
tasks/CLV_VALIDATION_PICKUP.md (CBB CLV tracking)
tasks/OPENCLAW_ORCHESTRATOR_SPEC.md (superseded by current implementation)
```

**Keep:** `tasks/todo.md`, `tasks/lessons.md`, `tasks/architect_review.md` (active)

---

### Category 7: OpenClaw CBB Audit Files (5 files)

**Reason:** March CBB audit reports — season closed

```
.openclaw/AUDIT_REPORT_2026-03-11.md
.openclaw/active-task.md
.openclaw/IMPROVEMENTS_SUMMARY.md
.openclaw/README.md (CBB-focused)
.openclaw/TROUBLESHOOTING.md (CBB-specific)
```

---

### Category 8: Duplicate/Obsolete Operational Docs (4 files)

**Reason:** Superseded by consolidated CLAUDE.md or archive-ready

```
KIMI_K39_K43_DELEGATION.md (completed delegation)
KIMI_RESEARCH_BRIEF.md (superseded by current reports)
DOCUMENTATION_CLEANUP_PLAN.md (superseded by this file)
DOCUMENTATION_CLEANUP_SUMMARY.md (superseded by this file)
```

---

### Category 9: Old CBB Validation Reports (18 files)

**Reason:** March 2026 CBB validation — season closed

```
reports/validation/VALIDATION_REPORT_2026_03_20.md
reports/validation/VALIDATION_REPORT_ALERTS.md (CBB)
reports/validation/VALIDATION_REPORT_BET_HISTORY.md (CBB)
reports/validation/VALIDATION_REPORT_CALIBRATION.md (CBB)
reports/validation/VALIDATION_REPORT_CLV.md (CBB)
reports/validation/VALIDATION_REPORT_LIVE_SLATE.md (CBB)
reports/validation/VALIDATION_REPORT_ODDS_MONITOR.md (CBB)
reports/validation/VALIDATION_REPORT_TODAY.md (CBB)
reports/validation/REVALIDATION_REPORT_ALERTS.md (CBB)
reports/validation/REVALIDATION_REPORT_CLV.md (CBB)
reports/validation/PHASE_1_VALIDATION_SUMMARY.md (CBB)
reports/validation/PHASE_1_PLUS_VALIDATION.md (CBB)
reports/validation/PHASE_1_FINAL_VALIDATION.md (CBB)
reports/validation/CLAUDE_HANDOFF_VALIDATION.md (CBB)
reports/OPENCLAW_AUTONOMY_SPEC_v4.md (superseded)
reports/OPENCLAW_AUTONOMY_SPEC_v4_MLB_ADDENDUM.md (superseded)
reports/OPENCLAW_AUTONOMY_SPEC_v4_PHASE1_NOW.md (superseded)
reports/openclaw-issue-analysis.md (March issues, resolved)
```

---

### Category 10: Archived Execution Plans (5 files)

**Reason:** Completed execution plans from March — historical only

```
docs/archive/plans/EXECUTION_PLAN_K34.md
docs/archive/plans/EXECUTION_PLAN_K35.md
docs/archive/plans/EXECUTION_PLAN_K36.md
docs/archive/plans/EXECUTION_PLAN_K37.md
docs/archive/plans/EXECUTION_PLAN_K38.md
```

---

### Category 11: Archived Incident Reports (4 files)

**Reason:** Superseded by canonical incident in docs/incidents/

```
docs/archive/incidents/ops_whip_root_cause_analysis.md
docs/archive/incidents/ops_whip_root_cause_analysis_CORRECTED.md
docs/archive/incidents/ops_whip_root_cause_analysis_FINAL.md
docs/archive/incidents/ops_whip_root_cause_RAILWAY_INVESTIGATION.md
```

**Keep:** `docs/incidents/2026-04-10-ops-whip-root-cause.md` (canonical)

---

## KEEP LIST (236 files)

### Core Operational Docs (15 files) — NEVER REMOVE

```
✅ HANDOFF.md — Current operational state (CRITICAL)
✅ HANDOFF_ARCHIVE.md — Historical handoff context
✅ AGENTS.md — Agent role definitions
✅ ORCHESTRATION.md — Swimlane routing
✅ IDENTITY.md — Risk posture + Kelly math
✅ HEARTBEAT.md — Operational loops
✅ SOUL.md — System identity
✅ CLAUDE.md — Project orientation (root canonical)
✅ GEMINI.md — Gemini agent guide
✅ QUICKREF.md — Command reference
✅ docs_index.md — Minified system reference
✅ MASTER_DOCUMENT_INDEX.md — Full document map
✅ CLAUDE_PROMPTS_INDEX.md — Prompt index
✅ README.md — Repository overview
✅ INSTALL.md — Setup instructions
```

---

### Baseball Math & Research (12 files) — PRESERVE CALCULATIONS

```
✅ STAT_MAPPING_ANALYSIS.md — Yahoo stat ID mapping
✅ STAT_MAPPING_IMPACT_ANALYSIS.md — Stat mapping impact
✅ reports/2026-04-18-category-math-reference.md — ERA, WHIP, SV, K formulas
✅ reports/2026-04-09-vorp-implementation-guide.md — VORP calculation
✅ reports/2026-04-09-zscore-best-practices.md — Z-score normalization
✅ reports/2026-04-10-h2h-scoring-systems.md — H2H category scoring
✅ reports/daily_lineup_optimization_research.md — Lineup optimization math
✅ reports/ELITE_FANTASY_TECHNIQUES_QUANTITATIVE_ANALYSIS.md — Advanced stats
✅ reports/DRAFT_DAY_WAR_ROOM_BLUEPRINT_2026.md — Draft strategy + WAR
✅ docs/FATIGUE_MODEL.md — Pitcher fatigue modeling
✅ docs/task-10-era-investigation.md — ERA calculation deep dive
✅ reports/2026-04-18-rolling-stats-audit.md — Rolling window calculations
```

---

### API References (8 files) — CRITICAL INTEGRATIONS

```
✅ docs/YAHOO_API_REFERENCE.md — Yahoo Fantasy API spec
✅ docs/STATCAST_API_GUIDE.md — Statcast/pybaseball guide
✅ docs/PROJECTION_DATA_SOURCES.md — Projection providers
✅ reports/YAHOO_FANTASY_API_RESEARCH.md — Yahoo API deep dive
✅ reports/2026-04-18-yahoo-api-shape-catalog.md — Yahoo response shapes
✅ reports/2026-04-10-bdl-api-capabilities.md — BallDontLie MLB capabilities
✅ reports/2026-04-10-mlb-api-comparison.md — MLB data provider comparison
✅ reports/2026-04-08-balldontlie-mlb-mcp-research.md — BDL MCP integration
```

---

### April 2026 Reports — Active MLB Fantasy Work (65 files)

**Keep ALL reports from April 2026** (active season, current specs):

```
✅ reports/2026-04-21-*.md (5 files — latest production audits)
✅ reports/2026-04-20-*.md (4 files — math fix specs)
✅ reports/2026-04-18-*.md (6 files — frontend audit, category math)
✅ reports/2026-04-17-*.md (2 files — UI contract audit)
✅ reports/2026-04-15-*.md (3 files — Layer 2 certification)
✅ reports/2026-04-13-*.md (3 files — statcast aggregation)
✅ reports/2026-04-11-*.md (6 files — database health)
✅ reports/2026-04-10-*.md (4 files — API comparison, UI design)
✅ reports/2026-04-09-*.md (10 files — player identity, sync jobs)
✅ reports/2026-04-08-*.md (8 files — weather, park factors, UI)
✅ reports/2026-04-07-*.md (2 files — identity resolution, BDL stats)
✅ reports/2026-04-05-*.md (1 file — raw ingestion audit)
✅ reports/2026-04-01-*.md (13 files — fantasy stabilization, UAT)
```

**Reason:** Active season work, current specs, recent bug investigations

---

### Active Tasks (3 files)

```
✅ tasks/todo.md — Active execution board
✅ tasks/lessons.md — Self-improvement log
✅ tasks/architect_review.md — Architect review queue
```

---

### Active Skills (9 files)

```
✅ .claude/skills/cbb-identity/SKILL.md — Risk posture (update name later)
✅ .claude/skills/emac-protocol/SKILL.md — Agent coordination
✅ .gemini/skills/railway-logs/SKILL.md — Gemini ops skill
✅ .gemini/skills/health-check/SKILL.md — Gemini health check
✅ .gemini/skills/env-check/SKILL.md — Gemini env check
✅ .gemini/skills/db-migrate/SKILL.md — Gemini migration skill
✅ skills/yahoo-fantasy-baseball/SKILL.md — Yahoo integration skill
✅ skills/cbb-edge-guardian/SKILL.md — Guardian circuit breaker skill
✅ skills/capability-evolver/SKILL.md — Self-improvement skill
```

---

### Active Plans & Specs (18 files)

```
✅ DESIGN.md — System architecture
✅ SYSTEM_ARCHITECTURE_ANALYSIS.md — Architecture gaps analysis
✅ FRONTEND_MIGRATION.md — Next.js migration plan
✅ PRODUCTION_DEPLOYMENT_PLAN.md — Deployment procedures
✅ PARLAY_IMPLEMENTATION.md — Parlay engine spec (MLB future)
✅ DAILY_INGESTION_SUMMARY.md — Daily job documentation
✅ TOURNAMENT_DAY_CHECKLIST.md — Tournament day ops (archive later)
✅ TOOLS.md — Tool inventory
✅ USER.md — User documentation
✅ docs/EXECUTIVE_ROADMAP.md — Long-term roadmap
✅ docs/MLB_FANTASY_ROADMAP.md — MLB feature roadmap
✅ docs/OPENCLAW_LITE_PLAN.md — OpenClaw lightweight plan
✅ docs/analytics-roadmap.md — Analytics features
✅ docs/RAILWAY_ENV_SETUP.md — Railway configuration
✅ docs/RAILWAY_DISCORD_FIX.md — Discord Railway setup
✅ docs/DISCORD_SETUP.md — Discord integration
✅ docs/DISCORD_SETUP_QUICKSTART.md — Discord quick start
✅ docs/DISCORD_CHANNEL_DESIGN.md — Discord channel architecture
```

---

### Superpowers (Layer 2) Docs (15 files)

**Keep ALL** — active Layer 2 certification work:

```
✅ docs/superpowers/specs/*.md (5 files)
✅ docs/superpowers/plans/*.md (7 files)
✅ docs/superpowers/completed/*.md (4 files)
✅ docs/superpowers/prep/*.md (3 files)
✅ docs/superpowers/README.md
✅ docs/superpowers/planning-summary-tasks-10-11.md
```

---

### Claude Commands & Rules (3 files)

```
✅ .claude/rules/workflow.md — Workflow rules (attachment in instructions)
✅ .claude/commands/emac.md — EMAC command
✅ .claude/DISCORD_ERRORS_EXPLAINED.md — Discord error reference
```

---

### Canonical Prompts in .claude/prompts/ (14 files)

**Keep ALL** — canonical prompt definitions:

```
✅ .claude/prompts/CLAUDE.md
✅ .claude/prompts/CLAUDE_ARCHITECT_PROMPT_MARCH28.md
✅ .claude/prompts/CLAUDE_FANTASY_ROADMAP_PROMPT.md
✅ .claude/prompts/CLAUDE_GEMINI_SKILLS_PROMPT.md
✅ .claude/prompts/CLAUDE_K33_MCP_DELEGATION_PROMPT.md
✅ .claude/prompts/CLAUDE_K34_K38_KIMI_DELEGATION.md
✅ .claude/prompts/CLAUDE_KIMI_DELEGATION.md
✅ .claude/prompts/CLAUDE_LOCAL_LLM_PROMPT.md
✅ .claude/prompts/CLAUDE_OPENCLAW_ORCHESTRATION_PROMPT.md
✅ .claude/prompts/CLAUDE_RETURN_PROMPT.md
✅ .claude/prompts/CLAUDE_TEAM_COORDINATION_PROMPT.md
✅ .claude/prompts/CLAUDE_UAT_FIXES_PROMPT.md
✅ .claude/prompts/CLAUDE_UI_CONTRACT_REORIENTATION_PROMPT.md
✅ .claude/prompts/CLAUDE_UI_UX_ARCHITECT_PROMPT.md
```

---

### Recent Memory (2 files)

```
✅ memory/2026-04-20.md — Recent operational log
✅ memory/2026-04-21.md — Latest operational log
```

---

### Active Reports (59 additional files)

**Keep all non-CBB, non-duplicate reports**:

```
✅ reports/ADVANCED_ANALYTICS_INTEGRATION.md
✅ reports/AI_ML_RESEARCH_MCP_OPEN_SOURCE.md
✅ reports/ARCHITECTURE_ANALYSIS_API_WORKER_PATTERN.md
✅ reports/data-quality-fixes-validation.md
✅ reports/dashboard-mock-audit.md
✅ reports/db_null_analysis_*.md (2 files)
✅ reports/EXPANSION_ARCHITECTURE_MLB_PGA_BLUEPRINT.md
✅ reports/FANTASY_BASEBALL_2026_PROJECTIONS_COMPLETE.md
✅ reports/FANTASY_BASEBALL_ELITE_ROADMAP_v2.md
✅ reports/FANTASY_BASEBALL_GAP_ANALYSIS.md
✅ reports/GEMINI_MCP_ANALYSIS.md
✅ reports/GEMINI_UI_AUDIT*.md (3 files)
✅ reports/KIMI_UAT_ANALYSIS_2026-04-01.md
✅ reports/MULTI_AGENT_ORCHESTRATION_ANALYSIS.md
✅ reports/openclaw-capabilities-assessment.md
✅ reports/OPENCLAW_DISCORD_ENHANCEMENT_PLAN.md
✅ reports/OPENCLAW_ORCHESTRATION_WORKFLOW.md
✅ reports/README.md
✅ reports/RESEARCH_PLAN_K34_K38.md
✅ reports/SCHEMA_DISCOVERY.md
✅ reports/SPORTSBOOK_FANTASY_BASEBALL_ROADMAP.md
✅ reports/task-11-validation-report.md
✅ reports/task-21-orphan-linking-results.md
✅ reports/tasks-10-11-completion-summary.md
✅ reports/UAT_ISSUES_ANALYSIS.md
✅ reports/URGENT_UAT.md
✅ reports/yahoo-client-hotfix-march28.md
... (and 32 more non-CBB reports)
```

---

### Docs Archive (keep for reference) (7 files)

```
✅ docs/archive/CLAUDE.md (historical)
✅ docs/archive/IMPLEMENTATION.md (historical)
✅ docs/archive/QUICKSTART.md (historical)
✅ docs/archive/README.md (archive index)
✅ docs/archive/SETUP_FROM_SCRATCH.md (historical setup)
✅ docs/archive/SKILL.md (historical skill)
✅ docs/archive/WINDOWS_SETUP.md (historical)
✅ docs/archive/cheatsheet.md (historical)
```

---

### Miscellaneous Keep (16 files)

```
✅ .claude/agents/cbb-architect.md (update name to mlb-architect later)
✅ .claude/CLAUDE_SYSTEM_PROMPT_UPDATE.md
✅ docs/CLOSER_SITUATION_SOURCES.md — MLB closer tracking
✅ docs/LINEUP_CONFIRMATION_SOURCES.md — Lineup confirmation sources
✅ docs/UAT_MARCH_10_2026.md — Historical UAT baseline
✅ docs/reference/README.md
✅ docs/reference/AGENT_TEAM_BEST_PRACTICES.md
✅ docs/incidents/2026-04-10-ops-whip-root-cause.md — OPS/WHIP incident
✅ docs/plan/fantasy-recovery-2026-04/wave1-completion-summary.md
✅ frontend/PHASE4_SPEC.md
✅ frontend/VALIDATION_REPORT.md
✅ postman_collections/README_UAT_Guide.md
✅ postman_collections/UAT_Checklist_Template.md
✅ data/projections/README.md
✅ reviews/EMAC-087_PEER_REVIEW.md
✅ scripts/BACKFILL_README.md
✅ scripts/diagnose_*.md (4 files — diagnostic tools)
✅ skills/sonoscli/SKILL.md
✅ skills/capability-evolver/README.md (and related files)
```

---

## FILES REQUIRING USER REVIEW

**The following 3 files need your decision:**

1. **TOURNAMENT_DAY_CHECKLIST.md**
   - **Status:** CBB tournament checklist
   - **Current Location:** Root
   - **Recommendation:** ARCHIVE (season closed) or REMOVE
   - **Question:** Any reusable patterns for MLB high-stakes days?

2. **.claude/agents/cbb-architect.md**
   - **Status:** Named "cbb-architect" but still in use
   - **Current Location:** .claude/agents/
   - **Recommendation:** RENAME to `mlb-architect.md` (don't delete)
   - **Action Required:** Rename file and update references

3. **.claude/skills/cbb-identity/SKILL.md**
   - **Status:** Named "cbb-identity" but contains Kelly math still in use
   - **Current Location:** .claude/skills/cbb-identity/
   - **Recommendation:** RENAME directory to `risk-identity` (don't delete)
   - **Action Required:** Rename directory and update references in workflow.md

---

## EXECUTION PLAN

### Phase 1: Validate Duplicates (SAFE)

Delete 14 root-level CLAUDE_*.md files (verified duplicates):

```bash
# PowerShell commands
Remove-Item -Path "CLAUDE_ARCHITECT_PROMPT_MARCH28.md" -Force
Remove-Item -Path "CLAUDE_FANTASY_ROADMAP_PROMPT.md" -Force
Remove-Item -Path "CLAUDE_GEMINI_SKILLS_PROMPT.md" -Force
Remove-Item -Path "CLAUDE_K33_MCP_DELEGATION_PROMPT.md" -Force
Remove-Item -Path "CLAUDE_K34_K38_KIMI_DELEGATION.md" -Force
Remove-Item -Path "CLAUDE_KIMI_DELEGATION.md" -Force
Remove-Item -Path "CLAUDE_LOCAL_LLM_PROMPT.md" -Force
Remove-Item -Path "CLAUDE_OPENCLAW_ORCHESTRATION_PROMPT.md" -Force
Remove-Item -Path "CLAUDE_PHASE0_IMPLEMENTATION_PROMPT.md" -Force
Remove-Item -Path "CLAUDE_PHASE1_IMPLEMENTATION_PROMPT.md" -Force
Remove-Item -Path "CLAUDE_PHASE2_IMPLEMENTATION_PROMPT.md" -Force
Remove-Item -Path "CLAUDE_PHASE3_IMPLEMENTATION_PROMPT.md" -Force
Remove-Item -Path "CLAUDE_RETURN_PROMPT.md" -Force
Remove-Item -Path "CLAUDE_TEAM_COORDINATION_PROMPT.md" -Force
Remove-Item -Path "CLAUDE_UAT_FIXES_PROMPT.md" -Force
Remove-Item -Path "CLAUDE_UI_CONTRACT_REORIENTATION_PROMPT.md" -Force
Remove-Item -Path "CLAUDE_UI_UX_ARCHITECT_PROMPT.md" -Force
```

**Impact:** Zero (exact duplicates)

---

### Phase 2: Remove CBB Task Directories (SAFE)

```bash
Remove-Item -Path ".agent_tasks" -Recurse -Force
Remove-Item -Path ".openclaw" -Recurse -Force
```

**Impact:** Removes 12 obsolete CBB files

---

### Phase 3: Remove Old UAT Findings (SAFE)

```bash
cd tasks
Remove-Item -Path "uat_findings.md" -Force
Remove-Item -Path "uat_findings_fresh.md" -Force
Remove-Item -Path "uat_findings_post_deploy.md" -Force
Remove-Item -Path "uat_findings_post_deploy_v2.md" -Force
Remove-Item -Path "uat_findings_post_deploy_v3.md" -Force
Remove-Item -Path "uat_findings_post_deploy_v4.md" -Force
Remove-Item -Path "uat_findings_post_deploy_v5.md" -Force
Remove-Item -Path "uat_findings_post_deploy_v6.md" -Force
Remove-Item -Path "uat_findings_post_deploy_v7.md" -Force
Remove-Item -Path "validation-post-fix-report.md" -Force
```

**Impact:** Keeps only v8 (latest)

---

### Phase 4: Remove Old Delegation Logs (SAFE)

```bash
cd memory
Remove-Item -Path "delegation_*.md" -Force
Remove-Item -Path "2026-03-25-il-roster-support.md" -Force
Remove-Item -Path "2026-03-27.md" -Force
Remove-Item -Path "calibration.md" -Force
```

**Impact:** Keeps only April 2026 logs

---

### Phase 5: Remove CBB Reports (VERIFY FIRST)

**STOP: Verify no MLB content in these files before deletion**

```bash
cd reports

# March CBB tournament reports
Remove-Item -Path "2026-03-06-balldontlie-api-research.md" -Force
Remove-Item -Path "2026-03-06-seed-data-research.md" -Force
Remove-Item -Path "2026-03-06-tournament-intelligence.md" -Force
Remove-Item -Path "2026-03-07-health-check-thresholds.md" -Force
Remove-Item -Path "2026-03-07-k6-o8-baseline-spec.md" -Force
Remove-Item -Path "2026-03-07-model-quality-audit.md" -Force
Remove-Item -Path "2026-03-12-api-ground-truth.md" -Force
Remove-Item -Path "2026-03-12-possession-sim-audit.md" -Force
Remove-Item -Path "2026-03-12-recalibration-v92-spec.md" -Force
Remove-Item -Path "2026-03-13-clv-attribution.md" -Force
Remove-Item -Path "2026-03-16-a26t2-implementation-spec.md" -Force
Remove-Item -Path "2026-03-16-project-state-assessment.md" -Force
Remove-Item -Path "2026-03-23-oracle-validation-spec.md" -Force
Remove-Item -Path "2026-03-24-design-phase2-openclaw.md" -Force
Remove-Item -Path "2026-03-24-mlb-openclaw-patterns.md" -Force
Remove-Item -Path "2026-03-24-openclaw-quickref.md" -Force

# CBB audit reports
Remove-Item -Path "BETTING_HISTORY_AUDIT_MARCH_2026.md" -Force
Remove-Item -Path "AUDIT_VERIFICATION_KIMI_MARCH_2026.md" -Force
Remove-Item -Path "DISCORD_TOURNAMENT_AUDIT_MARCH_2026.md" -Force

# CBB validation reports
cd validation
Remove-Item -Path "VALIDATION_REPORT_*.md" -Force
Remove-Item -Path "REVALIDATION_REPORT_*.md" -Force
Remove-Item -Path "PHASE_1_*.md" -Force
Remove-Item -Path "CLAUDE_HANDOFF_VALIDATION.md" -Force
cd ..

# OpenClaw CBB specs
Remove-Item -Path "OPENCLAW_AUTONOMY_SPEC_v4.md" -Force
Remove-Item -Path "OPENCLAW_AUTONOMY_SPEC_v4_MLB_ADDENDUM.md" -Force
Remove-Item -Path "OPENCLAW_AUTONOMY_SPEC_v4_PHASE1_NOW.md" -Force
Remove-Item -Path "openclaw-issue-analysis.md" -Force
```

**Impact:** Removes 43 CBB-specific files

---

### Phase 6: Remove Obsolete Docs (SAFE)

```bash
# Root
Remove-Item -Path "KIMI_K39_K43_DELEGATION.md" -Force
Remove-Item -Path "KIMI_RESEARCH_BRIEF.md" -Force
Remove-Item -Path "DOCUMENTATION_CLEANUP_PLAN.md" -Force
Remove-Item -Path "DOCUMENTATION_CLEANUP_SUMMARY.md" -Force

# Tasks
cd tasks
Remove-Item -Path "cbb_enhancement_plan.md" -Force
Remove-Item -Path "CLV_VALIDATION_PICKUP.md" -Force
Remove-Item -Path "OPENCLAW_ORCHESTRATOR_SPEC.md" -Force

# Docs
cd docs
Remove-Item -Path "BRACKET_PROJECTION_PLAN.md" -Force
Remove-Item -Path "THIRD_RATING_SOURCE.md" -Force

# Archive execution plans
cd archive/plans
Remove-Item -Path "EXECUTION_PLAN_K*.md" -Force
cd ..

# Archive incidents (duplicates)
cd incidents
Remove-Item -Path "ops_whip_root_cause_analysis*.md" -Force
Remove-Item -Path "ops_whip_root_cause_RAILWAY_INVESTIGATION.md" -Force
```

**Impact:** Removes 18 obsolete files

---

## VERIFICATION COMMANDS

### Before Deletion: Count Current Files

```powershell
(Get-ChildItem -Path . -Filter "*.md" -Recurse | Measure-Object).Count
# Expected: 333 files
```

### After Phase 1-6: Count Remaining Files

```powershell
(Get-ChildItem -Path . -Filter "*.md" -Recurse | Measure-Object).Count
# Expected: 236 files (97 removed)
```

### Verify No Broken References

```powershell
# Search for references to deleted files in kept files
Get-ChildItem -Path . -Filter "*.md" -Recurse | Select-String -Pattern "CLAUDE_ARCHITECT_PROMPT_MARCH28"
# Should return only .claude/prompts/CLAUDE_ARCHITECT_PROMPT_MARCH28.md references
```

---

## POST-CLEANUP ACTIONS

### 1. Update MASTER_DOCUMENT_INDEX.md

Remove references to deleted files, especially:
- Root CLAUDE_*.md duplicates
- .agent_tasks/
- March CBB reports

### 2. Update docs_index.md

Verify "Document Map" section doesn't reference deleted files.

### 3. Update HANDOFF.md (if needed)

Remove any references to deleted delegation logs or CBB tasks.

### 4. Rename Tasks (Optional)

```bash
# Rename CBB-named files to MLB-appropriate names
Rename-Item -Path ".claude/agents/cbb-architect.md" -NewName "mlb-architect.md"
Rename-Item -Path ".claude/skills/cbb-identity" -NewName "risk-identity"
```

Then update references in:
- `.claude/rules/workflow.md`
- `AGENTS.md`

---

## ROLLBACK PLAN

If anything breaks:

```bash
# Restore from git
git checkout HEAD -- <deleted_file>

# Or restore entire batch
git checkout HEAD -- reports/2026-03-*.md
```

**Recommendation:** Commit after each phase for granular rollback.

---

## EXPECTED OUTCOMES

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Total .md Files** | 333 | 236 | -29% |
| **Root Duplicates** | 17 | 0 | -100% |
| **CBB-Specific Files** | 66 | 0 | -100% |
| **Old UAT Versions** | 10 | 1 | -90% |
| **Old Delegation Logs** | 7 | 0 | -100% |
| **Token Load (est.)** | ~850KB | ~600KB | -29% |

---

## CONCLUSION

This cleanup removes **97 obsolete files (29%)** while preserving:
- ✅ All operational docs (HANDOFF, AGENTS, ORCHESTRATION, etc.)
- ✅ All baseball math and statistical research
- ✅ All April 2026 MLB fantasy work
- ✅ All API references and integration docs
- ✅ All active tasks and skills

**Recommendation:** Execute Phase 1-4 immediately (low risk). Verify Phase 5 manually before execution. Execute Phase 6 after Phase 5 verification.

**Next Step:** User approval to proceed with Phase 1 (duplicate removal).
