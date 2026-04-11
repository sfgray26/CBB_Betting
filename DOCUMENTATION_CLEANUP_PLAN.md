# Repository Documentation Cleanup Plan

> **Created:** April 11, 2026  
> **Approach:** Big Bang (Option A)  
> **Estimated Duration:** 11-14 hours  
> **Risk Level:** LOW (with preservation rules followed)

---

## 1. EXECUTIVE SUMMARY

### Current State
- **250 markdown files** total
- **15 CLAUDE*.md files** in root (significant sprawl)
- **4 versioned ops_whip files** (redundancy)
- **Inconsistent naming** in reports/ (dated vs undated vs K-prefixed)
- **Good foundation exists:** docs/archive/, MASTER_DOCUMENT_INDEX.md

### Target State
- **~175-190 files** (20-25% reduction through consolidation)
- **Single CLAUDE_PROMPTS_INDEX.md** replaces 15 scattered prompt files
- **Consistent naming convention** across reports/
- **Clear archive/completed/active distinction**
- **All cross-references valid**

---

## 2. INVENTORY BY CATEGORY

### 2.1 Root Level CLAUDE*.md Files (15 files)

| File | Size | Status | Decision | Target Location |
|------|------|--------|----------|-----------------|
| **CLAUDE.md** | 7.4KB | ⚠️ ACTIVE | KEEP in root | (no move) |
| **CLAUDE_ARCHITECT_PROMPT_MARCH28.md** | 10.7KB | ⚠️ EMERGENCY | KEEP in root | (no move - referenced by HANDOFF) |
| **CLAUDE_TEAM_COORDINATION_PROMPT.md** | 7.1KB | ⚠️ ACTIVE | KEEP in root | (no move - authoritative) |
| **CLAUDE_COORDINATION_PROMPT.md** | 8.1KB | ✅ DEPRECATED | Archive | `.claude/prompts/archive/` |
| **CLAUDE_RETURN_PROMPT.md** | 12.5KB | ⚠️ ACTIVE | Keep accessible | `.claude/prompts/` |
| **CLAUDE_FANTASY_ROADMAP_PROMPT.md** | 14.5KB | ⚠️ ACTIVE | Keep accessible | `.claude/prompts/` |
| **CLAUDE_UAT_FIXES_PROMPT.md** | 9.9KB | ⚠️ ACTIVE | Keep accessible | `.claude/prompts/` |
| **CLAUDE_OPENCLAW_ORCHESTRATION_PROMPT.md** | 6.6KB | ⚠️ ACTIVE | Keep accessible | `.claude/prompts/` |
| **CLAUDE_GEMINI_SKILLS_PROMPT.md** | 3.4KB | ⚠️ ACTIVE | Keep accessible | `.claude/prompts/` |
| **CLAUDE_LOCAL_LLM_PROMPT.md** | 3.6KB | ⚠️ ACTIVE | Keep accessible | `.claude/prompts/` |
| **CLAUDE_UI_UX_ARCHITECT_PROMPT.md** | 11.0KB | ⚠️ ACTIVE | Keep accessible | `.claude/prompts/` |
| **CLAUDE_PROMPT_K26_MATCHUP_FIX.md** | 8.6KB | ✅ COMPLETED | Archive | `.claude/prompts/archive/` |
| **CLAUDE_K33_MCP_DELEGATION_PROMPT.md** | 7.7KB | ⚠️ ACTIVE | Keep accessible | `.claude/prompts/` |
| **CLAUDE_K34_K38_KIMI_DELEGATION.md** | 19.4KB | ⚠️ ACTIVE | Keep accessible | `.claude/prompts/` |
| **CLAUDE_KIMI_DELEGATION.md** | 11.6KB | ⚠️ ACTIVE | Keep accessible | `.claude/prompts/` |

**NEW FILE TO CREATE:** `CLAUDE_PROMPTS_INDEX.md` (in root) - master index of all prompts

### 2.2 Version Chain: ops_whip_root_cause

| File | Size | Decision | Notes |
|------|------|----------|-------|
| ops_whip_root_cause_analysis.md | 6.7KB | ARCHIVE | Superseded |
| ops_whip_root_cause_analysis_CORRECTED.md | 7.7KB | ARCHIVE | Superseded |
| ops_whip_root_cause_analysis_FINAL.md | 8.7KB | KEEP | Authoritative version |
| ops_whip_root_cause_RAILWAY_INVESTIGATION.md | 6.7KB | MERGE | Merge into FINAL |

**Action:** Consolidate into single `docs/incidents/2026-04-10-ops-whip-root-cause.md`

### 2.3 docs/superpowers/plans/ Analysis

| File | Size | Date | Status | Decision |
|------|------|------|--------|----------|
| 2026-04-10-data-quality-fixes.md | 29.5KB | Apr 10 | ✅ COMPLETE (Tasks 1-11) | Move to `completed/` |
| 2026-04-09-data-quality-remediation.md | 44.2KB | Apr 9 | ✅ COMPLETE (Tasks 1-11) | Move to `completed/` |
| 2026-04-09-position-eligibility-remediation.md | 41.7KB | Apr 9 | ✅ COMPLETE | Move to `completed/` |
| 2026-04-04-fantasy-edge-decoupling-structural.md | 52.3KB | Apr 5 | ⚠️ PARTIAL | Keep in `plans/` (active) |
| 2026-03-31-fantasy-baseball-perf-fixes.md | 24.9KB | Mar 31 | ✅ COMPLETE | Move to `completed/` |

### 2.4 Reports Naming Issues

**Inconsistent patterns found:**

| Pattern | Count | Example | Action |
|---------|-------|---------|--------|
| YYYY-MM-DD-name.md | 32 | 2026-04-10-bdl-api-capabilities.md | ✅ KEEP |
| K[0-9]+_NAME.md | 28 | K11_CLV_ATTRIBUTION_MARCH_2026.md | RENAME to dated format |
| undated_name.md | 15 | balldontlie-api-research.md | RENAME if date determinable |
| EXECUTION_PLAN_K*.md | 5 | EXECUTION_PLAN_K34.md | ARCHIVE (research complete) |

**Reports to RENAME:**
- `balldontlie-api-research.md` → `2026-03-07-balldontlie-api-research.md` (from content)
- `api_ground_truth.md` → `2026-03-18-api-ground-truth.md` (from git history)
- `K11_CLV_ATTRIBUTION_MARCH_2026.md` → `2026-03-12-clv-attribution.md`
- `K12_RECALIBRATION_SPEC_V92.md` → `2026-03-12-recalibration-v92.md`
- `K13_POSSESSION_SIM_AUDIT.md` → `2026-03-12-possession-sim-audit.md`
- ... (and all other K-prefixed reports)

---

## 3. TARGET TAXONOMY

### Proposed Directory Structure

```
cbb-edge/
├── AGENTS.md                         # (keep - core authority)
├── CLAUDE_PROMPTS_INDEX.md           # (NEW - replaces 15 scattered files)
├── HANDOFF.md                        # (keep - active state)
├── HEARTBEAT.md                      # (keep - operational)
├── IDENTITY.md                       # (keep - operational)
├── MASTER_DOCUMENT_INDEX.md          # (update)
├── ORCHESTRATION.md                  # (keep - operational)
├── QUICKREF.md                       # (keep)
├── README.md                         # (keep)
│
├── .claude/
│   ├── prompts/
│   │   ├── CLAUDE.md                 # (symlink or copy from root)
│   │   ├── CLAUDE_ARCHITECT_PROMPT_MARCH28.md
│   │   ├── CLAUDE_TEAM_COORDINATION_PROMPT.md
│   │   ├── CLAUDE_RETURN_PROMPT.md
│   │   ├── CLAUDE_FANTASY_ROADMAP_PROMPT.md
│   │   ├── CLAUDE_UAT_FIXES_PROMPT.md
│   │   ├── CLAUDE_OPENCLAW_ORCHESTRATION_PROMPT.md
│   │   ├── CLAUDE_GEMINI_SKILLS_PROMPT.md
│   │   ├── CLAUDE_LOCAL_LLM_PROMPT.md
│   │   ├── CLAUDE_UI_UX_ARCHITECT_PROMPT.md
│   │   ├── CLAUDE_K33_MCP_DELEGATION_PROMPT.md
│   │   ├── CLAUDE_K34_K38_KIMI_DELEGATION.md
│   │   ├── CLAUDE_KIMI_DELEGATION.md
│   │   └── archive/
│   │       ├── CLAUDE_COORDINATION_PROMPT.md
│   │       └── CLAUDE_PROMPT_K26_MATCHUP_FIX.md
│   ├── commands/
│   ├── rules/
│   └── skills/
│
├── docs/
│   ├── archive/                      # (existing - add to it)
│   │   ├── incidents/
│   │   │   └── ops_whip_root_cause/  # (consolidated versions)
│   │   ├── plans/
│   │   │   └── superpowers/          # (completed old plans)
│   │   └── README.md
│   │
│   ├── incidents/                    # (NEW - active incident docs)
│   │   └── 2026-04-10-ops-whip-root-cause.md
│   │
│   ├── superpowers/
│   │   ├── README.md
│   │   ├── plans/                    # (ACTIVE plans only)
│   │   │   └── 2026-04-04-fantasy-edge-decoupling-structural.md
│   │   ├── completed/                # (NEW - executed plans)
│   │   │   ├── 2026-04-10-data-quality-fixes.md
│   │   │   ├── 2026-04-09-data-quality-remediation.md
│   │   │   ├── 2026-04-09-position-eligibility-remediation.md
│   │   │   └── 2026-03-31-fantasy-baseball-perf-fixes.md
│   │   └── prep/
│   │
│   └── reference/
│       └── AGENT_TEAM_BEST_PRACTICES.md
│
├── reports/
│   ├── README.md                     # (NEW - index with naming convention)
│   ├── YYYY-MM-DD-*.md               # (standardized names)
│   └── validation/
│
└── memory/
    └── YYYY-MM-DD-*.md               # (already standardized)
```

---

## 4. MIGRATION MAP

### Phase 2: Root Level Consolidation

| Source | Destination | Type |
|--------|-------------|------|
| CLAUDE_COORDINATION_PROMPT.md | .claude/prompts/archive/ | Move |
| CLAUDE_PROMPT_K26_MATCHUP_FIX.md | .claude/prompts/archive/ | Move |
| CLAUDE_RETURN_PROMPT.md | .claude/prompts/ | Move |
| CLAUDE_FANTASY_ROADMAP_PROMPT.md | .claude/prompts/ | Move |
| CLAUDE_UAT_FIXES_PROMPT.md | .claude/prompts/ | Move |
| CLAUDE_OPENCLAW_ORCHESTRATION_PROMPT.md | .claude/prompts/ | Move |
| CLAUDE_GEMINI_SKILLS_PROMPT.md | .claude/prompts/ | Move |
| CLAUDE_LOCAL_LLM_PROMPT.md | .claude/prompts/ | Move |
| CLAUDE_UI_UX_ARCHITECT_PROMPT.md | .claude/prompts/ | Move |
| CLAUDE_K33_MCP_DELEGATION_PROMPT.md | .claude/prompts/ | Move |
| CLAUDE_K34_K38_KIMI_DELEGATION.md | .claude/prompts/ | Move |
| CLAUDE_KIMI_DELEGATION.md | .claude/prompts/ | Move |

**NEW:** Create `CLAUDE_PROMPTS_INDEX.md` in root

### Phase 3: Operational Documents

| Source | Destination | Action |
|--------|-------------|--------|
| docs/ops_whip_root_cause_*.md (4 files) | docs/incidents/2026-04-10-ops-whip-root-cause.md | Consolidate |
| docs/ops_whip_root_cause_*.md (originals) | docs/archive/incidents/ | Archive with headers |
| reports/EXECUTION_PLAN_K*.md (5 files) | docs/archive/plans/ | Move (research complete) |

### Phase 4: Reports Standardization

| Current Name | New Name | Status |
|--------------|----------|--------|
| balldontlie-api-research.md | 2026-03-07-balldontlie-api-research.md | Rename |
| api_ground_truth.md | 2026-03-18-api-ground-truth.md | Rename |
| K11_CLV_ATTRIBUTION_MARCH_2026.md | 2026-03-12-clv-attribution.md | Rename |
| K12_RECALIBRATION_SPEC_V92.md | 2026-03-12-recalibration-v92.md | Rename |
| K13_POSSESSION_SIM_AUDIT.md | 2026-03-12-possession-sim-audit.md | Rename |
| K14_STAT57_CONFIRMATION.md | 2026-04-01-stat57-confirmation.md | Rename |
| K15_NAME_CONCAT_SPEC.md | 2026-04-01-name-concat-spec.md | Rename |
| K15_ORACLE_VALIDATION_SPEC.md | 2026-03-23-oracle-validation-spec.md | Rename |
| K16_INGESTION_AUDIT.md | 2026-04-01-ingestion-audit.md | Rename |
| K17_DATETIME_AUDIT.md | 2026-04-01-datetime-audit.md | Rename |
| K17_INSEASON_PIPELINE_OVERHAUL.md | 2026-04-01-inseason-pipeline-overhaul.md | Rename |
| K18_STAT_VALIDATION_SPEC.md | 2026-04-01-stat-validation-spec.md | Rename |
| K19_INSEASON_PIPELINE_SPEC.md | 2026-04-01-inseason-pipeline-spec.md | Rename |
| K20_WAIVER_WIRE_UAT_DEEP_DIVE.md | 2026-04-01-waiver-wire-uat.md | Rename |
| K21_DAILY_LINEUP_UAT_DEEP_DIVE.md | 2026-04-01-daily-lineup-uat.md | Rename |
| K22_MATCHUP_UAT_DEEP_DIVE.md | 2026-04-01-matchup-uat.md | Rename |
| K23_SETTINGS_UAT_DEEP_DIVE.md | 2026-04-01-settings-uat.md | Rename |
| K24_YAHOO_PLAYER_STATS_SPEC.md | 2026-04-01-yahoo-player-stats-spec.md | Rename |
| K25_FANGRAPHS_COLUMN_MAP.md | 2026-04-01-fangraphs-column-map.md | Rename |
| K26_MATCHUP_CATEGORY_ALIGNMENT_SPEC.md | 2026-04-01-matchup-category-alignment.md | Rename |
| K27_RAW_INGESTION_AUDIT.md | 2026-04-05-raw-ingestion-audit.md | Rename |
| K_A_BDL_STATS_SPEC.md | 2026-04-07-bdl-stats-spec.md | Rename |
| K_B_IDENTITY_RESOLUTION_SPEC.md | 2026-04-07-identity-resolution-spec.md | Rename |

### Phase 5: Superpowers Plans

| Source | Destination | Reason |
|--------|-------------|--------|
| docs/superpowers/plans/2026-04-10-data-quality-fixes.md | docs/superpowers/completed/ | Tasks 1-11 complete |
| docs/superpowers/plans/2026-04-09-data-quality-remediation.md | docs/superpowers/completed/ | Tasks 1-11 complete |
| docs/superpowers/plans/2026-04-09-position-eligibility-remediation.md | docs/superpowers/completed/ | Complete |
| docs/superpowers/plans/2026-03-31-fantasy-baseball-perf-fixes.md | docs/superpowers/completed/ | Complete |
| docs/superpowers/plans/2026-04-04-fantasy-edge-decoupling-structural.md | (keep) | Still active |

---

## 5. RISK ASSESSMENT

### Critical Documents (DO NOT MOVE/DELETE)

| Document | Reason |
|----------|--------|
| AGENTS.md | Core authority - referenced everywhere |
| HANDOFF.md | Active operational state |
| HEARTBEAT.md | Referenced by AGENTS.md |
| IDENTITY.md | Referenced by AGENTS.md |
| ORCHESTRATION.md | Referenced by AGENTS.md |
| CLAUDE_ARCHITECT_PROMPT_MARCH28.md | Active emergency reference |
| MASTER_DOCUMENT_INDEX.md | Navigation hub - update in place |
| QUICKREF.md | Operational reference |

### Cross-Reference Risk

**Files that may link to others:**
- MASTER_DOCUMENT_INDEX.md (links to ~20 docs)
- AGENTS.md (references HANDOFF, HEARTBEAT, IDENTITY, ORCHESTRATION)
- HANDOFF.md (references many reports and docs)
- CLAUDE*.md files (may reference each other)

**Mitigation:** After all moves, grep for broken `.md` links

---

## 6. EXECUTION CHECKLIST

### Phase 1: Analysis (COMPLETED)
- [x] Inventory all 250 markdown files
- [x] Identify version chains and duplicates
- [x] Categorize CLAUDE*.md files by status
- [x] Map superpowers plans by completion status
- [x] Identify naming inconsistencies
- [x] Create this plan document

### Phase 2: Root Level Consolidation
- [ ] Create `.claude/prompts/` directory
- [ ] Create `.claude/prompts/archive/` directory
- [ ] Move deprecated prompts to archive/
- [ ] Move active prompts to prompts/
- [ ] Create `CLAUDE_PROMPTS_INDEX.md`
- [ ] Update moved files with redirect headers
- [ ] Commit: "Phase 2: Consolidate CLAUDE prompts"

### Phase 3: Operational Documents
- [ ] Create `docs/incidents/` directory
- [ ] Create `docs/archive/incidents/` directory
- [ ] Consolidate ops_whip files into single doc
- [ ] Archive original ops_whip versions with headers
- [ ] Create `docs/archive/plans/` directory
- [ ] Move EXECUTION_PLAN_K*.md to archive/plans/
- [ ] Commit: "Phase 3: Consolidate operational docs"

### Phase 4: Reports Standardization
- [ ] Rename K-prefixed reports to dated format
- [ ] Rename undated reports (if date determinable)
- [ ] Create `reports/README.md` with index
- [ ] Commit: "Phase 4: Standardize report naming"

### Phase 5: Superpowers Archival
- [ ] Create `docs/superpowers/completed/` directory
- [ ] Move completed plans to completed/
- [ ] Update docs/superpowers/README.md
- [ ] Commit: "Phase 5: Archive completed plans"

### Phase 6: INDEX Update
- [ ] Update MASTER_DOCUMENT_INDEX.md
- [ ] Add archive structure section
- [ ] Add documentation maintenance section
- [ ] Commit: "Phase 6: Update master index"

### Phase 7: Validation
- [ ] Grep for broken .md links
- [ ] Verify no critical files lost
- [ ] Count files before/after
- [ ] Final commit: "Phase 7: Documentation cleanup complete"

---

## 7. POST-CLEANUP FILE COUNT TARGETS

| Location | Current | Target | Change |
|----------|---------|--------|--------|
| Root (*.md) | ~40 | ~25 | -15 (moved prompts) |
| docs/superpowers/plans/ | 5 | 1 | -4 (archived) |
| docs/ (new structure) | - | - | +incidents, +archive |
| reports/ | ~100 | ~95 | -5 (renamed, not deleted) |
| .claude/prompts/ | 0 | 13 | +13 (organized) |
| **Total** | **250** | **~190** | **~24% reduction** |

---

## 8. DOCUMENTATION MAINTENANCE POLICY (NEW)

**To be added to MASTER_DOCUMENT_INDEX.md:**

```markdown
## Documentation Maintenance Policy

### Naming Conventions
- **Reports:** `YYYY-MM-DD-descriptive-name.md`
- **Plans:** `YYYY-MM-DD-brief-description.md`
- **Incidents:** `YYYY-MM-DD-incident-name.md`

### Archive Rules
1. Plans move to `completed/` when all tasks done
2. Superseded docs get archive header, move to `archive/`
3. Version chains consolidate to single authoritative doc
4. Archive at end of each sprint

### Responsibilities
- **Claude:** Maintain MASTER_DOCUMENT_INDEX.md
- **Kimi:** Archive completed research plans
- **All:** Follow naming convention for new docs
```

---

**Plan Version:** 1.0  
**Ready for Execution:** YES  
**Estimated Time:** 11-14 hours  
**Risk Level:** LOW (with preservation rules)
