# Layer 2 Certification Design

> **Date:** April 15, 2026
> **Author:** Claude Code (Master Architect)
> **Status:** Design Complete — Pending Implementation Planning
> **Scope:** Layer 2 (Data and Adaptation) certification for MLB Fantasy Platform

---

## Executive Summary

This design establishes a **critical-path-first certification process** for Layer 2 completion. The architecture follows a strict dependency chain where each stage operates on a confirmed truth surface before proceeding. No parallel analysis across potentially misaligned realities.

**Core principle**: Deployment freshness is the truth gate. Until production is confirmed to match the repo, no analytical work, validation design, or implementation should proceed — all downstream observations would be contaminated.

**Completion model**: Binary pass/fail. Layer 2 is either complete (all 6 substantive criteria satisfied + certification record present) or blocked. No caveated completion, no partial credit.

---

## Architecture Overview

```
[Deployment Truth] → [Gap Analysis] → [Validation Framework] → [Gap Closure] → [Certification] → [Completion Marker]
         ↓                    ↓                  ↓                  ↓              ↓                ↓
      Gemini              Kimi               Kimi              Claude          Gemini          Claude
```

**Stage Flow**:

```
Stage 1 (Gemini) → [Claude confirms] → Stage 2 (Kimi) → [Claude confirms] →
Stage 3 (Kimi) → [Claude confirms] → Stage 4 (Claude) → [Claude self-approves] →
Stage 5 (Gemini) → [Claude confirms] → Stage 6 (Kimi, conditional) →
Completion Marker (Claude)
```

---

## Canonical Acceptance Criteria

From HANDOFF.md § "Layer 2 Acceptance Criteria":

Layer 2 is not complete until **all** of the following are true:

1. Production is confirmed to be running the latest repo code.
2. `data_ingestion_logs` has recent durable rows from real job runs.
3. `/admin/pipeline-health` and `/admin/validation-audit` correctly degrade on empty critical tables.
4. `probable_pitchers` contains usable rows, or a documented source outage explains a zero-row run with log evidence.
5. Raw MLB source tables used by the system are fresh and internally consistent.
6. Weather and park context are persisted canonically rather than trapped in request-time logic.
7. A short Layer 2 completion note is added here before any Layer 3 work is activated.

**Revised Certification Model**:

- **Criteria 1-6**: Substantive requirements validated by Stage 5 (Gemini)
- **Criterion 7**: Completion marker added by Claude **after** Stage 5 PASS — as a **consequence** of certification, not a prerequisite

**Layer 2 PASS**: Criteria 1-6 all TRUE, AND Stage 5 certification record is present.
**Layer 2 BLOCKED**: Any of criteria 1-6 is FALSE, OR Stage 5 certification is absent/incomplete.

---

## Stage Specifications

### Stage 1: Deployment Truth Establishment

**Owner**: Gemini CLI

**Objective**: Confirm production is running the latest repo code.

**Method**:
1. Redeploy current repo state to Railway
2. Query canonical version endpoint for deployment fingerprint
3. Run the validation queries from HANDOFF.md
4. Report factual production state only

**Canonical Deployment Fingerprint**:

Production exposes `/admin/version` (new endpoint) returning:

```json
{
  "git_commit_sha": "abc123def456...",
  "git_commit_date": "2026-04-15T10:30:00Z",
  "build_timestamp": "2026-04-15T10:31:15Z",
  "app_version": "dev"
}
```

Stage 1 verifies: `git_commit_sha` matches `git rev-parse HEAD` from repo.

**First-Cycle Bootstrap Rule**:

If `/admin/version` returns 404 or does not exist (first certification cycle):
1. Fallback: Compare deployed startup logs timestamp against repo commit timestamp
2. Add creation of `/admin/version` endpoint to Stage 4 implementation scope
3. Proceed with Stage 1 using fallback method; Stage 5 will use canonical endpoint after Stage 4

**Exit Criteria** (three-state):

| State | Meaning | Next Step |
|-------|---------|-----------|
| **STALE** | `git_commit_sha` does not match repo HEAD | Loop at Stage 1 |
| **FRESH but AMBIGUOUS** | SHA matches, but other observations inconsistent/incomplete | Claude requests targeted rerun OR explicitly marks ambiguity for Stage 2 |
| **FRESH and CLEAR** | SHA matches, observations are consistent | Proceed to Stage 2 |

**Deliverable**: HANDOFF.md update with:
1. Deployment verdict (STALE / FRESH-AMBIGUOUS / FRESH-CLEAR) with evidence
2. Production `git_commit_sha` and repo HEAD SHA
3. Raw production observations (row counts, timestamps, endpoint behaviors)
4. If AMBIGUOUS: explicit ambiguity note that Stage 2 must account for

**Validation Queries** (from HANDOFF.md):
```bash
# Canonical deployment fingerprint (PRIMARY)
railway ssh python -c "import json, os, requests; base=os.getenv('API_URL') or os.getenv('NEXT_PUBLIC_API_URL') or 'https://cbb-edge-production.up.railway.app'; key=os.getenv('API_KEY') or os.getenv('ADMIN_API_KEY') or os.getenv('X_API_KEY'); headers={'X-API-Key': key} if key else {}; r=requests.get(f'{base}/admin/version', headers=headers, timeout=10); print(r.status_code); print(json.dumps(r.json(), indent=2))"

# data_ingestion_logs row count
railway ssh python scripts/devops/db_query.py "SELECT COUNT(*) AS row_count, MAX(started_at) AS latest_started_at, MAX(completed_at) AS latest_completed_at FROM data_ingestion_logs;"

# newest ingestion log rows
railway ssh python scripts/devops/db_query.py "SELECT job_type, status, target_date, started_at, completed_at, records_processed, records_failed, error_message FROM data_ingestion_logs ORDER BY started_at DESC LIMIT 15;"

# probable_pitchers row count
railway ssh python scripts/devops/db_query.py "SELECT COUNT(*) AS row_count, MAX(game_date) AS latest_game_date FROM probable_pitchers;"

# sample probable_pitchers rows
railway ssh python scripts/devops/db_query.py "SELECT game_date, team, pitcher_name, bdl_player_id, mlbam_id, opponent, is_home, is_confirmed, created_at FROM probable_pitchers ORDER BY game_date ASC, team ASC LIMIT 20;"

# pipeline-health endpoint
railway ssh python -c "import json, os, requests; base=os.getenv('API_URL') or os.getenv('NEXT_PUBLIC_API_URL') or 'https://cbb-edge-production.up.railway.app'; key=os.getenv('API_KEY') or os.getenv('ADMIN_API_KEY') or os.getenv('X_API_KEY'); headers={'X-API-Key': key} if key else {}; r=requests.get(f'{base}/admin/pipeline-health', headers=headers, timeout=30); print(r.status_code); print(json.dumps(r.json(), indent=2))"

# validation-audit endpoint
railway ssh python -c "import json, os, requests; base=os.getenv('API_URL') or os.getenv('NEXT_PUBLIC_API_URL') or 'https://cbb-edge-production.up.railway.app'; key=os.getenv('API_KEY') or os.getenv('ADMIN_API_KEY') or os.getenv('X_API_KEY'); headers={'X-API-Key': key} if key else {}; r=requests.get(f'{base}/admin/validation-audit', headers=headers, timeout=60); print(r.status_code); data=r.json(); print(json.dumps({'critical': data.get('critical', []), 'high': data.get('high', []), 'medium': data.get('medium', []), 'low': data.get('low', []), 'info': data.get('info', [])}, indent=2))"
```

**Stage 5 Reuse**: Stage 5 uses the **same** `/admin/version` endpoint as Stage 1 — single source of truth for deployment freshness.

---

### Stage 2: Layer 2 Gap Analysis

**Owner**: Kimi CLI

**Objective**: Compare spec vs. repo behavior vs. live production to identify the real gaps.

**Input**: Fresh production state from Stage 1 (must be FRESH-CLEAR or FRESH-AMBIGUOUS with explicit notes)

**Scope**:
- All Layer 2 components: ingestion orchestrator, logging, health semantics, probable-pitcher fallback, raw table population, canonical context persistence
- For each component:
  - What is the contract (spec)?
  - What does the code do (repo)?
  - What does production show (live)?
- Identify: silent failures, partial wiring, missing dependencies, misleading signals

**Output Mapping**:

| Component | Contract | Repo Behavior | Production Behavior | Gap Classification |
|-----------|----------|---------------|---------------------|-------------------|
| Ingestion orchestrator | Should leave durable logs | Code exists | 0 rows observed | Deploy gap or wiring bug |
| Health semantics | Should degrade on empty tables | Degradation logic exists | Returns healthy on empty | Deploy gap or logic bug |
| Probable pitchers | Should populate or fail explicitly | Fallback code exists | 0 rows, no error | Source outage or wiring gap |
| ... | ... | ... | ... | ... |

**Gap Classifications**:
- **Deploy gap**: Code exists in repo but not reflected in production
- **Wiring bug**: Integration not connected in code
- **Missing feature**: No implementation exists for contract requirement
- **Source outage**: External dependency unavailable
- **Silent failure**: Error path produces no observable evidence

**Deliverable**: `reports/layer-2-gap-analysis-{date}.md` with:
- Component-by-component gap table
- Gap classification for each
- Prioritized remediation list (grouped by gap class)
- Mapping to acceptance criteria (which gaps block which criteria)

---

### Stage 3: Validation Framework Design

**Owner**: Kimi CLI

**Objective**: Define the exact validation needed to certify Layer 2 completion with high confidence.

**Input**: Gap analysis from Stage 2

**Method**:
- Map each Layer 2 acceptance criterion (1-6) to specific tests/probes
- Define success/failure thresholds for each
- Identify what must be automated vs. manual verification
- Design repeatable certification process
- **NON-DESTRUCTIVE**: No validation should require mutating live production data

**Validation Mapping Structure**:

| Acceptance Criterion | Validation Method | Test Specification | Success Threshold | Failure Behavior |
|---------------------|-------------------|-------------------|-------------------|------------------|
| 1. Production latest | `/admin/version` endpoint | Query SHA, compare to repo HEAD | Exact match | STALE, loop to Stage 1 |
| 2. Ingestion logs | Database query + job trigger | Query row count, trigger job, re-query | >0 rows with recent timestamps | BLOCKED, investigate wiring |
| 3. Health degradation | Diagnostic mode probe | Call `/admin/pipeline-health?diagnostic=true` with simulated empty state | Returns degraded status in diagnostic mode | BLOCKED, fix logic |
| 4. Probable pitchers | Database query + log inspection | Query rows, check logs for explicit evidence | >0 usable rows OR documented outage | BLOCKED, investigate |
| 5. Raw source tables | Database query + freshness check | Query row counts, max dates | All tables fresh (<48h stale) | BLOCKED, investigate |
| 6. Context persisted | Database query + code inspection + consumer verification | Query weather/park tables, verify schema, identify at least one real consumer | Tables exist, have schema, AND at least one non-test consumer queries persisted context | BLOCKED, implement |

**Criterion 6 — Consumer Proof Requirement**:

Validation must confirm that persisted weather/park context is actually consumed by at least one real (non-test) code path. Test code does not count. The requirement is satisfied when:
1. Tables exist with correct schema
2. At least one production code path (not in `tests/`) queries these tables for use in scoring, optimization, or decision logic

**Criterion 3 — Non-Destructive Validation**:

**Problem**: We cannot empty critical tables in production just to test degradation.

**Solution — Diagnostic Mode**:

1. **Repo-level contract test** (runs in test suite, not production):
   - Fixture: mock empty critical tables
   - Test: verify `/admin/pipeline-health` returns `overall_healthy: false`
   - This proves the degradation branch works at code level

2. **Production diagnostic probe** (runs in production without mutation):
   - `/admin/pipeline-health?diagnostic=true` accepts a `simulate_empty={table_names}` query param
   - Endpoint evaluates: "IF tables were empty, what would health status be?"
   - Returns `{"diagnostic_mode": true, "simulated_health": {...}, "actual_health": {...}}`
   - No tables are actually emptied — this is a dry-run evaluation

3. **Combined proof**: Contract test proves degradation logic + diagnostic probe proves logic is reachable in deployed binary = complete certification without destructive production state mutation

**Deliverable**: `reports/layer-2-validation-framework-{date}.md` with:
- Criterion → validation mapping
- Test specifications (what to probe, how to interpret)
- Success/failure thresholds
- Repeatable certification checklist
- Expected execution time
- Explicit note on non-destructive validation for Criterion 3

---

### Stage 4: Gap Closure

**Owner**: Claude Code

**Objective**: Implement **only** the repo changes required to satisfy the 6 substantive Layer 2 acceptance criteria.

**Input**: Gap analysis from Stage 2 (remediation list), validation framework from Stage 3 (test specs)

**Scope Boundary**:
- **IN**: Fixes directly required for acceptance criteria 1-6
- **OUT**: Any opportunistic expansion, nice-to-haves, or "while we're here" improvements

**Layer 5 Exception — Minimal Rewiring Allowed**:

Criterion 6 requires weather/park context to be persisted rather than trapped in request-time logic. If existing consumers in Layer 5 currently call request-time functions, **minimal rewiring is permitted** to consume persisted context instead. This is not a scope change — it's the minimal adaptation required to satisfy the criterion. HANDOFF.md already allows "changes that expose Layer 2 truth" in Layer 5.

**Implementation Mapping** (example):

| Gap | Acceptance Criterion | Implementation Scope | Test Method |
|-----|---------------------|---------------------|-------------|
| `/admin/version` endpoint missing | 1 | Add endpoint returning git SHA, build timestamp | Query SHA, compare to repo |
| data_ingestion_logs empty | 2 | Verify orchestrator writes logs, add if missing | Run job, query table |
| Health degradation logic | 3 | Fix degradation logic, add diagnostic mode | Contract test + diagnostic probe |
| probable_pitchers wiring | 4 | Connect fallback logic, add explicit failure | Trigger sync, check logs |
| Weather/park not persisted | 6 | Create tables, wire into pipeline, minimal Layer 5 consumer rewiring | Query tables, verify schema, check consumers |
| ... | ... | ... | ... |

**Dependency Discovery** (split):

| Class | Example | Resolution |
|-------|---------|------------|
| **Implementation-local** | Missing import, minor wiring fix, schema edge case | Fix stays in Stage 4; no loop |
| **Scope-changing** | New contract required, architectural boundary crossed, acceptance criteria reinterpretation needed | Loop back to Stage 2; update HANDOFF.md with revised scope |
| **Layer 5 truth-exposure** | Rewiring Layer 5 to consume persisted context per criterion 6 | Permitted in Stage 4; not scope-changing |

**Rule**: Only scope-changing dependencies trigger Stage 2 loop. Implementation-local and truth-exposure changes are resolved within Stage 4.

**Constraints**:
- No new features beyond the 6 substantive acceptance criteria
- No changes to frozen layers (3-4) except as required for criteria 1-6
- Layer 5 changes limited to truth-exposure and context consumption per criterion 6
- All changes tested before marked complete
- PyCompile passes for all modified files
- Relevant tests pass

**Deliverable**:
- Repo changes **committed and validated**
- HANDOFF.md updated with implementation summary
- Evidence that each change satisfies its criterion

---

### Stage 5: Certification Validation

**Owner**: Gemini CLI

**Objective**: Run the validation framework from Stage 3 against the closed implementation.

**Input**: Validation framework from Stage 3, implementation from Stage 4

**Method**: Execute the certification checklist and report pass/fail for each substantive acceptance criterion (1-6).

**Execution**:
1. Run each test specification from validation framework
2. Record actual results vs. expected thresholds
3. Document any deviations or failures
4. Produce final verdict

**Deliverable**: HANDOFF.md update with:

```markdown
## Layer 2 Certification Results

Date: [Date]
Validator: Gemini CLI
Validation Framework: [link to Stage 3 output]

Per-Criterion Results:
✓ 1. Production confirmed running latest repo code — PASS (SHA: [commit])
✓ 2. data_ingestion_logs has recent durable rows — PASS ([N] rows, latest: [timestamp])
✓ 3. Health endpoints degrade correctly — PASS (diagnostic probe verified)
✓ 4. probable_pitchers usable or documented outage — [PASS/BLOCKED]
✓ 5. Raw MLB source tables fresh and consistent — PASS
✓ 6. Weather/park context persisted canonically — PASS (tables verified, consumer confirmed: [path])

Overall Verdict: [PASS / BLOCKED]

If PASS: Criteria 1-6 all satisfied. Certification record present. Awaiting completion marker from Claude.
If BLOCKED: Remaining gaps are [list].
```

**Key Change**: Stage 5 certifies criteria 1-6 and produces the certification record. It does **not** wait for criterion 7 (completion marker) — that is added by Claude as the final step after Stage 5 PASS.

---

### Stage 6: Source Research (Conditional)

**Owner**: Kimi CLI

**Trigger**: Only if Stage 5 confirms criterion 4 (probable_pitchers) is blocked due to **source coverage**, not deploy/wiring issue.

**Objective**: Research alternative data sources and produce procurement recommendation.

**Scope**:
- Evaluate alternative probable pitcher sources: MLB.com, ESPN, Baseball Press, Rotowire, etc.
- Assess: API availability, data quality, licensing terms, integration complexity
- Compare to current implementation (BDL gap, MLB Stats API limitation)

**Analysis Template**:

| Source | Availability | Data Quality | Licensing | Integration Complexity | Recommendation |
|--------|--------------|--------------|-----------|----------------------|----------------|
| MLB.com (official) | Public API, no key | High (official) | Fair use | Low | [Primary recommendation] |
| ESPN | Web scrape required | Medium | Unknown | Medium | [Fallback] |
| Baseball Press | Scrape or API | Medium | Unknown | Low | [Fallback] |
| ... | ... | ... | ... | ... | ... |

**Deliverable**: `reports/probable-pitcher-source-alternatives-{date}.md` with:
- Source evaluation matrix
- Procurement recommendation (primary + fallbacks)
- Integration effort estimate
- Risk assessment for each option

**Important**: Stage 6 **cannot by itself authorize a caveated Layer 2 pass**. If no viable source alternatives exist:
- **Option 1**: Document as a permanent known limitation; Layer 2 remains BLOCKED
- **Option 2**: Claude explicitly revises HANDOFF.md acceptance criteria — this is an **architectural decision**, not a soft exception
- No silent caveats, no "complete but..." status

Only Claude (as architect) can choose Option 2, and only after explicit documentation.

---

### Completion Marker (Post-Certification)

**Owner**: Claude Code

**Trigger**: Stage 5 returns PASS (all criteria 1-6 satisfied)

**Precondition**: Claude has reviewed Stage 5 certification results and explicitly confirmed all criteria 1-6 are TRUE in HANDOFF.md

**Post-Stage-5 State Machine**:

```
[Gemini: Stage 5 Complete]
       ↓
[Claude: Review Stage 5 Results] — MUST explicitly confirm "criteria 1-6 all TRUE" in HANDOFF.md
       ↓
[Claude: Add Completion Marker] — Only after explicit confirmation above
       ↓
[Layer 2: COMPLETE] — Layers 3-6 unblocked
```

This interval is **defined explicitly**, not implicit. Claude must not add the completion marker without first documenting the confirmation in HANDOFF.md.

**Objective**: Add the Layer 2 completion marker to HANDOFF.md, formally unblocking Layers 3-6.

**Method**: After reviewing Stage 5 certification results, Claude adds:

```markdown
## Layer 2 Status: COMPLETE

Certified: [Date]
Validation: [Link to Stage 5 output]
Authorization: Layers 3-6 are now unblocked for new work.

All acceptance criteria satisfied:
✓ 1. Production confirmed running latest repo code (SHA: [commit])
✓ 2. data_ingestion_logs has recent durable rows
✓ 3. Health endpoints degrade correctly
✓ 4. probable_pitchers usable or documented outage
✓ 5. Raw MLB source tables fresh and consistent
✓ 6. Weather/park context persisted canonically (consumer: [path])
✓ 7. Completion note added (this section)
```

**This resolves the circularity**: The completion marker is added **after** Stage 5 PASS, not as a prerequisite for Stage 5 to run.

---

## Agent Coordination Protocol

### Handoff Contract

Per-stage handoff rules:

1. The current stage owner updates HANDOFF.md with the stage deliverable in the required format.
2. The next stage does not begin until that deliverable is present and explicitly marked complete.
3. If a stage returns stale, blocked, or ambiguous results, the process loops at that stage boundary rather than allowing downstream work to continue.
4. Only one agent owns the active stage at a time.
5. **Claude remains the control point** for interpreting whether a stage output is sufficient to unlock the next stage.

### Loop Behavior

| Loop Trigger | Resolution Path | Example |
|--------------|-----------------|---------|
| Stage 1 returns STALE | Loop at Stage 1 until FRESH | Redeploy failed, must retry |
| Stage 1 returns FRESH-AMBIGUOUS | Claude requests targeted rerun OR marks ambiguity | Inconsistent observations need clarification |
| Stage 2 reveals deploy gap | Loop back to Stage 1 | Code exists but not in production |
| Stage 2 gap analysis ambiguous | Claude reviews, may request targeted re-analysis | Gap classification unclear |
| Stage 3 validation incomplete | Claude identifies gaps, Kimi extends framework | Missing test for criterion 4 |
| Stage 4 discovers scope-changing dependency | Loop back to Stage 2, update HANDOFF.md | New contract required |
| Stage 4 discovers Layer 5 truth-exposure need | Permitted in Stage 4, no loop | Minimal consumer rewiring per criterion 6 |
| Stage 5 certification fails | Specific failures route to Stage 4; structural issues to Stage 2 | Criterion 3 fails: health logic bug |
| Stage 5 confirms probable_pitchers is source problem | Proceed to Stage 6 | MLB Stats API returns 0 |
| Stage 5 returns PASS | Claude adds completion marker | Criteria 1-6 satisfied |

### Error Handling

| Stage | Failure Mode | Resolution Path |
|-------|--------------|-----------------|
| 1 | Deployment fails, remains stale | Loop at Stage 1; Claude escalates to operator if needed |
| 1 | Fresh but ambiguous | Claude requests targeted rerun OR explicitly marks ambiguity for Stage 2 |
| 2 | Gap analysis returns ambiguous | Claude reviews findings; may request targeted re-analysis |
| 3 | Validation design incomplete | Claude identifies gaps; Kimi extends framework |
| 4 | Implementation-local dependency | Fix stays in Stage 4; no loop |
| 4 | Scope-changing dependency | Loop back to Stage 2; update HANDOFF.md |
| 5 | Certification fails (specific) | Route back to Stage 4 for fix |
| 5 | Certification fails (structural) | Route back to Stage 2 for gap re-analysis |
| 6 | No viable source alternatives | Document as known limitation; Layer 2 remains BLOCKED unless Claude explicitly revises acceptance criteria |

---

## Completion Definition

### Binary Pass/Fail

Layer 2 is either:
- **COMPLETE**: Criteria 1-6 all TRUE, Stage 5 certification record present, completion marker added
- **BLOCKED**: Any of criteria 1-6 is FALSE, OR Stage 5 certification is absent/incomplete

No partial credit, no caveated completion.

### Certification Flow

```
[Gemini: Stage 5] → Produces certification record (criteria 1-6 results)
       ↓
[Claude: Review] → Confirms all 1-6 TRUE; documents in HANDOFF.md
       ↓
[Claude: Add Marker] → Adds completion marker to HANDOFF.md
       ↓
[Layer 2: COMPLETE] → Layers 3-6 unblocked
```

**Critical**: The "Claude: Review" step must explicitly document the TRUE confirmation in HANDOFF.md before the completion marker is added. This creates an auditable handoff interval rather than an implicit transition.

### Exception Handling

If a criterion cannot be satisfied (e.g., criterion 4 permanently unavailable):

- **Option 1**: Keep Layer 2 BLOCKED; document as permanent limitation
- **Option 2**: Claude explicitly revises HANDOFF.md acceptance criteria — this is an **architectural decision**, not a soft exception
- No silent caveats, no "complete but..." status

---

## Agent Responsibilities Summary

| Agent | Stages | Core Responsibilities |
|-------|--------|---------------------|
| **Claude Code** | Inter-stage gates, Stage 4, Completion Marker | Control point for handoffs; gap closure implementation; acceptance criteria interpretation; HANDOFF.md updates; adds completion marker after Stage 5 PASS |
| **Gemini CLI** | Stage 1, Stage 5 | DevOps truth-establishment; deployment; production validation; factual reporting only; uses `/admin/version` as canonical fingerprint |
| **Kimi CLI** | Stage 2, Stage 3, Stage 6 (conditional) | Deep analysis; gap audit; validation framework design (including non-destructive Criterion 3 approach); source research |

---

## Design Rationale

**Why Critical-Path-First?**

Deployment freshness gates everything. If production is stale, every downstream observation is contaminated: gap analysis, validation design, probable-pitcher diagnosis, and even judgments about production readiness.

**Why Synchronous Handoffs?**

Each stage depends on a confirmed output from the prior stage. Parallelism would reintroduce the exact failure mode we're eliminating: agents reasoning from different realities at the same time.

**Why Binary Pass/Fail?**

The hard gate doctrine is weakened if we allow caveated completion. If a criterion truly cannot be satisfied, that should be a conscious architectural decision with explicit HANDOFF.md revision, not a soft default.

**Why `/admin/version` as Canonical Fingerprint?**

- Single source of truth for deployment freshness
- Stage 1 and Stage 5 use the same endpoint — no ambiguity
- Deterministic comparison: SHA match or no match
- No judgment calls about "stale strings" or "stale behavior"

**Why Non-Destructive Validation for Criterion 3?**

- We cannot empty critical tables in production just to test degradation
- Diagnostic mode (`?diagnostic=true`) simulates empty state without mutation
- Repo-level contract test proves degradation logic works
- Combined proof without operational risk

**Why Separate Completion Marker from Stage 5?**

- Resolves circularity: Stage 5 certifies 1-6, Claude adds marker after
- Clear separation: validation (Gemini) vs. declaration (Claude)
- Marker is consequence of PASS, not prerequisite for it

**Why Allow Minimal Layer 5 Rewiring for Criterion 6?**

- Criterion 6 requires context to be persisted rather than request-time
- If Layer 5 consumers currently use request-time functions, they must be rewired
- This is not scope expansion — it's the minimal change needed to satisfy the criterion
- HANDOFF.md already allows Layer 5 "changes that expose Layer 2 truth"

**Why Kimi for Analysis, Gemini for Ops?**

- **Kimi**: 1M-token context window for comprehensive gap analysis and framework design
- **Gemini**: DevOps expertise for Railway deployment and production validation
- **Claude**: Architecture authority and implementation control

---

## Next Steps

1. **Immediate**: Gemini executes Stage 1 (Deployment Truth Establishment)
2. **After Stage 1**: Claude confirms handoff, Kimi executes Stage 2 (Gap Analysis)
3. **After Stage 2**: Claude confirms handoff, Kimi executes Stage 3 (Validation Framework)
4. **After Stage 3**: Claude executes Stage 4 (Gap Closure)
5. **After Stage 4**: Gemini executes Stage 5 (Certification)
6. **After Stage 5 PASS**: Claude adds completion marker
7. **If needed**: Kimi executes Stage 6 (Source Research)

---

*Design document complete. Ready for implementation planning via writing-plans skill.*
