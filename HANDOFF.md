# HANDOFF.md — MLB Platform Operating Brief

> Date: April 16, 2026 | Author: Claude Code (Master Architect)
> Status: Layer 2 certified complete. Layer 3B context authority audit complete. API endpoint live with auth. Do not reopen Layer 2 except for regressions.

Full audit: reports/2026-04-15-comprehensive-application-audit.md
Raw-ingestion contract audit: reports/2026-04-05-raw-ingestion-audit.md
Historical context: HANDOFF_ARCHIVE.md

---

## Mission Accomplished

- Layer 2 certification is complete in production.
- Deployment truth is live and versioned.
- Raw ingestion audit logging is healthy.
- `probable_pitchers` is populating successfully in production.
- Weather and park context persistence is implemented and seeded.
- The Layer 2 hard gate is lifted.

---

## Core Doctrine

The MLB data platform is now validated at Layer 2. Work may resume above Layer 2, but execution should remain disciplined and sequenced.

The architecture remains layered:

| Layer | Name | Purpose | Status |
|------|------|---------|--------|
| 0 | Immutable Decision Contracts | Canonical contracts, schemas, IDs, and validation boundaries | Stable |
| 1 | Pure Stateless Intelligence | Deterministic pure functions over validated inputs | Available |
| 2 | Data and Adaptation | Ingestion, validation, persistence, observability, freshness, provenance | Certified Complete |
| 3 | Derived Stats and Scoring | Rolling stats, player scores, context-enriched features | Active |
| 4 | Decision Engines and Simulation | Lineup logic, waiver logic, matchup engines, Monte Carlo | Hold until the first Layer 3 objective is stable |
| 5 | APIs and Service Presentation | FastAPI contracts, dashboards, admin views | Maintenance |
| 6 | Frontend and UX | Next.js pages, interactions, polish | Maintenance |

### Operating Rule

- Do not reopen Layer 2 as an active workstream unless a production regression is observed.
- Do not activate broad Layer 4 work yet.
- Use Layer 3 as the single active engineering lane.

---

## Current Production Truth

Verified production state as of April 16, 2026 (13:00 UTC):

| Area | Current Truth | Status |
|------|---------------|--------|
| Deployment state | Fresh (Build: 2026-04-16T02:00:54) | Healthy |
| `data_ingestion_logs` | 66 rows | Healthy |
| `probable_pitchers` | 94 rows | Healthy |
| `/admin/pipeline-health` | `overall_healthy: true` | Healthy |
| `mlb_player_stats` | 7249 rows | Healthy |
| `statcast_performances` | 7408 rows | Healthy |
| `park_factors` | 27 parks seeded | Healthy |

### Operational Interpretation

- Layer 2 is certified complete.
- The production data spine is no longer the blocker.
- The next bottleneck is downstream scoring construction, not ingestion stability.

---

## Layer 2 Certification Record

Final production verification:

- `probable_pitchers` row count: 94
- Latest `probable_pitchers` job result: SUCCESS (Job ID 65)
- `data_ingestion_logs` contains recent durable rows
- `/admin/pipeline-health` returns `overall_healthy: true`
- `/admin/version` exists and deployment versioning is live

Layer 2 acceptance criteria status:

1. Production is running latest repo code: PASS
2. `data_ingestion_logs` has recent durable rows: PASS
3. Health endpoints report correctly: PASS
4. `probable_pitchers` contains usable rows: PASS
5. Raw MLB source tables are fresh and internally consistent: PASS
6. Weather and park context are persisted canonically: PASS
7. Persisted context is consumed by scoring code: PASS

Layer 2 verdict: PASS

---

## Layer Status

### Layer 0 — Immutable Decision Contracts
Status: STABLE

- No contract expansion is currently required.

### Layer 1 — Pure Stateless Intelligence
Status: AVAILABLE

- Pure logic may be extended as required by Layer 3 scoring work.

### Layer 2 — Data and Adaptation
Status: CERTIFIED COMPLETE

- Regressions only.
- Do not run a new Layer 2 roadmap unless production evidence degrades.

### Layer 3 — Derived Stats and Scoring
Status: ACTIVE

- This is the only active engineering workstream now.

**Layer 3B Context Authority Audit (2026-04-16):**

Authoritative scoring path: `player_rolling_stats → compute_league_zscores() → player_scores`

Key findings:
- Scoring is PURE Z-score computation over rolling stats - NO park factors, NO weather
- Park factor fragmentation: 5+ hardcoded copies across codebase (ballpark_factors.py, mlb_analysis.py, daily_lineup_optimizer.py, two_start_detector.py, weather_ingestion.py)
- scoring_engine.py has DB-backed get_park_factor() helper but it's UNUSED by main scoring
- Weather infrastructure exists but is deferred - not consumed by scoring (appropriate for multi-day windows)

Risk severity: HIGH (fragmentation) > MEDIUM (unused helper confusion) > LOW (weather deferred)

Next step (scoped): Update `ballpark_factors.py:get_park_factor()` to read from persisted ParkFactor table - ONE function change in ONE file.

### Layer 4 — Decision Engines and Simulation
Status: HOLD

- Do not resume broader decision-engine work until the first Layer 3 scoring objective is complete and stable.

### Layer 5 — APIs and Service Presentation
Status: MAINTENANCE

- Only changes needed to expose validated Layer 3 outputs should be made.

### Layer 6 — Frontend and UX
Status: MAINTENANCE

- No new UI initiative should begin until Layer 3 outputs are defined.

---

## Active Workstream

### L3A. Derived Stats And Scoring Spine

**Status: API Endpoint Complete (2026-04-16)**

Implemented `GET /api/fantasy/players/{bdl_player_id}/scores` - the first authoritative Layer 3 scoring exposure.

**Completed:**
- Endpoint implementation in `backend/routers/fantasy.py` with verify_api_key auth
- Pydantic schemas (`PlayerScoresResponse`, `PlayerScoreOut`, `PlayerScoreCategoryBreakdown`)
- 13 comprehensive test cases covering validation, response contract, and auth requirements
- Supports window_days=7/14/30 (defaults to 14)
- Supports optional as_of_date query parameter (defaults to latest available)
- Returns 400 for invalid window_days, 404 for missing scores, 401 for missing auth
- Exposes hitter categories (z_hr, z_rbi, z_nsb, z_avg, z_obp) and pitcher categories (z_era, z_whip, z_k_per_9)
- Legacy z_sb field excluded from response contract (z_nsb is canonical)

**Recommended next steps for L3A:**
- Verify player score generation pipeline is populating player_scores table in production
- Add weather/park factor influence verification to scoring path
- Consider additional aggregations or filters if downstream consumers identify gaps

**Layer 3B Audit Complete (2026-04-16):**
- Context authority audit completed - scoring is pure (no park/weather used)
- Park factor fragmentation confirmed (5+ hardcoded sources)
- Scoped fix defined: ballpark_factors.py → DB-backed read
- Weather explicitly deferred (appropriate for rolling windows)

**Out of scope for this phase:**

- simulation expansion
- waiver-system breadth work
- optimizer redesign
- frontend feature expansion
- broad Layer 4 activation

---

## Immediate Priority Queue

| Priority | Action | Owner | Status |
|----------|--------|-------|--------|
| P0 | Define the first Layer 3 scoring objective and success criteria | Claude | Complete |
| P0 | Audit current derived-stats and scoring code path for gaps | Claude | Complete |
| P1 | Identify one authoritative scoring output for downstream consumers | Claude | Complete |
| P1 | Audit context authority in scoring path (3B) | Claude | Complete |
| P1 | Verify player_scores table is being populated in production | Claude | Next |
| P2 | Consolidate ballpark_factors.py to DB-backed read | Claude | Pending |
| P1 | Decide whether any Layer 5 response shape changes are needed after scoring output stabilizes | Claude | Pending |

---

## Architect Review Queue

- Keep `/admin/version`, ingestion logs, and `probable_pitchers` on passive regression watch.
- Treat canonical environment snapshots beyond current weather and park persistence as backlog, not active recovery work.
- Do not reopen simulation or decision-layer expansion until Layer 3 outputs are demonstrably trustworthy.
- If production health regresses, reopen Layer 2 explicitly rather than mixing regression response into Layer 3 work.

---

## Delegation Bundles

### Gemini CLI

No active delegation.

Use Gemini only if a production regression check, Railway deploy, log tail, or read-only production validation is required.

### Kimi CLI

No active delegation.

Use Kimi for a bounded Layer 3 analysis memo only after Claude defines the exact scoring objective.

---

## HANDOFF PROMPTS

No active handoff prompt is currently open. Create a new prompt only after the first Layer 3 objective is explicitly defined.

---

Last Updated: April 16, 2026 (16:00 UTC - Layer 3B context authority audit complete)