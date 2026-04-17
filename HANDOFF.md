# HANDOFF.md — MLB Platform Operating Brief

> Date: April 16, 2026 | Author: Claude Code (Master Architect)
> Status: Layer 2 certified complete. Layer 3B consolidation complete. Layer 3D observability complete. API endpoint live with auth. Decision pipeline observability complete. L3F (Decision Output Read Surface) complete. L3E (Market-Implied Probabilities) is deferred as a future enhancement. Do not reopen Layer 2 except for regressions.

Full audit: reports/2026-04-15-comprehensive-application-audit.md
Raw-ingestion contract audit: reports/2026-04-05-raw-ingestion-audit.md
Historical context: HANDOFF_ARCHIVE.md

---

## Mission Accomplished

- Layer 2 certification is complete in production.
- Deployment truth is live and versioned.
- Raw ingestion audit logging is healthy.
- `probable_pitchers` is populating successfully in production.
- `park_factors` is seeded and available for DB-backed reads with fallback.
- `weather_forecasts` exists in schema but remains deferred; request-time weather is the live path.
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

Verified production state as of April 16, 2026 (18:00 UTC):

| Area | Current Truth | Status |
|------|---------------|--------|
| Deployment state | Fresh (Build: 2026-04-16T02:00:54) | Healthy |
| `data_ingestion_logs` | 66 rows | Healthy |
| `probable_pitchers` | 94 rows | Healthy |
| `/admin/pipeline-health` | `overall_healthy: true` | Healthy |
| `mlb_player_stats` | 7249 rows | Healthy |
| `statcast_performances` | 7408 rows | Healthy |
| `park_factors` | 27 parks seeded, DB-backed reads active | Healthy |
| `weather_forecasts` | Table exists, EMPTY (request-time weather used instead) | Deferred |
| `/admin/diagnose-scoring/layer3-freshness` | Endpoint live, 13 tests passing | Healthy |
| `/admin/diagnose-decision/pipeline-freshness` | Endpoint live, 8 tests passing | Healthy |

### Operational Interpretation

- Layer 2 is certified complete.
- Park factor authority is DB-backed with fallback (ballpark_factors.py → ParkFactor table → PARK_FACTORS constant → neutral 1.0).
- Weather context exists in schema but is NOT populated; request-time weather (weather_fetcher.py) remains the live path for consumers like smart_lineup_selector.py.
- Layer 3 scoring (player_scores) does NOT consume weather - pure rolling-window Z-score computation remains appropriate for multi-day windows.
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
6. `park_factors` is persisted canonically; `weather_forecasts` remains deferred by design: PASS
7. Scoring code remains pure and does not depend on persisted weather context: PASS

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
- L3A (scoring spine), L3B (context authority), L3D (observability), L3F (decision read surface), and decision pipeline observability are complete.
- L3E (Market-Implied Probabilities) is preserved as backlog, deferred pending explicit policy gate.

**Layer 3B Context Authority Audit (2026-04-16):**

Authoritative scoring path: `player_rolling_stats → compute_league_zscores() → player_scores`

Key findings:
- Scoring is PURE Z-score computation over rolling stats - NO park factors, NO weather
- Park factor fragmentation: 5+ hardcoded copies across codebase (ballpark_factors.py, mlb_analysis.py, daily_lineup_optimizer.py, two_start_detector.py, weather_ingestion.py)
- scoring_engine.py has DB-backed get_park_factor() helper but it's UNUSED by main scoring
- Weather infrastructure exists (weather_ingestion.py) but weather_forecasts table is EMPTY; request-time weather (weather_fetcher.py) is the live path

Risk severity: HIGH (fragmentation) > MEDIUM (unused helper confusion) > LOW (weather table empty but request-time path exists)

**Scoped consolidation COMPLETE (2026-04-16):**
- `ballpark_factors.py:get_park_factor()` now reads from ParkFactor table first, with fallback to PARK_FACTORS constant → neutral 1.0
- 9 focused tests added (test_ballpark_factors.py)
- Weather remains deferred for Layer 3 scoring (appropriate for rolling windows; request-time weather via weather_fetcher.py serves immediate-decision needs)

### L3E. Market-Implied Probability Integration

**Status: DEFERRED — future enhancement backlog (not active)**

> **NOTE:** This work requires an explicit policy gate before becoming active. It proposes using The Odds API for MLB player props, which currently conflicts with CLAUDE.md hard-stop rules ("Do NOT use THE_ODDS_API_KEY for new MLB features"). If this lane is activated in the future, CLAUDE.md must be updated first to reflect the policy change.

**Objective:** Enrich the daily player score with forward-looking Vegas market sentiment.
The current `player_scores` table is built entirely from backward-looking rolling Z-scores
(wOBA, ISO, HR rate, etc.). Layer 3E introduces a second signal axis — what the implied
market currently believes about each player's performance for *today's* slate — and
synthesises both into an **+EV Player Score** that can weight fantasy lineup decisions.

---

#### Data-Source Decision (READ BEFORE CODING)

This is an active architectural decision with budget implications.

| Source | What it offers | Constraint |
|--------|----------------|------------|
| **BallDontLie GOAT MLB** (`/mlb/v1/odds`) | Game-level moneylines and totals per sportsbook | **Already wired.** Does NOT expose player prop markets (O/U hits, HRs, TB, etc.) |
| **The Odds API** (`/v4/sports/baseball_mlb/events/{id}/odds?markets=batter_*`) | Granular player prop lines per book | OddsAPI Basic = 20 k calls/month. This budget is currently reserved for CBB archival closing lines only. Player props require separate endpoint calls per event per market. |

**Decision required before L3E coding begins:**

1. Confirm whether the OddsAPI Basic 20 k call budget has headroom once CBB archival traffic
   drops to near-zero (it should — CBB season is closed).
2. Estimate call volume: ~15 MLB games/day × ~30 active props/game × ~1 bookmaker pass = ~450 calls/day × 30 days ≈ 13 500/month. This fits within 20 k with margin.
3. If budget is confirmed: add `MLB_PROPS_ENABLED=true` env var gate before the ingestion job runs.
4. Do NOT route game-level MLB moneylines through OddsAPI — those remain on BDL exclusively.

**Provisional decision: proceed with The Odds API for player props only.** Rationale: BDL has no player-prop endpoint, CBB archival call volume is now negligible, and 13 500 estimated calls/month fits the 20 k budget. Gate behind env var `MLB_PROPS_ENABLED` so the job is opt-in.

---

#### Engineering Task Breakdown

##### L3E-1. Pydantic Data Contracts (Layer 0 extension)

New contracts required in `backend/data_contracts.py` (create file if missing) or
the canonical contracts module used by the rest of the pipeline.

```
PlayerPropRaw         — raw API response record, strict field types, no defaults for
                        critical fields (must reject if over/under odds missing)
PlayerPropContract    — validated, de-vigged record ready for DB write
PlayerPropBatch       — list[PlayerPropContract] + fetch metadata (source, fetched_at)
```

Rules (matches Layer 0 doctrine):
- All monetary/odds fields typed as `int` (American odds are always integers).
- `over_american_odds` and `under_american_odds` must both be present; if either is
  null the record is rejected at validation, not silently coerced to None.
- `prop_type` must come from an allowlist enum (`PropType`): hits, home_runs,
  total_bases, rbis, strikeouts, walks, stolen_bases, runs_scored.
- `bdl_player_id` linkage must be resolved *before* the contract is written to DB;
  records with unresolved player IDs are logged and dropped, not stored.
- Never pass raw dicts to DB write functions — always validate through `PlayerPropContract`
  first.

##### L3E-2. Schema: `player_prop_odds` Table

New ORM model in `backend/models.py`. Natural key:
`(bdl_player_id, game_date, prop_type, prop_line, bookmaker)`.

```
id                 BigInteger PK autoincrement
bdl_player_id      Integer  NOT NULL  — FK-style reference to BDL player entity
game_date          Date     NOT NULL  — calendar date of the game (ET)
prop_type          String   NOT NULL  — enum: hits / home_runs / total_bases / rbis /
                                        strikeouts / walks / stolen_bases / runs_scored
prop_line          Float    NOT NULL  — the O/U threshold (e.g. 0.5, 1.5, 2.5)
over_american_odds Integer  NOT NULL  — e.g. -130
under_american_odds Integer NOT NULL  — e.g. +110
over_implied_raw   Float    NOT NULL  — raw implied prob before vig removal
under_implied_raw  Float    NOT NULL
vig_pct            Float    NOT NULL  — (over_implied_raw + under_implied_raw) - 1.0
over_true_prob     Float    NOT NULL  — de-vigged: over_implied_raw / total_implied
under_true_prob    Float    NOT NULL  — de-vigged: under_implied_raw / total_implied
bookmaker          String   NOT NULL  — e.g. "draftkings", "fanduel", "pinnacle"
fetched_at         DateTime NOT NULL  — timestamp of API call (ET, timezone-aware)

UniqueConstraint: (bdl_player_id, game_date, prop_type, prop_line, bookmaker)
Index: (game_date, prop_type) — for daily slate queries
Index: (bdl_player_id, game_date) — for player-centric lookups
```

Alembic migration required; do not use `Base.metadata.create_all()` in production.

##### L3E-3. De-Vigging Math (Pure Layer 1 Function)

Add a pure, zero-I/O function to `backend/core/odds_math.py` (or equivalent pure-logic
module). No numpy/pandas — standard library arithmetic only, consistent with
`scoring_engine.py` precedent.

**Algorithm:**

```
American → decimal probability:
  If odds > 0:  raw_prob = 100 / (odds + 100)
  If odds < 0:  raw_prob = abs(odds) / (abs(odds) + 100)

De-vig (Multiplicative method — most neutral):
  total_implied = over_raw_prob + under_raw_prob   # always > 1.0 due to vig
  over_true_prob  = over_raw_prob  / total_implied
  under_true_prob = under_raw_prob / total_implied
  vig_pct = total_implied - 1.0

Validation guard:
  Reject if total_implied < 1.0 (market inversion — data error).
  Reject if total_implied > 1.25 (vig > 25% — implausible; API data error).
  Reject if either raw_prob <= 0 or >= 1.
```

Example: Over -130, Under +110
```
  over_raw  = 130/230 = 0.5652
  under_raw = 100/210 = 0.4762
  total     = 1.0414  (vig = 4.14%)
  over_true  = 0.5652 / 1.0414 = 0.5427
  under_true = 0.4762 / 1.0414 = 0.4573
```

Unit test coverage required:
- Positive American odds case, negative American odds case, symmetric -110/-110 case.
- Rejection guards: inversion, extreme vig, zero/boundary odds.

##### L3E-4. Ingestion Job

New daily job in `backend/services/daily_ingestion.py`.

```
Lock ID:  100_016  (next available per CLAUDE.md advisory lock registry)
Schedule: 10:00 AM ET (after probable pitchers are confirmed for the day)
Gate:     env var MLB_PROPS_ENABLED=true (default false — must be explicitly enabled)
```

Job flow:
1. Fetch today's MLB game IDs from BDL (`get_mlb_games`).
2. For each game, call The Odds API player props endpoint for the configured markets.
3. For each prop record, resolve player name → `bdl_player_id` via `PlayerIdResolver`
   (existing service). Drop unresolvable records with a WARNING log entry.
4. Validate each record through `PlayerPropContract`. Reject invalid records; log counts.
5. De-vig via the Layer 1 pure function.
6. Upsert into `player_prop_odds` using the natural key.
7. Write a `data_ingestion_logs` row with `job_name="mlb_player_props"`, row counts, and any
   rejection summary.

Error handling: if The Odds API returns a non-200 or the response shape is unexpected,
log the failure and write a FAILED ingestion log row. Do not raise — job must not crash the
scheduler.

Call-budget telemetry: log the `X-Requests-Remaining` header from The Odds API response on
every call. Emit a WARNING if remaining calls drop below 2 000.

##### L3E-5. Synthesis: +EV Player Score

**Target end-state (may be a Layer 3F task depending on L3E execution scope).**

The goal is a daily `ev_player_scores` table (or a new column set on `player_scores`) that
merges two orthogonal signals:

| Signal | Source | Nature |
|--------|--------|--------|
| Z-score composite | `player_scores.composite_z` (14d window) | Backward-looking: recent form |
| Market-implied probability | `player_prop_odds.over_true_prob` (today's slate) | Forward-looking: market consensus |

**Synthesis approach (to be refined in implementation):**

For each hitter on the daily slate, for each relevant prop type:

```
base_score  = composite_z_14d  (normalised rolling form)
market_edge = over_true_prob - historical_hit_rate_baseline
              where baseline = rolling w_avg (hits proxy) or w_home_runs (HR proxy)
              from PlayerRollingStats (14d window)

ev_score = α × base_score + β × market_edge

Initial calibration: α=0.6, β=0.4 (market-weighted; tunable via env var)
```

The `market_edge` term is the critical quant insight: if the market implies a 54% chance
a player gets a hit today, and their 14-day rolling average implies ~48%, this positive
delta (+6 pp) is a meaningful forward-looking signal. A negative delta suggests the market
is pricing in something not captured by recent form (injury concern, tough matchup, travel).

This synthesis must remain a **read-only computation** at output time (pure function over
`player_scores` + `player_prop_odds`). Do not mutate existing `player_scores` rows.

---

#### L3E Acceptance Criteria

- [ ] `devig_american_odds()` pure function with unit tests (all guard cases pass)
- [ ] `PlayerPropContract` Pydantic model with rejection validation
- [ ] `player_prop_odds` ORM model + Alembic migration applied in production
- [ ] Ingestion job at lock 100_016, gated by `MLB_PROPS_ENABLED`
- [ ] `data_ingestion_logs` row written on each run (success and failure)
- [ ] Call-budget telemetry: remaining-requests logged on every Odds API call
- [ ] At least 1 day of production data in `player_prop_odds` before synthesis work begins

---

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

## Frontend Readiness Brief

Frontend is NOT the active workstream. When frontend execution resumes, use the documents below as the canonical briefing set and preserve backend-first sequencing.

### Frontend source-of-truth docs

1. `DESIGN.md`
	- Primary visual authority for the current design direction.
	- Use the Revolut-inspired system: Aeonik Pro display typography, Inter body, near-black/white binary, pill buttons, zero shadows.

2. `reports/2026-04-10-revolut-design-implementation-plan.md`
	- Token and component implementation plan for Tailwind/CSS.
	- Use for concrete frontend build execution once UI work is officially active.

3. `docs/superpowers/plans/2026-04-12-next-steps-assessment.md`
	- Fantasy-first frontend roadmap.
	- Important product guidance: do NOT spend cycles redesigning dead CBB surfaces before the fantasy product has a usable interface.

4. `FRONTEND_MIGRATION.md`
	- Historical frontend implementation record and guardrails.
	- Useful for patterns, auth, client-fetching rules, and type-discipline; NOT the source of current product priority.

5. `reports/2026-03-12-api-ground-truth.md`
	- Contract authority for frontend TypeScript shapes.
	- Frontend types should be derived from backend truth, never guessed from browser errors.

6. `docs/superpowers/specs/2026-04-04-fantasy-edge-decoupling-design.md`
	- Architectural guardrail for product separation.
	- Frontend work should reinforce Fantasy as the active product and avoid deepening coupling to frozen CBB concerns.

### Frontend activation gates

- Do not start frontend implementation while backend decision trust is still under validation.
- Frontend may consume validated backend outputs; it must not invent or pressure backend contracts prematurely.
- The first frontend initiative, when opened, should be fantasy-first and use existing `/api/fantasy/*` endpoints rather than redesigning archived CBB-first views.
- Any frontend type work must be grounded in backend route/schema truth or the API ground-truth report, not inferred from runtime UI errors.
- Treat `DESIGN.md` as the style guide and `reports/2026-04-10-revolut-design-implementation-plan.md` as the implementation recipe.
- If frontend work starts, scope it as a bounded execution lane with its own prompt and do not mix it into backend stabilization tasks.

---

## Active Workstream

### L3F. Decision Output Read Surface

**Status: COMPLETE (2026-04-16)**

Implemented `GET /api/fantasy/decisions` endpoint exposing trusted P17/P19 outputs.

**Completed:**
- Endpoint implementation in `backend/routers/fantasy.py` with verify_api_key auth
- Pydantic schemas (`DecisionsResponse`, `DecisionWithExplanation`, `DecisionResultOut`, `DecisionExplanationOut`, `FactorDetail`)
- 13 comprehensive test cases covering filtering, pagination, auth, response contract, and empty result handling
- Query params: `decision_type` (lineup/waiver, optional), `as_of_date` (optional, defaults to latest), `limit` (1-500, default 50)
- Returns decisions ordered by confidence desc, value_gain desc
- Empty list returned for dates with no data (not 404)
- Explanation data attached when available

**Out of scope (remains deferred):**
- New decision computation logic (Layer 4 remains HOLD)
- Frontend UI for decisions
- OddsAPI-based player props (L3E remains deferred)

### L3E. Market-Implied Probability Integration

**Status: DEFERRED — see Layer Status section for full spec (preserved as backlog)**

This work is preserved as a complete specification for future consideration, but requires an explicit policy gate before becoming active. The proposed use of The Odds API for MLB player props currently conflicts with CLAUDE.md hard-stop rules.

### L3A. Derived Stats And Scoring Spine

**Status: Complete (2026-04-16)**

Implemented `GET /api/fantasy/players/{bdl_player_id}/scores` - the first authoritative Layer 3 scoring exposure.

**Completed:**
- Endpoint implementation in `backend/routers/fantasy.py` with verify_api_key auth
- Pydantic schemas (`PlayerScoresResponse`, `PlayerScoreOut`, `PlayerScoreCategoryBreakdown`)
- 13 comprehensive test cases covering validation, response contract, and auth requirements
- Supports window_days=7/14/30 (defaults to 14)
- Supports optional as_of_date query parameter (defaults to latest available)
- Returns 400 for invalid window_days, 404 for missing scores, 401 for missing auth
- Exposes hitter categories (z_hr, z_rbi, z_nsb, z_avg, z_obp) and pitcher categories (z_era, z_whip, z_k_per_9)

### L3B. Context Authority Consolidation

**Status: Complete (2026-04-16)**

Scoped park factor consolidation completed:
- `ballpark_factors.py:get_park_factor()` now resolves: DB → PARK_FACTORS constant → 1.0 neutral
- 9 focused tests added (test_ballpark_factors.py)
- Weather remains explicitly deferred for Layer 3 scoring (request-time weather via weather_fetcher.py serves immediate-decision needs)

### L3D. Layer 3 Observability

**Status: Complete (2026-04-16)**

Layer 3 freshness endpoint `/admin/diagnose-scoring/layer3-freshness` is live and fully tested:
- Returns freshness verdict (healthy/stale/partial/missing)
- Provides row counts by window (7/14/30)
- Shows latest audit log entries for rolling_windows and player_scores jobs
- 13 comprehensive tests in test_admin_scoring_diagnostics.py

**Decision Pipeline Observability (2026-04-16):**

Decision pipeline freshness endpoint `/admin/diagnose-decision/pipeline-freshness` is live and fully tested:
- Provides observability for P17-P19 stages (DecisionResult and DecisionExplanation tables)
- Returns freshness verdict (healthy/stale/partial/missing)
- Shows breakdown_by_type (lineup/waiver), row counts, and latest computed_at timestamps
- Includes schedule expectations (~7 AM for decision_results, ~9 AM for decision_explanations)
- 8 comprehensive tests in test_admin_scoring_diagnostics.py
- Total test count for admin_scoring_diagnostics.py: 34 tests passing

**Out of scope for this phase:**

- simulation expansion
- waiver-system breadth work
- optimizer redesign
- frontend feature expansion
- broad Layer 4 activation
- weather_forecasts table population (request-time weather remains sufficient)

---

## Immediate Priority Queue

| Priority | Action | Owner | Status |
|----------|--------|-------|--------|
| P0 | Define the first Layer 3 scoring objective and success criteria | Claude | Complete |
| P0 | Audit current derived-stats and scoring code path for gaps | Claude | Complete |
| P1 | Identify one authoritative scoring output for downstream consumers | Claude | Complete |
| P1 | Audit context authority in scoring path (3B) | Claude | Complete |
| P2 | Consolidate ballpark_factors.py to DB-backed read | Claude | Complete |
| P2 | Add Layer 3 freshness observability endpoint | Claude | Complete |
| P2 | Add decision pipeline freshness observability endpoint | Claude | Complete |
| P1 | Expose DecisionResult read API (lineup/waiver outputs) with verify_api_key auth (L3F) | Claude | Complete |
| P2 | Decide whether any Layer 5 response shape changes are needed after scoring output stabilizes | Claude | Complete (2026-04-16: consumer audit found no changes needed) |

---

## Architect Review Queue

- Keep `/admin/version`, ingestion logs, and `probable_pitchers` on passive regression watch.
- Treat canonical environment snapshots beyond current weather and park persistence as backlog, not active recovery work.
- Do not reopen simulation or decision-layer expansion until Layer 3 outputs are demonstrably trustworthy.
- Do not start frontend implementation until the backend decision pipeline is trusted and the frontend lane is opened explicitly.
- When frontend work opens, use `DESIGN.md` plus the April 10 and April 12 planning docs as the briefing pack; treat `FRONTEND_MIGRATION.md` as historical implementation context only.
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

Last Updated: April 16, 2026 (21:00 UTC - L3F Decision Output Read Surface complete; GET /api/fantasy/decisions live with 13 tests passing; L3E remains deferred pending policy gate)