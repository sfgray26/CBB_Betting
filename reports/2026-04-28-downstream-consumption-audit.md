# Downstream Consumption Audit — Session H Fields

> **Date:** 2026-04-28 | **Auditor:** Kimi CLI (K-34)  
> **Scope:** `scarcity_rank` → `position_eligibility` and `quality_score` → `probable_pitchers`  
> **Mandate:** Research only — no code changes.

---

## Executive Summary

| Field | Table | Downstream Consumer | Reads It? | Impact if Null |
|-------|-------|---------------------|-----------|----------------|
| `scarcity_rank` | `position_eligibility` | `waiver_edge_detector.py` | ❌ No | Scarcity ignored; hardcoded position groups used |
| `scarcity_rank` | `position_eligibility` | `daily_lineup_optimizer.py` | ❌ No | Scarcity-first slot order is hardcoded in `_DEFAULT_BATTER_SLOTS` |
| `scarcity_rank` | `position_eligibility` | `lineup_constraint_solver.py` | ❌ No | Static `SLOT_CONFIG` has its own `scarcity_rank` column (unrelated to DB) |
| `quality_score` | `probable_pitchers` | `two_start_detector.py` | ✅ Yes | Already SELECTs and surfaces it; null → all ratings neutral (0.0) |
| `quality_score` | `probable_pitchers` | Waiver recommendations endpoint | ❌ No | Endpoint never queries `probable_pitchers` |

**Key finding:** `two_start_detector.py` is the *only* downstream consumer that already reads `quality_score`. The other major consumers (`waiver_edge_detector.py`, `daily_lineup_optimizer.py`, `lineup_constraint_solver.py`) do not query `position_eligibility` at all and therefore cannot benefit from `scarcity_rank` without additional wiring.

**Schema mismatch discovered:** `backend/schemas.py` `MatchupRatingSchema.quality_score` is documented as `-2.0 to +2.0` (line 722), but the Session H heuristic will produce `0.0–1.0`. This is a breaking contract change for any frontend that expects the old range.

---

## Q1: `waiver_edge_detector.py` and `scarcity_rank`

### CURRENT STATE
`waiver_edge_detector.py` (404 lines) **never queries the `position_eligibility` table**. Position handling is entirely based on hardcoded Python structures:

- `_POS_GROUP` (lines 185–199): maps FA positions to roster position groups for drop pairing (e.g., `"C": ["C"]`, `"OF": ["OF", "LF", "CF", "RF"]`).
- `_count_position_coverage` (lines 322–334): counts roster players covering a position group using `p.get("positions")` from Yahoo roster dicts.
- `_weakest_droppable_at` (lines 336–368): finds the weakest droppable player at a given position group.
- `drop_candidate_value` (lines 115–144): uses `z_score`, `tier`, `adp`, `owned_pct` — no scarcity weighting.

There is **no reference** to `scarcity_rank`, `league_rostered_pct`, `PositionEligibility`, or any SQL query against the `position_eligibility` table.

### GAP
The waiver edge detector evaluates free agents purely on category need + z-score + ownership. It does not weight scarce positions (C, SS, 2B) more heavily than abundant ones (OF, 1B). A replacement-level catcher and a replacement-level outfielder receive the same `need_score` even though the catcher is harder to replace.

### MINIMUM FIX
Add a scarcity multiplier in `_score_fa_against_deficits` (or in `get_top_moves` before sorting) that looks up `scarcity_rank` from `position_eligibility` via the player's `bdl_player_id` or name, and boosts the `need_score` for players with `scarcity_rank <= 3` (C, SS, 2B).

---

## Q2: `daily_lineup_optimizer.py` and `lineup_constraint_solver.py`

### CURRENT STATE — `daily_lineup_optimizer.py`
- Never queries `position_eligibility`.
- Scarcity-first slot filling is hardcoded in `_DEFAULT_BATTER_SLOTS` (lines 172–182): `C → 1B → 2B → 3B → SS → OF×3 → Util`.
- `solve_lineup()` (lines 603–729) iterates slots in that fixed order and greedily assigns the highest-`lineup_score` eligible batter.
- Position eligibility comes from `player.get("positions")` in the Yahoo roster dict, not from the database.

### CURRENT STATE — `lineup_constraint_solver.py`
- Never queries `position_eligibility`.
- `SLOT_CONFIG` (lines 83–93) defines its own internal `scarcity_rank` (1=C, 2=SS, … 9=Util) used only for iteration order.
- `eligibility` is passed in as a `Dict[str, List[str]]` parameter from the caller.
- `analyze_scarcity()` (lines 313–356) computes scarcity from the provided `eligibility` dict by counting eligible players per position (`len(eligible) <= 1` → scarce), not from the database.

### GAP
Both optimizers use static/hardcoded scarcity ordering. They cannot dynamically adjust scarcity weighting based on league-wide roster percentages or true positional depth because they never read `position_eligibility`.

### MINIMUM FIX
- **For `daily_lineup_optimizer.py`:** Before `solve_lineup()`, query `position_eligibility` for all roster players, join `scarcity_rank`, and use it to re-order `_DEFAULT_BATTER_SLOTS` dynamically (e.g., if a league has only 1 SS rostered total, boost SS priority above C).
- **For `lineup_constraint_solver.py`:** Inject `scarcity_rank` into the `EliteScore` or `PlayerSlotAssignment` dataclass and add a scarcity bonus term to the CP-SAT objective function (e.g., `+ (10 - scarcity_rank) * 50` scaled points for filling scarce slots).

---

## Q3: `two_start_detector.py` and `quality_score`

### CURRENT STATE
`two_start_detector.py` **already reads `quality_score`** from the database:

- SQL SELECT (lines 132–147) explicitly includes `quality_score`.
- Row unpacking (line 170): `"quality_score": row[8] or 0.0`.
- `MatchupRating` dataclass (lines 23–31) includes `quality_score: float`.
- `TwoStartOpportunity` dataclass (lines 34–64) includes `average_quality_score` and `streamer_rating`.
- `_build_opportunity()` (lines 185–254) copies `quality_score` into `game_1` and `game_2`, computes the mean, and derives `streamer_rating` from it:
  - `avg_quality >= 1.0` → `"EXCELLENT"`
  - `avg_quality >= 0.0` → `"GOOD"`
  - else → `"AVOID"`

The API response schema `TwoStartOpportunitySchema` (lines 727–776 in `backend/schemas.py`) includes `average_quality_score` and nested `MatchupRatingSchema.quality_score`.

### GAP
The detector is fully wired to consume `quality_score`, but because the column is 100% null in production, every pitcher currently gets `quality_score=0.0` (the `or 0.0` fallback), which means:
- `avg_quality = 0.0` → `"GOOD"` for every pitcher
- No differentiation between aces and streamers
- The `EXCELLENT` and `AVOID` buckets are effectively unreachable

### MINIMUM FIX
No downstream code change is required — the consumption path is complete. Session H only needs to implement the heuristic in `_sync_probable_pitchers` and the Two-Start Command Center will automatically surface real ratings.

**Note:** There is a schema contract mismatch. `MatchupRatingSchema` documents `quality_score` as `-2.0 to +2.0` (line 722), but the Session H heuristic produces `0.0–1.0`. The `streamer_rating` thresholds (`>= 1.0` → EXCELLENT, `>= 0.0` → GOOD) happen to work with the 0–1 range, but any frontend that expects negative values for "terrible" matchups will see no AVOID ratings. Either update the schema docstring or adjust the heuristic to emit -2.0 to +2.0.

---

## Q4: Waiver Recommendations Endpoint and `quality_score`

### CURRENT STATE
The endpoint `GET /api/fantasy/waiver/recommendations` (`backend/routers/fantasy.py:2077`) does **not** surface `quality_score` anywhere in its response.

Trace:
1. Endpoint fetches free agents via `YahooFantasyClient.get_free_agents()` (line 2198).
2. Scores each FA with `_score_fa()` (line 2200) using projections + category need vector.
3. Finds drop candidates via `_weakest_safe_to_drop()` (line 2373) using roster projections.
4. Builds `RosterMoveRecommendation` objects (lines 2377–2392, 2419–2441) with fields: `action`, `add_player` (`WaiverPlayerOut`), `drop_player_name`, `rationale`, `need_score`, `confidence`, `statcast_signals`, `regression_delta`, and MCMC win-probability fields.

`WaiverPlayerOut` (lines 399–426 in `schemas.py`) does **not** contain a `quality_score` field.
`RosterMoveRecommendation` (lines 450–467 in `schemas.py`) also does **not** contain a `quality_score` field.

The endpoint never queries `probable_pitchers`, so even if a FA is a starting pitcher, there is no matchup quality data in the response.

### GAP
Users cannot see matchup quality when evaluating waiver-add pitchers. A streamer SP recommendation lacks context about whether the pitcher faces a weak offense in a pitcher-friendly park or a strong offense in Coors Field.

### MINIMUM FIX
For pitcher recommendations, add a `probable_pitchers` subquery in `_score_fa()` (or in the endpoint loop) that looks up `quality_score` by player name + team for today's game date, and inject it into `WaiverPlayerOut` as a new `matchup_quality` field (requires schema update).

---

## Q5: Database Schema Verification

```sql
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name IN ('position_eligibility', 'probable_pitchers')
  AND column_name IN ('scarcity_rank', 'league_rostered_pct', 'quality_score')
ORDER BY table_name, column_name;
```

### Results

| table_name | column_name | data_type | is_nullable |
|------------|-------------|-----------|-------------|
| position_eligibility | league_rostered_pct | double precision | YES |
| position_eligibility | scarcity_rank | integer | YES |
| probable_pitchers | quality_score | double precision | YES |

All three columns exist and are **nullable**. None have `NOT NULL` constraints. `scarcity_rank` and `league_rostered_pct` live on `position_eligibility`; `quality_score` lives on `probable_pitchers`.

---

## Recommendations for Session H (Claude Code)

| Priority | Action | File(s) |
|----------|--------|---------|
| P0 | Implement `scarcity_rank` in `_sync_position_eligibility` (already planned) | `backend/services/daily_ingestion.py` |
| P0 | Implement `quality_score` in `_sync_probable_pitchers` (already planned) | `backend/services/daily_ingestion.py` |
| P1 | **Fix schema docstring mismatch** — `MatchupRatingSchema.quality_score` claims `-2.0 to +2.0` but heuristic emits `0.0–1.0` | `backend/schemas.py:722` |
| P2 | Add `scarcity_rank` lookup in `waiver_edge_detector.py` to weight FA need scores by position scarcity | `backend/services/waiver_edge_detector.py` |
| P2 | Add `scarcity_rank` dynamic re-ordering in `daily_lineup_optimizer.py` | `backend/fantasy_baseball/daily_lineup_optimizer.py` |
| P2 | Inject `scarcity_rank` into CP-SAT objective in `lineup_constraint_solver.py` | `backend/fantasy_baseball/lineup_constraint_solver.py` |
| P3 | Surface `matchup_quality` in waiver recommendations endpoint for pitcher FAs | `backend/routers/fantasy.py`, `backend/schemas.py` |

---

## Files Audited

- `backend/services/waiver_edge_detector.py` (404 lines, full read)
- `backend/fantasy_baseball/daily_lineup_optimizer.py` (1000 lines, full read)
- `backend/fantasy_baseball/lineup_constraint_solver.py` (422 lines, full read)
- `backend/fantasy_baseball/two_start_detector.py` (309 lines, full read)
- `backend/routers/fantasy.py` (lines 2077–2473, waiver recommendations endpoint)
- `backend/services/dashboard_service.py` (lines 446–519, waiver target aggregation)
- `backend/schemas.py` (relevant Pydantic models)
- PostgreSQL `information_schema.columns` (live DB query)
