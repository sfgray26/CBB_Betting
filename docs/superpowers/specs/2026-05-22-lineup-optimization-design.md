# Lineup Optimization Design

**Goal:** Detect sub-optimal pitcher slot assignments in `_get_lineup_gaps()` and emit actionable swap suggestions using `score_0_100` percentile rank.

**Architecture:** Two-phase `_get_lineup_gaps()`. Phase 1 = existing empty-slot detection (unchanged). Phase 2 = score-based sub-optimal placement detection added as a second loop within the same method. No changes to call signature, `asyncio.gather()`, or `DashboardData`.

**Tech Stack:** Python 3.11, SQLAlchemy, PlayerScore / PlayerIDMapping models, Yahoo roster API (already fetched in Phase 1).

---

## Components

### Constant
`SUBOPTIMAL_SCORE_THRESHOLD = 10.0` — minimum score_0_100 gap to emit a swap suggestion. Defined at module level in `dashboard_service.py`.

### New severity value
`"optimization"` added to the `LineupGap` docstring comment (no schema change — severity is a free string).

### Phase 2 algorithm (inside `_get_lineup_gaps()`)

1. **Separate pitchers from roster** by `selected_position`:
   - Starting pitchers: `selected_position in {"SP", "RP", "P"}`
   - Bench pitchers: `selected_position == "BN"` AND any of SP/RP in `positions`

2. **Fetch scores** — one batch DB query via `SessionLocal()`:
   ```
   normalize each player name
   → PlayerIDMapping.normalized_name → bdl_player_id
   → PlayerScore (window_days=14, latest as_of_date) → score_0_100
   ```
   Players with no score row are assigned `score_0_100 = 0.0` (safe fallback).

3. **Swap detection loop** — for each starting pitcher:
   - Find all bench/misslotted pitchers eligible for the starting pitcher's slot
     (eligibility = `"SP" in player["positions"]` for SP slots, etc.)
   - If `bench_score - starting_score > SUBOPTIMAL_SCORE_THRESHOLD`:
     emit `LineupGap(position=slot, severity="optimization", message=..., suggested_add=bench_player_name)`

4. **Message format:**
   ```
   ⚠️ SUB-OPTIMAL: {bench_name} (Score {bench_score:.0f}/100, {bench_pos_str}) is on BN.
   Consider moving to {slot}. {starter_name} (Score {starter_score:.0f}/100) would move to BN.
   ```

### DB access pattern
`SessionLocal()` created at start of Phase 2, closed in `finally`. Same pattern as `_get_streaks()`. Import `PlayerScore` added to existing import line for `backend.models`.

---

## Error handling
- If DB query fails: log WARNING, skip Phase 2 entirely (empty list, no crash)
- If a player has no score row: use `0.0` (treated as unknown/worst)
- Phase 2 never raises — all exceptions caught and logged

---

## Testing
- `test_lineup_gap_detects_suboptimal_sp_placement()` — mock roster with Harrison in RP (score 90) and Pérez in SP (score 20); assert "optimization" gap emitted
- `test_lineup_gap_no_false_positive_when_scores_within_threshold()` — scores within 10pts; assert no optimization gap
- `test_lineup_gap_phase2_skips_gracefully_on_db_error()` — DB raises; assert function returns normally (no crash)

---

## Out of scope
- Hitter slot optimization (pitcher-only for this iteration; YAGNI)
- Automatic roster moves (detection only)
- Frontend changes (Kimi owns)
