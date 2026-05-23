# Lineup Optimization + Pitcher Diagnostic — 2026-05-22

## Task A: Diagnostic Logging (daily_lineup_optimizer.py)

Added 4 DIAG-level INFO log lines to `flag_pitcher_starts()`:

1. **Entry** — `roster_size=N, sp_rp_p_eligible=N` (addresses: empty roster / no positions parsed)
2. **After probable_pitchers fetch** — `probable_pitchers_teams=N` or `FAILED: <error>` (addresses: silent crash)
3. **Per-pitcher pass-through** — one INFO line per pitcher that clears the `_INACTIVE_STATUSES` filter
4. **Exit** — `returning N pitchers`

Also: wrapped `_fetch_probable_pitchers_for_date()` in `try/except` — on crash, `probable_pitchers`
defaults to `{}` and all SPs fall through the existing Fallback 1 path (`has_start=True` when no data
and team has a game scheduled). This defensive change ships regardless of root cause.

**Next step:** Deploy commit `1d9a2d8`. Check Railway logs for `[flag_pitcher_starts] DIAG` lines
to identify which failure mode is active, then code the targeted fix.

## Task B: Sub-optimal Pitcher Placement Detection (dashboard_service.py)

### What was added

- `SUBOPTIMAL_SCORE_THRESHOLD = 10.0` — named constant at module level
- `PlayerScore` added to model imports
- `LineupGap.severity` docstring updated to include `"optimization"`
- `_detect_pitcher_swap_gaps(roster, db)` — new method on `DashboardService`
- Phase 2 call wired into `_get_lineup_gaps()` before the return, wrapped in its own `try/except`

### How Phase 2 works

1. Splits roster pitchers into starting (`selected_position ∈ {"SP","RP","P"}`) and bench
2. Resolves names → `bdl_id` via `PlayerIDMapping.normalized_name`
3. Fetches 14-day `PlayerScore.score_0_100` for all resolved pitchers (latest per player, 3-day lag tolerance)
4. For each starting pitcher, checks all bench pitchers eligible for that slot
5. If `bench_score − starter_score > 10.0`: emits `LineupGap(severity="optimization", ...)`
6. Message format: `⚠️ SUB-OPTIMAL: {bench} (Score X/100, SP/RP) is on BN. Consider moving to SP. {starter} (Score Y/100) would move to BN.`

### Isolation guarantee

Phase 2 DB errors **never** affect Phase 1 results. Two layers of protection:
- `_detect_pitcher_swap_gaps()` has an inner `try/except` returning `[]` on any failure
- The Phase 2 call in `_get_lineup_gaps()` is wrapped in a separate outer `try/except`

## Verification

- `py_compile` both files → clean
- `pytest tests/test_dashboard_service_lineup_optimization.py` → 4 passed
- `pytest tests/test_dashboard_service*.py tests/test_lineup_optimizer.py` → 29 passed, 1 skipped
