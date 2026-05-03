# Optimizer Performance Investigation — May 3, 2026

## Baseline Performance
- Response time (Dashboard): 19.34s
- Response time (Waiver): 23.78s
- Query count: Estimated > 100 per request (based on `park_factors` loops)
- Top bottleneck: `db.query(ParkFactor)` in `ballpark_factors.py` (N+1)

## Caching Implementation
- Changes: `lru_cache` added to `_get_player_board`, TTL managed via 5-min cache buster.
- Files modified: `backend/fantasy_baseball/mcmc_calibration.py`
- Lines changed: ~15 lines added/modified.

## Results After Caching
- Response time (Dashboard): 9.95s (50% reduction)
- Response time (Waiver): 26.82s (No significant change, as expected)
- Target <5s achieved: NO (but significant progress made for dashboard)

## Critical Findings
- **Missing Indexes:** All key indexes (player_id, as_of_date, etc.) are present in production.
- **N+1 Query Problem:** YES. Confirmed in `backend/fantasy_baseball/ballpark_factors.py`. Every `get_park_factor` call triggers a database query.
- **Caching Insufficient:** While caching the player board helped the dashboard, the N+1 queries in other services remain the primary bottleneck.

## Recommendations
- **Fix N+1 in ballpark_factors.py:** Implement bulk loading or process-level caching for the `park_factors` table.
- **Investigate Dashboard Nulls:** Fix why `matchup_preview` returns null, preventing win probability verification.
- **Algorithm Rewrite:** If caching doesn't get us under 5s, consider pre-calculating and caching the entire board projections in Redis.
