# Wave 4 — Weekly Matchup Preview Backend

**Date:** 2026-05-20
**Severity:** P2 FEATURE — next-week H2H matchup projection + streaming recommendations

---

## Problem

War Room only showed the current matchup. Users had no visibility into next week's opponent
or which categories they should target on waivers before the week starts.

---

## Solution

### Backend: `GET /api/fantasy/matchup-preview`

**File:** `backend/routers/fantasy.py` (new endpoint after `bulk_apply_roster_moves`)
**Schema:** `backend/contracts.py` (`MatchupPreviewCategoryProjection`, `MatchupPreviewStreamingRec`, `MatchupPreviewResponse`)

Key design:
- **Week resolution**: Calls `client.get_league()` for `current_week`, tries `get_matchup_stats(week=current+1)` for next week's opponent. Falls back to current-week opponent gracefully when Yahoo hasn't published next week's matchup.
- **MCMC simulation**: Calls existing `simulate_weekly_matchup(my_roster, opponent_roster=[], n_sims=2000, remaining_fraction=1.0)`. Empty opponent list triggers the league-average (z=0) baseline in the simulator.
- **Roster enrichment**: Reuses `_fetch_rosters_for_simulate(db)` which already queries `player_projections` and fuzzy-name-matches to build `cat_scores` dicts per player.
- **Category classification**: `win_prob > 0.6` → "win", `< 0.4` → "loss", else "bubble".
- **Streaming recommendations**: Automatically surfaces human-readable strings for every "loss" category.

```
GET /api/fantasy/matchup-preview
→ 200: {
    "opponent_team_name": "Team Rocket",
    "week_num": 9,
    "win_prob": 0.62,
    "expected_cats_won": 10.4,
    "category_projections": [
      {"category": "hr_b", "my_win_prob": 0.72, "status": "win"},
      {"category": "era",  "my_win_prob": 0.34, "status": "loss"},
      {"category": "rbi",  "my_win_prob": 0.53, "status": "bubble"}
    ],
    "streaming_recommendations": [
      {"category": "era", "reason": "Projected to lose ERA (34% win rate) — target streamers with ERA upside"}
    ],
    "data_quality": "ok",
    "freshness": {...}
  }
```

---

## Files Changed

| File | Change |
|------|--------|
| `backend/contracts.py` | Added `MatchupPreviewCategoryProjection`, `MatchupPreviewStreamingRec`, `MatchupPreviewResponse` |
| `backend/routers/fantasy.py` | New `get_matchup_preview` endpoint; added 3 schema imports |
| `tests/test_matchup_preview.py` | 4 new tests covering structure, classification, recs, and fallback |

**Note on ScheduleFetcher:** `ScheduleFetcher.get_todays_schedule(date: Optional[datetime] = None)` already accepts a future date parameter — no extension needed. The existing method can be called with any date for multi-day schedule lookups when `starts_this_week` enrichment is added in a future pass.

**Test result:** 4/4 passing.
**py_compile:** zero errors on contracts.py and routers/fantasy.py.
