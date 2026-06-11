# Wave2: Within-League Transaction Feed
**Date:** 2026-05-22  
**Branch:** `agent/copilot/wave2-transaction-feed`  
**Status:** ✅ Complete

---

## Summary
Built a within-league drop intelligence feed that pulls recent drop transactions from the Yahoo Fantasy API and annotates waiver wire candidates with "was dropped by X, N days ago" context.

---

## Files Changed

| File | Change |
|------|--------|
| `backend/services/league_transaction_feed.py` | **New** — `LeagueDrop` dataclass, `get_recent_league_drops()`, `build_drop_lookup()` |
| `backend/schemas.py` | Added `league_drop: Optional[dict] = None` to `WaiverPlayerOut` |
| `backend/routers/fantasy.py` | Integration block after IL separation (~line 2278) |
| `tests/test_league_transaction_feed.py` | **New** — 26 unit tests (100% pass) |

---

## Implementation Details

### `backend/services/league_transaction_feed.py`
- **`LeagueDrop` dataclass**: `player_key`, `player_name`, `team`, `positions`, `dropped_by_team`, `dropped_by_name`, `dropped_at`, `days_ago`
- **`get_recent_league_drops(client, days=7)`**: Calls `client.get_transactions(t_type="drop")`, filters by timestamp within look-back window, returns `List[LeagueDrop]` sorted newest-first. Returns `[]` on API failure (non-fatal).
- **`build_drop_lookup(drops)`**: Builds `{by_key: {player_key → LeagueDrop}, by_name: {lower_name → LeagueDrop}}` for O(1) waiver annotation.
- Defensively handles Yahoo API's inconsistent response shapes (lists-of-dicts vs plain dicts, wrapped vs unwrapped `transaction_data`).
- Accepts both `"drop"` and `"add/drop"` transaction types; filters to drop legs only.

### Schema change (`WaiverPlayerOut`)
```python
league_drop: Optional[dict] = None
# Shape when populated: {"dropped_by": "Marte Partay", "days_ago": 1.0, "team_key": "469.l.72586.t.3"}
```

### Router integration (fantasy.py)
- Runs after IL separation (so both `top_available` and `il_watch` are annotated)
- Wrapped in `try/except` — a transaction API failure never blocks the waiver response
- Lookup: `by_key` (player_key exact match) → `by_name` (lowercase name fallback)

---

## Test Results
```
26 new tests: 26 passed, 0 failed
Full suite:   3033 passed, 3 skipped, 4 pre-existing failures (test_row_projector date arithmetic — not in diff)
```

---

## Verification
```
py_compile: PASS (league_transaction_feed.py, schemas.py, fantasy.py)
pytest tests/test_league_transaction_feed.py: 26/26 PASS
pytest tests/: 3033 passed, 4 pre-existing failures only
```
