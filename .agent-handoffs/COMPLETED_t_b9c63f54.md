# Completed: t_b9c63f54 — BDL #2 Player Search Auto-Heal

## Status
COMPLETE — 17/17 tests pass, all acceptance criteria met.

## What Was Done

### New file: `backend/services/player_autoheal.py`
- `PlayerAutoHealService` with `heal_player()` and `batch_heal()` methods
- BDL name-search fallback via `search_mlb_players()`
- `source='bdl_search'` tag on all auto-healed rows — never overwrites `source='manual'`
- 7-day TTL: fresh rows skipped, stale rows refreshed, failed heals retried every run
- Unicode-normalized name matching with 0.85 confidence threshold
- bdl_id conflict guard on insert (skips if BDL ID already mapped elsewhere)

### Modified: `backend/services/daily_ingestion.py`
- Added `yahoo_key` to all four `unmatched.append()` call sites (was missing — auto-heal needs it for DB lookup)
- Added auto-heal phase after main name-match loop: `PlayerAutoHealService.batch_heal()` on healable unmatched players
- Returns `auto_healed` count in job result dict
- Skips `bdl_name_collision` and `bdl_id_conflict` reasons in auto-heal (manual resolution required)

### New file: `tests/test_player_autoheal.py`
17 tests covering:
- Helper functions: `_normalize`, `_name_confidence`, `_is_fresh`
- New call-up insert path
- `source='bdl_search'` and `resolution_confidence` set correctly
- Manual override protection
- Fresh TTL skip
- Stale TTL refresh
- No BDL results → False
- BDL exception → graceful False, no crash
- `batch_heal` counts (healed/skipped/failed)
- Missing `yahoo_key` → skip
- bdl_id conflict on insert → skip

## Acceptance Criteria
- [x] New call-ups show ownership % within 24 hours (auto-heal runs in `yahoo_id_sync` nightly)
- [x] Auto-heal events logged (`auto_heal: healed yahoo_key=... -> bdl_id=...`)
- [x] No regression in existing player mapping (manual rows protected)
- [x] All tests pass (17/17)
- [x] Handoff summary written
