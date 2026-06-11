# Wave 5 P2 Feature: Trade Analyzer Backend

**Date:** 2026-05-20  
**Agent:** Copilot CLI (Wave 5, P2)  
**Status:** ✅ Complete — py_compile clean, 8/8 tests pass

---

## Problem

Fantasy managers evaluate trades constantly but have no tool to project category impact.
No endpoint existed for trade analysis; users had to mentally compare two player lists
across 10+ scoring categories.

---

## What Was Built

### 1. `backend/contracts.py` — Four new Pydantic schemas

| Schema | Purpose |
|--------|---------|
| `TradePlayerInput` | Player in a trade request (`player_key` + optional `player_name`) |
| `TradeAnalyzeRequest` | Request body: `give`, `receive` lists, optional `league_id` |
| `TradeCategoryDelta` | Per-category impact: `give_z`, `receive_z`, `delta`, `direction` |
| `TradeAnalysis` | Full result: player summaries, deltas, `total_z_delta`, `recommendation`, `summary` |

Also added `Any` to the `typing` import (was missing from the module).

### 2. `backend/fantasy_baseball/trade_analyzer.py` — Core logic module

- **`_sum_cat_scores(players)`** — sums z-scores per category across a list of player dicts
- **`_player_summary(player)`** — compact serialisable summary (name, key, z_score, type, team, positions)
- **`analyze_trade(give_players, receive_players, league_settings=None)`**:
  1. Sums `cat_scores` from each player dict (format produced by `get_or_create_projection`)
  2. Computes `delta = receive_z - give_z` per category
  3. Sums all deltas → `total_z_delta`
  4. Maps delta to five-level recommendation:

| `total_z_delta` | Recommendation |
|-----------------|---------------|
| ≥ +1.5 | `strong_accept` |
| ≥ +0.5 | `accept` |
| (−0.5, +0.5) | `neutral` |
| ≤ −0.5 | `reject` |
| ≤ −1.5 | `strong_reject` |

### 3. `backend/routers/trade.py` — Isolated REST router

- `POST /api/fantasy/trade/analyze` (authenticated via `verify_api_key`)
- `_resolve_player()` helper: calls `get_or_create_projection` with minimal Yahoo dict; falls back to an empty-cat-scores stub so one bad player key can't crash the whole request
- Validates non-empty `give` / `receive` lists (422 if either is empty)
- 500 error only on truly unexpected internal failures

### 4. `backend/main.py` — Router mount

```python
from backend.routers.trade import router as _trade_router
app.include_router(_trade_router)
```

Mounted at the same level as `_edge_router`, `_fantasy_router`, `_admin_router` — fully isolated from `fantasy.py` as required.

---

## Design Notes

### Data flow

```
POST /api/fantasy/trade/analyze
  → validate TradeAnalyzeRequest
  → _resolve_player() × N  (calls player_board.get_or_create_projection)
  → analyze_trade(give_projections, receive_projections)
  → return TradeAnalysis
```

### Why `cat_scores` from `player_board`?

`scoring_engine.py` and `category_math.py` referenced in the task spec do not exist.
The equivalent production-ready module is `backend/fantasy_baseball/player_board.py`
(`get_or_create_projection`) which returns board-compatible dicts with a `cat_scores`
field (`{category → z_score}`) sourced from `PlayerProjection.cat_scores` (DB) or
the draft board (static fallback).  Category weights live in
`backend/services/cat_scores_builder.py` (`BATTER_WEIGHTS`, `PITCHER_WEIGHTS`).

The trade analyzer's `total_z_delta` is an unweighted sum of category deltas.
A future enhancement could apply `BATTER_WEIGHTS` / `PITCHER_WEIGHTS` to make
it more sensitive to high-value categories (HR, SB, ERA, WHIP).

---

## Files Created / Modified

| File | Action |
|------|--------|
| `backend/contracts.py` | Modified — added `Any` import + 4 new schemas |
| `backend/fantasy_baseball/trade_analyzer.py` | Created |
| `backend/routers/trade.py` | Created |
| `backend/main.py` | Modified — import + mount of `_trade_router` |
| `tests/test_trade_analyzer.py` | Created — 8 unit tests |

---

## Verification

| Check | Result |
|-------|--------|
| `py_compile backend/contracts.py` | ✅ OK |
| `py_compile backend/fantasy_baseball/trade_analyzer.py` | ✅ OK |
| `py_compile backend/routers/trade.py` | ✅ OK |
| `py_compile backend/main.py` | ✅ OK |
| `pytest tests/test_trade_analyzer.py -v` | ✅ 8/8 passed |

---

## Hand-off Notes for Claude Code

1. **Category weighting** — `total_z_delta` is a plain sum. Apply `BATTER_WEIGHTS` /
   `PITCHER_WEIGHTS` from `cat_scores_builder.py` for a more accurate recommendation.

2. **Yahoo player enrichment** — The router uses `get_or_create_projection` which hits
   the DB. Adding a Yahoo API call (`client.get_player(player_key)`) before the projection
   lookup would enrich the name → better DB resolution for recently-rostered players.

3. **`league_settings` parameter** is reserved but unused. Future use: per-league category
   weights or disabled categories.

4. **Frontend integration** — endpoint is `POST /api/fantasy/trade/analyze`.
   Request shape: `{"give": [{"player_key": "...", "player_name": "..."}], "receive": [...]}`.
   Response shape: see `TradeAnalysis` in `backend/contracts.py`.
