# Wave 3A — Proxy Projection Fallback Fix

**Date:** 2026-05-20
**Severity:** P3 MEDIUM — 8+ diverse players all returned identical 58.0 proxy_projection score

---

## Root Cause

`_projection_fallback_score` in `fantasy.py` (line 355) computes:
```
score = 50.0 + (z_score * 8.0) + ownership * 0.05
if is_proxy: score = min(score, 58.0)
```

Any proxy player with `z_score >= 1.0` hits the 58.0 ceiling. The fusion path in
`player_board.py` was hardcoding `z_score = 0.0` for ALL players with no DB projection data,
and `scoring_engine.py` silently returned `composite_z = 0.0` when a player had no valid
rolling-stat Z-scores — with no log output to indicate data quality failures.

---

## Fixes

### Fix 1 — Tiered Position-Based Proxy z_score (`player_board.py`)

**Before:** `z_score = 0.0` (flat neutral for every unknown player)

**After:**
```python
_PROXY_Z_BY_POSITION = {
    "C":   0.2,    # scarcest — slight positive
    "SS":  0.1,
    "2B":  0.0,
    "3B":  0.0,
    "OF": -0.1,  "LF": -0.1,  "CF": -0.1,  "RF": -0.1,
    "1B": -0.2,
    "DH": -0.3,
    "SP": -0.1,
    "RP": -0.4,   # lowest marginal fantasy value
}
_primary_pos = positions[0] if positions else None
z_score = _PROXY_Z_BY_POSITION.get(_primary_pos or "", 0.0)
logger.warning("[player_board] proxy fallback: %s pos=%s z_score=%.2f", name, _primary_pos, z_score)
```

**Effect:** Proxy scores now range ~46–52 by position instead of clustering at the 58.0 cap.
A catcher proxy scores ~51.6 while a reliever proxy scores ~46.8 — meaningful differentiation.

### Fix 2 — Explicit Warning in Scoring Engine (`scoring_engine.py`)

Added `import logging` and `logger = logging.getLogger(__name__)`.

When `kv_pairs` is empty (no category reached MIN_SAMPLE):
```python
if not kv_pairs:
    logger.warning(
        "player_id=%d (%s) has no valid Z-score categories in %d-day window — "
        "composite_z=0.0 (below MIN_SAMPLE=%d or all rate stats suppressed)",
        pid, player_type, window_days, MIN_SAMPLE,
    )
```

This surfaces data quality failures instead of silently returning 0.0.

---

## Files Changed

| File | Change |
|------|--------|
| `backend/fantasy_baseball/player_board.py` | Added `_PROXY_Z_BY_POSITION` dict; replaced `z_score=0.0` with position lookup + logger.warning |
| `backend/services/scoring_engine.py` | Added logging import + logger.warning when composite_z falls back to 0.0 |
| `tests/test_player_board_fuzzy.py` | 2 new position-tiered z_score tests |

**Test result:** 41/41 passing (39 existing + 2 new).

---

## Why Not Fix in fantasy.py

The `min(score, 58.0)` cap in `_projection_fallback_score` (fantasy.py line 355) is intentional —
it prevents proxy players from ever outscoring known players. The real fix is upstream: give
proxy players differentiated (sub-1.0) z_scores so they naturally stay below the cap while
still reflecting positional value differences.
