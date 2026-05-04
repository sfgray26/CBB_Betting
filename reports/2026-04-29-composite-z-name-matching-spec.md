# Composite-Z Name Matching + `rank_streamers()` Integration Spec

> **Agent:** Kimi CLI (research-only audit)  
> **Scope:** Verify name-format consistency across all optimizer data paths; spec `composite_z` wire-up for `rank_streamers()`  
> **Branch:** stable/cbb-prod, HEAD 79a644f

---

## 1. Name Matching Verdict: **SAFE**

### 1.1 Name Source Chain

All player names that reach the optimizer flow through **`_parse_player()`** in `yahoo_client_resilient.py` (lines 1284–1402). This function normalizes names identically for every Yahoo API endpoint:

| Caller | Method | Line | Name Source |
|--------|--------|------|-------------|
| `rank_batters()` | `get_roster()` → `_parse_player()` | 591 | `meta.get("full_name")` → `meta["name"].get("full")` → `meta.get("name")` |
| `_sync_position_eligibility()` | `get_league_rosters()` → `_parse_player()` | 474 | Same as above |
| `rank_streamers()` | `get_free_agents()` → `_parse_players_block()` → `_parse_player()` | 683 | Same as above |

### 1.2 Normalization Applied by `_parse_player()`

**File:** `backend/fantasy_baseball/yahoo_client_resilient.py:1362–1390`

```python
# 1. Extract from Yahoo's nested structure
name = meta.get("full_name")
if not name and isinstance(meta.get("name"), dict):
    name = meta["name"].get("full")
if not name:
    name = meta.get("name", "Unknown")

# 2. Unicode NFC normalization (ensures "Díaz" stays composed)
if isinstance(name, str):
    name = unicodedata.normalize("NFC", name)

# 3. Strip injury descriptions appended by Yahoo
# e.g. "Jason Adam Quadriceps" → "Jason Adam"
name = re.sub(
    r"\s+(?:Quadriceps|Hamstring|Shoulder|Elbow|Hip|Knee|Back|Wrist|Ankle|"
    r"Oblique|Forearm|Calf|Groin|Thumb|Finger|Ribs?|Concussion|"
    r"Strain|Sprain|Fracture|Tear|Surgery|Illness|Fatigue|IL|DL)\b.*$",
    "", name, flags=re.IGNORECASE,
).strip()
```

**Result:** `position_eligibility.player_name` and `roster`/`free_agents` dicts both receive the **exact same normalized string** because they share the same `_parse_player()` call path.

### 1.3 Evidence from `_sync_position_eligibility`

**File:** `backend/services/daily_ingestion.py:5443–5450`

```python
for player_data in all_players:
    player_key = player_data.get("player_key")
    name = player_data.get("name", "Unknown")  # ← already normalized by _parse_player()
```

`all_players` comes from `get_league_rosters()`, which calls `_parse_player()` for every player before returning.

### 1.4 Evidence from `rank_batters()` and `rank_streamers()`

**File:** `backend/fantasy_baseball/daily_lineup_optimizer.py:531,627`

```python
# rank_batters()
name = player.get("name", "")  # ← from get_roster() → _parse_player()

# rank_streamers()
name = player.get("name", "")  # ← from get_free_agents() → _parse_players_block() → _parse_player()
```

### 1.5 Edge Cases Handled

| Edge Case | Handling | Risk Level |
|-----------|----------|------------|
| Accented characters (Díaz, Soto) | NFC unicode normalization | **None** |
| Injury suffixes ("Quadriceps", "IL") | Stripped by regex in `_parse_player()` | **None** |
| Jr./Sr./III suffixes | **Not stripped** — remains part of name | **Low** — Yahoo is consistent |
| Middle names | Yahoo `full_name` includes them | **None** — consistent across endpoints |
| All-caps vs mixed case | Lowercased at lookup time (`name.lower()`) | **None** |

### 1.6 Conclusion

**SAFE — straight lowercase match is sufficient.** No additional normalization (fuzzy matching, Soundex, etc.) is needed because all name strings originate from the same `_parse_player()` normalization pipeline.

> **One caution:** If `_sync_position_eligibility` is ever changed to call a different Yahoo endpoint that bypasses `_parse_player()`, the name formats could diverge. Document this dependency in the code.

---

## 2. `rank_streamers()` Integration Spec

### 2.1 Current `rank_streamers()` Behavior

**File:** `backend/fantasy_baseball/daily_lineup_optimizer.py:598–674`

**Inputs:**
- `free_agents`: List of dicts from `YahooFantasyClient.get_free_agents('SP')` — keys: `name`, `team`, `status`, `positions`
- `projections`: List of dicts from `PlayerProjection` — keys consumed: `k9` (`k_per_nine`), `era`, `k` (`k_pit`), `ip`

**Score formula (lines 647–650):**
```python
env_score  = max(0.0, (5.5 - implied_opp_runs) / 2.0) * 10   # 0–10
k_score    = min(10.0, k9 - 5.0)                             # 0–10
park_score = (2.0 - park_factor) * 5                         # –5 to +5
stream_score = env_score * 0.5 + k_score * 0.3 + park_score * 0.2
```

**Current stream_score range:** approximately **–1 to +9** (realistic range: 2–8).

### 2.2 Pitchers Are in `position_eligibility`

Confirmed by production query:

```
 player_type | total | has_name
-------------+-------+----------
 pitcher     |  1209 |     1209
 batter      |  1180 |     1180
```

Pitchers have `can_play_sp = True` or `can_play_rp = True` in `position_eligibility`. The same `player_name` → `composite_z` join path works for them.

### 2.3 Pitcher `composite_z` Composition

**File:** `backend/services/scoring_engine.py:56–62`

```python
PITCHER_CATEGORIES: dict[str, tuple[str, bool]] = {
    "z_era":     ("w_era",         True),   # lower ERA is better
    "z_whip":    ("w_whip",        True),   # lower WHIP is better
    "z_k_per_9": ("w_k_per_9",     False),
    "z_k_p":     ("w_strikeouts_pit", False),
    "z_qs":      ("w_qs",          False),
}
```

Pitcher `composite_z` = weighted sum of `z_era`, `z_whip`, `z_k_per_9`, `z_k_p`, `z_qs`. It already captures live rolling performance across all pitcher categories.

### 2.4 Recommended Integration

**Option: Additive augmentation (RECOMMENDED)**

Add `composite_z * 0.5` to the existing `stream_score`. This preserves the current formula while augmenting it with live data.

```python
# AFTER existing stream_score computation:
composite_z = composite_z_lookup.get(name.lower(), 0.0)
stream_score = env_score * 0.5 + k_score * 0.3 + park_score * 0.2 + composite_z * 0.5
```

**Rationale:**
- `composite_z` range: –3.0 to +3.0 (capped at ±3.0 by `Z_CAP`)
- Contribution to stream_score: –1.5 to +1.5
- This is **material** (comparable to `park_score`'s full range of ±1.0 after weighting)
- But **not dominant** — environment (implied runs) and K/9 still drive the score

**Why not replacement?** Replacing `k_score` with `composite_z` would discard the matchup-specific `env_score` and `park_score` adjustments, which are the optimizer's unique value-add over raw Z-scores.

### 2.5 Exact Line Insertions

**File:** `backend/fantasy_baseball/daily_lineup_optimizer.py`

**Step 1:** Add composite_z lookup alongside existing `proj_by_name` (after line 619):

```python
# Existing code (lines 619-620):
proj_by_name = {p["name"].lower(): p for p in projections
                if (p.get("type") or p.get("player_type", "")) == "pitcher"}

# NEW: Fetch live composite_z for pitchers
target_date = self._parse_game_date(game_date)
composite_z_lookup: Dict[str, float] = {}
if target_date is not None:
    from backend.models import PlayerScore, PositionEligibility
    _db = SessionLocal()
    try:
        # Join position_eligibility (has player_name) → player_scores (has composite_z)
        rows = (
            _db.query(PositionEligibility.player_name, PlayerScore.composite_z)
            .join(PlayerScore, PositionEligibility.bdl_player_id == PlayerScore.bdl_player_id)
            .filter(
                PlayerScore.as_of_date == target_date,
                PlayerScore.window_days == 7,
                PositionEligibility.player_type == "pitcher",
            )
            .all()
        )
        composite_z_lookup = {r.player_name.lower(): r.composite_z for r in rows if r.composite_z is not None}
    except Exception:
        pass
    finally:
        _db.close()
```

**Step 2:** Apply composite_z inside the free_agents loop (after line 650):

```python
# Existing code (lines 647-650):
env_score = max(0.0, (5.5 - implied_opp_runs) / 2.0) * 10
k_score = min(10.0, k9 - 5.0)
park_score = (2.0 - park_factor) * 5
stream_score = env_score * 0.5 + k_score * 0.3 + park_score * 0.2

# NEW: Augment with live composite_z
live_z = composite_z_lookup.get(name.lower(), 0.0)
stream_score += live_z * 0.5

# Update reason string
if live_z > 1.0:
    reason_parts.append(f"hot form (z={live_z:.1f})")
elif live_z < -1.0:
    reason_parts.append(f"cold form (z={live_z:.1f})")
```

### 2.6 Fallback Behavior

- If `composite_z_lookup` is empty (DB failure or no rows): `live_z = 0.0` → no change to stream_score.
- If pitcher not found in lookup (name mismatch): `live_z = 0.0` → falls back to projection-only scoring.
- If `target_date` is None: skip lookup entirely.

---

## 3. Other Integration Points (Task S3)

### 3.1 `solve_lineup()`

**Verdict:** No direct changes needed.

`solve_lineup()` (line 680) only consumes the output of `rank_batters()`:

```python
ranked: List[BatterRanking] = self.rank_batters(roster, projections, game_date)
```

It then performs slot assignment using `BatterRanking.lineup_score`, which is already computed by `rank_batters()`. As long as `rank_batters()` incorporates `composite_z` into `lineup_score`, `solve_lineup()` automatically benefits.

**No DB queries inside `solve_lineup()`** — it is pure constraint logic.

### 3.2 `LineupConstraintSolver`

**Verdict:** NOT connected to `daily_lineup_optimizer.py`.

`LineupConstraintSolver` (in `lineup_constraint_solver.py`) is a standalone OR-Tools-based optimizer. It is **not imported or called** by `daily_lineup_optimizer.py`. The grep returned zero matches.

If Claude ever wires `LineupConstraintSolver` into the main pipeline, it would need its own `player_scores` dict (currently `Dict[str, EliteScore]`), which is a different data structure.

### 3.3 `EliteLineupScorer`

**Verdict:** NOT connected to `daily_lineup_optimizer.py`.

`EliteLineupScorer` (in `elite_lineup_scorer.py`) is a research-grade multi-factor scorer that uses `BatterProfile` and `PitcherProfile` dataclasses. It is **not imported or called** by `daily_lineup_optimizer.py`.

Importantly, `EliteLineupScorer` does **not** query `player_scores` or `composite_z` either. It operates entirely on hand-constructed `BatterProfile` objects.

### 3.4 Cache Invalidation

**Verdict:** None needed.

`DailyLineupOptimizer` is **stateless**. Every call to `rank_batters()`, `rank_streamers()`, or `solve_lineup()` computes scores fresh. There are no cached rankings, no memoized DB results, and no background workers that precompute optimizer outputs.

The `self._odds_cache` (line 245) caches Odds API responses by date, but this is independent of player data.

---

## 4. Summary

| Item | Verdict | Evidence |
|------|---------|----------|
| Name matching | **SAFE** | All paths converge on `_parse_player()` which applies identical NFC normalization + injury suffix stripping |
| `rank_streamers()` composite_z | **Ready to wire** | Same name-based join as batters; pitchers exist in `position_eligibility` (1,209 rows); additive `composite_z * 0.5` recommended |
| `solve_lineup()` changes | **None needed** | Consumes `rank_batters()` output; no direct DB access |
| `LineupConstraintSolver` | **Out of scope** | Not imported by `daily_lineup_optimizer.py` |
| Cache invalidation | **None needed** | Optimizer is stateless |

---

*Report generated by Kimi CLI at 2026-04-29. Read-only audit — no files modified.*
