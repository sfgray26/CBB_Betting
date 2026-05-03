# Player Scoring Pipeline Audit — 2026-04-29

> **Agent:** Kimi CLI (research-only audit)  
> **Scope:** End-to-end audit of raw stats → computed tables → optimizer consumption  
> **Branch:** stable/cbb-prod, HEAD 79a644f

---

## 1. Executive Summary

- **The optimizer uses ONLY static Steamer projections from March 2026.** All 69,000+ rows in `player_rolling_stats` and `player_scores` are **computed daily but never read** by the lineup optimizer. This is a massive waste of compute and an untapped opportunity.
- **Every z_* column in `player_scores` is dead code** from the optimizer's perspective. They are computed at 4 AM ET daily by `_compute_player_scores` (confirmed wired in scheduler) but no method in `daily_lineup_optimizer.py` queries the `player_scores` table.
- **`rank_batters()` score formula is entirely projection-driven:** `hr`, `r`, `rbi`, `nsb`, `avg` from the `projections` parameter (which maps to `PlayerProjection` columns). No live rolling data, no Z-scores, no quality_score.
- **`rank_streamers()` is also projection-only:** Uses `k9`, `era`, `k`, `ip` from `PlayerProjection`. Does NOT use `probable_pitchers.quality_score` or any `player_scores` columns.
- **`score_0_100` and `composite_z` are computed for 100% of rows** (797/797 on the 7-day window) but have zero consumers.

---

## 2. Data Flow Diagram

```
MLB Stats API ──► mlb_player_stats ──┐
                                     │
                                     ▼
                         _compute_rolling_windows (3 AM ET)
                                     │
                                     ▼
                         player_rolling_stats (69,860 rows)
                         │ w_runs, w_tb, w_qs, w_era, etc.
                         │
                         ▼
              _compute_player_scores (4 AM ET)
                         │
                         ▼
              player_scores (69,599 rows)
              │ z_r, z_hr, z_rbi, z_ops, z_era, z_k_p, etc.
              │ composite_z, score_0_100
              │
              ▼
           ┌──────────────────────────────────────────────┐
           │                                              │
           │   NOBODY READS THIS TABLE                  │
           │                                              │
           │   daily_lineup_optimizer.py never queries  │
           │   player_scores or player_rolling_stats    │
           │                                              │
           └──────────────────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  rank_batters()     │ ◄── projections param
              │  rank_streamers()   │     (from PlayerProjection)
              │  solve_lineup()     │
              └─────────────────────┘
```

**The dashed box is the problem:** ~70K rows of rolling stats and Z-scores are computed every morning, then discarded.

---

## 3. TASK 1 — DB Query & Column Usage Audit

### 3.1 `daily_lineup_optimizer.py` — Every DB Query

| Method | Line | Query | Table | Columns Fetched | Used In |
|--------|------|-------|-------|-----------------|---------|
| `_load_schedule_fallback_games` | 324–332 | `SELECT team, opponent, is_home, park_factor FROM probable_pitchers WHERE game_date = ?` | `ProbablePitcherSnapshot` | `team`, `opponent`, `is_home`, `park_factor` | `fetch_mlb_odds()` fallback when Odds API fails |
| `_get_scarcity_rank` | 224–231 | `SELECT MIN(scarcity_rank) FROM position_eligibility WHERE primary_position = ? AND scarcity_rank IS NOT NULL` | `PositionEligibility` | `scarcity_rank` | `solve_lineup()` tiebreaker |
| `_fetch_probable_pitchers_for_date` | 877 | `load_probable_pitchers_from_snapshot(db, date)` | `ProbablePitcherSnapshot` | `team`, `pitcher_name` | `flag_pitcher_starts()` — start detection only |
| `_fetch_probable_pitchers_for_date` | 924 | `infer_probable_pitcher_map(db, date)` | `MLBPlayerStats`, `PlayerIDMapping` | `game_date`, `innings_pitched`, `raw_payload`, `bdl_player_id` | Fallback when MLB API fails |

**Tables NEVER queried by optimizer:**
- `player_projections` — data arrives via the `projections` parameter (passed in from caller)
- `player_scores` — **never queried**
- `player_rolling_stats` — **never queried**
- `player_daily_metrics` (VORP) — **never queried**

### 3.2 `rank_batters()` — Score Formula (lines 525–536)

```python
base_score = implied_runs * park_factor
proj_avg = proj.get("avg", 0.0)
stat_bonus = (
    proj.get("hr", 0) * 2.0
    + proj.get("r", 0) * 0.3
    + proj.get("rbi", 0) * 0.3
    + proj.get("nsb", 0) * 0.5
    + proj_avg * 5.0
)
lineup_score = base_score + stat_bonus * 0.1
```

**Projection keys used:** `avg`, `hr`, `r`, `rbi`, `nsb`  
**Source:** `projections` parameter → `PlayerProjection` rows (Steamer CSV, static March 2026)  
**Live data used:** NONE

### 3.3 `rank_streamers()` — Score Formula (lines 604–622)

```python
k9 = proj.get("k9", 0.0)
era = proj.get("era", 5.0)
...
env_score = max(0.0, (5.5 - implied_opp_runs) / 2.0) * 10
k_score = min(10.0, k9 - 5.0)
park_score = (2.0 - park_factor) * 5
stream_score = env_score * 0.5 + k_score * 0.3 + park_score * 0.2
```

**Projection keys used:** `k9`, `era`, `k`, `ip`  
**Source:** `projections` parameter → `PlayerProjection` rows  
**Live data used:** NONE (not even `quality_score` from probable pitchers)

### 3.4 `PlayerProjection` Model — All Columns

**File:** `backend/models.py:796–853`

| Column | Type | Used by Optimizer? |
|--------|------|-------------------|
| `player_id` | String | No (identifier only) |
| `player_name` | String | No (lookup by name from projections param) |
| `team` | String | No |
| `positions` | JSON | No |
| `woba` | Float | **No** |
| `avg` | Float | **YES** (`rank_batters` line 528) |
| `obp` | Float | **No** |
| `slg` | Float | **No** |
| `ops` | Float | **No** |
| `xwoba` | Float | **No** |
| `hr` | Integer | **YES** (`rank_batters` line 530) |
| `r` | Integer | **YES** (`rank_batters` line 531) |
| `rbi` | Integer | **YES** (`rank_batters` line 532) |
| `sb` | Integer | **No** (optimizer uses `nsb` key instead) |
| `era` | Float | **YES** (`rank_streamers` line 605) |
| `whip` | Float | **No** |
| `k_per_nine` | Float | **YES** (`rank_streamers` line 604 as `k9`) |
| `bb_per_nine` | Float | **No** |
| `w`, `l`, `hr_pit`, `k_pit`, `qs`, `nsv` | Int | **No** |
| `cat_scores` | JSONB | **No** |

**Utilization rate:** 6 of ~25 columns (24%) are actively used by the optimizer.

### 3.5 `PlayerScore` Model — All Columns

**File:** `backend/models.py:1341–1400`

| Column | Type | Used by Optimizer? | Production Coverage (7d window) |
|--------|------|-------------------|--------------------------------|
| `z_r` | Float | **DEAD** | 399/797 (50.1%) |
| `z_h` | Float | **DEAD** | 399/797 (50.1%) |
| `z_hr` | Float | **DEAD** | 399/797 (50.1%) |
| `z_rbi` | Float | **DEAD** | 399/797 (50.1%) |
| `z_sb` | Float | **DEAD** | 399/797 (50.1%) |
| `z_nsb` | Float | **DEAD** | 399/797 (50.1%) |
| `z_k_b` | Float | **DEAD** | 399/797 (50.1%) |
| `z_tb` | Float | **DEAD** | 399/797 (50.1%) |
| `z_avg` | Float | **DEAD** | 395/797 (49.6%) |
| `z_obp` | Float | **DEAD** | 395/797 (49.6%) |
| `z_ops` | Float | **DEAD** | 395/797 (49.6%) |
| `z_era` | Float | **DEAD** | 403/797 (50.6%) |
| `z_whip` | Float | **DEAD** | 403/797 (50.6%) |
| `z_k_per_9` | Float | **DEAD** | 403/797 (50.6%) |
| `z_k_p` | Float | **DEAD** | 405/797 (50.8%) |
| `z_qs` | Float | **DEAD** | 405/797 (50.8%) |
| `composite_z` | Float | **DEAD** | 797/797 (100%) |
| `score_0_100` | Float | **DEAD** | 797/797 (100%) |
| `confidence` | Float | **DEAD** | 797/797 (100%) |

**Verdict:** Every single column is **DEAD (computed, not consumed)**.

### 3.6 `PlayerRollingStats` Model — Key Columns

**File:** `backend/models.py:1263–1338`

All rolling columns (`w_runs`, `w_tb`, `w_qs`, `w_era`, `w_k_per_9`, etc.) are **intermediate inputs** to `scoring_engine.compute_league_zscores()`. They are consumed only by the scoring engine to produce Z-scores. The optimizer never reads them directly.

---

## 4. TASK 2 — "Dead Column" Classification

| Column | Status | Evidence |
|--------|--------|----------|
| `z_r` | **DEAD (computed, not consumed)** | Computed daily from `w_runs`. Never queried by optimizer. |
| `z_h` | **DEAD (computed, not consumed)** | Computed daily from `w_hits`. Never queried by optimizer. |
| `z_hr` | **DEAD (computed, not consumed)** | Computed daily from `w_home_runs`. Never queried by optimizer. |
| `z_rbi` | **DEAD (computed, not consumed)** | Computed daily from `w_rbi`. Never queried by optimizer. |
| `z_sb` | **DEAD (computed, not consumed)** | Computed daily from `w_stolen_bases`. Excluded from composite_z. Never queried. |
| `z_nsb` | **DEAD (computed, not consumed)** | Computed daily from `w_net_stolen_bases`. Never queried by optimizer. |
| `z_k_b` | **DEAD (computed, not consumed)** | Computed daily from `w_strikeouts_bat`. Never queried by optimizer. |
| `z_tb` | **DEAD (computed, not consumed)** | Computed daily from `w_tb`. Never queried by optimizer. |
| `z_avg` | **DEAD (computed, not consumed)** | Computed daily from `w_avg`. Never queried by optimizer. |
| `z_obp` | **DEAD (computed, not consumed)** | Computed daily from `w_obp`. Never queried by optimizer. |
| `z_ops` | **DEAD (computed, not consumed)** | Computed daily from `w_ops`. Never queried by optimizer. |
| `z_era` | **DEAD (computed, not consumed)** | Computed daily from `w_era`. Never queried by optimizer. |
| `z_whip` | **DEAD (computed, not consumed)** | Computed daily from `w_whip`. Never queried by optimizer. |
| `z_k_per_9` | **DEAD (computed, not consumed)** | Computed daily from `w_k_per_9`. Never queried by optimizer. |
| `z_k_p` | **DEAD (computed, not consumed)** | Computed daily from `w_strikeouts_pit`. Never queried by optimizer. |
| `z_qs` | **DEAD (computed, not consumed)** | Computed daily from `w_qs`. Never queried by optimizer. |
| `composite_z` | **DEAD (computed, not consumed)** | Computed for 100% of rows. Never queried. |
| `score_0_100` | **DEAD (computed, not consumed)** | Computed for 100% of rows. Never queried. |
| `confidence` | **DEAD (computed, not consumed)** | Computed for 100% of rows. Never queried. |

---

## 5. TASK 3 — Steamer vs. Live Data

### Explicit Answer: **The optimizer uses ONLY static Steamer projections.**

**Proof:**

1. `rank_batters()` (line 497) builds `proj_by_name` from the `projections` parameter:
   ```python
   proj_by_name = {p["name"].lower(): p for p in projections
                   if p.get("type") == "batter" or p.get("player_type") == "batter"}
   ```
   This is a Python dict passed in from the caller, not a DB query.

2. The projection dict keys consumed are: `avg`, `hr`, `r`, `rbi`, `nsb` (batters) and `k9`, `era`, `k`, `ip` (pitchers). These match the `PlayerProjection` schema.

3. **No `SessionLocal()` call** inside `rank_batters()` or `rank_streamers()`.

4. **No reference to `PlayerScore`, `PlayerRollingStats`, or `player_scores`, `player_rolling_stats`** anywhere in `daily_lineup_optimizer.py`.

5. The `PlayerProjection` model's `prior_source` defaults to `'steamer'`, and the docstring says it combines prior (Steamer/ZiPS) with likelihood (recent performance) using shrinkage. However, the `update_method` defaults to `'prior'`, meaning **no Bayesian update is actually applied** unless explicitly triggered.

### The Opportunity

If the optimizer were to use live rolling data instead of (or in addition to) Steamer projections:
- **Early-season** (now, April 29): Steamer is still somewhat reliable, but rolling data captures breakouts and slumps.
- **Mid-season** (July+): Rolling data should dominate; Steamer priors are stale.
- **Waiver decisions**: Rolling Z-scores (`composite_z`, `score_0_100`) are designed exactly for this — comparing players across categories on a common scale.

---

## 6. TASK 4 — Pitcher Ranking Audit

### `rank_streamers()` — Current Behavior

**File:** `backend/fantasy_baseball/daily_lineup_optimizer.py:570–646`

| Input | Source | Key |
|-------|--------|-----|
| `free_agents` | Yahoo API (`get_free_agents('SP')`) | `name`, `team`, `status`, `positions` |
| `projections` | `PlayerProjection` rows | `k9` (`k_per_nine`), `era`, `k` (`k_pit`), `ip` |
| `team_odds` | Odds API / ProbablePitcherSnapshot fallback | `opponent`, `implied_runs`, `park_factor`, `is_home` |

**Score formula:**
```python
env_score = max(0.0, (5.5 - implied_opp_runs) / 2.0) * 10   # 0-10
k_score   = min(10.0, k9 - 5.0)                             # 0-10
park_score = (2.0 - park_factor) * 5                         # pitcher parks bonus
stream_score = env_score * 0.5 + k_score * 0.3 + park_score * 0.2
```

**Gaps:**
1. **`quality_score` from `probable_pitchers` is NOT used.** The optimizer knows who the opponent's pitcher is (via `team_odds`), but doesn't adjust the SP's score based on whether they're facing an ace or a replacement-level pitcher.
2. **No use of `z_era`, `z_k_p`, `z_qs` from `player_scores`.** Live performance Z-scores could replace or augment the static `era` and `k9` projections.
3. **No use of `w` (wins) or `qs` (quality starts) projections.** The optimizer only cares about K/9, ERA, and matchup environment.

### `flag_pitcher_starts()` — Start Detection

**File:** `backend/fantasy_baseball/daily_lineup_optimizer.py:791–854`

This method determines if an SP on the roster has a start today. It:
1. Fetches probable pitchers from `ProbablePitcherSnapshot` (line 809)
2. Fuzzy-matches roster pitcher names against probable starters (line 837)
3. Returns `has_start: bool`

**Does NOT use `quality_score` for start/sit decisions.** It only answers "does he start?" not "should we start him?"

---

## 7. Top 3 Live Data Columns with Highest Value/Effort Ratio

If the optimizer were to start using live data, these three columns offer the best ROI:

### 1. `composite_z` / `score_0_100` from `player_scores` — **HIGHEST VALUE**
- **What it is:** A single number (-3 to +3 for `composite_z`, 0-100 for `score_0_100`) that summarizes all batting or pitching categories into one league-relative score.
- **How to use it:** Replace the entire `stat_bonus` computation in `rank_batters()` with `composite_z * some_scale`. This immediately gives the optimizer access to 11 categories of live rolling data with one DB lookup.
- **Effort:** LOW. One query: `SELECT bdl_player_id, composite_z, score_0_100 FROM player_scores WHERE as_of_date = ? AND window_days = 7`.
- **Risk:** MEDIUM. Need to map `yahoo_player_key` → `bdl_player_id` (the ID mismatch documented in previous audits).

### 2. `z_nsb` (Net Stolen Bases) — **HIGH VALUE**
- **What it is:** The Z-score for `SB - CS`, the most scarce and valuable H2H category.
- **How to use it:** Add `z_nsb * weight` to the `stat_bonus` in `rank_batters()`. Currently the optimizer uses `nsb` from Steamer projections, which doesn't capture recent hot streaks (e.g., a player who suddenly started running more).
- **Effort:** LOW. Already computed; just read the column.
- **Risk:** LOW. Direct replacement for existing `nsb` projection input.

### 3. `z_k_p` (Pitching Strikeouts Z-score) — **HIGH VALUE**
- **What it is:** Z-score for raw pitching strikeouts, a dominant H2H category.
- **How to use it:** Replace or augment `k9` in `rank_streamers()` with `z_k_p`. A pitcher with a static Steamer K/9 of 9.5 might actually be striking out 12 batters per 9 over the last 14 days (breakout) or 6.5 (slump/injury).
- **Effort:** LOW. Already computed; just read the column.
- **Risk:** LOW. Direct replacement for existing `k9` input.

---

## 8. Top 3 Recommendations

| Rank | Recommendation | Value | Effort | File(s) |
|------|---------------|-------|--------|---------|
| 1 | **Wire `composite_z` into `rank_batters()`** as the primary stat bonus component. Replace the static `hr*2 + r*0.3 + rbi*0.3 + nsb*0.5 + avg*5` formula with a live `composite_z`-driven score. This instantly activates 11 categories of rolling data. | **Very High** — makes the entire P13/P14 pipeline useful | Medium — requires ID bridge + formula tuning | `daily_lineup_optimizer.py` |
| 2 | **Wire `z_k_p` into `rank_streamers()`** to replace or augment static `k9`. Also wire `z_era` for `era`. This makes SP streaming recommendations reactive to breakouts and slumps. | **High** — immediate waiver/FAAB impact | Low — simple column swap | `daily_lineup_optimizer.py` |
| 3 | **Stop computing unused columns** (optional cleanup). If the optimizer will never use `z_obp`, `z_whip`, etc., consider dropping them from `_compute_player_scores` to reduce CPU and storage. However, future features (waiver edge detector, dashboard) may need them, so **keep them** and just wire the ones listed above. | **Low** — saves ~5% compute | Low | `scoring_engine.py` |

---

## 9. Raw Evidence

### `daily_lineup_optimizer.py:497–498` — projection-only input
```python
proj_by_name = {p["name"].lower(): p for p in projections
                if p.get("type") == "batter" or p.get("player_type") == "batter"}
```

### `daily_lineup_optimizer.py:528–536` — score formula using only projection keys
```python
proj_avg = proj.get("avg", 0.0)
stat_bonus = (
    proj.get("hr", 0) * 2.0
    + proj.get("r", 0) * 0.3
    + proj.get("rbi", 0) * 0.3
    + proj.get("nsb", 0) * 0.5
    + proj_avg * 5.0
)
lineup_score = base_score + stat_bonus * 0.1
```

### `daily_ingestion.py:724–730` — _compute_player_scores IS wired in scheduler
```python
self._scheduler.add_job(
    self._compute_player_scores,
    CronTrigger(hour=4, minute=0, timezone=tz),
    id="player_scores",
    name="Player Z-Score Scoring",
    replace_existing=True,
)
```

### `scoring_engine.py:42–62` — category mapping from rolling_stats → z-scores
```python
HITTER_CATEGORIES: dict[str, tuple[str, bool]] = {
    "z_r":    ("w_runs",              False),
    "z_h":    ("w_hits",              False),
    "z_hr":   ("w_home_runs",         False),
    ...
}
PITCHER_CATEGORIES: dict[str, tuple[str, bool]] = {
    "z_era":     ("w_era",         True),
    "z_whip":    ("w_whip",        True),
    ...
}
```

### Production `player_scores` coverage (7-day window, latest date)
```
 total | z_r | z_h | z_hr | z_rbi | z_sb | z_nsb | z_k_b | z_tb | z_avg | z_obp | z_ops | z_era | z_whip | z_k_per_9 | z_k_p | z_qs | composite_z | score_0_100
-------+-----+-----+------+-------+------+-------+-------+------+-------+-------+-------+-------+--------+-----------+-------+------+-------------+-------------
   797 | 399 | 399 |  399 |   399 |  399 |   399 |   399 |  399 |   395 |   395 |   395 |   403 |    403 |       403 |   405 |  405 |         797 |         797
```

---

*Report generated by Kimi CLI at 2026-04-29. Read-only audit — no files modified.*
