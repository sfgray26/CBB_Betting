# K1 Rolling Stats & Database Code Audit
**Date:** 2026-04-18  
**Author:** Kimi CLI  
**Scope:** Read-only forensic audit of `player_rolling_stats`, `mlb_player_stats`, scoring pipeline, and v2 canonical stat contract alignment.

---

## Executive Summary

The `player_rolling_stats` table stores **descriptive column names** (`w_home_runs`, `w_rbi`, `w_stolen_bases`, etc.), not v1 codes or raw Yahoo `stat_id`s. This is the *correct* naming convention — no column rename migration is needed.

The gap is **coverage**, not naming. Only **9 of 18** v2 canonical scoring categories have Z-score computation paths in `scoring_engine.py`. The rolling window engine captures most raw weighted sums needed for the missing categories, but **5 pitching counting stats are absent from the upstream `mlb_player_stats` table entirely** (W, L, QS, SV, BS), blocking their inclusion in rolling windows and therefore in Z-scores.

---

## 1. What Stat Codes Are Stored in `player_rolling_stats`?

### Column Naming Convention

All columns use descriptive snake_case prefixed with `w_` (weighted):

| Column | Semantics |
|--------|-----------|
| `w_ab`, `w_hits`, `w_doubles`, `w_triples`, `w_home_runs`, `w_rbi`, `w_walks`, `w_strikeouts_bat`, `w_stolen_bases`, `w_caught_stealing`, `w_net_stolen_bases` | Batting weighted sums |
| `w_avg`, `w_obp`, `w_slg`, `w_ops` | Batting derived rates |
| `w_ip`, `w_earned_runs`, `w_hits_allowed`, `w_walks_allowed`, `w_strikeouts_pit` | Pitching weighted sums |
| `w_era`, `w_whip`, `w_k_per_9` | Pitching derived rates |
| `w_games`, `games_in_window` | Metadata |

**Verdict:** These are **v2-style descriptive names**, not v1 codes like `HR`, `HRA`, `K(B)`. There is no translation layer between a legacy code system and the DB columns — the columns were designed with the current canonical contract in mind.

### Source of Truth
- `backend/models.py:1143-1216` — `PlayerRollingStats` ORM definition
- `backend/services/rolling_window_engine.py:68-113` — `RollingWindowResult` dataclass mirrors the ORM exactly

---

## 2. Which of the 18 v2 Categories Are Already Computed?

The v2 canonical contract defines 18 scoring categories:

**Batting (9):** `R`, `H`, `HR_B`, `RBI`, `K_B`, `TB`, `AVG`, `OPS`, `NSB`  
**Pitching (9):** `W`, `L`, `HR_P`, `K_P`, `ERA`, `WHIP`, `K_9`, `QS`, `NSV`

### Currently Computed Z-Scores (scoring_engine.py)

`scoring_engine.py` computes exactly **9 Z-score categories**:

| Z-key | Canonical Mapping | Column in `player_rolling_stats` | Status |
|-------|-------------------|----------------------------------|--------|
| `z_hr` | `HR_B` | `w_home_runs` | ✅ Computed |
| `z_rbi` | `RBI` | `w_rbi` | ✅ Computed |
| `z_sb` | `SB` (legacy) | `w_stolen_bases` | ⚠️ Legacy, excluded from composite |
| `z_nsb` | `NSB` | `w_net_stolen_bases` | ✅ Computed (P27) |
| `z_avg` | `AVG` | `w_avg` | ✅ Computed |
| `z_obp` | `OBP` (display, not scoring) | `w_obp` | ✅ Computed |
| `z_era` | `ERA` | `w_era` | ✅ Computed |
| `z_whip` | `WHIP` | `w_whip` | ✅ Computed |
| `z_k_per_9` | `K_9` | `w_k_per_9` | ✅ Computed |

### Missing Z-Score Computations

| Canonical Code | Rolling Raw Data Present? | Z-Score Path Exists? | Blocker |
|----------------|---------------------------|----------------------|---------|
| `R` | ✅ `runs` in `mlb_player_stats` → would need `w_runs` in rolling result | ❌ No | Add `w_runs` to rolling engine + scoring_engine |
| `H` | ✅ `hits` in `mlb_player_stats` → `w_hits` exists | ❌ No | Add to scoring_engine only |
| `TB` | ⚠️ Derivable from `w_hits`, `w_doubles`, `w_triples`, `w_home_runs` | ❌ No | Add derived column `w_tb` to rolling result + scoring_engine |
| `K_B` | ✅ `strikeouts_bat` in `mlb_player_stats` → `w_strikeouts_bat` exists | ❌ No | Add to scoring_engine only |
| `OPS` | ✅ `w_ops` exists | ❌ No | Add to scoring_engine only |
| `W` | ❌ **NOT in `mlb_player_stats`** | ❌ No | Upstream data source missing |
| `L` | ❌ **NOT in `mlb_player_stats`** | ❌ No | Upstream data source missing |
| `HR_P` | ❌ **NOT in `mlb_player_stats`** | ❌ No | Upstream data source missing |
| `K_P` | ✅ `strikeouts_pit` → `w_strikeouts_pit` exists | ❌ No | Add to scoring_engine only |
| `QS` | ❌ **NOT in `mlb_player_stats`** | ❌ No | Upstream data source missing |
| `NSV` | ❌ **NOT in `mlb_player_stats`** (no SV/BS columns) | ❌ No | Upstream data source missing |

**Cumulative:** 9 Z-scores exist. 3 batting Z-scores need only scoring_engine wiring (`H`, `TB`, `OPS` — raw data present). 2 batting Z-scores need rolling engine + scoring_engine wiring (`R`, `K_B`). 1 pitching Z-score needs only scoring_engine wiring (`K_P`). **5 pitching Z-scores are blocked on upstream data** (`W`, `L`, `HR_P`, `QS`, `NSV`).

---

## 3. What Data Source Feeds Each Computed Category?

| Category | Data Source | Pipeline Stage |
|----------|-------------|----------------|
| All batting raw sums | BDL `mlb_player_stats` | `_ingest_mlb_box_stats` (2 AM) → `_compute_rolling_windows` (3 AM) |
| All pitching raw sums (existing) | BDL `mlb_player_stats` | Same |
| `W`, `L`, `QS`, `SV`, `BS`, `HLD` | **NOT CURRENTLY INGESTED** | Would need Yahoo `team_stats` or MLB Stats API `pitching` endpoint |
| `HR_P` (home runs allowed) | **NOT CURRENTLY INGESTED** | Would need Yahoo or MLB Stats API |

The BDL box stats feed (`mlb_player_stats`) is per-game batting/pitching lines. It includes runs, hits, HR, RBI, K, SB, CS, IP, ER, H_allowed, BB_allowed, K_pit, WHIP, ERA. It **does not** include team-context stats like W/L (which depend on game outcome) or QS/NSV (which require start-specific logic).

---

## 4. For Missing Categories: Data Source & Aggregation Method

| Missing Category | Feasible Data Source | Aggregation Method | Implementation Complexity |
|------------------|----------------------|--------------------|---------------------------|
| `R` | Already in BDL (`runs`) | `sum(w * runs)` — same as existing | Low |
| `H` | Already in BDL (`hits`) | `sum(w * hits)` — same as existing | Low |
| `TB` | Derivable from BDL (`hits`, `doubles`, `triples`, `home_runs`) | `sum(w * (H + 2B + 2*3B + 3*HR))` — add to rolling engine | Low |
| `K_B` | Already in BDL (`strikeouts_bat`) | `sum(w * so_bat)` — same as existing | Low |
| `OPS` | Already in BDL (derived `ops`) or compute from `w_obp + w_slg` | Use `w_ops` already computed in rolling engine | Low |
| `K_P` | Already in BDL (`strikeouts_pit`) | `sum(w * k_pit)` — same as existing | Low |
| `W` | Yahoo `team_stats` or MLB Stats API `pitching` → `wins` | `sum(w * wins)` per start | Medium — new data source |
| `L` | Same as W | `sum(w * losses)` per start | Medium — new data source |
| `HR_P` | Yahoo `team_stats` or MLB Stats API `homeRunsAllowed` | `sum(w * hr_allowed)` | Medium — new data source |
| `QS` | MLB Stats API `qualityStarts` or compute from game log (6+ IP, ≤3 ER) | `sum(w * qs)` or compute per-game | Medium-High — may need game-log parsing |
| `NSV` | Yahoo `team_stats` or MLB Stats API `saves`/`blownSaves` | `sum(w * (sv - bs))` | Medium — new data source |

---

## 5. Is There a Translation Layer?

**Yes, but it's simple and one-directional.**

In `scoring_engine.py:42-55`:

```python
HITTER_CATEGORIES: dict[str, tuple[str, bool]] = {
    "z_hr":  ("w_home_runs",     False),
    "z_rbi": ("w_rbi",           False),
    ...
}
```

This maps `z_key` → `column_name_on_PlayerRollingStats`. There is no v1→v2 translation because the DB was never populated with v1 codes. The "translation" needed is **expanding this dictionary** to cover all 18 categories, not renaming existing columns.

A separate mapping exists in `compute_league_params()` (`scoring_engine.py:393-396`) that maps `z_key` → short stat key for the simulation engine:

```python
_Z_TO_SHORT = {
    "z_hr": "hr", "z_rbi": "rbi", "z_sb": "sb", "z_nsb": "nsb",
    "z_avg": "avg", "z_obp": "obp",
    "z_era": "era", "z_whip": "whip", "z_k_per_9": "k",
}
```

This also needs expansion for the missing categories.

---

## 6. What Would a v1→v2 Migration Look Like?

**Answer: No migration is needed.**

The DB columns already use v2-style descriptive names. The required changes are:

1. **Add new weighted-sum columns to `PlayerRollingStats`** / `RollingWindowResult`:
   - `w_runs` (for `R`)
   - `w_tb` (for `TB`)
   - `w_hr_allowed` (for `HR_P`) — pending upstream data
   - `w_wins`, `w_losses`, `w_qs`, `w_sv`, `w_bs` — pending upstream data

2. **Expand `HITTER_CATEGORIES` and `PITCHER_CATEGORIES`** in `scoring_engine.py` to cover all 18 codes.

3. **Expand `_Z_TO_SHORT`** in `compute_league_params()`.

4. **Expand `PlayerScoreResult` dataclass** and `PlayerScore` ORM to persist the new Z-scores.

5. **Add upstream ingestion** for W, L, QS, SV, BS, HR_P from Yahoo or MLB Stats API into `mlb_player_stats` (or a new `mlb_pitcher_outcomes` table).

---

## Critical Finding: The 5 Pitching Counting Stats Are Greenfield Data

The rolling window engine cannot compute what it does not ingest. `mlb_player_stats` has **no columns** for:
- `wins`
- `losses`
- `home_runs_allowed`
- `quality_starts`
- `saves` / `blown_saves`

These require either:
- **Yahoo augmentation:** Fetch per-player season stats from Yahoo (`get_player_stats_batch`) and merge into the rolling window computation at scoring time.
- **MLB Stats API augmentation:** Add new ingestion pipeline stage for pitcher outcome stats.
- **Compute QS from game logs:** A start is a QS if `ip >= 6.0` and `er <= 3`. The existing `mlb_player_stats` has both `innings_pitched` and `earned_runs`, so QS can be derived per-game during rolling window computation without a new data source.

**QS is the only missing category that can be computed from existing data.**

---

## Recommended Next Actions (for Claude)

| Priority | Action | Files | Effort |
|----------|--------|-------|--------|
| P0 | Add `w_runs`, `w_tb` to rolling engine + ORM + upsert | `rolling_window_engine.py`, `models.py`, `daily_ingestion.py` | 2 hrs |
| P0 | Expand `HITTER_CATEGORIES` in scoring engine to cover R, H, TB, K_B, OPS | `scoring_engine.py`, `models.py`, `daily_ingestion.py` | 2 hrs |
| P0 | Expand `PITCHER_CATEGORIES` to cover K_P | `scoring_engine.py`, `models.py`, `daily_ingestion.py` | 1 hr |
| P1 | Derive QS per-game in rolling engine (`ip >= 6.0 and er <= 3`) | `rolling_window_engine.py`, `models.py` | 2 hrs |
| P1 | Add Yahoo season-stats fallback for W, L, HR_P, NSV when BDL data missing | `yahoo_client_resilient.py`, `daily_ingestion.py` | 1 day |
| P2 | Add MLB Stats API ingestion for W, L, HR_P, SV, BS | New module or `statsapi_supplement` | 2-3 days |
