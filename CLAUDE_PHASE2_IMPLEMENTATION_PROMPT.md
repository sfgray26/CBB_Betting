# Phase 2 Implementation Prompt — 18-Category Rolling Stats + ROW Projection Pipeline

> **Self-contained prompt.** No prior conversation context assumed.
> **Prerequisite:** Phase 0 COMPLETE (stat_contract + 6 UI contracts). Phase 1 COMPLETE (v1→v2 migration + 7 helpers, 2029 tests passing). Kimi K1–K5 research complete.

---

## 1. READ FIRST

Before writing any code, read these files in order:

**Architecture & State:**
1. `HANDOFF.md` — current operational state (Phase 1 COMPLETE, Phase 2 NEXT)
2. `backend/stat_contract/__init__.py` — v2 contract singleton (`SCORING_CATEGORY_CODES`, `BATTING_CODES`, `PITCHING_CODES`, `LOWER_IS_BETTER`, `YAHOO_ID_INDEX`)
3. `backend/contracts.py` — `CategoryStats`, `CanonicalPlayerRow` (esp. `row_projection` field, always `None` today), `MatchupScoreboardRow`, `FreshnessMetadata`

**Kimi Research (critical — contains verified formulas and data inventories):**
4. `reports/2026-04-18-rolling-stats-audit.md` — K1: which of 18 categories exist, which are wirable, which are greenfield
5. `reports/2026-04-18-row-projection-spec.md` — K2: projection math, ratio stat aggregation, function signature, risk assessment
6. `reports/2026-04-18-category-math-reference.md` — K4: margin sign convention, delta-to-flip formulas, per-category math sheets

**Code to modify:**
7. `backend/services/rolling_window_engine.py` — `RollingWindowResult` dataclass (lines 65–103), computation loop (lines 150–280)
8. `backend/services/scoring_engine.py` — `HITTER_CATEGORIES` (lines 31–38), `PITCHER_CATEGORIES` (lines 40–44), `_Z_TO_SHORT` (lines 404–412), `PlayerScoreResult` (lines 59–79)
9. `backend/models.py` — `PlayerRollingStats` (lines 1143–1217), `PlayerScore` (lines 1218–1280), `MLBPlayerStats` (lines 1016–1073)
10. `backend/fantasy_baseball/h2h_monte_carlo.py` — `HITTING_CATS` / `PITCHING_CATS` (lines 56–57), `_aggregate_roster` (lines 128–136)
11. `backend/services/constraint_helpers.py` — Phase 1 helpers (available for reuse)
12. `tests/test_constraint_helpers.py` — pattern reference for test style

---

## 2. MISSION

Phase 2 builds the **ROW projection pipeline** — the single highest-value gap in the system (blocks 18 UI fields across 4 pages). It has four workstreams executed in order:

| Workstream | Summary | Depends On |
|------------|---------|-----------|
| **A** | Expand rolling stats from 9→15 Z-score categories (wire existing data + derive QS) | Nothing |
| **B** | Build `row_projector.py` — pure function computing team-level ROW projections for all 18 categories | A |
| **C** | Build `category_math.py` — margin, delta-to-flip, classification | B |
| **D** | Wire into contracts + test harness | A + B + C |

**Out of scope for Phase 2 (deferred to Phase 3+):**
- Updating `h2h_monte_carlo.py` to use v2 codes (Phase 3 deliverable)
- Building API endpoints (Phase 4)
- Yahoo API augmentation for greenfield pitching stats W, L, HR_P, NSV (see §6)
- Frontend work (Phase 5+)

---

## 3. GROUND TRUTH: CURRENT STATE OF DATA

### What exists in `mlb_player_stats` (BDL box stats)

| Column | Canonical Map | Present? |
|--------|---------------|----------|
| `runs` | R | ✅ |
| `hits` | H | ✅ |
| `home_runs` | HR_B | ✅ |
| `rbi` | RBI | ✅ |
| `strikeouts_bat` | K_B | ✅ |
| `doubles`, `triples`, `home_runs` → TB | TB | ✅ (derivable) |
| `avg` | AVG | ✅ |
| `ops` | OPS | ✅ |
| `stolen_bases`, `caught_stealing` → NSB | NSB | ✅ |
| `strikeouts_pit` | K_P | ✅ |
| `earned_runs`, `innings_pitched` → ERA | ERA | ✅ |
| `hits_allowed`, `walks_allowed`, `innings_pitched` → WHIP | WHIP | ✅ |
| `strikeouts_pit`, `innings_pitched` → K_9 | K_9 | ✅ |
| `innings_pitched`, `earned_runs` → QS | QS | ✅ (derivable: IP≥6.0 AND ER≤3) |
| wins | W | ❌ NOT IN TABLE |
| losses | L | ❌ NOT IN TABLE |
| home_runs_allowed | HR_P | ❌ NOT IN TABLE |
| saves, blown_saves | NSV | ❌ NOT IN TABLE |

### What exists in `player_rolling_stats` (rolling window engine output)

| Field | v2 Code | Z-Score Exists? |
|-------|---------|-----------------|
| `w_home_runs` | HR_B | ✅ `z_hr` |
| `w_rbi` | RBI | ✅ `z_rbi` |
| `w_net_stolen_bases` | NSB | ✅ `z_nsb` |
| `w_avg` | AVG | ✅ `z_avg` |
| `w_obp` | OBP (display) | ✅ `z_obp` |
| `w_era` | ERA | ✅ `z_era` |
| `w_whip` | WHIP | ✅ `z_whip` |
| `w_k_per_9` | K_9 | ✅ `z_k_per_9` |
| `w_hits` | H | ⚠️ Data present, **no Z-score** |
| `w_ops` | OPS | ⚠️ Data present, **no Z-score** |
| `w_strikeouts_bat` | K_B | ⚠️ Data present, **no Z-score** |
| `w_strikeouts_pit` | K_P | ⚠️ Data present, **no Z-score** |
| (TB computed in loop as `sum_w_tb`) | TB | ⚠️ Computed but **NOT stored in dataclass or ORM** |
| (runs in mlb_player_stats) | R | ⚠️ Raw data exists, **not accumulated in rolling engine** |
| — | W | ❌ No upstream data |
| — | L | ❌ No upstream data |
| — | HR_P | ❌ No upstream data |
| — | QS | ❌ No upstream data (but **derivable** from IP + ER per game) |
| — | NSV | ❌ No upstream data |

### Summary: 9 Z-scores exist → Workstream A targets 15

- **6 categories need only scoring_engine wiring** (raw weighted sums already in DB or engine): `H`, `OPS`, `K_B`, `K_P` + `TB` (store existing computation) + `R` (add accumulator)
- **1 category derivable from existing data**: `QS` (IP≥6.0 AND ER≤3, per game)
- **4 categories are greenfield** — no upstream data at all: `W`, `L`, `HR_P`, `NSV`
- **Phase 2 delivers 15 Z-scores.** The 4 greenfield categories get `None` placeholders with a `TODO` comment; upstream data ingestion is Phase 2b/3.

---

## 4. WORKSTREAM A: EXPAND ROLLING STATS FROM 9→15 Z-SCORES

### A1. Add `w_runs` and `w_tb` to `RollingWindowResult` dataclass

**File:** `backend/services/rolling_window_engine.py`

In the `RollingWindowResult` dataclass (line ~83), add after `w_rbi`:
```python
w_runs: Optional[float] = None              # sum(weight × runs) — for R category
```

After `w_net_stolen_bases` (line ~89), add:
```python
w_tb: Optional[float] = None                # sum(weight × total_bases) — for TB category
```

**Note:** `w_tb` is already COMPUTED in the loop (line ~250: `sum_w_tb += w * tb`) but never stored in the result. You must:
1. Add the field to the dataclass
2. Assign `sum_w_tb` to `result.w_tb` at the end of computation
3. Do the same for `w_runs`: accumulate `sum_w_runs += w * _runs` in the loop, assign to result

### A2. Add `w_qs` to `RollingWindowResult` dataclass

After the pitching weighted sums section, add:
```python
w_qs: Optional[float] = None                # sum(weight × quality_start_flag) — derivable per-game
```

In the computation loop, for each pitching game line, derive QS:
```python
# QS derivation: start with IP >= 6.0 and ER <= 3
_ip_decimal = parse_ip(row.innings_pitched)  # already computed
_er = float(row.earned_runs or 0)
_qs_flag = 1.0 if (_ip_decimal >= 6.0 and _er <= 3.0) else 0.0
sum_w_qs += w * _qs_flag
```

**Important:** A QS requires a *start*, not just any appearance. If `mlb_player_stats` does not distinguish starts from relief appearances, treat any game with IP ≥ 6.0 as a start (relief pitchers virtually never pitch 6+ IP). This is a safe heuristic.

### A3. Add `w_runs`, `w_tb`, `w_qs` to `PlayerRollingStats` ORM

**File:** `backend/models.py`

Add columns to the `PlayerRollingStats` class after the existing batting/pitching weighted sums:
```python
# Phase 2: new weighted sums for 18-category expansion
w_runs          = Column(Float, nullable=True)   # for R category
w_tb            = Column(Float, nullable=True)   # for TB category (H + 2B + 2×3B + 3×HR)
w_qs            = Column(Float, nullable=True)   # for QS category (IP≥6, ER≤3)
```

**Alembic migration:** Create `alembic revision --autogenerate -m "phase2_rolling_stats_expansion"`. The migration adds 3 nullable float columns — safe online with no downtime.

### A4. Update `_upsert_rolling_stats()` in daily_ingestion.py

Wherever `PlayerRollingStats` is upserted (likely in `daily_ingestion.py` or `rolling_window_engine.py`), ensure the new fields (`w_runs`, `w_tb`, `w_qs`) are included in the upsert dict.

### A5. Expand scoring_engine.py Z-score dictionaries

**File:** `backend/services/scoring_engine.py`

Expand `HITTER_CATEGORIES` (line 31):
```python
HITTER_CATEGORIES: dict[str, tuple[str, bool]] = {
    "z_hr":   ("w_home_runs",       False),
    "z_rbi":  ("w_rbi",             False),
    "z_sb":   ("w_stolen_bases",    False),      # legacy — excluded from composite
    "z_nsb":  ("w_net_stolen_bases", False),      # P27 canonical basestealing
    "z_avg":  ("w_avg",             False),
    "z_obp":  ("w_obp",             False),
    # Phase 2 additions:
    "z_r":    ("w_runs",            False),       # R — runs scored
    "z_h":    ("w_hits",            False),       # H — hits
    "z_tb":   ("w_tb",              False),       # TB — total bases
    "z_k_b":  ("w_strikeouts_bat",  True),        # K_B — batter K (lower is better)
    "z_ops":  ("w_ops",             False),       # OPS — on-base + slugging
}
```

Expand `PITCHER_CATEGORIES` (line 40):
```python
PITCHER_CATEGORIES: dict[str, tuple[str, bool]] = {
    "z_era":     ("w_era",           True),       # lower ERA is better
    "z_whip":    ("w_whip",          True),       # lower WHIP is better
    "z_k_per_9": ("w_k_per_9",      False),
    # Phase 2 additions:
    "z_k_p":     ("w_strikeouts_pit", False),     # K_P — pitcher strikeouts (higher is better)
    "z_qs":      ("w_qs",            False),       # QS — quality starts
}
```

**Note on 4 greenfield categories:** `W`, `L`, `HR_P`, `NSV` are NOT added to these dicts yet because there is no upstream data. They will be `None` in the `CategoryStats` output. Add a comment:
```python
# Phase 2b TODO: Add z_w, z_l, z_hr_p, z_nsv when upstream data (Yahoo/MLB Stats API) is available.
# These 4 categories require new ingestion pipeline — see K1 report.
```

### A6. Expand `_Z_TO_SHORT` mapping

```python
_Z_TO_SHORT = {
    "z_hr": "hr", "z_rbi": "rbi", "z_sb": "sb", "z_nsb": "nsb",
    "z_avg": "avg", "z_obp": "obp",
    "z_era": "era", "z_whip": "whip", "z_k_per_9": "k_9",
    # Phase 2:
    "z_r": "r", "z_h": "h", "z_tb": "tb", "z_k_b": "k_b", "z_ops": "ops",
    "z_k_p": "k_p", "z_qs": "qs",
}
```

### A7. Expand `PlayerScoreResult` dataclass + `PlayerScore` ORM

**File:** `backend/services/scoring_engine.py` — Add fields to `PlayerScoreResult`:
```python
# Phase 2 additions
z_r:       Optional[float] = None
z_h:       Optional[float] = None
z_tb:      Optional[float] = None
z_k_b:     Optional[float] = None
z_ops:     Optional[float] = None
z_k_p:     Optional[float] = None
z_qs:      Optional[float] = None
```

**File:** `backend/models.py` — Add columns to `PlayerScore`:
```python
# Phase 2 additions
z_r         = Column(Float, nullable=True)
z_h         = Column(Float, nullable=True)
z_tb        = Column(Float, nullable=True)
z_k_b       = Column(Float, nullable=True)
z_ops       = Column(Float, nullable=True)
z_k_p       = Column(Float, nullable=True)
z_qs        = Column(Float, nullable=True)
```

**Alembic migration:** Include these columns in the same migration as A3.

### A8. Tests for Workstream A

Write `tests/test_rolling_stats_expansion.py`:

```python
# Test 1: w_runs accumulates correctly with decay weights
# Test 2: w_tb = H + 2B + 2×3B + 3×HR with decay
# Test 3: w_qs = 1.0 only when IP >= 6.0 AND ER <= 3
# Test 4: w_qs = 0.0 when IP = 5.2 (not a QS even with 0 ER)
# Test 5: w_qs = 0.0 when IP = 7.0 but ER = 4
# Test 6: New Z-scores (z_r, z_h, z_tb, z_k_b, z_ops, z_k_p, z_qs) are computed
# Test 7: z_k_b is negated (lower is better)
# Test 8: Composite Z includes new categories
# Test 9: _Z_TO_SHORT maps all new keys
```

Target: **15+ tests**, all passing.

---

## 5. WORKSTREAM B: ROW PROJECTION PIPELINE

### B1. Create `backend/fantasy_baseball/row_projector.py`

This is a **new file** — the greenfield ROW projection function. It is a pure function with no database access.

**Function signature:**
```python
from __future__ import annotations
from typing import Dict, List, Optional
from backend.stat_contract import SCORING_CATEGORY_CODES, LOWER_IS_BETTER, BATTING_CODES, PITCHING_CODES


def compute_row_projections(
    player_stats: List[PlayerStatInput],
    games_remaining: Dict[str, int],
    current_matchup_totals: Optional[MatchupTotals] = None,
) -> ROWProjectionResult:
    """
    Compute team-level Rest-of-Week projections for all 18 scoring categories.

    Parameters
    ----------
    player_stats : Per-player rolling + season rates for active roster.
    games_remaining : player_key -> expected remaining games this matchup week.
    current_matchup_totals : Current accumulated matchup values (from Yahoo scoreboard).
                             If provided, projected finals = current + ROW increment.

    Returns
    -------
    ROWProjectionResult containing:
      - row_increment: Dict[str, float] — projected additional stats for rest of week
      - projected_final: Dict[str, Optional[float]] — current + increment (None if no current provided)
      - per_player: Dict[str, Dict[str, float]] — individual player ROW contributions
    """
```

**Input dataclasses:**
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class PlayerStatInput:
    """Inputs for one player's ROW projection."""
    player_key: str
    is_pitcher: bool

    # Rolling 14-day rates (from player_rolling_stats — preferred)
    rolling_14d: Dict[str, Optional[float]]  # canonical_code -> value
    games_in_window: int

    # Season totals (for blended rate — stabilizes early-season)
    season_totals: Optional[Dict[str, float]] = None  # canonical_code -> season total
    season_games: Optional[int] = None

    # Supporting stats needed for ratio projections
    rolling_ab: Optional[float] = None
    rolling_ip: Optional[float] = None
    rolling_hits_allowed: Optional[float] = None
    rolling_walks_allowed: Optional[float] = None
    rolling_earned_runs: Optional[float] = None
    rolling_walks: Optional[float] = None  # batter BB
    rolling_tb: Optional[float] = None


@dataclass(frozen=True)
class MatchupTotals:
    """Current accumulated matchup values from Yahoo scoreboard."""
    my_totals: Dict[str, float]        # canonical_code -> accumulated value
    opp_totals: Dict[str, float]       # canonical_code -> accumulated value
    # Supporting stats for ratio computation
    my_ab: float = 0.0
    my_ip: float = 0.0
    my_er: float = 0.0
    my_hits: float = 0.0
    my_bb: float = 0.0
    my_tb: float = 0.0
    my_hits_allowed: float = 0.0
    my_walks_allowed: float = 0.0
    my_k_p: float = 0.0
    opp_ab: float = 0.0
    opp_ip: float = 0.0


@dataclass(frozen=True)
class ROWProjectionResult:
    """Output of ROW projection computation."""
    row_increment: Dict[str, float]
    projected_final: Dict[str, Optional[float]]
    per_player: Dict[str, Dict[str, float]]
```

### B2. Counting stat projection — blended daily rate

For each counting stat category:
```python
COUNTING_STATS = {"R", "H", "HR_B", "RBI", "K_B", "TB", "NSB", "W", "L", "HR_P", "K_P", "QS", "NSV"}

# Per-player daily rate:
rolling_rate = (rolling_14d_value / games_in_window) if games_in_window > 0 else 0.0
season_rate  = (season_total / season_games)          if season_games and season_games > 0 else 0.0

# Blend: 60% rolling (recency), 40% season (stability)
if games_in_window >= 7:
    daily_rate = 0.6 * rolling_rate + 0.4 * season_rate
elif games_in_window >= 3:
    daily_rate = 0.4 * rolling_rate + 0.6 * season_rate  # trust season more with thin rolling data
else:
    daily_rate = season_rate  # rolling window too thin to trust

player_row_value = daily_rate * games_remaining[player_key]
```

**Sum across roster for team total.**

### B3. Ratio stat projection — weighted aggregation (CRITICAL)

Ratio stats **MUST NOT** be averaged across players. They require numerator/denominator aggregation.

**AVG:**
```python
# Per-player: project remaining H and AB separately
row_h  = sum(daily_rate_h(player) × games_remaining[player] for player in hitters)
row_ab = sum(daily_rate_ab(player) × games_remaining[player] for player in hitters)

# Projected final (if current matchup totals provided):
proj_final_avg = (current_H + row_h) / (current_AB + row_ab)  # guard div-by-zero
```

**OPS:**
```python
row_obp_num = sum(daily_rate_h(p) + daily_rate_bb(p)) × games_remaining[p] for p in hitters)
row_obp_den = sum((daily_rate_ab(p) + daily_rate_bb(p)) × games_remaining[p] for p in hitters)
row_slg_num = sum(daily_rate_tb(p) × games_remaining[p] for p in hitters)
row_slg_den = row_ab  # same as AVG denominator

proj_obp = (current_OBP_num + row_obp_num) / (current_OBP_den + row_obp_den)
proj_slg = (current_SLG_num + row_slg_num) / (current_SLG_den + row_slg_den)
proj_final_ops = proj_obp + proj_slg
```

**ERA:**
```python
row_er = sum(daily_rate_er(p) × games_remaining[p] for p in pitchers)
row_ip = sum(daily_rate_ip(p) × games_remaining[p] for p in pitchers)

proj_final_era = 9.0 * (current_ER + row_er) / (current_IP + row_ip)  # guard div-by-zero
```

**WHIP:**
```python
row_h_allowed = sum(daily_rate_h_allowed(p) × games_remaining[p] for p in pitchers)
row_bb_allowed = sum(daily_rate_bb_allowed(p) × games_remaining[p] for p in pitchers)
row_ip_whip = row_ip  # same IP as ERA

proj_final_whip = (current_H_all + row_h_allowed + current_BB_all + row_bb_allowed) / (current_IP + row_ip_whip)
```

**K/9:**
```python
row_k_p = sum(daily_rate_k_p(p) × games_remaining[p] for p in pitchers)
row_ip_outs = row_ip * 3.0  # convert IP to outs

proj_final_k_9 = 27.0 * (current_K_P + row_k_p) / (current_IP_outs + row_ip_outs)
```

**All division operations must guard against zero denominators.** Return `None` if denominator is 0.

### B4. Greenfield categories (W, L, HR_P, NSV)

For the 4 categories with no upstream data:
```python
# These categories have no data in mlb_player_stats.
# Return 0.0 in row_increment (no data to project) and None in projected_final.
# Phase 2b will add upstream ingestion from Yahoo season stats.
for code in ("W", "L", "HR_P", "NSV"):
    row_increment[code] = 0.0
    projected_final[code] = None  # cannot project without data
```

### B5. Tests for Workstream B

Write `tests/test_row_projector.py`:

```python
# Test 1: Pure counting stat (R) — blended rate × games_remaining
# Test 2: Ratio stat (AVG) — weighted aggregation, not average of player AVGs
# Test 3: Ratio stat (ERA) — 9 × ER / IP, team-level
# Test 4: Ratio stat (WHIP) — (H_all + BB_all) / IP
# Test 5: Ratio stat (K_9) — 27 × K_P / IP_outs
# Test 6: OPS — composite OBP + SLG from components
# Test 7: Zero games remaining → all zeros
# Test 8: Division by zero guard (0 AB, 0 IP → None for ratio stats)
# Test 9: Blending weights: ≥7 games → 60/40, 3-6 games → 40/60, <3 games → season only
# Test 10: Greenfield categories (W, L, HR_P, NSV) return 0.0 increment, None final
# Test 11: projected_final = current + row_increment for counting stats
# Test 12: projected_final uses component aggregation for ratio stats
# Test 13: per_player dict has correct keys and values
# Test 14: Lower-is-better categories (K_B) still project as positive sums (margin handles sign)
```

Target: **15+ tests**, all passing.

---

## 6. WORKSTREAM C: CATEGORY MATH MODULE

### C1. Create `backend/fantasy_baseball/category_math.py`

**Pure functions, no DB access.**

```python
from backend.stat_contract import SCORING_CATEGORY_CODES, LOWER_IS_BETTER


COUNTING_CATEGORIES = SCORING_CATEGORY_CODES - {"AVG", "OPS", "ERA", "WHIP", "K_9"}
RATIO_CATEGORIES = {"AVG", "OPS", "ERA", "WHIP", "K_9"}


def compute_margin(
    my_value: float,
    opp_value: float,
    canonical_code: str,
) -> float:
    """
    Compute margin for a single category.

    Returns positive when 'I am winning'. For lower-is-better categories,
    the subtraction is reversed so the sign convention is consistent.

    >>> compute_margin(3.50, 4.00, "ERA")  # I have lower ERA → I'm winning
    0.5
    >>> compute_margin(10, 8, "R")  # I have more R → I'm winning
    2
    >>> compute_margin(5, 3, "K_B")  # I have more K_B → I'm losing (lower is better)
    -2
    """
    if canonical_code in LOWER_IS_BETTER:
        return opp_value - my_value
    return my_value - opp_value


def compute_delta_to_flip(
    my_value: float,
    opp_value: float,
    canonical_code: str,
    *,
    my_denominator: Optional[float] = None,
) -> Optional[float]:
    """
    Compute the change needed to flip who is winning this category.

    For counting stats: simple integer difference.
    For ratio stats: requires denominator to solve for numerator change.

    Returns None if the category is already won (no flip needed),
    or if it's a ratio stat without a denominator.

    Positive return = "need this many more units to flip."
    """
```

### C2. Category classification

```python
from enum import Enum
from backend.contracts import CategoryStatusTag

def classify_category(
    my_proj_final: float,
    opp_proj_final: float,
    canonical_code: str,
) -> CategoryStatusTag:
    """
    Classify a category into one of 5 status tags.

    Returns one of: WINNING, CLOSE_LEAD, TIED, CLOSE_TRAIL, LOSING
    
    "Close" threshold: within 10% of opponent value for ratio stats,
    within 2 units for counting stats.
    """
```

### C3. Tests for Workstream C

Write `tests/test_category_math.py`:

```python
# Test 1: compute_margin — higher-is-better, winning → positive
# Test 2: compute_margin — higher-is-better, losing → negative
# Test 3: compute_margin — lower-is-better (ERA), winning → positive
# Test 4: compute_margin — lower-is-better (K_B), losing → negative
# Test 5: compute_delta_to_flip — counting stat, losing by 3 → delta = 4
# Test 6: compute_delta_to_flip — already winning → None
# Test 7: compute_delta_to_flip — ratio stat (ERA) with denominator
# Test 8: compute_delta_to_flip — ratio stat without denominator → None
# Test 9: classify_category — clear winner → WINNING
# Test 10: classify_category — close lead → CLOSE_LEAD
# Test 11: classify_category — tied → TIED
# Test 12: classify_category — all 5 LOWER_IS_BETTER codes tested
```

Target: **12+ tests**, all passing.

---

## 7. WORKSTREAM D: WIRING + INTEGRATION

### D1. Ensure `CanonicalPlayerRow.row_projection` can be populated

`CanonicalPlayerRow.row_projection` is typed as `Optional[CategoryStats]`. `CategoryStats` is a `Dict[str, Optional[float]]` validated against the 18 `SCORING_CATEGORY_CODES`.

The ROW projector's `per_player` output must be convertible to `CategoryStats`. For the 4 greenfield categories, set values to `None`:
```python
row_projection = CategoryStats(
    values={code: per_player.get(player_key, {}).get(code) for code in SCORING_CATEGORY_CODES}
)
```

### D2. Export public API from `row_projector.py` and `category_math.py`

Ensure both modules have clean `__all__` exports:
```python
# row_projector.py
__all__ = ["compute_row_projections", "PlayerStatInput", "MatchupTotals", "ROWProjectionResult"]

# category_math.py
__all__ = ["compute_margin", "compute_delta_to_flip", "classify_category",
           "COUNTING_CATEGORIES", "RATIO_CATEGORIES"]
```

### D3. Integration test

Write `tests/test_phase2_integration.py`:

```python
# Test 1: End-to-end: rolling_window_result → scoring_engine → row_projector → category_math
# Test 2: 15 Z-scores are non-None for a player with full data
# Test 3: 4 greenfield categories (W, L, HR_P, NSV) are None in row_projection
# Test 4: Ratio stat projected finals match manual calculation
# Test 5: FreshnessMetadata propagation — row_projection has correct as_of timestamp
```

Target: **5+ tests**.

---

## 8. V2 CANONICAL CODES (LOCKED — DO NOT MODIFY)

**Batting (9):** R, H, HR_B, RBI, K_B, TB, AVG, OPS, NSB
**Pitching (9):** W, L, HR_P, K_P, ERA, WHIP, K_9, QS, NSV
**Lower-is-better (5):** ERA, WHIP, L, K_B, HR_P

---

## 9. WHAT NOT TO DO

1. **Do NOT average player ratios for team totals.** `team_AVG ≠ mean(player_AVGs)`. Use `sum(H)/sum(AB)`.
2. **Do NOT modify `h2h_monte_carlo.py` in this phase.** That's Phase 3. Leave its v1 codes (`K`, `K/9`) untouched.
3. **Do NOT ingest new data from Yahoo for W/L/HR_P/NSV.** That's a separate ingestion task. Return `None`/`0.0` placeholders.
4. **Do NOT modify the stat_contract package.** It is immutable. Consume it.
5. **Do NOT modify existing Z-scores.** Only add new ones. Existing `z_hr`, `z_rbi`, `z_nsb`, `z_avg`, `z_obp`, `z_era`, `z_whip`, `z_k_per_9` must remain unchanged.
6. **Do NOT use `datetime.utcnow()`.** Always use `datetime.now(ZoneInfo("America/New_York"))`.
7. **Do NOT put test files outside `tests/`.**
8. **Do NOT use the wrong sign convention.** `margin > 0` always means "I am winning." For lower-is-better categories, reverse the subtraction.

---

## 10. GATE CRITERIA — Phase 2 Complete When:

| # | Criterion | How to verify |
|---|-----------|---------------|
| G1 | 15 Z-score categories computed (9 existing + 6 new) | `pytest tests/test_rolling_stats_expansion.py` passes |
| G2 | `w_runs`, `w_tb`, `w_qs` stored in `PlayerRollingStats` ORM | Alembic migration generated, `py_compile` passes |
| G3 | `row_projector.py` produces correct team-level projections | `pytest tests/test_row_projector.py` — ratio stat tests are critical |
| G4 | Ratio stats use weighted aggregation (not average of ratios) | AVG test: 2 players with .300/.200 AVG, different AB → team AVG ≠ .250 |
| G5 | `category_math.py` margin sign convention correct for all 5 lower-is-better categories | `pytest tests/test_category_math.py` |
| G6 | Greenfield categories (W, L, HR_P, NSV) return `None`/`0.0` gracefully | Integration test |
| G7 | All existing 2029 tests still pass | `venv/Scripts/python -m pytest tests/ -q --tb=short` |
| G8 | All new files compile | `venv/Scripts/python -m py_compile backend/fantasy_baseball/row_projector.py` etc. |

**Gate phrase:** "Phase 2 gate passed: 15 Z-scores computed, ROW projections stable for full matchup week, all tests passing."

---

## 11. AFTER GATE PASSES

1. Update `HANDOFF.md`: Phase 2 → COMPLETE, Phase 3 → NEXT
2. Update Layer 3 status to reflect 15/18 categories covered
3. Note the 4 greenfield categories (W, L, HR_P, NSV) as Phase 2b backlog items
4. Phase 3 scope: Wire `h2h_monte_carlo.py` to v2 codes, connect ROW projections to simulation, build remaining L1 pure functions (ratio risk, IP pace classifier, etc.)

---

## 12. EXECUTION ORDER

```
A1-A3  Add w_runs, w_tb, w_qs to RollingWindowResult + ORM + migration
A4     Update upsert logic
A5-A6  Expand scoring_engine dictionaries
A7     Expand PlayerScoreResult + PlayerScore ORM
A8     Write + run Workstream A tests
---checkpoint: 15 Z-scores computed, existing tests still pass---
B1-B4  Build row_projector.py with all projection logic
B5     Write + run Workstream B tests
---checkpoint: ROW projections stable---
C1-C2  Build category_math.py
C3     Write + run Workstream C tests
---checkpoint: category math verified---
D1-D3  Wire contracts + integration tests
---GATE: all criteria met---
```

---

## 13. FILE MANIFEST (New + Modified)

| File | Action | Workstream |
|------|--------|-----------|
| `backend/services/rolling_window_engine.py` | MODIFY — add `w_runs`, `w_tb`, `w_qs` fields + accumulation | A |
| `backend/models.py` | MODIFY — add 3 cols to `PlayerRollingStats`, 7 cols to `PlayerScore` | A |
| `backend/services/scoring_engine.py` | MODIFY — expand category dicts, `PlayerScoreResult`, `_Z_TO_SHORT` | A |
| `backend/services/daily_ingestion.py` | MODIFY — include new fields in upsert | A |
| `backend/fantasy_baseball/row_projector.py` | CREATE — ROW projection pipeline | B |
| `backend/fantasy_baseball/category_math.py` | CREATE — margin, delta-to-flip, classification | C |
| `tests/test_rolling_stats_expansion.py` | CREATE — 15+ tests | A |
| `tests/test_row_projector.py` | CREATE — 15+ tests | B |
| `tests/test_category_math.py` | CREATE — 12+ tests | C |
| `tests/test_phase2_integration.py` | CREATE — 5+ tests | D |
| `alembic/versions/xxx_phase2_rolling_stats_expansion.py` | CREATE — migration | A |
