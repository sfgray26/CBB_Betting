# Phase 3 Implementation Prompt — Pure Functions + Engine Wiring

> **Self-contained prompt.** No prior conversation context assumed.
> **Prerequisite:** Phase 0 COMPLETE. Phase 1 COMPLETE. Phase 2 COMPLETE (15 Z-scores, ROW projector, category math — 96 tests). All 18 scoring categories flow through the pipeline.

---

## 1. READ FIRST

Before writing any code, read these files in order:

**Architecture & State:**
1. `HANDOFF.md` — current operational state (Phase 2 COMPLETE, Phase 3 NEXT)
2. `backend/stat_contract/__init__.py` — v2 contract singleton (`SCORING_CATEGORY_CODES`, `BATTING_CODES`, `PITCHING_CODES`, `LOWER_IS_BETTER`)

**Phase 2 Deliverables (your inputs):**
3. `backend/services/row_projector.py` — `ROWProjectionResult` dataclass + `compute_row_projection()` + `compute_row_projection_from_canonical_rows()` — produces team-level projections for all 18 categories
4. `backend/services/category_math.py` — `CategoryMathResult` dataclass + `compute_category_math()` + `compute_all_category_math()` — margin, delta-to-flip, classification
5. `backend/contracts.py` — `MatchupScoreboardRow`, `CategoryMathResult`, `CategoryMathSummary`, `CanonicalPlayerRow` (with `rolling_14d` field)

**Phase 3 Targets (files to modify):**
6. `backend/fantasy_baseball/h2h_monte_carlo.py` — H2H One Win Monte Carlo simulator. Uses v1 category codes. **Must be updated to v2.**
7. `backend/fantasy_baseball/mcmc_simulator.py` — MCMC weekly matchup simulator. Uses non-canonical z-score keys. **Must be aligned.**

**Reference (context only — do not modify):**
8. `backend/services/scoring_engine.py` — `HITTER_CATEGORIES`, `PITCHER_CATEGORIES`, `_Z_TO_SHORT` (already v2-aligned in Phase 2)
9. `backend/services/constraint_helpers.py` — Phase 1 pure functions (IP pace, acquisition count, games remaining, etc.)
10. `reports/2026-04-18-category-math-reference.md` — K4: margin sign convention and delta-to-flip formulas
11. `reports/2026-04-18-row-projection-spec.md` — K2: ratio stat aggregation formulas

---

## 2. MISSION

Phase 3 wires the **simulation layer** (Layer 4) to the v2 canonical codes and the ROW projection pipeline delivered in Phase 2. It also builds 2 remaining L1 pure functions.

**Gate criterion:** H2H Monte Carlo with projected finals produces non-degenerate results.

| Workstream | Summary | Depends On |
|------------|---------|-----------|
| **A** | Align `h2h_monte_carlo.py` to v2 18-category contract | Nothing |
| **B** | Align `mcmc_simulator.py` to v2 canonical keys | Nothing |
| **C** | Build ROW→Simulation bridge adapter | Phase 2 deliverables |
| **D** | Build remaining L1 pure functions (ratio risk, category-count delta) | Phase 2 category_math |
| **E** | Integration tests — simulation with projected data | A + B + C |

**Out of scope for Phase 3:**
- Building API endpoints (Phase 4)
- Frontend work (Phase 5+)
- Lineup solver full-roster extension (batting+pitching — deferred)
- Yahoo API augmentation for greenfield pitching stats (W, L, HR_P, NSV)

---

## 3. GROUND TRUTH: CURRENT STATE OF SIMULATION ENGINES

### H2H Monte Carlo (`h2h_monte_carlo.py`) — V1 Codes

**Current categories (WRONG — v1):**
```python
HITTING_CATS = ["R", "HR", "RBI", "SB", "NSB", "AVG", "OPS"]
PITCHING_CATS = ["W", "QS", "K", "K/9", "ERA", "WHIP"]
```

**Problems:**
| Current | Should Be | Issue |
|---------|-----------|-------|
| `"SB"` | remove | Not a scoring category — `NSB` (already present) is canonical |
| `"K"` | `"K_P"` | v1 pitching strikeout code |
| `"K/9"` | `"K_9"` | v1 strikeout rate code |
| missing | `"H"` | Hits — v2 scoring category, not in simulator |
| missing | `"K_B"` | Batter strikeouts — v2 scoring category, not in simulator |
| missing | `"TB"` | Total bases — v2 scoring category, not in simulator |
| missing | `"L"` | Losses — v2 scoring category, not in simulator |
| missing | `"HR_P"` | HR allowed — v2 scoring category, not in simulator |
| missing | `"NSV"` | Net saves — v2 scoring category, not in simulator |

**Current lower-is-better handling:** Only ERA and WHIP are treated as lower-is-better in `_run_simulation()`. K_B, L, and HR_P are missing entirely.

**`STAT_CV` dict:** Uses v1 keys (`"K"`, `"K/9"`). Must be updated to v2 (`"K_P"`, `"K_9"`). New categories need CV values.

### MCMC Simulator (`mcmc_simulator.py`) — Non-Canonical Keys

**Current category keys (from `_PLAYER_WEEKLY_STD`):**
```python
# Batting
"hr", "r", "rbi", "nsb", "h", "tb", "avg", "ops"
# Pitching
"k_pit", "w", "nsv", "qs", "k9", "era", "whip"
```

**Problems:**
| Current | Should Be | Issue |
|---------|-----------|-------|
| `"k_pit"` | `"k_p"` | Non-canonical key for pitcher strikeouts |
| `"k9"` | `"k_9"` | Non-canonical key for K/9 (missing underscore) |
| missing | `"k_b"` | Batter strikeouts — not in simulator |
| missing | `"l"` | Losses — not in simulator |
| missing | `"hr_p"` | HR allowed — not in simulator |

**MCMC uses z-scores (all higher=better).** ERA and WHIP z-scores are already inverted by the scoring engine, so the MCMC's approach of "all z-scores higher is better" is correct — it just needs canonical keys.

**Input contract:** Each player dict needs `cat_scores: dict[str, float]` (z-score per category). The MCMC sums z-scores across roster and compares.

---

## 4. WORKSTREAM A: H2H MONTE CARLO V2 ALIGNMENT

### A1. Update category lists to v2 canonical codes

**File:** `backend/fantasy_baseball/h2h_monte_carlo.py`

Replace the category constants:
```python
# V2 canonical scoring categories (18-cat H2H One Win format)
# Batting (9): higher is better except K_B
HITTING_CATS = ["R", "H", "HR_B", "RBI", "K_B", "TB", "AVG", "OPS", "NSB"]

# Pitching (9): higher is better except ERA, WHIP, L, HR_P
PITCHING_CATS = ["W", "L", "HR_P", "K_P", "ERA", "WHIP", "K_9", "QS", "NSV"]

# Categories where lower is better (margin sign reversal + simulation comparison reversal)
LOWER_IS_BETTER = {"ERA", "WHIP", "K_B", "L", "HR_P"}
```

**Important:** The current code hardcodes `if cat in ["ERA", "WHIP"]` in `_run_simulation()`. This must be replaced with `if cat in self.LOWER_IS_BETTER` so all 5 lower-is-better categories are handled correctly.

### A2. Update `STAT_CV` dict

Replace with v2 keys and add new categories:
```python
STAT_CV = {
    # Batting counting stats
    "R": 0.35,
    "H": 0.30,          # NEW — hits are relatively stable
    "HR_B": 0.40,       # was "HR"
    "RBI": 0.35,
    "K_B": 0.30,        # NEW — batter strikeouts
    "TB": 0.35,         # NEW — total bases
    "NSB": 0.50,
    # Batting rate stats
    "AVG": 0.08,
    "OPS": 0.10,
    # Pitching counting stats
    "W": 0.30,
    "L": 0.30,          # NEW — losses (similar volatility to wins)
    "HR_P": 0.40,       # NEW — HR allowed
    "K_P": 0.25,        # was "K"
    "QS": 0.25,
    "NSV": 0.50,        # NEW — net saves (very volatile)
    # Pitching rate stats
    "ERA": 0.15,
    "WHIP": 0.12,
    "K_9": 0.12,        # was "K/9"
}
```

### A3. Update `_run_simulation()` lower-is-better logic

**Current (WRONG):**
```python
if cat in ["ERA", "WHIP"]:
    category_win_matrix[:, i] = (my_samples < opp_samples).astype(float)
else:
    category_win_matrix[:, i] = (my_samples > opp_samples).astype(float)
```

**Replace with:**
```python
if cat in self.LOWER_IS_BETTER:
    # Lower is better: I win when my value < opponent's value
    category_win_matrix[:, i] = (my_samples < opp_samples).astype(float)
else:
    # Higher is better: I win when my value > opponent's value
    category_win_matrix[:, i] = (my_samples > opp_samples).astype(float)
```

### A4. Update `_aggregate_roster()` — ratio stats need weighted aggregation

**Current problem:** `_aggregate_roster()` sums ALL stats across players, including ratio stats (AVG, OPS, ERA, WHIP, K_9). Summing ratio stats is incorrect — team AVG ≠ sum of player AVGs.

**Two approaches (choose the simpler one):**

**Approach 1 (Recommended — minimal change):** Since `simulate_week()` receives **team-level projected values** (not per-player), the aggregation is already done by the ROW projector. The Monte Carlo should accept pre-aggregated team projections directly.

Add an alternative entry point that accepts pre-aggregated projections:
```python
def simulate_week_from_projections(
    self,
    my_proj: Dict[str, float],
    opp_proj: Dict[str, float],
    n_sims: int = 10000,
    as_of_date: date = None,
) -> H2HWinResult:
    """
    Run simulation from pre-aggregated team projections.

    Use this when projections come from ROWProjectionResult.to_dict()
    which already handles ratio-stat weighted aggregation correctly.
    """
    if as_of_date is None:
        as_of_date = date.today()

    categories_won, category_win_matrix = self._run_simulation(my_proj, opp_proj, n_sims)

    win_prob = np.mean(categories_won >= len(self.HITTING_CATS + self.PITCHING_CATS) // 2 + 1)
    locked, swing, vulnerable = self._classify_categories(category_win_matrix)

    return H2HWinResult(
        win_probability=float(win_prob),
        locked_categories=locked,
        swing_categories=swing,
        vulnerable_categories=vulnerable,
        category_win_probs=self._compute_category_probs(category_win_matrix),
        mean_categories_won=float(np.mean(categories_won)),
        std_categories_won=float(np.std(categories_won)),
        n_simulations=n_sims,
        as_of_date=as_of_date,
    )
```

**Keep the existing `simulate_week()` for backward compatibility** but add a deprecation comment noting that ratio stats in the per-player dicts are summed naively.

### A5. Update win threshold

The current code uses `categories_won >= 6` (assumes 10 categories, need 6+ to win). With 18 categories, the win threshold is `>= 10` (majority of 18).

```python
# Win threshold: need majority of categories
_WIN_THRESHOLD = len(HITTING_CATS + PITCHING_CATS) // 2 + 1  # 10 for 18 categories
```

In `simulate_week()`, replace:
```python
win_prob = np.mean(categories_won >= 6)
```
with:
```python
win_prob = np.mean(categories_won >= self._WIN_THRESHOLD)
```

Or compute dynamically:
```python
n_cats = len(self.HITTING_CATS + self.PITCHING_CATS)
win_threshold = n_cats // 2 + 1  # majority
win_prob = np.mean(categories_won >= win_threshold)
```

### A6. Update class docstring

Update the docstring from "Categories (10 total)" to "Categories (18 total)" and list all v2 canonical codes.

---

## 5. WORKSTREAM B: MCMC SIMULATOR V2 ALIGNMENT

### B1. Update `_PLAYER_WEEKLY_STD` keys to canonical lowercase

**File:** `backend/fantasy_baseball/mcmc_simulator.py`

Replace:
```python
_PLAYER_WEEKLY_STD: dict[str, float] = {
    # Batting — counting
    "hr": 0.65,       # → "hr_b" (canonical)
    "r": 0.70,
    "rbi": 0.70,
    "nsb": 0.90,
    "h": 0.55,
    "tb": 0.65,
    # Batting — rate
    "avg": 0.40,
    "ops": 0.40,
    # Pitching — counting
    "k_pit": 0.75,    # → "k_p" (canonical)
    "w": 0.85,
    "nsv": 1.00,
    "qs": 0.80,
    "k9": 0.40,       # → "k_9" (canonical)
    # Pitching — rate
    "era": 0.65,
    "whip": 0.55,
}
```

With canonical keys:
```python
_PLAYER_WEEKLY_STD: dict[str, float] = {
    # Batting — counting (canonical v2 codes, lowercase)
    "r": 0.70,
    "h": 0.55,
    "hr_b": 0.65,
    "rbi": 0.70,
    "k_b": 0.60,       # NEW — batter K (volatile week-to-week)
    "tb": 0.65,
    "nsb": 0.90,
    # Batting — rate
    "avg": 0.40,
    "ops": 0.40,
    # Pitching — counting
    "w": 0.85,
    "l": 0.85,          # NEW — losses (same volatility as wins)
    "hr_p": 0.70,       # NEW — HR allowed
    "k_p": 0.75,        # was "k_pit"
    "qs": 0.80,
    "nsv": 1.00,
    # Pitching — rate
    "era": 0.65,
    "whip": 0.55,
    "k_9": 0.40,        # was "k9"
}
```

### B2. Update `_detect_categories()` and `_roster_means_stds()`

These functions auto-detect categories from player `cat_scores` dicts. They should recognize the new canonical keys. If they use `_PLAYER_WEEKLY_STD.keys()` as the valid key set, the update in B1 is sufficient.

Verify that any key-name filtering or mapping inside these functions uses the updated canonical keys.

### B3. Update module docstring

Update the "Category keys" section:
```python
Category keys (canonical v2 codes, lowercase):
  Batting: r, h, hr_b, rbi, k_b, tb, avg, ops, nsb
  Pitching: w, l, hr_p, k_p, era, whip, k_9, qs, nsv
```

### B4. Ensure lower-is-better z-scores are already inverted

The MCMC simulator's approach is correct: all z-scores should be higher=better. The scoring engine already negates Z for lower-is-better categories (ERA, WHIP, K_B, L, HR_P). **No change needed in the MCMC comparison logic** — just ensure the input z-scores use canonical keys.

---

## 6. WORKSTREAM C: ROW→SIMULATION BRIDGE

### C1. Create `backend/services/simulation_bridge.py`

This adapter converts between the ROW projector's output and the simulation engines' expected inputs.

```python
"""
Simulation Bridge — adapts ROW projections to simulation engine input formats.

Bridges the gap between:
  - ROWProjectionResult (canonical codes, team-level) → H2H Monte Carlo (per-category dict)
  - Player Z-scores (canonical codes) → MCMC simulator (lowercase cat_scores dict)

Pure functions. No I/O.
"""
from typing import Dict, List, Optional

from backend.stat_contract import SCORING_CATEGORY_CODES, LOWER_IS_BETTER


def row_projection_to_monte_carlo(
    my_row: "ROWProjectionResult",
    opp_row: "ROWProjectionResult",
) -> tuple[Dict[str, float], Dict[str, float]]:
    """
    Convert two ROWProjectionResult objects into the dict format
    expected by H2HOneWinSimulator.simulate_week_from_projections().

    Returns (my_proj_dict, opp_proj_dict) keyed by canonical codes.
    """
    return my_row.to_dict(), opp_row.to_dict()


def player_scores_to_mcmc_roster(
    players: List[Dict],
    z_score_key_map: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """
    Convert player score dicts to MCMC roster format.

    The MCMC expects each player dict to have:
      - cat_scores: dict[str, float]  (lowercase canonical codes → z-scores)
      - positions: list[str]
      - starts_this_week: int (for pitchers)
      - name: str

    Parameters
    ----------
    players : list of dicts
        Each dict has z-score fields (z_r, z_h, z_hr, etc.) from PlayerScoreResult.
    z_score_key_map : dict, optional
        Mapping from z_key (e.g. "z_hr") to MCMC key (e.g. "hr_b").
        If None, uses the default _Z_TO_MCMC mapping.

    Returns
    -------
    List of player dicts with cat_scores in MCMC format.
    """
    if z_score_key_map is None:
        z_score_key_map = _Z_TO_MCMC

    result = []
    for p in players:
        cat_scores = {}
        for z_key, mcmc_key in z_score_key_map.items():
            val = p.get(z_key)
            if val is not None:
                cat_scores[mcmc_key] = val
        result.append({
            "name": p.get("name", p.get("player_name", "Unknown")),
            "positions": p.get("positions", p.get("eligible_positions", [])),
            "starts_this_week": p.get("starts_this_week", 1),
            "cat_scores": cat_scores,
        })
    return result


# Default mapping from scoring_engine z_key → MCMC canonical lowercase key
_Z_TO_MCMC: Dict[str, str] = {
    "z_r": "r",
    "z_h": "h",
    "z_hr": "hr_b",
    "z_rbi": "rbi",
    "z_nsb": "nsb",
    "z_k_b": "k_b",
    "z_tb": "tb",
    "z_avg": "avg",
    "z_obp": "obp",   # display stat, not scoring — include for completeness
    "z_ops": "ops",
    "z_era": "era",
    "z_whip": "whip",
    "z_k_per_9": "k_9",
    "z_k_p": "k_p",
    "z_qs": "qs",
    # z_w, z_l, z_hr_p, z_nsv → not yet computed (greenfield). Omitted.
}
```

### C2. Export `__all__`

```python
__all__ = [
    "row_projection_to_monte_carlo",
    "player_scores_to_mcmc_roster",
]
```

---

## 7. WORKSTREAM D: REMAINING L1 PURE FUNCTIONS

### D1. Ratio Risk Quantifier

**File:** `backend/services/category_math.py` (add to existing module)

```python
def compute_ratio_risk(
    my_ip: float,
    my_er: float,
    my_hits_allowed: float,
    my_bb_allowed: float,
    opp_era: float,
    opp_whip: float,
    remaining_ip: float,
) -> Dict[str, str]:
    """
    Quantify the risk that a single bad start could blow a ratio category.

    A "ratio blowup" occurs when adding a bad start's stats to the current
    accumulator would flip a winning category to losing.

    Parameters
    ----------
    my_ip : current accumulated IP this week
    my_er : current accumulated ER this week
    my_hits_allowed : current accumulated H allowed this week
    my_bb_allowed : current accumulated BB allowed this week
    opp_era : opponent's current or projected ERA
    opp_whip : opponent's current or projected WHIP
    remaining_ip : expected IP remaining this week

    Returns
    -------
    Dict with keys:
      "era_risk": "SAFE" | "AT_RISK" | "CRITICAL"
      "whip_risk": "SAFE" | "AT_RISK" | "CRITICAL"
      "era_cushion_er": float (ER allowed before flipping)
      "whip_cushion_baserunners": float (H+BB allowed before flipping)
    """
    result = {}

    # ERA risk: how many ER can I allow in remaining IP before my ERA exceeds opponent's?
    if my_ip > 0:
        total_ip = my_ip + remaining_ip
        # Max ER to stay below opponent's ERA: opp_ERA × total_IP_outs / 27
        ip_outs = total_ip * 3
        max_total_er = opp_era * ip_outs / 27.0
        er_cushion = max_total_er - my_er
        result["era_cushion_er"] = round(er_cushion, 1)

        if er_cushion > remaining_ip * 0.5:
            result["era_risk"] = "SAFE"
        elif er_cushion > 0:
            result["era_risk"] = "AT_RISK"
        else:
            result["era_risk"] = "CRITICAL"
    else:
        result["era_risk"] = "SAFE"
        result["era_cushion_er"] = 99.0

    # WHIP risk: how many H+BB can I allow in remaining IP before WHIP exceeds opponent's?
    if my_ip > 0:
        total_ip = my_ip + remaining_ip
        ip_outs = total_ip * 3
        max_total_baserunners = opp_whip * ip_outs / 3.0
        current_baserunners = my_hits_allowed + my_bb_allowed
        br_cushion = max_total_baserunners - current_baserunners
        result["whip_cushion_baserunners"] = round(br_cushion, 1)

        if br_cushion > remaining_ip * 1.5:
            result["whip_risk"] = "SAFE"
        elif br_cushion > 0:
            result["whip_risk"] = "AT_RISK"
        else:
            result["whip_risk"] = "CRITICAL"
    else:
        result["whip_risk"] = "SAFE"
        result["whip_cushion_baserunners"] = 99.0

    return result
```

### D2. Category-Count Delta Extractor

**File:** `backend/services/category_math.py` (add to existing module)

```python
def compute_category_count_delta(
    category_results: Dict[str, "CategoryMathResult"],
) -> Dict[str, int]:
    """
    Compute category win/loss/swing counts from a full set of CategoryMathResults.

    Returns
    -------
    Dict with keys:
      "winning": int — categories where margin > 0
      "losing": int — categories where margin < 0
      "tied": int — categories where margin == 0
      "swing": int — categories where |margin| < close_threshold
      "projected_result": str — "WIN", "LOSS", or "TOSS_UP"
    """
    winning = sum(1 for r in category_results.values() if r.is_winning)
    losing = sum(1 for r in category_results.values() if not r.is_winning and r.margin < 0)
    tied = sum(1 for r in category_results.values() if r.margin == 0.0)

    n_cats = len(category_results)
    win_threshold = n_cats // 2 + 1  # majority

    if winning >= win_threshold:
        projected_result = "WIN"
    elif losing >= win_threshold:
        projected_result = "LOSS"
    else:
        projected_result = "TOSS_UP"

    # Swing categories: close margin (within 10% for ratio, within 2 for counting)
    swing = 0
    for r in category_results.values():
        if abs(r.margin) < 2.0 and not r.is_winning:
            swing += 1
        elif abs(r.margin) < 0.1 and r.canonical_code in {"AVG", "OPS", "ERA", "WHIP", "K_9"}:
            swing += 1

    return {
        "winning": winning,
        "losing": losing,
        "tied": tied,
        "swing": swing,
        "projected_result": projected_result,
    }
```

---

## 8. WORKSTREAM E: TESTS

### E1. Tests for H2H Monte Carlo v2 alignment

Write `tests/test_h2h_monte_carlo_v2.py`:

```python
# Test 1: HITTING_CATS has exactly 9 entries matching BATTING_CODES
# Test 2: PITCHING_CATS has exactly 9 entries matching PITCHING_CODES
# Test 3: LOWER_IS_BETTER includes ERA, WHIP, K_B, L, HR_P (5 total)
# Test 4: STAT_CV has keys for all 18 categories
# Test 5: simulate_week_from_projections() returns H2HWinResult
# Test 6: win_probability is between 0.0 and 1.0
# Test 7: category_win_probs has all 18 category keys
# Test 8: Lower-is-better categories: team with lower ERA wins more often
# Test 9: K_B lower-is-better: team with fewer K_B wins K_B category
# Test 10: Win threshold = 10 (majority of 18)
# Test 11: Degenerate input (all zeros) → 50/50 result
# Test 12: Dominant team (all categories 2× opponent) → win_prob > 0.95
# Test 13: _aggregate_roster still works for backward compat (counting stats only)
# Test 14: locked/swing/vulnerable lists contain v2 canonical codes
```

Target: **14+ tests**, all passing.

### E2. Tests for MCMC simulator v2 alignment

Write `tests/test_mcmc_v2.py`:

```python
# Test 1: _PLAYER_WEEKLY_STD has canonical keys (hr_b not hr, k_p not k_pit, k_9 not k9)
# Test 2: simulate_weekly_matchup() accepts canonical cat_scores keys
# Test 3: simulate_roster_move() returns win_prob_gain
# Test 4: category_win_probs keys are canonical lowercase
# Test 5: Empty roster → 50/50 result
# Test 6: Dominant roster → win_prob > 0.80
# Test 7: Adding strong player increases win_prob
```

Target: **7+ tests**, all passing.

### E3. Tests for simulation bridge

Write `tests/test_simulation_bridge.py`:

```python
# Test 1: row_projection_to_monte_carlo returns two dicts with 18 keys each
# Test 2: player_scores_to_mcmc_roster maps z_hr → hr_b correctly
# Test 3: player_scores_to_mcmc_roster maps z_k_per_9 → k_9
# Test 4: Missing z-scores are omitted (not set to 0)
# Test 5: player dict format includes name, positions, cat_scores, starts_this_week
```

Target: **5+ tests**, all passing.

### E4. Tests for L1 pure functions

Write `tests/test_ratio_risk.py`:

```python
# Test 1: ERA SAFE — large cushion (my ERA 3.00, opp ERA 4.50, 20 IP remaining)
# Test 2: ERA AT_RISK — small cushion (my ERA 3.80, opp ERA 4.00, 10 IP remaining)
# Test 3: ERA CRITICAL — already losing (my ERA 4.50, opp ERA 3.50)
# Test 4: WHIP SAFE — large cushion
# Test 5: WHIP CRITICAL — already losing
# Test 6: Zero IP → SAFE (no data yet)
# Test 7: compute_category_count_delta — 10 winning → projected "WIN"
# Test 8: compute_category_count_delta — 10 losing → projected "LOSS"
# Test 9: compute_category_count_delta — 9-9 split → "TOSS_UP"
# Test 10: compute_category_count_delta — swing count detects close margins
```

Target: **10+ tests**, all passing.

### E5. End-to-end integration test

Write `tests/test_phase3_integration.py`:

```python
# Test 1: ROW projection → bridge → Monte Carlo → H2HWinResult (full pipeline)
# Test 2: Player scores → bridge → MCMC → win_prob (full pipeline)
# Test 3: Ratio risk correctly identifies when streaming a pitcher is dangerous
# Test 4: category_count_delta matches Monte Carlo category breakdown
# Test 5: Non-degenerate results: win_prob ≠ 0.0 and ≠ 1.0 for balanced teams
```

Target: **5+ tests**, all passing.

---

## 9. V2 CANONICAL CODES (LOCKED — DO NOT MODIFY)

**Batting (9):** R, H, HR_B, RBI, K_B, TB, AVG, OPS, NSB
**Pitching (9):** W, L, HR_P, K_P, ERA, WHIP, K_9, QS, NSV
**Lower-is-better (5):** ERA, WHIP, L, K_B, HR_P
**Win threshold:** 10 categories (majority of 18)

---

## 10. WHAT NOT TO DO

1. **Do NOT break backward compatibility gratuitously.** Keep `simulate_week()` working (add `simulate_week_from_projections()` alongside it). Keep `simulate_roster_move()` working with updated keys.
2. **Do NOT sum ratio stats across players.** The ROW projector already handles weighted aggregation. The Monte Carlo should accept pre-aggregated team projections.
3. **Do NOT change the scoring engine.** `scoring_engine.py` was updated in Phase 2 and is correct. Do not modify.
4. **Do NOT modify `row_projector.py` or `category_math.py`.** These are Phase 2 deliverables. Consume them, do not alter.
5. **Do NOT modify `backend/stat_contract/`.** It is immutable.
6. **Do NOT use `datetime.utcnow()`.** Always use `datetime.now(ZoneInfo("America/New_York"))`.
7. **Do NOT put test files outside `tests/`.**
8. **Do NOT build API endpoints.** That's Phase 4. Phase 3 is pure engine wiring.
9. **Do NOT import numpy at module level in bridge/pure-function files.** Only simulation engines use numpy.

---

## 11. GATE CRITERIA — Phase 3 Complete When:

| # | Criterion | How to verify |
|---|-----------|---------------|
| G1 | H2H Monte Carlo uses all 18 v2 canonical codes | `HITTING_CATS + PITCHING_CATS` has 18 entries, all matching `SCORING_CATEGORY_CODES` |
| G2 | Lower-is-better handled for all 5 categories (ERA, WHIP, K_B, L, HR_P) | Test: team with lower ERA wins ERA category more often |
| G3 | Win threshold = 10 (majority of 18), not hardcoded 6 | `win_prob = np.mean(categories_won >= 10)` or dynamic |
| G4 | MCMC simulator uses canonical keys (hr_b, k_p, k_9, k_b, l, hr_p) | `_PLAYER_WEEKLY_STD` keys match canonical lowercase codes |
| G5 | `simulate_week_from_projections()` accepts `ROWProjectionResult.to_dict()` output | Integration test passes |
| G6 | Simulation bridge converts between all formats | `tests/test_simulation_bridge.py` passes |
| G7 | Ratio risk quantifier produces SAFE/AT_RISK/CRITICAL | `tests/test_ratio_risk.py` passes |
| G8 | Non-degenerate Monte Carlo results with projected data | Win prob ∈ (0.05, 0.95) for balanced projected teams |
| G9 | All existing tests still pass | `venv/Scripts/python -m pytest tests/ -q --tb=short` |
| G10 | All new files compile | `venv/Scripts/python -m py_compile backend/services/simulation_bridge.py` etc. |

**Gate phrase:** "Phase 3 gate passed: H2H Monte Carlo produces non-degenerate results with 18-category v2 projections, all simulation engines aligned to canonical codes."

---

## 12. AFTER GATE PASSES

1. Update `HANDOFF.md`: Phase 3 → COMPLETE, Phase 4 → NEXT
2. Update Layer 4 status to COMPLETE
3. Phase 4 scope: Build P1 page APIs (scoreboard, budget, roster, optimize). All endpoints return complete data per Layer 0 contracts.

---

## 13. EXECUTION ORDER

```
A1-A3  Update h2h_monte_carlo.py categories + STAT_CV + lower-is-better
A4-A5  Add simulate_week_from_projections() + dynamic win threshold
A6     Update docstring
---checkpoint: Monte Carlo compiles, uses v2 codes---
B1-B3  Update mcmc_simulator.py keys + docstring
B4     Verify lower-is-better z-score inversion is correct
---checkpoint: MCMC compiles, uses canonical keys---
C1-C2  Build simulation_bridge.py
---checkpoint: bridge compiles---
D1-D2  Add ratio risk + category count delta to category_math.py
---checkpoint: pure functions compile---
E1-E5  Write + run all test suites
---GATE: all criteria met---
```

---

## 14. FILE MANIFEST (New + Modified)

| File | Action | Workstream |
|------|--------|-----------|
| `backend/fantasy_baseball/h2h_monte_carlo.py` | MODIFY — v2 categories, STAT_CV, lower-is-better, new entry point, dynamic threshold | A |
| `backend/fantasy_baseball/mcmc_simulator.py` | MODIFY — canonical keys in _PLAYER_WEEKLY_STD + docstring | B |
| `backend/services/simulation_bridge.py` | CREATE — ROW→MC adapter + PlayerScore→MCMC adapter | C |
| `backend/services/category_math.py` | MODIFY — add `compute_ratio_risk()` + `compute_category_count_delta()` | D |
| `tests/test_h2h_monte_carlo_v2.py` | CREATE — 14+ tests | E |
| `tests/test_mcmc_v2.py` | CREATE — 7+ tests | E |
| `tests/test_simulation_bridge.py` | CREATE — 5+ tests | E |
| `tests/test_ratio_risk.py` | CREATE — 10+ tests | E |
| `tests/test_phase3_integration.py` | CREATE — 5+ tests | E |
