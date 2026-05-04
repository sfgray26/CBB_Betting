# K2 ROW Projection Architecture Spec
**Date:** 2026-04-18  
**Author:** Kimi CLI  
**Scope:** Read-only audit of existing projection infrastructure + spec for Rest-of-Week (ROW) projection pipeline.

---

## Executive Summary

**There is no existing ROW projection pipeline.** `projections_bridge.py` does not exist. The `PlayerProjection` ORM table exists but has **zero rows** in production. The `h2h_monte_carlo.py` simulator expects pre-projected player dicts but produces none itself. `CanonicalPlayerRow.row_projection` (PR-18) is always `None` today.

This memo specifies the greenfield ROW projection function that Claude must implement in Phase 2. The math differs materially between counting stats (linear sum) and ratio stats (weighted harmonic aggregation). Getting ratio stats wrong is the primary implementation risk.

---

## 1. What Projection Infrastructure Exists Today?

### `PlayerProjection` ORM (`models.py:681-733`)

```python
class PlayerProjection(Base):
    __tablename__ = "player_projections"
    id = Column(Integer, primary_key=True)
    player_id = Column(String(50), nullable=False, unique=True)
    player_name = Column(String(100))
    team = Column(String(10))
    positions = Column(JSON)
    # Stats: woba, avg, obp, slg, ops, xwoba, hr, r, rbi, sb, era, whip, k_per_nine, bb_per_nine
    # Bayesian metadata: shrinkage, data_quality_score, sample_size, prior_source, update_method
    cat_scores = Column(JSONB, default=dict)
```

**Status in prod:** `player_projections` table has **0 rows** per `HANDOFF.md` database audit. No pipeline writes to it.

### `h2h_monte_carlo.py` (`backend/fantasy_baseball/h2h_monte_carlo.py`)

This module **consumes** projections but does not **produce** them.

```python
def _aggregate_roster(self, roster: List[Dict[str, Any]]) -> Dict[str, float]:
    aggregates = {cat: 0.0 for cat in self.HITTING_CATS + self.PITCHING_CATS}
    for player in roster:
        for cat in aggregates:
            aggregates[cat] += player.get(cat, 0.0)
    return aggregates
```

It expects each player dict to already contain projected values for:
- Hitting: `R`, `HR`, `RBI`, `SB`, `NSB`, `AVG`, `OPS`
- Pitching: `W`, `QS`, `K`, `K/9`, `ERA`, `WHIP`

Note: The simulator uses `K` (not `K_P`) and `K/9` (not `K_9`), and omits `H`, `K_B`, `TB`, `L`, `HR_P`, `NSV`. It is **not aligned** with the v2 18-category contract. This misalignment must be resolved in Phase 2.

### `CanonicalPlayerRow` Contract (`contracts.py:332-359`)

```python
class CanonicalPlayerRow(BaseModel):
    player_name: str
    team: str
    eligible_positions: List[str]
    status: str
    game_context: Optional[PlayerGameContext]
    season_stats: Optional[CategoryStats]     # PR-13
    rolling_7d: Optional[CategoryStats]       # PR-14
    rolling_15d: Optional[CategoryStats]      # PR-15
    rolling_30d: Optional[CategoryStats]      # PR-16
    ros_projection: Optional[CategoryStats]   # PR-17
    row_projection: Optional[CategoryStats]   # PR-18 (Phase 2 deliverable — always None today)
    ownership_pct: Optional[float]
    injury_status: Optional[str]
    injury_return_timeline: Optional[str]
    freshness: FreshnessMetadata
    yahoo_player_key: Optional[str]
    bdl_player_id: Optional[int]
    mlbam_id: Optional[int]
```

`CategoryStats` is a `Dict[str, Optional[float]]` validated against `SCORING_CATEGORY_CODES` (the 18 v2 canonical codes).

---

## 2. ROW Projection Method by Category

### Assumptions

- **ROW = Rest of Week.** A standard H2H matchup runs Monday–Sunday. If today is Thursday, the ROW projects Thursday–Sunday.
- **Games remaining** per player = `days_remaining_in_matchup × expected_starts_per_day`. For hitters, this is usually `1.0` per day they are active. For pitchers, it depends on rotation slot and probable starter status.
- **Daily rate** for counting stats = `rolling_14d_value / 14` (or `/ games_in_window` for better accuracy).
- **Daily rate** for ratio stats = the ratio itself (AVG, OPS, ERA, WHIP, K/9) — but **team-level aggregation must use weighted formulas**.

### Counting Stats: `season_daily_rate × games_remaining`

| Canonical Code | Formula | Notes |
|----------------|---------|-------|
| `R` | `proj_r = (rolling_14d_r / 14) × games_remaining` | Use `games_in_window` denominator for precision |
| `H` | `proj_h = (rolling_14d_h / 14) × games_remaining` | |
| `HR_B` | `proj_hr_b = (rolling_14d_hr / 14) × games_remaining` | |
| `RBI` | `proj_rbi = (rolling_14d_rbi / 14) × games_remaining` | |
| `K_B` | `proj_k_b = (rolling_14d_so_bat / 14) × games_remaining` | Lower-is-batter, but projection is still sum |
| `TB` | `proj_tb = (rolling_14d_tb / 14) × games_remaining` | Need `w_tb` in rolling stats (see K1) |
| `NSB` | `proj_nsb = (rolling_14d_nsb / 14) × games_remaining` | Already have `w_net_stolen_bases` |
| `W` | `proj_w = (season_w / season_gs) × remaining_starts` | Season rate, not rolling — small sample in rolling window |
| `L` | `proj_l = (season_l / season_gs) × remaining_starts` | |
| `HR_P` | `proj_hr_p = (season_hr_p / season_ip) × projected_remaining_ip` | Rate per IP |
| `K_P` | `proj_k_p = (rolling_14d_k_pit / 14) × games_remaining` | Or season rate for pitchers |
| `QS` | `proj_qs = (season_qs / season_gs) × remaining_starts` | |
| `NSV` | `proj_nsv = (season_nsv / season_appearances) × remaining_appearances` | Closer role volatility is high |

**Recommendation for counting stats:** Use a **blended rate**:
```
daily_rate = 0.6 × (rolling_14d / 14) + 0.4 × (season_total / days_into_season)
```
This stabilizes early-season projections where 14-day rolling windows have high variance.

### Ratio Stats: Weighted Aggregation Across Roster

Ratio stats **cannot** be projected by summing player ratios. The correct team-level formula requires numerators and denominators.

| Canonical Code | Team-Level Aggregation Formula | Display Precision |
|----------------|--------------------------------|-------------------|
| `AVG` | `sum(H) / sum(AB)` | .3 |
| `OPS` | `sum(OBP_num) / sum(AB) + sum(SLG_num) / sum(AB)` | .3 |
| `ERA` | `9 × sum(ER) / sum(IP)` | .2 |
| `WHIP` | `sum(H_allowed + BB_allowed) / sum(IP)` | .3 |
| `K_9` | `27 × sum(K_P) / sum(IP_outs)` | .2 |

**Important:** For ROW projection, the "current" matchup value already includes some innings/AB. The projected final must add the ROW numerator/denominator to the current numerator/denominator:

```python
# Example: ERA
current_er = yahoo_scoreboard_era_numerator   # ER accumulated so far this week
current_ip = yahoo_scoreboard_ip              # IP accumulated so far this week

row_er = sum(proj_er for each_active_pitcher)
row_ip = sum(proj_ip for each_active_pitcher)

projected_final_era = 9.0 * (current_er + row_er) / (current_ip + row_ip)
```

The same pattern applies to AVG, OPS, WHIP, and K/9.

---

## 3. What Inputs Does the H2H Monte Carlo Need?

`h2h_monte_carlo.py:87-132`:

```python
def simulate_week(
    self,
    my_roster: List[Dict[str, Any]],
    opponent_roster: List[Dict[str, Any]],
    n_sims: int = 10000,
    as_of_date: date = None,
) -> H2HWinResult:
```

Each player dict must contain projected values keyed by the **simulator's category strings**:
```python
HITTING_CATS = ["R", "HR", "RBI", "SB", "NSB", "AVG", "OPS"]
PITCHING_CATS = ["W", "QS", "K", "K/9", "ERA", "WHIP"]
```

**Required shape per player:**
```python
{
    "name": str,
    "R": float, "HR": float, "RBI": float, "SB": float, "NSB": float,
    "AVG": float, "OPS": float,
    "W": float, "QS": float, "K": float, "K/9": float, "ERA": float, "WHIP": float,
}
```

**Critical gap:** The simulator uses `"K"` (not `"K_P"`) and `"K/9"` (not `"K_9"`). It also omits `H`, `K_B`, `TB`, `L`, `HR_P`, `NSV`. To align with the v2 18-category contract, the simulator must be updated to use canonical codes and accept all 18 categories.

---

## 4. Proposed Function Signature

```python
from typing import Dict, List, Optional
from backend.contracts import CanonicalPlayerRow, CategoryStats
from backend.stat_contract import SCORING_CATEGORY_CODES, LOWER_IS_BETTER

def compute_row_projection(
    roster: List[CanonicalPlayerRow],
    games_remaining: Dict[str, int],  # player_key -> expected games remaining this week
    season_rates: Optional[Dict[str, Dict[str, float]]] = None,  # player_key -> {canonical: season_rate}
) -> Dict[str, float]:
    """
    Compute team-level Rest-of-Week (ROW) projections for all 18 scoring categories.

    Parameters
    ----------
    roster: List of CanonicalPlayerRow for active roster (not bench).
    games_remaining: Expected remaining games per player this matchup week.
    season_rates: Optional season-long rates for blended projection (recommended for pitchers).

    Returns
    -------
    Dict mapping canonical_code -> projected team total for the rest of the week.
    Counting stats: projected sum.
    Ratio stats: projected weighted ratio.
    """
```

### Implementation Sketch

```python
def compute_row_projection(roster, games_remaining, season_rates=None):
    # Accumulators for ratio-stat numerators/denominators
    proj = {code: 0.0 for code in SCORING_CATEGORY_CODES}
    
    # Ratio stat component accumulators
    sum_h = sum_ab = 0.0
    sum_obp_num = sum_slg_num = sum_ab_ops = 0.0
    sum_er = sum_ip = 0.0
    sum_h_allowed = sum_bb_allowed = sum_ip_whip = 0.0
    sum_k_p = sum_ip_outs = 0.0
    
    for player in roster:
        gr = games_remaining.get(player.yahoo_player_key, 0)
        if gr == 0:
            continue
        
        rolling = player.rolling_14d.values if player.rolling_14d else {}
        season = season_rates.get(player.yahoo_player_key, {}) if season_rates else {}
        
        # --- Counting stats ---
        for code in ["R", "H", "HR_B", "RBI", "K_B", "TB", "NSB"]:
            rolling_rate = (rolling.get(code, 0.0) / 14.0) if rolling.get(code) else 0.0
            season_rate = season.get(code, 0.0)
            blended = 0.6 * rolling_rate + 0.4 * season_rate
            proj[code] += blended * gr
        
        # Pitcher counting stats
        for code in ["W", "L", "HR_P", "K_P", "QS", "NSV"]:
            # For pitchers, season rate is more stable than rolling
            season_rate = season.get(code, 0.0)
            proj[code] += season_rate * gr
        
        # --- Ratio stat components ---
        # AVG / OPS
        h = rolling.get("H", 0.0) / 14.0 * gr
        ab = rolling.get("AB", 0.0) / 14.0 * gr
        sum_h += h
        sum_ab += ab
        
        # OBP numerator = H + BB
        bb = rolling.get("BB", 0.0) / 14.0 * gr
        sum_obp_num += h + bb
        sum_ab_ops += ab
        
        # SLG numerator = TB
        tb = rolling.get("TB", 0.0) / 14.0 * gr
        sum_slg_num += tb
        
        # ERA
        er = rolling.get("ER", 0.0) / 14.0 * gr
        ip = rolling.get("IP", 0.0) / 14.0 * gr
        sum_er += er
        sum_ip += ip
        
        # WHIP
        h_all = rolling.get("H_P", 0.0) / 14.0 * gr  # hits allowed
        bb_all = rolling.get("BB_P", 0.0) / 14.0 * gr  # walks allowed
        sum_h_allowed += h_all
        sum_bb_allowed += bb_all
        sum_ip_whip += ip
        
        # K/9
        k_p = rolling.get("K_P", 0.0) / 14.0 * gr
        sum_k_p += k_p
        sum_ip_outs += ip * 3.0  # convert IP to outs
    
    # --- Compute team-level ratio stats ---
    if sum_ab > 0:
        proj["AVG"] = sum_h / sum_ab
    if sum_ab_ops > 0:
        obp = sum_obp_num / sum_ab_ops
        slg = sum_slg_num / sum_ab_ops
        proj["OPS"] = obp + slg
    if sum_ip > 0:
        proj["ERA"] = 9.0 * sum_er / sum_ip
        proj["WHIP"] = (sum_h_allowed + sum_bb_allowed) / sum_ip
    if sum_ip_outs > 0:
        proj["K_9"] = 27.0 * sum_k_p / sum_ip_outs
    
    return proj
```

**Note:** This sketch assumes `CanonicalPlayerRow.rolling_14d` contains the 18 canonical codes. Today it does not — the rolling stats only store 9 Z-score equivalents. The `CategoryStats` validator in `contracts.py` enforces that all 18 keys are present, so either:
1. The rolling stats pipeline must be expanded first (see K1), OR
2. The ROW projection function must tolerate missing keys and fall back to season rates.

---

## 5. Ratio Stat Aggregation: Detailed Derivation

### Team AVG

Player-level AVG = H / AB. Team AVG is NOT the average of player AVGs.

```
team_avg = sum(H_i) / sum(AB_i)
```

For ROW projection, each player contributes `proj_H_i` and `proj_AB_i` to the remaining-week totals. The projected final AVG adds these to the current matchup totals:

```
projected_final_avg = (current_H + sum(proj_H_i)) / (current_AB + sum(proj_AB_i))
```

### Team OPS

OPS = OBP + SLG. Team OPS = team_OBP + team_SLG.

```
team_obp = sum(H_i + BB_i) / sum(AB_i + BB_i)
team_slg = sum(TB_i) / sum(AB_i)
team_ops = team_obp + team_slg
```

Note: OBP denominator is `AB + BB` (no HBP/SF in our data model), while SLG denominator is `AB` only.

### Team ERA

```
team_era = 9 × sum(ER_i) / sum(IP_i)
```

For projected final:
```
projected_final_era = 9 × (current_ER + sum(proj_ER_i)) / (current_IP + sum(proj_IP_i))
```

### Team WHIP

```
team_whip = sum(H_allowed_i + BB_allowed_i) / sum(IP_i)
```

### Team K/9

```
team_k_9 = 27 × sum(K_P_i) / sum(IP_outs_i)
```

where `IP_outs_i = IP_i × 3`.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Early-season rolling rates are noisy | High | Blend 60% rolling + 40% season rate |
| Pitcher start count uncertainty | High | Use probable pitcher feed + rotation math |
| Two-way players (Ohtani) double-counted | Medium | Ensure Ohtani appears in both hitter and pitcher pools, but roster logic deduplicates |
| Ratio stat division by zero | Medium | Guard all denominators; return `None` if zero |
| Missing `AB`/`IP` in rolling stats for some players | Medium | Fallback to season averages or league mean |

---

## Recommended Next Actions (for Claude)

1. **Implement `compute_row_projection()`** as a pure function in a new module (e.g., `backend/fantasy_baseball/row_projector.py`).
2. **Update `h2h_monte_carlo.py`** to accept all 18 canonical codes and use canonical key names (`K_P`, `K_9`, etc.).
3. **Wire ROW projections into the scoreboard endpoint** (`GET /api/fantasy/scoreboard`) so `my_projected_final` and `opp_projected_final` are populated.
4. **Expand rolling stats pipeline** (per K1) so `CanonicalPlayerRow.rolling_14d` has all 18 canonical codes.
