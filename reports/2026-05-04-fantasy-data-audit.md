# Fantasy Baseball Data Pipeline — Deep Audit Report

**Auditor:** Kimi CLI (Deep Intelligence Unit)  
**Date:** 2026-05-04  
**Scope:** All production Python files in `backend/fantasy_baseball/`, `backend/routers/fantasy.py`, `backend/services/scoring_engine.py`, `backend/schemas.py`, `backend/models.py`, `backend/stat_contract/`, and associated test modules.  
**Test Baseline:** 2475 pass / 3 skip / 0 fail  
**Files Read:** 22 (all required source + 5 test modules)

---

## Executive Summary

The Fantasy Baseball subsystem is **functionally operational** but carries **three P0 mathematical inconsistencies** that distort player valuation, win-probability simulation, and API contract guarantees. The most severe issue is a **scale mismatch between live Z-scores (population std) and board Z-scores (sample std)**, which means a board z_score of 2.0 and a live composite_z of 2.0 are **not comparable metrics**. This undermines every downstream decision engine (waiver, lineup, recommendations) that blends both data sources.

Additionally, the MCMC simulator uses a **normal distribution for all categories**, including discrete counting stats (SB, SV, W, QS) over a 7-day window, materially understating tail risk for binary events.

**Risk posture:** The system will not crash, but it will systematically mis-rank players and generate low-confidence roster moves until these math issues are resolved.

---

## 1. Data Quality & Consistency

### 1.1 P0 — Z-Score Scale Mismatch: Live vs. Board

| Component | Std Method | Location |
|-----------|-----------|----------|
| `scoring_engine.py` | Population std (`ddof=0`) | Line 267-274 |
| `player_board.py` | Sample std (`statistics.stdev`, `ddof=1`) | Line 628-629 |

**Impact:** `scoring_engine.py` computes league Z-scores across all players with non-null values (correct for a finite league pool). `player_board.py` computes Z-scores against its hardcoded Steamer/ZiPS board using sample standard deviation. The two scales differ by a factor of `sqrt(N/(N-1))` (~1.005 for large N, but diverges for small N).

Because waiver and lineup endpoints blend board projections (`get_or_create_projection`) with live scoring data (`compute_league_zscores`), the composite rankings are **internally inconsistent**. A player valued at z=2.0 on the board is not the same quality as a player valued at composite_z=2.0 in the live table.

**Fix:** Change `player_board.py:628` to use population std, matching `scoring_engine.py`.

```python
# player_board.py line 628-629 (CURRENT)
mean = statistics.mean(values)
std = statistics.stdev(values)   # ddof=1 — WRONG

# FIX
def _population_std(values):
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((x - mean) ** 2 for x in values) / n)
std = _population_std(values)
```

### 1.2 P0 — Composite_z Is Weighted Sum (Not Mean), Scale Varies by Player Type

`scoring_engine.py:525-528`:
```python
result.composite_z = (
    sum(_CATEGORY_WEIGHTS.get(k, 1.0) * v for k, v in kv_pairs)
    if kv_pairs else 0.0
)
```

**Impact:** Two-way players (Ohtani) accumulate 8+ category Z-scores; pure hitters only 5. Unnormalized sum means the **player_type directly drives the composite_z scale**. A two-way player with mediocre Z-scores in many categories can outrank a specialist who dominates 3 categories.

The `models.py` docstring for `PlayerScore.composite_z` explicitly says:
> "composite_z = mean of all applicable non-None per-category Z-scores"

But the implementation computes a **weighted sum**, not a mean. The docstring and the code disagree.

**Fix:** Divide by the sum of weights or the count of categories:
```python
total_weight = sum(_CATEGORY_WEIGHTS.get(k, 1.0) for k, _ in kv_pairs)
result.composite_z = (
    sum(_CATEGORY_WEIGHTS.get(k, 1.0) * v for k, v in kv_pairs) / total_weight
    if kv_pairs and total_weight > 0 else 0.0
)
```

### 1.3 P1 — player_board Hardcoded Data Quality Issues

`player_board.py:46-332` contains `_BATTER_RAW` / `_PITCHER_RAW` with the following defects:
- **Duplicates:** Yordan Alvarez appears twice (lines ~78 and ~112 in the hardcoded tuple).
- **Late additions in "missed" section:** Aaron Judge was appended outside the main sorted block, indicating the board was manually patched rather than regenerated from a clean source.
- **Approximation:** `slg = ops - 0.330` (line 595) assumes a fixed league-average OBP of .330. This is not real SLG and introduces systematic bias for players with OBP far from .330.

### 1.4 P1 — Field Validators Silently Coerce Bad Data

`schemas.py:420-426`:
```python
@field_validator("need_score", "owned_pct", "projected_saves", "projected_points", mode="before")
@classmethod
def default_floats(cls, v):
    if v is None or (isinstance(v, float) and v != v):
        return 0.0
    return v
```

**Impact:** `0.0` is indistinguishable from "no data". A player with genuinely zero projected saves looks identical to a missing projection. This causes the waiver wire endpoint to present players as "0.0 projected saves" when the real issue is "no projection available."

`schemas.py:539-545` `default_z_score` is slightly better (converts to `None`), but the `default_floats` validator remains lossy for the fields it covers.

### 1.5 P1 — Statcast DB Tier Coverage Gaps

`models.py` defines `StatcastBatterMetrics` and `StatcastPitcherMetrics` with missing columns:
- `StatcastBatterMetrics` lacks `sprint_speed`
- `StatcastPitcherMetrics` lacks `stuff_plus`, `location_plus`

These fields **are populated in the pybaseball loader** (`pybaseball_loader.py`) and are referenced in briefing/waiver enrichment, but they **cannot be persisted to the database** because the ORM models don't have columns for them. This forces reliance on in-memory/cached pybaseball data rather than the DB tier.

---

## 2. Scoring Math & Statistical Correctness

### 2.1 P0 — MCMC Normal Distribution Misuse for Counting Stats

`mcmc_simulator.py:302-303`:
```python
my_noise = rng.normal(0.0, my_stds, size=(n_sims,) + my_means.shape)
my_totals = (my_means + my_noise).sum(axis=1)
```

**Impact:** The simulator uses `numpy.random.normal` for **all categories**, including:
- SB (stolen bases) — discrete, often 0-1 per week
- SV (saves) — binary for most closers (0 or 1 per appearance)
- W (wins) — binary, zero-inflated
- QS (quality starts) — binary

A normal distribution can produce **negative values** for these stats (which are physically impossible) and understates tail risk. For example, a closer with a mean of 2.0 saves/week and std=1.4 could be simulated at -1.2 saves.

**Fix:** Use Poisson or zero-inflated Poisson for counting stats with `mean <= 5`:
```python
# Pseudocode for fix
my_totals = np.zeros(n_sims)
for i, cat in enumerate(categories):
    if cat in COUNTING_CATS and my_means[i] <= 5:
        my_totals += rng.poisson(max(0, my_means[i]), size=n_sims)
    else:
        my_totals += rng.normal(my_means[i], my_stds[i], size=n_sims)
```

### 2.2 P1 — category_aware_scorer Ignores Cross-Category Correlation

`category_aware_scorer.py:145`:
```python
blended_score = 0.4 * player_z_score + 0.6 * (cat_score / n_cats_safe)
```

**Impact:** A player who helps HR but hurts AVG gets **linearly summed treatment**. In reality, power hitters who help HR often hurt AVG (high strikeout rates). The scorer treats these as independent contributions and may recommend a player who creates a new deficit larger than the gain.

### 2.3 P1 — DailyLineupOptimizer Uses Magic Numbers

`daily_lineup_optimizer.py:536-542`:
```python
talent_prior = (
    proj.get("hr", 0) * 2.0 / _GAMES_ROS
    + proj.get("r", 0) * 0.3 / _GAMES_ROS
    + proj.get("rbi", 0) * 0.3 / _GAMES_ROS
    + proj.get("nsb", 0) * 0.5 / _GAMES_ROS
    + proj.get("avg", 0.0) * 5.0
) * 10 + cz_val * 1.0
```

**Impact:** The coefficients (`2.0`, `0.3`, `0.5`, `5.0`) are **not derived from category weights or league scoring**. They are hardcoded heuristics. `avg * 5.0` is a rate stat multiplied by 5 to fit a 0-20 per-game scale. These weights may not match the actual category values in the user's league.

### 2.4 P1 — player_board Uses Approximate Derivations

`player_board.py:954`:
```python
obp = avg + 0.070  # approximation when OBP is missing
woba = ops * 0.95   # approximation when wOBA is missing
```

**Impact:** These approximations assume fixed relationships that don't hold across player profiles. A slap hitter (high AVG, low BB) has a much smaller AVG→OBP gap than a walk-heavy power hitter.

---

## 3. API Contract & Schema Integrity

### 3.1 P0 — WaiverPlayerOut.projected_points Permanently 0.0

`schemas.py:411`:
```python
projected_points: float = 0.0
```

**Impact:** No upstream consumer populates this field. It is always `0.0` in API responses, which misleads frontend consumers into thinking every free agent has zero projected points. Either populate it from `PlayerProjection` or remove it from the schema to avoid contract confusion.

### 3.2 P0 — RosterMoveRecommendation.category_targets Always Empty

`schemas.py:458`:
```python
category_targets: List[str] = []
```

**Impact:** The waiver recommendations endpoint (`fantasy.py:2644`) hardcodes `category_targets=[]` for every recommendation. This field was intended to tell the user *which categories* the move targets, but it is never computed. This is dead schema surface area.

### 3.3 P1 — models.py Uses datetime.utcnow() (Violates AGENTS.md Gate)

`models.py:1018-1019` and multiple other locations:
```python
created_at = Column(DateTime, default=datetime.utcnow)
updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

**Impact:** `AGENTS.md` Code Quality Gate #3 explicitly forbids `datetime.utcnow()`. The `FantasyProfile`, `MLBTeam`, `MLBGameLog`, `PlayerRollingStats`, `PlayerScore`, `PlayerMomentum`, and `SimulationResult` models all use it. These should use `datetime.now(ZoneInfo("America/New_York"))` for consistency with the rest of the codebase.

### 3.4 P1 — Stat Contract Dual Source-of-Truth Risk

The system has **two authoritative stat mappings**:
1. `backend/data/fantasy_stat_contract.json` — loaded by `stat_contract/loader.py`
2. `backend/stat_contract/registry.py` — Python dataclass registry

`registry.py` is imported directly by `yahoo_client_resilient.py` (line 1161: `from backend.stat_contract import load_contract`), while `fantasy.py` uses the JSON contract indirectly through the registry's exported constants. If `registry.py` and `fantasy_stat_contract.json` diverge, the system will have inconsistent stat semantics between Yahoo client parsing and waiver endpoint display.

**Evidence:** `registry.py` defines `yahoo_stat_id` mappings that were corrected in `STAT_MAPPING_ANALYSIS.md`, but there is no automated check that `fantasy_stat_contract.json` contains the same corrections.

---

## 4. Yahoo Integration & External API Risks

### 4.1 P1 — Silent Truncation in Batch Stats Fetching

`yahoo_client_resilient.py:822-824`:
```python
keys_str = ",".join(player_keys[:25])
data = self._get(
    f"league/{self.league_key}/players;player_keys={keys_str}/stats;type={stat_type}"
)
```

**Impact:** Calling with >25 keys **silently drops extras** with no warning or error. The `get_roster` endpoint in `fantasy.py:2886` passes all player keys to `get_players_stats_batch`. A 25-man roster will have 0 players truncated, but larger requests (e.g., free agent pages of 50+) will lose data.

**Fix:** Either raise if `len(player_keys) > 25`, or implement chunking with multiple requests.

### 4.2 P1 — No 429 / Rate-Limit Backoff

`yahoo_client_resilient.py:316-351` handles status 401 (single retry) and 999 (exponential backoff), but **does not handle HTTP 429** (Too Many Requests). Yahoo's API rate limits are documented as 2,000 requests/hour per app. Without 429 handling, a burst of requests (e.g., during waiver wire scanning) will fail hard.

### 4.3 P1 — Hardcoded Token Refresh Sleep

`yahoo_client_resilient.py:236`:
```python
time.sleep(2)  # allow token propagation
```

**Impact:** A fixed 2-second sleep assumes Yahoo's token propagation is always ≤2 seconds. This is a race condition that could cause intermittent auth failures under load.

### 4.4 P1 — Free Agent Deduplication Not Handled Before Stats Merge

`yahoo_client_resilient.py:get_free_agents()` paginates by position. The same player can appear in multiple position pages (e.g., a 1B/3B player appears in both 1B and 3B pages). Deduplication is **not handled before stats are merged** in the batch call. While the batch stats call uses `player_keys`, if the same key appears twice in the input list, the Yahoo API may return it once or twice depending on its own dedup logic, leading to inconsistent response sizes.

---

## 5. Test Coverage & Observability

### 5.1 Untested Edge Cases in scoring_engine

`test_scoring_engine.py` (21 tests) and `test_scoring_engine_fixes.py` (3 tests) cover happy-path Z-score computation, but **do not test**:
- Empty roster (all `cat_scores` missing)
- All-zero `cat_scores` (division by zero in std)
- Week 1 cold start (`MIN_SAMPLE=3` masks this, but what if only 2 players have data?)
- IL-full roster (no active players in a category)
- Two-way player composite_z (Ohtani-style mixed hitter/pitcher categories)

### 5.2 No Waiver Statcast Field Verification

No test verifies that the waiver endpoint (`/api/fantasy/waiver`) returns **non-empty `statcast_stats`** or **`statcast_signals`** for players that exist in the Statcast database. The `statcast_loader.py` name-matching logic could fail silently, and the endpoint would still return HTTP 200 with null statcast fields.

### 5.3 MCMC win_prob_gain Absent from Production Logs

During prior Railway log checks, **no `win_prob_gain` log lines were found**. The `mcmc_simulator.py` debug logs at line 2608-2615 should produce:
```
[MCMC] <name>: enabled=True win_prob 0.450->0.480 gain=0.030 opp_roster=12
```

Absence of these logs indicates either:
1. `opponent_roster` is empty for most matchups (opponent_team_key resolution fails)
2. MCMC is disabled (`mcmc_enabled=False`) because `opponent_roster_scored` has no `cat_scores`
3. The log level is above INFO in production

This is a **blind spot**: the most sophisticated feature in the waiver engine may not be running in production.

### 5.4 models.py datetime.utcnow() in Test Fixtures

Multiple test fixtures may inherit the `datetime.utcnow()` default from `models.py`. If tests assert on timezone-aware timestamps, they will fail when run in ET. This is a latent test fragility.

---

## 6. Recommended Fixes (Prioritized)

### P0 — Must Fix Before Next Waiver Cycle

| # | File | Line(s) | Issue | Fix Complexity |
|---|------|---------|-------|----------------|
| 1 | `player_board.py` | 628-629 | Sample std (ddof=1) inconsistent with scoring_engine | Low — replace with population std helper |
| 2 | `scoring_engine.py` | 525-528 | Composite_z is weighted sum, not mean | Low — divide by total_weight |
| 3 | `mcmc_simulator.py` | 302-303 | Normal distribution for counting stats | Medium — Poisson for low-mean counting cats |
| 4 | `schemas.py` | 411 | `projected_points` permanently 0.0 | Low — populate from projection or remove field |
| 5 | `schemas.py` | 458 | `category_targets` always empty | Low — compute from cat_scores or remove field |

### P1 — Fix Before Mid-Season Lineup Optimization

| # | File | Line(s) | Issue | Fix Complexity |
|---|------|---------|-------|----------------|
| 6 | `yahoo_client_resilient.py` | 822-824 | Silent truncation at 25 keys | Low — raise or chunk |
| 7 | `yahoo_client_resilient.py` | 316-351 | No 429 backoff | Low — add status code check |
| 8 | `models.py` | Multiple | `datetime.utcnow()` violations | Low — replace with ET-aware now |
| 9 | `player_board.py` | 595, 954 | `slg = ops - 0.330`, `obp = avg + 0.070` | Medium — derive from real components |
| 10 | `schemas.py` | 420-426 | Silent None/NaN → 0.0 coercion | Low — preserve None, let frontend handle |
| 11 | `models.py` | Statcast models | Missing `sprint_speed`, `stuff_plus`, `location_plus` | Low — add columns + migration |
| 12 | `daily_lineup_optimizer.py` | 536-542 | Magic-number talent prior | Medium — derive from `_CATEGORY_WEIGHTS` |
| 13 | `category_aware_scorer.py` | 145 | No cross-category correlation | Medium — add anti-correlation penalty |

### P2 — Quality-of-Life / Tech Debt

| # | File | Line(s) | Issue | Fix Complexity |
|---|------|---------|-------|----------------|
| 14 | `yahoo_client_resilient.py` | 236 | Hardcoded `time.sleep(2)` | Low — retry with exponential backoff |
| 15 | `player_board.py` | 46-332 | Duplicate entries in hardcoded board | Low — deduplicate and sort |
| 16 | `stat_contract/` | All | Dual source-of-truth with JSON | Medium — single codegen pipeline |
| 17 | `fantasy.py` | 2608-2615 | MCMC logs absent — verify production | Low — check log level, add metric |

---

## Appendix: File Inventory

| # | File | Lines Read | Key Finding |
|---|------|------------|-------------|
| 1 | `backend/services/scoring_engine.py` | ~580 | Population std, weighted sum composite_z |
| 2 | `backend/fantasy_baseball/player_board.py` | ~1050 | Sample std, hardcoded approximations |
| 3 | `backend/fantasy_baseball/mcmc_simulator.py` | ~410 | Normal distribution for counting stats |
| 4 | `backend/fantasy_baseball/category_aware_scorer.py` | ~155 | Linear cross-category scoring |
| 5 | `backend/fantasy_baseball/waiver_edge_detector.py` | ~470 | Sort instability on tie |
| 6 | `backend/fantasy_baseball/daily_lineup_optimizer.py` | ~920 | Magic-number talent prior |
| 7 | `backend/routers/fantasy.py` | ~2824 | Dead fields, complex waiver logic |
| 8 | `backend/schemas.py` | ~580 | Silent coercion, unpopulated fields |
| 9 | `backend/fantasy_baseball/yahoo_client_resilient.py` | ~1534 | Silent truncation, no 429 |
| 10 | `backend/fantasy_baseball/statcast_loader.py` | ~380 | Fixed name→mlbam_id join |
| 11 | `backend/models.py` | ~1500+ | utcnow violations, missing Statcast cols |
| 12 | `backend/stat_contract/registry.py` | ~772 | Dual source-of-truth risk |
| 13 | `backend/stat_contract/__init__.py` | ~46 | Export constants |
| 14 | `tests/test_scoring_engine.py` | ~470 | Good coverage, missing edge cases |
| 15 | `tests/test_scoring_engine_fixes.py` | ~80 | Winsorization tests |
| 16 | `tests/test_fantasy_router.py` | ~130 | Endpoint contract tests |
| 17 | `tests/test_yahoo_client.py` | ~290 | Mock-based, no 429 test |
| 18 | `tests/test_waiver_edge.py` | ~180 | Drop candidate tests |
| 19 | `backend/fantasy_baseball/smart_lineup_selector.py` | ~370 | Hardcoded game time |
| 20 | `backend/fantasy_baseball/daily_briefing.py` | ~330 | Briefing generation |
| 21 | `backend/fantasy_baseball/pybaseball_loader.py` | ~220 | Statcast CSV loading |
| 22 | `backend/fantasy_baseball/category_tracker.py` | ~180 | Category need tracking |

---

*Report generated by Kimi CLI v1.17.0. All line numbers reference the state of the repository as of 2026-05-04 on branch `stable/cbb-prod`.*
