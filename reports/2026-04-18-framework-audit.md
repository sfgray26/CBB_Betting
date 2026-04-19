# Framework-Driven Audit of the Fantasy Baseball Platform
**Date:** 2026-04-18  
**Author:** Kimi CLI  
**Scope:** Apply elite fantasy baseball principles (Bayesian inference, EV, signal vs noise, market inefficiency, portfolio theory, process discipline, asymmetric risk, systems thinking, Occam's razor, antifragility) to audit mathematical models, data gaps, and implementation architecture.

---

## Executive Summary

**Verdict: The platform has solid foundations but systematically underutilizes its own data.**

The rolling window engine, scoring engine, and Monte Carlo simulator are mathematically sound and correctly implemented. However, the **decision layer** — the code that actually recommends lineups and waivers — operates on a crude heuristic (0.6×score_0_100 + 0.3×momentum + 0.1×projection) that ignores matchup context, category covariance, and tail risk. This is like building a Formula 1 engine and then steering with a toy wheel.

**Critical finding:** The `smart_lineup_selector.py` module contains sophisticated contextual logic (platoon splits, opposing pitcher quality, weather, park factors, category needs) but is **completely orphaned** — it is not called by `daily_ingestion.py`, `decision_engine.py`, or any API endpoint. The production pipeline uses the primitive `decision_engine.py` greedy optimizer instead.

**Data Quality: B+ (improving)** — Fresh through 2026-04-13, 30K+ rolling stats, NSB pipeline live.  
**Model Quality: B** — Core Z-score and simulation math is correct. Missing 5 pitching categories (W, L, HR_P, QS, NSV).  
**Decision Quality: C** — No category awareness, no matchup context, no covariance modeling.  
**System Integration: C+** — Sophisticated modules exist but are not wired together.

---

## 1. Principle-by-Principle Audit

### 1.1 Bayesian Inference — PARTIALLY IMPLEMENTED ⚠️

**What exists:**
- `PlayerProjection` ORM (`models.py:681`) has Bayesian metadata fields: `shrinkage`, `data_quality_score`, `sample_size`, `prior_source`, `update_method`.
- The rolling window engine uses exponential decay (λ=0.95), which is a form of time-weighted Bayesian updating — recent games count more than old games.
- The simulation engine blends rolling rates with league means for composite Z-score computation.

**What is missing:**
- `player_projections` table has **0 rows** in production. No prior distributions are loaded.
- No Steamer/ZiPS/The BAT ingestion pipeline exists (FanGraphs returns 403 locally, unclear if production works).
- The `row_projection` field in `CanonicalPlayerRow` is always `None`.
- There is no explicit shrinkage formula: `posterior = (precision_prior × prior + precision_likelihood × likelihood) / (precision_prior + precision_likelihood)`.

**Gap severity: HIGH.** Early-season projections (April) are extremely noisy because the 14-day rolling window has high variance. Without a strong prior (Steamer/ZiPS), the system overreacts to small samples.

**Implementation suggestion:**
```python
# New module: backend/fantasy_baseball/bayesian_projector.py
def blend_projection(prior: float, likelihood: float, prior_games: int, likelihood_games: int) -> float:
    """Precision-weighted Bayesian blend."""
    prior_precision = prior_games / 100  # Steamer based on ~3 seasons = 300+ games
    likelihood_precision = likelihood_games / 14  # rolling window
    return (prior_precision * prior + likelihood_precision * likelihood) / (prior_precision + likelihood_precision)
```

---

### 1.2 Expected Value (EV) — NOT IMPLEMENTED 🔴

**What exists:**
- `value_gain` in `optimize_waivers()` computes a raw delta: `_composite_value(candidate) - _composite_value(drop_candidate)`.
- `confidence` is a normalized score_0_100 + momentum blend.

**What is missing:**
- No probabilistic framing. The system asks "is this player better?" not "what is the probability this player improves my championship equity?"
- No H2H Monte Carlo integration into waiver decisions. The `h2h_monte_carlo.py` exists but is never called by `decision_engine.py` or `daily_ingestion.py`.
- No category-specific value. A player who helps HR but hurts AVG has no net EV computed.

**Gap severity: HIGH.** In H2H One Win, the correct question is "does adding this player increase P(win 10+ categories)?" not "does this player have a higher composite score?"

**Implementation suggestion:**
```python
# In optimize_waivers:
for candidate in waiver_pool:
    simulated_roster = current_roster + [candidate] - [drop_candidate]
    my_proj = compute_row_projection(simulated_roster, games_remaining)
    opp_proj = fetch_opponent_projection()  # from Yahoo scoreboard + ROW
    ev = h2h_sim.simulate_week_from_projections(my_proj, opp_proj, n_sims=1000)
    gain = ev.win_probability - baseline_win_probability
```

---

### 1.3 Signal vs. Noise — PARTIALLY IMPLEMENTED ⚠️

**What exists:**
- Winsorization at 5th/95th percentile (`scoring_engine.py:161-174`) clips outliers before Z computation.
- MAD-based robust Z-scores available via `use_mad=True` (`scoring_engine.py:189-203`).
- Exponential decay weights recent games more heavily.

**What is missing:**
- **No x-stats (expected stats) in the pipeline.** The system uses raw BDL box stats (HR, RBI, AVG) but never ingests Statcast xwOBA, barrel%, hard-hit%, xERA, xFIP, or CSW%.
- The `statcast_performances` table exists with 6,971 rows but is **never consumed** by the scoring engine, decision engine, or simulation engine.
- No BABIP regression modeling. A hitter with .400 BABIP is treated as genuinely elite rather than lucky.
- No HR/FB regression. A pitcher with 5% HR/FB is treated as elite rather than lucky.

**Gap severity: CRITICAL.** This is the single biggest information asymmetry gap. Elite managers use x-stats to buy low/sell high. The platform ignores them entirely.

**Implementation suggestion:**
1. Ingest Statcast xwOBA, xBA, barrel%, hard-hit% into `player_rolling_stats` as new columns.
2. Compute "luck-adjusted Z-scores": `z_luck_adj = z_raw + (z_xstat - z_raw) * 0.5` — blend 50% raw, 50% expected.
3. Add `xERA`, `xFIP`, `CSW%` to pitcher rolling stats.

---

### 1.4 Market Inefficiency & Arbitrage — NOT IMPLEMENTED 🔴

**What exists:**
- `waiver_edge_detector.py` exists (referenced in `main.py` and `dashboard_service.py`) but was not found in the expected path. The waiver endpoint in `main.py` returns Yahoo free agents with basic scoring.
- `smart_lineup_selector.py` has category need logic but is not wired into the waiver flow.

**What is missing:**
- No ownership% tracking. Cannot identify players whose underlying skill (x-stats) has improved faster than their ownership.
- No ADP tracking. Cannot identify draft values vs. market prices.
- No automated "buy low / sell high" signals based on x-stat vs. surface-stat divergence.
- The waiver pool in `daily_ingestion.py` is silently empty because `player_id_mapping` has no free agent yahoo_keys (see prior investigation).

**Gap severity: MEDIUM-HIGH.** In public leagues, the edge comes from being first to add breakout players. The platform cannot do this because it doesn't track the right signals.

**Implementation suggestion:**
1. Add `ownership_pct` and `adp` columns to `player_scores` or new `player_market` table.
2. Compute `value_gap = z_xstat - z_adp` (expected performance minus market expectation).
3. Surface top 10 positive gaps in the waiver endpoint.

---

### 1.5 Portfolio Theory (Covariance) — NOT IMPLEMENTED 🔴

**What exists:**
- `optimize_lineup()` fills slots greedily by composite score. It does not model category correlations.
- `h2h_monte_carlo.py` simulates categories independently (no covariance matrix).

**What is missing:**
- No category correlation matrix. Power hitters correlate HR/RBI but anti-correlate AVG/SB.
- No "punting" strategy detection. The system doesn't recognize when a category is unwinnable and resources should be reallocated.
- No roster construction optimization. The lineup optimizer picks the 13 best players by score; it doesn't ask "do I have too many power hitters and not enough speed?"

**Gap severity: MEDIUM.** In 18-category H2H, covariance matters less than in 5×5 because diversification is built into the category spread. But for streaming decisions ("should I add a speedster or a power bat?"), covariance is essential.

**Implementation suggestion:**
```python
# Category correlation matrix from league-wide player data
corr_matrix = np.corrcoef(player_vectors)  # shape: (18, 18)
# Roster diversification score: penalize concentrated category exposure
diversity_penalty = -sum(corr_matrix[i][j] * exposure[i] * exposure[j] for i, j in pairs)
```

---

### 1.6 Process Over Outcomes — NOT IMPLEMENTED 🔴

**What exists:**
- `_record_job_run()` tracks pipeline success/failure.
- `decision_results` stores what the system recommended.

**What is missing:**
- No backtesting of decision quality. Did the recommended lineup actually outperform the bench? Did the waiver add actually help?
- `backtest_results` table exists but has no documented pipeline populating it.
- No "process score" metric. The system doesn't know if it made the *right* decision given the information available at the time.

**Gap severity: MEDIUM.** Without backtesting, the system cannot learn from its mistakes.

**Implementation suggestion:**
1. After each matchup week, compare `decision_results` recommendations against actual Yahoo outcomes.
2. Compute `process_score = sum(category_wins_recommended) / sum(category_wins_optimal)`.
3. Feed this into a feedback loop that adjusts lineup_score weights.

---

### 1.7 Asymmetric Risk/Reward — PARTIALLY IMPLEMENTED ⚠️

**What exists:**
- `simulation_engine.py` computes `downside_p25`, `upside_p75`, and `prob_above_median`.
- These fields flow into `PlayerDecisionInput` and are displayed in `explainability_layer.py` risk narratives.

**What is missing:**
- **Tail risk is never used in the optimizer.** `optimize_lineup()` ranks by `_lineup_score()` which ignores `downside_p25` / `upside_p75`.
- No convexity-seeking behavior. A high-upside, high-risk player (e.g., a rookie with 30% barrel% but 50% K%) is treated the same as a safe veteran with the same mean projection.
- The `LineupConstraintSolver` (OR-Tools) maximizes total score but does not model variance.

**Gap severity: MEDIUM.** In H2H, upside variance is valuable when trailing. The current system is risk-neutral when it should be risk-seeking in close matchups.

**Implementation suggestion:**
```python
def _lineup_score(player: PlayerDecisionInput, need_upside: bool = False) -> float:
    base = 0.6 * score_component + 0.3 * momentum + 0.1 * proj
    if need_upside and player.upside_p75:
        upside_bonus = (player.upside_p75 - player.composite_z) * 0.2
        return base + upside_bonus
    return base
```

---

### 1.8 Systems Thinking (Feedback Loops) — PARTIALLY IMPLEMENTED ⚠️

**What exists:**
- The scheduler runs jobs in correct dependency order: box stats → rolling windows → player scores → momentum → simulation → decision optimization.
- Advisory locks prevent race conditions.

**What is missing:**
- **No auto-tuning.** The scoring engine weights (0.6/0.3/0.1), composite Z formula, and momentum thresholds are hardcoded. They never adjust based on backtesting results.
- No feedback from Yahoo outcomes into the pipeline. If the system consistently overvalues speedsters, it never learns.
- `data_ingestion_logs` table exists but has 0 rows.

**Gap severity: MEDIUM.** The system is a feed-forward pipeline, not a closed-loop control system.

**Implementation suggestion:**
1. Populate `data_ingestion_logs` with structured records.
2. Run weekly backtests comparing predictions to outcomes.
3. Use simple gradient descent to adjust `_lineup_score` weights based on backtest MAE.

---

### 1.9 Occam's Razor — IMPLEMENTED WELL ✅

**What exists:**
- Pure Python implementations, no numpy/pandas in scoring engine.
- Simple normal sampling in Monte Carlo (no GARCH, no ML).
- Greedy slot-filling in lineup optimizer (not a neural network).
- Z-score methodology is transparent and explainable.

**Verdict:** The models are appropriately simple. The complexity that *does* exist (`smart_lineup_selector.py`) is the right kind of complexity — domain-specific contextual layering.

**Caution:** The `smart_lineup_selector.py` module has many moving parts (weather, platoon, pitcher deep dive, park factors). If not maintained, this could become a brittle "kitchen sink." The current isolation (it's not wired into the main pipeline) actually protects the system from this fragility.

---

### 1.10 Antifragility — IMPLEMENTED WELL ✅

**What exists:**
- `YahooFantasyClient` has extensive fallback paths (auth retry, API retry, basic parsing fallback).
- `LineupConstraintSolver` falls back from OR-Tools to greedy if OR-Tools is unavailable.
- `daily_ingestion.py` catches exceptions at every stage and continues with degraded data (e.g., empty waiver pool, missing simulation results).
- `scoring_engine.py` skips categories with `< MIN_SAMPLE` players rather than crashing.

**Verdict:** The system degrades gracefully under failure. This is a genuine strength.

---

## 2. Mathematical Models Deep Dive

### 2.1 Rolling Window Engine — CORRECT ✅

```python
weight = 0.95 ** days_back
```

- IP parsing correctly handles baseball notation ("6.2" → 6.667).
- Exponential decay is the right choice for time-weighted performance.
- λ=0.95 implies a half-life of ~13.5 days, which is reasonable for capturing "hot streaks" without excessive noise.
- QS derivation (IP≥6, ER≤3) is correct and requires no new data source.

**Recent improvement:** `w_runs`, `w_tb`, `w_qs` were added since the April 15 audit. This expands coverage from 9 to 12 categories in rolling stats.

### 2.2 Scoring Engine — CORRECT BUT INCOMPLETE ⚠️

**Strengths:**
- Type-appropriate pools prevent pitcher batting nulls from diluting hitter Z-scores.
- Winsorization and MAD options are available.
- Z-cap at ±3.0 prevents single outliers from distorting rankings.
- `_COMPOSITE_EXCLUDED` correctly avoids double-counting `z_sb` vs `z_nsb`.

**Weaknesses:**
- **Composite Z is an unweighted mean of applicable Z-scores.** This assumes all categories are equally important. In H2H One Win, some categories are more "swingable" than others (e.g., NSB has high variance; AVG has low variance). A weighted composite that reflects category volatility would be better.
- **Percentile rank is within player_type cohort only.** This means a pitcher with composite_z=1.5 and a hitter with composite_z=1.5 both score ~90. But in a 9-9 category split, pitcher value and hitter value are not directly comparable.

### 2.3 Simulation Engine — CORRECT BUT INCOMPLETE ⚠️

**Strengths:**
- Thread-safe RNG (`random.Random(seed)` instance per call).
- Positive-floored normal draws prevent negative counting stats.
- Pitching appearance estimation based on IP/appearance is smart.

**Weaknesses:**
- **Only simulates 7 stats:** HR, RBI, SB, AVG (batting); K, ERA, WHIP (pitching).
- Missing: R, H, TB, K_B, OPS, K_P, QS, W, L, HR_P, NSV, K/9.
- **No correlated draws.** HR and RBI are drawn independently, but in reality they correlate strongly. Simulated rosters may have unrealistic covariance structures.
- **Rate stats (AVG, ERA, WHIP) are simulated as ratios of independent draws.** This introduces variance that doesn't match reality — a player's AVG doesn't vary as much as `hit_draw / ab_draw` would suggest because AB and hits are correlated.

**Implementation suggestion:** Simulate components, not ratios:
```python
# Instead of: avg = total_hit / total_ab
# Simulate hits and AB with correlation:
cov = [[var_ab, cov_ab_hits], [cov_ab_hits, var_hits]]
ab, hits = draw_correlated(ab_rate, hit_rate, cov, n_games)
```

### 2.4 H2H Monte Carlo — CORRECT BUT UNWIRED ⚠️

**Strengths:**
- Now aligned with v2 canonical codes (18 categories).
- Uses `LOWER_IS_BETTER` from stat_contract.
- Vectorized NumPy for performance.
- `simulate_week_from_projections()` is the right interface for ROW → Simulation bridge.

**Weaknesses:**
- **Never called in production.** The endpoint `GET /api/fantasy/scoreboard` does not invoke the Monte Carlo.
- **CV values are rough estimates.** `NSB: 0.50` may be too low for a stat that can be negative. `ERA: 0.15` may be too high for a rate stat.
- **No opponent roster projection.** The opponent is simulated from the same distribution as the user, but in reality opponents have different roster constructions.

### 2.5 Decision Engine — CRUDE AND MISALIGNED 🔴

**Critical critique:**

```python
def _lineup_score(player) -> float:
    mb_norm = (momentum_bonus + 10) / 2  # [-10,10] -> [0,10]
    pb = _proj_bonus(player)             # [0,10]
    score_component = score_0_100 / 10   # [0,10]
    return 0.6 * score_component + 0.3 * mb_norm + 0.1 * pb
```

**Problems:**
1. **No category awareness.** A team leading HR by 10 and trailing SB by 2 gets the same lineup as a team trailing HR by 2 and leading SB by 10.
2. **No matchup context.** The opponent's strengths/weaknesses are ignored.
3. **No weather/park context.** `smart_lineup_selector.py` has this but is not used.
4. **No positional scarcity in score.** The Catcher slot is filled by whoever has the highest composite score, not by the best catcher relative to replacement level.
5. **Projection bonus is simplistic.** Hitters get credit for HR + RBI only. No credit for R, H, TB, OPS, NSB. Pitchers get credit for K only. No credit for QS, W, ERA, WHIP.

**The `smart_lineup_selector.py` module solves most of these problems** but is completely disconnected from the production pipeline.

---

## 3. Data Gaps

| Category | Rolling Stats | Z-Score | Simulation | Decision Engine | Monte Carlo | Status |
|----------|--------------|---------|------------|-----------------|-------------|--------|
| R | ✅ w_runs | ✅ z_r | ❌ | ❌ | ❌ | Partial |
| H | ✅ w_hits | ✅ z_h | ❌ | ❌ | ❌ | Partial |
| HR_B | ✅ w_home_runs | ✅ z_hr | ✅ | ✅ (proxy) | ✅ | Complete |
| RBI | ✅ w_rbi | ✅ z_rbi | ✅ | ✅ (proxy) | ✅ | Complete |
| K_B | ✅ w_strikeouts_bat | ✅ z_k_b | ❌ | ❌ | ❌ | Partial |
| TB | ✅ w_tb | ✅ z_tb | ❌ | ❌ | ❌ | Partial |
| AVG | ✅ w_avg | ✅ z_avg | ✅ | ❌ | ❌ | Partial |
| OPS | ✅ w_ops | ✅ z_ops | ❌ | ❌ | ❌ | Partial |
| NSB | ✅ w_net_stolen_bases | ✅ z_nsb | ✅ | ✅ (proxy) | ✅ | Complete |
| W | ❌ | ❌ | ❌ | ❌ | ❌ | **Missing** |
| L | ❌ | ❌ | ❌ | ❌ | ❌ | **Missing** |
| HR_P | ❌ | ❌ | ❌ | ❌ | ❌ | **Missing** |
| K_P | ✅ w_strikeouts_pit | ✅ z_k_p | ❌ | ❌ | ❌ | Partial |
| ERA | ✅ w_era | ✅ z_era | ✅ | ❌ | ❌ | Partial |
| WHIP | ✅ w_whip | ✅ z_whip | ✅ | ❌ | ❌ | Partial |
| K_9 | ✅ w_k_per_9 | ✅ z_k_per_9 | ❌ | ❌ | ❌ | Partial |
| QS | ✅ w_qs | ✅ z_qs | ❌ | ❌ | ❌ | Partial |
| NSV | ❌ | ❌ | ❌ | ❌ | ❌ | **Missing** |

**Notes:**
- "Complete" means the category flows from raw data → Z-score → simulation → decision → Monte Carlo.
- "Partial" means some stages are implemented but not all.
- **5 categories are entirely missing** from the upstream data model: W, L, HR_P, QS (raw exists but not simulated/decided), NSV.

---

## 4. The Orphaned Module Problem

The following sophisticated modules exist in the codebase but **do not participate in the production pipeline:**

| Module | Capability | Why It's Orphaned | Integration Point |
|--------|-----------|-------------------|-------------------|
| `smart_lineup_selector.py` | Platoon, weather, park, pitcher quality, category needs | Not called by `decision_engine.py` or `daily_ingestion.py` | Replace `optimize_lineup()` call in `_run_decision_optimization()` |
| `weather_fetcher.py` (736 lines) | OpenWeatherMap integration, HR physics | `OPENWEATHER_API_KEY` status unknown; not in scheduler | Add to `smart_lineup_selector` pipeline |
| `park_weather.py` (549 lines) | Stadium microclimates, wind analysis | No DB table; not called | Create `ParkFactor` table, integrate into scoring |
| `ballpark_factors.py` (270 lines) | Hardcoded park factors | Only used by `daily_lineup_optimizer.py` line 263 | Expand to full scoring adjustment |
| `h2h_monte_carlo.py` | Win probability simulation | Not called by scoreboard endpoint | Wire into `GET /api/fantasy/scoreboard` |
| `category_tracker.py` | Category need/urgency computation | Referenced but not integrated into decision flow | Feed into `smart_lineup_selector` |
| `elite_lineup_scorer.py` | Advanced scoring with weather/park | Not imported by decision engine | Merge into `_lineup_score()` |

**This is a classic "invented here but never shipped" pattern.** The hard work of building sophisticated contextual logic has been done, but the plumbing to connect it to the daily decision pipeline has not.

---

## 5. OODA Loop Assessment

| Phase | Current State | Target State | Gap |
|-------|--------------|--------------|-----|
| **Observe** | BDL box stats + Statcast raw data + Yahoo roster | Add x-stats, ownership%, weather, probable pitchers | Missing x-stat ingestion, weather not scheduled |
| **Orient** | Z-scores + momentum + ROS simulation (7 stats) | ROW projections + category margins + Monte Carlo win prob | Missing ROW projector, Monte Carlo unwired |
| **Decide** | Greedy lineup by composite score; empty waiver pool | Category-aware lineup; EV-based waiver; smart context | Using primitive engine instead of smart selector |
| **Act** | DecisionResults upsert at 7 AM; manual Yahoo lineup set | Auto-lineup push to Yahoo API; Discord alerts for waivers | No Yahoo write API; no auto-lineup |

---

## 6. Implementation Roadmap

### Phase A: Close the Wiring Gap (1 week)

1. **Replace `optimize_lineup()` call** in `daily_ingestion.py:_run_decision_optimization()` with `SmartLineupSelector.select_optimal_lineup()`.
   - Pass category needs from Yahoo scoreboard.
   - Pass weather context if `OPENWEATHER_API_KEY` is available.
   - Fallback to `optimize_lineup()` if smart selector fails.

2. **Wire `h2h_monte_carlo.py`** into `GET /api/fantasy/scoreboard`.
   - Compute ROW projections first (see Phase B).
   - Run 1,000 simulations.
   - Return `win_probability`, `locked_categories`, `swing_categories`.

3. **Fix waiver pool empty issue.**
   - Implement name-based fuzzy matching for free agents (see prior investigation).
   - Add diagnostic logging.

### Phase B: Implement ROW Projector (1 week)

4. **Build `backend/fantasy_baseball/row_projector.py`.**
   - Counting stats: blended daily rate × games remaining.
   - Ratio stats: component accumulation + team-level division.
   - Input: `list[CanonicalPlayerRow]` + `games_remaining` dict.
   - Output: `dict[canonical_code, float]`.

5. **Expand `simulation_engine.py`** to cover all 18 categories.
   - Add R, H, TB, K_B, OPS, K_P, QS, W, L, HR_P, NSV, K/9 simulation paths.
   - For missing upstream data (W, L, HR_P, NSV), use Yahoo season stats as fallback.

### Phase C: Add Bayesian Layer (1 week)

6. **Build projection ingestion pipeline.**
   - Ingest Steamer/ZiPS from FanGraphs (resolve 403 issue or use CSV export).
   - Populate `player_projections` table.
   - Add `prior_source`, `shrinkage`, `data_quality_score` computation.

7. **Blend priors into rolling stats.**
   - `daily_rate = 0.6 × rolling + 0.4 × prior` (adjust weights by season month).

### Phase D: Add Signal Layer (1 week)

8. **Ingest Statcast x-stats into rolling window.**
   - Add `xwoba`, `barrel_pct`, `hard_hit_pct`, `xera`, `csw_pct` columns to `player_rolling_stats`.
   - Compute "luck-adjusted Z-scores" that blend raw and expected performance.

9. **Build buy/sell signal detector.**
   - `signal_gap = z_xstat - z_raw`.
   - Surface top 20 positive gaps (buy low) and top 20 negative gaps (sell high).

### Phase E: Process Discipline (ongoing)

10. **Populate `data_ingestion_logs`** with structured records.
11. **Build weekly backtest job** comparing recommendations to outcomes.
12. **Auto-tune `_lineup_score` weights** based on backtest MAE.

---

## 7. Bottom Line

**The platform's mathematical foundation is sound. Its decision-making layer is not.**

The rolling windows, Z-scores, and simulations are implemented correctly. The problem is that the **decision engine operates on a 3-factor heuristic from 2024** while **2026 modules with contextual awareness sit unused in the repository.**

**The highest-ROI fix is not adding new math — it's wiring existing math together.**

1. **Wire `smart_lineup_selector.py` into the daily pipeline.** (Biggest immediate impact)
2. **Wire `h2h_monte_carlo.py` into the scoreboard endpoint.** (Enables strategic decision-making)
3. **Implement `row_projector.py`.** (Unblocks the Monte Carlo)
4. **Ingest x-stats.** (Creates information asymmetry vs. casual managers)
5. **Add Bayesian priors.** (Stabilizes early-season projections)

After these changes, the platform will have a genuine competitive advantage. Until then, it is a data pipeline with a primitive decision layer on top.
