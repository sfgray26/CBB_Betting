# Phase 9 Research: Statcast Predictive Proxy Engine

**Date:** 2026-04-24  
**Researcher:** Kimi CLI (Deep Intelligence Unit)  
**Authority:** Proposed architecture — requires Claude Code approval before implementation  
**Scope:** Replace static proxy fallback for unknown/rookie players with dynamic Bayesian projections fueled by Statcast advanced metrics.

---

## 1. Executive Summary

The Fantasy Baseball platform currently suffers from a **critical proxy generation gap**: when a player is not on the hardcoded 200-player board AND not present in `player_projections`, `get_or_create_projection()` returns an empty proxy (`z_score=0.0`, empty `cat_scores`, empty `proj`). This causes:

- 21/25 waiver free agents scoring `need_score=0.0`
- Lineup optimizer receiving no stat bonuses for unknown players
- MCMC simulation treating unknown players as zero-value

**This report documents:**
1. Local codebase audit of existing Bayesian and Statcast infrastructure
2. Research on stabilization rates, Empirical Bayes, and Statcast translation models
3. A proposed `StatcastProxyEngine` architecture that bridges the gap
4. Concrete implementation steps with pseudocode and parameter tables

---

## 2. Local Codebase Audit

### 2.1 Existing Bayesian Infrastructure (`statcast_ingestion.py`)

A `BayesianProjectionUpdater` class already exists (lines 797–1046) with the following capabilities:

| Component | Status | Notes |
|-----------|--------|-------|
| `UpdatedProjection` dataclass | ✅ Implemented | Stores prior, likelihood, posterior, shrinkage, confidence intervals |
| `get_prior_projection()` | ✅ Implemented | Queries `player_projections` → falls back to `player_board` |
| `get_recent_performance()` | ✅ Implemented | Aggregates last 14 days from `statcast_performances`; requires ≥10 PA |
| `bayesian_update()` | ✅ Implemented | Conjugate normal update on **wOBA only** |
| `update_all_projections()` | ✅ Implemented | Loops all `statcast_performances` player_ids; stores results back to DB |
| `_store_updated_projection()` | ✅ Implemented | Upserts `PlayerProjection` row with `update_method='bayesian'` |

**Critical limitation:** The updater **requires a prior projection** to exist. If `get_prior_projection()` returns `None` (no `player_projections` row, no board match), the player is **skipped entirely**. This means unknown/rookie players with Statcast data receive no projection update.

### 2.2 Statcast Data Pipeline

The `StatcastIngestionAgent` (lines 324–794) fetches daily data from Baseball Savant and stores it in `statcast_performances`:

```
Baseball Savant CSV → _fetch_by_player_type() → _aggregate_to_daily()
→ transform_to_performance() → store_performances() → DB
```

**Stored metrics per player-day:**
- Counting stats: `pa`, `ab`, `h`, `hr`, `r`, `rbi`, `bb`, `so`, `sb`, `cs`
- Quality metrics: `exit_velocity_avg`, `launch_angle_avg`, `hard_hit_pct`, `barrel_pct`
- Expected stats: `xba`, `xslg`, `xwoba`, `woba`
- Pitcher stats: `ip`, `er`, `k_pit`, `bb_pit`, `pitches`

**Current DB state:** 11,230 rows in `statcast_performances`.

### 2.3 Proxy Generation Path (`player_board.py`)

The `get_or_create_projection()` function is the **critical chokepoint**:

```
Yahoo API Player
    ├── Runtime cache check
    ├── Exact name match on hardcoded board (200 players)
    ├── Fuzzy name match (90% similarity)
    └── DB fallback:
        ├── Yahoo ID → PlayerIDMapping → mlbam_id
        ├── mlbam_id → PlayerProjection
        └── If ANY step fails → return EMPTY PROXY
```

**Empty proxy definition (current):**
```python
proxy = {
    "id": player_key or name.lower().replace(" ", "_"),
    "name": name, "team": ..., "positions": positions,
    "type": player_type, "tier": 10, "rank": 9999, "adp": 9999.0,
    "z_score": 0.0,
    "cat_scores": {},      # ← EMPTY
    "proj": {},            # ← EMPTY
    "is_proxy": True,
}
```

**Root cause of zero-need-score FAs:** 21/25 free agents are completely absent from `player_projections`. The DB lookup path fails. Yahoo ID mapping also fails (0/10,000 rows have `yahoo_id`). The proxy falls back to empty.

### 2.4 The Disconnect

```
┌─────────────────────┐         ┌─────────────────────┐
│  statcast_performances│         │  player_projections  │
│  (11,230 rows)        │         │  (628 rows)          │
│  Has xwOBA, Barrel%,  │   X     │  Has Steamer/ZiPS    │
│  Exit Velocity, etc.  │────────→│  priors for known    │
│                       │  NO BRIDGE  │  players only       │
└─────────────────────┘         └─────────────────────┘
           │
           │  (data exists but never queried for unknowns)
           ▼
    ┌──────────────┐
    │ get_or_create_projection()  ← returns EMPTY for unknowns
    └──────────────┘
```

---

## 3. State-of-the-Art Research Findings

### 3.1 Stabilization Rates & Sample Size (Russell Carleton / FanGraphs)

Understanding when stats become "real" is essential for weighting early-season data.

**Batter stabilization points (PA or BIP):**

| Statistic | Stabilization Point | Source |
|-----------|---------------------|--------|
| K% | ~60 PA | Carleton 2007 |
| BB% | ~120 PA | Carleton 2007 |
| HR rate | ~170 PA | Carleton 2007 |
| OBP | ~460 PA | Carleton 2007 |
| SLG | ~320 AB | Carleton 2007 |
| ISO | ~160 AB | Carleton 2007 |
| GB/FB rate | ~80 BIP | Carleton 2007 |
| **Barrel%** | ~45–50 BBE (~15–20 games) | Freeze 2019 / Statcast literature |
| **Exit Velocity** | ~45–50 BBE | Freeze 2019 |
| **xwOBA** | ~100–150 BBE | Industry consensus |

**Key insight:** Statcast process metrics (Barrel%, Exit Velocity, xwOBA) stabilize **much faster** than outcome stats (AVG, OBP, SLG). A rookie with 3 weeks of MLB data (~60–80 PA, ~40–50 BBE) has **meaningful** Barrel% and Exit Velocity signals, even though their batting average is still noisy.

### 3.2 Empirical Bayes & Shrinkage

Empirical Bayes estimates outperform raw mid-season batting averages for predicting end-of-season performance (Brill 2023, Ryan 2024). The core formula:

```
posterior_mean = (prior_precision × prior_mean + likelihood_precision × sample_mean) / (prior_precision + likelihood_precision)
shrinkage = prior_precision / posterior_precision
```

Where:
- `prior_precision = 1 / prior_variance`
- `likelihood_precision = sample_size / sample_variance`

For **players WITHOUT priors** (rookies, unknowns), we must construct a **population prior** — the league-average distribution of the metric — and shrink the player's small sample toward that.

### 3.3 Statcast → Counting Stats Translation Models

**xwOBA to wOBA mapping:**
- xwOBA is more predictive of future wOBA than current wOBA (MLB.com, Sharpe 2019)
- xwOBA ≈ 0.320 is league average
- Each 0.010 xwOBA ≈ ~5 wRAA over 600 PA

**Barrel% to HR/FB and ISO:**
- Barrel% strongly correlates with ISO and HR/FB
- Elite: >12%, Good: 8%, Average: 5%, Poor: <3% (per `advanced_metrics.py` thresholds)
- A player with 15% Barrel% and 600 PA projects to ~35–40 HR (depending on FB%)

**Exit Velocity to AVG/SLG:**
- 92+ mph avg EV → above-average slugging
- 88–90 mph avg EV → league-average contact
- <85 mph avg EV → well below average

### 3.4 Minor League Equivalency (MLE) for True Rookies

For players with **no MLB data at all** (pure rookies called up from minors):

- AAA-to-MLB translation: ~18% offensive ability loss (Bill James / Dan Szymborski)
- MLE formula (simplified): `MLE_stat = Minor_stat × √m × PM` where `m ≈ 0.82` for AAA
- Chain from AA → AAA → MLB using successive league factors
- **Important caveat:** MLEs are translations, not projections. They tell you "what would this have been in MLB?" not "what will this player do next year?"

For the proxy engine, MLEs are most useful when we have minor league Statcast data (Arizona Fall League, Triple-A) and need a starting prior before MLB data accumulates.

---

## 4. Proposed Architecture: `StatcastProxyEngine`

### 4.1 Design Goal

Create a new module `backend/fantasy_baseball/statcast_proxy_engine.py` that:

1. Queries `statcast_performances` for players absent from `player_projections`
2. Aggregates their Statcast metrics over a rolling window (last 7–30 days)
3. Translates those metrics into **synthetic counting stats** (HR, R, RBI, AVG, OPS, etc.)
4. Computes **synthetic z-scores** using the same pool standard deviations as `cat_scores_builder.py`
5. Returns a populated proxy dict compatible with `get_or_create_projection()`

### 4.2 Module Interface

```python
class StatcastProxyEngine:
    """
    Generates synthetic projections for unknown/rookie players
    using Statcast batted-ball data and Empirical Bayes shrinkage.
    """

    def __init__(self, db: Session):
        self.db = db
        self.bayesian_updater = BayesianProjectionUpdater()

    def get_proxy_projection(self, player_name: str, yahoo_id: str = None,
                            positions: list = None) -> dict:
        """
        Main entry point. Returns a projection dict compatible with
        get_or_create_projection().

        Strategy:
        1. Check if player exists in player_projections (fast path)
        2. Query statcast_performances for recent data
        3. If data exists → build Bayesian proxy
        4. If no data → return population-prior proxy (better than empty)
        """

    def _build_batter_proxy(self, player_id: str, recent_data: list) -> dict:
        """
        Translate Statcast metrics to synthetic counting stats.
        Uses regression models calibrated on 2024–2025 data.
        """

    def _build_pitcher_proxy(self, player_id: str, recent_data: list) -> dict:
        """
        Translate pitcher Statcast (velocity, whiff%, xwOBA allowed)
        to synthetic pitching stats.
        """

    def _compute_synthetic_cat_scores(self, proj: dict, player_type: str) -> dict:
        """
        Compute z-scores against the current player pool using
        cat_scores_builder's BATTER_WEIGHTS / PITCHER_WEIGHTS.
        """
```

### 4.3 Batter Translation Model (Pseudocode)

```python
def _translate_statcast_to_counting(self, metrics: StatcastMetrics) -> dict:
    """
    Translates batted-ball quality into expected counting stats
    over a 600-PA season.
    """
    # --- Inputs ---
    pa_observed = metrics.total_pa          # e.g., 45 PA
    xwoba = metrics.weighted_xwoba          # e.g., 0.355
    barrel_pct = metrics.weighted_barrel_pct # e.g., 0.086 (8.6%)
    ev = metrics.weighted_exit_velocity     # e.g., 89.5 mph
    hard_hit_pct = metrics.weighted_hard_hit_pct
    bb_rate_observed = metrics.bb / metrics.pa
    k_rate_observed = metrics.so / metrics.pa

    # --- Step 1: Population priors (league average) ---
    league_avg_xwoba = 0.320
    league_avg_barrel_pct = 0.055
    league_avg_ev = 88.4
    league_avg_bb_rate = 0.085
    league_avg_k_rate = 0.220

    # --- Step 2: Shrinkage weights by sample size ---
    # Barrel% stabilizes ~50 BBE; xwOBA ~100 BBE; BB/K ~60–120 PA
    bbe = metrics.total_batted_balls
    shrinkage_barrel = 50 / (50 + bbe) if bbe > 0 else 1.0
    shrinkage_xwoba = 100 / (100 + bbe) if bbe > 0 else 1.0
    shrinkage_bb = 120 / (120 + pa_observed)
    shrinkage_k = 60 / (60 + pa_observed)

    # --- Step 3: Posterior estimates (shrunk toward league) ---
    posterior_barrel_pct = (
        shrinkage_barrel * league_avg_barrel_pct +
        (1 - shrinkage_barrel) * barrel_pct
    )
    posterior_xwoba = (
        shrinkage_xwoba * league_avg_xwoba +
        (1 - shrinkage_xwoba) * xwoba
    )
    posterior_bb_rate = (
        shrinkage_bb * league_avg_bb_rate +
        (1 - shrinkage_bb) * bb_rate_observed
    )
    posterior_k_rate = (
        shrinkage_k * league_avg_k_rate +
        (1 - shrinkage_k) * k_rate_observed
    )

    # --- Step 4: Translate to counting stats (600 PA basis) ---
    # HR from Barrel% (regression: ~3.5 HR per 1% Barrel over 600 PA)
    projected_hr = max(5, posterior_barrel_pct * 100 * 3.5)

    # AVG from xwOBA (simplified: xwOBA ≈ 1.25×OPS - 0.08; invert for AVG)
    # Better: use xBA directly if available
    projected_avg = metrics.weighted_xba if metrics.weighted_xba > 0.150 else 0.250

    # R and RBI from xwOBA and team context
    # League avg: ~75 R, ~75 RBI per 600 PA for middle-of-order hitters
    # Scale by xwOBA / 0.320
    run_factor = posterior_xwoba / 0.320
    projected_r = 75 * run_factor
    projected_rbi = 72 * run_factor

    # TB from SLG estimate
    projected_slg = metrics.weighted_xslg if metrics.weighted_xslg > 0.300 else 0.400
    projected_tb = projected_slg * 550  # ~550 AB over 600 PA

    # SB from sprint speed (if available) or position/age heuristic
    projected_sb = self._estimate_sb(metrics)

    # K_bat from K%
    projected_k_bat = posterior_k_rate * 600

    return {
        "pa": 600, "ab": 550,
        "r": round(projected_r),
        "h": round(projected_avg * 550),
        "hr": round(projected_hr),
        "rbi": round(projected_rbi),
        "k_bat": round(projected_k_bat),
        "tb": round(projected_tb),
        "avg": round(projected_avg, 3),
        "ops": round(projected_slg + projected_avg + 0.070, 3),  # crude OBP estimate
        "nsb": projected_sb,
    }
```

### 4.4 Pitcher Translation Model (Pseudocode)

```python
def _translate_pitcher_statcast(self, metrics: StatcastMetrics) -> dict:
    """
    Translate pitcher quality metrics to synthetic stats.
    Key inputs: xwOBA allowed, K%, BB%, velocity, whiff%
    """
    # Population prior for pitchers
    league_avg_xwoba_allowed = 0.320
    league_avg_k_rate = 0.220
    league_avg_bb_rate = 0.085

    # Stabilization: K% ~70 BF, BB% ~170 BF
    bf = metrics.total_bf
    shrinkage_k = 70 / (70 + bf) if bf > 0 else 1.0
    shrinkage_bb = 170 / (170 + bf) if bf > 0 else 1.0

    posterior_k_rate = shrinkage_k * league_avg_k_rate + (1 - shrinkage_k) * metrics.k_rate
    posterior_bb_rate = shrinkage_bb * league_avg_bb_rate + (1 - shrinkage_bb) * metrics.bb_rate

    # ERA from xwOBA allowed (rough: xwOBA × 9 ≈ ERA-ish, but use xERA formula)
    # Simplified: elite xwOBA allowed (<0.280) → ERA ~3.00; poor (>0.360) → ERA ~5.00
    projected_era = 2.50 + (metrics.weighted_xwoba_allowed - 0.280) * 15.0
    projected_era = max(2.00, min(6.00, projected_era))

    # WHIP from BB% and hit rate (approximate)
    projected_whip = 1.00 + posterior_bb_rate * 2.0 + (metrics.weighted_xwoba_allowed - 0.300) * 2.0
    projected_whip = max(0.90, min(1.80, projected_whip))

    # K/9 from K% and innings
    projected_k9 = posterior_k_rate * 27.0  # ~27 BF per 9 IP

    # IP estimate (for starters vs relievers)
    projected_ip = 160 if "SP" in positions else 65

    # Wins/QS from ERA and IP (very rough)
    projected_w = round(projected_ip / 30) if projected_era < 4.00 else round(projected_ip / 40)
    projected_qs = round(projected_ip / 8) if projected_era < 4.00 else round(projected_ip / 12)

    return {
        "ip": projected_ip,
        "w": projected_w,
        "l": round(projected_ip / 35),  # rough
        "qs": projected_qs,
        "k_pit": round(projected_k9 * projected_ip / 9),
        "era": round(projected_era, 2),
        "whip": round(projected_whip, 2),
        "k9": round(projected_k9, 1),
        "hr_pit": round(projected_ip * 0.12),  # league avg ~1.2 HR/9
        "nsv": 0,  # closer detection is separate
    }
```

### 4.5 Integration with `get_or_create_projection()`

Modify the DB fallback path in `player_board.py`:

```python
def get_or_create_projection(yahoo_player: dict) -> dict:
    # ... existing cache and board lookup ...

    # NEW: Statcast proxy engine for unknown players
    try:
        from backend.fantasy_baseball.statcast_proxy_engine import StatcastProxyEngine
        engine = StatcastProxyEngine(db)
        proxy = engine.get_proxy_projection(
            player_name=name,
            yahoo_id=yahoo_id,
            positions=positions
        )
        if proxy and proxy.get("z_score", 0.0) != 0.0:
            return proxy
    except Exception as e:
        logger.warning("Statcast proxy engine failed for %s: %s", name, e)

    # FALLBACK: empty proxy (existing behavior)
    return _empty_proxy(name, player_key, positions)
```

---

## 5. Implementation Recommendations

### 5.1 Phase 1: Quick Win — Population-Prior Proxy (1–2 days)

Before building the full translation model, immediately improve the empty proxy by assigning **league-average z-scores** rather than zeros:

```python
# In get_or_create_projection() BEFORE the Statcast engine:
league_avg_proxy = {
    "z_score": -0.5,  # Slightly below average (replacement level)
    "cat_scores": {
        "r": -0.3, "h": -0.3, "hr": -0.3, "rbi": -0.3,
        "k_bat": 0.0, "tb": -0.3, "avg": -0.3, "ops": -0.3, "nsb": -0.1,
    },
    "proj": {"avg": 0.250, "ops": 0.720, "hr": 15, "r": 65, "rbi": 65, "nsb": 5},
}
```

This single change would fix the `need_score=0.0` problem for all 21 unknown FAs **immediately**, giving them plausible (replacement-level) values instead of complete emptiness.

### 5.2 Phase 2: Statcast-Aware Proxy Engine (3–5 days)

1. **Create `statcast_proxy_engine.py`** with the translation models above
2. **Add rolling aggregation** query to `statcast_performances` (last 14 days, weighted by PA)
3. **Calibrate translation coefficients** using the existing 11,230 rows:
   - Regress `barrel_pct` → `hr_per_600_pa` on known players
   - Regress `xwoba` → `r + rbi_per_600_pa`
   - Regress `xba` → `avg`
4. **Integrate into `get_or_create_projection()`** as the DB fallback
5. **Run backfill** for all current unknown FAs

### 5.3 Phase 3: Full Bayesian Pipeline (1–2 weeks)

1. Extend `BayesianProjectionUpdater` to handle **no-prior players** by using population priors
2. Add MLE support for true rookies with minor league data
3. Add daily automated run after `StatcastIngestionAgent`
4. Monitor proxy quality via data-quality dashboard

### 5.4 Key Parameters Table

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Barrel% → HR/600 PA factor | 3.5 | 10% Barrel% ≈ 35 HR (MLB avg ~3.3) |
| xwOBA → Run factor base | 0.320 | League-average wOBA |
| K% stabilization (batters) | 60 PA | Carleton 2007 |
| BB% stabilization (batters) | 120 PA | Carleton 2007 |
| Barrel% stabilization | 50 BBE | Freeze 2019 / Statcast |
| xwOBA stabilization | 100 BBE | Industry consensus |
| K% stabilization (pitchers) | 70 BF | Carleton 2007 |
| BB% stabilization (pitchers) | 170 BF | Carleton 2007 |
| League avg proxy z_score | -0.5 | Replacement level (12-team, 23-roster) |
| Shrinkage formula | `N / (N + sample)` | Simple reliability-weighted |

---

## 6. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Translation coefficients miscalibrated | Medium | Use 2024–2025 retrospective data to validate before deploying |
| Statcast data stale (no recent games) | Low | Engine gracefully falls back to population prior |
| Performance impact on waiver endpoint | Medium | Cache proxy results for 1 hour; DB query is indexed on `player_id` |
| Overvaluing small-sample rookies | Medium | Aggressive shrinkage (50 BBE → 50% trust in Barrel%) |
| Pitcher proxy harder than batter proxy | High | Defer pitcher proxy to Phase 3; use population prior for pitchers initially |

---

## 7. References

1. **Carleton, Russell** (2007). "Sample Size." *Baseball Prospectus*. Stabilization points for traditional stats.
2. **FanGraphs** (2017). "Sample Size." https://library.fangraphs.com/principles/sample-size/ — Updated reliability thresholds.
3. **Freeze, Michael** (2019). "Tracking systems derive metrics for which the stabilization rate... is 45 to 50 balls in play." Cited in York University thesis on baseball performance metrics.
4. **Szymborski, Dan** (c. 2000). "How to Calculate MLEs." *Baseball Think Factory*. AAA→MLB translation methodology.
5. **MacAree, Graham** (2010). "League Equivalencies." *FanGraphs Library*. Park and level adjustments.
6. **Brill, Ryan** (2023). "Empirical Bayes Estimates of End-of-Season Batting Averages." Demonstrates EB outperforms raw small-sample stats.
7. **Chatterjee, Rajit** (2026). "Stochastic Differential Equation Treatment of OPS in Baseball." *NHSJS*. SDE modeling with Statcast predictors.
8. **Sharpe, A.** (2019). MLB.com expected stats methodology. Forward-looking batted-ball outcomes.

---

## 8. Next Steps (Claude Code Decision Required)

1. **Approve Phase 1 quick win** (population-prior proxy) → fixes 21/25 zero-need-score FAs immediately
2. **Approve Phase 2 scope** (StatcastProxyEngine) → delegate implementation to Kimi CLI or implement directly
3. **Provide calibration data** → run retrospective regression on 2024–2025 `statcast_performances` to derive precise Barrel%→HR and xwOBA→R/RBI coefficients
4. **Update AGENTS.md** → add `statcast_proxy_engine.py` to Kimi CLI ownership if delegated

---

*Report compiled by Kimi CLI v1.17.0 | Context window: ~850K tokens consumed | Files read: 4 | Web sources: 8*
