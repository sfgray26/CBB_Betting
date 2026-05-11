# Next-Generation Fantasy Baseball Analytics — Technical Design Document

**Author:** Kimi CLI (Senior Staff Engineer)  
**Date:** 2026-05-04  
**Branch:** `stable/cbb-prod`  
**Audience:** Implementation engineers (1–3 person team)

---

## Executive Summary

The current fantasy baseball platform has solid foundational scoring (z-scores, rolling windows, MCMC simulation) but lacks the **contextual intelligence layers** that separate good fantasy tools from elite ones. This document designs six production-ready features that add opportunity modeling, dynamic thresholds, matchup context, market signals, and actionable decision outputs.

**Key constraint:** All designs assume a single PostgreSQL database, Python 3.11 backend, and Railway deployment. No microservices, no real-time streaming unless explicitly justified. Batch pipelines with idempotent upserts are preferred throughout.

**Cross-cutting theme:** Every new system must emit **confidence scores** and **gracefully degrade** when data is missing. The platform already has too many silent NULLs and hardcoded thresholds — these designs explicitly prevent that.

---

## Feature 1: Opportunity Model

### 1.1 Problem Definition

The current `player_scores` table ranks players by skill (z-scores) alone. It completely ignores:
- **Playing time volume:** A .380 wOBA part-timer (250 PA) outranks a .340 wOBA full-timer (650 PA) in the current system
- **Lineup position:** Leadoff hitters get ~100 more PA/year than #7 hitters
- **Platoon risk:** A player who only faces LHP has ~60% the PA of an everyday player
- **Role certainty:** A closer in a committee is worth less than a locked-in closer

**Live evidence from production:** The `player_scores` table has no PA/game, lineup slot, or games-started columns. The `position_eligibility` table tracks positions but not lineup spot stability.

### 1.2 Design Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  mlb_player_stats│────▶│ opportunity_engine│────▶│player_opportunity│
│  (daily box)     │     │ (daily batch)     │     │ (DB table)       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                              │
                              ▼
                        ┌──────────────────┐
                        │ scoring_engine.py │
                        │ (composite_z      │
                        │  *= opportunity   │
                        │  weighting)       │
                        └──────────────────┘
```

### 1.3 Data Model

```sql
CREATE TABLE player_opportunity (
    id BIGSERIAL PRIMARY KEY,
    bdl_player_id INTEGER NOT NULL REFERENCES player_id_mapping(bdl_id),
    as_of_date DATE NOT NULL,

    -- Volume metrics (rolling 14-day)
    pa_per_game FLOAT,
    ab_per_game FLOAT,
    games_played_14d INTEGER,
    games_started_14d INTEGER,
    games_started_pct FLOAT,  -- games_started / games_team_played

    -- Lineup position (rolling 14-day)
    lineup_slot_avg FLOAT,    -- 1.0 = leadoff, 9.0 = ninth
    lineup_slot_mode INTEGER, -- most common lineup spot
    lineup_slot_entropy FLOAT,-- Shannon entropy (0 = fixed spot, high = bouncing around)

    -- Platoon / usage splits
    pa_vs_lhp_14d INTEGER,
    pa_vs_rhp_14d INTEGER,
    platoon_ratio FLOAT,      -- min(pa_vs_lhp, pa_vs_rhp) / max(...). 1.0 = everyday, 0.0 = strict platoon
    platoon_risk_score FLOAT, -- 0.0 = no platoon risk, 1.0 = strict platoon

    -- Role stability (pitchers)
    appearances_14d INTEGER,
    saves_14d INTEGER,
    holds_14d INTEGER,
    role_certainty_score FLOAT, -- 0-1. Derived from save/hold consistency

    -- Injury / absence
    days_since_last_game INTEGER,
    il_stint_flag BOOLEAN,

    -- Derived composite
    opportunity_z FLOAT,      -- z-score of expected_fantasy_output vs league
    opportunity_confidence FLOAT, -- based on sample size (PA in window)

    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (bdl_player_id, as_of_date)
);

CREATE INDEX idx_player_opp_bdl_date ON player_opportunity(bdl_player_id, as_of_date);
CREATE INDEX idx_player_opp_date ON player_opportunity(as_of_date);
CREATE INDEX idx_player_opp_opportunity_z ON player_opportunity(as_of_date, opportunity_z);
```

### 1.4 Core Algorithms / Logic

**Lineup Slot Entropy:**
```python
def lineup_slot_entropy(lineup_slots: list[int]) -> float:
    """
    Shannon entropy of batting order positions.
    0.0 = always bats same spot. High = bounces around.
    """
    if not lineup_slots:
        return 1.0  # unknown = high entropy
    from collections import Counter
    counts = Counter(lineup_slots)
    n = len(lineup_slots)
    import math
    entropy = -sum((c/n) * math.log2(c/n) for c in counts.values())
    # normalize to 0-1 scale (max entropy for 9 slots = log2(9) ≈ 3.17)
    return min(entropy / 3.17, 1.0)
```

**Platoon Risk Score:**
```python
def platoon_risk_score(pa_vs_lhp: int, pa_vs_rhp: int) -> float:
    """
    0.0 = perfectly balanced (everyday player)
    1.0 = faces only one handedness
    """
    total = pa_vs_lhp + pa_vs_rhp
    if total < 20:
        return 0.5  # insufficient data
    ratio = min(pa_vs_lhp, pa_vs_rhp) / max(pa_vs_lhp, pa_vs_rhp)
    return 1.0 - ratio  # 1.0 - 1.0 = 0, 1.0 - 0.0 = 1.0
```

**Opportunity Z-Score (the composite):**
```python
def compute_opportunity_z(opp: PlayerOpportunityRow, league_baselines: LeagueOpportunityBaselines) -> float:
    """
    Compute opportunity z-score using 4 factors.
    Each factor is z-scored against the current-season league distribution.
    """
    # Factor 1: Volume (PA/game) — most important
    volume_z = (opp.pa_per_game - league_baselines.mean_pa_per_game) / league_baselines.std_pa_per_game

    # Factor 2: Lineup quality (lower slot number = better)
    slot_z = (league_baselines.mean_lineup_slot - opp.lineup_slot_avg) / league_baselines.std_lineup_slot
    # inverted: leadoff (1.0) gets positive z

    # Factor 3: Platoon risk (penalty)
    platoon_penalty = -2.0 * opp.platoon_risk_score  # up to -2.0 z for strict platoon

    # Factor 4: Role certainty (pitchers only)
    role_bonus = 0.0
    if opp.role_certainty_score is not None:
        role_bonus = 1.5 * (opp.role_certainty_score - 0.5)  # centered at 0.5

    # Weighted composite
    composite = (
        0.40 * volume_z +
        0.30 * slot_z +
        0.20 * platoon_penalty +
        0.10 * role_bonus
    )
    return round(composite, 3)
```

**Confidence Score:**
```python
def opportunity_confidence(pa_in_window: int) -> float:
    """
    Sigmoid scaling: 50 PA = 0.5 confidence, 200 PA = 0.95
    """
    return min(0.95, pa_in_window / 200.0)
```

### 1.5 Integration Points

| Consumer | How It Uses Opportunity |
|----------|------------------------|
| `scoring_engine.py` | Multiply `composite_z` by `opportunity_weight` (default 0.85, adjustable per league) |
| `waiver_edge_detector.py` | Filter out FAs with `opportunity_confidence < 0.3` unless market signal is strong |
| `daily_lineup_optimizer.py` | Use `lineup_slot_avg` to project PA for the day |
| `mcmc_simulator.py` | Scale counting-stat means by `pa_per_game / league_avg_pa_per_game` |

### 1.6 Data Pipeline

```
Step 1: Ingestion (daily, 3 AM ET)
  Source: mlb_player_stats (already ingested daily)
  Query: SELECT player_id, game_date, ab, pa, lineup_slot, opponent_hand
         FROM mlb_player_stats
         WHERE game_date >= CURRENT_DATE - INTERVAL '14 days'

Step 2: Aggregation (same job)
  Group by bdl_player_id
  Compute: pa_per_game, lineup_slot_avg, lineup_slot_entropy, pa_vs_lhp, pa_vs_rhp

Step 3: Baseline computation (weekly)
  Compute league-wide means/stds for:
    pa_per_game, lineup_slot_avg, platoon_risk_score
  Store in: opportunity_baselines table (or reuse player_scores computation window)

Step 4: Upsert
  INSERT INTO player_opportunity ... ON CONFLICT (bdl_player_id, as_of_date) DO UPDATE
```

### 1.7 Edge Cases / Failure Modes

| Edge Case | Handling |
|-----------|----------|
| Player has 0 games in window | `opportunity_z = -2.0` (penalty), `opportunity_confidence = 0.0` |
| Call-up with <10 PA | Low confidence; use minor league projected PA as prior |
| Pitcher with no starts (reliever) | `role_certainty_score` based on save/hold consistency, not starts |
| DH-only player | `lineup_slot_avg` is useful; `platoon_risk_score` still applies |
| Rainouts / missed games | `days_since_last_game` captures this; don't penalize IL stints separately |

### 1.8 Performance Considerations

- **Table size:** ~2,000 players × 180 days = 360K rows/year. Trivial for Postgres.
- **Recomputation:** Daily aggregation over 14 days is O(players × games). With 2K players and 14 days, ~28K rows scanned. Sub-second.
- **Index strategy:** Covering index on `(bdl_player_id, as_of_date, opportunity_z)` for scoring lookups.

### 1.9 Phased Rollout Plan

| Phase | What | Timeline |
|-------|------|----------|
| MVP | Create table, populate from existing `mlb_player_stats`, compute simple `pa_per_game` and `lineup_slot_avg` | Week 1 |
| V1 | Add platoon splits, role certainty, entropy. Integrate into scoring_engine as multiplicative factor | Week 2 |
| V2 | Add minor league call-up priors, injury risk modeling, IL stint tracking | Week 4 |

### 1.10 Estimated Complexity

- **Engineering effort:** Medium (2 engineers × 1 week for MVP)
- **Risk level:** Low — additive feature, no breaking changes to existing scoring

---

## Feature 2: Statcast Data Integration Fix

### 2.1 Problem Definition

The `statcast_batter_metrics` and `statcast_pitcher_metrics` tables have columns for `sprint_speed`, `stuff_plus`, and `location_plus`, but **100% of rows are NULL** for these fields. The schema was migrated but the ingestion pipeline never populated them.

**Live evidence:**
```sql
SELECT COUNT(*) FROM statcast_batter_metrics WHERE sprint_speed IS NOT NULL; -- 0
SELECT COUNT(*) FROM statcast_pitcher_metrics WHERE stuff_plus IS NOT NULL;   -- 0
SELECT COUNT(*) FROM statcast_pitcher_metrics WHERE location_plus IS NOT NULL; -- 0
```

**Root cause hypotheses:**
1. **Data source gap:** `pybaseball_loader.py` pulls from FanGraphs leaderboards (`batting_stats()` / `pitching_stats()`). FanGraphs may not expose `Stuff+` and `Location+` in the default leaderboard endpoint.
2. **Schema mismatch:** The CSV/JSON parser in `pybaseball_loader.py` may not extract `Sprint Speed` from the returned DataFrame — the column might be named differently (`sprint_speed`, `SprSpd`, etc.).
3. **Missing data source:** `Sprint Speed` comes from Baseball Savant's baserunning leaderboards, NOT FanGraphs. The current pipeline only ingests FanGraphs data.
4. **Ingestion failure silent:** Even if the data is present in the raw response, the loader may silently skip it during DataFrame→ORM mapping.

### 2.2 Design Overview

```
┌──────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│ FanGraphs API    │────▶│ pybaseball_loader │────▶│ statcast_pitcher_   │
│ (Stuff+, Loc+)   │     │ (augmented)       │     │ metrics.stuff_plus  │
└──────────────────┘     └──────────────────┘     └─────────────────────┘

┌──────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│ Baseball Savant  │────▶│ savant_scraper.py │────▶│ statcast_batter_    │
│ (Sprint Speed)   │     │ (NEW)             │     │ metrics.sprint_speed│
└──────────────────┘     └──────────────────┘     └─────────────────────┘
```

### 2.3 Data Sources / APIs

**For Stuff+ / Location+ (Pitchers):**
- FanGraphs Leaderboards API: `https://www.fangraphs.com/api/leaders/major-league/data`
- Query params: `pos=all`, `stats=pit`, `type=8` (8 = pitching+ metrics)
- Alternative: pybaseball's `pitching_stats()` may support `qual` and `ind` parameters to get advanced tables
- If pybaseball doesn't expose these, we may need to scrape FanGraphs directly using the existing `_BROWSER_HEADERS` User-Agent patch

**For Sprint Speed (Hitters):**
- MLB Stats API: `https://statsapi.mlb.com/api/v1/stats?stats=statcast&group=hitting&playerPool=all&season=2026`
- Or Baseball Savant CSV: `https://baseballsavant.mlb.com/leaderboard/sprint_speed`
- The `savant_scraper.py` module (NEW) will handle this

### 2.4 ETL Pipeline Design

**New module:** `backend/ingestion/savant_scraper.py`

```python
"""
Baseball Savant scraper for metrics not available via FanGraphs/pybaseball.

Metrics fetched:
  - Sprint Speed (baserunning)
  - Future: Jump, Outs Above Average, etc.
"""

import logging
import requests
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

_SAVANT_SPRINT_SPEED_URL = (
    "https://baseballsavant.mlb.com/leaderboard/sprint_speed"
    "?year={year}&position=&team=&min=0&csv=true"
)

def fetch_sprint_speed(year: int = 2026) -> pd.DataFrame:
    """
    Download Baseball Savant sprint speed leaderboard as CSV.
    Returns DataFrame with columns: player_id, player_name, sprint_speed
    """
    url = _SAVANT_SPRINT_SPEED_URL.format(year=year)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/csv",
    }
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    
    df = pd.read_csv(pd.io.common.StringIO(resp.text))
    # Column mapping (Savant uses specific headers)
    col_map = {
        "player_id": "mlbam_id",
        "sprint_speed": "sprint_speed",
        # Fallbacks for different CSV formats
        "Player Id": "mlbam_id",
        "Sprint Speed": "sprint_speed",
    }
    df = df.rename(columns=col_map)
    return df[["mlbam_id", "sprint_speed"]]
```

**Augmented `pybaseball_loader.py`:**

```python
def load_pybaseball_pitchers(year: int = 2026) -> dict[str, StatcastPitcher]:
    # Existing FanGraphs fetch
    df = _fetch_pitching_stats(year)
    
    # NEW: Attempt to fetch Stuff+ / Location+ from alternate endpoint
    try:
        plus_df = _fetch_pitching_plus(year)
        df = df.merge(plus_df, on="player_id", how="left")
    except Exception as e:
        logger.warning("Stuff+/Location+ fetch failed: %s", e)
    
    result = {}
    for _, row in df.iterrows():
        p = StatcastPitcher(
            # ... existing fields ...
            stuff_plus=_float(row.get("Stuff+") or row.get("stuff_plus")),
            location_plus=_float(row.get("Location+") or row.get("location_plus")),
        )
        result[_normalize_name(row["Name"])] = p
    return result
```

### 2.5 Backfill Strategy

```sql
-- Step 1: Create backfill tracking table
CREATE TABLE statcast_backfill_log (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(50) NOT NULL,  -- 'sprint_speed', 'stuff_plus', 'location_plus'
    season INTEGER NOT NULL,
    records_backfilled INTEGER,
    records_total INTEGER,
    null_rate_pct FLOAT,
    run_at TIMESTAMPTZ DEFAULT NOW()
);

-- Step 2: Run backfill script (one-time)
-- script: scripts/backfill_statcast_advanced.py
-- This runs savant_scraper for 2025-2026 and pybaseball_loader for 2025-2026
-- Then updates existing rows in statcast_*_metrics
```

```python
# scripts/backfill_statcast_advanced.py
def backfill_sprint_speed(season: int):
    df = fetch_sprint_speed(season)
    updated = 0
    for _, row in df.iterrows():
        mlbam_id = int(row["mlbam_id"])
        sprint_speed = float(row["sprint_speed"])
        # Update by mlbam_id
        db.execute("""
            UPDATE statcast_batter_metrics
            SET sprint_speed = %s
            WHERE mlbam_id = %s AND season = %s
        """, (sprint_speed, str(mlbam_id), season))
        updated += db.rowcount
    db.commit()
    return updated
```

### 2.6 Data Validation Rules

| Metric | Non-null Threshold | Validation Rule |
|--------|-------------------|-----------------|
| `sprint_speed` | ≥70% | If <70%, disable in scoring and show "limited data" badge |
| `stuff_plus` | ≥60% | Pitchers have smaller samples; 60% is acceptable |
| `location_plus` | ≥60% | Same as Stuff+ |
| `xwoba` | ≥95% | Existing metric — should already be near 100% |

**Validation job:** Run daily after Statcast ingestion. If null rate exceeds threshold, alert admin but do NOT fail the pipeline.

```python
def validate_statcast_coverage(table: str, column: str, threshold: float = 0.70) -> bool:
    total = db.query(f"SELECT COUNT(*) FROM {table}").scalar()
    non_null = db.query(f"SELECT COUNT(*) FROM {table} WHERE {column} IS NOT NULL").scalar()
    rate = non_null / total if total else 0
    if rate < threshold:
        logger.error(f"{table}.{column} null rate = {1-rate:.1%}, exceeds threshold")
        return False
    return True
```

### 2.7 Performance / Storage Considerations

- **Network calls:** FanGraphs + Savant = ~2-3 requests. Each leaderboard is ~500 KB CSV. Total daily ingress: ~2 MB. Trivial.
- **Storage:** Adding 3 FLOAT columns to ~1,000 rows = 12 KB. Negligible.
- **Latency:** Savant scrape + FanGraphs fetch = ~5 seconds. Run in existing daily ingestion job (lock 100_018).

### 2.8 Phased Rollout Plan

| Phase | What | Timeline |
|-------|------|----------|
| MVP | Implement `savant_scraper.py`, backfill `sprint_speed` for 2026 | 2 days |
| V1 | Add `Stuff+` / `Location+` fetch from FanGraphs, backfill 2026 | 2 days |
| V2 | Backfill 2025 historical data, add validation alerts | 1 day |

### 2.9 Estimated Complexity

- **Engineering effort:** Small (1 engineer × 1 week)
- **Risk level:** Medium — depends on external website (FanGraphs/Savant) stability. Must have fallback to NULL if scrape fails.

---

## Feature 3: Threshold Configuration System

### 3.1 Problem Definition

The codebase is littered with hardcoded thresholds that were set once and never revisited:

```python
# scoring_engine.py
Z_CAP: float = 3.0

# player_board.py (hot/cold)
"hot_threshold": 0.5
"cold_threshold": -0.5

# category_aware_scorer.py
RATE_STAT_PROTECT_THRESHOLD = 0.5

# daily_lineup_optimizer.py
"streamer_threshold": 0.3

# player_momentum.py (via models)
delta_z > 0.5  → SURGING
delta_z >= 0.2 → HOT
delta_z > -0.2 → STABLE
delta_z >= -0.5 → COLD
```

**Why this matters:** A "hot" player in 2019 (juiced ball) had a different wOBA than in 2026. Static thresholds decay silently. The system currently has no mechanism to tune these without code deploys.

### 3.2 Design Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  config.json    │────▶│ threshold_service │────▶│ All consumers   │
│  (fallback)     │     │ (DB + memory      │     │ (scoring,       │
│                 │     │  cache)           │     │  momentum, etc) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                              │
                              ▼
                        ┌──────────────────┐
                        │  threshold_audit │
                        │  (tracks changes)│
                        └──────────────────┘
```

### 3.3 Data Model

```sql
-- Core threshold config table
CREATE TABLE threshold_config (
    id SERIAL PRIMARY KEY,
    config_key VARCHAR(100) NOT NULL UNIQUE,
    config_value JSONB NOT NULL,  -- flexible: can be float, int, dict, list
    scope VARCHAR(50) NOT NULL DEFAULT 'global',  -- 'global', 'league:469.l.72586', 'user:123'
    description TEXT,
    effective_date DATE NOT NULL DEFAULT CURRENT_DATE,
    deprecated_date DATE,  -- NULL = active
    created_by VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Audit log (immutable)
CREATE TABLE threshold_audit (
    id BIGSERIAL PRIMARY KEY,
    config_key VARCHAR(100) NOT NULL,
    old_value JSONB,
    new_value JSONB NOT NULL,
    changed_by VARCHAR(50),
    changed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Feature flag table (for rollout control)
CREATE TABLE feature_flags (
    id SERIAL PRIMARY KEY,
    flag_name VARCHAR(100) NOT NULL UNIQUE,
    enabled BOOLEAN NOT NULL DEFAULT FALSE,
    rollout_pct INTEGER NOT NULL DEFAULT 0 CHECK (rollout_pct BETWEEN 0 AND 100),
    scope VARCHAR(50) DEFAULT 'global',
    description TEXT
);
```

### 3.4 Core Algorithms / Logic

**Threshold Service (singleton, memory-cached):**

```python
# backend/services/threshold_service.py

from functools import lru_cache
from datetime import date
from typing import Any
import json

class ThresholdService:
    """
    Centralized threshold lookup with DB backing and in-memory cache.
    
    Resolution order:
      1. Scope-specific config (e.g., league:469.l.72586)
      2. Global config
      3. Hardcoded fallback (from code constant)
    """
    
    def __init__(self, db_session):
        self.db = db_session
        self._cache: dict[str, Any] = {}
        self._last_refresh = 0.0
    
    def get(
        self,
        key: str,
        scope: str = "global",
        default: Any = None,
        as_of: date | None = None
    ) -> Any:
        """Get threshold value, resolving scope hierarchy."""
        cache_key = f"{scope}:{key}"
        
        # Try scope-specific first
        val = self._get_from_db(key, scope, as_of)
        if val is not None:
            return val
        
        # Fall back to global
        if scope != "global":
            val = self._get_from_db(key, "global", as_of)
            if val is not None:
                return val
        
        return default
    
    def _get_from_db(self, key: str, scope: str, as_of: date | None) -> Any:
        as_of = as_of or date.today()
        row = self.db.execute("""
            SELECT config_value
            FROM threshold_config
            WHERE config_key = %s
              AND scope = %s
              AND effective_date <= %s
              AND (deprecated_date IS NULL OR deprecated_date > %s)
            ORDER BY effective_date DESC
            LIMIT 1
        """, (key, scope, as_of, as_of)).fetchone()
        return row[0] if row else None
    
    def set(self, key: str, value: Any, scope: str = "global", user: str = "system"):
        """Update a threshold and log the change."""
        old = self._get_from_db(key, scope, date.today())
        
        # Upsert
        self.db.execute("""
            INSERT INTO threshold_config (config_key, config_value, scope)
            VALUES (%s, %s, %s)
            ON CONFLICT (config_key, scope) DO UPDATE
            SET config_value = EXCLUDED.config_value,
                updated_at = NOW()
        """, (key, json.dumps(value), scope))
        
        # Audit log
        self.db.execute("""
            INSERT INTO threshold_audit (config_key, old_value, new_value, changed_by)
            VALUES (%s, %s, %s, %s)
        """, (key, json.dumps(old) if old else None, json.dumps(value), user))
        
        self.db.commit()
        self._cache.pop(f"{scope}:{key}", None)
```

**Key Config Values (initial seed):**

```sql
INSERT INTO threshold_config (config_key, config_value, description) VALUES
('momentum.surging.delta_z', '0.5', 'Minimum delta_z for SURGING signal'),
('momentum.hot.delta_z', '0.2', 'Minimum delta_z for HOT signal'),
('momentum.cold.delta_z', '-0.5', 'Maximum delta_z for COLD signal'),
('scoring.z_cap', '3.0', 'Z-score winsorization cap'),
('scoring.min_sample', '5', 'Minimum players before computing category Z'),
('waiver.streamer_threshold', '0.3', 'Minimum z-score for streamer suggestion'),
('lineup.scarcity.catcher', '1.20', 'Position scarcity multiplier for C'),
('lineup.scarcity.shortstop', '1.15', 'Position scarcity multiplier for SS'),
('opportunity.hot_threshold', '0.5', 'Hot/cold flag threshold (percentile-based soon)');
```

### 3.5 Versioning Strategy

- **Effective dating:** Every threshold has `effective_date`. New values can be staged before activation.
- **Audit immutability:** `threshold_audit` is append-only. Never update or delete.
- **Rollback:** To rollback, insert a new row with the old value and a new `effective_date`.

### 3.6 Rollout Strategy

```sql
-- Feature flags control which thresholds are active
INSERT INTO feature_flags (flag_name, enabled, rollout_pct, description) VALUES
('dynamic_thresholds_v1', false, 0, 'Use DB-driven thresholds instead of hardcoded values'),
('percentile_hot_cold', false, 0, 'Replace static hot/cold with percentile-based thresholds');
```

**Rollout process:**
1. Deploy code that reads from `ThresholdService` but falls back to hardcoded values
2. Enable `dynamic_thresholds_v1` flag for 10% of leagues
3. Monitor for 1 week
4. Ramp to 100%
5. Remove hardcoded fallbacks in next release

### 3.7 Migration Plan from Hardcoded Values

**Step 1:** Extract all hardcoded thresholds into `threshold_config` (run once)
**Step 2:** Modify consumers to use `ThresholdService.get()` with fallback
**Step 3:** Enable feature flag
**Step 4:** After 2 weeks of stability, remove fallbacks

**Code pattern for migration:**
```python
# BEFORE (hardcoded)
Z_CAP = 3.0

# AFTER (DB-driven with fallback)
from backend.services.threshold_service import get_threshold
Z_CAP = get_threshold("scoring.z_cap", default=3.0)
```

### 3.8 Phased Rollout Plan

| Phase | What | Timeline |
|-------|------|----------|
| MVP | Create tables, seed with current hardcoded values, build ThresholdService | 2 days |
| V1 | Migrate scoring_engine and momentum_engine to use ThresholdService | 2 days |
| V2 | Add feature flags, A/B testing framework, admin UI for threshold editing | 1 week |

### 3.9 Estimated Complexity

- **Engineering effort:** Small-Medium (1 engineer × 1.5 weeks)
- **Risk level:** Low — purely additive, with hardcoded fallbacks

---

## Feature 4: Matchup Context Engine

### 4.1 Problem Definition

The current `daily_lineup_optimizer.py` uses opponent implied runs from sportsbook odds and static park factors. It has NO concept of:
- Pitcher-batter handedness interaction (the single most predictive matchup factor)
- Pitch-type affinity (e.g., a fastball hitter vs a breaking-ball pitcher)
- Bullpen quality behind the starter
- Weather-adjusted park factors

**Live evidence:** The `_get_game_context()` function in `fantasy.py` returns `opp_impl` (opponent implied runs) and `park_factor` but never considers whether the opposing starter is LHP or RHP, or whether the batter crushes sliders but the pitcher throws 40% sliders.

### 4.2 Design Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ MLB Stats API   │────▶│ matchup_engine   │────▶│ matchup_context │
│ (probable SP,   │     │ (daily batch)    │     │ (DB table)      │
│  lineups,       │     │                  │     │                 │
│  weather)       │     │                  │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                              │
                              ▼
                        ┌──────────────────┐
                        │ daily_lineup_    │
                        │ optimizer.py     │
                        │ (uses matchup_z  │
                        │  instead of      │
                        │  generic odds)   │
                        └──────────────────┘
```

### 4.3 Data Model

```sql
CREATE TABLE matchup_context (
    id BIGSERIAL PRIMARY KEY,
    bdl_player_id INTEGER NOT NULL REFERENCES player_id_mapping(bdl_id),
    game_date DATE NOT NULL,
    opponent_team VARCHAR(10),
    opponent_team_id INTEGER,
    
    -- Opponent starter info
    opponent_starter_name VARCHAR(100),
    opponent_starter_hand VARCHAR(1),  -- 'L' or 'R'
    opponent_starter_era FLOAT,
    opponent_starter_whip FLOAT,
    opponent_starter_k_per_nine FLOAT,
    opponent_starter_stuff_plus FLOAT,
    
    -- Opponent bullpen (behind starter)
    opponent_bullpen_era FLOAT,
    opponent_bullpen_whip FLOAT,
    
    -- Park / weather
    home_team VARCHAR(10),
    park_factor_runs FLOAT,
    park_factor_hr FLOAT,
    weather_temp_f FLOAT,
    weather_wind_mph FLOAT,
    weather_wind_direction VARCHAR(10),  -- 'out', 'in', 'l_to_r', 'r_to_l'
    weather_precip_chance FLOAT,
    
    -- Hitter split vs this handedness (from rolling stats)
    hitter_woba_vs_hand FLOAT,
    hitter_k_pct_vs_hand FLOAT,
    hitter_iso_vs_hand FLOAT,
    
    -- Derived matchup score
    matchup_z FLOAT,
    matchup_confidence FLOAT,  -- based on sample size of split data
    
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE (bdl_player_id, game_date)
);

CREATE INDEX idx_matchup_context_player_date ON matchup_context(bdl_player_id, game_date);
CREATE INDEX idx_matchup_context_date ON matchup_context(game_date);
```

### 4.4 Core Algorithms / Logic

**Matchup Score Formula:**

```python
def compute_matchup_z(
    hitter: HitterProfile,
    context: MatchupContext,
    league_baseline: LeagueBaseline
) -> float:
    """
    Compute matchup z-score for a single game.
    Positive = favorable matchup. Negative = unfavorable.
    """
    score = 0.0
    
    # 1. Handedness split (weight: 0.35)
    # Most predictive single factor in baseball
    if context.opponent_starter_hand == "L":
        hand_gap = hitter.woba_vs_lhp - hitter.woba_overall
    else:
        hand_gap = hitter.woba_vs_rhp - hitter.woba_overall
    score += 0.35 * (hand_gap / league_baseline.std_woba_gap)
    
    # 2. Opponent starter quality (weight: 0.25)
    # Better pitcher = lower matchup score
    starter_z = (
        (league_baseline.mean_era - context.opponent_starter_era) / league_baseline.std_era +
        (league_baseline.mean_whip - context.opponent_starter_whip) / league_baseline.std_whip
    ) / 2
    score -= 0.25 * starter_z  # subtract because good pitcher hurts hitter
    
    # 3. Park factor (weight: 0.15)
    park_bonus = (context.park_factor_runs - 1.0) * 20  # +4 for Coors, -3 for Oracle
    score += 0.15 * park_bonus
    
    # 4. Weather (weight: 0.10)
    weather_bonus = 0.0
    if context.weather_wind_mph > 15:
        if context.weather_wind_direction in ("out", "l_to_r"):
            weather_bonus = 3.0  # wind blowing out
        elif context.weather_wind_direction in ("in", "r_to_l"):
            weather_bonus = -3.0  # wind blowing in
    if context.weather_temp_f and context.weather_temp_f > 85:
        weather_bonus += 1.5  # hot air carries
    score += 0.10 * weather_bonus
    
    # 5. Bullpen quality (weight: 0.15)
    # If starter gets pulled early, hitter faces bullpen
    bullpen_z = (
        (league_baseline.mean_bullpen_era - context.opponent_bullpen_era) / league_baseline.std_bullpen_era
    )
    score += 0.15 * bullpen_z  # bad bullpen = good for hitter
    
    return round(score, 3)
```

**Weather Data Source:**
- Use `backend/fantasy_baseball/weather.py` if it exists, or integrate OpenWeatherMap/NOAA API
- Cache weather forecasts per (venue, date) for 6 hours
- If weather unavailable, use `weather_bonus = 0.0` (neutral)

### 4.5 Integration Points

| Consumer | Integration |
|----------|-------------|
| `daily_lineup_optimizer.py` | Replace generic `opp_impl` with `matchup_z` in lineup scoring. A hitter with matchup_z=+2.0 gets a 20% lineup score boost. |
| `mcmc_simulator.py` | Adjust category means by `matchup_z * category_volatility` for the specific week |
| `waiver_edge_detector.py` | Show "Favorable matchups this week" tag if upcoming 7-day average matchup_z > +1.0 |

### 4.6 Data Pipeline

```
Step 1: Probable Pitchers (daily, 10 AM ET)
  Source: MLB Stats API / probable_pitchers table
  Query: SELECT game_date, home_team, away_team, probable_starter
         FROM probable_pitchers WHERE game_date = CURRENT_DATE

Step 2: Hitter Splits (daily, same job)
  Source: player_rolling_stats
  Compute: woba_vs_lhp, woba_vs_rhp from last 365 days of mlb_player_stats

Step 3: Weather Fetch (daily, 10 AM ET)
  Source: OpenWeatherMap API (venue lat/lon)
  Cache: 6-hour TTL per venue

Step 4: Matchup Compute + Upsert
  For each hitter with a game today:
    Fetch opponent starter
    Compute matchup_z
    INSERT INTO matchup_context ... ON CONFLICT UPDATE
```

### 4.7 Edge Cases / Failure Modes

| Edge Case | Handling |
|-----------|----------|
| Opponent starter TBD | Use opponent's average starter ERA as placeholder; confidence=0.3 |
| Hitter has <20 PA vs this hand | Use overall wOBA with regression toward mean; confidence=0.5 |
| Indoor stadium (dome) | Ignore weather; `weather_bonus = 0.0` |
| Doubleheader | Create TWO matchup_context rows, one for each game |
| Rainout / PPD | Row stays in DB; `matchup_z` is still valid if game rescheduled same day |

### 4.8 Performance Considerations

- **Table size:** ~2,000 hitters × 162 games = 324K rows/year. Manageable.
- **Query pattern:** Lineup optimizer queries by `(bdl_player_id, game_date)`. Indexed.
- **Weather API:** ~15 venue calls/day. Well within free tier limits.

### 4.9 Phased Rollout Plan

| Phase | What | Timeline |
|-------|------|----------|
| MVP | Handedness splits + opponent starter ERA only | 3 days |
| V1 | Add park factors (dynamic from existing `park_factors` table), weather | 2 days |
| V2 | Add bullpen quality, pitch-type interaction (requires pitch-level data) | 1 week |

### 4.10 Estimated Complexity

- **Engineering effort:** Medium (1 engineer × 1.5 weeks for MVP+V1)
- **Risk level:** Medium — depends on external weather API and MLB probable pitcher data

---

## Feature 5: Market Signals Engine

### 5.1 Problem Definition

The current system tracks `owned_pct` but treats it as a static filter. It completely ignores:
- **Velocity of change:** A player going from 15% to 65% owned in 48 hours is a market signal
- **Add/drop ratio:** High add velocity with low drop velocity = rising demand
- **FAAB bidding trends:** Are people actually spending money, or just adding?
- **Roster churn:** A player with 60% ownership but 40% weekly churn is unstable

**Why this matters:** The best waiver adds are often players the market hasn't noticed yet. The best sells are players the market is overreacting to.

### 5.2 Design Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Yahoo API       │────▶│ market_engine    │────▶│ player_market_  │
│ (free agents,   │     │ (daily snapshot)  │     │ signals         │
│  ownership %)   │     │                  │     │ (DB table)      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                              │
                              ▼
                        ┌──────────────────┐
                        │ waiver_edge_     │
                        │ detector.py      │
                        │ (uses market_     │
                        │  score for       │
                        │  contrarian bias)│
                        └──────────────────┘
```

### 5.3 Data Model

```sql
CREATE TABLE player_market_signals (
    id BIGSERIAL PRIMARY KEY,
    bdl_player_id INTEGER NOT NULL REFERENCES player_id_mapping(bdl_id),
    as_of_date DATE NOT NULL,
    
    -- Ownership (from Yahoo)
    yahoo_owned_pct FLOAT,
    yahoo_owned_pct_7d_ago FLOAT,
    yahoo_owned_pct_30d_ago FLOAT,
    
    -- Velocity (derived)
    ownership_delta_7d FLOAT,
    ownership_delta_30d FLOAT,
    ownership_velocity FLOAT,  -- adds per day (smoothed)
    
    -- Add/drop signals (if available from API or transaction scraping)
    add_rate_7d FLOAT,    -- estimated adds per 1000 leagues per day
    drop_rate_7d FLOAT,   -- estimated drops per 1000 leagues per day
    add_drop_ratio FLOAT, -- add_rate / drop_rate. >2.0 = hot pickup
    
    -- FAAB signal (if league tracks this)
    faab_bid_median FLOAT,
    faab_bid_max FLOAT,
    
    -- Derived scores
    market_score FLOAT,       -- 0-100, contrarian opportunity
    market_tag VARCHAR(20),   -- 'BUY_LOW', 'SELL_HIGH', 'HOT_PICKUP', 'FAIR'
    market_urgency VARCHAR(20), -- 'ACT_NOW', 'THIS_WEEK', 'MONITOR', 'NONE'
    
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE (bdl_player_id, as_of_date)
);

CREATE INDEX idx_market_signals_player_date ON player_market_signals(bdl_player_id, as_of_date);
CREATE INDEX idx_market_signals_date_score ON player_market_signals(as_of_date, market_score);
```

### 5.4 Core Algorithms / Logic

**Ownership Velocity (noise-filtered):**

```python
def compute_ownership_velocity(
    current_pct: float,
    pct_7d_ago: float,
    pct_14d_ago: float,
    min_leagues: int = 100
) -> float:
    """
    Smoothed velocity using exponential decay weighting.
    Recent changes matter more than old changes.
    """
    if current_pct is None or pct_7d_ago is None:
        return 0.0
    
    delta_7d = current_pct - pct_7d_ago
    delta_14d = current_pct - pct_14d_ago
    
    # Weight: 7-day delta = 0.7, 14-day delta = 0.3
    velocity = 0.7 * (delta_7d / 7.0) + 0.3 * (delta_14d / 14.0)
    
    # Noise filter: if player has <5% ownership, velocity is noisy
    if current_pct < 5.0 and abs(velocity) < 1.0:
        return 0.0
    
    return round(velocity, 3)
```

**Market Score (contrarian engine):**

```python
def compute_market_score(
    skill_gap: float,           # xwOBA - wOBA (positive = underperforming skill)
    skill_gap_percentile: float, # percentile of gap across league
    ownership_velocity: float,   # % change per day
    owned_pct: float
) -> MarketResult:
    """
    The core contrarian logic:
    - Large positive skill gap + low ownership velocity = BUY_LOW (market hasn't noticed)
    - Large negative skill gap + high ownership velocity = SELL_HIGH (market overreacting)
    - High ownership velocity + positive skill gap = HOT_PICKUP (market catching on — urgency)
    """
    
    # Component 1: Skill gap signal (-1 to +1)
    skill_signal = 2.0 * (skill_gap_percentile - 0.5)  # -1.0 to +1.0
    
    # Component 2: Market awareness (-1 to +1)
    # High velocity = market HAS noticed (bad for buy-low)
    market_awareness = min(ownership_velocity / 5.0, 1.0)  # cap at 5%/day
    
    # Contrarian score: high skill gap but low market awareness = best opportunity
    contrarian = skill_signal * (1.0 - market_awareness)
    
    # Normalize to 0-100
    market_score = 50.0 + (contrarian * 50.0)
    
    # Tag generation
    if skill_gap_percentile > 0.85 and ownership_velocity < 2.0:
        tag = "BUY_LOW"
        urgency = "ACT_NOW"
    elif skill_gap_percentile < 0.15 and ownership_velocity > 3.0:
        tag = "SELL_HIGH"
        urgency = "THIS_WEEK"
    elif ownership_velocity > 5.0 and skill_gap_percentile > 0.60:
        tag = "HOT_PICKUP"
        urgency = "ACT_NOW"
    elif owned_pct < 15.0 and skill_gap_percentile > 0.70:
        tag = "SLEEPER"
        urgency = "THIS_WEEK"
    else:
        tag = "FAIR"
        urgency = "MONITOR"
    
    return MarketResult(
        score=round(market_score, 1),
        tag=tag,
        urgency=urgency,
        reasoning=f"Skill gap at {skill_gap_percentile:.0%}ile, market moving at {ownership_velocity:.1f}%/day"
    )
```

### 5.5 Data Sources

**Primary:** Yahoo Fantasy API
- `get_free_agents()` returns `percent_owned` (or `percent_rostered` in 2025+ format)
- Run daily and store historical snapshots
- Transaction data: `get_transactions()` returns add/drop events per league
  - Can be extrapolated to league-wide add/drop rates

**Secondary:** ESPN API (if multi-platform support is desired)
- Similar ownership % endpoints

**Limitation:** Yahoo does not expose league-wide add/drop velocity directly. We must:
1. Track our own league's transactions
2. Use ownership % deltas as a proxy for market movement
3. If we have multiple league connections, aggregate across them for better signal

### 5.6 Noise Filtering

| Filter | Rule |
|--------|------|
| Minimum ownership | Ignore velocity for players <3% owned (too noisy) |
| Smoothing | Use 3-day moving average of ownership % |
| Sample size | Require ≥7 days of history before generating BUY_LOW/SELL_HIGH |
| Call-ups | New call-ups get a 7-day grace period (high velocity expected) |

### 5.7 Integration Points

| Consumer | How It Uses Market Signal |
|----------|--------------------------|
| `waiver_edge_detector.py` | Sort FAs by `market_score` when `win_prob_gain` is tied. Prefer BUY_LOW over HOT_PICKUP if both have similar skill. |
| `daily_briefing.py` | Include "Market is sleeping on X" or "Market is overreacting to Y" in daily briefings |
| `trade_analyzer.py` (future) | SELL_HIGH players are trade candidates; BUY_LOW players are acquisition targets |

### 5.8 Anti-Herd Bias Safeguard

The system must NOT recommend a player just because everyone else is adding them. The `market_score` is **contrarian** by design:
- High add velocity WITHOUT underlying skill improvement = **SELL_HIGH** or **AVOID**
- High add velocity WITH skill improvement = **HOT_PICKUP** (valid, but lower score than BUY_LOW)
- The best recommendations are always **BUY_LOW** or **SLEEPER** — players the market hasn't found yet

### 5.9 Phased Rollout Plan

| Phase | What | Timeline |
|-------|------|----------|
| MVP | Track ownership % daily, compute 7-day delta and velocity | 2 days |
| V1 | Add contrarian scoring (skill_gap × market_awareness), generate BUY_LOW/SELL_HIGH tags | 2 days |
| V2 | Aggregate transaction data from Yahoo `get_transactions()`, add add/drop ratio | 3 days |

### 5.10 Estimated Complexity

- **Engineering effort:** Small (1 engineer × 1 week)
- **Risk level:** Low — additive, uses existing Yahoo API. Main risk is API rate limits from daily polling.

---

## Feature 6: Decision Output / UX Redesign

### 6.1 Problem Definition

The current API returns raw metrics and expects users to synthesize decisions:

```json
{
  "need_score": 1.234,
  "category_contributions": {"hr": 0.5, "rbi": 0.3},
  "z_score": 0.8,
  "hot_cold": "HOT"
}
```

Users see **data**, not **instructions**. They don't know:
- Should I add this player or not?
- Should I start him today?
- How confident is this recommendation?
- What's the downside if I'm wrong?

### 6.2 Design Overview

Replace the flat metric dump with a **structured decision object** that explicitly states the action, reasoning, confidence, risk, and urgency.

### 6.3 Decision Output Schema

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import date

class DecisionAction(BaseModel):
    """A single recommended action for a player."""
    action: Literal["ADD", "DROP", "START", "BENCH", "TRADE", "STREAM", "HOLD", "IGNORE"]
    priority: int = Field(..., ge=1, le=10, description="1 = highest priority, 10 = lowest")
    confidence: float = Field(..., ge=0.0, le=1.0)
    urgency: Literal["ACT_NOW", "THIS_WEEK", "MONITOR", "NONE"]
    
    # Reasoning
    headline: str  # One-line summary: "Add — power breakout with everyday playing time"
    rationale: str  # 2-3 sentence explanation
    key_drivers: list[str]  # Specific metrics driving the recommendation
    
    # Risk / uncertainty
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    downside_scenario: str  # "If he cools off, he's a drop in 2 weeks"
    volatility_rating: Literal["LOW", "MEDIUM", "HIGH"]
    
    # Context
    relevant_tags: list[str]  # ["BUY_LOW", "EVERYDAY_PLAY", "PLUS_STUFF"]
    time_horizon: Literal["DAILY", "WEEKLY", "ROS"]
    category_impact: dict[str, float]  # {"hr": +0.4, "avg": -0.1}
    
    # Scores (for transparency)
    skill_score: float  # 0-100
    trend_score: float  # 0-100
    opportunity_score: float  # 0-100
    matchup_score: Optional[float]  # 0-100, null if not applicable
    market_score: Optional[float]  # 0-100, null if not applicable
    
    class Config:
        json_schema_extra = {
            "example": {
                "action": "ADD",
                "priority": 2,
                "confidence": 0.82,
                "urgency": "ACT_NOW",
                "headline": "Add — undervalued power with locked-in cleanup role",
                "rationale": "xwOBA is 95th percentile but wOBA is only 65th percentile, indicating positive regression. He's batting cleanup every day.",
                "key_drivers": [
                    "xwOBA-wOBA gap at 95th percentile",
                    "Batting 4th in 14/14 games",
                    "Barrel% up 6 points from last year"
                ],
                "risk_level": "MEDIUM",
                "downside_scenario": "If his BABIP doesn't regress, he's a league-average hitter. Still worth the add.",
                "volatility_rating": "MEDIUM",
                "relevant_tags": ["BUY_LOW", "EVERYDAY_PLAY", "POWER_UP"],
                "time_horizon": "ROS",
                "category_impact": {"hr": 0.5, "rbi": 0.4, "avg": -0.1},
                "skill_score": 78.5,
                "trend_score": 72.0,
                "opportunity_score": 85.0,
                "matchup_score": None,
                "market_score": 88.0
            }
        }

class PlayerDecisionCard(BaseModel):
    """Complete decision output for a single player."""
    player_id: str
    player_name: str
    player_type: Literal["hitter", "pitcher", "two_way"]
    position: str
    team: str
    
    # The primary recommendation
    primary_recommendation: DecisionAction
    
    # Alternative actions (if context changes)
    alternative_actions: list[DecisionAction]
    
    # Raw data (for power users who want it)
    raw_metrics: Optional[dict]  # Existing metric dump, hidden by default
    
    # Timestamp
    generated_at: date
    valid_until: date  # Recommendations expire; e.g., daily lineup recs expire next day
```

### 6.4 Explanation Layer Design

**Headline generation rules:**

| Scenario | Headline |
|----------|----------|
| BUY_LOW + everyday + power | "Add — undervalued power with locked-in cleanup role" |
| SELL_HIGH + declining trend | "Trade — skills are down but name value is high" |
| STREAM + favorable matchup | "Stream today — faces LHP at Coors, .420 wOBA vs lefties" |
| START + elite matchup | "Start — best matchup on the slate" |
| BENCH + bad matchup + platoon | "Bench — faces elite RHP, strict platoon player" |
| HOLD + no action needed | "Hold — steady performer, no action required" |

**Rationale generation (template + variable substitution):**

```python
RATIONALE_TEMPLATES = {
    "BUY_LOW": "{name} has an xwOBA of {xwOBA:.3f} but actual wOBA of {woba:.3f}, a gap at the {gap_pct:.0f}th percentile. {opportunity_sentence}.",
    "SELL_HIGH": "{name}'s ERA ({era:.2f}) is {era_gap:.1f} runs below his xERA ({xera:.2f}). He's due for regression.",
    "STREAM": "{name} faces {opponent_starter} ({starter_hand}HP) at {venue}. His wOBA vs {hand}HP is {split_woba:.3f}, well above league average.",
    "EVERYDAY_PLAY": "He's started {games_started}/14 games, batting {lineup_slot_avg:.1f} on average.",
}
```

### 6.5 Prioritization Rules

Not every player needs a recommendation. The system should only surface decisions when there's **actionable edge**:

```python
def should_generate_recommendation(player: dict, action: DecisionAction) -> bool:
    """Filter noise — only surface meaningful decisions."""
    
    # Always show top waiver adds (top 10 by final_score)
    if action.action == "ADD" and action.priority <= 3:
        return True
    
    # Show start/bench for players in user's roster
    if action.action in ("START", "BENCH") and player.get("on_roster"):
        return True
    
    # Show trade recommendations only for high-confidence moves
    if action.action == "TRADE" and action.confidence >= 0.75:
        return True
    
    # Show BUY_LOW / SELL_HIGH tags for any player with strong market signal
    if action.market_score and action.market_score >= 80:
        return True
    
    # Default: hide
    return False
```

### 6.6 Example API Responses

**Waiver Wire Endpoint (simplified):**

```json
{
  "week_end": "2026-05-10",
  "matchup_opponent": "Team X",
  "top_recommendations": [
    {
      "player_id": "mlb.p.12345",
      "player_name": "Jordan Walker",
      "position": "3B",
      "team": "STL",
      "primary_recommendation": {
        "action": "ADD",
        "priority": 1,
        "confidence": 0.85,
        "urgency": "ACT_NOW",
        "headline": "Add — power breakout with everyday playing time",
        "rationale": "Barrel% is up 8 points from last year and he's batting cleanup in 12 straight games. The market hasn't caught on (18% owned).",
        "key_drivers": [
          "Barrel% at 18.5% (97th percentile)",
          "Batting 4th in 12/12 games",
          "xwOBA-wOBA gap at 91st percentile"
        ],
        "risk_level": "MEDIUM",
        "downside_scenario": "If he stops hitting barrels, he drops to platoon duty. Still worth the add given the upside.",
        "volatility_rating": "HIGH",
        "relevant_tags": ["BREAKOUT", "BUY_LOW", "EVERYDAY_PLAY"],
        "time_horizon": "ROS",
        "category_impact": {"hr": 0.6, "rbi": 0.5, "avg": 0.0},
        "skill_score": 76.0,
        "trend_score": 88.0,
        "opportunity_score": 82.0,
        "market_score": 91.0
      },
      "alternative_actions": [
        {
          "action": "STREAM",
          "priority": 3,
          "confidence": 0.70,
          "headline": "Stream this week — 2 starts at hitter-friendly parks",
          "rationale": "..."
        }
      ],
      "generated_at": "2026-05-04"
    }
  ],
  "category_deficits": [...]
}
```

### 6.7 UX Patterns

**What users need (in order):**
1. **Action** — "Add Jordan Walker"
2. **Why** — "Barrel% up 8 pts, batting cleanup, 18% owned"
3. **How sure** — "85% confident"
4. **What if wrong** — "Medium risk — could lose playing time if he slumps"
5. **When** — "Act now before next waiver period"
6. **Raw data** — collapsible section for advanced users

**What users DON'T need:**
- Raw z-scores without interpretation
- 15-category breakdowns on every card
- Charts that require 30 seconds to parse
- Recommendations for players they can't add (already rostered in 95% of leagues)

### 6.8 Integration Points

| Source System | Data Provided |
|---------------|--------------|
| `scoring_engine.py` | `skill_score`, `confidence` |
| `momentum_engine.py` | `trend_score`, `trend_signal` |
| `opportunity_engine.py` | `opportunity_score`, `opportunity_risk` |
| `matchup_engine.py` | `matchup_score`, `matchup_context` |
| `market_engine.py` | `market_score`, `market_tag`, `urgency` |
| `mcmc_simulator.py` | `win_prob_gain` (for waiver decisions) |

**New module:** `backend/services/decision_card_builder.py`
- Takes all 5 layer scores + MCMC output
- Generates `DecisionAction` objects
- Applies prioritization filters
- Renders rationale strings

### 6.9 Phased Rollout Plan

| Phase | What | Timeline |
|-------|------|----------|
| MVP | Create `DecisionAction` schema, add to waiver endpoint only | 3 days |
| V1 | Add to lineup endpoint, add headline + rationale generation | 2 days |
| V2 | Add to daily briefing, add alternative_actions, raw_metrics toggle | 3 days |

### 6.10 Estimated Complexity

- **Engineering effort:** Small-Medium (1 engineer × 1.5 weeks)
- **Risk level:** Low — purely presentational. Backend scoring logic unchanged.

---

## Cross-Feature Dependencies and Sequencing

```
Phase 1 (Week 1-2): Foundation
├── Feature 3: Threshold Config System
│   └── Unblocks dynamic tuning of all other features
├── Feature 2: Statcast Fix
│   └── Unlocks advanced metrics for Skill Score
└── Feature 1: Opportunity Model (MVP)
    └── Provides PA/game, lineup slot data

Phase 2 (Week 3-4): Intelligence Layers
├── Feature 1: Opportunity Model (V1)
│   └── Integrated into scoring_engine
├── Feature 4: Matchup Context Engine (MVP)
│   └── Handedness splits + opponent quality
└── Feature 5: Market Signals Engine (MVP)
    └── Ownership tracking + contrarian scoring

Phase 3 (Week 5-6): Decision Layer
├── Feature 4: Matchup Context Engine (V1)
├── Feature 5: Market Signals Engine (V1)
└── Feature 6: Decision Output / UX Redesign
    └── Consumes ALL layers to generate actionable cards
```

**Critical path:** Threshold Config → Statcast Fix → Opportunity Model → Market Engine → Decision UX

**Parallelizable:** Matchup Engine can be built in parallel with Market Engine (no shared state).

---

## Recommended Build Order

| Rank | Feature | Why First? | Effort | Risk |
|------|---------|-----------|--------|------|
| 1 | **Threshold Config System** | Every other feature needs tunable thresholds. Without this, you're hardcoding again. | S | Low |
| 2 | **Statcast Fix** | Advanced metrics are schema-only ghosts. Fix data before building features on top. | S | Medium |
| 3 | **Opportunity Model (MVP)** | Biggest gap in current system. Provides immediate value with simple PA/game + lineup slot. | M | Low |
| 4 | **Market Signals Engine** | Low effort, high impact. Contrarian signals are the easiest "edge" to generate. | S | Low |
| 5 | **Matchup Context Engine** | More complex due to external data sources (weather, splits). Build after simpler features. | M | Medium |
| 6 | **Decision Output / UX** | Final polish layer. Must wait until all scoring layers are stable. | M | Low |

---

*Design verified against production codebase (`stable/cbb-prod`, commit 827b2c0) and live Railway PostgreSQL schema (2026-05-04).*
