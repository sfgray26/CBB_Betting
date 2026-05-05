# CLAUDE CODE — IMPLEMENTATION PROMPT
## Fantasy Baseball Next-Gen Scoring Engine

**Date:** 2026-05-04  
**Branch:** `stable/cbb-prod` (commit 827b2c0)  
**Test Baseline:** 2488 pass / 4 skip / 0 fail  
**Database:** PostgreSQL via Railway (`DATABASE_URL` in env)  
**Your Role:** Principal Architect & Lead Developer (per AGENTS.md)

---

## MISSION

Implement a 7-EPIC feature set that transforms the fantasy baseball platform from a static stat-dump into a dynamic, context-aware decision engine. **Do NOT implement all at once.** Work PR-by-PR in the exact order specified below. Each PR is atomic, testable, and reversible.

---

## CRITICAL ARCHITECTURE PRINCIPLES (NON-NEGOTIABLE)

1. **Keep scores SEPARATE until the decision layer**
   - `skill_score`, `opportunity_score`, `matchup_score`, `market_score` are independent
   - Combine ONLY in the final decision builder

2. **Opportunity is a bounded modifier, NOT a multiplier on raw skill**
   ```python
   opportunity_adj = clamp(opportunity_z * 0.15, -0.3, 0.2)
   opportunity_adj *= confidence
   final_score = skill_score * (1 + opportunity_adj)
   ```

3. **All weights configurable via threshold system**
   - No hardcoded weights in matchup/market/opportunity engines
   - Use `config_service.get_threshold(key, default)`

4. **Confidence gates EVERYTHING**
   - Low sample → dampen market signals, matchup boosts, opportunity impact
   - `if confidence < 0.5: contrarian *= 0.5`

5. **External data fails gracefully**
   - Statcast/weather failures must NOT break scoring
   - Use last-known-good or skip the feature layer

6. **Decision engine resolves conflicts**
   - Competing signals reconciled, not just displayed
   - Tag outputs: `LOW_CONFIDENCE`, `LIMITED_PLAYING_TIME`, `MARKET_HYPE`

7. **Every feature has a kill switch**
   - Feature flags in `threshold_config` or env vars
   - If flag disabled, scoring falls back to pre-feature behavior

8. **No datetime.utcnow()**
   - Use `datetime.now(ZoneInfo("America/New_York"))` or `datetime.now(timezone.utc)`
   - This is a project-wide code quality gate

---

## GOTCHAS (READ BEFORE CODING)

### Gotcha 1: Config Table Value Type
The `threshold_config` table **must use JSONB for `value`**, not FLOAT. Three critical constants are dicts:
- `_CATEGORY_WEIGHTS` in `scoring_engine.py`
- `_POSITION_MULT` in `mcmc_simulator.py`
- `_PLAYER_WEEKLY_STD` in `mcmc_simulator.py`

**Fix:** Use `JSONB` in PR 1.1. This is a one-line schema change that prevents a migration later.

### Gotcha 2: Hitter Split Data Doesn't Exist
`mlb_player_stats` has no `opponent_starter_hand` column. PR 5.2 (matchup splits) is **blocked** until this column exists.

**Fix:** Add `opponent_starter_hand` to `mlb_player_stats` ingestion in a mini-PR before EPIC 5.2.

### Gotcha 3: DB Naming Convention
The user's spec uses `player_id` but the existing production schema uses `bdl_player_id` (foreign key to `player_id_mapping.bdl_id`). **Always use `bdl_player_id`** to match existing tables.

### Gotcha 4: Composite_z Is a Weighted SUM (Bug)
`scoring_engine.py:525-528` computes `composite_z = sum(weights * z)` instead of `sum(weights * z) / sum(weights)`. **This is a pre-existing P0 bug.** Do NOT fix it in these PRs unless explicitly told — it affects 21 tests and requires its own PR.

### Gotcha 5: player_board Uses Sample Std (Bug)
`player_board.py:628-629` uses `statistics.stdev(values)` (ddof=1) while `scoring_engine.py` uses population std (ddof=0). **Pre-existing P0 bug.** Do NOT fix here.

---

## PR-BY-PR BREAKDOWN

Implement in this exact order. Each PR = 1 commit. Open a PR, get it reviewed/merged, then start the next.

---

### PR 1.1 — Create Threshold Config Tables

**Files:** `scripts/migration_threshold_config.sql`

**Schema:**
```sql
CREATE TABLE threshold_config (
    id SERIAL PRIMARY KEY,
    config_key TEXT NOT NULL,
    config_value JSONB NOT NULL,
    scope TEXT NOT NULL DEFAULT 'global',
    description TEXT,
    effective_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (config_key, scope)
);

CREATE INDEX idx_threshold_key_scope ON threshold_config(config_key, scope);

CREATE TABLE threshold_audit (
    id BIGSERIAL PRIMARY KEY,
    config_key TEXT NOT NULL,
    old_value JSONB,
    new_value JSONB NOT NULL,
    changed_by TEXT,
    changed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE feature_flags (
    id SERIAL PRIMARY KEY,
    flag_name TEXT NOT NULL UNIQUE,
    enabled BOOLEAN NOT NULL DEFAULT FALSE,
    rollout_pct INTEGER NOT NULL DEFAULT 0 CHECK (rollout_pct BETWEEN 0 AND 100),
    scope TEXT DEFAULT 'global',
    description TEXT
);
```

**Run:** `railway run psql -f scripts/migration_threshold_config.sql`

**Acceptance:**
- [ ] Tables exist in production DB
- [ ] `UNIQUE (config_key, scope)` constraint active

---

### PR 1.2 — Config Service with Caching

**New file:** `backend/services/config_service.py`

**Requirements:**
- In-memory dict cache with 60s TTL
- `get_threshold(key: str, default: Any = None, scope: str = "global") -> Any`
- Cache refresh: query all active configs once per minute
- Thread-safe (use `threading.Lock()`)
- Returns `default` if key not found in DB

**Pseudocode:**
```python
import threading
import time
from typing import Any
from backend.models import SessionLocal

_cache: dict[str, Any] = {}
_cache_expiry: float = 0.0
_lock = threading.Lock()

def get_threshold(key: str, default: Any = None, scope: str = "global") -> Any:
    global _cache, _cache_expiry
    now = time.time()
    
    if now > _cache_expiry:
        with _lock:
            if now > _cache_expiry:  # double-check
                db = SessionLocal()
                try:
                    rows = db.execute(
                        "SELECT config_key, config_value FROM threshold_config WHERE scope = :scope",
                        {"scope": scope}
                    ).fetchall()
                    _cache = {r.config_key: r.config_value for r in rows}
                    _cache_expiry = now + 60.0
                finally:
                    db.close()
    
    return _cache.get(key, default)
```

**Tests:** `tests/test_config_service.py`
- [ ] Returns default when DB empty
- [ ] Cache hit avoids DB call (mock `time.time` to verify)
- [ ] TTL refresh after 60s
- [ ] Scope fallback works

**Acceptance:**
- [ ] `py_compile` passes
- [ ] All tests pass
- [ ] No DB query on cache hit

---

### PR 1.3 — Migrate 3–5 Core Constants

**Files to modify:**
- `backend/services/scoring_engine.py`
- `backend/fantasy_baseball/player_momentum.py` (or wherever momentum thresholds live)
- `backend/fantasy_baseball/category_aware_scorer.py`

**Constants to replace:**
```python
# scoring_engine.py
Z_CAP = 3.0 → Z_CAP = get_threshold("scoring.z_cap", default=3.0)
MIN_SAMPLE = 5 → MIN_SAMPLE = get_threshold("scoring.min_sample", default=5)

# momentum ( wherever the delta_z thresholds are computed )
SURGING_THRESHOLD = 0.5 → get_threshold("momentum.surging.delta_z", default=0.5)
HOT_THRESHOLD = 0.2 → get_threshold("momentum.hot.delta_z", default=0.2)
COLD_THRESHOLD = -0.5 → get_threshold("momentum.cold.delta_z", default=-0.5)

# category_aware_scorer.py
RATE_STAT_PROTECT_THRESHOLD = 0.5 → get_threshold("scoring.rate_stat_protect", default=0.5)
```

**Import:** `from backend.services.config_service import get_threshold`

**Rules:**
- Keep the hardcoded value as the `default` parameter
- If DB is empty, behavior is IDENTICAL to before
- Do NOT touch dict constants yet (CategoryWeights, PositionMult)

**Tests:**
- [ ] All existing tests pass with no changes
- [ ] When DB has a custom value, the custom value is used

**Acceptance:**
- [ ] `pytest tests/test_scoring_engine.py` passes
- [ ] `pytest tests/test_scoring_engine_fixes.py` passes
- [ ] No behavior change when DB empty

---

### PR 1.4 — Backfill Threshold Values

**New file:** `scripts/seed_threshold_config.py`

**Script:**
```python
from backend.models import SessionLocal
from backend.services.config_service import get_threshold  # force cache refresh

def seed():
    db = SessionLocal()
    seeds = [
        ("scoring.z_cap", 3.0, "Winsorization cap for Z-scores"),
        ("scoring.min_sample", 5, "Minimum players before computing category Z"),
        ("momentum.surging.delta_z", 0.5, "Delta-Z threshold for SURGING signal"),
        ("momentum.hot.delta_z", 0.2, "Delta-Z threshold for HOT signal"),
        ("momentum.cold.delta_z", -0.5, "Delta-Z threshold for COLD signal"),
        ("momentum.collapsing.delta_z", -0.5, "Delta-Z threshold for COLLAPSING signal"),
        ("scoring.rate_stat_protect", 0.5, "Rate stat protection threshold"),
        ("waiver.streamer_threshold", 0.3, "Z-score threshold for streamer suggestions"),
    ]
    for key, value, desc in seeds:
        db.execute("""
            INSERT INTO threshold_config (config_key, config_value, description)
            VALUES (:key, :value, :desc)
            ON CONFLICT (config_key, scope) DO NOTHING
        """, {"key": key, "value": value, "desc": desc})
    db.commit()
    db.close()
    print(f"Seeded {len(seeds)} thresholds")

if __name__ == "__main__":
    seed()
```

**Run:** `railway run python scripts/seed_threshold_config.py`

**Acceptance:**
- [ ] All values match previous hardcoded behavior
- [ ] `SELECT COUNT(*) FROM threshold_config` returns expected count

---

### PR 2.1 — Savant Scraper

**New file:** `backend/ingestion/savant_scraper.py`

**Requirements:**
- Fetch Baseball Savant sprint speed leaderboard CSV
- Parse into DataFrame with columns: `mlbam_id`, `player_name`, `sprint_speed`
- Handle request failures gracefully (return empty DataFrame, log warning)
- Use browser headers (see existing `_BROWSER_HEADERS` in `pybaseball_loader.py`)

**URL:** `https://baseballsavant.mlb.com/leaderboard/sprint_speed?year={year}&position=&team=&min=0&csv=true`

**Function signature:**
```python
def fetch_sprint_speed(year: int = 2026) -> pd.DataFrame:
    """Returns DataFrame with mlbam_id, sprint_speed. Empty on failure."""
```

**Tests:** `tests/test_savant_scraper.py`
- [ ] Returns DataFrame with correct columns on success
- [ ] Returns empty DataFrame on HTTP error
- [ ] Handles malformed CSV gracefully

**Acceptance:**
- [ ] `py_compile` passes
- [ ] Unit tests pass

---

### PR 2.2 — Pipeline Integration

**Files:**
- `backend/ingestion/savant_scraper.py` (already exists from 2.1)
- `backend/services/daily_ingestion.py` (find the Statcast update job)

**Requirements:**
- Hook `fetch_sprint_speed()` into the daily Statcast ingestion job
- Update `statcast_batter_metrics.sprint_speed` by `mlbam_id`
- Use `ON CONFLICT` or `UPDATE ... WHERE mlbam_id = ...` (no duplicates)

**SQL pattern:**
```python
for _, row in df.iterrows():
    db.execute("""
        UPDATE statcast_batter_metrics
        SET sprint_speed = %s
        WHERE mlbam_id = %s AND season = %s
    """, (row["sprint_speed"], str(row["mlbam_id"]), year))
```

**Acceptance:**
- [ ] Daily ingestion job populates `sprint_speed`
- [ ] `SELECT COUNT(*) FROM statcast_batter_metrics WHERE sprint_speed IS NOT NULL` > 0 after run
- [ ] Failure in savant scraper does NOT crash the entire ingestion job

---

### PR 2.3 — Validation + Feature Flag

**Files:**
- `backend/services/daily_ingestion.py`
- `backend/services/config_service.py`

**Requirements:**
- After Statcast ingestion, compute null rate for `sprint_speed`
- If null rate > 30%, disable feature flag `statcast_sprint_speed_enabled`
- Feature flag checked before using sprint_speed in scoring

**Validation function:**
```python
def validate_statcast_coverage(table: str, column: str, threshold: float = 0.70) -> bool:
    total = db.query(f"SELECT COUNT(*) FROM {table}").scalar()
    non_null = db.query(f"SELECT COUNT(*) FROM {table} WHERE {column} IS NOT NULL").scalar()
    rate = non_null / total if total else 0
    if rate < threshold:
        logger.error(f"{table}.{column} coverage {rate:.1%} below {threshold}")
        return False
    return True
```

**Feature flag seed:**
```sql
INSERT INTO feature_flags (flag_name, enabled, description)
VALUES ('statcast_sprint_speed_enabled', true, 'Enable sprint_speed in scoring');
```

**Acceptance:**
- [ ] Validation runs after each ingestion
- [ ] Flag disables when coverage < 70%
- [ ] Flag re-enables when coverage recovers

---

### PR 2.4 — Backfill Script

**New file:** `scripts/backfill_statcast_advanced.py`

**Requirements:**
- Backfill `sprint_speed` for 2026 season
- Idempotent (running twice does not duplicate data)
- Log progress: `Updated X / Y players`

**Run:** `railway run python scripts/backfill_statcast_advanced.py`

**Acceptance:**
- [ ] `SELECT COUNT(*) FROM statcast_batter_metrics WHERE sprint_speed IS NOT NULL AND season = 2026` returns >70% of rows
- [ ] Running twice produces same result

---

### PR 3.1 — Opportunity Schema

**File:** `scripts/migration_player_opportunity.sql`

**Schema:**
```sql
CREATE TABLE player_opportunity (
    id BIGSERIAL PRIMARY KEY,
    bdl_player_id INTEGER NOT NULL REFERENCES player_id_mapping(bdl_id),
    as_of_date DATE NOT NULL,
    
    pa_per_game FLOAT,
    ab_per_game FLOAT,
    games_played_14d INTEGER,
    games_started_14d INTEGER,
    games_started_pct FLOAT,
    
    lineup_slot_avg FLOAT,
    lineup_slot_mode INTEGER,
    lineup_slot_entropy FLOAT,
    
    pa_vs_lhp_14d INTEGER,
    pa_vs_rhp_14d INTEGER,
    platoon_ratio FLOAT,
    platoon_risk_score FLOAT,
    
    appearances_14d INTEGER,
    saves_14d INTEGER,
    holds_14d INTEGER,
    role_certainty_score FLOAT,
    
    days_since_last_game INTEGER,
    il_stint_flag BOOLEAN,
    
    opportunity_score FLOAT,
    opportunity_z FLOAT,
    opportunity_confidence FLOAT,
    
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    UNIQUE (bdl_player_id, as_of_date)
);

CREATE INDEX idx_player_opp_bdl_date ON player_opportunity(bdl_player_id, as_of_date);
CREATE INDEX idx_player_opp_date ON player_opportunity(as_of_date);
CREATE INDEX idx_player_opp_opportunity_z ON player_opportunity(as_of_date, opportunity_z);
```

**Run:** `railway run psql -f scripts/migration_player_opportunity.sql`

**Acceptance:**
- [ ] Table exists in production
- [ ] Indexes created

---

### PR 3.2 — Raw Metric Computation

**New file:** `backend/services/opportunity_engine.py`

**Functions:**
```python
def compute_lineup_entropy(slots: list[int]) -> float:
    """Shannon entropy normalized to 0-1."""

def compute_platoon_risk(pa_vs_lhp: int, pa_vs_rhp: int) -> float:
    """0.0 = everyday, 1.0 = strict platoon."""

def compute_role_certainty(appearances: int, saves: int, holds: int, player_type: str) -> float:
    """0-1 score for pitchers based on save/hold consistency."""

def aggregate_player_opportunity(bdl_player_id: int, as_of_date: date, db: Session) -> dict:
    """Query mlb_player_stats for last 14 days, compute all raw metrics."""
```

**SQL for aggregation:**
```sql
SELECT 
    player_id,
    COUNT(DISTINCT game_date) as games_played,
    SUM(pa) as total_pa,
    AVG(pa) as pa_per_game,
    -- lineup_slot requires mlb_player_stats to have lineup_position
    -- if not present, compute from box score data
FROM mlb_player_stats
WHERE player_id = :bdl_id
  AND game_date >= :cutoff
GROUP BY player_id
```

**Tests:** `tests/test_opportunity_engine.py`
- [ ] Entropy calculation correct
- [ ] Platoon risk handles edge cases (0 PA, all vs one hand)
- [ ] Role certainty for closers = high, for swingmen = low

**Acceptance:**
- [ ] `py_compile` passes
- [ ] Tests pass

---

### PR 3.3 — Opportunity Score + Z-Score

**File:** `backend/services/opportunity_engine.py`

**Requirements:**
- Compute `opportunity_score` (0-1 normalized)
- Compute `opportunity_z` (z-score against league baseline)
- Compute `opportunity_confidence` (sigmoid of PA in window)

**League baseline computation:**
```python
def compute_opportunity_baselines(as_of_date: date, db: Session) -> dict:
    """Compute league-wide mean/std for pa_per_game, lineup_slot_avg, platoon_risk."""
```

**Acceptance:**
- [ ] Scores bounded correctly
- [ ] Z-scores sensible (everyday leadoff hitter ≈ +2.0, platoon bench player ≈ -1.5)

---

### PR 3.4 — Ingestion Hook (LOG ONLY)

**File:** `backend/services/daily_ingestion.py`

**Requirements:**
- Add new daily job: `opportunity_update`
- Compute opportunity for all players with `mlb_player_stats` in last 14 days
- UPSERT into `player_opportunity`
- **LOG ONLY** — do NOT use in scoring yet
- Log output: `opportunity_update: computed 1847 players, avg_opportunity_z=0.12`

**Acceptance:**
- [ ] Job runs daily without errors
- [ ] Table populated for active players
- [ ] No impact on existing scoring or waiver logic

---

### PR 3.5 — Safe Integration into Scoring

**File:** `backend/services/scoring_engine.py`

**Requirements:**
- Fetch `opportunity_z` and `opportunity_confidence` for player
- Apply bounded modifier:
  ```python
  opportunity_adj = clamp(opportunity_z * 0.15, -0.3, 0.2)
  opportunity_adj *= opportunity_confidence
  
  # Only apply if feature flag enabled
  if get_threshold("feature.opportunity_enabled", default=False):
      composite_z = composite_z * (1 + opportunity_adj)
  ```
- Feature flag seed:
  ```sql
  INSERT INTO feature_flags (flag_name, enabled, description)
  VALUES ('opportunity_enabled', false, 'Enable opportunity scoring modifier');
  ```

**Tests:**
- [ ] When flag disabled, scores identical to before
- [ ] When flag enabled, extreme opportunity_z (+3.0) capped at +20%
- [ ] When flag enabled, extreme negative opportunity_z (-3.0) capped at -30%

**Acceptance:**
- [ ] All existing tests pass with flag OFF
- [ ] New tests pass with flag ON
- [ ] No player score changes by >30% in either direction

---

### PR 4.1 — Market Signals Schema

**File:** `scripts/migration_market_signals.sql`

**Schema:**
```sql
CREATE TABLE player_market_signals (
    id BIGSERIAL PRIMARY KEY,
    bdl_player_id INTEGER NOT NULL REFERENCES player_id_mapping(bdl_id),
    as_of_date DATE NOT NULL,
    
    yahoo_owned_pct FLOAT,
    yahoo_owned_pct_7d_ago FLOAT,
    yahoo_owned_pct_30d_ago FLOAT,
    
    ownership_delta_7d FLOAT,
    ownership_delta_30d FLOAT,
    ownership_velocity FLOAT,
    
    add_rate_7d FLOAT,
    drop_rate_7d FLOAT,
    add_drop_ratio FLOAT,
    
    market_score FLOAT,
    market_tag VARCHAR(20),
    market_urgency VARCHAR(20),
    
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE (bdl_player_id, as_of_date)
);

CREATE INDEX idx_market_signals_player_date ON player_market_signals(bdl_player_id, as_of_date);
CREATE INDEX idx_market_signals_date_score ON player_market_signals(as_of_date, market_score);
```

**Acceptance:**
- [ ] Table exists
- [ ] Indexes created

---

### PR 4.2 — Ownership History Tracking

**File:** `backend/fantasy_baseball/yahoo_client_resilient.py` + `backend/services/daily_ingestion.py`

**Requirements:**
- During daily Yahoo free agent poll, store `percent_owned` for each player
- Track history in `player_market_signals`
- If player not polled today, copy forward from yesterday

**Acceptance:**
- [ ] Daily snapshots stored
- [ ] 7-day and 30-day deltas computable

---

### PR 4.3 — Market Score Calculation

**File:** `backend/services/market_engine.py`

**Algorithm:**
```python
def compute_market_score(
    skill_gap: float,
    skill_gap_percentile: float,
    ownership_velocity: float,
    owned_pct: float,
    confidence: float,
) -> MarketResult:
    skill_signal = 2.0 * (skill_gap_percentile - 0.5)
    market_awareness = min(abs(ownership_velocity) / 5.0, 1.0)
    contrarian = skill_signal * (1.0 - market_awareness)
    
    # Confidence gate
    if confidence < 0.5:
        contrarian *= 0.5
    
    market_score = 50.0 + (contrarian * 50.0)
    
    # Tag logic
    if skill_gap_percentile > 0.85 and abs(ownership_velocity) < 2.0:
        tag, urgency = "BUY_LOW", "ACT_NOW"
    elif skill_gap_percentile < 0.15 and ownership_velocity > 3.0:
        tag, urgency = "SELL_HIGH", "THIS_WEEK"
    elif ownership_velocity > 5.0 and skill_gap_percentile > 0.60:
        tag, urgency = "HOT_PICKUP", "ACT_NOW"
    elif owned_pct < 15.0 and skill_gap_percentile > 0.70:
        tag, urgency = "SLEEPER", "THIS_WEEK"
    else:
        tag, urgency = "FAIR", "MONITOR"
    
    return MarketResult(score=market_score, tag=tag, urgency=urgency)
```

**Tests:** `tests/test_market_engine.py`
- [ ] High skill gap + low velocity = BUY_LOW
- [ ] Low skill gap + high velocity = SELL_HIGH
- [ ] Low confidence dampens score

---

### PR 4.4 — Confidence Gating

**File:** `backend/services/market_engine.py`

**Already implemented in 4.3** — verify the `if confidence < 0.5: contrarian *= 0.5` line exists and is tested.

**Acceptance:**
- [ ] Unit test confirms low-confidence players get dampened scores

---

### PR 4.5 — Decision Integration

**File:** `backend/services/waiver_edge_detector.py`

**Requirements:**
- Add market_score as tiebreaker in `get_top_moves()`
- When `win_prob_gain` is tied, prefer BUY_LOW over FAIR
- Do NOT let market_score dominate skill — max 10% weight in final ranking

**Code change:**
```python
# In get_top_moves() sort key
moves.sort(key=lambda m: (
    m["win_prob_gain"],
    m["need_score"],
    m.get("market_score", 50.0)  # tiebreaker
), reverse=True)
```

**Acceptance:**
- [ ] Market signal influences ranking but does not override skill
- [ ] When market_score missing, falls back to 50.0 (neutral)

---

### PR 5.1 — Matchup Context Schema

**File:** `scripts/migration_matchup_context.sql`

**Schema:**
```sql
CREATE TABLE matchup_context (
    id BIGSERIAL PRIMARY KEY,
    bdl_player_id INTEGER NOT NULL REFERENCES player_id_mapping(bdl_id),
    game_date DATE NOT NULL,
    opponent_team VARCHAR(10),
    opponent_starter_name VARCHAR(100),
    opponent_starter_hand VARCHAR(1),
    opponent_starter_era FLOAT,
    opponent_starter_whip FLOAT,
    opponent_starter_k_per_nine FLOAT,
    opponent_bullpen_era FLOAT,
    opponent_bullpen_whip FLOAT,
    home_team VARCHAR(10),
    park_factor_runs FLOAT,
    park_factor_hr FLOAT,
    weather_temp_f FLOAT,
    weather_wind_mph FLOAT,
    weather_wind_direction VARCHAR(10),
    hitter_woba_vs_hand FLOAT,
    hitter_k_pct_vs_hand FLOAT,
    hitter_iso_vs_hand FLOAT,
    matchup_score FLOAT,
    matchup_z FLOAT,
    matchup_confidence FLOAT,
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (bdl_player_id, game_date)
);

CREATE INDEX idx_matchup_context_player_date ON matchup_context(bdl_player_id, game_date);
CREATE INDEX idx_matchup_context_date ON matchup_context(game_date);
```

**Acceptance:**
- [ ] Table exists

---

### PR 5.1b — Augment mlb_player_stats with Opponent Starter Hand

**File:** `backend/services/daily_ingestion.py` (MLB box stats ingestion)

**Requirements:**
- Add `opponent_starter_hand` column to `mlb_player_stats`
- Populate during daily ingestion from `probable_pitchers` table or MLB Stats API

**Migration:**
```sql
ALTER TABLE mlb_player_stats ADD COLUMN opponent_starter_hand VARCHAR(1);
```

**Acceptance:**
- [ ] Column exists
- [ ] Populated for new games

---

### PR 5.2 — Data Collection (Minimal)

**File:** `backend/services/matchup_engine.py`

**Requirements:**
- Fetch probable pitcher for each game
- Compute hitter split vs that hand from `mlb_player_stats` history
- Store in `matchup_context`

**Acceptance:**
- [ ] `matchup_context` populated for upcoming games
- [ ] `hitter_woba_vs_hand` computed from last 365 days

---

### PR 5.3 — Basic Matchup Score

**File:** `backend/services/matchup_engine.py`

**Algorithm:**
```python
def compute_matchup_z(hitter, context, baselines) -> float:
    # 1. Handedness split (35% weight)
    hand_gap = hitter.woba_vs_hand - hitter.woba_overall
    hand_z = hand_gap / baselines.std_woba_gap
    
    # 2. Pitcher quality (25% weight)
    pitcher_z = (4.50 - context.starter_era) / baselines.std_era
    
    # 3. Park factor (15% weight)
    park_bonus = (context.park_factor_runs - 1.0) * 20
    
    return 0.35 * hand_z - 0.25 * pitcher_z + 0.15 * park_bonus
```

**Acceptance:**
- [ ] Scores directionally correct (favorable matchup = positive)

---

### PR 5.4 — Configurable Weights

**File:** `backend/services/matchup_engine.py`

**Replace hardcoded weights:**
```python
weights = {
    "handedness": get_threshold("matchup.weight.handedness", default=0.35),
    "pitcher": get_threshold("matchup.weight.pitcher", default=0.25),
    "park": get_threshold("matchup.weight.park", default=0.15),
    "weather": get_threshold("matchup.weight.weather", default=0.10),
    "bullpen": get_threshold("matchup.weight.bullpen", default=0.15),
}
```

**Seed values:**
```sql
INSERT INTO threshold_config (config_key, config_value, description) VALUES
('matchup.weight.handedness', 0.35, 'Matchup weight for handedness split'),
('matchup.weight.pitcher', 0.25, 'Matchup weight for pitcher quality'),
('matchup.weight.park', 0.15, 'Matchup weight for park factor'),
('matchup.weight.weather', 0.10, 'Matchup weight for weather'),
('matchup.weight.bullpen', 0.15, 'Matchup weight for bullpen quality');
```

**Acceptance:**
- [ ] Changing a weight in DB changes scoring without code deploy

---

### PR 5.5 — Integration (Bounded Boost)

**File:** `backend/services/daily_lineup_optimizer.py`

**Requirements:**
- Apply matchup boost to lineup scoring:
  ```python
  matchup_boost = clamp(matchup_z * 0.1, -0.2, 0.2)
  matchup_boost *= matchup_confidence
  
  if get_threshold("feature.matchup_enabled", default=False):
      lineup_score = lineup_score * (1 + matchup_boost)
  ```
- Feature flag: `matchup_enabled`

**Acceptance:**
- [ ] Boost bounded to ±20%
- [ ] Confidence gates the boost
- [ ] No impact when flag disabled

---

### PR 6.1 — Extend Decision Inputs

**File:** `backend/schemas.py`

**Extend `WaiverPlayerOut` and `RosterMoveRecommendation`:**
```python
class WaiverPlayerOut(BaseModel):
    # ... existing fields ...
    skill_score: Optional[float] = None
    opportunity_score: Optional[float] = None
    matchup_score: Optional[float] = None
    market_score: Optional[float] = None
    confidence: Optional[float] = None
    tags: List[str] = []
```

**Acceptance:**
- [ ] Schema validates
- [ ] API responses include new fields

---

### PR 6.2 — Final Score Composition

**File:** `backend/services/decision_card_builder.py` (NEW)

**Requirements:**
- Combine all components:
  ```python
  final_score = skill_score * (1 + opportunity_adj) * (1 + matchup_boost)
  final_score += market_score * 0.1
  ```
- Cap final_score to 0-100 range

**Acceptance:**
- [ ] Final score bounded
- [ ] Market score does not dominate

---

### PR 6.3 — Conflict Tagging

**File:** `backend/services/decision_card_builder.py`

**Tag rules:**
```python
def generate_tags(skill_score, opportunity_z, market_score, confidence) -> list[str]:
    tags = []
    if confidence < 0.4:
        tags.append("LOW_CONFIDENCE")
    if opportunity_z < -1.0 and skill_score > 70:
        tags.append("LIMITED_PLAYING_TIME")
    if market_score > 80 and skill_score < 50:
        tags.append("MARKET_HYPE")
    if market_score < 20 and skill_score > 70:
        tags.append("BUY_LOW")
    return tags
```

**Acceptance:**
- [ ] Tags generated correctly for conflicting signals

---

### PR 6.4 — DecisionAction Schema

**File:** `backend/schemas.py`

**New schema:**
```python
class DecisionAction(BaseModel):
    action: Literal["ADD", "DROP", "START", "BENCH", "TRADE", "STREAM", "HOLD", "IGNORE"]
    priority: int = Field(..., ge=1, le=10)
    confidence: float = Field(..., ge=0.0, le=1.0)
    urgency: Literal["ACT_NOW", "THIS_WEEK", "MONITOR", "NONE"]
    headline: str
    rationale: str
    key_drivers: List[str]
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    downside_scenario: str
    volatility_rating: Literal["LOW", "MEDIUM", "HIGH"]
    relevant_tags: List[str]
    time_horizon: Literal["DAILY", "WEEKLY", "ROS"]
    category_impact: Dict[str, float]
    skill_score: float
    opportunity_score: Optional[float]
    matchup_score: Optional[float]
    market_score: Optional[float]
```

**Acceptance:**
- [ ] Schema validates with example data

---

### PR 6.5 — Decision Card Builder

**File:** `backend/services/decision_card_builder.py`

**Requirements:**
- Generate `headline`, `rationale`, `downside_scenario` from scores
- Use template strings:
  ```python
  HEADLINES = {
      ("BUY_LOW", " hitter"): "Add — undervalued power with everyday playing time",
      ("SELL_HIGH", "pitcher"): "Trade — ERA well below xERA, regression likely",
      ("STREAM", "hitter"): "Stream today — faces LHP at Coors, elite split",
  }
  ```
- Rationale includes specific metrics: "xwOBA is .380 but wOBA is .320, a gap at the 94th percentile"

**Acceptance:**
- [ ] Headlines generated for all major scenarios
- [ ] Rationale includes specific metric references

---

### PR 6.6 — Filtering Layer

**File:** `backend/routers/fantasy.py` (waiver endpoint)

**Requirements:**
- Suppress recommendations where `confidence < 0.5`
- Suppress where `priority > 7`
- Show "Low confidence — data insufficient" message instead of hiding silently

**Acceptance:**
- [ ] Low-confidence players still appear but with warning
- [ ] No hidden suppressions

---

### PR 7.1 — Decision Logging

**File:** `backend/services/decision_card_builder.py`

**Requirements:**
- Log every decision to `decision_explanations` table (already exists)
- JSON payload includes all sub-scores:
  ```json
  {
    "player_id": 123,
    "skill_score": 82.3,
    "opportunity_adj": -0.12,
    "matchup_boost": 0.08,
    "market_score": 0.3,
    "confidence": 0.61,
    "final_score": 88.1,
    "action": "ADD",
    "tags": ["BUY_LOW", "EVERYDAY_PLAY"]
  }
  ```

**Acceptance:**
- [ ] Logs written for every recommendation
- [ ] Queryable by player_id and date

---

### PR 7.2 — Debug Endpoint

**File:** `backend/routers/fantasy.py` or `backend/routers/admin.py`

**New endpoint:**
```python
@router.get("/debug/player/{bdl_player_id}")
def debug_player_scores(bdl_player_id: int, db: Session = Depends(get_db)):
    return {
        "skill_score": get_skill_score(bdl_player_id),
        "opportunity_score": get_opportunity_score(bdl_player_id),
        "matchup_score": get_matchup_score(bdl_player_id),
        "market_score": get_market_score(bdl_player_id),
        "confidence": get_confidence(bdl_player_id),
        "final_score": compute_final_score(...),
        "tags": generate_tags(...),
    }
```

**Acceptance:**
- [ ] Returns all sub-scores for any player
- [ ] Works with admin API key

---

## EXECUTION ORDER (STRICT)

### Phase 1: Foundation (Sequential)
1. PR 1.1 — Threshold tables
2. PR 1.2 — Config service
3. PR 1.3 — Migrate constants
4. PR 1.4 — Backfill values
5. PR 2.1 — Savant scraper
6. PR 2.2 — Pipeline integration
7. PR 2.3 — Validation + flag
8. PR 2.4 — Backfill script

### Phase 2: Opportunity (Sequential)
9. PR 3.1 — Opportunity schema
10. PR 3.2 — Raw metrics
11. PR 3.3 — Score + Z
12. PR 3.4 — Ingestion hook (LOG ONLY)
13. PR 3.5 — Safe integration

### Phase 3: Parallel Tracks
**Track A: Market**
14. PR 4.1 — Market schema
15. PR 4.2 — Ownership tracking
16. PR 4.3 — Market score
17. PR 4.4 — Confidence gating
18. PR 4.5 — Decision integration

**Track B: Matchup**
19. PR 5.1 — Matchup schema
20. PR 5.1b — Opponent starter hand column
21. PR 5.2 — Data collection
22. PR 5.3 — Basic matchup score
23. PR 5.4 — Configurable weights
24. PR 5.5 — Bounded integration

### Phase 4: Decision Layer (Sequential)
25. PR 6.1 — Extend inputs
26. PR 6.2 — Final composition
27. PR 6.3 — Conflict tags
28. PR 6.4 — DecisionAction schema
29. PR 6.5 — Card builder
30. PR 6.6 — Filtering

### Phase 5: Observability
31. PR 7.1 — Decision logging
32. PR 7.2 — Debug endpoint

---

## TESTING CHECKLIST (Every PR)

- [ ] `venv\Scripts\python -m py_compile <new_or_modified_file>`
- [ ] `venv\Scripts\python -m pytest tests/ -q --tb=short` (must stay green)
- [ ] New functionality has unit tests
- [ ] Feature flag exists (can be disabled without code change)
- [ ] No `datetime.utcnow()` introduced
- [ ] No breaking changes to existing API responses (unless explicitly migrating)

---

## ROLLBACK PLAN

Every PR is reversible:
- **Schema PRs:** Include `DROP TABLE IF EXISTS` in rollback script (do not run unless emergency)
- **Config PRs:** Disable feature flag = instant rollback
- **Integration PRs:** Removing the `if flag_enabled:` block reverts to old behavior
- **DB migrations:** Save rollback SQL in `scripts/rollback_pr_XX.sql`

---

## FINAL NOTES FOR CLAUDE

1. **Do NOT fix the composite_z sum bug or player_board std bug in these PRs.** Those are separate P0s.
2. **Use `bdl_player_id` not `player_id`** in all new tables to match existing schema.
3. **JSONB for threshold_config.value** — don't use FLOAT.
4. **Every feature has a feature flag** — no exceptions.
5. **Log-only before integration** — opportunity (3.4) is the model. Do not touch scoring until 3.5.
6. **Ask before expanding scope** — if a PR seems bigger than 4 hours, flag it and split.

Ready to start with PR 1.1.
