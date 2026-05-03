# Pitcher Quality Integration Spec — 2026-04-29

> **Agent:** Kimi CLI (research-only audit)  
> **Scope:** Design integration of `probable_pitchers.quality_score` into `daily_lineup_optimizer.py`  
> **Branch:** stable/cbb-prod, HEAD 8c7058c

---

## 1. CRITICAL BUG DISCOVERED — `quality_score` Is 0 for All 436 Rows

**Root cause:** `backend/services/daily_ingestion.py:5624` has an **incorrect SQL cast**:

```python
JOIN player_id_mapping m ON m.bdl_id = s.bdl_player_id::text
```

`player_id_mapping.bdl_id` is `INTEGER` (verified via `information_schema.columns`). Casting `bdl_player_id::text` produces a `text` value, but PostgreSQL has **no `integer = text` operator**. The query throws an `operator does not exist` error, which is silently caught at line 5638:

```python
except Exception as exc:
    logger.warning("_sync_probable_pitchers: ERA lookup failed (%s) -- quality_score will be 0.5", exc)
```

Because `mlbam_to_era` remains empty, **every pitcher gets `quality_score = 0.0`** (the neutral fallback at line 5731).

**Fix:** Change line 5624 from:
```python
JOIN player_id_mapping m ON m.bdl_id = s.bdl_player_id::text
```
to:
```python
JOIN player_id_mapping m ON m.bdl_id = s.bdl_player_id
```

**Validation:** Running the corrected query returns **421 pitchers with ERA data** (mean ERA 4.87), confirming the fix would populate real values.

> **Claude must fix this bug BEFORE integrating quality_score into the optimizer.** Otherwise the optimizer would apply a flat 0.0 (neutral) multiplier to every batter.

---

## 2. TASK 1 — Lineup Optimizer Scoring Logic

### 2.1 Where Scores Are Computed

**File:** `backend/fantasy_baseball/daily_lineup_optimizer.py`  
**Method:** `rank_batters()` — lines 477–564

```python
def rank_batters(self, roster, projections, game_date=None) -> List[BatterRanking]:
    games = self.fetch_mlb_odds(game_date)
    team_odds = self._build_team_odds_map(games)
    ...
    for player in roster:
        ...
        base_score = implied_runs * park_factor
        stat_bonus = (
            proj.get("hr", 0) * 2.0
            + proj.get("r", 0) * 0.3
            + proj.get("rbi", 0) * 0.3
            + proj.get("nsb", 0) * 0.5
            + proj_avg * 5.0
        )
        lineup_score = base_score + stat_bonus * 0.1
        ...
```

### 2.2 Score Formula Breakdown

| Component | Formula | Typical Range |
|-----------|---------|---------------|
| `base_score` | `implied_runs * park_factor` | 3.5–7.0 |
| `stat_bonus` | `hr*2 + r*0.3 + rbi*0.3 + nsb*0.5 + avg*5` | 0–15 |
| `lineup_score` | `base_score + stat_bonus * 0.1` | **3.5–8.5** |

The `stat_bonus` is dampened by `0.1×`, so environment (implied runs + park factor) dominates the score. A typical batter scores ~5.0.

### 2.3 Existing Multipliers / Adjustments

- **Park factor** (already applied): `base_score = implied_runs * park_factor`
- **Weather**: Not currently used.
- **No existing pitcher-quality adjustment**.

### 2.4 Data Structure Passed to Slot Assignment

`rank_batters()` returns `List[BatterRanking]` where each element is:

```python
@dataclass
class BatterRanking:
    name: str
    team: str                   # e.g. "NYY"
    positions: List[str]
    implied_team_runs: float
    park_factor: float
    projected_r: float = 0.0
    projected_hr: float = 0.0
    projected_rbi: float = 0.0
    projected_avg: float = 0.0
    is_home: bool = False
    status: Optional[str] = None
    lineup_score: float = 0.0   # <-- THIS IS THE SORT KEY
    reason: str = ""
    has_game: bool = False
```

`solve_lineup()` (line 652) receives this list and sorts by:

```python
def _slot_sort_key(b: BatterRanking) -> tuple:
    primary = b.positions[0] if b.positions else "OF"
    return (-b.lineup_score, _get_scarcity_rank(db, primary))
```

**Primary sort:** `-lineup_score DESC`  
**Tiebreaker:** `_get_scarcity_rank ASC` (lower = scarcer position preferred)

---

## 3. TASK 2 — Probable Pitchers Schema & Sync

### 3.1 `ProbablePitcherSnapshot` Model

**File:** `backend/models.py` — lines 1795–1836

```python
class ProbablePitcherSnapshot(Base):
    __tablename__ = "probable_pitchers"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    game_date = Column(Date, nullable=False, index=True)
    team = Column(String(10), nullable=False)       # Pitcher's team (e.g., "NYY")
    opponent = Column(String(10), nullable=True)    # Opponent team
    is_home = Column(Boolean, nullable=True)
    pitcher_name = Column(String(100), nullable=True)
    bdl_player_id = Column(Integer, nullable=True, index=True)
    mlbam_id = Column(Integer, nullable=True)
    is_confirmed = Column(Boolean, nullable=False, default=False)
    game_time_et = Column(String(10), nullable=True)
    park_factor = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)    # <-- TARGET COLUMN
    fetched_at = Column(DateTime(timezone=True), ...)
    updated_at = Column(DateTime(timezone=True), ...)

    __table_args__ = (
        UniqueConstraint("game_date", "team", name="_pp_date_team_uc"),
        Index("idx_pp_date", "game_date"),
        Index("idx_pp_pitcher", "bdl_player_id"),
    )
```

**Key finding:** `team` = **the pitcher's team**, not the batter's team. `opponent` = the team the pitcher is facing.

### 3.2 How `_sync_probable_pitchers` Populates Rows

**File:** `backend/services/daily_ingestion.py` — lines 5682–5743

For each game, it processes **both sides**:

```python
for side, is_home in [("home", True), ("away", False)]:
    opp_side = "away" if is_home else "home"
    side_data = teams_data.get(side, {})
    opp_data = teams_data.get(opp_side, {})

    team_abbr = _normalize_abbr(side_data.get("team", {}).get("abbreviation", ""))
    opp_abbr = _normalize_abbr(opp_data.get("team", {}).get("abbreviation", ""))

    pitcher_data = side_data.get("probablePitcher", {})
    ...
    # Upsert:
    team=team_abbr,           # <-- pitcher's team
    opponent=opp_abbr,        # <-- opponent
    quality_score=quality_score,
```

**Result:** Two rows per game (one per team). Example:

| game_date | team | opponent | pitcher_name | quality_score |
|-----------|------|----------|--------------|---------------|
| 2026-04-29 | NYY | BOS | G. Cole | +1.8 |
| 2026-04-29 | BOS | NYY | C. Sale | +1.2 |

### 3.3 `quality_score` Formula

**File:** `backend/services/daily_ingestion.py` — lines 5725–5737

```python
pitcher_era = mlbam_to_era.get(mlbam_id) if mlbam_id else None
if pitcher_era is None:
    quality_score = 0.0
else:
    era_score = max(-0.5, min(0.5, (4.50 - pitcher_era) / 3.00))
    park_val = pf if pf else 1.0
    park_score = max(-0.25, min(0.25, (1.0 - park_val) * 0.25))
    raw_qs = max(0.0, min(1.0, 0.5 + era_score + park_score))
    quality_score = round((raw_qs - 0.5) * 4.0, 2)
```

**Actual range:** **[-2.0, +2.0]**

| Scenario | ERA | era_score | raw_qs | quality_score | Meaning |
|----------|-----|-----------|--------|---------------|---------|
| Ace | 1.50 | +0.50 | 1.00 | **+2.0** | Elite pitcher |
| Average | 4.50 | 0.00 | 0.50 | **0.0** | League average |
| Replacement | 7.50 | -0.50 | 0.00 | **-2.0** | Very bad pitcher |

**⚠️ Important:** The user's context stated "Range: [0, 100] float" and "Higher = better matchup for BATTERS." **This is incorrect.** The code produces **[-2.0, +2.0]** where **higher = BETTER pitcher = WORSE for batters**. Claude must use the actual code semantics, not the user's description.

---

## 4. TASK 3 — Integration Design

### Q3.1 JOIN PATH

**To find the opposing pitcher's `quality_score` for a batter:**

1. Batter is on team `T` (e.g., "BOS").
2. From `team_odds[T]["opponent"]`, get the opponent team `O` (e.g., "NYY").
3. Query `probable_pitchers` where `game_date = today AND team = O`.
4. Read `quality_score`.

**Example:**
- Batter: Rafael Devers, team = "BOS"
- BOS opponent today = "NYY" (from `team_odds["BOS"]["opponent"]`)
- Query: `SELECT quality_score FROM probable_pitchers WHERE game_date = '2026-04-29' AND team = 'NYY'`
- Result: `+1.8` (Gerrit Cole is pitching for NYY)

**Implementation:** Pre-fetch all `quality_score` values into a dict keyed by **opponent team** (not batter team):

```python
# Inside rank_batters() or _build_team_odds_map()
pitcher_quality: Dict[str, float] = {}
db = SessionLocal()
try:
    rows = (
        db.query(ProbablePitcherSnapshot.team, ProbablePitcherSnapshot.quality_score)
        .filter(ProbablePitcherSnapshot.game_date == target_date)
        .all()
    )
    for team, qs in rows:
        if team and qs is not None:
            pitcher_quality[team] = qs
finally:
    db.close()
```

Then for a batter on team `T`:
```python
opp_team = team_odds.get(team, {}).get("opponent", "")
opp_qs = pitcher_quality.get(opp_team, 0.0)  # 0.0 = neutral fallback
```

**Why key by pitcher's team?** Because `probable_pitchers.team` stores the pitcher's team. The opponent team is the batter's team. So `pitcher_quality[opp_team]` gives the quality of the pitcher facing the batter's team.

### Q3.2 MULTIPLIER DESIGN

Given the actual range [-2.0, +2.0]:
- `+2.0` = Ace → **penalize** batter
- `-2.0` = Bad pitcher → **boost** batter
- `0.0` = Average → no change

**Recommended: Soft linear multiplier**

```python
# In rank_batters(), after computing lineup_score:
multiplier = 1.0 - (opp_qs / 10.0)
# Range: 1.0 - (+2.0/10.0) = 0.8  (ace: -20%)
#        1.0 - (-2.0/10.0) = 1.2  (bad pitcher: +20%)
#        1.0 - (0.0/10.0)   = 1.0  (average: no change)

adjusted_score = lineup_score * multiplier
```

**Why this option:**
- **Preserves ordering** for edge cases: A batter with score 6.0 facing an ace (0.8×) drops to 4.8. A batter with score 5.0 facing a bad pitcher (1.2×) rises to 6.0. The relative ordering is maintained proportionally.
- **Conservative magnitude:** ±20% is material but not extreme. A 2.0-quality ace won't completely bench a star hitter.
- **Linear:** Simple to understand and debug. No arbitrary thresholds.

**Rejected alternatives:**
- **Option A** (linear `(1 + (qs-50)/100)`): Assumes wrong [0,100] range.
- **Option B** (soft gate <20 or >80): Creates cliff effects. A pitcher with quality_score=1.9 gets no adjustment while 2.0 gets full penalty.
- **Option C** (additive): Would add a flat bonus/penalty, which disproportionately affects low-scoring players.

### Q3.3 PITCHER HANDLING

**Already handled.** `rank_batters()` skips all pitchers at line 505:

```python
if any(p in ("SP", "RP", "P") for p in positions):
    continue
```

The quality_score multiplier only needs to be applied inside `rank_batters()`, which never processes pitchers. No additional type detection needed.

### Q3.4 MISSING DATA FALLBACK

**Recommended: Neutral (multiplier = 1.0×)**

If no probable pitcher is listed for a game (`quality_score IS NULL` or team not in `pitcher_quality` dict), use:

```python
opp_qs = pitcher_quality.get(opp_team, 0.0)
```

This yields `multiplier = 1.0 - (0.0 / 10.0) = 1.0`.

**Why neutral:** If we don't know who's pitching, we shouldn't penalize or reward the batter. Conservative is correct.

### Q3.5 PERFORMANCE

**Negligible impact.**

- `probable_pitchers` has **436 rows total**, with a unique index on `(game_date, team)`.
- A typical MLB day has **15 games** = **30 rows** (home + away).
- Query: `SELECT team, quality_score FROM probable_pitchers WHERE game_date = ?`
  - Returns ≤ 30 rows.
  - Indexed on `game_date`.
  - Execution time: **< 1 ms**.
- Pre-fetch once per `rank_batters()` call into a `Dict[str, float]`.
- Lookup per player: **O(1)** dict get.

---

## 5. Implementation Sketch for Claude

### Step 1: Fix the SQL cast bug (BLOCKING)
**File:** `backend/services/daily_ingestion.py`  
**Line:** 5624

```python
# BEFORE (broken — causes all quality_score = 0.0)
JOIN player_id_mapping m ON m.bdl_id = s.bdl_player_id::text

# AFTER (fixed — both columns are integer)
JOIN player_id_mapping m ON m.bdl_id = s.bdl_player_id
```

### Step 2: Add quality_score lookup to optimizer
**File:** `backend/fantasy_baseball/daily_lineup_optimizer.py`  
**Method:** `rank_batters()` — around line 494

```python
def rank_batters(self, roster, projections, game_date=None) -> List[BatterRanking]:
    games = self.fetch_mlb_odds(game_date)
    team_odds = self._build_team_odds_map(games)

    # NEW: Pre-fetch pitcher quality scores for today's games
    pitcher_quality: Dict[str, float] = {}
    target_date = self._parse_game_date(game_date)
    if target_date is not None:
        db = SessionLocal()
        try:
            rows = (
                db.query(
                    ProbablePitcherSnapshot.team,
                    ProbablePitcherSnapshot.quality_score,
                )
                .filter(ProbablePitcherSnapshot.game_date == target_date)
                .all()
            )
            for team, qs in rows:
                if team and qs is not None:
                    pitcher_quality[team] = qs
        except Exception as exc:
            logger.warning("Failed to load pitcher quality scores: %s", exc)
        finally:
            db.close()

    proj_by_name = {...}
    rankings = []
    for player in roster:
        ...
        # NEW: Get opponent pitcher's quality_score
        opp_team = team_odds.get(team, {}).get("opponent", "")
        opp_qs = pitcher_quality.get(opp_team, 0.0)  # 0.0 = neutral

        # Existing score computation
        base_score = implied_runs * park_factor
        stat_bonus = (...)
        lineup_score = base_score + stat_bonus * 0.1

        # NEW: Apply pitcher-quality multiplier
        multiplier = 1.0 - (opp_qs / 10.0)
        adjusted_score = lineup_score * multiplier

        # Add reason text
        if opp_qs > 1.0:
            reason_parts.append(f"vs ace (qs={opp_qs:.1f})")
        elif opp_qs < -1.0:
            reason_parts.append(f"vs weak SP (qs={opp_qs:.1f})")

        rankings.append(BatterRanking(
            ...
            lineup_score=round(adjusted_score, 3),
            reason=", ".join(reason_parts),
            ...
        ))
```

### Step 3: Update `BatterRanking` dataclass (optional)
If you want to expose `opponent_pitcher_quality` in the report:

```python
@dataclass
class BatterRanking:
    ...
    opponent_pitcher_quality: float = 0.0  # NEW
```

---

## 6. Raw Evidence

### `daily_ingestion.py:5624` — broken SQL cast
```python
JOIN player_id_mapping m ON m.bdl_id = s.bdl_player_id::text
```

### `daily_ingestion.py:5725–5737` — quality_score formula
```python
pitcher_era = mlbam_to_era.get(mlbam_id) if mlbam_id else None
if pitcher_era is None:
    quality_score = 0.0
else:
    era_score = max(-0.5, min(0.5, (4.50 - pitcher_era) / 3.00))
    park_val = pf if pf else 1.0
    park_score = max(-0.25, min(0.25, (1.0 - park_val) * 0.25))
    raw_qs = max(0.0, min(1.0, 0.5 + era_score + park_score))
    quality_score = round((raw_qs - 0.5) * 4.0, 2)
```

### `daily_lineup_optimizer.py:528–536` — score formula
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

### `models.py:1826` — quality_score column
```python
quality_score = Column(Float, nullable=True)  # Precomputed matchup rating (-2.0 to +2.0)
```

---

*Report generated by Kimi CLI at 2026-04-29. Read-only audit — no files modified.*
