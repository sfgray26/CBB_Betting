# Kimi CLI — K-5 Pipeline Audit: MLB Odds Routing & Advisory Lock Completeness

**Date:** 2026-05-01  
**Agent:** Kimi CLI (Deep Intelligence Unit)  
**Assigned by:** Claude Code (Master Architect)  
**HEAD:** aea2cda (stable/cbb-prod)  
**Files read:** `backend/services/mlb_analysis.py`, `backend/models.py`, `backend/services/daily_ingestion.py`, `backend/services/odds.py`, `backend/fantasy_baseball/daily_lineup_optimizer.py`, `backend/main.py`, `backend/services/bet_tracker.py`, `backend/tournament/fetch_tournament_odds*.py`

---

## 1. mlb_analysis._fetch_mlb_odds() Current Behavior

**File:** `backend/services/mlb_analysis.py` (lines 356–438)

**Surprise finding:** `_fetch_mlb_odds()` does **NOT** call `THE_ODDS_API_KEY`. The docstring at line 8 mentions "The Odds API" but the implementation (lines 356–438) already queries `mlb_odds_snapshot` via SQLAlchemy.

**However, the implementation is broken.** The query uses incorrect SQLAlchemy expressions:

```python
# Line 386–391 — BROKEN: .has() returns a boolean SQL expression, not the abbreviation string
latest_odds_q = (
    db.query(
        MLBOddsSnapshot,
        MLBTeam.away_team_obj.has(abbreviation=MLBTeam.abbreviation),  # ← returns bool, not str
        MLBTeam.home_team_obj.has(abbreviation=MLBTeam.abbreviation),  # ← returns bool, not str
        vendor_priority.label("vendor_priority")
    )
    .join(MLBGameLog, MLBOddsSnapshot.game_id == MLBGameLog.game_id)
    .join(MLBTeam, MLBGameLog.away_team_id == MLBTeam.team_id)        # ← no alias; conflicts with next join
    .join(
        MLBTeam.home_team_obj,                                          # ← alias collision
        MLBGameLog.home_team_id == MLBTeam.team_id
    )
    ...
)
```

**Result unpacking (lines 410–413):**
```python
for row in latest_odds_q:
    odds = row[0]
    away_abbr = row[1]   # ← receives a SQL boolean, not a team abbreviation
    home_abbr = row[2]   # ← same problem
```

**Current return type:** `dict[str, dict]` keyed by `"{away_abbr}@{home_abbr}"` — but because `away_abbr` and `home_abbr` are booleans, the keys are garbage like `"True@True"` or `"False@False"`.

**Graceful degradation:** The method is wrapped in `try/except` and returns `{}` on any exception. In practice, the SQL query likely raises an exception immediately due to the alias collision, so the function silently returns an empty dict every time.

---

## 2. mlb_analysis._calculate_edge() Current Behavior

**File:** `backend/services/mlb_analysis.py` (lines 444–464)

The current `_calculate_edge()` already expects a **flat dict** (not the nested OddsAPI structure) and its logic is **mostly correct**:

```python
def _calculate_edge(self, projection: MLBGameProjection, market: dict) -> float:
    if not market or "ml_home_odds" not in market or market["ml_home_odds"] == 0:
        return 0.0

    try:
        american_odds = market["ml_home_odds"]
        if american_odds < 0:
            market_prob = abs(american_odds) / (abs(american_odds) + 100)
        else:
            market_prob = 100 / (american_odds + 100)
        return round(projection.home_win_prob - market_prob, 4)
    except Exception:
        return 0.0
```

**What it does:**
- Receives `market` dict with key `"ml_home_odds"` (integer, American format)
- Converts American odds to implied probability using the standard formula
- Returns `projection.home_win_prob - market_prob`, rounded to 4 decimals
- Returns `0.0` on any missing data or exception

**Minor issue:** It uses moneyline (`ml_home_odds`) rather than runline, but the docstring says "runline". Since `ml_home_odds` is what we have in `mlb_odds_snapshot`, the docstring is outdated, not the logic.

**Where it's called:** `run_analysis()` line 156:
```python
market_key = f"{proj.away_team}@{proj.home_team}"
market = market_odds.get(market_key, {})
proj.edge = self._calculate_edge(proj, market)
```

---

## 3. Schema Summaries

### 3a. MLBOddsSnapshot (`backend/models.py` lines 1098–1134)

| Column | Type | Notes |
|--------|------|-------|
| id | BigInteger (PK, auto) | Surrogate key |
| odds_id | Integer | BDL MLBBettingOdd.id |
| game_id | Integer (FK → mlb_game_log) | Indexed |
| vendor | String(50) | "draftkings", "fanduel", "pinnacle", etc. |
| snapshot_window | DateTime(tz=True) | Rounded to 30-min bucket |
| spread_home | String(10) | **VARCHAR** — e.g. "1.5" (NOT float) |
| spread_away | String(10) | **VARCHAR** — e.g. "-1.5" |
| spread_home_odds | Integer | American odds |
| spread_away_odds | Integer | American odds |
| ml_home_odds | Integer | American odds |
| ml_away_odds | Integer | American odds |
| total | String(10) | **VARCHAR** — e.g. "8.5" |
| total_over_odds | Integer | American odds |
| total_under_odds | Integer | American odds |
| raw_payload | JSONB | Full BDL dict (dual-write) |

**Unique constraint:** `(game_id, vendor, snapshot_window)` — name `_mlb_odds_game_vendor_window_uc`

**Indexes:** `idx_mlb_odds_vendor_window` on `(vendor, snapshot_window)`

### 3b. MLBGameLog (`backend/models.py` lines 1053–1095)

| Column | Type | Notes |
|--------|------|-------|
| game_id | Integer (PK) | BDL MLBGame.id |
| game_date | Date | ET date, indexed |
| season | Integer | e.g. 2026 |
| season_type | String(20) | "regular" \| "postseason" \| "preseason" |
| status | String(30) | "STATUS_FINAL" etc., indexed |
| home_team_id | Integer (FK → mlb_team.team_id) | NOT NULL |
| away_team_id | Integer (FK → mlb_team.team_id) | NOT NULL |
| home_runs | Integer | NULL pre-game |
| away_runs | Integer | NULL pre-game |
| home_hits | Integer | |
| away_hits | Integer | |
| home_errors | Integer | |
| away_errors | Integer | |
| venue | String(200) | |
| attendance | Integer | NULL pre-game |
| period | Integer | Current/final inning |
| raw_payload | JSONB | Full BDL MLBGame dict |
| ingested_at | DateTime(tz=True) | |
| updated_at | DateTime(tz=True) | |

**Relationships:**
- `home_team_obj` → `MLBTeam` (foreign_keys=[home_team_id])
- `away_team_obj` → `MLBTeam` (foreign_keys=[away_team_id])
- `odds_snapshots` → `MLBOddsSnapshot` (back_populates="game")

### 3c. MLBTeam (`backend/models.py` lines 1026–1051)

| Column | Type | Notes |
|--------|------|-------|
| team_id | Integer (PK) | BDL MLBTeam.id |
| abbreviation | String(10) | **"LAA", "NYY"** — this is the key we need |
| name | String(100) | "Angels" |
| display_name | String(150) | "Los Angeles Angels" |
| short_name | String(50) | "Angels" |
| location | String(100) | "Los Angeles" |
| slug | String(50) | "los-angeles-angels" |
| league | String(10) | "National" \| "American" |
| division | String(10) | "East" \| "Central" \| "West" |
| ingested_at | DateTime(tz=True) | |

---

## 4. Replacement Spec: _fetch_mlb_odds()

**Goal:** Fix the broken DB-query implementation so it returns correct `{"Away@Home": {...}}` dicts.

**Pseudocode (Claude to implement):**

```python
def _fetch_mlb_odds(self) -> dict[str, dict]:
    """
    Fetch current MLB odds from mlb_odds_snapshot table.

    Returns dict keyed by "AwayAbbr@HomeAbbr" -> flat odds dict.
    Queries the most recent snapshot_window per game, preferring vendors by quality.
    """
    # Lazy imports per AGENTS.md rule
    from backend.models import SessionLocal, MLBOddsSnapshot, MLBGameLog, MLBTeam
    from sqlalchemy import func, case

    db = SessionLocal()
    try:
        # 1. Find max snapshot_window per game_id
        max_window_subq = (
            db.query(
                MLBOddsSnapshot.game_id,
                func.max(MLBOddsSnapshot.snapshot_window).label("max_window")
            )
            .group_by(MLBOddsSnapshot.game_id)
            .subquery()
        )

        # 2. Vendor priority: lower number = preferred
        preferred = ["pinnacle", "draftkings", "fanduel", "betmgm", "caesars"]
        vendor_priority = case(
            {v: i for i, v in enumerate(preferred)},
            else_=len(preferred)
        )

        # 3. Aliased joins to avoid table alias collision
        from sqlalchemy.orm import aliased
        AwayTeam = aliased(MLBTeam)
        HomeTeam = aliased(MLBTeam)

        rows = (
            db.query(
                MLBOddsSnapshot,
                AwayTeam.abbreviation.label("away_abbr"),
                HomeTeam.abbreviation.label("home_abbr"),
                vendor_priority.label("vp")
            )
            .join(max_window_subq,
                  (MLBOddsSnapshot.game_id == max_window_subq.c.game_id) &
                  (MLBOddsSnapshot.snapshot_window == max_window_subq.c.max_window))
            .join(MLBGameLog, MLBOddsSnapshot.game_id == MLBGameLog.game_id)
            .join(AwayTeam, MLBGameLog.away_team_id == AwayTeam.team_id)
            .join(HomeTeam, MLBGameLog.home_team_id == HomeTeam.team_id)
            .order_by(MLBOddsSnapshot.game_id, "vp")
            .all()
        )

        result: dict[str, dict] = {}
        seen_games: set[int] = set()

        for odds, away_abbr, home_abbr, _ in rows:
            if odds.game_id in seen_games:
                continue
            seen_games.add(odds.game_id)
            key = f"{away_abbr}@{home_abbr}"
            result[key] = {
                "ml_home_odds": odds.ml_home_odds,
                "ml_away_odds": odds.ml_away_odds,
                "spread_home": odds.spread_home,
                "spread_home_odds": odds.spread_home_odds,
                "spread_away": odds.spread_away,
                "spread_away_odds": odds.spread_away_odds,
                "total": odds.total,
                "vendor": odds.vendor,
            }

        logger.debug("mlb_analysis: loaded odds for %d games from DB", len(result))
        return result
    except Exception as exc:
        logger.warning("mlb_analysis: odds DB fetch failed: %s", exc)
        return {}
    finally:
        db.close()
```

**Key fixes from current broken code:**
1. Use `aliased(MLBTeam)` for AwayTeam and HomeTeam to avoid join alias collision
2. Select actual columns (`AwayTeam.abbreviation`) instead of boolean `.has()` expressions
3. Correct result unpacking: `for odds, away_abbr, home_abbr, _ in rows:`
4. Add `spread_away` and `spread_away_odds` to the returned dict (currently missing)

---

## 5. Replacement Spec: _calculate_edge()

**Current code is already correct** (lines 444–464). No rewrite needed.

**One-line docstring fix:** Change "runline" to "moneyline" in the docstring to match the actual metric used:

```python
def _calculate_edge(self, projection: MLBGameProjection, market: dict) -> float:
    """
    Edge = projected_win_prob - market_implied_win_prob on the moneyline.
    ...
    """
```

**Verification that current logic is sound:**
- Input: `market["ml_home_odds"]` = American odds integer (e.g. -150 or +130)
- Negative odds: `150 / (150 + 100) = 0.600` → 60% implied probability ✓
- Positive odds: `100 / (130 + 100) = 0.435` → 43.5% implied probability ✓
- Returns `0.0` on empty market or zero odds ✓
- Catches all exceptions and returns `0.0` ✓

---

## 6. OddsAPI Violations Inventory

**HARD RULE (CLAUDE.md):** Do NOT route MLB odds through OddsAPI. All MLB odds MUST come from BDL.

### Search: `THE_ODDS_API_KEY` in `backend/`

| # | File | Line | Context | MLB Violation? |
|---|------|------|---------|----------------|
| 1 | `backend/services/odds.py` | 46 | `API_KEY = os.getenv("THE_ODDS_API_KEY")` | **NO** — docstring says "CBB odds". Uses `basketball_ncaab` endpoint. Permitted (CBB archival). |
| 2 | `backend/tournament/fetch_tournament_odds_simple.py` | 37 | `API_KEY = os.getenv("THE_ODDS_API_KEY")` | **NO** — fetches `basketball_ncaab` tournament odds. CBB archival. Permitted. |
| 3 | `backend/tournament/fetch_tournament_odds.py` | 199 | `api_key = os.getenv("THE_ODDS_API_KEY")` | **NO** — same as #2. CBB archival. Permitted. |
| 4 | `backend/services/bet_tracker.py` | 26 | `API_KEY = os.getenv("THE_ODDS_API_KEY")` | **NO** — fetches NCAAB scores (`basketball_ncaab/scores`). CBB archival. Permitted. |
| 5 | `backend/main.py` | 1402 | `api_key = os.getenv("THE_ODDS_API_KEY")` | **NO** — tournament bracket notification job (March 14–20 only). Calls `basketball_ncaab/events`. CBB archival. Permitted. |
| 6 | `backend/fantasy_baseball/daily_lineup_optimizer.py` | 244 | `self._api_key = os.getenv("THE_ODDS_API_KEY", "")` | **YES — VIOLATION** — `fetch_mlb_odds()` calls `ODDS_API_BASE/sports/baseball_mlb/odds`. This is **active MLB odds routing through OddsAPI**, directly violating the HARD RULE. |

### ⚠️ REAL VIOLATION FOUND

**`backend/fantasy_baseball/daily_lineup_optimizer.py` (line 244)**

```python
def fetch_mlb_odds(self, game_date: Optional[str] = None) -> List[MLBGameOdds]:
    resp = requests.get(
        f"{ODDS_API_BASE}/sports/baseball_mlb/odds",
        params={"apiKey": self._api_key, "regions": "us", ...}
    )
```

**Impact:** The lineup optimizer fetches MLB odds from OddsAPI instead of `mlb_odds_snapshot` (which is already populated every 5 minutes by BDL via `_poll_mlb_odds`).

**Fix spec for Claude:**
1. In `daily_lineup_optimizer.py`, replace `fetch_mlb_odds()` with a DB query against `mlb_odds_snapshot` + `mlb_game_log` + `mlb_team` (same join pattern as Task 4).
2. Remove `self._api_key` and `THE_ODDS_API_KEY` dependency from this file entirely.
3. Keep the `MLBGameOdds` dataclass return type — just populate it from DB columns instead of OddsAPI JSON.

---

## 7. Advisory Lock Audit — Gaps Found

### 7a. Lock IDs defined but NO method implementation

| Lock ID | Name | Status |
|---------|------|--------|
| 100_007 | `waiver_scan` | **NO method** — not in `DailyIngestionOrchestrator`, not scheduled |
| 100_008 | `mlb_brief` | **NO method** — not in `DailyIngestionOrchestrator`, not scheduled |
| 100_009 | `openclaw_perf` | **NO method in daily_ingestion.py** — OpenClaw has its own scheduler in `backend/services/openclaw/scheduler.py` |
| 100_010 | `openclaw_sweep` | **NO method in daily_ingestion.py** — same as above |

**Recommendation:** Either remove these four unused entries from `LOCK_IDS` or implement the corresponding methods.

### 7b. Scheduled jobs missing from `run_job()` manual trigger map

The `run_job()` handler map (lines 1039–1066) is missing several actively scheduled jobs. They cannot be triggered manually via `/admin/ingestion/run/{job_id}`:

| Job ID | Scheduled? | In `run_job` handlers? | Has advisory lock? |
|--------|------------|------------------------|-------------------|
| `mlb_odds` | ✅ Every 5 min | ❌ Missing | ✅ 100_001 |
| `clv` | ✅ Daily 11 PM | ❌ Missing | ✅ 100_005 |
| `cleanup` | ✅ Daily 3:30 AM | ❌ Missing | ✅ 100_006 |
| `rolling_z` | ✅ Daily 4 AM | ❌ Missing | ✅ 100_012 |
| `yahoo_adp_injury` | ✅ Every 4 hours | ❌ Missing | ✅ 100_013 |
| `ensemble_update` | ✅ Daily 5 AM | ❌ Missing | ✅ 100_014 |
| `fangraphs_ros` | ✅ Daily 3 AM | ❌ Missing | ✅ 100_012 |
| `statsapi_supplement` | ✅ Daily 2:30 AM | ❌ Missing | ✅ 100_026 |
| `yahoo_id_sync` | ✅ Daily 4:30 AM | ❌ Missing | ✅ 100_034 |
| `bdl_injuries` | ✅ Hourly | ❌ Missing | ✅ 100_033 |
| `probable_pitchers_morning` | ✅ 8:30 AM | ❌ Missing (generic `probable_pitchers` mapped but not time-of-day variants) | ✅ 100_028 |
| `probable_pitchers_afternoon` | ✅ 4:00 PM | ❌ Missing | ✅ 100_028 |
| `probable_pitchers_evening` | ✅ 8:00 PM | ❌ Missing | ✅ 100_028 |
| `valuation_cache` | ✅ Conditional (6 AM) | ❌ Missing | ✅ 100_011 |

**All of these DO acquire their advisory locks** when the scheduler fires them. The gap is only in the **manual trigger path** (`run_job()`).

### 7c. Scheduled job WITHOUT advisory lock (outside daily_ingestion.py)

| Job | Location | Lock? |
|-----|----------|-------|
| MLB nightly analysis (`_run_mlb_analysis_job`) | `backend/main.py` lines 1797–1805 | **NO advisory lock** — runs in main APScheduler, not `DailyIngestionOrchestrator` |

This is a **scheduling gap**: the MLB analysis job is scheduled at 9:00 AM ET but does not use `_with_advisory_lock`. It should either use an advisory lock or be moved into the `DailyIngestionOrchestrator`.

---

## 8. Recommended Implementation Order for Claude

### Priority A — Fix broken MLB odds (affects analysis correctness)

1. **Fix `_fetch_mlb_odds()` in `backend/services/mlb_analysis.py`**
   - Apply the spec from Section 4 (aliased MLBTeam joins, correct column selection)
   - Test: after fix, `_fetch_mlb_odds()` should return non-empty dict on days with games

2. **Fix `fetch_mlb_odds()` in `backend/fantasy_baseball/daily_lineup_optimizer.py`**
   - Replace OddsAPI call with DB query against `mlb_odds_snapshot`
   - This is the **real OddsAPI violation** — higher priority than #1 because it violates the HARD RULE
   - Remove `self._api_key` and `THE_ODDS_API_KEY` dependency from this file

### Priority B — Advisory lock cleanup (operational hygiene)

3. **Add missing jobs to `run_job()` handler map**
   - Map `mlb_odds`, `clv`, `cleanup`, `rolling_z`, `yahoo_adp_injury`, `ensemble_update`, `fangraphs_ros`, `statsapi_supplement`, `yahoo_id_sync`, `bdl_injuries`, `probable_pitchers_morning/afternoon/evening`, `valuation_cache`

4. **Remove or implement unused LOCK_IDS entries**
   - `waiver_scan` (100_007), `mlb_brief` (100_008), `openclaw_perf` (100_009), `openclaw_sweep` (100_010)

### Priority C — Minor fixes

5. **Docstring fix in `_calculate_edge()`** — change "runline" to "moneyline"
6. **Add advisory lock to MLB analysis job** in `backend/main.py` (or document why it doesn't need one)

---

**Sign-off:** Kimi CLI K-5  
**Report saved to:** `reports/2026-05-01-kimi-k5-pipeline-audit.md`
