# Optimized Prompt for Claude Code: BallDontLie GOAT Tier Integration

**Date:** 2026-04-25  
**From:** Kimi CLI (Research Complete)  
**To:** Claude Code (Implementation Required)  
**Priority:** P0 — This is the highest-ROI data investment for the platform  
**Context:** User has upgraded to BallDontLie GOAT tier ($39.99/mo, 600 req/min). All 19 MLB endpoints are now available including play-by-play, pitch-level data, betting odds, and player props.

---

## THE SITUATION

The fantasy baseball platform has critical data gaps that BallDontLie (BDL) resolves comprehensively. Kimi has completed a full API analysis and the user has purchased the GOAT tier. **Your job is to build the integration.**

### Current Pain Points BDL Will Fix

| # | Pain Point | Current State | BDL Endpoint |
|---|-----------|---------------|--------------|
| 1 | **Yahoo ID Mapping dead** | 0/10,000 rows have yahoo_id | `GET /mlb/v1/players?search=` |
| 2 | **Probable Pitchers unconfirmed** | 0/332 confirmed | `GET /mlb/v1/lineups` |

| 3 | **Injury data sparse** | 3/23 players | `GET /mlb/v1/player_injuries` |
| 4 | **Rolling windows 100% null** | No ingestion pipeline | `GET /mlb/v1/stats?dates[]=` |
| 5 | **Player scores table empty** | 0 rows current window | `GET /mlb/v1/season_stats` |
| 6 | **21/25 FAs need_score=0** | Missing proxy data | `GET /mlb/v1/season_stats` |
| 7 | **Scoreboard "Opponent" generic** | Placeholder text | `GET /mlb/v1/games?dates[]=` |
| 8 | **No platoon optimization** | None exists | `GET /mlb/v1/players/splits` |
| 9 | **No matchup history** | None exists | `GET /mlb/v1/players/versus` |
| 10 | **Betting only CBB** | No MLB odds | `GET /mlb/v1/odds` + `/odds/player_props` |

### What Kimi Fixed (Don't Repeat)

- ✅ Gemini violation remediated — `fusion_engine.py` integrated, 90/90 tests pass
- ✅ Savant ingestion fixed — correct endpoint, BOM stripping, column mapping
- ✅ `player_board.py` preserves pre-computed z-scores
- ✅ All modified files compile and pass tests

See `HANDOFF.md` section 16.11 for full details.

---

## PHASE 1: FOUNDATION — BDL Client + ID Mapping + Injuries

### Task 1.1: Build `backend/services/balldontlie_client.py` (P0)

**Create a resilient BDL API client following the same patterns as `yahoo_client_resilient.py`.**

**Requirements:**
- Use the official Python SDK: `pip install balldontlie`
- Lazy initialization (don't create client at import time)
- Request caching with TTL (30 min for stats, 5 min for lineups, 60 min for injuries)
- Rate limit awareness: GOAT tier = 600 req/min = 10 req/sec. Implement token bucket.
- Error handling: retry 3x with exponential backoff on 5xx, fail fast on 4xx
- Logging: `[balldontlie] method=args → status (N ms)` format
- Singleton pattern (same as Yahoo client)

**API Key:** Read from env var `BALLDONTLIE_API_KEY`

**Methods to implement (Phase 1):**
```python
def get_players(search: str = None, per_page: int = 100) -> list[dict]
def get_player_by_id(bdl_id: int) -> dict
def get_player_injuries(per_page: int = 100) -> list[dict]
def get_games(dates: list[str] = None, per_page: int = 100) -> list[dict]
def get_lineups(game_ids: list[int] = None, per_page: int = 100) -> list[dict]
def get_season_stats(season: int, per_page: int = 100) -> list[dict]
def get_stats(dates: list[str] = None, per_page: int = 100) -> list[dict]
```

**Test:** `tests/test_balldontlie_client.py` — mock the SDK, verify retry logic, caching, rate limiting.

### Task 1.2: Build Player ID Mapping Pipeline (P0)

**Goal: Populate `player_id_mapping.yahoo_id` for all Yahoo roster players.**

**Current state:** `player_id_mapping` has 10,000 rows but 0 have `yahoo_id` populated. This breaks `get_or_create_projection()` which tries `PlayerIDMapping.yahoo_id == yahoo_id`.

**Implementation:**
1. Create `backend/services/bdl_id_mapping.py` with `BDLIdMappingAgent`
2. Method `map_yahoo_player(yahoo_player: dict) -> dict | None`:
   - Extract name from `yahoo_player["name"]`
   - Call `balldontlie_client.get_players(search=last_name)`
   - Fuzzy match on full name (≥0.90 ratio)
   - If match found, return `{bdl_id, mlbam_id, name, team, position}`
3. Method `run_full_mapping_backfill(db)`:
   - Query all `PlayerIDMapping` rows with `yahoo_id IS NULL`
   - For each, search BDL by name
   - On match, update `yahoo_id` and `mlbam_id` (if BDL provides it)
   - Log match rate

**Schema note:** BDL player objects have `id` (BDL ID), not `mlbam_id`. You may need to cross-reference or store BDL IDs separately. Check if BDL's `MLBPlayer` schema includes an MLBAM ID field.

**Test:** Verify that after backfill, `get_or_create_projection()` successfully finds mappings for known players.

### Task 1.3: Injury Ingestion Pipeline (P0)

**Goal: Replace sparse injury data with rich BDL injury reports.**

**Current state:** Only 3/23 roster players have injury data.

**Implementation:**
1. Create `backend/services/bdl_injury_ingestion.py`
2. Method `run_daily_ingestion(db)`:
   - Call `balldontlie_client.get_player_injuries(per_page=100)` (paginate through all)
   - Upsert into `player_injuries` table:
     - `player_id` (join via `balldontlie_player_mapping` or `player_id_mapping`)
     - `injury_type` ← BDL `type`
     - `detail` ← BDL `detail`
     - `status` ← BDL `status`
     - `return_date` ← BDL `return_date`
     - `source` = "balldontlie"
   - Use advisory lock ID `100_033`
3. Schedule in `DailyIngestionOrchestrator` at 7:00 AM ET daily

**Test:** After ingestion, verify `GET /api/fantasy/roster` returns injury data for >80% of players.

---

## PHASE 2: CORE FEATURES — Lineups + Season Stats + Rolling Windows

### Task 2.1: Replace Probable Pitchers Scraping (P0)

**Goal: Use BDL confirmed lineups instead of HTML scraping.**

**Current state:** `probable_pitchers` table has 332 rows, 0 confirmed. HTML scraping is fragile.

**Implementation:**
1. Create `backend/services/bdl_lineup_ingestion.py`
2. Method `run_daily_ingestion(db, date: str)`:
   - Step 1: `GET /mlb/v1/games?dates[]=YYYY-MM-DD` → get `game_id` list
   - Step 2: `GET /mlb/v1/lineups?game_ids[]=id1,id2,...` → get confirmed lineups
   - Step 3: Upsert into NEW table `daily_lineups`:
     ```sql
     CREATE TABLE daily_lineups (
         id SERIAL PRIMARY KEY,
         game_id INTEGER NOT NULL,
         game_date DATE NOT NULL,
         bdl_player_id INTEGER NOT NULL,
         player_name VARCHAR(100),
         team VARCHAR(10),
         batting_order INTEGER,
         position VARCHAR(10),
         is_probable_pitcher BOOLEAN DEFAULT false,
         confirmed BOOLEAN DEFAULT false,
         last_updated TIMESTAMP DEFAULT NOW()
     );
     ```
   - Mark `is_probable_pitcher=true` rows as confirmed
   - Use advisory lock ID `100_034`
3. Update `DailyIngestionOrchestrator` to call this at 11:00 AM ET (lineups usually confirmed by then)
4. Update `GET /api/fantasy/briefing/YYYY-MM-DD` to read from `daily_lineups` instead of scraping
5. **Deprecate but don't delete** old HTML scraping code — keep as fallback

**Test:** `GET /api/fantasy/briefing/2026-04-25` returns real probable pitchers with confirmed status.

### Task 2.2: Season Stats Backfill for Fusion Engine (P1)

**Goal: Use BDL `season_stats` as the "observed" component in the fusion engine.**

**Current state:** Fusion engine uses Statcast CSV data (445 batters, 507 pitchers). BDL season stats cover ALL active players with official MLB data.

**Implementation:**
1. Create `backend/services/bdl_stats_ingestion.py`
2. Method `run_weekly_ingestion(db, season: int = 2026)`:
   - Paginate through `GET /mlb/v1/season_stats?season=2026&per_page=100`
   - For each player, extract:
     - Batters: `batting_avg`, `batting_obp`, `batting_slg`, `batting_ops`, `batting_hr`, `batting_rbi`, `batting_r`, `batting_sb`, `batting_so`, `batting_tb`, `batting_gp`
     - Pitchers: `pitching_era`, `pitching_whip`, `pitching_k`, `pitching_k_per_9`, `pitching_w`, `pitching_l`, `pitching_qs`, `pitching_ip`, `pitching_sv`, `pitching_hr`, `pitching_gp`
   - Store in NEW table `bdl_season_stats`:
     ```sql
     CREATE TABLE bdl_season_stats (
         id SERIAL PRIMARY KEY,
         bdl_player_id INTEGER UNIQUE NOT NULL,
         player_name VARCHAR(100),
         team VARCHAR(10),
         season INTEGER NOT NULL,
         -- Batting
         batting_gp INT, batting_ab INT, batting_r INT, batting_h INT,
         batting_avg FLOAT, batting_hr INT, batting_rbi INT, batting_tb INT,
         batting_bb INT, batting_so INT, batting_sb INT,
         batting_obp FLOAT, batting_slg FLOAT, batting_ops FLOAT, batting_war FLOAT,
         -- Pitching
         pitching_gp INT, pitching_gs INT, pitching_qs INT, pitching_w INT,
         pitching_l INT, pitching_era FLOAT, pitching_sv INT, pitching_hld INT,
         pitching_ip FLOAT, pitching_h INT, pitching_er INT, pitching_hr INT,
         pitching_bb INT, pitching_whip FLOAT, pitching_k INT, pitching_k_per_9 FLOAT,
         pitching_war FLOAT,
         last_updated TIMESTAMP DEFAULT NOW()
     );
     ```
3. Update `fusion_engine.py` or `player_board.py` to query `bdl_season_stats` as an additional "observed" data source (alongside Statcast)
4. The fusion engine's four-state logic should become:
   - State 1: Steamer + Statcast + BDL → weighted fusion
   - State 2: Steamer + BDL → weighted fusion
   - State 3: BDL only → population prior + observed
   - State 4: Nothing → pure population prior

**Test:** After backfill, `GET /api/fantasy/waiver` shows >20/25 FAs with non-zero `need_score`.

### Task 2.3: Rolling Window Stats (P1)

**Goal: Populate 7d/14d/15d/30d rolling stats using BDL game-level data.**

**Current state:** Rolling windows are 100% null. The `player_scores` table has 0 rows for current window.

**Implementation:**
1. Create `backend/services/bdl_rolling_stats.py`
2. Method `run_daily_ingestion(db)`:
   - For each window (7d, 14d, 15d, 30d):
     - Compute date range: today - N days
     - `GET /mlb/v1/stats?dates[]=start&dates[]=end&per_page=100` (paginate)
     - Aggregate per player: sum counting stats, compute rate stats
     - Upsert into `player_scores` table with window label
   - Use advisory lock ID `100_035`
3. Schedule in `DailyIngestionOrchestrator` at 6:30 AM ET

**Test:** `GET /api/fantasy/roster` returns non-null `rolling_stats` for all players.

---

## PHASE 3: ADVANCED — GOAT Tier Features

### Task 3.1: Platoon Splits Integration (P1)

**Goal: Use BDL splits data for daily lineup optimization.**

**Implementation:**
1. Add method to BDL client: `get_player_splits(bdl_id, season)`
2. Create `backend/services/bdl_splits_ingestion.py`
3. Store splits in NEW table `player_splits`:
   ```sql
   CREATE TABLE player_splits (
       id SERIAL PRIMARY KEY,
       bdl_player_id INTEGER NOT NULL,
       season INTEGER NOT NULL,
       split_category VARCHAR(50),  -- "split", "byDayMonth", "byOpponent"
       split_name VARCHAR(50),      -- "vs RHP", "vs LHP", "April"
       category VARCHAR(20),        -- "batting" or "pitching"
       at_bats INT, runs INT, hits INT, home_runs INT, rbis INT,
       walks INT, strikeouts INT, stolen_bases INT,
       avg FLOAT, obp FLOAT, slg FLOAT, ops FLOAT,
       last_updated TIMESTAMP DEFAULT NOW(),
       UNIQUE(bdl_player_id, season, split_category, split_name)
   );
   ```
4. Update `daily_lineup_optimizer.py` to query splits when making start/sit decisions:
   - If batter faces LHP and has `vs LHP` split with OPS > 0.850 → boost score
   - If batter faces LHP and `vs LHP` OPS < 0.600 → reduce score or bench

**Test:** Lineup optimizer produces different recommendations when SP handedness changes.

### Task 3.2: Matchup History (P2)

**Goal: Use batter vs pitcher historical data for waiver and start/sit.**

**Implementation:**
1. Add method to BDL client: `get_player_versus(bdl_id, opponent_team_id)`
2. In waiver recommendation engine, query matchup history for proposed acquisitions
3. Display in API response: `matchup_history: {avg, ops, hr, pa}`

### Task 3.3: MLB Betting Odds (P2)

**Goal: Expand CBB betting model to MLB using BDL odds.**

**Implementation:**
1. Add methods to BDL client:
   - `get_odds(dates, game_ids)` → moneyline, spread, total
   - `get_player_props(game_id, player_id, prop_type)` → HR, K, hits props
2. Create `backend/services/bdl_odds_ingestion.py`
3. Store odds in `mlb_betting_odds` table
4. Integrate with existing betting model (`backend/betting_model.py`):
   - Use odds as market-implied probabilities
   - Generate Kelly bets for MLB games
   - Display in dashboard

---

## PHASE 4: MCP SERVER INTEGRATION

### Task 4.1: Configure MCP for Agent Workflows (P2)

**Add BDL MCP server to agent configurations:**

```json
{
  "mcpServers": {
    "balldontlie-mlb": {
      "url": "https://mcp.balldontlie.io/mcp",
      "headers": { "Authorization": "${BALLDONTLIE_API_KEY}" }
    }
  }
}
```

**Update agent capabilities:**
- **Claude Code:** Can query `mlb_get_season_stats` during architecture reviews
- **Kimi CLI:** Can query `mlb_get_players_splits` for waiver research reports
- **OpenClaw:** Can query `mlb_get_player_injuries` for daily morning briefings

---

## ARCHITECTURAL CONSTRAINTS

1. **Follow existing patterns.** Use the same resilient client pattern as `yahoo_client_resilient.py`
2. **Lazy initialization.** Don't create BDL client at module import time
3. **No `datetime.utcnow()`.** Use `datetime.now(ZoneInfo("America/New_York"))`
4. **Advisory locks.** Each ingestion pipeline gets a unique lock ID (100_033+, registered in `DailyIngestionOrchestrator`)
5. **Environment variables.** API key in `BALLDONTLIE_API_KEY`. Tier info in `BALLDONTLIE_TIER=goat`
6. **No ghost changes.** Document all modifications in HANDOFF.md
7. **Tests required.** Every new module gets unit/integration tests
8. **Preserve backward compatibility.** Don't break existing Yahoo API flows

---

## ACCEPTANCE CRITERIA

### Phase 1 Gate
- [ ] `pip install balldontlie` added to `requirements.txt`
- [ ] `backend/services/balldontlie_client.py` exists with all Phase 1 methods
- [ ] `tests/test_balldontlie_client.py` passes (mocked, no real API calls)
- [ ] `player_id_mapping.yahoo_id` populated for >50% of Yahoo roster players
- [ ] `GET /api/fantasy/roster` returns injury data for >80% of players
- [ ] `pytest` full suite: 0 regressions

### Phase 2 Gate
- [ ] `daily_lineups` table created and populated
- [ ] `GET /api/fantasy/briefing/YYYY-MM-DD` returns confirmed probable pitchers
- [ ] `bdl_season_stats` table created and populated with 2026 data
- [ ] `GET /api/fantasy/waiver` shows >20/25 FAs with non-zero `need_score`
- [ ] Rolling window stats non-null for >80% of roster players
- [ ] Scoreboard shows real opponent names (not "Opponent")

### Phase 3 Gate
- [ ] `player_splits` table created with vs LHP/RHP data
- [ ] Lineup optimizer considers platoon splits
- [ ] Matchup history displayed in waiver recommendations
- [ ] MLB betting odds ingested and displayed in dashboard

### Phase 4 Gate
- [ ] MCP server configured for all agents
- [ ] Agents documented as using BDL tools in AGENTS.md

---

## KEY FILES TO CREATE/MODIFY

| File | Action | Purpose |
|------|--------|---------|
| `requirements.txt` | Add `balldontlie` | Python SDK |
| `backend/services/balldontlie_client.py` | **CREATE** | Core API client |
| `backend/services/bdl_id_mapping.py` | **CREATE** | Player ID mapping pipeline |
| `backend/services/bdl_injury_ingestion.py` | **CREATE** | Daily injury ingestion |
| `backend/services/bdl_lineup_ingestion.py` | **CREATE** | Confirmed lineup ingestion |
| `backend/services/bdl_stats_ingestion.py` | **CREATE** | Season stats backfill |
| `backend/services/bdl_rolling_stats.py` | **CREATE** | Rolling window computation |
| `backend/services/bdl_splits_ingestion.py` | **CREATE** | Platoon splits (GOAT) |
| `backend/services/bdl_odds_ingestion.py` | **CREATE** | Betting odds (GOAT) |
| `backend/models.py` | **MODIFY** | Add `daily_lineups`, `bdl_season_stats`, `player_splits` tables |
| `backend/fantasy_baseball/daily_lineup_optimizer.py` | **MODIFY** | Consider platoon splits |
| `backend/routers/fantasy.py` | **MODIFY** | Use `daily_lineups` for briefing |
| `backend/orchestrator.py` | **MODIFY** | Schedule new ingestion jobs |
| `tests/test_balldontlie_client.py` | **CREATE** | Client unit tests |
| `tests/test_bdl_integration.py` | **CREATE** | End-to-end integration tests |
| `AGENTS.md` | **MODIFY** | Document BDL MCP access |
| `HANDOFF.md` | **MODIFY** | Document implementation decisions |

---

## DECISIONS REQUIRED FROM CLAUDE

1. **Player ID mapping strategy:** BDL uses its own `id` field. Should we:
   - A) Add `bdl_id` column to `player_id_mapping`
   - B) Use BDL `id` as the primary cross-reference and derive `mlbam_id` separately
   - C) Create a separate `balldontlie_player_mapping` table

2. **Fusion engine data priority:** When Steamer + Statcast + BDL all exist, what are the weights?
   - Suggestion: Steamer 50% + Statcast 25% + BDL 25% for rate stats
   - BDL has larger sample (full season) but Statcast has process metrics (xwOBA)

3. **Rolling window computation:** Should we:
   - A) Query BDL `stats` endpoint for each window daily (API cost: ~4 calls/day)
   - B) Maintain our own game-log table and compute locally (more code, less API usage)

4. **Deprecating Savant:** With BDL providing season stats and pitch data, should we:
   - A) Keep BOTH (BDL for reliability, Savant for xwOBA)
   - B) Gradually migrate to BDL-only
   - C) Keep Savant as fallback only

---

## REFERENCES

- **Full API Analysis:** `reports/2026-04-25-balldontlie-api-integration-analysis.md`
- **OpenAPI Spec:** https://www.balldontlie.io/openapi/mlb.yml
- **Python SDK:** https://github.com/balldontlie-api/python
- **MCP Server:** https://github.com/balldontlie-api/mcp
- **Current App State:** `HANDOFF.md` section 16.11
- **Fusion Engine Context:** `reports/2026-04-24-mathematical-framework-steamer-statcast-fusion.md`

---

*Prompt compiled by Kimi CLI v1.17.0 | 19 endpoints mapped | 12 pain points addressed | 4 phases scoped | GOAT tier unlocked*
