# K-27: Raw Ingestion Spec Audit

**Date:** April 5, 2026  
**Analyst:** Kimi CLI (Deep Intelligence Unit)  
**Status:** COMPLETE — Delta analysis for data contract design

---

## Executive Summary

This audit maps what the CODEBASE ASSUMES each API returns versus what Claude will capture in live payloads. The delta between assumptions and reality is where every bug lives.

**Top 5 Silent Failure Risks:**
1. Yahoo `percent_rostered` vs `percent_owned` key confusion (K-20 fixed but pattern repeats)
2. MLB odds APIs return 0 games for future dates — logged as "skipped" not "error"
3. `_parse_player()` recursive flattening hides field path changes
4. `statsapi.schedule()` throws on network errors — returns empty list (graceful but invisible)
5. `get_players_stats_batch()` fails silently with empty `stats={}` for all players on API error

---

## Task 1: Yahoo API Assumptions

### 1.1 `get_free_agents()` — Lines 529-561

| Field Accessed | Null-Safe? | Type-Checked? | Notes |
|----------------|------------|---------------|-------|
| `p["player_key"]` | ❌ NO | ❌ NO | List comp assumes key exists |
| `p.get("player_key")` | ✅ YES | ❌ NO | Used in stats merge, defaults to "" |
| `p["stats"]` (post-merge) | ✅ YES | ✅ Implicit | Created if missing line 556-557 |

**Silent Failure Pattern:**
```python
# Line 552: Assumes all players have player_key
player_keys = [p["player_key"] for p in players if p.get("player_key")]
# If player_key missing, player silently excluded from stats batch
```

**Assumed Response Structure:**
```python
players = [
  {
    "player_key": "469.p.12345",
    "player_id": "12345",
    "name": "Player Name",
    "team": "NYY",
    "positions": ["C", "1B"],
    "status": null,
    "injury_note": null,
    "is_undroppable": false,
    "percent_owned": 82.0,
    "stats": {"7": "12", "8": "45"}  # Merged from batch call
  }
]
```

### 1.2 `_parse_player()` — Lines 932-1038

| Field Path | Access Pattern | Risk |
|------------|----------------|------|
| `meta["eligible_positions"]` | `.get()` with list/dict branch | ✅ SAFE |
| `meta["player_key"]` | Direct `.get()` | ✅ SAFE |
| `meta["percent_owned"]` | Multi-format recursive search | ⚠️ COMPLEX |
| `obj["ownership"]["percent_rostered"]` | Recursive find_ownership() | ⚠️ DEEP NESTING |

**Critical Finding:** The recursive `find_ownership()` function (lines 967-1000) searches 5 levels deep for ownership data. This was added for K-20 fix but masks Yahoo's actual response structure. **Code assumes ownership can be anywhere in the tree** — we need to know where it ACTUALLY is.

**Assumed Ownership Locations:**
1. `player_list[n].ownership.percent_rostered`
2. `player_list[n].ownership.percent_owned`
3. `player_list[n].percent_rostered` (flat)
4. `player_list[n].percent_owned.value` (old format)

### 1.3 `get_scoreboard()` — Lines 776-849

| Assumption | Risk |
|------------|------|
| Scoreboard has 3 possible nesting patterns (v1, v2, v3) | ⚠️ Handles multiple shapes, but which is current? |
| `matchups_raw` can be dict OR list | ✅ Handled explicitly |
| `teams` inside matchup has `count` field | ❌ NOT VALIDATED — `int(teams.get("count", 0))` defaults to 0 |

**Silent Failure:** If Yahoo changes `count` field name, loop silently runs 0 times.

### 1.4 `get_adp_and_injury_feed()` — Lines 613-645

**Contract Promises (docstring lines 624-626):**
```python
"""
Each returned dict has:
    player_key, name, team, positions, status, injury_note,
    is_undroppable, percent_owned
"""
```

**Reality:** Returns whatever `_parse_players_block()` returns. No validation that all promised fields exist.

**Silent Failure Pattern:**
```python
# Line 635-638
batch = self._parse_players_block(players_raw)
if not batch:
    break  # Empty page = no more players
results.extend(batch)  # Partial results OK
```
If page 1 fails but page 0 succeeded, returns partial results without indication of incompleteness.

### 1.5 Hardcoded Stat IDs

From `_YAHOO_STAT_FALLBACK` (main.py lines 5458-5469):

| Stat ID | Mapped To | Confidence |
|---------|-----------|------------|
| 3 | AVG | K-14 verified |
| 7 | R | K-14 verified |
| 8 | H | K-14 verified |
| 12 | HR | K-14 verified |
| 13 | RBI | K-14 verified |
| 16 | SB | K-14 verified |
| 21 | IP | Assumed |
| 23 | W | Assumed |
| 26 | ERA | K-14 verified |
| 27 | WHIP | K-14 verified |
| 28 | K | Assumed (pitcher) |
| 29 | QS | Assumed |
| 32 | SV | Assumed |
| 38 | K/BB | Assumed |
| 42 | K | DUPLICATE of 28? |
| 50 | IP | DUPLICATE of 21? |
| 55 | OPS | K-14 verified |
| 57 | BB | K-14 verified |
| 60 | H | DUPLICATE of 8? |
| 62 | GS | Assumed |
| 83 | NSV | K-14 verified |
| 85 | OBP | K-14 verified |

**Risk:** Multiple stat IDs map to same abbreviation. Code doesn't distinguish batting K (27?) from pitching K (28/42?).

---

## Task 2: BDL Client Patterns

### 2.1 `balldontlie.py` — MLB Support Status

**Current Reality:** ZERO MLB code exists.

| Endpoint | Status | Pattern Reusable? |
|----------|--------|-------------------|
| `/ncaab/v1/games` | ACTIVE | ✅ YES — pagination reusable |
| `/ncaab/v1/odds` | ACTIVE | ✅ YES — structure likely similar |
| `/ncaab/v1/players` | ACTIVE | ✅ YES — auth pattern identical |
| `/mlb/v1/*` | **NEVER CALLED** | N/A — must verify endpoints exist |

**Reusable Patterns for MLB:**

```python
# Auth pattern (line 42-49)
self.api_key = api_key or os.getenv("BALLDONTLIE_API_KEY", "")
self.session.headers.update({"Authorization": self.api_key})

# Pagination pattern (lines 61-81)
def _paginate(self, path, params=None, max_pages=20):
    params.setdefault("per_page", 100)
    while page < max_pages:
        data = self._get(path, params)
        results.extend(data.get("data", []))
        next_cursor = data.get("meta", {}).get("next_cursor")
```

**Hypothesized MLB Endpoints (TO VERIFY):**
- `GET /mlb/v1/games?dates[]={date}` — should match NCAAB format
- `GET /mlb/v1/odds?game_id={id}` — should match NCAAB format
- `GET /mlb/v1/players?search={name}` — should match NCAAB format
- `GET /mlb/v1/injuries` — unknown if exists

### 2.2 Required Pydantic Contracts (Layer 0)

Based on NCAAB patterns, MLB contracts need:

```python
class BDLMLBGame(BaseModel):
    id: int
    date: str  # ISO format
    home_team: str
    away_team: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    status: str  # "scheduled", "in_progress", "final"
    
class BDLMLBOdds(BaseModel):
    game_id: int
    bookmakers: list[BDLBookmaker]
    
class BDLBookmaker(BaseModel):
    key: str  # "pinnacle", "draftkings", etc.
    markets: list[BDLMarket]
    
class BDLMarket(BaseModel):
    key: str  # "spreads", "totals", "h2h"
    outcomes: list[BDLOutcome]
```

---

## Task 3: `mlb_analysis.py` OddsAPI Calls

### 3.1 Direct OddsAPI Calls (TO BE REBUILT)

**Location:** Lines 364-388

```python
def _fetch_mlb_odds(self) -> dict[str, dict]:
    api_key = os.getenv("THE_ODDS_API_KEY")
    if not api_key:
        return {}  # SILENT RETURN — no log, no error
    
    resp = requests.get(
        "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds",
        params={...},
        timeout=10,
    )
    resp.raise_for_status()
    games = resp.json()  # NO VALIDATION — assumes list
    
    for g in (games if isinstance(games, list) else []):
        key = f"{g.get('away_team', '')}@{g.get('home_team', '')}"
        result[key] = g  # STORES ENTIRE PAYLOAD — no field selection
```

**Silent Failure Paths:**
1. Missing API key → returns `{}` (logged by caller)
2. Network timeout → exception caught, returns `{}`
3. Non-list response → `else []` branch silently drops all data
4. Missing `away_team`/`home_team` → key becomes `"@"` or `"@Home"`

### 3.2 Field Assumptions for OddsAPI

From `_calculate_edge()` (lines 394+):

| Field Path | Assumed Type | Risk |
|------------|--------------|------|
| `market["bookmakers"]` | list | ❌ NOT CHECKED before iteration |
| `bookmaker["markets"]` | list | ❌ NOT CHECKED |
| `mkt["key"]` | str | ✅ Compared to "spreads" |
| `mkt["outcomes"]` | list | ❌ NOT CHECKED |
| `outcome["name"]` | str | ✅ Compared to home_team |
| `outcome["price"]` | int | ⚠️ Defaults to -110 if missing |

**Silent Failure Pattern:**
```python
for bookmaker in market.get("bookmakers", []):  # Empty list = silent skip
    for mkt in bookmaker.get("markets", []):    # Empty list = silent skip
        if mkt.get("key") == "spreads":
            for outcome in mkt.get("outcomes", []):  # Empty list = silent skip
```
Three nested loops with `.get(..., [])` defaults = invisible no-ops if structure changes.

---

## Task 4: Silent Failure Audit in `daily_ingestion.py`

### 4.1 `_poll_mlb_odds()` — Lines 403-448

| Validation | Status | Issue |
|------------|--------|-------|
| `isinstance(games, list)` | ✅ YES | Line 429 — but only for counting |
| Field validation per game | ❌ NO | No schema validation |
| DB persistence | ❌ NO | "Does NOT persist to DB" — comment line 406 |
| Job run recorded | ✅ YES | `_record_job_run()` called |

**Silent Success Pattern:**
```python
n = len(games) if isinstance(games, list) else 0
if n == 0:
    logger.info("...only publishes lines 1-3 days out...expected not error")
    return {"status": "skipped", ...}
```
Zero games is treated as "expected" — but could also mean API key expired, endpoint changed, etc.

### 4.2 `_update_statcast()` — Lines 450+

```python
result = await asyncio.to_thread(run_daily_ingestion)
status = "success" if result.get("success") else "failed"
```

**Risk:** `run_daily_ingestion` returns dict with `"success": bool`. But if it throws exception, whole job fails without partial results.

### 4.3 Pattern: `_with_advisory_lock()` Wrappers

All ingestion jobs use:
```python
return await _with_advisory_lock(LOCK_IDS["mlb_odds"], _run)
```

**Risk:** If lock contention, job is skipped silently (no log entry).

---

## Recommended Field List for Pydantic Contracts

### Yahoo Fantasy Contracts

```python
class YahooPlayer(BaseModel):
    player_key: str
    player_id: str
    name: str
    team: str  # editorial_team_abbr
    positions: list[str]
    status: Optional[str] = None  # injury status
    injury_note: Optional[str] = None
    is_undroppable: bool = False
    percent_owned: float = 0.0  # percent_rostered mapped

class YahooPlayerWithStats(YahooPlayer):
    stats: dict[str, str]  # stat_id -> value (raw strings from Yahoo)

class YahooRosterEntry(BaseModel):
    player: YahooPlayer
    selected_position: str  # where they're slotted
    is_flex: bool = False
```

### BDL MLB Contracts (Hypothesized — VERIFY WITH CLAUDE'S CAPTURE)

```python
class BDLMLBGame(BaseModel):
    id: int
    date: str
    home_team_id: int
    away_team_id: int
    home_team: str  # full name
    away_team: str  # full name
    status: Literal["scheduled", "in_progress", "final", "postponed"]
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    
class BDLMLBOddsLine(BaseModel):
    game_id: int
    bookmaker_key: str
    market_key: Literal["spreads", "totals", "h2h"]
    home_spread: Optional[float] = None
    home_odds: Optional[int] = None  # american format
    total: Optional[float] = None
    over_odds: Optional[int] = None
```

---

## Delta Summary: Assumptions vs Reality

| Source | Code Assumes | Reality Check Needed |
|--------|--------------|----------------------|
| Yahoo | `player_key` always present | Verify in live roster/FA payloads |
| Yahoo | Ownership at 4+ possible paths | Claude's capture will show actual path |
| Yahoo | Stat IDs 28/42 both = K | Need to distinguish batting vs pitching |
| BDL | `/mlb/v1/` mirrors `/ncaab/v1/` | ENDPOINTS MAY NOT EXIST — verify first |
| BDL | Pagination uses `meta.next_cursor` | Verify in MLB response |
| OddsAPI | Response is list of games | Verify structure unchanged |
| OddsAPI | `away_team`/`home_team` present | Check if ID-based in MLB |

---

## Action Items for Claude (Priority 1)

1. **Capture Yahoo payloads** — focus on field paths for:
   - `player_key` location
   - `percent_rostered` vs `percent_owned` actual key names
   - `ownership` block nesting depth

2. **Capture BDL MLB** — verify these endpoints exist:
   - `GET /mlb/v1/games?dates[]={date}`
   - `GET /mlb/v1/odds?game_id={id}`
   - `GET /mlb/v1/players?search={name}`

3. **Document actual field names** — especially for odds (spread/total key names may differ from NCAAB)

4. **Test validation** — feed captured payloads through Pydantic contracts before declaring them final

---

*Audit complete. Awaiting Claude's payload capture for delta resolution.*
