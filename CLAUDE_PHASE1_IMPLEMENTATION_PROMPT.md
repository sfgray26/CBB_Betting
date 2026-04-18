# Phase 1 Implementation Prompt — V1→V2 Consumer Migration + 7 Data Gap Closures

> **Self-contained prompt.** No prior conversation context assumed.
> **Prerequisite:** Phase 0 is COMPLETE. 30/30 tests passing. The `backend/stat_contract/` package and 6 UI contracts in `backend/contracts.py` are delivered.

---

## 1. READ FIRST

Before writing any code, read these files in order:

1. `HANDOFF.md` — current operational state (Phase 0 COMPLETE, Phase 1 NEXT)
2. `backend/stat_contract/__init__.py` — the v2 contract singleton and derived constants
3. `backend/stat_contract/fantasy_stat_contract.json` — the v2 authoritative JSON
4. `backend/stat_contract/schema.py` — Pydantic v2 schema models
5. `backend/stat_contract/registry.py` — frozen dataclass stat definitions + `YAHOO_ID_TO_CANONICAL`
6. `backend/utils/fantasy_stat_contract.py` — the OLD v1 loader (being replaced)
7. `backend/utils/fantasy_stat_contract.json` — the OLD v1 JSON (being retired)
8. `backend/contracts.py` — decision contracts + 6 UI contracts from Phase 0
9. `backend/main.py` line 107 (import) and lines 6415–6510 (matchup endpoint usage)
10. `backend/routers/fantasy.py` lines 66, 85, 1213, 2204, 2245–2246 (import + all usages)
11. `backend/fantasy_baseball/category_tracker.py` lines 1–80 (import + usage)
12. `tests/test_fantasy_stat_contract.py` — old v1 tests (will be replaced)
13. `tests/test_stat_contract_schema.py` — Phase 0 schema tests (30/30 passing)
14. `tests/test_ui_contracts.py` — Phase 0 UI contract tests (30/30 passing)
15. `reports/2026-04-17-ui-specification-contract-audit.md` — the authoritative field-level mapping

---

## 2. MISSION

Phase 1 has **two workstreams** that must be completed in order:

**Workstream A — V1→V2 Consumer Migration (3 files)**
Repoint the 3 production files that import from `backend/utils/fantasy_stat_contract` to use `backend/stat_contract` instead. This is a **code rename, not a code rewrite** — the goal is import substitution with minimal behavioral change. The critical difference is the v2 canonical codes: `HR_B` (was `HR` batting), `HR_P` (was `HRA`), `K_B` (was `K(B)`), `K_P` (was `K` pitching), `K_9` (was `K/9`).

**Workstream B — 7 Data Gap Closures (pure functions + helpers)**
Implement 7 small, self-contained data extraction/computation helpers that close the L2 gaps identified in the UI specification audit. Each produces a value needed by the `ConstraintBudget`, `CanonicalPlayerRow`, or `MatchupScoreboardRow` contracts.

---

## 3. CANONICAL CODE MAPPING (V1 → V2)

This table governs every rename. The v2 contract uses disambiguated codes for stats that appear in both batting and pitching:

| V1 Code (old) | V2 Code (new) | Yahoo stat_id | Context |
|----------------|----------------|---------------|---------|
| `HR` (batting) | `HR_B` | 12 | Batting home runs |
| `HRA` | `HR_P` | 35 | Pitcher HR allowed |
| `K(B)` | `K_B` | 42 | Batter strikeouts (lower-is-better) |
| `K` (pitching) | `K_P` | 28 | Pitcher strikeouts |
| `K/9` or `K9` | `K_9` | 57 | Strikeouts per 9 innings |
| All others | **Unchanged** | — | R, H, RBI, TB, AVG, OPS, NSB, W, L, ERA, WHIP, QS, NSV |

The v2 `YAHOO_ID_INDEX` (from `backend/stat_contract/__init__.py`) already maps Yahoo stat_ids to canonical v2 codes. Use it as the authoritative lookup instead of the old `YAHOO_STAT_ID_FALLBACK`.

---

## 4. WORKSTREAM A: V1→V2 CONSUMER MIGRATION

### A1. `backend/main.py` — Matchup Endpoint

**Current import (line 107):**
```python
from backend.utils.fantasy_stat_contract import YAHOO_STAT_ID_FALLBACK, LEAGUE_SCORING_CATEGORIES
```

**Replace with:**
```python
from backend.stat_contract import YAHOO_ID_INDEX, SCORING_CATEGORY_CODES
```

**Changes required:**

1. **Line 6415** — Replace:
   ```python
   _YAHOO_STAT_FALLBACK: dict[str, str] = dict(YAHOO_STAT_ID_FALLBACK)
   ```
   With:
   ```python
   _YAHOO_STAT_FALLBACK: dict[str, str] = dict(YAHOO_ID_INDEX)
   ```

2. **Lines 6505–6507** — Replace:
   ```python
   if not active_stat_abbrs and LEAGUE_SCORING_CATEGORIES:
       active_stat_abbrs = set(LEAGUE_SCORING_CATEGORIES)
       logger.info("Using hardcoded LEAGUE_SCORING_CATEGORIES as fallback: %s", sorted(active_stat_abbrs))
   ```
   With:
   ```python
   if not active_stat_abbrs and SCORING_CATEGORY_CODES:
       active_stat_abbrs = set(SCORING_CATEGORY_CODES)
       logger.info("Using stat_contract SCORING_CATEGORY_CODES as fallback: %s", sorted(active_stat_abbrs))
   ```

3. **Lines ~6482–6492 — Update the disambiguation map.** The existing code uses `_PITCHER_RENAME = {"HR": "HRA", "K": "K(P)"}` and `_BATTER_RENAME = {"K": "K(B)", "HR": "HR"}`. These old codes must change to v2:
   ```python
   _PITCHER_RENAME = {"HR": "HR_P", "K": "K_P"}
   _BATTER_RENAME = {"K": "K_B", "HR": "HR_B"}
   ```
   **Important:** The disambiguation logic itself stays — Yahoo still sends `"HR"` and `"K"` as raw abbreviations for both batting and pitching. We just change what we rename them *to*.

4. **Line ~5364** — same pattern, update the copy of `_YAHOO_STAT_FALLBACK` in the waiver endpoint.

### A2. `backend/routers/fantasy.py` — Fantasy Router

**Current import (line 66):**
```python
from backend.utils.fantasy_stat_contract import YAHOO_STAT_ID_FALLBACK, LEAGUE_SCORING_CATEGORIES
```

**Replace with:**
```python
from backend.stat_contract import YAHOO_ID_INDEX, SCORING_CATEGORY_CODES
```

**Changes required:**

1. **Line 85** — Replace:
   ```python
   _YAHOO_STAT_FALLBACK: dict = dict(YAHOO_STAT_ID_FALLBACK)
   ```
   With:
   ```python
   _YAHOO_STAT_FALLBACK: dict = dict(YAHOO_ID_INDEX)
   ```

2. **Lines 2245–2246** — Same pattern as main.py:
   ```python
   if not active_stat_abbrs and SCORING_CATEGORY_CODES:
       active_stat_abbrs = set(SCORING_CATEGORY_CODES)
   ```

3. **All disambiguation maps in this file** — update `_PITCHER_RENAME` and `_BATTER_RENAME` to use v2 codes, same as in A1.

4. **Line 1213** — update the `sid_map` copy.

### A3. `backend/fantasy_baseball/category_tracker.py`

**Current import (line 13):**
```python
from backend.utils.fantasy_stat_contract import BATTING_CATEGORIES, CATEGORY_NEED_STAT_MAP
```

**This is the trickiest migration.** The old `BATTING_CATEGORIES` is `("r", "h", "hr", "rbi", "k", "tb", "avg", "ops", "nsb")` — **lowercase** strings used as dict keys in `_calculate_needs()`. The old `CATEGORY_NEED_STAT_MAP` maps Yahoo stat_ids to these lowercase keys: `{"3": "avg", "7": "r", "8": "h", "12": "hr", "13": "rbi", "42": "k", "55": "ops", "60": "nsb"}`.

**Problem:** The v2 `YAHOO_ID_INDEX` maps stat_ids to **uppercase canonical codes** (`"12" → "HR_B"`, `"42" → "K_B"`). The v2 `BATTING_CODES` is `frozenset({"R", "H", "HR_B", "RBI", "K_B", "TB", "AVG", "OPS", "NSB"})`.

**Solution:** Replace import with v2 constants and update the category tracker to use uppercase v2 canonical codes:

```python
from backend.stat_contract import BATTING_CODES, YAHOO_ID_INDEX
```

Then:
1. Replace `YAHOO_STAT_MAP = dict(CATEGORY_NEED_STAT_MAP)` with a filtered copy of `YAHOO_ID_INDEX` that only includes batting stat_ids:
   ```python
   # Filter YAHOO_ID_INDEX to only batting category stat_ids
   _BATTING_YAHOO_IDS = {sid: code for sid, code in YAHOO_ID_INDEX.items() if code in BATTING_CODES}
   YAHOO_STAT_MAP = dict(_BATTING_YAHOO_IDS)
   ```

2. Replace `for category in BATTING_CATEGORIES:` with `for category in sorted(BATTING_CODES):` in `_calculate_needs()`.

3. The `CategoryNeed` objects created in `_calculate_needs()` will now use v2 canonical codes (uppercase: `"R"`, `"HR_B"`, `"K_B"`) instead of lowercase (`"r"`, `"hr"`, `"k"`). **Verify no downstream consumer depends on lowercase keys.** Check all callers of `get_category_needs()` and `_calculate_needs()`:
   - `smart_lineup_selector.py` uses `CategoryNeed` — check how it references the `.category` field.
   - If any consumer string-matches against lowercase like `"hr"` or `"k"`, update it.

### A4. Retire old v1 artifacts

After all 3 consumers are migrated:

1. **Delete `backend/utils/fantasy_stat_contract.py`** — no remaining imports.
2. **Delete `backend/utils/fantasy_stat_contract.json`** — no remaining loader references it.
3. **Delete `frontend/lib/fantasy-stat-contract.json`** if it exists — the old v1 loader checked this path first.
4. **Update `tests/test_fantasy_stat_contract.py`** — rewrite or delete. The old tests assert v1 codes (`"K/9"`, `"NSB"` in CATEGORY_NEED_STAT_MAP). Either:
   - Delete the file entirely (Phase 0 tests in `test_stat_contract_schema.py` cover the v2 contract), or
   - Rewrite to test v2 equivalents.

### A5. Verify no other v1 references

Run:
```bash
grep -r "fantasy_stat_contract" backend/ tests/ --include="*.py" -l
```

The only results should be:
- `backend/stat_contract/fantasy_stat_contract.json` (the v2 JSON — this is fine)
- `backend/stat_contract/loader.py` (loads v2 JSON — this is fine)

If any other file still references `backend/utils/fantasy_stat_contract`, migrate it.

---

## 5. WORKSTREAM B: 7 DATA GAP CLOSURES

Each gap produces a pure function or thin helper. All go in a **new file**: `backend/services/constraint_helpers.py`. This keeps them isolated, testable, and importable by the Phase 4 API endpoints.

### File placement:
```
backend/
  services/
    constraint_helpers.py    ← NEW: all 7 gap closure functions
tests/
  test_constraint_helpers.py ← NEW: tests for all 7 functions
```

### B1. Acquisition Counter

**UI field:** GH-6 (acquisitions_used), GH-7 (acquisitions_remaining)
**Source:** Yahoo transactions API via `YahooFantasyClient.get_transactions()`
**Contract ref:** `ConstraintBudget.acquisitions_used`, `.acquisitions_remaining`, `.acquisition_limit`

```python
def count_weekly_acquisitions(
    transactions: list[dict],
    my_team_key: str,
    week_start: datetime,
    week_end: datetime,
) -> int:
    """Count add transactions by my team within the matchup week window.
    
    Args:
        transactions: Raw Yahoo transactions list from get_transactions(t_type="add")
        my_team_key: My Yahoo team key (e.g. "mlb.l.12345.t.3")
        week_start: Matchup week start date (inclusive)
        week_end: Matchup week end date (inclusive)
    
    Returns:
        Number of "add" type transactions by my team in the window.
    """
```

**Implementation notes:**
- Yahoo transaction entries have a `"type"` field (`"add"`, `"drop"`, `"trade"`) and a `"timestamp"` field (Unix epoch).
- Filter to `type == "add"` only.
- Filter to transactions where the destination team is `my_team_key`.
- Filter to transactions within `[week_start, week_end]`.
- Return the count.

### B2. IP Extractor

**UI field:** GH-12 (weekly IP accumulated)
**Source:** Yahoo scoreboard (stat_id 50 = IP)
**Contract ref:** `ConstraintBudget.ip_accumulated`

```python
def extract_ip_from_scoreboard(
    matchup_stats: dict[str, float],
) -> float:
    """Extract innings pitched value from a parsed matchup stats dict.
    
    The matchup stats dict is keyed by canonical codes (after disambiguation).
    IP is a display-only stat that Yahoo includes in scoreboard responses.
    
    Returns:
        Innings pitched as a float (e.g. 15.2 means 15 innings + 2 outs).
    """
```

**Implementation notes:**
- After A1 migration, the matchup stats dict will be keyed by v2 canonical codes.
- IP maps to `"IP"` in the v2 `YAHOO_ID_INDEX` (stat_ids 21 and 50 both map to `"IP"`).
- Simply return `matchup_stats.get("IP", 0.0)`.
- This is trivially simple but exists as a named function for contract traceability.

### B3. IP Pace Classifier

**UI field:** GH-14 (IP pace flag)
**Contract ref:** `ConstraintBudget.ip_pace` → `IPPaceFlag`

```python
from backend.contracts import IPPaceFlag

def classify_ip_pace(
    ip_accumulated: float,
    ip_minimum: float,
    days_elapsed: int,
    days_total: int,
) -> IPPaceFlag:
    """Classify weekly IP pace relative to league minimum.
    
    Args:
        ip_accumulated: IP so far this matchup week (e.g. 12.1)
        ip_minimum: League IP minimum (18.0)
        days_elapsed: Days completed in matchup week (1-7)
        days_total: Total days in matchup week (7)
    
    Returns:
        IPPaceFlag.BEHIND if projected to miss minimum
        IPPaceFlag.ON_TRACK if projected to hit ±10% of minimum
        IPPaceFlag.AHEAD if projected to exceed minimum comfortably
    """
```

**Implementation notes:**
- If `days_elapsed == 0`, return `BEHIND` (no data yet).
- Calculate `daily_rate = ip_accumulated / days_elapsed`.
- Calculate `projected_total = daily_rate * days_total`.
- BEHIND: `projected_total < ip_minimum * 0.9`
- AHEAD: `projected_total > ip_minimum * 1.1`
- ON_TRACK: everything else.

### B4. Per-Player Games Remaining

**UI field:** MS-11 (games remaining this week), PR-18 (ROW projection input)
**Source:** MLB schedule + roster
**Contract ref:** `MatchupScoreboardRow.games_remaining`

```python
def count_games_remaining(
    team_abbr: str,
    schedule: dict[str, list[datetime]],
    today: datetime,
    week_end: datetime,
) -> int:
    """Count remaining games for a team between today (exclusive) and week_end (inclusive).
    
    Args:
        team_abbr: MLB team abbreviation (e.g. "NYY", "LAD")
        schedule: Dict mapping team abbreviation to list of game datetimes
        today: Current date (games today are INCLUDED if not yet started)
        week_end: Matchup week end date (inclusive)
    
    Returns:
        Number of games remaining in the matchup week.
    """
```

**Implementation notes:**
- The schedule dict can be built from `lineup_validator.ScheduleFetcher._fetch_mlb_schedule()` or from `statsapi.schedule()`.
- Count games where `game_date > today` or `game_date == today and game not started`.
- For simplicity in Phase 1, count games where `game_date.date() >= today.date()` and `game_date.date() <= week_end.date()`.

### B5. Standings Record Extractor

**UI field:** GH-2 (my season record W-L-T)
**Source:** Yahoo standings API via `YahooFantasyClient.get_standings()`

```python
def extract_team_record(
    standings: list[dict],
    my_team_key: str,
) -> tuple[int, int, int]:
    """Extract W-L-T record from Yahoo standings response.
    
    Args:
        standings: Raw Yahoo standings list from get_standings()
        my_team_key: My Yahoo team key
    
    Returns:
        (wins, losses, ties) tuple
    """
```

**Implementation notes:**
- Yahoo standings response contains `outcome_totals` with `wins`, `losses`, `ties`.
- Existing code in `dashboard_service.py._fetch_team_record()` already does this — extract the logic into a pure function.
- Walk the standings list, find the entry where `team_key == my_team_key`, extract `outcome_totals`.
- Return `(int(wins), int(losses), int(ties))`.

### B6. Opposing SP Lookup

**UI field:** PR-5/PR-6 (opponent + home/away for probable pitchers), PR-11 (opposing SP for hitters)
**Source:** `probable_pitchers` table + MLB schedule

```python
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class OpposingSPInfo:
    """Opposing starting pitcher context for a hitter."""
    sp_name: Optional[str]
    sp_handedness: Optional[str]  # "L" or "R"
    opponent_team: str
    home_away: str  # "home" or "away"

def lookup_opposing_sp(
    player_team: str,
    schedule_entry: dict,
    probable_pitchers: list[dict],
) -> Optional[OpposingSPInfo]:
    """Look up the opposing starting pitcher for a hitter's game today.
    
    Args:
        player_team: Hitter's team abbreviation
        schedule_entry: Today's game entry with home_team, away_team
        probable_pitchers: List of probable pitcher records from DB
    
    Returns:
        OpposingSPInfo or None if no game today.
    """
```

**Implementation notes:**
- Determine if `player_team` is home or away from the schedule entry.
- The opposing team is the other team in the matchup.
- Look up the opposing team's probable pitcher from the `probable_pitchers` list.
- Existing logic in `smart_lineup_selector.py._fetch_probable_pitchers()` and `OpposingPitcher` dataclass provides the pattern.
- Return `OpposingSPInfo` with the SP's name, handedness, opponent team, and home/away.

### B7. Playing-Today Status Resolver

**UI field:** PR-4 (playing / not_playing / probable / IL / minors)
**Source:** Yahoo roster status + MLB schedule

```python
def resolve_playing_status(
    roster_entry: dict,
    team_schedule_today: bool,
) -> str:
    """Resolve a player's playing-today status.
    
    Args:
        roster_entry: Yahoo roster player dict with status fields
        team_schedule_today: Whether the player's team has a game today
    
    Returns:
        One of: "playing", "not_playing", "probable", "IL", "minors"
    """
```

**Implementation notes:**
- If roster entry has `status == "IL"` or `"IL10"` or `"IL60"` or `"IL15"` → return `"IL"`.
- If roster entry has `status == "NA"` or `"minors"` → return `"minors"`.
- If `not team_schedule_today` → return `"not_playing"`.
- If the player is a pitcher and in the `probable_pitchers` table for today → return `"probable"`.
- Otherwise → return `"playing"`.
- Existing logic in `lineup_validator._get_player_game_status()` provides the pattern.

---

## 6. TESTS

### `tests/test_constraint_helpers.py`

Write tests for all 7 functions:

1. **test_count_weekly_acquisitions_basic** — 3 adds in window, 1 outside → returns 3
2. **test_count_weekly_acquisitions_filters_other_teams** — adds by other teams excluded
3. **test_count_weekly_acquisitions_filters_drops** — drop transactions excluded
4. **test_extract_ip_from_scoreboard** — `{"IP": 15.2, "ERA": 3.50}` → returns 15.2
5. **test_extract_ip_missing** — empty dict → returns 0.0
6. **test_classify_ip_pace_behind** — 5.0 IP after 4 days → BEHIND
7. **test_classify_ip_pace_on_track** — 12.0 IP after 4 days → ON_TRACK
8. **test_classify_ip_pace_ahead** — 16.0 IP after 4 days → AHEAD
9. **test_classify_ip_pace_day_zero** — 0 days elapsed → BEHIND
10. **test_count_games_remaining** — team with 3 games left in week → returns 3
11. **test_count_games_remaining_no_games** — team not in schedule → returns 0
12. **test_extract_team_record** — standings with my team at 5-3-1 → returns (5, 3, 1)
13. **test_extract_team_record_not_found** — my team not in standings → returns (0, 0, 0)
14. **test_lookup_opposing_sp_found** — hitter vs known SP → returns OpposingSPInfo
15. **test_lookup_opposing_sp_no_game** — no game today → returns None
16. **test_resolve_playing_status_il** — IL status → "IL"
17. **test_resolve_playing_status_no_game** — no game today → "not_playing"
18. **test_resolve_playing_status_playing** — has game, not IL → "playing"

### Migration tests

19. **test_v2_yahoo_id_index_covers_v1_fallback** — every stat_id that was in the old `YAHOO_STAT_ID_FALLBACK` has a corresponding entry in `YAHOO_ID_INDEX` (may use different canonical code)
20. **test_v2_scoring_codes_cover_v1** — `SCORING_CATEGORY_CODES` has 18 entries, covering all v1 scoring categories under their new names
21. **test_disambiguation_maps_use_v2_codes** — verify the `_PITCHER_RENAME` and `_BATTER_RENAME` dicts in `main.py` use v2 codes (`HR_P`, `K_P`, `K_B`, `HR_B`)

Add these migration-verification tests to `tests/test_stat_contract_schema.py` (extending the existing file).

---

## 7. IMPLEMENTATION ORDER

Execute in this exact sequence:

1. **Create `backend/services/constraint_helpers.py`** with all 7 functions (Workstream B)
2. **Create `tests/test_constraint_helpers.py`** with all 18 tests
3. **Run tests:** `venv/Scripts/python -m pytest tests/test_constraint_helpers.py -q --tb=short` — all must pass
4. **Migrate `backend/fantasy_baseball/category_tracker.py`** (A3) — smallest consumer, lowest risk
5. **Migrate `backend/routers/fantasy.py`** (A2)
6. **Migrate `backend/main.py`** (A1)
7. **Run py_compile on all modified files:**
   ```bash
   venv/Scripts/python -m py_compile backend/services/constraint_helpers.py
   venv/Scripts/python -m py_compile backend/fantasy_baseball/category_tracker.py
   venv/Scripts/python -m py_compile backend/routers/fantasy.py
   venv/Scripts/python -m py_compile backend/main.py
   ```
8. **Run full test suite:** `venv/Scripts/python -m pytest tests/ -q --tb=short` — note any failures
9. **Add migration-verification tests** to `tests/test_stat_contract_schema.py` (tests 19-21)
10. **Delete old v1 artifacts:**
    - `backend/utils/fantasy_stat_contract.py`
    - `backend/utils/fantasy_stat_contract.json`
    - `frontend/lib/fantasy-stat-contract.json` (if it exists)
11. **Update or delete `tests/test_fantasy_stat_contract.py`** — old v1 tests
12. **Verify no remaining v1 references:**
    ```bash
    grep -r "from backend.utils.fantasy_stat_contract" backend/ tests/ --include="*.py"
    grep -r "utils/fantasy_stat_contract" backend/ tests/ --include="*.py"
    ```
    Both must return zero results (excluding `.pyc` files).
13. **Final full test run:** `venv/Scripts/python -m pytest tests/ -q --tb=short`

---

## 8. GATE 1 VERIFICATION

Gate 1 checklist — all must pass:

### Workstream A (migration)
- [ ] `backend/main.py` imports from `backend.stat_contract`, not `backend.utils.fantasy_stat_contract`
- [ ] `backend/routers/fantasy.py` imports from `backend.stat_contract`
- [ ] `backend/fantasy_baseball/category_tracker.py` imports from `backend.stat_contract`
- [ ] All disambiguation maps use v2 codes: `HR_B`, `HR_P`, `K_B`, `K_P` (not `HRA`, `K(B)`, `K(P)`)
- [ ] `backend/utils/fantasy_stat_contract.py` is deleted
- [ ] `backend/utils/fantasy_stat_contract.json` is deleted
- [ ] `grep -r "from backend.utils.fantasy_stat_contract" backend/ tests/` returns zero results
- [ ] `tests/test_fantasy_stat_contract.py` is deleted or rewritten for v2

### Workstream B (data gaps)
- [ ] `backend/services/constraint_helpers.py` exists with 7 functions
- [ ] `count_weekly_acquisitions()` counts add-only transactions within a week window
- [ ] `extract_ip_from_scoreboard()` returns float IP from stats dict
- [ ] `classify_ip_pace()` returns correct `IPPaceFlag` for all 3 cases
- [ ] `count_games_remaining()` counts games in week window
- [ ] `extract_team_record()` returns (W, L, T) tuple from standings
- [ ] `lookup_opposing_sp()` returns `OpposingSPInfo` or `None`
- [ ] `resolve_playing_status()` returns one of 5 valid status strings

### Tests
- [ ] `tests/test_constraint_helpers.py` has 18+ tests, all passing
- [ ] 3 migration-verification tests added to `tests/test_stat_contract_schema.py`, all passing
- [ ] All existing Phase 0 tests still pass (30/30)
- [ ] Full test suite has zero new failures from migration
- [ ] All `py_compile` checks pass on modified files

---

## 9. WHAT YOU MUST NOT DO

1. **Do not modify `backend/stat_contract/` package** — Phase 0 deliverables are frozen. Do not change schema.py, registry.py, builder.py, loader.py, __init__.py, or the JSON.
2. **Do not modify existing UI contracts in `backend/contracts.py`** — Phase 0 UI contracts are frozen. You may import from them (e.g. `IPPaceFlag`) but not change them.
3. **Do not create API endpoints** — Phase 1 produces helpers only. Endpoints come in Phase 4.
4. **Do not modify database models or create migrations** — data gap closures use existing tables and Yahoo API responses.
5. **Do not refactor the matchup endpoint logic** — only change imports and disambiguation maps. The endpoint's control flow stays the same.
6. **Do not introduce new dependencies** — all 7 functions use only stdlib + existing project imports.
7. **Do not change the v2 canonical codes** — `HR_B`, `HR_P`, `K_B`, `K_P`, `K_9` are locked.
8. **Do not update HANDOFF.md until Gate 1 passes.**

---

## 10. AFTER GATE 1 PASSES

Once all tests pass and the gate checklist is complete:

1. Update HANDOFF.md:
   - Change Phase 1 status from "NEXT" to "COMPLETE"
   - Change Phase 2 status from "Blocked by Phase 1" to "NEXT"
   - Remove the old "Active Workstream: Phase 0" section (replace with Phase 1 completion note)
   - Note: "V1 consumer migration complete. All 3 consumers now use backend.stat_contract. Old v1 loader and JSON deleted."
   - Update "Last Updated" timestamp

2. Do NOT start Phase 2 in this session.

**Phase 2 preview** (for context only — do not implement):
- Expand rolling stats from 9 to 18 scoring categories
- Expand projections from 8 to 18 scoring categories
- Build ROW (Rest-of-Week) projection pipeline
- Team-level ROW aggregation
- Opponent ROW projection
- Category classification with directionality
- Freshness timestamp propagation

---

*End of prompt. This prompt is self-contained. No prior conversation context is assumed.*
