# Schema Discovery Report
Date: 2026-04-05
Status: COMPLETE — live payloads captured, field tables populated
Captured by: `railway run python scripts/capture_api_payloads.py`
Fixtures: `tests/fixtures/bdl_mlb_*.json`

---

## BDL MLB `/mlb/v1/games` Response

### Pagination structure
```json
{
  "data": [...],
  "meta": { "per_page": 25 }
}
```
Cursor pagination available (not present on this response — no `next_cursor` means page is complete).
Observed: 19 games for 2026-04-05 (one full day, no cursor needed).

### Data item fields (verified across all 19 games)
| Field | Type | Nullable | Notes |
|-------|------|----------|-------|
| `id` | `int` | No | Game ID — use as FK in odds lookup |
| `home_team_name` | `str` | No | Denormalized display string |
| `away_team_name` | `str` | No | Denormalized display string |
| `home_team` | `MLBTeam` | No | Nested object (see MLBTeam below) |
| `away_team` | `MLBTeam` | No | Nested object |
| `season` | `int` | No | e.g. `2026` |
| `postseason` | `bool` | No | `false` for regular season |
| `season_type` | `str` | No | Observed: `"regular"`. Enum: `"regular"`, `"postseason"`, `"preseason"` |
| `date` | `str` | No | ISO 8601 UTC: `"2026-04-05T00:10:00.000Z"` |
| `home_team_data` | `MLBTeamGameData` | No | Hits/runs/errors/inning_scores — zeros for pre-game |
| `away_team_data` | `MLBTeamGameData` | No | Hits/runs/errors/inning_scores — zeros for pre-game |
| `venue` | `str` | No | Stadium name |
| `attendance` | `int` | No | 0 for pre-game/in-progress games |
| `conference_play` | `bool` | No | Always `false` for MLB |
| `status` | `str` | No | Observed: `"STATUS_FINAL"`. Others: `"STATUS_IN_PROGRESS"`, `"STATUS_SCHEDULED"` |
| `period` | `int` | No | Current/final inning number |
| `clock` | `int` | No | Always `0` for MLB |
| `display_clock` | `str` | No | Always `"0:00"` for MLB |
| `scoring_summary` | `list[MLBScoringPlay]` | No | Empty list `[]` for pre-game |

**CRITICAL: `attendance` appears non-null (int=0 for pre-game).** Mark as `Optional[int]` in Pydantic to handle edge cases — API may return null for future games.

### Nested: MLBTeam
| Field | Type | Nullable | Notes |
|-------|------|----------|-------|
| `id` | `int` | No | BDL team ID |
| `slug` | `str` | No | e.g. `"colorado-rockies"` |
| `abbreviation` | `str` | No | e.g. `"COL"` |
| `display_name` | `str` | No | Full name |
| `short_display_name` | `str` | No | e.g. `"Rockies"` |
| `name` | `str` | No | Nickname only e.g. `"Rockies"` |
| `location` | `str` | No | City/state e.g. `"Colorado"` |
| `league` | `str` | No | `"National"` or `"American"` |
| `division` | `str` | No | `"East"`, `"Central"`, `"West"` |

### Nested: MLBTeamGameData
| Field | Type | Nullable | Notes |
|-------|------|----------|-------|
| `hits` | `int` | No | 0 pre-game |
| `runs` | `int` | No | 0 pre-game |
| `errors` | `int` | No | 0 pre-game |
| `inning_scores` | `list[int]` | No | Empty list `[]` pre-game; 9+ items post-game |

### Nested: MLBScoringPlay
| Field | Type | Nullable | Notes |
|-------|------|----------|-------|
| `play` | `str` | No | Plain-text description |
| `inning` | `str` | No | `"top"` or `"bottom"` |
| `period` | `str` | No | e.g. `"1st"` |
| `away_score` | `int` | No | Cumulative score at time of play |
| `home_score` | `int` | No | Cumulative score at time of play |

---

## BDL MLB `/mlb/v1/odds` Response

### Pagination structure
```json
{
  "data": [...],
  "meta": { "per_page": 25 }
}
```
No `next_cursor` on this response (6 vendors for one game — fits in one page).

### Data item fields (verified — 6 vendors for game 5057892)
| Field | Type | Nullable | Notes |
|-------|------|----------|-------|
| `id` | `int` | No | Odds record ID |
| `game_id` | `int` | No | FK to game |
| `vendor` | `str` | No | Observed: `"fanduel"`, `"fanatics"`, `"betrivers"`, `"caesars"`, `"draftkings"`, `"betmgm"` |
| `spread_home_value` | `str` | No | **STRING not float.** e.g. `"1.5"`, `"-1.5"`, `"1"`. Parse with `float()`. |
| `spread_home_odds` | `int` | No | American odds integer. e.g. `-1000`, `520` |
| `spread_away_value` | `str` | No | **STRING not float.** Opposite sign of home. |
| `spread_away_odds` | `int` | No | American odds integer |
| `moneyline_home_odds` | `int` | No | American odds integer |
| `moneyline_away_odds` | `int` | No | American odds integer |
| `total_value` | `str` | No | **STRING not float.** e.g. `"3.5"`, `"4.5"`, `"5.5"` |
| `total_over_odds` | `int` | No | American odds integer |
| `total_under_odds` | `int` | No | American odds integer |
| `updated_at` | `str` | No | ISO 8601 UTC timestamp |

**CRITICAL CONTRACT REQUIREMENT:**
- `spread_home_value`, `spread_away_value`, `total_value` are **strings** — the Pydantic model must NOT type these as `float`. Use `str` and expose a `@property` or `field_validator` that casts to `float` when needed.
- Vendors vary per game and are not an exhaustive enum — type as `str`, not `Literal[...]`.
- Spread values differ across vendors for same game (betrivers shows `"1"` vs others showing `"1.5"`) — this is expected, not a data error.

---

## BDL MLB `/mlb/v1/player_injuries` Response

### Pagination structure
```json
{
  "data": [...],
  "meta": { "per_page": 25, "next_cursor": 409031 }
}
```
Cursor pagination confirmed active — must page through to get full injury list.

### Data item fields (verified across 25 items — first page)
| Field | Type | Nullable | Notes |
|-------|------|----------|-------|
| `player` | `MLBPlayer` | No | Nested player object |
| `date` | `str` | No | ISO 8601 UTC — injury report date |
| `return_date` | `str` | No | ISO 8601 UTC — estimated return |
| `type` | `str` | No | Body part e.g. `"Triceps"`, `"Hamstring"` |
| `detail` | `str` | **Yes** (4/25 null) | Injury sub-type e.g. `"Strain"` |
| `side` | `str` | **Yes** (2/25 null) | `"Right"`, `"Left"`, or null for bilateral/unknown |
| `status` | `str` | No | IL type: `"15-Day-IL"`, `"60-Day-IL"`, `"10-Day-IL"`, `"DTD"` |
| `long_comment` | `str` | No | Paragraph-length note |
| `short_comment` | `str` | No | One-sentence note |

### Nested: MLBPlayer (same shape in games, injuries, players endpoints)
| Field | Type | Nullable | Notes |
|-------|------|----------|-------|
| `id` | `int` | No | BDL player ID |
| `first_name` | `str` | No | |
| `last_name` | `str` | No | |
| `full_name` | `str` | No | Canonical display name |
| `debut_year` | `int` | **Yes** (3/25) | null for rookies with no debut yet |
| `jersey` | `str` | No | Number as string e.g. `"17"` |
| `college` | `str` | **Yes** (18/25) | Most players null — international/no college |
| `position` | `str` | No | Full position name e.g. `"Starting Pitcher"`, `"Designated Hitter"` |
| `active` | `bool` | No | `false` for IL players — **not** a reliable "rostered" signal |
| `birth_place` | `str` | No | City, state/country |
| `dob` | `str` | **Yes** (1/25) | Format: `"D/M/YYYY"` — **non-standard, not ISO 8601** |
| `age` | `int` | **Yes** (1/25) | Computed field — may be null when dob is null |
| `height` | `str` | No | e.g. `"6' 4\""` |
| `weight` | `str` | No | e.g. `"210 lbs"` |
| `draft` | `str` | **Yes** (16/25) | e.g. `"2014: Rd 5, Pk 158 (CLE)"`. null for undrafted/international |
| `bats_throws` | `str` | No | e.g. `"Right/Right"`, `"Left/Right"` |
| `team` | `MLBTeam` | No | Same nested shape as in games |

**CRITICAL CONTRACT NOTES:**
- `dob` uses `"D/M/YYYY"` format (day first!), not ISO 8601. Do NOT parse with naive `datetime.fromisoformat()`. Use a custom validator.
- `college` is null in the vast majority of cases — do NOT use it as a required field.
- `draft` is null in ~64% of cases. International players and undrafted players both return null.
- `active=false` on an IL player does NOT mean they left the team. Treat as unreliable for roster logic.

---

## BDL MLB `/mlb/v1/players` Response

### Pagination structure
Same as odds: `{ "data": [...], "meta": { "per_page": 25 } }`
No cursor on Ohtani search (1 result).

### Fields
Same `MLBPlayer` schema as in injuries response — confirmed consistent across endpoints.

---

## Summary: Pydantic Contract Requirements (Priority 2 inputs)

### Models to build in `backend/data_contracts/`

| Model | Source fields | Key gotchas |
|-------|--------------|-------------|
| `MLBTeam` | `id`, `slug`, `abbreviation`, `display_name`, `short_display_name`, `name`, `location`, `league`, `division` | All required, none null |
| `MLBTeamGameData` | `hits`, `runs`, `errors`, `inning_scores` | `inning_scores` is `list[int]` not nullable but can be `[]` |
| `MLBScoringPlay` | `play`, `inning`, `period`, `away_score`, `home_score` | |
| `MLBGame` | All fields in games table above | `scoring_summary` as `list[MLBScoringPlay]`, `home_team_data`/`away_team_data` as `MLBTeamGameData` |
| `MLBBettingOdd` | All fields in odds table above | Spread/total values are `str` — expose float via `@property` or `field_validator` |
| `MLBPlayer` | All fields in player table above | `dob` custom validator, 5 nullable fields |
| `MLBInjury` | All fields in injuries table above, with `player: MLBPlayer` | `detail`, `side` nullable |

### Pagination model
```python
class BDLMeta(BaseModel):
    per_page: int
    next_cursor: Optional[int] = None

class BDLResponse(BaseModel, Generic[T]):
    data: list[T]
    meta: BDLMeta
```

### Non-obvious decisions locked here
1. **Spread values stay as `str` in the model** — float conversion is downstream concern
2. **`attendance` should be `Optional[int]`** — observed as 0, but pre-game may return null in some cases
3. **`dob` requires custom parser** — `"D/M/YYYY"` format is not Python-parseable by default
4. **`scoring_summary` is `list[MLBScoringPlay]`** — empty list is valid, not null
5. **`return_date` in injuries observed non-null** — but mark `Optional[str]` for players with no expected return date

---

## Yahoo Schema
Date captured: 2026-04-05
Fixtures: `tests/fixtures/yahoo_roster.json`, `yahoo_free_agents.json`, `yahoo_adp_injury.json`
Captured via: `railway run venv/Scripts/python.exe scripts/yahoo_capture.py`

---

### `get_roster()` Response — 24 players

#### Fields (verified across all 24)
| Field | Type | Nullable | Notes |
|-------|------|----------|-------|
| `player_key` | `str` | No | e.g. `"469.p.12435"` — league_id.p.player_id format |
| `player_id` | `str` | No | Numeric string e.g. `"12435"` |
| `name` | `str` | No | Full display name |
| `team` | `str` | No | Team abbreviation e.g. `"HOU"` |
| `positions` | `list[str]` | No | Eligible positions e.g. `["C", "Util"]` |
| `status` | `bool` | **Yes** (22/24 null) | **`True` = player is injured.** NOT a string. `None` = healthy. |
| `injury_note` | `str` | **Yes** (19/24 null) | Body part string e.g. `"Calf"`. Present when injured. |
| `is_undroppable` | `bool` | No | `False` for most players |
| `percent_owned` | `float` | No | 0.0–100.0. **Already normalized** by client — K27 `percent_rostered` concern is resolved. |
| `selected_position` | `str` | No | Where currently slotted e.g. `"C"`, `"SP"`, `"BN"` |

**CRITICAL:** `status` is `bool` (`True` = injured), NOT a string like `"IL"`. The K27 audit assumed string. This is a breaking assumption in any code that checks `if player["status"] == "IL"`.

---

### `get_free_agents()` Response — 25 players per page

#### Fields (verified across 25)
| Field | Type | Nullable | Notes |
|-------|------|----------|-------|
| `player_key` | `str` | No | Same format as roster |
| `player_id` | `str` | No | |
| `name` | `str` | No | |
| `team` | `str` | No | |
| `positions` | `list[str]` | No | |
| `status` | `bool` | **Yes** (25/25 null in this page) | Same bool type as roster |
| `injury_note` | `str` | **Yes** (25/25 null in this page) | |
| `is_undroppable` | `bool` | No | |
| `percent_owned` | `float` | No | |
| `stats` | `dict[str, str]` | No | **Only on FA, not roster.** Stat ID strings → value strings |

**`stats` field:** dictionary of Yahoo stat ID → value string. Values are raw strings (not floats).
Stat ID 60 returns `"8/20"` format (H/AB combined) — NOT the same as stat ID 8 (raw H count).

**Stat IDs observed in live FA data:**
| ID | Likely meaning | Notes |
|----|----------------|-------|
| 3 | AVG | e.g. `".400"` |
| 7 | R | |
| 8 | H | Raw count |
| 12 | HR | |
| 13 | RBI | |
| 21 | IP | |
| 23 | W | |
| 26 | ERA | |
| 27 | WHIP | |
| 28 | K (pitching) | |
| 29 | QS | |
| 38 | K/BB | |
| 42 | K (batting?) | K27 flagged as duplicate of 28 — likely batting K, not pitching |
| 50 | IP (duplicate?) | K27 flagged as duplicate of 21 |
| 55 | OPS | e.g. `"1.380"` |
| 57 | BB | |
| 60 | H/AB | Returns `"8/20"` format — NOT same as stat ID 8 |
| 62 | GS | |
| 83 | NSV | |
| 85 | OBP | |

**Stat IDs 28 vs 42 and 21 vs 50:** K27 correctly flagged these as potential duplicates. IDs 42 and 50 are probably batting K and total IP respectively — need league settings API to confirm exact mappings.

---

### `get_adp_and_injury_feed()` Response — 100 players

This is the **embargo-critical method** for job 100_013.

#### Fields (verified across 100)
| Field | Type | Nullable | Notes |
|-------|------|----------|-------|
| `player_key` | `str` | No | |
| `player_id` | `str` | No | |
| `name` | `str` | No | |
| `team` | `str` | No | |
| `positions` | `list[str]` | No | Includes `"IL"` as a position for injured players |
| `status` | `bool` | **Yes** (98/100 null) | `True` = injured per Yahoo flag. Can be `None` even when `injury_note` is present. |
| `injury_note` | `str` | **Yes** (82/100 null) | Body part only e.g. `"Hip"`. NOT a full description. |
| `is_undroppable` | `bool` | No | |
| `percent_owned` | `float` | No | |

**CRITICAL:** `status=None` + `injury_note="Hip"` is valid — Verlander has no Yahoo status flag but does have an injury note. Both fields must be checked independently. Code that checks only `status is not None` to detect injuries will miss these players.

**`positions` includes `"IL"`** for players on the IL — this is a reliable injury signal in addition to `injury_note`.

---

### Delta: K27 Assumptions vs Reality

| K27 Assumption | Actual | Impact |
|----------------|--------|--------|
| `status` is a string e.g. `"IL"` | `status` is `bool` (`True`/`None`) | Any `status == "IL"` check silently misses all injuries |
| `percent_rostered` path depth varies (4+ locations) | Client normalizes to `percent_owned: float` — always present | Non-issue: client already handles the path complexity |
| `_parse_player()` hides response structure | `get_roster()` returns clean flat dicts — structure is known now | No hidden fields found in live data |
| Stat 28/42 both = pitching K | 42 likely = batting K (returned for hitters, 28 for pitchers) | Need to build separate fields in contracts |
| Stat 60 = H (duplicate of 8) | Stat 60 = `"H/AB"` combined format e.g. `"8/20"` | Not a duplicate — different data shape entirely |

---

### Yahoo Pydantic Contract Requirements (Priority 4)

```
backend/data_contracts/yahoo_player.py   -- YahooPlayer (shared base)
backend/data_contracts/yahoo_roster.py   -- YahooRosterEntry (adds selected_position)
backend/data_contracts/yahoo_waiver.py   -- YahooWaiverCandidate (adds stats dict)
```

Key contract decisions:
- `status: Optional[bool]` — NOT `Optional[str]`
- `injury_note: Optional[str]` — independent of status; must check both
- `percent_owned: float` — always present, no null guard needed
- `selected_position: str` — roster only, not present in FA or ADP feed
- `stats: Optional[dict[str, str]]` — FA and stats-enriched paths only; None on roster/ADP
- `positions: list[str]` — check for `"IL"` as reliable injury signal
