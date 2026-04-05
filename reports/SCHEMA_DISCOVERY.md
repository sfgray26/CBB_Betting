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
Status: PENDING — capture requires live OAuth session. Kimi K27 audit findings in HANDOFF.md cover the code-side assumptions. Live capture to be done in a separate Railway session once Yahoo token is confirmed active.
