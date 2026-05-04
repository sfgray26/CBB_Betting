# Yahoo API Response Shape Catalog

**Scope:** `backend/fantasy_baseball/yahoo_client_resilient.py`  
**Date:** 2026-04-18  
**Canonical Client:** `YahooFantasyClient`  
**Resilience Wrapper:** `ResilientYahooClient`

---

## 1. Global HTTP & Resilience Behavior

### Base URL
```
https://fantasysports.yahooapis.com/fantasy/v2
```

### `_get(path, params=None)` — Central Dispatcher
| Concern | Behavior |
|---------|----------|
| **URL build** | `f"{YAHOO_API_BASE}/{path.lstrip('/')}?format=json"` |
| **Comma quirk** | Param `out` is popped and appended **raw** (`?out=stats`) because Yahoo rejects `%2C` encoded commas. |
| **Auth header** | `Authorization: Bearer {access_token}` |
| **401 handling** | Refreshes token mid-flight and retries (`continue` on same attempt index). |
| **Rate limit (HTTP 999)** | Exponential backoff `wait = 2 ** attempt` (1s → 2s → 4s). Up to 3 total attempts. |
| **Other errors** | Non-2xx / non-999 raises `YahooAPIError` immediately. |
| **Circuit breaker** | `_CoreCircuitBreaker(failure_threshold=3, recovery_timeout=60, window_seconds=300)`. Open → raises 503 before network. |
| **Timeout** | 10 s per `requests.get`. |
| **Success path** | `self._cb.record_success()` then `resp.json()`. |

### Token Refresh
- `_ensure_token()` uses double-checked locking (`_token_lock`).
- `_store_tokens()` persists to `.env` best-effort; silent no-op on Railway (tokens live in-memory).

### `_flatten_league_section(raw)`
Normalizes Yahoo’s inconsistent `league[N]` shape:
- **(a)** merged dict → returned as-is
- **(b)** list of single-key dicts → merged into one flat dict
- Anything else → `{}`

---

## 2. Method Catalog

### `get_league_settings`

**URL Pattern**
```
GET /league/{game_id}.l.{league_id}/settings?format=json
```

**Response Parsing**
```python
data = self._get(f"league/{self.league_key}/settings")
return self._league_section(data, 0)   # _flatten_league_section(data["fantasy_content"]["league"][0])
```

**Representative Shape**
```json
{
  "fantasy_content": {
    "league": [
      [{"league_key": "469.l.72586"}, {"name": "Test League"}],
      {
        "settings": [{
          "stat_categories": {
            "stats": [
              {"stat": {"stat_id": "1", "display_name": "Runs", "position_type": "B"}},
              {"stat": {"stat_id": "60", "display_name": "H/AB", "position_type": "B"}}
            ]
          },
          "roster_positions": [...]
        }]
      }
    ]
  }
}
```

**Consumed vs Ignored (Downstream)**
- **Consumed:** `settings[0].stat_categories.stats[].stat.{stat_id, display_name, abbreviation, name, position_type, is_only_display_stat}` — used by `fantasy.py` and `main.py` to build `sid_map` and disambiguate batting/pitching stat collisions (e.g., HR, K).
- **Ignored:** `roster_positions`, `max_teams`, `scoring_type`, `weekly_deadline`, etc. (returned raw but not read by known backend callers).

**Bugs / Fragile Assumptions**
- Relies on `_flatten_league_section` to turn `league[0]` into a flat dict. If Yahoo returns a scalar or unexpected nesting, returns `{}` silently.

---

### `get_scoreboard`

**URL Pattern**
```
GET /league/{league_key}/scoreboard?format=json
GET /league/{league_key}/scoreboard?format=json&week={week}
```

**Response Parsing**
```python
sec = self._league_section(data, 1)
scoreboard = sec.get("scoreboard", {})

# Multi-version fallback:
# v2: scoreboard.matchups
# v1: scoreboard.0.matchups
# v3: deep recursive search (depth <= 3)
matchups_raw = scoreboard.get("matchups", {}) \
    or scoreboard.get("0", {}).get("matchups", {}) \
    or _deep_search(scoreboard, "matchups", depth=3)

# Then flatten indexed dict or list into a list of matchup dicts.
```

**Representative Shape**
```json
{
  "fantasy_content": {
    "league": [
      {...},
      {
        "scoreboard": {
          "matchups": {
            "count": "6",
            "0": {"matchup": {"week": "1", "week_start": "2026-03-26", "teams": {...}}},
            "1": {"matchup": {"week": "1", "teams": {...}}}
          }
        }
      }
    ]
  }
}
```

**Consumed vs Ignored (Downstream)**
- **Consumed:** Callers receive a `list[dict]` of raw matchup objects. Known consumers:
  - `dashboard_service.py`: iterates matchups, calls `_extract_opponent_key(matchup["teams"], my_team_key)`.
  - `category_tracker.py`: digs into `matchup["teams"][str(i)]["team"]` to find `team_key` and `stats`.
  - `fantasy.py` / `main.py`: same opponent-key extraction pattern.
- **Ignored by client:** `week_start`, `status`, `is_tied`, `winner_team_key` (passed through raw).

**Bugs / Fragile Assumptions**
- Deep recursive search `find_matchups` is capped at depth 3. If Yahoo nests matchups deeper, returns `[]` silently.
- `get_scoreboard` does **not** normalize the matchup internals; every downstream consumer re-implements its own traversal of `teams` blocks.

---

### `get_standings`

**URL Pattern**
```
GET /league/{league_key}/standings?format=json
```

**Response Parsing**
```python
sec = self._league_section(data, 1)
teams_raw = sec.get("standings", [{}])[0].get("teams", {})
count = int(teams_raw.get("count", 0))
for i in range(count):
    team_data = teams_raw[str(i)]["team"]
    teams.append(self._parse_team(team_data))
```

**Representative Shape**
```json
{
  "fantasy_content": {
    "league": [
      {...},
      {
        "standings": [{
          "teams": {
            "count": "12",
            "0": {"team": [[{"team_key": "469.l.72586.t.1"}, {"name": "Team A"}], {"roster": {...}}]},
            "1": {"team": [[{"team_key": "469.l.72586.t.2"}, {"name": "Team B"}], {"roster": {...}}]}
          }
        }]
      }
    ]
  }
}
```

**Consumed vs Ignored (Downstream)**
- **Consumed by client:** `_parse_team` extracts `team_key`, `team_id`, `name`, `manager`.
- **Downstream:** **No known callers in backend** (dead code as of 2026-04-15). Returned list is unused.

**Bugs / Fragile Assumptions**
- `sec.get("standings", [{}])[0]` assumes `standings` is a list. If `_flatten_league_section` produces a dict, `[0]` raises `TypeError`. In practice the current shape is a list, but this is a ticking time-bomb.
- `_parse_team` assumes `team_list[0]` is a list. If Yahoo flattens to a list of dicts, `meta` stays empty.

---

### `get_roster`

**URL Pattern**
```
GET /team/{team_key}/roster/players?format=json
```

**Response Parsing**
```python
data = self._get(f"team/{team_key}/roster/players")
team_data = self._team_section(data)          # flatten fantasy_content.team
roster = self._safe_get(team_data, "roster")
slot_0 = self._safe_get(roster, "0")
players_raw = self._safe_get(slot_0, "players")

count = int(players_raw.get("count", 0))
for i in range(count):
    entry = players_raw.get(str(i), {})
    entry = self._flatten_league_section(entry) if isinstance(entry, list) else entry
    player_data = entry.get("player", entry)
    p = self._parse_player(player_data)
    p["selected_position"] = self._extract_selected_position(player_data)
    # deduplicate by player_key
```

**Representative Shape**
```json
{
  "fantasy_content": {
    "team": [
      [{"team_key": "469.l.72586.t.1"}, {"name": "My Team"}],
      {"roster": {
        "0": {
          "players": {
            "count": "23",
            "0": {"player": [
              [{"player_key": "469.p.1234"}, {"player_id": "1234"}],
              {"name": {"full": "Juan Soto"}},
              {"selected_position": {"position": "OF"}}
            ]},
            "1": {"player": [...]}
          }
        }
      }}
    ]
  }
}
```

**Consumed vs Ignored (Downstream)**
- **Consumed:** `_parse_player` returns `player_key`, `player_id`, `name`, `team` (editorial_team_abbr), `positions`, `status`, `injury_note`, `is_undroppable`, `percent_owned`. `get_roster` adds `selected_position`.
- **Pydantic contract (`YahooRosterEntry`):** expects `selected_position: str` plus all `YahooPlayer` fields. `get_validated_roster()` in `yahoo_ingestion.py` validates every dict against this model.
- **Ignored:** `headshot`, `uniform_number`, `editorial_team_full_name`, `has_player_notes`, `starting_status`, `ownership` sub-block details, etc.

**Bugs / Fragile Assumptions**
- Deduplication logic (Bugfix March 28) uses `player_key` as primary key; fallback to `player_id` or `name` string can collide if keys are missing.
- `_extract_selected_position` only scans lists, not dicts. If Yahoo ever returns `selected_position` as a flat dict sibling, it is missed.
- Relies on `players_raw["count"]`; if Yahoo omits `count` (as it does in some other endpoints), loop body never executes. `_iter_block` (used elsewhere) is **not** used here.
- `_safe_get` auto-flattens lists, but `roster` → `0` → `players` chain assumes a three-deep numeric-key nesting that has changed before.

---

### `get_transactions`

**URL Pattern**
```
GET /league/{league_key}/transactions?format=json&type={t_type}
```
Default `t_type="add,drop,trade"`.

**Response Parsing**
```python
txns_raw = self._league_section(data, 1).get("transactions", {})
count = int(txns_raw.get("count", 0))
for i in range(count):
    txns.append(txns_raw[str(i)].get("transaction", {}))
```

**Representative Shape**
```json
{
  "fantasy_content": {
    "league": [
      {...},
      {
        "transactions": {
          "count": "3",
          "0": {"transaction": {"type": "add", "timestamp": "1234567890", "trader_team_key": "469.l.72586.t.1", "players": {...}}},
          "1": {"transaction": {"type": "drop", ...}}
        }
      }
    ]
  }
}
```

**Consumed vs Ignored (Downstream)**
- **Downstream:** **No known callers in backend** (dead code as of 2026-04-15). Returned raw transaction dicts.

**Bugs / Fragile Assumptions**
- If `transactions` is a list rather than an indexed dict, `count` becomes 0 and returns `[]` silently.

---

### `get_player_stats`

**URL Pattern**
```
GET /player/{player_key}/stats;type={stat_type}?format=json
```
`stat_type`: `season` | `average` | `projected_season`

**Response Parsing**
```python
player = data["fantasy_content"]["player"]
return self._parse_player_with_stats(player)
```

**Representative Shape**
```json
{
  "fantasy_content": {
    "player": [
      [[{"player_key": "469.p.1234"}, {"player_id": "1234"}], {"name": {"full": "Juan Soto"}}],
      {"player_stats": {
        "coverage_type": "season",
        "stats": [
          {"stat": {"stat_id": "1", "value": "42"}},
          {"stat": {"stat_id": "60", "value": "12/30"}}
        ]
      }}
    ]
  }
}
```

**Consumed vs Ignored (Downstream)**
- **Consumed:** `_parse_player_with_stats` produces all `_parse_player` fields plus `stats: {stat_id_str: value_str, ...}`.
- **Downstream:** Not directly called by known backend routers. Used ad-hoc for individual player enrichment.

**Bugs / Fragile Assumptions**
- `_parse_player_with_stats` calls `_parse_player(player[0] if isinstance(player[0], list) else player)`. If `player[0]` is a dict, the entire `player` list is passed to `_parse_player`, which deep-flattens it — slower but safe.
- Stats extraction scans only the top-level `player` list for a dict containing `"player_stats"`. If Yahoo nests it deeper, stats are silently empty.

---

### `get_free_agents`

**URL Pattern**
```
GET /league/{league_key}/players?format=json&status=A&start={start}&count={count}&sort=AR&position={position}
```

**Response Parsing**
```python
params = {"status": "A", "start": start, "count": count, "sort": "AR"}
if position: params["position"] = position
data = self._get(f"league/{self.league_key}/players", params=params)
players_raw = self._league_section(data, 1).get("players", {})
players = self._parse_players_block(players_raw)

# Best-effort batch stats enrichment (non-blocking)
player_keys = [p["player_key"] for p in players if p.get("player_key")]
if player_keys:
    stats_map = self.get_players_stats_batch(player_keys)
    for p in players:
        p["stats"] = stats_map.get(p.get("player_key"), {})
```

**Representative Shape**
Same as `get_waiver_players` (see below), with optional `stats` dict added post-hoc.

**Consumed vs Ignored (Downstream)**
- **Consumed:** Same `_parse_player` fields + `stats` dict.
- **Pydantic contract (`YahooWaiverCandidate`):** validates `stats: Optional[dict[str, str]]`.
- **Known callers:** `waiver_edge_detector.py`, `daily_ingestion.py`, `fantasy.py`, `main.py`.
- **Ignored:** `percent_change`, `percent_started`, `rank`, `rank_projected`, etc.

**Bugs / Fragile Assumptions**
- **K-24 HOTFIX:** `out=stats` is **invalid** on this endpoint (Yahoo returns 400). Stats are fetched separately via `get_players_stats_batch`. If the batch call fails, players are returned with empty `stats` — the waiver surface does not 503.
- `get_players_stats_batch` silently truncates to 25 keys; callers requesting >25 FAs receive stats for only the first 25.
- `ResilientYahooClient.get_waiver_players` fallback to metadata-only invokes `_make_request`, which is **undefined** — will `AttributeError` if triggered.

---

### `get_players_stats_batch`

**URL Pattern**
```
GET /league/{league_key}/players;player_keys={k1},{k2},.../stats;type={stat_type}?format=json
```

**Response Parsing**
```python
keys_str = ",".join(player_keys[:25])
data = self._get(f"league/{self.league_key}/players;player_keys={keys_str}/stats;type={stat_type}")
players_raw = self._league_section(data, 1).get("players", {})
for p in self._iter_block(players_raw, "player"):
    player_key = None
    stats_raw = {}
    if isinstance(p, list):
        for item in p:
            if isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict) and "player_key" in sub:
                        player_key = sub["player_key"]
            elif isinstance(item, dict):
                if "player_stats" in item:
                    for stat_entry in item["player_stats"].get("stats", []):
                        s = stat_entry.get("stat", {})
                        sid = s.get("stat_id")
                        if sid is not None:
                            stats_raw[str(sid)] = s.get("value", "")
    if player_key and stats_raw:
        result[player_key] = stats_raw
```

**Representative Shape**
```json
{
  "fantasy_content": {
    "league": [
      {...},
      {
        "players": {
          "count": "2",
          "0": {"player": [
            [{"player_key": "469.p.1234"}],
            {"player_stats": {"stats": [{"stat": {"stat_id": "1", "value": "42"}}]}}
          ]},
          "1": {"player": [...]}
        }
      }
    ]
  }
}
```

**Consumed vs Ignored (Downstream)**
- **Consumed:** Returns `{player_key: {stat_id_str: value_str}}`. Merged into `YahooWaiverCandidate.stats` by `get_free_agents`.
- **Ignored:** Player metadata returned by this endpoint is discarded; only `player_key` and `player_stats` are kept.

**Bugs / Fragile Assumptions**
- **Hard 25-player limit** enforced by slicing; no warning or pagination for larger inputs.
- Parsing is extremely fragile: requires `p` to be a list, then nested lists, then dicts. If Yahoo returns a flatter dict format, yields empty stats silently.
- Only records stats when **both** `player_key` and `stats_raw` are truthy. A player with no stats (e.g., rookie with no MLB innings) is omitted from the result map entirely.

---

### `get_waiver_players`

**URL Pattern**
```
GET /league/{league_key}/players?format=json&status=W&start={start}&count={count}
```

**Response Parsing**
```python
params = {"status": "W", "start": start, "count": count}
data = self._get(f"league/{self.league_key}/players", params=params)
players_raw = self._league_section(data, 1).get("players", {})
return self._parse_players_block(players_raw)
```

**Representative Shape**
```json
{
  "fantasy_content": {
    "league": [
      {...},
      {
        "players": {
          "count": "25",
          "0": {"player": [
            [{"player_key": "469.p.5678"}, {"player_id": "5678"}],
            {"name": {"full": "John Doe"}},
            {"ownership": {"percent_rostered": {"value": "15.2"}}}
          ]},
          "1": {"player": [...]}
        }
      }
    ]
  }
}
```

**Consumed vs Ignored (Downstream)**
- **Consumed:** Same `_parse_player` fields. No stats enrichment in this method.
- **Known callers:** `ResilientYahooClient.get_waiver_players` wraps the sync parent.

**Bugs / Fragile Assumptions**
- Same `_parse_players_block` / `_parse_player` fragilities as `get_free_agents`.
- `ResilientYahooClient` fallback path broken (see `get_free_agents`).

---

### `get_all_teams` (Note: `get_team` missing)

> **The requested `get_team(team_key)` method does not exist in `YahooFantasyClient`.** The closest equivalent is `get_all_teams()`, which returns metadata for every team.

**URL Pattern**
```
GET /league/{league_key}/teams?format=json
```

**Response Parsing**
```python
data = self._get(f"league/{self.league_key}/teams")
teams_raw = self._league_section(data, 1).get("teams", {})
return [self._parse_team(team_data) for team_data in self._iter_block(teams_raw, "team")]
```

**Representative Shape**
```json
{
  "fantasy_content": {
    "league": [
      {...},
      {
        "teams": {
          "count": "12",
          "0": {"team": [[{"team_key": "469.l.72586.t.1"}, {"name": "Team A"}], {"managers": [{"manager": {"nickname": "Alice"}}]}]},
          "1": {"team": [[{"team_key": "469.l.72586.t.2"}, {"name": "Team B"}], {"managers": [{"manager": {"nickname": "Bob"}}]}]}
        }
      }
    ]
  }
}
```

**Consumed vs Ignored (Downstream)**
- **Consumed by client:** `_parse_team` extracts `team_key`, `team_id`, `name`, `manager`.
- **Downstream callers:** `fantasy.py`, `main.py`, `dashboard_service.py` use `team_key` and `name` to build matchup/opponent mappings.
- **Ignored:** `team_logos`, `waiver_priority`, `faab_balance`, `number_of_moves`, `number_of_trades`, `roster_adds`, etc.

**Bugs / Fragile Assumptions**
- `_parse_team` assumes `team_list[0]` is a list. If Yahoo returns a flat list of dicts, `meta` stays empty and all fields become `None`.
- `_iter_block` handles missing `count`, but `_parse_team` does not handle the resulting shapes uniformly.

---

### `get_my_team_key`

**URL Pattern**
```
GET /league/{league_key}/teams?format=json
```
(Same as `get_all_teams`; re-fetched independently.)

**Response Parsing**
Deep recursive flatten (`flatten_team_data`, depth ≤ 5) across every `team` block to find `is_owned_by_current_login == 1` (or `True`).

**Representative Shape**
Same as `get_all_teams`, with `is_owned_by_current_login: 1` inside one team’s metadata.

**Consumed vs Ignored (Downstream)**
- **Consumed:** Returns only the `team_key` string.
- **Known callers:** Used pervasively — `dashboard_service.py`, `fantasy.py`, `main.py`, `category_tracker.py`, etc.

**Bugs / Fragile Assumptions**
- Recursion depth capped at 5. If Yahoo buries `is_owned_by_current_login` deeper, falls back to `YAHOO_TEAM_KEY` env var.
- Re-fetches the full `/teams` list every time instead of caching; adds ~1 extra API call to nearly every user-facing flow.

---

### `get_player`

**URL Pattern**
```
GET /player/{player_key}?format=json
```

**Response Parsing**
```python
data = self._get(f"player/{player_key}")
return self._parse_player(data["fantasy_content"]["player"][0])
```

**Consumed vs Ignored (Downstream)**
- **Consumed:** All `_parse_player` fields.
- **Downstream:** Not directly called by known backend routers. Exposed as a utility.

**Bugs / Fragile Assumptions**
- `data["fantasy_content"]["player"][0]` assumes `player` is a list. If Yahoo returns a dict, raises `TypeError` / `KeyError`.

---

### `get_league_rosters`

**URL Pattern**
```
GET /league/{league_key}/teams/roster?format=json
```

**Response Parsing**
Most complex parser in the file. Handles:
- `teams` as list vs indexed dict
- `team_data[0]` as list vs dict (team meta extraction)
- `roster_wrapper` search by key `"roster"`
- **Format A:** `roster_wrapper["players"]` (old)
- **Format B:** `roster_wrapper["0"]["players"]` (2026)
- `players_raw` as list vs indexed dict
- `_iter_block` for players

**Representative Shape (Format B — 2026)**
```json
{
  "fantasy_content": {
    "league": [
      {...},
      {
        "teams": {
          "0": {"team": [
            [{"team_key": "469.l.72586.t.1"}, {"name": "My Team"}],
            {"roster": {
              "0": {
                "players": {
                  "count": "23",
                  "0": {"player": [...]},
                  "1": {"player": [...]}
                }
              }
            }}
          ]}
        }
      }
    ]
  }
}
```

**Consumed vs Ignored (Downstream)**
- **Consumed:** `_parse_player` fields + injected `team_key`.
- **Known callers:** Not directly referenced in backend routers; used for league-wide roster audits.

**Bugs / Fragile Assumptions**
- This is the most fragile method. Three nested format branches with manual list/dict sniffing.
- Exception handler returns `[]` silently, swallowing the stack trace from production debugging.

---

### `get_lineup`

**URL Pattern**
```
GET /team/{team_key}/roster/players?format=json&date=YYYY-MM-DD
```

**Response Parsing**
```python
data = self._get(f"team/{team_key}/roster/players", params={"date": date})
players_raw = (
    self._team_section(data)
    .get("roster", {})
    .get("0", {})
    .get("players", {})
)
count = int(players_raw.get("count", 0))
for i in range(count):
    player_data = players_raw[str(i)]["player"]
    p = self._parse_player(player_data)
    # manual selected_position extraction (duplicated logic)
    ...
    players.append(p)
```

**Consumed vs Ignored (Downstream)**
- **Consumed:** Same as `get_roster` plus `selected_position`.
- **Known callers:** `set_lineup_resilient` in `ResilientYahooClient` uses it for pre-submission validation.

**Bugs / Fragile Assumptions**
- Duplicates `_extract_selected_position` logic inline with slightly different nesting traversal.
- Same `count` reliance and deduplication concerns as `get_roster`.

---

### `get_draft_results`

**URL Pattern**
```
GET /league/{league_key}/draftresults?format=json
```

**Response Parsing**
```python
picks_raw = (
    self._league_section(data, 1)
    .get("draft_results", {})
    .get("0", {})
    .get("draft_results", {})
)
count = int(picks_raw.get("count", 0))
for i in range(count):
    pick = picks_raw[str(i)]["draft_result"][0]
    picks.append({
        "pick": pick.get("pick"),
        "round": pick.get("round"),
        "team_key": pick.get("team_key"),
        "player_key": pick.get("player_key"),
    })
```

**Consumed vs Ignored (Downstream)**
- **Consumed:** `pick`, `round`, `team_key`, `player_key`.
- **Known callers:** `fantasy.py`, `main.py`, `draft_tracker.py`.
- **Ignored:** `player_name` (if present), `cost`, `destination_team_key`, etc.

**Bugs / Fragile Assumptions**
- Fixed deep path `draft_results.0.draft_results` is extremely brittle. If Yahoo removes the `0` index wrapper, returns empty list silently.

---

### `search_players`

**URL Pattern**
```
GET /league/{league_key}/players?format=json&search={name}&status={status}
```

**Response Parsing**
Identical to `get_free_agents` / `get_waiver_players` via `_parse_players_block`.

**Consumed vs Ignored (Downstream)**
- **Consumed:** Same `_parse_player` fields.
- **Downstream:** Not directly called by known backend routers.

---

### `get_adp_and_injury_feed`

**URL Pattern**
```
GET /league/{league_key}/players?format=json&start={start}&count={count}&sort=DA
```

**Response Parsing**
Paginated loop (default 4 pages × 25 = 100 players). Calls `_parse_players_block` per page.

**Consumed vs Ignored (Downstream)**
- **Consumed:** `_parse_player` fields (name, injury_note, status, percent_owned, positions).
- **Known callers:** `get_validated_adp_feed()` in `yahoo_ingestion.py` validates against `YahooPlayer`.
- **Ignored:** `stats` (not requested), `rank`, `rank_projected`.

**Bugs / Fragile Assumptions**
- If any page fails with `YahooAPIError`, loop breaks and returns partial results. No retry per page.
- `_parse_player` name-stripping regex (injury suffix removal) can over-strip legitimate names if they collide with body-part tokens.

---

### `get_faab_balance`

**URL Pattern**
```
GET /league/{league_key}/teams?format=json
```

**Response Parsing**
Shallow flatten of team blocks to find `is_owned_by_current_login`, then reads `faab_balance`.

**Consumed vs Ignored (Downstream)**
- **Consumed:** Returns `float(faab_balance)` or `None`.
- **Downstream:** Not directly called by known backend routers.

**Bugs / Fragile Assumptions**
- Flattening is shallow compared to `get_my_team_key`. If `faab_balance` is nested deeper than one level, missed.
- Returns `None` on any exception, swallowing the error.

---

## 3. Summary Table

| Method | URL Path | Downstream Consumers | Known Bugs |
|--------|----------|---------------------|------------|
| `get_league_settings` | `/league/{lk}/settings` | `fantasy.py`, `main.py` (sid_map builder) | `_flatten_league_section` can return `{}` silently |
| `get_scoreboard` | `/league/{lk}/scoreboard` | `dashboard_service.py`, `category_tracker.py`, `fantasy.py`, `main.py` | Multi-version fallback; depth-capped recursive search |
| `get_standings` | `/league/{lk}/standings` | **None** (dead code) | `standings[0]` TypeError risk if dict shape changes |
| `get_roster` | `/team/{tk}/roster/players` | `yahoo_ingestion.py`, `dashboard_service.py`, `fantasy.py`, `main.py`, `daily_lineup_optimizer.py` | `count` reliance; dedup fallback collision risk |
| `get_transactions` | `/league/{lk}/transactions` | **None** (dead code) | Assumes indexed dict; list format → empty |
| `get_player_stats` | `/player/{pk}/stats;type={t}` | Ad-hoc enrichment | `player[0]` list assumption; stats sibling scan only |
| `get_free_agents` | `/league/{lk}/players` | `waiver_edge_detector.py`, `daily_ingestion.py`, `fantasy.py`, `main.py` | `out=stats` invalid; batch stats truncate at 25 |
| `get_players_stats_batch` | `/league/{lk}/players;player_keys=.../stats` | Called by `get_free_agents` | Hard 25-key limit; extremely fragile nested list parsing |
| `get_waiver_players` | `/league/{lk}/players?status=W` | `ResilientYahooClient` wrapper | Same as `get_free_agents`; resilient fallback broken |
| `get_all_teams` | `/league/{lk}/teams` | `fantasy.py`, `main.py`, `dashboard_service.py` | `_parse_team` list assumption |
| `get_my_team_key` | `/league/{lk}/teams` | Pervasive (every flow) | Depth-5 recursion cap; extra API call per invocation |
| `get_player` | `/player/{pk}` | Ad-hoc | `player[0]` list assumption |
| `get_league_rosters` | `/league/{lk}/teams/roster` | Ad-hoc / audit scripts | Most fragile parser; silent `[]` on exception |
| `get_lineup` | `/team/{tk}/roster/players?date=` | `set_lineup_resilient` | Duplicated `selected_position` logic |
| `get_draft_results` | `/league/{lk}/draftresults` | `fantasy.py`, `main.py`, `draft_tracker.py` | Fixed deep path `draft_results.0.draft_results` |
| `search_players` | `/league/{lk}/players?search=` | **None** (utility) | Same `_parse_players_block` fragilities |
| `get_adp_and_injury_feed` | `/league/{lk}/players?sort=DA` | `yahoo_ingestion.py` (ADP job) | Page failure → partial results; name-regex over-stripping |
| `get_faab_balance` | `/league/{lk}/teams` | **None** | Shallow flatten; swallows all exceptions |

---

## 4. Missing Methods

- **`get_team(team_key)`** — Not implemented. Use `get_all_teams()` and filter by `team_key`, or call `get_roster(team_key)` for roster details.

---

## 5. Resilient Wrapper Notes (`ResilientYahooClient`)

- **Circuit breaker:** 3 failures in 300 s → open for 300 s (5 min recovery).
- **Stale cache:** TTL default 24 h; disabled via `YAHOO_CACHE_DISABLED=true`.
- **Broken fallback:** `_fetch_waiver_metadata_only` → `_fetch_with_subresources` → `_make_request` is **undefined**. If Yahoo rejects `percent_owned` subresource, the resilient path will raise `AttributeError` rather than degrade gracefully.
- **Rate limit budget:** With the base `_get` retry logic, a single method call can issue up to 3 requests. The circuit breaker guards against cascading retries but does not throttle overall volume.
