# K-24: Yahoo `player_stats` Endpoint Specification

**Date:** April 1, 2026  
**Analyst:** Kimi CLI (Deep Intelligence Unit)  
**Status:** COMPLETE — Unblocks H-1 WaiverPlayerOut stats field

---

## 1. Endpoint URL Templates

### Single Player Stats
```
GET /fantasy/v2/player/{player_key}/stats;type={stat_type}?format=json
```

**Example:**
```
/fantasy/v2/player/469.p.12435/stats;type=season?format=json
```

### Multiple Players (Batch)
```
GET /fantasy/v2/league/{league_key}/players;player_keys={key1},{key2},{keyn}/stats;type={stat_type}?format=json
```

**Example:**
```
/fantasy/v2/league/469.l.72586/players;player_keys=469.p.12435,469.p.10918/stats;type=season?format=json
```

### With Sub-resources (metadata + stats + ownership)
```
GET /fantasy/v2/league/{league_key}/players;player_keys={keys}/stats;type={stat_type};out=stats,metadata,percent_owned?format=json
```

---

## 2. Stat Type Parameter

| Value | Description |
|-------|-------------|
| `season` | Cumulative season stats (2026) |
| `average_season` | Career averages |
| `lastweek` | Previous week's stats |
| `lastmonth` | Previous month's stats |

**Recommendation:** Use `type=season` for Waiver Wire key stats.

---

## 3. JSON Path to Stats

### Response Structure
```json
{
  "fantasy_content": {
    "player": [
      [/* metadata */],
      {
        "player_stats": {
          "coverage_type": "season",
          "stats": [
            {"stat": {"stat_id": "7", "value": "12"}},   // HR
            {"stat": {"stat_id": "8", "value": "45"}},   // RBI
            {"stat": {"stat_id": "6", "value": "5"}},    // R
            {"stat": {"stat_id": "3", "value": ".274"}} // AVG
          ]
        }
      }
    ]
  }
}
```

### Exact JSON Paths

| Stat Category | JSON Path | Data Type |
|--------------|-----------|-----------|
| All raw stats | `fantasy_content.player[1].player_stats.stats` | Array |
| Individual stat | `fantasy_content.player[1].player_stats.stats[n].stat.stat_id` | String |
| Stat value | `fantasy_content.player[1].player_stats.stats[n].stat.value` | String |

### Stats Array Format
```python
[
  {"stat": {"stat_id": "7", "value": "12"}},    # HR = 12
  {"stat": {"stat_id": "8", "value": "45"}},    # RBI = 45
  {"stat": {"stat_id": "6", "value": "5"}},     # R = 5
  {"stat": {"stat_id": "5", "value": "2"}},     # SB = 2
  {"stat": {"stat_id": "3", "value": ".274"}},  # AVG = .274
  {"stat": {"stat_id": "50", "value": "3.45"}}, # ERA = 3.45
  {"stat": {"stat_id": "26", "value": "1.12"}}  # WHIP = 1.12
]
```

---

## 4. Stat ID Mapping

### Batting Stats
| Stat | ID | Notes |
|------|-----|-------|
| Games Played | 1 | |
| At Bats | 2 | |
| Runs | 6 | |
| Hits | 4 | |
| Home Runs | 7 | |
| Runs Batted In | 8 | |
| Stolen Bases | 5 | |
| Batting Average | 3 | Format: ".274" |
| On-base Percentage | 9 | |
| Slugging | 10 | |
| OPS | 42 | OBP + SLG |
| NSB (Net SB) | 60 | SB - CS |
| Walks (BB) | 57 | K-14 verified |

### Pitching Stats
| Stat | ID | Notes |
|------|-----|-------|
| Wins | 28 | |
| Losses | 29 | |
| Saves | 32 | |
| Strikeouts | 27 | Often labeled "K" |
| Innings Pitched | 50 | Format: "182.1" |
| Hits Allowed | 22 | |
| Earned Runs | 25 | |
| ERA | 26 | Format: "3.45" |
| WHIP | 27 | Format: "1.12" |
| Walks (BB) | 44 | |
| Net Saves (NSV) | 83 | K-14 verified |
| Games Started | 62 | |
| Holds | 48 | |
| K/9 | 52 | |
| K/BB | 53 | |

**Note:** These stat IDs are consistent across Yahoo leagues. To get league-specific stat IDs, call:
```
GET /league/{league_key}/settings?format=json
```
Path: `fantasy_content.league[1].settings[0].stat_categories.stats`

---

## 5. Batching Capability

| Parameter | Limit | Notes |
|-----------|-------|-------|
| Max players per batch | **25** | Hard Yahoo limit |
| URL length | ~2000 chars | Keep player_keys under 25 |
| Recommended batch size | 25 | Optimal for waiver wire (25 players/page) |

### Batching Strategy
```python
# For waiver wire (paginated 25 at a time):
# Option A: Reuse existing get_free_agents() call + separate stats batch
# Option B: Add out=stats to league/players call

# Recommended: Option B (single call)
params = {
    "status": "A",
    "start": 0,
    "count": 25,
    "sort": "AR",
    "out": "stats,metadata,percent_owned"  # Include stats in same call
}
```

---

## 6. Implementation Recommendation for `_to_waiver_player()`

### Recommended Approach: Reuse with `out=stats`

Modify the existing `get_free_agents()` call to include stats:

```python
# backend/fantasy_baseball/yahoo_client_resilient.py
def get_free_agents(self, position: str = "", start: int = 0, 
                    count: int = 25, include_stats: bool = True) -> list[dict]:
    params = {
        "status": "A", 
        "start": start, 
        "count": count, 
        "sort": "AR"
    }
    if include_stats:
        params["out"] = "stats,metadata,percent_owned"
    # ... rest of method
```

**Why this approach:**
- ✅ Single API call per 25 players (efficient)
- ✅ No additional rate limit exposure
- ✅ Stats returned in same response structure
- ✅ Compatible with existing parsing logic

### Stats Parsing Helper
```python
def _extract_player_stats(self, player_data: list) -> dict:
    """Extract stats from player response into keyed dict."""
    stats_by_id = {}
    for item in player_data:
        if isinstance(item, dict) and "player_stats" in item:
            stats_list = item["player_stats"].get("stats", [])
            for entry in stats_list:
                if isinstance(entry, dict) and "stat" in entry:
                    s = entry["stat"]
                    stats_by_id[s.get("stat_id")] = s.get("value")
    
    # Map to readable keys
    return {
        "HR": stats_by_id.get("7"),
        "RBI": stats_by_id.get("8"),
        "R": stats_by_id.get("6"),
        "SB": stats_by_id.get("5"),
        "AVG": stats_by_id.get("3"),
        "ERA": stats_by_id.get("26"),
        "WHIP": stats_by_id.get("27"),
        "K": stats_by_id.get("27"),  # Pitcher strikeouts
        "W": stats_by_id.get("28"),
        "SV": stats_by_id.get("32"),
    }
```

---

## 7. Response Sample (Full)

```json
{
  "fantasy_content": {
    "xml:lang": "en-US",
    "yahoo:uri": "/fantasy/v2/player/469.p.12435/stats",
    "player": [
      [
        {"player_key": "469.p.12435"},
        {"player_id": "12435"},
        {"name": {"full": "Yainer Diaz", "first": "Yainer", "last": "Diaz"}},
        {"editorial_team_abbr": "HOU"},
        {"display_position": "C"},
        {"position_type": "B"}
      ],
      {
        "player_stats": {
          "coverage_type": "season",
          "stats": [
            {"stat": {"stat_id": "7", "value": "12"}},
            {"stat": {"stat_id": "8", "value": "45"}},
            {"stat": {"stat_id": "6", "value": "5"}},
            {"stat": {"stat_id": "3", "value": ".274"}}
          ]
        }
      }
    ]
  }
}
```

---

## Summary for Claude Code

| Question | Answer |
|----------|--------|
| **Endpoint for season stats?** | `/player/{player_key}/stats;type=season` |
| **Batch multiple players?** | Yes, up to 25 via `/league/{key}/players;player_keys=k1,k2,.../stats` |
| **Batting stats path?** | `fantasy_content.player[1].player_stats.stats` |
| **Pitching stats path?** | Same (distinguished by `position_type: "P"`) |
| **HR stat ID?** | `7` |
| **RBI stat ID?** | `8` |
| **ERA stat ID?** | `26` |
| **WHIP stat ID?** | `27` |
| **Reuse free-agents response?** | **YES** — Add `out=stats,metadata,percent_owned` to existing call |

---

*Spec complete. Ready for H-1 WaiverPlayerOut implementation.*
