# BallDontLie API Research Summary

**Date:** 2026-03-06  
**Researcher:** Kimi CLI  
**Purpose:** A-26 T2 Seed-Spread Kelly Scalars Implementation  
**API Version:** 1.0.0  
**OpenAPI Spec:** https://www.balldontlie.io/openapi.yml

---

## Executive Summary

The BallDontLie API provides comprehensive NCAAB (college basketball) data including a **March Madness bracket endpoint** that returns tournament seeds. This is exactly what's needed for the A-26 T2 implementation. The API uses simple header-based authentication and returns well-structured JSON with team seed information.

---

## Authentication

**Type:** API Key (Header-based)  
**Header Name:** `Authorization`  
**Header Format:** `Authorization: YOUR_API_KEY`

```python
headers = {
    "Authorization": "balldontlie_api_key_here"
}
```

**Required for:** All NCAAB endpoints (bracket endpoint requires GOAT tier subscription)

---

## Key Endpoint: NCAAB Tournament Bracket

### Endpoint
```
GET https://api.balldontlie.io/ncaab/v1/bracket
```

### Description
Retrieve NCAA Men's Basketball tournament bracket data with optional filters. **Requires GOAT tier subscription.**

**Important Note:** Season values are offset by -1 from database values. To get the 2026 tournament bracket, query with `season=2025`.

### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `season` | integer | No | Filter by season (e.g., 2025 queries database for 2026) |
| `round_id` | integer (1-7) | No | Filter by tournament round |
| `region_id` | integer | No | Filter by region ID |
| `cursor` | integer | No | Pagination cursor |
| `per_page` | integer | No | Results per page (default varies) |

### Round ID Mapping

| Round ID | Round Name |
|----------|------------|
| 1 | First Four / Play-in games |
| 2 | Round of 64 (First Round) |
| 3 | Round of 32 (Second Round) |
| 4 | Sweet 16 (Regional Semifinals) |
| 5 | Elite 8 (Regional Finals) |
| 6 | Final Four (National Semifinals) |
| 7 | Championship (National Championship) |

### Response Structure

```json
{
  "data": [
    {
      "game_id": 12345,
      "season": 2025,
      "round": 2,
      "region_id": 1,
      "region_label": "East",
      "bracket_location": 1,
      "date": "2026-03-20T12:15:00Z",
      "location": "Dayton, OH",
      "status": "pre",
      "status_detail": null,
      "broadcasts": ["CBS"],
      "home_team": {
        "id": 1,
        "conference_id": 1,
        "name": "Duke",
        "full_name": "Duke Blue Devils",
        "abbreviation": "DUKE",
        "seed": "1",
        "score": null,
        "winner": null
      },
      "away_team": {
        "id": 2,
        "conference_id": 8,
        "name": "Mount St. Mary's",
        "full_name": "Mount St. Mary's Mountaineers",
        "abbreviation": "MSM",
        "seed": "16",
        "score": null,
        "winner": null
      }
    }
  ],
  "meta": {
    "next_cursor": null,
    "prev_cursor": null,
    "per_page": 100
  }
}
```

### Data Types

**NCAABBracket Object:**
| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `game_id` | integer | Yes | Associated game ID |
| `season` | integer | No | Season year (offset by -1) |
| `round` | integer | No | Round ID (1-7) |
| `region_id` | integer | Yes | Region ID |
| `region_label` | string | Yes | Region name (East, West, South, Midwest) |
| `bracket_location` | integer | Yes | Position in bracket visualization |
| `date` | string (date-time) | Yes | Game date and time |
| `location` | string | Yes | Physical venue |
| `status` | string | Yes | Game status (pre, in, post) |
| `status_detail` | string | Yes | Detailed status info |
| `broadcasts` | array | Yes | TV/streaming channels |
| `home_team` | NCAABBracketTeam | Yes | Home team with seed |
| `away_team` | NCAABBracketTeam | Yes | Away team with seed |

**NCAABBracketTeam Object (KEY FOR A-26 T2):**
| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `id` | integer | Yes | Team ID (null if TBD) |
| `conference_id` | integer | Yes | Conference ID |
| `name` | string | No | Team name |
| `full_name` | string | Yes | Full display name |
| `abbreviation` | string | Yes | Team abbreviation |
| **`seed`** | **string** | **Yes** | **Tournament seed (e.g., "1", "16")** |
| `score` | integer | Yes | Game score |
| `winner` | boolean | Yes | Whether team won |

---

## Implementation Notes for A-26 T2

### Seed Data Extraction

The seed information is available as a **string** in the `seed` field of each team's `NCAABBracketTeam` object:

```python
# Example seed extraction
home_seed_str = bracket_game["home_team"]["seed"]  # "1"
away_seed_str = bracket_game["away_team"]["seed"]  # "16"

# Convert to integer for comparison
home_seed = int(home_seed_str) if home_seed_str else None
away_seed = int(away_seed_str) if away_seed_str else None
```

### Season Parameter for 2026 Tournament

To get the March 2026 tournament bracket:

```python
# 2026 tournament requires season=2025 (offset by -1)
params = {"season": 2025}
response = requests.get(
    "https://api.balldontlie.io/ncaab/v1/bracket",
    headers={"Authorization": API_KEY},
    params=params
)
```

### Rate Limits & Error Handling

**Error Codes:**
- `401` - Unauthorized (invalid or missing API key)
- `400` - Bad Request
- `429` - Rate Limit Exceeded
- `500` - Server Error

**Note:** The bracket endpoint requires GOAT tier subscription. Standard API keys will return 401 or 403.

---

## Additional Useful NCAAB Endpoints

### Get NCAAB Teams
```
GET /ncaab/v1/teams
```
Returns all NCAAB teams with IDs, names, and conference information.

### Get NCAAB Games
```
GET /ncaab/v1/games
```
Query parameters:
- `dates[]` - Filter by dates (YYYY-MM-DD)
- `team_ids[]` - Filter by team IDs
- `season` - Season year (offset by -1)
- `postseason` - Boolean for tournament games

### Get NCAAB Odds
```
GET /ncaab/v1/odds
```
Returns betting odds (spreads, moneylines, totals) for games.

### Get NCAAB Rankings
```
GET /ncaab/v1/rankings
```
Returns AP Poll and Coaches Poll rankings by week.

---

## Mapping to CBB Edge Analyzer

### Seed-Spread Scalar Implementation

```python
# Pseudocode for integration
class TournamentDataClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.balldontlie.io"
        self._seed_cache = {}  # team_name -> seed
        self._cache_ttl = 6 * 3600  # 6 hours
    
    def fetch_bracket(self, season: int = 2025) -> dict:
        """Fetch tournament bracket and extract seed mappings."""
        response = requests.get(
            f"{self.base_url}/ncaab/v1/bracket",
            headers={"Authorization": self.api_key},
            params={"season": season, "per_page": 100}
        )
        response.raise_for_status()
        return response.json()
    
    def build_seed_map(self, bracket_data: dict) -> dict:
        """Map team names to seeds from bracket data."""
        seed_map = {}
        for game in bracket_data.get("data", []):
            for team_key in ["home_team", "away_team"]:
                team = game.get(team_key)
                if team and team.get("seed"):
                    seed_map[team["name"]] = int(team["seed"])
        return seed_map
    
    def get_seed(self, team_name: str) -> Optional[int]:
        """Get tournament seed for a team (cached)."""
        if team_name not in self._seed_cache:
            bracket = self.fetch_bracket()
            self._seed_cache = self.build_seed_map(bracket)
        return self._seed_cache.get(team_name)
```

### Integration with Betting Model

```python
# In betting_model.py - analyze_game() method

# After integrity scalar, apply seed-spread scalar
def analyze_game(self, game_data: dict) -> dict:
    # ... existing analysis ...
    
    # Get seeds from tournament data client
    home_seed = self.tournament_client.get_seed(game_data["home_team"])
    away_seed = self.tournament_client.get_seed(game_data["away_team"])
    
    # Calculate seed-spread scalar
    seed_scalar = self._seed_spread_kelly_scalar(
        home_seed, away_seed, game_data.get("spread")
    )
    
    # Apply to Kelly stake
    final_stake = base_kelly_stake * snr_scalar * integrity_scalar * seed_scalar
    
    # ... rest of analysis ...
```

---

## API Tier Requirements

| Feature | Required Tier |
|---------|--------------|
| NCAAB Teams/Games | Any paid tier |
| NCAAB Odds | Premium+ |
| **NCAAB Bracket** | **GOAT** |
| Real-time updates | GOAT |

---

## Next Steps for Implementation

1. **Obtain API Key** - Sign up at balldontlie.io and upgrade to GOAT tier for bracket access
2. **Add to Railway Env Vars** - Store `BALLDONTLIE_API_KEY` in environment variables
3. **Test Endpoint** - Verify bracket data is available for 2025 season
4. **Implement Client** - Build `TournamentDataClient` class with caching
5. **Integrate Scalars** - Add seed-spread logic to `betting_model.py`
6. **Add Tests** - Verify seed extraction and scalar calculations

---

## References

- **OpenAPI Spec:** https://www.balldontlie.io/openapi.yml
- **Base URL:** https://api.balldontlie.io
- **Documentation:** https://www.balldontlie.io/
- **Pricing:** GOAT tier required for bracket endpoint
