# BDL API Capabilities Analysis for MLB Fantasy Baseball

> **Research Date:** April 11, 2026  
> **Researcher:** Kimi CLI (Deep Intelligence)  
> **Task ID:** K-34  
> **Status:** COMPLETE

---

## 1. Executive Summary

This report provides a comprehensive analysis of the **BallDontLie (BDL) MLB API** capabilities, with specific focus on features relevant to the Fantasy Baseball platform. The platform currently uses BDL as the primary data source for MLB games, odds, injuries, players, and statistics.

### Key Findings

| Aspect | Finding |
|--------|---------|
| **Base URL** | `https://api.balldontlie.io/mlb/v1` |
| **Data Coverage** | 2002 to current (real-time) |
| **Rate Limit (GOAT tier)** | 600 requests/minute |
| **Cost** | $39.99/month per sport |
| **Authentication** | API key in `Authorization` header (no Bearer prefix) |
| **Pagination** | Cursor-based, max 100 per page |
| **Total Endpoints** | 18 MLB-specific endpoints identified |

### Recommendations

1. **Use `/stats` endpoint with date filters** for daily player box score ingestion
2. **Implement proper rate limiting** at 10 req/sec (600/min) for GOAT tier
3. **Use `per_page=100`** for bulk data fetching to minimize API calls
4. **Consider `/season_stats`** for derived stats calculation (pre-aggregated season data)
5. **Monitor 429 responses** and implement exponential backoff

---

## 2. Authentication & Configuration

### API Key Setup

```bash
# Environment variable
export BALLDONTLIE_API_KEY="your_api_key_here"

# Header format (no "Bearer" prefix)
Authorization: YOUR_API_KEY
```

### Client Libraries

**Official Python SDK:**
```bash
pip install balldontlie
```

```python
from balldontlie import BalldontlieAPI

api = BalldontlieAPI(api_key="YOUR_API_KEY")
stats = api.mlb.stats.list(dates=["2026-04-09"])
```

**Manual requests (current implementation approach):**
```python
import requests

session = requests.Session()
session.headers.update({"Authorization": os.getenv("BALLDONTLIE_API_KEY")})

response = session.get(
    "https://api.balldontlie.io/mlb/v1/stats",
    params={"dates[]": "2026-04-09", "per_page": 100}
)
```

---

## 3. Rate Limiting Analysis

### Tier Comparison

| Tier | Requests/Min | Price | MLB Access |
|------|--------------|-------|------------|
| Free | 5 | $0 | Teams, Players, Games |
| All-Star | 60 | $9.99/mo | + Injuries, Active Players, Standings, Player Stats |
| GOAT | 600 | $39.99/mo | Full access including season stats, splits, odds, props |
| All-Access | 600 | $499.99/mo | All sports, all endpoints |

### Practical Limits

At **600 req/min = 10 req/sec**:
- Daily games fetch: ~1-2 requests
- Player stats for a full slate: ~10-20 requests
- Injuries list: ~5-10 requests
- Season stats: ~3-5 requests

**Conclusion:** GOAT tier provides sufficient headroom for aggressive data ingestion strategies.

### Rate Limit Handling

```python
import time
from functools import wraps

def rate_limited(max_per_second=10):
    min_interval = 1.0 / max_per_second
    last_call = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_call[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            last_call[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Or simpler: 0.1s sleep between calls
@rate_limited(10)
def fetch_stats():
    pass
```

### Error Response (429)

```json
{
  "error": "Rate limit exceeded",
  "retry_after": 60
}
```

**Best Practice:** Implement exponential backoff on 429 errors:
- Retry 1: wait 1 second
- Retry 2: wait 2 seconds
- Retry 3: wait 4 seconds
- Retry 4+: wait 60 seconds

---

## 4. Endpoint Matrix

### Complete MLB Endpoint Catalog

| Endpoint | Method | Auth | Tier | Pagination | Primary Use Case |
|----------|--------|------|------|------------|------------------|
| `/teams` | GET | Required | Free | Yes | Team lookup, division/league info |
| `/teams/{id}` | GET | Required | Free | No | Single team details |
| `/players` | GET | Required | Free | Yes | Player search, roster building |
| `/players/{id}` | GET | Required | Free | No | Single player details |
| `/players/active` | GET | Required | All-Star | Yes | Active roster filtering |
| `/player_injuries` | GET | Required | All-Star | Yes | IL + DTD status |
| `/games` | GET | Required | Free | Yes | Schedule, scores, line scores |
| `/games/{id}` | GET | Required | Free | No | Single game details |
| `/stats` | GET | Required | All-Star | Yes | **Per-game box scores** |
| `/season_stats` | GET | Required | GOAT | Yes | **Pre-aggregated season stats** |
| `/standings` | GET | Required | All-Star | Yes | Division/league standings |
| `/teams/season_stats` | GET | Required | GOAT | Yes | Team aggregate stats |
| `/players/splits` | GET | Required | GOAT | Yes | Situational stats (home/away, etc.) |
| `/players/vs_player` | GET | Required | GOAT | Yes | Batter vs pitcher matchups |
| `/plays` | GET | Required | GOAT | Yes | Play-by-play data |
| `/plate_appearances` | GET | Required | GOAT | Yes | Individual PA details |
| `/odds` | GET | Required | GOAT | Yes | Sportsbook lines |
| `/player_props` | GET | Required | GOAT | Yes | Prop bet odds |

### Currently Used Endpoints (Platform)

Based on codebase analysis of `backend/services/balldontlie.py`:

| Endpoint | Used In | Status |
|----------|---------|--------|
| `/games` | `get_mlb_games()` | ✅ Active |
| `/odds` | `get_mlb_odds()` | ✅ Active |
| `/player_injuries` | `get_mlb_injuries()` | ✅ Active |
| `/players` | `get_all_mlb_players()`, `search_mlb_players()` | ✅ Active |
| `/players/{id}` | `get_mlb_player()` | ✅ Active |
| `/stats` | `get_mlb_stats()` | ✅ Active |

### Recommended Additional Endpoints

| Endpoint | Use Case | Priority |
|----------|----------|----------|
| `/season_stats` | Derived stats (YTD totals, averages) | HIGH |
| `/players/splits` | Platoon advantage calculations | MEDIUM |
| `/standings` | Playoff race context | LOW |

---

## 5. Data Field Reference

### Player Object Structure

```json
{
  "id": 208,
  "first_name": "Shohei",
  "last_name": "Ohtani",
  "full_name": "Shohei Ohtani",
  "debut_year": 2018,
  "jersey": "17",
  "college": null,
  "position": "Designated Hitter",
  "active": true,
  "birth_place": "Oshu, Japan",
  "dob": "5/7/1994",
  "age": 30,
  "height": "6' 4\"",
  "weight": "210 lbs",
  "draft": null,
  "bats_throws": "Left/Right",
  "team": {
    "id": 14,
    "slug": "los-angeles-dodgers",
    "abbreviation": "LAD",
    "display_name": "Los Angeles Dodgers",
    "short_display_name": "Dodgers",
    "name": "Dodgers",
    "location": "Los Angeles",
    "league": "National",
    "division": "West"
  }
}
```

### Stats Object Structure (per-game)

**Key insight from API docs:**
> "Batting fields are populated for position players, while pitching fields (`ip`, `p_hits`, `era`, etc.) are populated for pitchers. Fielding stats are included for all players. Detail fields (`doubles`, `triples`, `stolen_bases`, `batters_faced`, `putouts`, etc.) may be `null` if detailed box score data is not yet available for the game."

#### Batting Fields

| Field | Type | Nullable | Notes |
|-------|------|----------|-------|
| `at_bats` | int | Yes | AB |
| `runs` | int | Yes | R |
| `hits` | int | Yes | H |
| `rbi` | int | Yes | RBI |
| `hr` | int | Yes | HR |
| `bb` | int | Yes | Walks |
| `k` | int | Yes | Strikeouts |
| `avg` | float | Yes | Season AVG to date |
| `obp` | float | Yes | Season OBP to date |
| `slg` | float | Yes | Season SLG to date |
| `doubles` | int | Yes | 2B |
| `triples` | int | Yes | 3B |
| `stolen_bases` | int | Yes | SB |
| `caught_stealing` | int | Yes | CS |
| `plate_appearances` | int | Yes | PA |
| `left_on_base` | int | Yes | LOB |
| `ground_outs` | int | Yes | GO |
| `air_outs` | int | Yes | AO |
| `gidp` | int | Yes | Double plays |
| `sac_bunts` | int | Yes | SH |
| `sac_flies` | int | Yes | SF |

#### Pitching Fields

| Field | Type | Nullable | Notes |
|-------|------|----------|-------|
| `ip` | string | Yes | Format: "6.2" (6 IP, 2 outs) |
| `p_hits` | int | Yes | Hits allowed |
| `p_runs` | int | Yes | Runs allowed |
| `er` | int | Yes | Earned runs |
| `p_bb` | int | Yes | BB allowed |
| `p_k` | int | Yes | Strikeouts |
| `p_hr` | int | Yes | HR allowed |
| `era` | float | Yes | Season ERA |
| `whip` | float | Yes | Season WHIP |
| `pitch_count` | int | Yes | Total pitches |
| `strikes` | int | Yes | Strikes thrown |
| `batters_faced` | int | Yes | BF |
| `pitching_outs` | int | Yes | Outs recorded |
| `wins` | int | Yes | Season W |
| `losses` | int | Yes | Season L |
| `saves` | int | Yes | Season SV |
| `holds` | int | Yes | Season HLD |
| `blown_saves` | int | Yes | BS |
| `games_started` | int | Yes | GS |
| `wild_pitches` | int | Yes | WP |
| `balks` | int | Yes | BK |

#### Fielding Fields (all players)

| Field | Type | Nullable | Notes |
|-------|------|----------|-------|
| `putouts` | int | Yes | PO |
| `assists` | int | Yes | A |
| `errors` | int | Yes | E |
| `fielding_chances` | int | Yes | TC |
| `fielding_pct` | float | Yes | FPCT |

### Season Stats Object Structure

The `/season_stats` endpoint returns **pre-aggregated** season totals:

```json
{
  "player": { /* player object */ },
  "team_name": "Dodgers",
  "season": 2024,
  "postseason": false,
  "season_type": "regular",
  "batting_gp": 157,
  "batting_ab": 576,
  "batting_r": 128,
  "batting_h": 166,
  "batting_avg": 0.288,
  "batting_2b": 31,
  "batting_3b": 4,
  "batting_hr": 41,
  "batting_rbi": 109,
  "batting_tb": 328,
  "batting_bb": 129,
  "batting_so": 119,
  "batting_sb": 7,
  "batting_obp": 0.419,
  "batting_slg": 0.569,
  "batting_ops": 0.989,
  "batting_war": 7.92,
  "pitching_gp": null,
  "pitching_era": null,
  "pitching_w": null,
  "pitching_l": null,
  "pitching_sv": null,
  "pitching_ip": null,
  "pitching_k": null,
  "pitching_whip": null,
  "pitching_war": null,
  "fielding_gp": 157,
  "fielding_fp": 0.994,
  "fielding_e": 2
}
```

**Key Observation:** The `batting_` prefix on all batting stats indicates this endpoint is designed to clearly separate batting vs pitching data.

---

## 6. Pagination Strategy

### Cursor-Based Pagination

```json
{
  "data": [...],
  "meta": {
    "next_cursor": 90,
    "per_page": 25
  }
}
```

### Implementation Pattern

```python
def fetch_all_pages(endpoint, params=None, max_pages=50):
    """Fetch all pages using cursor pagination."""
    results = []
    cursor = None
    page = 0
    
    while page < max_pages:
        request_params = dict(params or {})
        request_params["per_page"] = 100  # Maximize page size
        
        if cursor:
            request_params["cursor"] = cursor
            
        response = session.get(f"{BASE_URL}{endpoint}", params=request_params)
        data = response.json()
        
        results.extend(data.get("data", []))
        
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
            
        page += 1
        time.sleep(0.1)  # Rate limiting
        
    return results
```

### Pagination Parameters

| Parameter | Default | Max | Description |
|-----------|---------|-----|-------------|
| `per_page` | 25 | 100 | Results per request |
| `cursor` | - | - | Opaque cursor for next page |

### Estimated Page Counts

| Data Type | Typical Count | Pages (@100/page) | API Calls |
|-----------|---------------|-------------------|-----------|
| Daily games | 15 | 1 | 1 |
| All players | ~1500 | 15 | 15 |
| Daily stats | ~300 | 3 | 3 |
| All injuries | ~150 | 2 | 2 |
| Full season stats | ~1500 | 15 | 15 |

---

## 7. Optimization Guide

### 7.1 Request Optimization

**❌ Inefficient: Multiple single-date calls**
```python
for date in week_dates:
    stats = fetch_stats(dates=[date])  # 7 calls
```

**✅ Efficient: Batch date filter**
```python
stats = fetch_stats(dates=week_dates)  # 1 call with all dates
```

### 7.2 Batching Strategies

**Filter Parameters Available:**

| Parameter | Applies To | Format | Example |
|-----------|-----------|--------|---------|
| `dates[]` | games, stats | Array of YYYY-MM-DD | `dates[]=2026-04-09&dates[]=2026-04-10` |
| `player_ids[]` | players, stats, injuries | Array of ints | `player_ids[]=208&player_ids[]=1106` |
| `team_ids[]` | games, players, stats | Array of ints | `team_ids[]=14&team_ids[]=19` |
| `game_ids[]` | stats, odds | Array of ints | `game_ids[]=58590` |
| `seasons[]` | games, stats | Array of years | `seasons[]=2025&seasons[]=2026` |

### 7.3 Caching Recommendations

| Data Type | Cache Duration | Rationale |
|-----------|---------------|-----------|
| Teams | 24 hours | Rarely changes |
| Players | 6 hours | Roster changes, trades |
| Games (scheduled) | 1 hour | Lineups, weather |
| Games (completed) | Permanent | Immutable |
| Stats | 15 minutes | Real-time updates |
| Injuries | 5 minutes | Time-sensitive |
| Standings | 30 minutes | Nightly updates |
| Odds | 5 minutes | Live line movement |
| Season Stats | 6 hours | Nightly aggregation |

### 7.4 Time-of-Day Strategy

```python
# Early morning (6-9 AM ET): Aggressive refresh
INGESTION_SCHEDULE = {
    "06:00": ["games", "odds"],           # Day's schedule
    "07:00": ["player_injuries"],         # Morning injury report
    "08:00": ["stats"],                   # Previous night's stats
    "09:00": ["season_stats"],            # Updated aggregates
}

# Daytime (9 AM - 7 PM ET): Live updates
LIVE_SCHEDULE = {
    "every_5min": ["odds"],
    "every_15min": ["games", "injuries"],
}

# Evening (7 PM - 11 PM ET): Real-time
EVENING_SCHEDULE = {
    "every_2min": ["games"],
    "every_5min": ["stats"],
}
```

---

## 8. Error Handling

### HTTP Status Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200 | OK | Process normally |
| 400 | Bad Request | Check parameters |
| 401 | Unauthorized | Verify API key |
| 403 | Forbidden | Upgrade tier |
| 404 | Not Found | Resource doesn't exist |
| 406 | Not Acceptable | Requested non-JSON format |
| 429 | Too Many Requests | Implement backoff |
| 500 | Internal Server Error | Retry with backoff |
| 503 | Service Unavailable | Retry later |

### Error Response Format

```json
{
  "error": "Description of error",
  "message": "Detailed message"
}
```

### Recommended Error Handler

```python
def handle_api_error(response: requests.Response) -> bool:
    """Returns True if should retry, False otherwise."""
    
    if response.status_code == 429:
        retry_after = int(response.headers.get('Retry-After', 60))
        logger.warning(f"Rate limited. Retry after {retry_after}s")
        time.sleep(retry_after)
        return True
        
    elif response.status_code in (500, 502, 503):
        logger.warning(f"Server error {response.status_code}. Retrying...")
        time.sleep(5)
        return True
        
    elif response.status_code == 401:
        logger.error("Invalid API key")
        raise AuthenticationError("Check BALLDONTLIE_API_KEY")
        
    elif response.status_code == 403:
        logger.error("Insufficient tier for endpoint")
        raise TierError("Upgrade to GOAT tier required")
        
    response.raise_for_status()
    return False
```

### Edge Cases

| Edge Case | Behavior | Handling |
|-----------|----------|----------|
| Doubleheader | Two game records | Filter by game_id |
| Postponed game | `status: "POSTPONED"` | Skip stats fetch |
| Incomplete game | Partial stats | Retry in 15 minutes |
| Player traded | Updated team_id | Handle gracefully |
| Missing detailed stats | null fields | Use available fields only |

---

## 9. Code Examples

### Example 1: Fetch Daily Stats

```python
def get_daily_player_stats(target_date: str, per_page: int = 100) -> List[Dict]:
    """
    Fetch all player stats for a specific date.
    
    Args:
        target_date: YYYY-MM-DD format
        per_page: Results per page (max 100)
    
    Returns:
        List of player stat records
    """
    url = "https://api.balldontlie.io/mlb/v1/stats"
    headers = {"Authorization": os.getenv("BALLDONTLIE_API_KEY")}
    
    all_stats = []
    cursor = None
    page = 0
    max_pages = 50
    
    while page < max_pages:
        params = {
            "dates[]": target_date,
            "per_page": per_page
        }
        if cursor:
            params["cursor"] = cursor
            
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        stats = data.get("data", [])
        all_stats.extend(stats)
        
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
            
        page += 1
        time.sleep(0.1)  # Rate limiting
    
    return all_stats
```

### Example 2: Fetch Season Stats for Derived Calculations

```python
def get_season_stats(season: int, player_ids: Optional[List[int]] = None) -> List[Dict]:
    """
    Fetch aggregated season stats for VORP/z-score calculations.
    
    Args:
        season: Season year (e.g., 2026)
        player_ids: Optional list to filter specific players
    
    Returns:
        List of season stat aggregates
    """
    url = "https://api.balldontlie.io/mlb/v1/season_stats"
    headers = {"Authorization": os.getenv("BALLDONTLIE_API_KEY")}
    
    params = {
        "season": season,
        "per_page": 100
    }
    
    # Optional: filter to specific players
    if player_ids:
        for pid in player_ids:
            params.setdefault("player_ids[]", []).append(pid)
    
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    
    return response.json().get("data", [])
```

### Example 3: Batch Player Lookup

```python
def get_players_by_ids(player_ids: List[int]) -> Dict[int, Dict]:
    """
    Efficiently fetch multiple players in batch.
    
    Note: BDL doesn't support player_ids[] on /players endpoint,
    so we fetch all and filter client-side.
    """
    url = "https://api.balldontlie.io/mlb/v1/players"
    headers = {"Authorization": os.getenv("BALLDONTLIE_API_KEY")}
    
    # Fetch all players (cached for efficiency)
    all_players = fetch_all_pages("/players", max_pages=20)
    
    # Build lookup by ID
    return {p["id"]: p for p in all_players if p["id"] in player_ids}
```

### Example 4: Monitor Injuries

```python
def get_current_injuries(team_ids: Optional[List[int]] = None) -> List[Dict]:
    """
    Fetch current injury report with optional team filter.
    
    Args:
        team_ids: Optional filter by team
    
    Returns:
        List of injury records
    """
    url = "https://api.balldontlie.io/mlb/v1/player_injuries"
    headers = {"Authorization": os.getenv("BALLDONTLIE_API_KEY")}
    
    params = {"per_page": 100}
    
    if team_ids:
        for tid in team_ids:
            params.setdefault("team_ids[]", []).append(tid)
    
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    
    injuries = response.json().get("data", [])
    
    # Filter to active/relevant injuries
    return [
        injury for injury in injuries
        if injury.get("status") in ["Out", "Day-To-Day", "IL"]
    ]
```

---

## 10. Comparison: Current vs Optimized Implementation

### Current Implementation (from `balldontlie.py`)

```python
# Good: Uses cursor pagination
# Good: Returns Pydantic-validated objects
# Good: Error handling with logging
# Could improve: No explicit rate limiting
# Could improve: No retry logic for 429/5xx
```

### Recommended Enhancements

```python
class OptimizedBDLClient:
    """Enhanced BDL client with optimizations."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Authorization": api_key})
        
        # Rate limiting: 600/min = 10/sec
        self.min_request_interval = 0.1
        self.last_request_time = 0
        
        # Simple LRU cache for teams/players
        self._cache = {}
        self._cache_ttl = {}
    
    def _request(self, method: str, path: str, **kwargs) -> Dict:
        """Make rate-limited request with retry logic."""
        
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
        url = f"{BASE_URL}{path}"
        
        for attempt in range(3):
            try:
                response = self.session.request(method, url, **kwargs)
                self.last_request_time = time.time()
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited, waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue
                
                # Handle server errors with backoff
                if response.status_code in (500, 502, 503):
                    wait = 2 ** attempt
                    logger.warning(f"Server error, retrying in {wait}s")
                    time.sleep(wait)
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        
        raise RuntimeError("Max retries exceeded")
```

---

## 11. Summary & Action Items

### Key Capabilities Confirmed

1. ✅ **Comprehensive MLB Data**: 2002-present, real-time updates
2. ✅ **Per-Game Stats**: Full box score data via `/stats`
3. ✅ **Aggregated Stats**: Season totals via `/season_stats`
4. ✅ **Injury Data**: IL + DTD via `/player_injuries`
5. ✅ **Player Search**: Name, team, ID lookups
6. ✅ **Game Schedules**: Past and future games
7. ✅ **Odds Data**: Sportsbook lines (GOAT tier)

### Limitations Identified

1. ⚠️ **No Probable Pitchers**: BDL does NOT provide starting pitcher projections
2. ⚠️ **No Advanced Metrics**: No Statcast data (xwOBA, barrel%, exit velocity)
3. ⚠️ **No Live Play-by-Play**: Real-time pitch-level data not available
4. ⚠️ **Per-Sport Pricing**: $39.99/mo per sport (not multi-sport)

### Recommendations for Platform

| Priority | Action | Impact |
|----------|--------|--------|
| HIGH | Use `/season_stats` for derived stats calculation | Reduces computation, improves accuracy |
| HIGH | Implement rate limiting at 10 req/sec | Prevents 429 errors |
| MEDIUM | Add retry logic with exponential backoff | Improves reliability |
| MEDIUM | Use `per_page=100` for all bulk requests | Reduces API calls by 75% |
| LOW | Cache teams/players for 6 hours | Reduces redundant calls |
| LOW | Consider `/players/splits` for platoon advantages | Enhances lineup optimization |

### Probable Pitcher Gap

The platform currently uses MLB Stats API for probable pitchers:
```python
url = "https://statsapi.mlb.com/api/v1/schedule"
```

This is the **correct approach** as BDL does not offer this endpoint.

---

## Sources

1. **Official Documentation**: https://mlb.balldontlie.io/
2. **API Spec**: https://www.balldontlie.io/openapi/mlb.yml
3. **Pricing**: https://www.balldontlie.io/
4. **Python SDK**: https://github.com/balldontlie-api/python

---

*Report generated: April 11, 2026*  
*Research confidence: HIGH (official documentation + live endpoint verification)*
