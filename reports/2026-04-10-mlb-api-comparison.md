# MLB API Comparison Analysis

> **Research Date:** April 11, 2026  
> **Researcher:** Kimi CLI (Deep Intelligence)  
> **Task ID:** K-37  
> **Status:** COMPLETE

---

## 1. Executive Summary

This report compares three major MLB data sources for fantasy baseball platforms: **MLB Stats API** (official), **Baseball Savant (Statcast)**, and **BallDontLie (BDL)**. The platform currently uses BDL as the primary source with MLB Stats API for probable pitchers.

### Recommendation Summary

| Data Need | Primary Source | Fallback | Notes |
|-----------|---------------|----------|-------|
| Live games & scores | BDL | MLB Stats API | BDL more reliable |
| Player stats (box scores) | BDL | MLB Stats API | BDL has cleaner format |
| Probable pitchers | MLB Stats API | ESPN API | Only MLB Stats has this |
| Advanced metrics (xwOBA, barrels) | Statcast (pybaseball) | None | BDL doesn't provide |
| Real-time odds | BDL (GOAT tier) | None | Requires paid tier |
| Injuries | BDL | MLB Stats API | BDL more comprehensive |
| Historical data (2002+) | BDL | MLB Stats API | Both excellent |
| Play-by-play | BDL (GOAT) | MLB Stats API | Both available |

### Key Finding: Probable Pitcher Gap

**BDL does NOT provide probable pitcher data.** The platform's current approach using MLB Stats API is correct:
```
https://statsapi.mlb.com/api/v1/schedule?sportId=1&hydrate=probablePitcher
```

---

## 2. API Profiles

### 2.1 MLB Stats API (Official)

**Overview:** The official MLB Stats API is a comprehensive REST API maintained by Major League Baseball. It powers MLB.com, team websites, and the MLB app.

| Attribute | Details |
|-----------|---------|
| **Base URL** | `https://statsapi.mlb.com` |
| **Versions** | v1 (most), v1.1 (live feeds) |
| **Authentication** | None required (public API) |
| **Rate Limits** | Undocumented; be respectful (suggest <1 req/sec) |
| **Cost** | Free |
| **Data Range** | Historical (1900s) to real-time |

**Key Endpoints:**

| Endpoint | Purpose | Documentation Quality |
|----------|---------|----------------------|
| `/api/v1/schedule` | Games, probable pitchers | Good |
| `/api/v1.1/game/{pk}/feed/live` | Live play-by-play | Good |
| `/api/v1/people/{id}` | Player info | Good |
| `/api/v1/people/{id}/stats` | Player statistics | Complex |
| `/api/v1/teams` | Team rosters | Good |
| `/api/v1/standings` | Division standings | Good |
| `/api/v1/stats/leaders` | League leaders | Spotty |

**Python Library:** `mlbstatsapi`
```bash
pip install mlbstatsapi
```

**Example Usage:**
```python
import requests

# Probable pitchers (used by current platform)
url = "https://statsapi.mlb.com/api/v1/schedule"
params = {
    "sportId": 1,
    "startDate": "2026-04-09",
    "endDate": "2026-04-09",
    "gameType": "R",
    "hydrate": "probablePitcher"
}
response = requests.get(url, params=params)
games = response.json()["dates"][0]["games"]

for game in games:
    home_pitcher = game.get("teams", {}).get("home", {}).get("probablePitcher", {})
    away_pitcher = game.get("teams", {}).get("away", {}).get("probablePitcher", {})
    print(f"Home: {home_pitcher.get('fullName', 'TBD')}")
    print(f"Away: {away_pitcher.get('fullName', 'TBD')}")
```

**Strengths:**
- ✅ Official source, authoritative
- ✅ No authentication required
- ✅ Probable pitcher data available
- ✅ Comprehensive historical data
- ✅ Real-time play-by-play (v1.1)

**Weaknesses:**
- ⚠️ Complex response structures (deeply nested)
- ⚠️ Undocumented rate limits
- ⚠️ Inconsistent documentation
- ⚠️ Some endpoints unreliable (league leaders)
- ⚠️ No advanced Statcast metrics

---

### 2.2 Baseball Savant (Statcast)

**Overview:** Baseball Savant provides advanced metrics via Statcast, a high-speed tracking system using Hawk-Eye cameras (12 per stadium). Data available from 2015-present.

| Attribute | Details |
|-----------|---------|
| **Base URL** | `https://baseballsavant.mlb.com` |
| **Access Method** | CSV export via web interface / scraping |
| **Authentication** | None (web scraping) |
| **Rate Limits** | Implicit; use responsibly |
| **Cost** | Free |
| **Data Range** | 2015-present (Statcast era) |

**Key Metrics Available:**

| Metric | Description | Fantasy Use |
|--------|-------------|-------------|
| **xwOBA** | Expected weighted on-base average | True skill indicator |
| **Barrel%** | Batted balls with ideal EV/LA | Power predictor |
| **Exit Velocity (EV)** | Speed of batted ball (mph) | Hard contact |
| **Launch Angle (LA)** | Vertical angle of batted ball | Batted ball profile |
| **Hard Hit%** | Balls hit 95+ mph | Quality of contact |
| **Sprint Speed** | Feet/second on bases | Stolen base potential |
| **Bat Speed** | Speed of bat through zone | Power indicator |
| **Swing Length** | Length of swing path | Contact ability |

**Python Library:** `pybaseball`
```bash
pip install pybaseball
```

**Example Usage:**
```python
from pybaseball import statcast, statcast_batter, playerid_lookup

# Get all Statcast data for date range
# (This is what statcast_ingestion.py does)
data = statcast(start_dt='2026-04-01', end_dt='2026-04-09')

# Key columns: launch_speed, launch_angle, barrel, xwoba, etc.
print(data[['player_name', 'launch_speed', 'xwoba', 'barrel']].head())

# Batter-specific stats
from pybaseball import statcast_batter
player_id = playerid_lookup('Ohtani', 'Shohei').iloc[0]['key_mlbam']
ohtani_data = statcast_batter(start_dt='2026-04-01', end_dt='2026-04-09', player_id=player_id)
```

**Statcast Data Available:**

The `statcast()` function returns pitch-level data with 90+ columns including:
- Release speed, spin rate, pitch type
- Exit velocity, launch angle, hit distance
- Expected stats (xBA, xSLG, xwOBA)
- Batted ball classification (barrel, solid, flare, etc.)

**Strengths:**
- ✅ Advanced metrics unavailable elsewhere
- ✅ Predictive power for fantasy performance
- ✅ Pitch-level granularity
- ✅ xwOBA > wOBA for predictive modeling
- ✅ Free access via pybaseball

**Weaknesses:**
- ⚠️ Data delay (typically next day, not real-time)
- ⚠️ Web scraping approach (not official API)
- ⚠️ Limited to 2015-present
- ⚠️ Large data volumes (every pitch)
- ⚠️ Requires data aggregation for player-level insights

**Current Platform Usage:**
The platform already uses Statcast via `statcast_ingestion.py`:
```python
# From backend/fantasy_baseball/statcast_ingestion.py
base_url = "https://baseballsavant.mlb.com/statcast_search/csv"
# Fetches: exit_velocity_avg, launch_angle_avg, hard_hit_pct, 
#          barrel_pct, xba, xslg, xwoba
```

---

### 2.3 BallDontLie (BDL)

**Overview:** Third-party sports data API providing clean, well-documented MLB data. Used as the platform's primary data source.

| Attribute | Details |
|-----------|---------|
| **Base URL** | `https://api.balldontlie.io/mlb/v1` |
| **Authentication** | API key (Authorization header) |
| **Rate Limits** | 600 req/min (GOAT tier) |
| **Cost** | $39.99/mo (GOAT tier) |
| **Data Range** | 2002-present |

**Key Endpoints:**

| Endpoint | Data | Tier |
|----------|------|------|
| `/games` | Schedule, scores, line scores | Free |
| `/stats` | Per-game player box scores | All-Star+ |
| `/season_stats` | Aggregated season totals | GOAT |
| `/players` | Player info, rosters | Free |
| `/player_injuries` | IL and DTD status | All-Star+ |
| `/odds` | Sportsbook lines | GOAT |
| `/plays` | Play-by-play | GOAT |

**Strengths:**
- ✅ Clean, consistent JSON responses
- ✅ Excellent documentation
- ✅ No complex hydration parameters
- ✅ Betting odds integration
- ✅ Fast response times
- ✅ Pydantic-friendly schemas

**Weaknesses:**
- ⚠️ **No probable pitcher data**
- ⚠️ **No Statcast advanced metrics**
- ⚠️ Monthly cost ($39.99/sport)
- ⚠️ Per-sport pricing (not bundled)

---

## 3. Feature Comparison Matrix

| Feature | MLB Stats API | Statcast (pybaseball) | BDL | Winner |
|---------|---------------|----------------------|-----|--------|
| **Authentication** | None | None | API Key | MLB Stats/Statcast |
| **Cost** | Free | Free | $39.99/mo | MLB Stats/Statcast |
| **Rate Limit** | Undocumented | N/A (scraping) | 600/min | BDL (clear) |
| **Documentation** | Fair | Good | Excellent | BDL |
| **Response Format** | Complex nested | DataFrame | Clean JSON | BDL |
| **Historical Data** | 1900s-present | 2015-present | 2002-present | MLB Stats |
| **Real-time** | Yes | No (next day) | Yes | Tie |
| **Probable Pitchers** | ✅ Yes | ❌ No | ❌ No | **MLB Stats** |
| **xwOBA/Barrel%** | ❌ No | ✅ Yes | ❌ No | **Statcast** |
| **Exit Velocity** | ❌ No | ✅ Yes | ❌ No | **Statcast** |
| **Live Odds** | ❌ No | ❌ No | ✅ Yes | **BDL** |
| **Injuries** | ✅ Yes | ❌ No | ✅ Yes | Tie |
| **Python Library** | mlbstatsapi | pybaseball | balldontlie | pybaseball |
| **Play-by-Play** | ✅ Yes | ❌ No | ✅ Yes (GOAT) | Tie |
| **Data Freshness** | Real-time | ~24h delay | Real-time | Tie |

---

## 4. Probable Pitchers Deep Dive

### Current Implementation Analysis

The platform correctly uses MLB Stats API for probable pitchers:

```python
# From backend/routers/fantasy.py and backend/main.py
url = (
    "https://statsapi.mlb.com/api/v1/schedule"
    f"?sportId=1&startDate={start_date}&endDate={end_date}"
    "&gameType=R&hydrate=probablePitcher"
)
```

### Alternative Sources Tested

| Source | Probable Pitchers? | Reliability | Notes |
|--------|-------------------|-------------|-------|
| **MLB Stats API** | ✅ Yes | High | Official source |
| **ESPN API** | ✅ Yes | Medium | `site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard` |
| **BDL** | ❌ No | N/A | Not available |
| **Baseball Reference** | ⚠️ Partial | Low | Scraping required |
| **Yahoo Fantasy API** | ⚠️ Partial | Medium | Requires OAuth |

### ESPN API Alternative

```python
# ESPN API (backup option)
url = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
params = {"dates": "20260409"}
response = requests.get(url, params=params)

for event in response.json().get("events", []):
    for competitor in event.get("competitions", [])[0].get("competitors", []):
        prob_pitcher = competitor.get("probables", [{}])[0].get("athlete", {})
        print(f"{competitor['team']['abbreviation']}: {prob_pitcher.get('displayName', 'TBD')}")
```

**Recommendation:** Keep MLB Stats API as primary for probable pitchers. ESPN API as fallback if MLB Stats is down.

---

## 5. Statcast Metrics for Fantasy

### Key Metrics and Their Fantasy Relevance

| Metric | What It Measures | Stabilizes | Fantasy Application |
|--------|-----------------|------------|---------------------|
| **xwOBA** | Expected production based on batted ball quality | ~100 PA | Better than wOBA for predicting future performance |
| **Barrel%** | Ideal contact (98+ mph, 25-30° LA) | ~50 BBE | Correlates with HR (r² = 0.73) |
| **Hard Hit%** | 95+ mph exit velocity | ~50 BBE | Quality of contact indicator |
| **Average EV** | Mean exit velocity | ~25 BBE | Raw power indicator |
| **EV50** | Mean of top 50% EV | ~25 BBE | Shows ceiling power |
| **Sprint Speed** | Fastest 1-second window | Immediate | SB predictor |
| **Sweet Spot%** | 8-32° launch angle | ~50 BBE | Line drive rate proxy |

### Statcast vs Traditional Stats

| Traditional | Statcast Equivalent | Why Statcast is Better |
|-------------|---------------------|----------------------|
| AVG | xBA | Removes defense/luck variance |
| SLG | xSLG | Based on quality of contact |
| wOBA | xwOBA | Predictive of future performance |
| HR | Barrel% | HR are noisy; barrels are skill |
| BABIP | Hard Hit% | BABIP is luck; hard hit is skill |

### Implementation Recommendation

The platform already has good Statcast integration in `statcast_ingestion.py`. Enhancements to consider:

```python
# Additional metrics to track from Statcast:
ADDITIONAL_STATCAST_METRICS = [
    "bat_speed",          # New in 2024 (bat tracking)
    "swing_length",       # New in 2024 (bat tracking)
    "sprint_speed",       # Baserunning speed
    "arm_strength",       # OF throwing
    "jump",               # OF reaction time
    "exchange",           # Catcher pop time component
]
```

---

## 6. Integration Strategy

### Recommended Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Source Strategy                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   BDL        │  │  MLB Stats   │  │  Statcast    │      │
│  │  ($39.99/mo) │  │    (Free)    │  │  (pybaseball)│      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │             │
│    Primary              Fallback           Advanced         │
│    Source               Source             Metrics          │
│         │                  │                  │             │
│    • Games            • Probable         • xwOBA           │
│    • Box scores         pitchers         • Barrels         │
│    • Injuries         • Deep historical  • Exit Velocity   │
│    • Odds             • Transactions     • Sprint Speed    │
│    • Standings                                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Source Assignment by Data Type

| Data Category | Primary | Fallback | Update Frequency |
|---------------|---------|----------|------------------|
| Game schedule | BDL | MLB Stats | Daily |
| Box scores | BDL | MLB Stats | Real-time |
| Probable pitchers | MLB Stats | ESPN API | Daily |
| Player stats (season) | BDL /season_stats | MLB Stats | Daily |
| Advanced metrics | Statcast | None | Daily (delayed) |
| Injuries | BDL | MLB Stats | Every 15 min |
| Odds | BDL | None | Every 5 min |
| Standings | BDL | MLB Stats | Hourly |

### Cost Optimization

**Current Monthly Costs:**
- BDL GOAT tier: $39.99
- Statcast (pybaseball): Free
- MLB Stats API: Free

**Total: $39.99/month**

If cost reduction needed:
- Downgrade BDL to All-Star ($9.99): Lose season_stats, odds, plays
- Implement more MLB Stats API as fallback
- Estimated savings: $30/month with moderate functionality loss

---

## 7. Implementation Roadmap

### Phase 1: Immediate (Week 1)

1. ✅ **Verify current MLB Stats API integration for probable pitchers**
   - Status: Already implemented correctly
   - No changes needed

2. **Add ESPN API as fallback for probable pitchers**
   ```python
   def get_probable_pitchers(date: str) -> Dict:
       try:
           return get_from_mlb_stats(date)
       except Exception:
           return get_from_espn(date)  # Fallback
   ```

### Phase 2: Short-term (Weeks 2-4)

1. **Enhance Statcast integration**
   - Add more metrics to `statcast_ingestion.py`
   - Implement barrel% tracking
   - Add sprint speed for SB prediction

2. **Implement data source failover**
   ```python
   class MLBDataManager:
       def get_games(self, date):
           try:
               return self.bdl.get_games(date)
           except BDLException:
               return self.mlb_stats.get_games(date)
   ```

### Phase 3: Long-term (Months 2-3)

1. **Evaluate All-Access tier**
   - BDL All-Access: $499.99/mo for all sports
   - Only worthwhile if adding NBA/NFL/others

2. **Historical backfill**
   - Use MLB Stats API for pre-2002 historical data
   - BDL only goes back to 2002

---

## 8. Code Examples

### Unified Data Client

```python
from typing import Optional, List, Dict
import requests

class MLBDataClient:
    """
    Unified MLB data client with automatic fallback.
    """
    
    def __init__(self, bdl_api_key: str):
        self.bdl = BDLClient(api_key=bdl_api_key)
        self.mlb_stats = MLBStatsClient()
        self.statcast = StatcastClient()
    
    def get_probable_pitchers(self, date: str) -> Dict[str, str]:
        """
        Get probable pitchers with fallback.
        
        Priority:
        1. MLB Stats API (official)
        2. ESPN API (fallback)
        """
        try:
            return self.mlb_stats.get_probable_pitchers(date)
        except Exception:
            # Fallback to ESPN
            return self._get_from_espn(date)
    
    def get_advanced_metrics(self, player_id: str) -> Dict:
        """
        Get Statcast advanced metrics.
        
        Source: Statcast (only source available)
        """
        return self.statcast.get_player_metrics(player_id)
    
    def get_games(self, date: str) -> List[Dict]:
        """
        Get games with fallback.
        
        Priority:
        1. BDL (cleaner responses)
        2. MLB Stats API (fallback)
        """
        try:
            return self.bdl.get_games(date)
        except Exception:
            return self.mlb_stats.get_games(date)
```

### MLB Stats API - Probable Pitchers

```python
def get_probable_pitchers_mlb_stats(date: str) -> Dict[str, Dict]:
    """
    Fetch probable pitchers from MLB Stats API.
    
    Returns dict mapping team abbreviation to pitcher info.
    """
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "startDate": date,
        "endDate": date,
        "gameType": "R",
        "hydrate": "probablePitcher"
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    data = response.json()
    pitchers = {}
    
    for date_data in data.get("dates", []):
        for game in date_data.get("games", []):
            for team_type in ["home", "away"]:
                team = game.get("teams", {}).get(team_type, {})
                team_abbr = team.get("team", {}).get("abbreviation", "")
                pitcher = team.get("probablePitcher", {})
                
                if team_abbr and pitcher:
                    pitchers[team_abbr] = {
                        "name": pitcher.get("fullName", "TBD"),
                        "id": pitcher.get("id"),
                        "hand": pitcher.get("pitchHand", {}).get("code", "R")
                    }
    
    return pitchers
```

### Statcast Integration

```python
from pybaseball import statcast
import pandas as pd

def get_daily_statcast_summary(date: str) -> pd.DataFrame:
    """
    Get aggregated Statcast metrics for a date.
    
    Returns player-level aggregates from pitch-level data.
    """
    # Fetch pitch-level data
    data = statcast(start_dt=date, end_dt=date)
    
    if data.empty:
        return pd.DataFrame()
    
    # Aggregate to player level
    summary = data.groupby("player_name").agg({
        "launch_speed": "mean",        # Avg exit velocity
        "launch_angle": "mean",        # Avg launch angle
        "barrel": "sum",               # Total barrels
        "xwoba": "mean",               # Avg xwOBA
        "estimated_ba_using_speedangle": "mean",  # xBA
    }).round(3)
    
    summary["barrel_rate"] = (
        data.groupby("player_name")["barrel"].sum() / 
        data.groupby("player_name").size() * 100
    ).round(1)
    
    return summary.reset_index()
```

---

## 9. Summary & Action Items

### Key Findings

1. **Current architecture is correct:** BDL for most data, MLB Stats API for probable pitchers
2. **Statcast integration is valuable:** Provides xwOBA, barrel% for advanced analysis
3. **No single source has everything:** Multi-source approach is necessary
4. **Cost is reasonable:** $39.99/month for comprehensive MLB data

### Recommendations

| Priority | Action | Owner | Timeline |
|----------|--------|-------|----------|
| HIGH | Document MLB Stats API as primary probable pitcher source | Architecture | Done |
| HIGH | Verify ESPN API fallback for probable pitchers | Backend | Week 1 |
| MEDIUM | Enhance Statcast metrics in ingestion pipeline | Data Eng | Week 2-3 |
| MEDIUM | Implement data source failover logic | Backend | Week 3-4 |
| LOW | Evaluate cost reduction (All-Star tier) | Product | Month 2 |

### Data Source Hierarchy

```
For each data type:
┌────────────────────────────────────────┐
│ 1. Try BDL (if available for data type)│
│    → Success: Return data              │
│    → Failure: Continue                 │
│ 2. Try MLB Stats API                   │
│    → Success: Return data              │
│    → Failure: Continue                 │
│ 3. Try ESPN API (limited endpoints)    │
│    → Success: Return data              │
│    → Failure: Return error             │
└────────────────────────────────────────┘
```

---

## Sources

1. **MLB Stats API Documentation:** https://github.com/toddrob99/MLB-StatsAPI/wiki
2. **pybaseball Documentation:** https://github.com/jldbc/pybaseball
3. **Baseball Savant:** https://baseballsavant.mlb.com
4. **BDL MLB API:** https://mlb.balldontlie.io/
5. **Statcast Glossary:** https://baseballsavant.mlb.com/csv-docs

---

*Report generated: April 11, 2026*  
*Research confidence: HIGH (official documentation + library verification)*
