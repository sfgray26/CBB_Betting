# Yahoo Fantasy Baseball API Research Report

**Research Date:** March 11, 2026  
**Draft Date:** March 23, 2026 @ 7:30am ET  
**Researcher:** Kimi CLI

---

## Executive Summary

The Yahoo Fantasy Sports API provides **limited live draft capabilities**. While it can retrieve draft results, it does **NOT** support true real-time streaming or websockets for live pick updates. For your March 23 draft, you'll need to implement a **polling-based approach** to track picks and available players.

---

## 1. Yahoo Fantasy API Draft Capabilities

### 1.1 Available Endpoints

#### `draft_results()` - Primary Draft Endpoint
```python
# Returns all picks made in the draft
GET https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/draftresults
```

**Key Characteristics:**
- ✅ Returns picks as they happen (during live draft)
- ✅ Updates in real-time when polled
- ❌ **NO websockets** - requires polling
- ❌ **NO push notifications** 
- ❌ Does NOT include currently nominating player (auction drafts)

**Response Format:**
```json
[
  {
    "pick": 1,
    "round": 1,
    "cost": "4",           // Auction price (if applicable)
    "team_key": "388.l.27081.t.4",
    "player_id": 9490
  },
  {
    "pick": 2,
    "round": 1,
    "team_key": "388.l.27081.t.1", 
    "player_id": 7569
  }
]
```

#### Player Pool Endpoint
```python
# Get all players in league context
GET https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/players

# Filter available players only
GET https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/players;status=A
```

**Player Status Filters:**
| Status | Description |
|--------|-------------|
| `A` | All available players |
| `FA` | Free agents only |
| `T` | All taken players |
| `K` | Keepers only |

### 1.2 Live Draft Polling Strategy

**Recommended Implementation:**
```python
import asyncio
import time

async def poll_draft_updates(league, last_pick_count=0):
    """
    Poll draft_results() every 5-10 seconds during live draft
    """
    while True:
        draft_results = league.draft_results()
        current_picks = len(draft_results)
        
        if current_picks > last_pick_count:
            # New pick(s) detected!
            new_picks = draft_results[last_pick_count:]
            for pick in new_picks:
                await process_new_pick(pick)
            last_pick_count = current_picks
            
        await asyncio.sleep(5)  # 5-second polling interval
```

**Rate Limiting Considerations:**
- Yahoo returns HTTP 999 when rate limited
- Recommended: Max 1 request per 5 seconds during draft
- Use exponential backoff on errors

### 1.3 Limitations for Live Draft Integration

| Feature | Available | Notes |
|---------|-----------|-------|
| Real-time pick updates | ⚠️ Partial | Requires polling |
| WebSocket streaming | ❌ No | Not supported |
| Draft clock/timer | ❌ No | Not in API |
| Pick notifications | ❌ No | Must poll for changes |
| On-the-clock team | ❌ No | Not exposed |
| Auction nominations | ❌ No | Not in API |

---

## 2. Alternative Baseball Data APIs

### 2.1 MLB Stats API (Official)

**Base URL:** `https://statsapi.mlb.com/api/v1/`

**Key Endpoints:**
```python
# Player stats
GET /people/{playerId}/stats?stats=season&season=2026

# League leaders
GET /stats/leaders?season=2026&statType=season

# Player search
GET /people/search?names={player_name}

# Teams roster
GET /teams/{teamId}/roster
```

**Pros:**
- ✅ Official MLB data
- ✅ Free, no API key required
- ✅ Real stats (not projections)
- ✅ Comprehensive player info

**Cons:**
- ❌ No fantasy-specific data
- ❌ No ADP or rankings
- ❌ Rate limits undocumented

### 2.2 Fangraphs (Projections Leader)

**Access Method:** Web scraping or CSV downloads

**Available Projections:**
| System | Type | Accuracy Rank (2024) |
|--------|------|----------------------|
| THE BAT X | Paid | #1 Overall |
| ATC | Aggregated | #3 Overall |
| Steamer | Free | #6 Overall |
| ZiPS | Free | #8 Overall |
| Depth Charts | Free | #5 Overall |

**Recommended:** Download projections CSV from:
- https://www.fangraphs.com/projections.aspx?pos=all&stats=bat&type=atc
- https://www.fangraphs.com/projections.aspx?pos=all&stats=pit&type=atc

### 2.3 FantasyPros API

**Access:** Unofficial/undocumented endpoints

**Available Data:**
- ADP (Average Draft Position)
- Expert consensus rankings
- Player notes/injuries

**ADP Endpoint Pattern:**
```
https://www.fantasypros.com/mlb/adp/overall.php
https://www.fantasypros.com/mlb/adp/sp.php  # Starting Pitchers
```

### 2.4 Baseball-Reference

**Access:** No official API, but data available via:
- `baseballreference` Python package
- Web scraping
- CSV exports

### 2.5 Sleeper API (Football-Only)

**Important:** Sleeper API currently supports **NFL only** - no MLB support as of 2026.

---

## 3. Recommended Integration Architecture

### 3.1 Pre-Draft Setup

```python
# 1. Load projection data from Fangraphs
fangraphs_projections = load_fangraphs_csv('atc_projections_2026.csv')

# 2. Load ADP data from FantasyPros  
fantasypros_adp = scrape_fantasypros_adp()

# 3. Merge with Yahoo player pool
yahoo_players = league.players()
combined_player_data = merge_player_data(
    yahoo=yahoo_players,
    fangraphs=fangraphs_projections,
    adp=fantasypros_adp
)
```

### 3.2 During Live Draft

```python
async def draft_tracker(league):
    """
    Real-time draft tracking with Yahoo polling
    """
    draft_results = league.draft_results()
    last_count = len(draft_results)
    
    while draft_in_progress(league):
        # Poll for new picks
        await asyncio.sleep(5)
        current_results = league.draft_results()
        
        if len(current_results) > last_count:
            new_picks = current_results[last_count:]
            for pick in new_picks:
                # Send to Discord
                await notify_discord_new_pick(pick)
                
                # Update available players
                update_player_pool(pick['player_id'])
                
            last_count = len(current_results)
            
        # Check if we're on the clock
        if is_my_pick(league, current_results):
            await send_draft_alert("YOU'RE ON THE CLOCK!")
```

### 3.3 Discord Integration

**Channel Recommendations:**
| Channel | Purpose |
|---------|---------|
| `#fantasy-draft-live` | Real-time pick updates |
| `#fantasy-draft-alerts` | On-the-clock notifications |
| `#fantasy-player-pool` | Available players by position |
| `#fantasy-recommendations` | AI pick suggestions |

---

## 4. Data Integration Strategy

### 4.1 Player ID Mapping

**Challenge:** Yahoo uses proprietary player IDs that don't match other sources.

**Solution:** Map by:
1. Player name (cleaned/normalized)
2. MLB team
3. Position

```python
def normalize_player_name(name):
    """Normalize for cross-source matching"""
    name = name.lower().strip()
    name = name.replace('.', '')  # J.T. Realmuto -> jt realmuto
    name = name.replace('-', ' ') # Juan-Paul -> juan paul
    return name
```

### 4.2 Recommended Data Sources by Use Case

| Use Case | Primary Source | Backup Source |
|----------|---------------|---------------|
| Draft picks (live) | Yahoo API | Manual entry |
| Player projections | Fangraphs ATC | THE BAT X (paid) |
| ADP/Rankings | FantasyPros | Yahoo rankings |
| Injury updates | Yahoo API | MLB.com |
| Player stats | MLB Stats API | Baseball-Reference |

---

## 5. Implementation Checklist

### Pre-Draft (Before March 23)
- [ ] Set up Yahoo OAuth flow (user needs to authenticate once)
- [ ] Download 2026 projections from Fangraphs
- [ ] Scrape current ADP from FantasyPros
- [ ] Build player ID mapping table
- [ ] Test polling loop with mock data
- [ ] Configure Discord webhook URLs

### Draft Day
- [ ] Start polling loop 15 min before draft
- [ ] Monitor rate limits
- [ ] Have manual override ready (if API fails)

### Post-Draft
- [ ] Save final rosters
- [ ] Set up season-long tracking

---

## 6. Code Samples

### Yahoo OAuth Setup
```python
from yahoo_oauth import OAuth2

# One-time auth (user must visit URL)
oauth = OAuth2(
    consumer_key=os.getenv('YAHOO_CLIENT_ID'),
    consumer_secret=os.getenv('YAHOO_CLIENT_SECRET')
)
# Save token for future use
```

### Get Available Players
```python
from yahoo_fantasy_api import League

sc = oauth.get_session()
league = League(sc, 'YOUR_LEAGUE_ID')

# Get all available players
available = league.players(status='A', count=1000)

# Get free agents by position
free_agent_pitchers = league.free_agents('SP')
```

### Draft Results Parser
```python
def parse_draft_results(league):
    """Get current draft state"""
    results = league.draft_results()
    
    drafted_players = []
    for pick in results:
        player_details = league.player_details(pick['player_id'])
        drafted_players.append({
            'pick_number': pick['pick'],
            'round': pick['round'],
            'team': pick['team_key'],
            'player': player_details['name']['full'],
            'position': player_details['display_position']
        })
    
    return drafted_players
```

---

## 7. Key Findings Summary

### ✅ What IS Possible
1. **Polling-based live updates** - Get draft picks within 5-10 seconds
2. **Available player tracking** - Filter by status='A' for remaining players
3. **Player details lookup** - Full stats and info via `player_details()`
4. **Post-draft roster sync** - Full team rosters accessible

### ❌ What is NOT Possible
1. **True real-time streaming** - No websockets/push notifications
2. **Auction nomination tracking** - Not exposed in API
3. **Draft timer visibility** - Must infer from pick timing
4. **Automated draft picks** - No write support for making picks

---

## 8. References

- [Yahoo Fantasy Sports API Guide](https://developer.yahoo.com/fantasysports/guide/)
- [yahoo_fantasy_api Python Library](https://yahoo-fantasy-api.readthedocs.io/)
- [Fangraphs Projections](https://www.fangraphs.com/projections.aspx)
- [FantasyPros ADP](https://www.fantasypros.com/mlb/adp/overall.php)
- [MLB Stats API Docs](https://statsapi.mlb.com/docs/)

---

**Report Generated:** March 11, 2026  
**Next Steps:** Implement polling-based draft tracker, set up OAuth flow
