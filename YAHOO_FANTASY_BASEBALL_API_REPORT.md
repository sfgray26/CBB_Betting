# Yahoo Fantasy Baseball API & Elite Fantasy Baseball Tools Report
## Comprehensive Research for CBB Edge Analyzer Expansion

**Prepared for:** Claude / CBB Edge Analyzer Team  
**Date:** March 24, 2026  
**Classification:** Internal Research Document  
**Version:** 1.0

---

## Executive Summary

This report provides a comprehensive analysis of the **Yahoo Fantasy Baseball API** and identifies the key characteristics of **successful, elite fantasy baseball applications**. The CBB Edge Analyzer's existing architecture (Python/FastAPI, statistical modeling, real-time data) provides a strong foundation for building a best-in-class fantasy baseball tool.

**Key Finding:** The most successful fantasy baseball tools (FantasyPros Draft Wizard, RotoLab, RotoWire) combine three core elements:
1. **Accurate, customizable projections** (with multiple sources)
2. **Real-time draft assistance** (ADP-aware, position-aware recommendations)
3. **Seamless platform integration** (Yahoo/ESPN/CBS league sync)

---

## Part 1: Yahoo Fantasy Baseball API Deep Dive

### 1.1 API Overview

**Base URL:** `https://fantasysports.yahooapis.com/fantasy/v2/`

**Authentication:** OAuth 2.0 (OAuth 1.0a legacy support)
- Requires Yahoo Developer Network app registration
- Consumer Key + Consumer Secret
- Access Token + Refresh Token flow

**Official Documentation:** https://developer.yahoo.com/fantasysports/guide/

**Important Note:** Yahoo's API documentation is notoriously difficult to navigate. The community has built wrapper libraries to simplify integration.

### 1.2 API Resources (Endpoints)

The Yahoo Fantasy API exposes 7 primary resources:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        YAHOO FANTASY API RESOURCES                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. GAME                                                                    │
│     ├── Game metadata (sport, season, game key)                            │
│     ├── League IDs for a game                                              │
│     └── URL: /game/{game_key}                                              │
│                                                                             │
│  2. LEAGUE                                                                  │
│     ├── League settings, scoring, roster positions                         │
│     ├── Standings                                                          │
│     ├── Scoreboard                                                         │
│     ├── Teams                                                              │
│     └── URL: /league/{league_key}                                          │
│                                                                             │
│  3. TEAM                                                                    │
│     ├── Roster (with week parameter)                                       │
│     ├── Stats (season or weekly)                                           │
│     ├── Standings                                                          │
│     ├── Draft Results                                                      │
│     ├── Matchups                                                           │
│     └── URL: /team/{team_key}                                              │
│                                                                             │
│  4. PLAYER                                                                  │
│     ├── Player stats, info, ownership %                                    │
│     ├── Percent drafted/started                                            │
│     └── URL: /player/{player_key}                                          │
│                                                                             │
│  5. ROSTER                                                                  │
│     ├── CRITICAL: Manage lineup positions                                  │
│     ├── PUT to set active/bench                                            │
│     └── URL: /team/{team_key}/roster                                       │
│                                                                             │
│  6. TRANSACTION                                                             │
│     ├── Add/drop/trade transactions                                        │
│     ├── Waiver claims                                                      │
│     ├── PUT to execute transactions                                        │
│     └── URL: /transaction/{transaction_key}                                │
│                                                                             │
│  7. USER                                                                    │
│     ├── User profile                                                       │
│     ├── Games played                                                       │
│     └── URL: /users;use_login=1                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Key Endpoint Details

#### Game Keys (Season Identifiers)

| Season | MLB Game Key |
|--------|--------------|
| 2025 | 412 |
| 2026 | 458 |

**League Key Format:** `{game_key}.l.{league_id}`  
**Example:** `458.l.123456` (2026 MLB League #123456)

**Team Key Format:** `{game_key}.l.{league_id}.t.{team_number}`  
**Example:** `458.l.123456.t.1` (Team 1 in that league)

#### Critical Endpoints for Fantasy Baseball Tool

**1. Get League Settings**
```http
GET https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/settings
```

Returns:
- Scoring categories (5x5, points, custom)
- Roster positions (C, 1B, 2B, 3B, SS, OF, Util, SP, RP, etc.)
- Bench slots, IL slots
- Draft type (snake, auction)
- Trade deadline, playoff settings

**2. Get User's Teams**
```http
GET https://fantasysports.yahooapis.com/fantasy/v2/users;use_login=1/games;game_keys=mlb/teams
```

Returns all MLB teams the authenticated user belongs to.

**3. Get Team Roster**
```http
GET https://fantasysports.yahooapis.com/fantasy/v2/team/{team_key}/roster;week={week}
```

Returns:
- Current lineup with positions
- Player keys for each roster spot
- Selected position (C, 1B, BN, IL, etc.)

**4. Get Player Stats**
```http
GET https://fantasysports.yahooapis.com/fantasy/v2/player/{player_key}/stats
```

Returns season stats or weekly stats for a player.

**5. Get Free Agents (Waiver Wire)**
```http
GET https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/players;status=A;position={pos};sort=AR
```

**Query Parameters:**
- `status=A` (Available)
- `position=B` (Batters), `P` (Pitchers), or specific position
- `sort=AR` (By rank)

**Pagination:** 25 players per request, use `start=` parameter

**6. Set Lineup (PUT Request)**
```http
PUT https://fantasysports.yahooapis.com/fantasy/v2/team/{team_key}/roster
```

Request Body (XML):
```xml
<?xml version="1.0" encoding="UTF-8"?>
<fantasy_content>
  <roster>
    <coverage_type>week</coverage_type>
    <week>1</week>
    <players>
      <player>
        <player_key>458.p.6619</player_key>
        <selected_position>
          <position>C</position>
        </selected_position>
      </player>
      <!-- More players... -->
    </players>
  </roster>
</fantasy_content>
```

**7. Execute Transaction (Add/Drop)**
```http
POST https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/transactions
```

Request Body (XML):
```xml
<?xml version="1.0" encoding="UTF-8"?>
<fantasy_content>
  <transaction>
    <type>add</type>
    <player>
      <player_key>458.p.6619</player_key>
    </player>
  </transaction>
</fantasy_content>
```

### 1.4 Python Integration Libraries

Several Python libraries wrap the Yahoo Fantasy API:

#### Option 1: yahoofantasy (Recommended)

```bash
pip install yahoofantasy
```

**Features:**
- OAuth2 authentication handling
- Clean Pythonic interface
- CLI tool included
- Active maintenance (GitHub: mattdodge/yahoofantasy)

**Basic Usage:**
```python
from yahoofantasy import Context

# Authenticate (opens browser for OAuth)
ctx = Context()

# Get MLB leagues for 2026
leagues = ctx.get_leagues('mlb', 2026)
for league in leagues:
    print(f"{league.id} - {league.name}")

# Access specific league
league = leagues[0]
print(f"Teams: {len(league.teams())}")

# Get standings
for team in league.standings():
    outcomes = team.team_standings.outcome_totals
    print(f"#{team.team_standings.rank}\t{team.name}\t"
          f"({outcomes.wins}-{outcomes.losses}-{outcomes.ties})")

# Get your team's roster
team = league.teams()[0]  # Your team
roster = team.players()
for player in roster:
    print(f"{player.name.full} - {player.display_position}")

# Get free agents
free_agents = league.free_agents('B')  # Available batters
```

#### Option 2: yfpy (YFPY)

```bash
pip install yfpy
```

**Features:**
- More low-level control
- Docker support
- Comprehensive test suite
- Documentation: https://yfpy.uberfastman.com

**Basic Usage:**
```python
from yfpy.query import YahooFantasySportsQuery

# Initialize
query = YahooFantasySportsQuery(
    auth_dir="auth/",
    league_id="123456",
    game_id="mlb",
    game_code="mlb",
    season=2026,
    consumer_key="YOUR_KEY",
    consumer_secret="YOUR_SECRET"
)

# Get league info
league_info = query.get_league_info()

# Get roster
roster = query.get_team_roster(team_id=1, week=1)

# Get player stats
player_stats = query.get_player_stats(player_id=6619, week=1)
```

#### Option 3: yahoo_fantasy_api (ReadTheDocs)

```bash
pip install yahoo_fantasy_api
```

**Features:**
- Well-documented on ReadTheDocs
- Direct mapping to API resources
- Team and League classes

### 1.5 Authentication Setup

**Step 1: Create Yahoo Developer App**
1. Go to https://developer.yahoo.com/apps/create/
2. Fill in application details:
   - Application Name: "CBB Edge Fantasy Baseball"
   - Application Type: "Installed Application"
   - API Permissions: Fantasy Sports (Read/Write)
   - Callback Domain: `localhost` (for development)
3. Note your **Consumer Key** and **Consumer Secret**

**Step 2: OAuth Flow (yahoofantasy CLI)**
```bash
# Install
pip install yahoofantasy

# Login (opens browser)
yahoofantasy login

# Tokens saved to ~/.yahoofantasy/token.json
```

**Step 3: Environment Variables**
```bash
export YAHOO_CLIENT_ID="your_consumer_key"
export YAHOO_CLIENT_SECRET="your_consumer_secret"
```

### 1.6 Rate Limits & Best Practices

**Rate Limits:**
- Yahoo doesn't publish explicit limits
- Conservative approach: 1 request per second
- Use caching for static data (player info, league settings)

**Best Practices:**
1. **Cache aggressively:** Player data changes rarely; cache for 1+ hours
2. **Batch requests:** Use collections (e.g., `/players;player_keys=key1,key2`)
3. **Refresh tokens:** Tokens expire; handle 401 errors with refresh
4. **Error handling:** Yahoo API can be flaky; implement retries with backoff

### 1.7 Data Available from API

**Player Data:**
- Name, team, positions
- Season stats (all categories)
- Weekly stats
- Percent owned, percent drafted, percent started
- Player notes (injuries, news)

**League Data:**
- Settings (scoring, roster, schedule)
- Standings
- Scoreboard
- Transactions
- Draft results

**Team Data:**
- Roster (current and historical weeks)
- Stats (cumulative and weekly)
- Matchups
- Draft picks

---

## Part 2: Successful Fantasy Baseball Apps Analysis

### 2.1 Market Leaders

| Tool | Company | Est. Users | Price | Key Differentiator |
|------|---------|------------|-------|-------------------|
| **MLB Draft Wizard** | FantasyPros | 500K+ | $3-12/mo | Expert Consensus Rankings (ECR) |
| **RotoLab** | RotoLab | 10K+ | $40/year | BaseballHQ projections, 24 years |
| **RotoWire Draft Assistant** | RotoWire | 100K+ | $8-15/mo | Real-time sync, customization |
| **Fantrax** | Fantrax | 50K+ | Free/$130 | Deep customization, dynasty focus |
| **FantasyPros My Playbook** | FantasyPros | 1M+ | $3-12/mo | Multi-platform sync |

### 2.2 FantasyPros MLB Draft Wizard

**Overview:** The industry standard for draft preparation.

**Key Features:**

1. **Expert Consensus Rankings (ECR)**
   - Aggregates 50+ fantasy baseball experts
   - Weighted by historical accuracy
   - Updates continuously

2. **Draft Simulator**
   - AI opponents draft based on ECR/ADP
   - Customizable league settings
   - Rapid mock drafts (5-10 minutes)
   - Practice against different draft strategies

3. **Draft Assistant**
   - Real-time pick recommendations
   - Accounts for:
     - Team needs (position scarcity)
     - Opponent roster construction
     - ADP vs. Rank disparity
     - Category balance
   - Integrates with Yahoo/ESPN/CBS live drafts

4. **Cheat Sheet Creator**
   - Customizable rankings
   - Tier-based organization
   - Position-specific views
   - Export to PDF/print

5. **Draft Analyzer**
   - Post-draft grades
   - Identifies reaches and values
   - Projects team strengths/weaknesses
   - Compares to optimal draft

6. **Salary Cap Tools**
   - Auction draft simulator
   - Budget allocation calculator
   - Value-based pricing

**Technical Architecture Insight:**
- Real-time ADP tracking across thousands of drafts
- Machine learning for opponent behavior prediction
- Multi-source projection aggregation

**What Makes It Elite:**
- **ECR accuracy:** Users trust the consensus more than any single expert
- **Integration depth:** Syncs directly to live drafts
- **Speed:** Rapid mock drafts allow many practice scenarios

### 2.3 RotoLab

**Overview:** Desktop application (Windows) for serious players. 24 years in business.

**Key Features:**

1. **BaseballHQ.com Projections**
   - Exclusive partnership with BaseballHQ
   - Sabermetric-focused projections
   - Forecaster notes included

2. **Valuation Flexibility**
   - Three methods: Z-Scores, SGP (Standings Gain Points), PVM
   - Category weight adjustments
   - Position adjustments
   - "Stars and Scrubs" vs. "Spread Risk" slider

3. **Draft Day Interface**
   - Players by Position view
   - Drag-and-drop roster management
   - Live budget tracking (auctions)
   - Inflation calculator
   - Draft log

4. **Player Editor**
   - Edit projections manually
   - Auto-import BaseballHQ updates
   - Flag players (sleepers, busts, etc.)
   - Custom notes

5. **Reports**
   - Customizable draft lists
   - Position-specific cheat sheets
   - Rookie lists for dynasty
   - Export to Excel

**What Makes It Elite:**
- **Projection quality:** BaseballHQ is considered gold standard for sabermetric analysis
- **Flexibility:** Deep customization for different league types
- **Auction support:** Best-in-class inflation calculation

### 2.4 RotoWire Draft Assistant

**Overview:** Browser-based tool with real-time league sync.

**Key Features:**

1. **Live Sync**
   - Connects to Yahoo/ESPN/CBS drafts
   - Auto-marks taken players
   - Real-time recommendations

2. **Customization**
   - Import custom rankings
   - Set scoring rules
   - Adjust position priorities
   - Create player tiers

3. **Player Comparison**
   - Side-by-side stat comparison
   - Category contribution analysis
   - Schedule analysis

4. **Waiver Wire Tools**
   - Top available by position
   - Waiver assistant
   - Add/drop recommendations

**What Makes It Elite:**
- **Real-time sync:** No manual entry during draft
- **Customization depth:** Works with any ranking system

### 2.5 Emerging/New Tools

| Tool | Innovation | Lesson |
|------|------------|--------|
| **Dabble** | Social DFS (copy picks) | Community features drive engagement |
| **Sleeper** | Modern UI, mobile-first | Design matters for younger users |
| **PrizePicks** | Simplified pick'em | Reducing complexity opens market |

---

## Part 3: What Makes an Elite Fantasy Baseball Tool

### 3.1 The Success Formula

Based on analysis of successful tools, the winning formula is:

```
ELITE TOOL = Accurate Projections × Customization × Usability × Integration
```

### 3.2 Core Requirements

#### 1. Accurate, Multi-Source Projections

**What Users Expect:**
- Multiple projection sources (Steamer, ZiPS, ATC, THE BAT)
- Historical accuracy tracking
- Confidence intervals or ranges
- Manual override capability

**Implementation:**
```python
class ProjectionEngine:
    """
    Aggregate multiple projection sources with weighting.
    """
    SOURCES = {
        'steamer': {'weight': 0.30, 'reliability': 0.85},
        'zips': {'weight': 0.25, 'reliability': 0.82},
        'atc': {'weight': 0.25, 'reliability': 0.88},
        'the_bat': {'weight': 0.20, 'reliability': 0.80},
    }
    
    def get_composite_projection(self, player_id):
        projections = {}
        for source, config in self.SOURCES.items():
            proj = self.fetch_source(source, player_id)
            projections[source] = {
                'data': proj,
                'weight': config['weight']
            }
        
        # Weighted average
        composite = self.weighted_average(projections)
        
        # Confidence interval
        std_dev = self.calculate_variance(projections)
        
        return {
            'mean': composite,
            'ci_low': composite - (1.96 * std_dev),
            'ci_high': composite + (1.96 * std_dev),
            'sources': projections
        }
```

#### 2. Dynamic Valuation Engine

**What Users Expect:**
- Z-Score or SGP calculations
- Position scarcity adjustments
- Category balance tracking
- Real-time value updates during draft

**Implementation:**
```python
class ValuationEngine:
    """
    Calculate player values based on league context.
    """
    def calculate_z_scores(self, players, scoring_categories):
        """
        Calculate Z-scores for each scoring category.
        """
        means = {}
        stds = {}
        
        for cat in scoring_categories:
            values = [p.projections[cat] for p in players]
            means[cat] = np.mean(values)
            stds[cat] = np.std(values)
        
        for player in players:
            player.z_scores = {}
            for cat in scoring_categories:
                player.z_scores[cat] = (
                    player.projections[cat] - means[cat]
                ) / stds[cat]
            
            player.total_z = sum(player.z_scores.values())
        
        return players
    
    def adjust_for_position_scarcity(self, players, roster_requirements):
        """
        Boost values for scarce positions (C, SS, 2B).
        """
        position_counts = defaultdict(int)
        for p in players:
            position_counts[p.primary_position] += 1
        
        # Calculate replacement level for each position
        replacement_levels = {}
        for pos, count in roster_requirements.items():
            pos_players = [p for p in players if pos in p.positions]
            pos_players.sort(key=lambda x: x.total_z, reverse=True)
            
            # Replacement level = best player not drafted
            replacement_level_idx = count * 12  # 12 team league
            if replacement_level_idx < len(pos_players):
                replacement_levels[pos] = pos_players[replacement_level_idx].total_z
            else:
                replacement_levels[pos] = -999
        
        # Adjust player values (VORP - Value Over Replacement Player)
        for player in players:
            best_pos_z = max(
                player.z_scores.get(pos, -999) for pos in player.positions
            )
            player.vorp = player.total_z - replacement_levels.get(
                player.primary_position, 0
            )
        
        return players
```

#### 3. Intelligent Draft Assistant

**What Users Expect:**
- Real-time pick recommendations
- Team need awareness
- Opponent roster tracking
- Reach/value detection

**Implementation:**
```python
class DraftAssistant:
    """
    Provide intelligent recommendations during draft.
    """
    def get_recommendations(self, my_team, available_players, round_num, pick_num):
        """
        Return ranked list of recommended picks.
        """
        # Score each available player
        scored_players = []
        for player in available_players:
            score = self.score_player(player, my_team, round_num)
            scored_players.append((player, score))
        
        # Sort by score
        scored_players.sort(key=lambda x: x[1], reverse=True)
        
        return scored_players[:10]  # Top 10 recommendations
    
    def score_player(self, player, my_team, round_num):
        """
        Calculate a recommendation score for a player.
        """
        score = 0
        
        # Base value (VORP)
        score += player.vorp * 100
        
        # Positional need bonus
        needs = self.get_team_needs(my_team)
        if player.primary_position in needs['critical']:
            score += 50  # Critical need
        elif player.primary_position in needs['moderate']:
            score += 25  # Moderate need
        elif player.primary_position in needs['filled']:
            score -= 30  # Already have enough
        
        # ADP-based value detection
        adp_diff = player.adp - pick_num
        if adp_diff > 15:
            score += 20  # Significant value
        elif adp_diff > 5:
            score += 10  # Moderate value
        elif adp_diff < -10:
            score -= 15  # Reaching
        
        # Category balance adjustment
        cat_balance = self.get_category_balance(my_team)
        for cat, z in player.z_scores.items():
            if cat_balance[cat] < -1 and z > 0:
                score += 10  # Need this category
            elif cat_balance[cat] > 2 and z > 0:
                score -= 5   # Already strong here
        
        # Pitcher/hitter balance
        roster_balance = self.get_roster_balance(my_team)
        if player.is_pitcher and roster_balance['pitchers'] < 0.4:
            score += 15
        elif not player.is_pitcher and roster_balance['hitters'] < 0.6:
            score += 15
        
        return score
    
    def get_team_needs(self, my_team):
        """
        Analyze roster and return position needs.
        """
        filled = defaultdict(int)
        for player in my_team.roster:
            filled[player.selected_position] += 1
        
        needs = {'critical': [], 'moderate': [], 'filled': []}
        
        for pos, required in LEAGUE_ROSTER_REQUIREMENTS.items():
            if filled[pos] < required * 0.5:
                needs['critical'].append(pos)
            elif filled[pos] < required * 0.8:
                needs['moderate'].append(pos)
            else:
                needs['filled'].append(pos)
        
        return needs
```

#### 4. Seamless Platform Integration

**What Users Expect:**
- One-click league import
- Real-time roster sync
- Automatic lineup optimization
- Waiver wire recommendations

**Yahoo API Integration:**
```python
class YahooLeagueSync:
    """
    Sync with Yahoo Fantasy Baseball leagues.
    """
    def __init__(self, access_token):
        self.client = yahoofantasy.Context()
        self.client.token = access_token
    
    def import_league(self, league_key):
        """
        Import league settings and rosters.
        """
        league = self.client.get_league(league_key)
        
        return {
            'name': league.name,
            'scoring_type': league.settings['scoring_type'],
            'roster_positions': league.settings['roster_positions'],
            'teams': [
                {
                    'team_key': team.team_key,
                    'name': team.name,
                    'manager': team.manager.nickname,
                    'roster': self.import_roster(team)
                }
                for team in league.teams()
            ]
        }
    
    def import_roster(self, team, week=None):
        """
        Import team's roster.
        """
        players = team.players()
        return [
            {
                'player_key': player.player_key,
                'name': player.name.full,
                'positions': player.display_position.split(','),
                'selected_position': player.selected_position.position,
                'stats': player.player_stats
            }
            for player in players
        ]
    
    def set_lineup(self, team_key, lineup, week):
        """
        Set active lineup for a week.
        """
        team = self.client.get_team(team_key)
        
        # Build roster XML
        roster_xml = self.build_roster_xml(lineup, week)
        
        # PUT to Yahoo API
        return team.set_roster(roster_xml)
    
    def get_waiver_wire(self, league_key, position=None, sort='AR'):
        """
        Get available free agents.
        """
        league = self.client.get_league(league_key)
        
        free_agents = []
        start = 0
        while True:
            batch = league.free_agents(
                position=position or 'B',
                start=start
            )
            if not batch:
                break
            free_agents.extend(batch)
            start += 25
            if start > 100:  # Limit to first 100
                break
        
        return free_agents
```

### 3.3 Differentiation Strategies

To create an **elite** tool, consider these differentiation angles:

#### 1. Superior Projections

**Opportunity:** Most tools use the same public projections (Steamer, ZiPS). Build proprietary models.

**Approach:**
- Integrate Statcast data (exit velocity, launch angle)
- Use machine learning for breakout/bust prediction
- Weather and park factor adjustments
- Injury risk modeling

#### 2. Real-Time Adaptation

**Opportunity:** Most tools are static. Build real-time adjustment capabilities.

**Approach:**
- Spring training performance integration
- Lineup change alerts
- Weather impact on games
- Bullpen usage tracking

#### 3. Advanced Analytics

**Opportunity:** Most tools use basic stats. Offer deeper analytics.

**Approach:**
- xwOBA, Barrel%, HardHit% integration
- Platoon splits analysis
- Rolling window performance
- Closer probability models

#### 4. Community/Social Features

**Opportunity:** Fantasy is social. Most tools are isolated.

**Approach:**
- League-specific chat
- Pick analysis sharing
- Expert pick tracking
- Crowdsourced rankings

#### 5. Multi-Sport Synergy

**Opportunity:** Your CBB expertise can transfer to MLB.

**Approach:**
- Similar analytical framework (KenPom → advanced stats)
- Fatigue/rest models (back-to-backs → doubleheaders)
- Player efficiency comparisons

---

## Part 4: Recommended Architecture for CBB Edge Fantasy Baseball Tool

### 4.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CBB EDGE FANTASY BASEBALL TOOL                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        DATA LAYER                                    │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │   Steamer   │  │    ZiPS     │  │  Statcast   │  │   Yahoo    │  │   │
│  │  │ Projections │  │ Projections │  │    API      │  │    API     │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  │                                                                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │   │
│  │  │ FanGraphs   │  │ Baseball    │  │   Roster    │                   │   │
│  │  │    API      │  │  Reference  │  │  Resource   │                   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   PROJECTION ENGINE                                  │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │  Composite Projection Model                                    │  │   │
│  │  │  • Weighted average across sources                            │  │   │
│  │  │  • Confidence intervals                                       │  │   │
│  │  │  • Manual override capability                                 │  │   │
│  │  │  • Update tracking                                            │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   VALUATION ENGINE                                   │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │   │
│  │  │   Z-Score Calc   │  │  Position Scarcity│  │   Category      │   │   │
│  │  │                  │  │    Adjustment     │  │    Balance      │   │   │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    DRAFT ASSISTANT                                   │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │   │
│  │  │  Recommendation  │  │   Opponent       │  │   Real-Time      │   │   │
│  │  │     Engine       │  │   Tracking       │  │     Sync         │   │   │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    USER INTERFACE                                    │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │   Web App   │  │ Discord Bot │  │  Mobile PWA │  │   API      │  │   │
│  │  │  (FastAPI)  │  │   (Cogs)    │  │  (React)    │  │  (REST)    │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Database Schema Additions

```sql
-- Fantasy Baseball specific tables

CREATE TABLE mlb_players (
    id SERIAL PRIMARY KEY,
    yahoo_player_key VARCHAR(50) UNIQUE,
    fangraphs_id VARCHAR(20),
    mlbam_id VARCHAR(20),
    name VARCHAR(100) NOT NULL,
    team VARCHAR(10),
    primary_position VARCHAR(10),
    eligible_positions VARCHAR(50)[],  -- Array of positions
    bats VARCHAR(1),  -- L, R, S
    throws VARCHAR(1),  -- L, R
    birthdate DATE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE mlb_projections (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES mlb_players(id),
    source VARCHAR(20),  -- 'steamer', 'zips', 'atc', 'composite'
    season INTEGER,
    
    -- Hitting projections
    projected_g INTEGER,
    projected_pa INTEGER,
    projected_ab INTEGER,
    projected_h INTEGER,
    projected_2b INTEGER,
    projected_3b INTEGER,
    projected_hr INTEGER,
    projected_r INTEGER,
    projected_rbi INTEGER,
    projected_sb INTEGER,
    projected_cs INTEGER,
    projected_bb INTEGER,
    projected_so INTEGER,
    projected_avg DECIMAL(4,3),
    projected_obp DECIMAL(4,3),
    projected_slg DECIMAL(4,3),
    projected_ops DECIMAL(4,3),
    
    -- Pitching projections
    projected_gs INTEGER,
    projected_ip DECIMAL(5,1),
    projected_w INTEGER,
    projected_l INTEGER,
    projected_era DECIMAL(4,2),
    projected_whip DECIMAL(4,2),
    projected_k INTEGER,
    projected_bb INTEGER,
    projected_sv INTEGER,
    projected_hld INTEGER,
    
    -- Metadata
    confidence_score DECIMAL(3,2),  -- 0.00 to 1.00
    last_updated TIMESTAMP DEFAULT NOW(),
    UNIQUE(player_id, source, season)
);

CREATE TABLE fantasy_leagues (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    platform VARCHAR(20),  -- 'yahoo', 'espn', 'cbs'
    external_league_key VARCHAR(50),
    league_name VARCHAR(100),
    league_type VARCHAR(20),  -- 'public', 'private'
    scoring_type VARCHAR(20),  -- 'headtohead', 'rotisserie', 'points'
    
    -- League settings
    num_teams INTEGER,
    roster_c INTEGER DEFAULT 1,
    roster_1b INTEGER DEFAULT 1,
    roster_2b INTEGER DEFAULT 1,
    roster_3b INTEGER DEFAULT 1,
    roster_ss INTEGER DEFAULT 1,
    roster_of INTEGER DEFAULT 3,
    roster_util INTEGER DEFAULT 1,
    roster_sp INTEGER DEFAULT 2,
    roster_rp INTEGER DEFAULT 2,
    roster_p INTEGER DEFAULT 3,
    roster_bn INTEGER DEFAULT 5,
    roster_il INTEGER DEFAULT 2,
    
    -- Scoring categories (JSON for flexibility)
    hitting_categories JSON,
    pitching_categories JSON,
    
    last_synced TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, platform, external_league_key)
);

CREATE TABLE fantasy_teams (
    id SERIAL PRIMARY KEY,
    league_id INTEGER REFERENCES fantasy_leagues(id),
    external_team_key VARCHAR(50),
    team_name VARCHAR(100),
    manager_name VARCHAR(100),
    is_my_team BOOLEAN DEFAULT FALSE,
    waiver_priority INTEGER,
    faab_balance INTEGER,
    
    -- Auto-sync enabled
    auto_sync BOOLEAN DEFAULT TRUE
);

CREATE TABLE fantasy_rosters (
    id SERIAL PRIMARY KEY,
    team_id INTEGER REFERENCES fantasy_teams(id),
    player_id INTEGER REFERENCES mlb_players(id),
    week INTEGER,
    selected_position VARCHAR(10),  -- 'C', '1B', 'BN', 'IL', etc.
    is_editable BOOLEAN DEFAULT TRUE,
    last_updated TIMESTAMP DEFAULT NOW(),
    UNIQUE(team_id, player_id, week)
);

CREATE TABLE draft_sessions (
    id SERIAL PRIMARY KEY,
    league_id INTEGER REFERENCES fantasy_leagues(id),
    session_type VARCHAR(20),  -- 'mock', 'live', 'auction'
    status VARCHAR(20),  -- 'active', 'completed', 'abandoned'
    current_round INTEGER DEFAULT 1,
    current_pick INTEGER DEFAULT 1,
    total_rounds INTEGER,
    my_draft_position INTEGER,
    
    -- Draft configuration
    scoring_settings JSON,
    roster_requirements JSON,
    
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

CREATE TABLE draft_picks (
    id SERIAL PRIMARY KEY,
    draft_session_id INTEGER REFERENCES draft_sessions(id),
    pick_number INTEGER,
    round INTEGER,
    team_id INTEGER REFERENCES fantasy_teams(id),
    player_id INTEGER REFERENCES mlb_players(id),
    pick_time TIMESTAMP,
    is_auto_pick BOOLEAN DEFAULT FALSE,
    recommended_by VARCHAR(20),  -- 'model', 'user', 'ecr'
    confidence_score DECIMAL(3,2)
);

CREATE TABLE player_tags (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES mlb_players(id),
    user_id INTEGER REFERENCES users(id),
    tag VARCHAR(20),  -- 'sleeper', 'bust', 'target', 'avoid', 'injury_risk'
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 4.3 Implementation Roadmap

#### Phase 1: Foundation (Weeks 1-2)
- [ ] Yahoo OAuth integration
- [ ] League import functionality
- [ ] Player database setup (Fangraphs IDs, positions)
- [ ] Basic projection import (Steamer)

#### Phase 2: Core Engine (Weeks 3-4)
- [ ] Z-score valuation engine
- [ ] Position scarcity adjustments
- [ ] Draft assistant algorithm
- [ ] ADP integration

#### Phase 3: Draft Tools (Weeks 5-6)
- [ ] Draft simulator (AI opponents)
- [ ] Real-time draft sync
- [ ] Recommendation engine
- [ ] Post-draft analyzer

#### Phase 4: In-Season (Weeks 7-8)
- [ ] Lineup optimizer
- [ ] Waiver wire recommendations
- [ ] Weekly matchup analysis
- [ ] Trade evaluator

#### Phase 5: Advanced Features (Ongoing)
- [ ] Statcast integration
- [ ] Custom projection models
- [ ] Community features
- [ ] Mobile app

---

## Part 5: Key Recommendations

### 5.1 Technical Recommendations

1. **Use yahoofantasy Python library** - Most maintained, easiest OAuth handling
2. **Cache aggressively** - Yahoo API can be slow; cache player data for 6+ hours
3. **Build async data pipeline** - Multiple projection sources require parallel fetching
4. **Implement token refresh** - OAuth tokens expire; handle gracefully
5. **Create fallback data sources** - If Yahoo API is down, use cached data

### 5.2 Product Recommendations

1. **Focus on draft assistant first** - Highest user value, easiest to demonstrate
2. **Differentiate with CBB-style analytics** - Apply your fatigue/rest models to MLB
3. **Build for H2H categories first** - Most popular format
4. **Integrate Statcast data** - Modern tool requirement (exit velocity, barrel%)
5. **Offer manual projection overrides** - Serious players want control

### 5.3 Business Recommendations

1. **Freemium model** - Free basic tools, premium for advanced features
2. **Partner with projection sources** - Negotiate access to Steamer/ZiPS
3. **Build community first** - Discord integration drives engagement
4. **Track accuracy publicly** - Build trust by showing projection performance
5. **Consider white-label** - Offer tool to content sites (BaseballHQ model)

---

## Appendices

### A. Yahoo API Code Examples

#### Complete OAuth Flow
```python
import os
from yahoofantasy import Context

# Set credentials
os.environ['YAHOO_CLIENT_ID'] = 'your_client_id'
os.environ['YAHOO_CLIENT_SECRET'] = 'your_client_secret'

# First time: login via browser
# yahoofantasy login

# In application
ctx = Context()

# Get leagues
leagues = ctx.get_leagues('mlb', 2026)
```

#### Get League Settings
```python
league = leagues[0]
settings = league.settings

print(f"Scoring: {settings['scoring_type']}")
print(f"Teams: {settings['num_teams']}")
print(f"Roster spots: {settings['roster_positions']}")
```

#### Get Free Agents
```python
# Get all available batters
batters = league.free_agents('B')

# Sort by percent owned
batters.sort(key=lambda x: x.percent_owned.value, reverse=True)

for b in batters[:10]:
    print(f"{b.name.full} - {b.percent_owned.value}% owned")
```

### B. Projection Source URLs

| Source | URL | Cost |
|--------|-----|------|
| Steamer | https://www.fangraphs.com/projections.aspx | Free |
| ZiPS | https://www.fangraphs.com/projections.aspx | Free |
| ATC | https://www.fangraphs.com/projections.aspx | Free |
| THE BAT | https://www.fangraphs.com/projections.aspx | Free |
| BaseballHQ | https://www.baseballhq.com | $99/year |
| RotoWire | https://www.rotowire.com | $8-15/month |

### C. Competitor Pricing Comparison

| Tool | Free Tier | Premium | Annual |
|------|-----------|---------|--------|
| FantasyPros | Limited | $3.99/mo | $35.88 |
| RotoWire | None | $8.99/mo | $99 |
| RotoLab | Demo | $40/year | $40 |
| Fantrax | Full | N/A | N/A |
| BaseballHQ | None | $99/year | $99 |

---

*Document Version: 1.0*  
*Prepared for: CBB Edge Analyzer Team*  
*Research Date: March 24, 2026*
