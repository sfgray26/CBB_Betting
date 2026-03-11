# MLB Fantasy Baseball — Full Application Roadmap

**League:** Treemendous · Yahoo ID 72586 · 12-team H2H One Win · 18 categories · Keeper League
**Draft:** March 23, 2026 @ 7:30am EDT · Snake · 90-sec clock
**Season:** March 27 – September 28, 2026

---

## Architecture Principle

The fantasy app is **intelligence-first**: every decision (keeper, draft pick, waiver, lineup, trade)
is backed by projection data + sportsbook market signals. The sportsbook odds layer gives us
information no pure fantasy platform provides — what the market expects from a game environment today.

---

## Phase 0 — Foundation (COMPLETE)

| Component | Status | File |
|-----------|--------|------|
| Yahoo OAuth client | Done | `yahoo_client.py` |
| Projections loader (Steamer CSV) | Done | `projections_loader.py` |
| Keeper engine (z-score surplus) | Done | `keeper_engine.py` |
| Draft engine scaffold | Done | `draft_engine.py` |
| Dashboard (11_Fantasy_Baseball.py) | Done | 6 tabs |
| Player board + z-scores | Done | `player_board.py` |

---

## Phase 1 — Auth + Pre-Draft (Mar 10–22) — ACTIVE NOW

### 1.1 Yahoo OAuth Fix (DONE)
- `_store_tokens()` is now Railway-safe (best-effort `.env` write, no crash)
- Dashboard Setup tab has full in-browser OAuth flow
- **User action required:** Complete OAuth locally → copy `YAHOO_REFRESH_TOKEN` to Railway

### 1.2 Draft Board Assistant (Mar 22–23)
**Priority: CRITICAL (2 days away)**

What it needs to do:
- Display real-time pick counter (pick N of 276)
- Show top available players ranked by z-score
- Filter by position need (track roster composition)
- Recommended pick with reasoning
- Color-code: must-have / good value / skip

Files to build:
- `backend/fantasy_baseball/draft_engine.py` — already scaffolded, needs logic
- Dashboard `Draft Board` tab — currently placeholder

**Draft strategy encoded:**
- R1-R3: Elite batters (Acuna, Soto, Alvarez, Judge tier)
- R4-R6: SP aces (Cole, Skubal, Wheeler, Sale tier)
- R7-R9: 2B/SS premium (mixed hitting value)
- R10-R14: SP depth, CI bats
- R15-R20: Closers (target 2-3, NSV is a category)
- R21-23: Upside flyers

**Odds integration:** Use sportsbook data to adjust projections for home/away context
(e.g., Rockies hitters get COL park factor boost even for away games at Coors)

### 1.3 Projection Data Refresh
Current data: hardcoded 50-80 player stubs
**Needed:** Full Steamer 2026 download (~750 batters, ~450 pitchers)

**Source:** FanGraphs Steamer projections
- URL: https://www.fangraphs.com/projections.aspx?pos=all&stats=bat&type=steamer
- Format: Export as CSV → drop in `data/projections/steamer_batting_2026.csv`
- Do same for pitching

**Gemini research task:** Find best free/bulk download method for 2026 Steamer data

---

## Phase 2 — Season Operations (Mar 27 – Sep 28)

### 2.1 Daily Lineup Optimizer (BUILT)
**File:** `backend/fantasy_baseball/daily_lineup_optimizer.py`

**How it works:**
1. Fetch MLB game odds from The Odds API (same key used for CBB)
2. Convert game total + spread → team implied runs
   - Formula: `home_implied = (total + spread_home) / 2`
3. Rank batters by team implied runs × park factor × projected stats
4. Rank SP streamers by opponent implied runs (lower = better)
5. Display daily dashboard with:
   - Best matchups today (high-total games = stack candidates)
   - Batters to start vs. bench (by implied run environment)
   - Streaming SP if any on waiver wire

**Still needed for Phase 2:**
- Dashboard tab: "Daily Lineup" (add as 7th tab)
- Integration with yahoo_client.get_lineup() to show current lineup vs. recommended
- "Apply Recommendations" button → calls set_lineup() via Yahoo API
- Weather layer (scrape weather.com or use OpenWeatherMap API for game-time conditions)
- Starting pitcher confirmation (pull from MLB API or RotoBaller)

**Odds-to-fantasy insights (key formulas):**
| Signal | Formula | Fantasy Use |
|--------|---------|-------------|
| Team implied runs | `(total + spread_home) / 2` | Stack batters from high-run teams |
| Pitcher opponent | Opponent's implied runs | Stream SP facing low-offense team |
| Strikeout environment | Use K prop lines when available | Prioritize high-K SPs on days you need Ks |
| Game total over 9.5 | Raw total > 9.5 | Flag as "stack game" |
| Close game (spread < 1) | Both teams live | Both lineups likely to see all 9 batters |

### 2.2 Waiver Wire Prioritization

**Category deficit tracker:**
- Pull weekly matchup from Yahoo scoreboard
- Show my team's category totals vs. opponent
- Identify which categories are deficits (at risk of losing)
- Rank waiver pickups by: (1) fills deficit categories, (2) available SP starts this week

**Streaming pitcher scheduler:**
- Given 7-day schedule, identify SPs with 2+ quality starts
- Quality start criteria: ERA < 4.0, opp implied runs < 4.2, road vs. weak team

**Priority queue logic:**
1. Fill injured player (auto-replace with same position)
2. Deficit closer (NSV is crucial — always monitor closer situations)
3. Hot hitter (last 7-day performance boost)
4. Two-start pitcher (two starts in a week = 2× production)

**Files to build:**
- `backend/fantasy_baseball/waiver_engine.py`
- `backend/fantasy_baseball/category_tracker.py`

### 2.3 Trade Analyzer

**Net category delta:**
For each proposed trade:
- Player giving: remove their ROS projections from your totals
- Player receiving: add their ROS projections to your totals
- Show: net change per category (+ or -)
- Label: "Win-now" (total categories gained) vs "rebuild" (future value)

**Roster depth view:**
- Show category standings before and after trade
- Rank among 12 teams for each category (position in standings)

**Files to build:**
- `backend/fantasy_baseball/trade_analyzer.py`

---

## Phase 3 — Advanced Intelligence (In-Season)

### 3.1 Statcast Integration
**File:** `backend/fantasy_baseball/statcast_scraper.py` (scaffolded)

Key metrics for lineup decisions:
- **xBA / xwOBA**: True talent signal vs. BABIP luck
- **Exit velocity / barrel rate**: Power indicator (HR regression signal)
- **Sprint speed**: Base-stealing readiness
- **Spin rate / pitch mix**: SP effectiveness vs. specific lineups

**Source:** Baseball Savant (baseballsavant.mlb.com)
- Free public API: `https://baseballsavant.mlb.com/statcast_search/csv?...`

### 3.2 Closer Situations Monitor
**File:** `backend/fantasy_baseball/closer_situations.py`

Critical because NSV is a league category. Track:
- Team's primary closer (name + handedness)
- Save opportunities per week (team's projected wins × blown save rate)
- "Next in line" when closer loses role
- Injury/demotion alerts

**Monitoring:** Scrape beat reporters or use Twitter/Bluesky for real-time news
**Alternative:** The Athletic, Rotowire, Baseball Reference

### 3.3 Injury Intelligence
**Current:** `data/projections/injury_flags_2026.csv` (static, built on startup)
**Needed:** Live injury monitoring

Options:
- ESPN fantasy player status (same CBS scraper pattern as CBB injuries)
- Rotowire news feed
- MLB.com transaction log

### 3.4 Recalibration
After first 30 games:
- Compare actual stats to Steamer projections
- Apply regression factors (hot/cold starts → regress to mean)
- Update z-scores in keeper_engine.py baselines

---

## Phase 4 — Full Automation

### 4.1 Auto-Lineup Setting
- Morning job (7 AM ET daily): pull optimal lineup from optimizer
- Compare to current lineup
- If material improvement: call `yahoo_client.set_lineup()` automatically
- Alert Discord with changes made

**Risk guard:** Verify player is actually starting today before auto-setting
(Use starting lineup confirmations from MLB API or FantasyLabs)

### 4.2 Weekly Waiver Assistant
- Sunday night job: analyze current week's category standings
- Monday morning: suggest 1-3 adds for deficit filling
- Alert via Discord

### 4.3 Performance Dashboard
- Weekly: ROI vs. projected (are projections calibrated?)
- Category win rate tracking (which categories are you reliably winning?)
- Regression monitoring (which players are over/underperforming xStats?)

---

## Sportsbook Odds Integration — Detailed Design

### MLB Odds from The Odds API (same key as CBB)
```
GET /v4/sports/baseball_mlb/odds
params:
  markets: spreads,totals,h2h
  regions: us
  oddsFormat: american
  commenceTimeFrom: 2026-04-01T00:00:00Z
  commenceTimeTo: 2026-04-01T23:59:59Z
```

### Implied Runs Formula
```python
home_implied = (total + spread_home) / 2   # spread_home negative if favored
away_implied = total - home_implied
```

Example:
- NYY vs BOS, total 9.5, NYY -1.5
- NYY implied: (9.5 + (-1.5)) / 2 = 4.0
- BOS implied: 9.5 - 4.0 = 5.5
- Stack BOS batters, stream pitcher vs. NYY

### Park Factor Adjustments
Built into `daily_lineup_optimizer.py`:
- COL: 1.25x (Coors — always stack)
- GB pitchers in COL: massive penalty
- SEA, TB, SF: 0.92-0.93x (pitcher parks)

### SP Streaming Score (0-100)
```
env_score  = (5.5 - opp_implied_runs) / 2.0 * 10   # 0-10
k_score    = min(10, k9 - 5.0)                       # 0-10
park_score = (2.0 - park_factor) * 5                 # 0-5
stream     = env_score*0.5 + k_score*0.3 + park*0.2
```

---

## Database Schema (Fantasy Extension)

Add to `backend/models.py`:

```python
class FantasyPlayer(Base):
    __tablename__ = "fantasy_players"
    id = Column(Integer, primary_key=True)
    yahoo_player_key = Column(String, unique=True, index=True)
    name = Column(String, nullable=False)
    mlb_team = Column(String)
    positions = Column(ARRAY(String))
    player_type = Column(String)    # 'batter' or 'pitcher'
    # Season projections
    proj_r = Column(Float); proj_hr = Column(Float); proj_rbi = Column(Float)
    proj_avg = Column(Float); proj_ops = Column(Float); proj_nsb = Column(Float)
    proj_w = Column(Float); proj_era = Column(Float); proj_k9 = Column(Float)
    proj_whip = Column(Float); proj_qs = Column(Float); proj_nsv = Column(Float)
    z_score = Column(Float)
    adp = Column(Float)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class FantasyLineupDecision(Base):
    __tablename__ = "fantasy_lineup_decisions"
    id = Column(Integer, primary_key=True)
    decision_date = Column(Date, nullable=False, index=True)
    player_key = Column(String)
    player_name = Column(String)
    action = Column(String)         # 'start', 'bench', 'add', 'drop'
    reason = Column(String)
    lineup_score = Column(Float)
    implied_team_runs = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
```

---

## Gemini Research Tasks (Assigned)

See HANDOFF.md Section 10 for verbatim Gemini prompts.

1. **G-R1:** Steamer 2026 projection data — best download method, bulk export format
2. **G-R2:** MLB starting lineup confirmation sources (7 AM daily) — free APIs
3. **G-R3:** Closer situation tracker sources — best real-time news feeds
4. **G-R4:** Statcast API endpoints — bulk player data download
5. **G-R5:** Yahoo Fantasy API — lineup slot codes, transaction XML format

---

## Timeline Summary

| Date | Milestone |
|------|-----------|
| **Mar 20** | Keeper deadline — evaluator active |
| **Mar 22** | Draft board assistant live |
| **Mar 23** | Draft day 7:30am |
| **Mar 27** | Season opener — daily optimizer active |
| **Apr 1** | Waiver engine MVP live |
| **Apr 14** | Trade analyzer MVP live |
| **May 1** | Statcast integration + recalibration |
| **Jun 1** | Auto-lineup setting enabled |
| **Aug 6** | Trade deadline |
| **Sep 28** | Season end |
