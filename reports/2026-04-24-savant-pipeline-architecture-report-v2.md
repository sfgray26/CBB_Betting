# SAVANT PIPELINE ARCHITECTURE REPORT v2.0

**Date:** 2026-04-24  
**Researcher:** Kimi CLI (Deep Intelligence Unit)  
**Authority:** Proposed architecture — requires Claude Code approval before implementation  
**Scope:** Comprehensive reverse-engineering of ALL Baseball Savant data sources: Custom Leaderboard (187 columns), Park Factors, Probable Pitchers, Bat Tracking, and Exit Velocity & Barrels leaderboards.

---

## EXECUTIVE SUMMARY

Baseball Savant is a **massive, mostly-unlocked data goldmine**. We have reverse-engineered **five distinct data sources**, each with different extraction methods and fantasy baseball applications:

| Source | Players/Parks | Extraction Method | Fantasy Use |
|--------|--------------|-------------------|-------------|
| **Custom Leaderboard** | 442 batters + 505 pitchers | `&csv=true` (direct HTTP) | **PRIMARY** — 187 column options including all counting stats, expected stats, batted-ball metrics, plate discipline, and new bat-tracking data |
| **Exit Velocity & Barrels** | 280 batters | `&csv=true` (direct HTTP) | Power metrics, EV percentiles, barrel rates |
| **Park Factors** | 30 parks | Regex extract from HTML | Park-adjusted projections, daily lineup optimization |
| **Bat Tracking** | 218 batters | Regex extract from HTML | Bat speed, swing length, attack angle, squared-up rate |
| **Probable Pitchers** | Daily matchups | HTML scraping | Daily lineup confirmation, opponent quality |

**Critical finding:** The Custom Leaderboard with the Statcast checkbox enabled exposes **187 selectable columns** — far more than the ~15 we initially tested. This includes traditional counting stats (R, RBI, SB, W, L, QS), expected stats (xBA, xSLG, xOBP, xISO), batted-ball profiles (GB%, FB%, LD%, Pull%), fielding (OAA), and baserunning (Sprint Speed, Bolts).

---

## 1. CUSTOM LEADERBOARD — THE PRIMARY DATA SOURCE

### 1.1 Reverse-Engineering Verdict

**✅ Direct HTTP requests work perfectly.** Appending `&csv=true` returns `text/csv; charset=utf-8`. No auth, no Cloudflare, no tokens.

### 1.2 The Statcast Checkbox Discovery

The Custom Leaderboard has a **"Statcast" checkbox** (`id="chkStatcast"`) that unlocks an additional ~100 column options beyond the basic stats. When checked, the column selector dropdown shows:

- **Standard Stats** tab: Age, AB, PA, H, 1B, 2B, 3B, HR, SO, BB, K%, BB%, AVG, SLG, OBP, OPS, ISO, BABIP, RBI, LOB, TB, CS, SB, R, G, Sac Bunt, Sac Fly, etc.
- **Statcast Stats** tab: xBA, xSLG, xOBP, xISO, wOBAcon, xwOBAcon, BACON, xBACON, BA-xBA diff, SLG-xSLG diff, wOBA-xwOBA diff, Bat Speed, Fast Swing %, Swing Length, Blasts, Squared-Up, Swords, Attack Angle, EV, LA, Sweet-Spot%, Barrels, Hard Hit%, Zone metrics, Whiff%, Swing%, Pull/Oppo%, GB/FB/LD%, Popups, etc.

**Total: 187 selectable columns.**

### 1.3 Complete Batter Endpoint (Recommended Selections)

For maximum fantasy coverage, we recommend fetching these columns:

```python
BATTER_SELECTIONS = (
    # Counting stats
    "pa%2Cab%2Ch%2Csingle%2Cdouble%2Ctriple%2Chome_run"
    "%2Cstrikeout%2Cwalk%2Cr_run%2Cb_rbi%2Cr_total_stolen_base"
    "%2Cr_total_caught_stealing%2Cb_total_bases%2Cb_game"
    # Rate stats
    "%2Ck_percent%2Cbb_percent%2Cbatting_avg%2Cslg_percent"
    "%2Con_base_percent%2Con_base_plus_slg%2Cisolated_power%2Cbabip"
    # Expected stats
    "%2Cxba%2Cxslg%2Cxwoba%2Cxobp%2Cxiso%2Cxwobacon%2Cxbacon"
    # Batted ball quality
    "%2Cexit_velocity_avg%2Claunch_angle_avg%2Csweet_spot_percent"
    "%2Cbarrel_batted_rate%2Chard_hit_percent%2Cavg_best_speed%2Cavg_hyper_speed"
    # Plate discipline
    "%2Cwhiff_percent%2Cswing_percent%2Cz_swing_percent%2Coz_swing_percent"
    "%2Ciz_contact_percent%2Coz_contact_percent"
    # Batted ball profile
    "%2Cgroundballs_percent%2Cflyballs_percent%2Clinedrives_percent%2Cpopups_percent"
    "%2Cpull_percent%2Cstraightaway_percent%2Copposite_percent"
    # Baserunning
    "%2Csprint_speed%2Cn_bolts%2Chp_to_1b"
    # Fielding
    "%2Cn_outs_above_average"
    # New bat tracking (2024+)
    "%2Cavg_swing_speed%2Cfast_swing_rate%2Cavg_swing_length"
    "%2Cblasts_contact%2Cblasts_swing%2Csquared_up_contact%2Csquared_up_swing"
    "%2Cswords%2Cattack_angle"
)
```

**Verified URL:**
```
https://baseballsavant.mlb.com/leaderboard/custom
?year=2026&type=batter&filter=&min=0
&selections={BATTER_SELECTIONS}
&chart=false&x=pa&y=pa&r=no&chartType=beeswarm
&sort=xwoba&sortDir=desc&csv=true
```

### 1.4 Complete Pitcher Endpoint (Recommended Selections)

```python
PITCHER_SELECTIONS = (
    # Counting stats (allowed by pitcher)
    "pa%2Cip%2Cw%2Cl%2Cqs"
    "%2Cer%2Ck_pit%2Cbb_pit"  # Note: verify exact column names
    # Rate stats (allowed)
    "%2Ck_percent%2Cbb_percent"
    # Traditional stats
    "%2Cera%2Cwhip%2Ck_9"
    # Expected stats (allowed)
    "%2Cwoba%2Cxwoba%2Cxera"
    # Batted ball quality allowed
    "%2Cbarrel_batted_rate%2Chard_hit_percent%2Cexit_velocity_avg"
    # Plate discipline (induced)
    "%2Cwhiff_percent%2Cswing_percent"
    # Batted ball profile allowed
    "%2Cgroundballs_percent%2Cflyballs_percent%2Clinedrives_percent"
)
```

**Note:** The exact column names for pitcher-specific counting stats (ER, K_pit, BB_pit) may differ. We verified `era`, `whip`, `k_9`, `ip`, `w`, `l`, `qs` work. Additional testing needed for `er`, `k_pit`, `bb_pit`.

### 1.5 CSV Parsing Notes

The CSV uses a combined name column: `"last_name, first_name"` (single quoted field). Python's `csv.DictReader` handles this, but the resulting key includes quotes.

Many numeric stats use **leading-dot notation** (`.551` for wOBA). Use this parser:

```python
def savant_float(val: Optional[str]) -> float:
    if not val:
        return 0.0
    val = val.strip()
    if val.startswith("."):
        val = "0" + val
    try:
        return float(val)
    except ValueError:
        return 0.0
```

---

## 2. PARK FACTORS LEADERBOARD

### 2.1 Endpoint

```
https://baseballsavant.mlb.com/leaderboard/statcast-park-factors
```

**❌ Does NOT support `&csv=true`** — returns HTML page.

**✅ Data is embedded** as `var data = [...]` in the page HTML.

### 2.2 Extraction Method

```python
import requests
import re
import json

def fetch_park_factors() -> list[dict]:
    url = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"
    resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
    resp.raise_for_status()
    
    match = re.search(r'var data = (\[.*?\]);', resp.text, re.DOTALL)
    if not match:
        raise ValueError("Could not find park factor data in HTML")
    
    return json.loads(match.group(1))
```

### 2.3 Available Metrics (28 Fields)

| Field | Description | Fantasy Application |
|-------|-------------|---------------------|
| `venue_id` | Venue ID | Join key |
| `venue_name` | Stadium name | Display |
| `main_team_id` | Team ID | Join to team |
| `name_display_club` | Team name | Display |
| `n_pa` | Sample size (PA) | Confidence weight |
| `index_runs` | Runs park factor | Expected runs scoring |
| `index_hr` | HR park factor | Power projection adjustment |
| `index_woba` | wOBA park factor | Overall offense adjustment |
| `index_wobacon` | wOBAcon park factor | Contact quality adjustment |
| `index_xwobacon` | xwOBAcon park factor | Expected contact adjustment |
| `index_obp` | OBP park factor | On-base adjustment |
| `index_so` | K park factor | Strikeout rate adjustment |
| `index_bb` | BB park factor | Walk rate adjustment |
| `index_hits` | Hits park factor | AVG adjustment |
| `index_1b` / `index_2b` / `index_3b` | Hit type factors | Hit distribution |
| `index_hardhit` | HardHit park factor | Contact quality |

**Values are indexed to 100** (100 = neutral, 110 = 10% above average, 90 = 10% below).

### 2.4 Fantasy Application

```python
def apply_park_factor(proj: dict, venue_id: int, park_factors: dict) -> dict:
    """
    Adjust projections based on home park factors.
    Example: Coors Field (index_runs=112) boosts R/RBI/AVG.
    """
    pf = park_factors.get(venue_id, {})
    
    # Offense boosts
    runs_factor = pf.get('index_runs', 100) / 100
    hr_factor = pf.get('index_hr', 100) / 100
    
    adjusted = {
        'r': proj['r'] * runs_factor,
        'rbi': proj['rbi'] * runs_factor,
        'hr': proj['hr'] * hr_factor,
        'avg': proj['avg'] * (pf.get('index_hits', 100) / 100),
    }
    return adjusted
```

---

## 3. BAT TRACKING LEADERBOARD

### 3.1 Endpoint

```
https://baseballsavant.mlb.com/leaderboard/bat-tracking
```

**❌ Does NOT support `&csv=true`** — visual scatterplot page with embedded data.

**✅ Data is embedded** as `var data = [...]` with 218 players and 52 metrics.

### 3.2 Extraction Method

Same regex approach as Park Factors:

```python
def fetch_bat_tracking() -> list[dict]:
    url = "https://baseballsavant.mlb.com/leaderboard/bat-tracking"
    resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
    match = re.search(r'var data = (\[.*?\]);', resp.text, re.DOTALL)
    return json.loads(match.group(1))
```

### 3.3 Key Metrics (52 Fields)

| Field | Description | Fantasy Relevance |
|-------|-------------|-------------------|
| `avg_sweetspot_speed_mph` | Bat speed at sweet spot | **Power predictor** |
| `avg_sweetspot_speed_mph_qualified` | Qualified competitive swings | More reliable |
| `avg_is_sweetspot_speed_high` | % of swings with high bat speed | Consistency |
| `swing_length_qualified` | Average swing length (ft) | Contact ability proxy |
| `attack_angle` | Average attack angle (degrees) | Launch angle tendency |
| `rate_ideal_attack_angle` | % swings at ideal angle | Optimization |
| `squared_up` | Total squared-up swings | Contact quality |
| `squared_up_qualified` | Qualified squared-up swings | Reliability |
| `squared_up_with_speed` | Squared-up + fast swing | **Elite contact** |
| `swords` | Times "sworded" (bad swing) | Weakness indicator |
| `delta_run_exp` | Run value from bat tracking | Overall contribution |
| `bat_contact` | Total bat-on-contact events | Sample size |
| `swings_qualified` | Qualified competitive swings | Sample size |

### 3.4 Application to Proxy Engine

Bat tracking metrics can **differentiate players with similar traditional stats**:

- **High bat speed + high squared-up rate** = likely to maintain or improve power
- **Low bat speed + declining attack angle** = aging risk or power decline
- **High swing length** = may struggle with contact (higher K%)

These can be used as **tiebreakers** when two waiver candidates have similar z-scores, or as **confidence modifiers** on projection ranges.

---

## 4. PROBABLE PITCHERS PAGE

### 4.1 Endpoint

```
https://baseballsavant.mlb.com/probable-pitchers
```

**❌ No CSV export, no embedded JSON data array.**

**⚠️ Requires HTML scraping** to extract daily matchup information.

### 4.2 Page Structure

The page shows daily game matchups. For each game:
- Away team @ Home team
- Game time, venue
- **Two probable pitchers** with photos and stats
- Each pitcher has a "Career vs Current [Opponent] Roster" table with:
  - PA, K%, BB%, AVG, wOBA
  - Exit Velo, Launch Angle, xBA, xSLG, xwOBA
- Links to `player_matchup` detail pages

### 4.3 Extractable Data

Via regex on the HTML:
- **60 matchup links** found per page
- Each link contains: `teamPitching`, `teamBatting`, `player_id`
- Pitcher names from `<h3>` tags
- Career vs roster stats from tables

### 4.4 Fantasy Application

This page is most valuable for **daily lineup optimization**, not proxy generation:

1. **Confirm probable pitchers** — cross-reference with our `probable_pitchers` table
2. **Opponent quality assessment** — a pitcher's career xwOBA vs the current opposing roster
3. **Start/sit decisions** — if a pitcher has terrible career numbers vs the opponent's current lineup, bench them

### 4.5 Scraping Strategy

```python
from bs4 import BeautifulSoup

def scrape_probable_pitchers(html: str) -> list[dict]:
    soup = BeautifulSoup(html, 'html.parser')
    matchups = []
    
    # Each game is a section with an h2 for teams
    for section in soup.find_all('div', class_=re.compile('game|matchup')):
        game_title = section.find('h2')
        if not game_title:
            continue
        
        pitchers = section.find_all('h3')
        stats_tables = section.find_all('table')
        
        matchup = {
            'game': game_title.text.strip(),
            'pitchers': []
        }
        
        for i, pitcher in enumerate(pitchers):
            stats = {}
            if i * 2 + 1 < len(stats_tables):
                tables = stats_tables[i*2:i*2+2]
                # Extract PA, K%, BB%, AVG, wOBA from first table
                # Extract EV, LA, xBA, xSLG, xwOBA from second table
            
            matchup['pitchers'].append({
                'name': pitcher.text.strip(),
                'career_vs_roster': stats
            })
        
        matchups.append(matchup)
    
    return matchups
```

---

## 5. EXIT VELOCITY & BARRELS LEADERBOARD

### 5.1 Endpoint

```
https://baseballsavant.mlb.com/leaderboard/statcast?csv=true
```

**✅ Supports `&csv=true`** — returns CSV with 280 rows.

### 5.2 Available Metrics

From the CSV we downloaded (33,006 bytes, 280 rows):
- `player_id`, `attempts` (BBE)
- `avg_hit_angle`, `anglesweetspotpercent`
- `max_hit_speed`, `avg_hit_speed`, `ev50`
- `fbld` (fly ball + line drive)
- `max_distance`, `avg_distance`, `avg_hr_distance`
- Hard hit counts and percentages
- Barrel counts and percentages

This leaderboard is a **subset** of what the Custom Leaderboard provides. Since the Custom Leaderboard can select ALL of these columns plus 150+ more, **the Custom Leaderboard should be the primary ingestion target**.

---

## 6. COMPLETE DATA SOURCE PRIORITY MATRIX

| Priority | Source | Extraction | Update Frequency | Fantasy Use |
|----------|--------|-----------|------------------|-------------|
| **P0** | Custom Leaderboard (batter) | `&csv=true` HTTP | Daily/Weekly | **Primary proxy data** — all stats needed for z-scores |
| **P0** | Custom Leaderboard (pitcher) | `&csv=true` HTTP | Daily/Weekly | **Primary proxy data** — ERA, WHIP, K9, xERA, xwOBA allowed |
| **P1** | Park Factors | HTML regex | Monthly/Seasonally | Park-adjusted projections, daily lineup slotting |
| **P2** | Bat Tracking | HTML regex | Weekly | Power/contact quality tiebreakers, breakout detection |
| **P2** | Probable Pitchers | HTML scrape | Daily | Lineup confirmation, start/sit decisions |
| **P3** | Exit Velocity & Barrels | `&csv=true` HTTP | Weekly | Redundant with Custom Leaderboard |

---

## 7. UPDATED PIPELINE ARCHITECTURE

### 7.1 Module Structure

```
backend/fantasy_baseball/
├── savant_ingestion.py              ← NEW: Multi-source ingestion client
│   ├── SavantIngestionClient
│   │   ├── fetch_custom_leaderboard(type, selections, min) → str (CSV)
│   │   ├── fetch_ev_barrels() → str (CSV)
│   │   ├── fetch_park_factors() → list[dict]
│   │   ├── fetch_bat_tracking() → list[dict]
│   │   ├── fetch_probable_pitchers() → list[dict]
│   │   └── _extract_data_array(html) → list[dict]
│   └── parse_savant_csv(text) → list[dict]
│
├── savant_proxy_engine.py           ← NEW: Proxy generation
│   ├── StatcastProxyEngine
│   │   ├── get_proxy_projection(name) → dict
│   │   ├── _build_batter_proxy(savant_row) → dict
│   │   ├── _build_pitcher_proxy(savant_row) → dict
│   │   ├── _apply_park_factors(proj, venue_id) → dict
│   │   └── _compute_synthetic_cat_scores(proj, type) → dict
│   └── _shrinkage_formula(sample, stabilization) → float
│
└── player_board.py                  ← MODIFIED
    └── get_or_create_projection()
        ├── Existing: cache → board → DB lookup
        ├── NEW: Custom Leaderboard lookup
        ├── NEW: Park factor adjustment
        └── FALLBACK: population-prior proxy
```

### 7.2 Recommended Ingestion Schedule

```python
SCHEDULE = {
    # Daily (6:00 AM ET)
    "custom_leaderboard_batter": "0 6 * * *",
    "custom_leaderboard_pitcher": "5 6 * * *",
    "probable_pitchers": "10 6 * * *",
    
    # Weekly (Mondays at 7:00 AM ET)
    "bat_tracking": "0 7 * * 1",
    
    # Monthly (1st of month at 8:00 AM ET)
    "park_factors": "0 8 1 * *",
}
```

### 7.3 Database Schema

```sql
-- Primary Savant data table
CREATE TABLE statcast_leaderboard (
    id SERIAL PRIMARY KEY,
    player_id INTEGER NOT NULL,
    player_name VARCHAR(255),
    player_type VARCHAR(10),  -- 'batter' or 'pitcher'
    season INTEGER NOT NULL,
    
    -- Counting stats
    pa INTEGER DEFAULT 0,
    ab INTEGER DEFAULT 0,
    h INTEGER DEFAULT 0,
    single INTEGER DEFAULT 0,
    double INTEGER DEFAULT 0,
    triple INTEGER DEFAULT 0,
    home_run INTEGER DEFAULT 0,
    strikeout INTEGER DEFAULT 0,
    walk INTEGER DEFAULT 0,
    r INTEGER DEFAULT 0,
    rbi INTEGER DEFAULT 0,
    sb INTEGER DEFAULT 0,
    cs INTEGER DEFAULT 0,
    tb INTEGER DEFAULT 0,
    games INTEGER DEFAULT 0,
    
    -- Rate stats
    k_percent NUMERIC(5,1),
    bb_percent NUMERIC(5,1),
    batting_avg NUMERIC(5,3),
    slg_percent NUMERIC(5,3),
    on_base_percent NUMERIC(5,3),
    ops NUMERIC(5,3),
    iso NUMERIC(5,3),
    babip NUMERIC(5,3),
    
    -- Expected stats
    xba NUMERIC(5,3),
    xslg NUMERIC(5,3),
    xwoba NUMERIC(5,3),
    xobp NUMERIC(5,3),
    xiso NUMERIC(5,3),
    xwobacon NUMERIC(5,3),
    xbacon NUMERIC(5,3),
    
    -- Batted ball quality
    exit_velocity_avg NUMERIC(5,1),
    launch_angle_avg NUMERIC(5,1),
    sweet_spot_percent NUMERIC(5,1),
    barrel_batted_rate NUMERIC(5,1),
    hard_hit_percent NUMERIC(5,1),
    avg_best_speed NUMERIC(6,2),
    avg_hyper_speed NUMERIC(6,2),
    
    -- Plate discipline
    whiff_percent NUMERIC(5,1),
    swing_percent NUMERIC(5,1),
    z_swing_percent NUMERIC(5,1),
    oz_swing_percent NUMERIC(5,1),
    iz_contact_percent NUMERIC(5,1),
    oz_contact_percent NUMERIC(5,1),
    
    -- Batted ball profile
    gb_percent NUMERIC(5,1),
    fb_percent NUMERIC(5,1),
    ld_percent NUMERIC(5,1),
    popup_percent NUMERIC(5,1),
    pull_percent NUMERIC(5,1),
    straightaway_percent NUMERIC(5,1),
    opposite_percent NUMERIC(5,1),
    
    -- Baserunning
    sprint_speed NUMERIC(5,1),
    bolts INTEGER DEFAULT 0,
    hp_to_1b NUMERIC(5,2),
    
    -- Fielding
    oaa NUMERIC(5,1),
    
    -- Bat tracking (2024+)
    avg_swing_speed NUMERIC(5,1),
    fast_swing_rate NUMERIC(5,1),
    avg_swing_length NUMERIC(5,2),
    blasts_contact NUMERIC(5,1),
    blasts_swing NUMERIC(5,1),
    squared_up_contact NUMERIC(5,1),
    squared_up_swing NUMERIC(5,1),
    swords INTEGER DEFAULT 0,
    attack_angle NUMERIC(5,1),
    
    -- Pitcher-specific (nullable for batters)
    ip NUMERIC(5,1),
    era NUMERIC(5,2),
    whip NUMERIC(5,3),
    k_9 NUMERIC(5,2),
    wins INTEGER,
    losses INTEGER,
    qs INTEGER,
    xera NUMERIC(5,2),
    
    -- Metadata
    data_source VARCHAR(50) DEFAULT 'savant_custom_leaderboard',
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(player_id, player_type, season)
);

-- Park factors table
CREATE TABLE statcast_park_factors (
    id SERIAL PRIMARY KEY,
    venue_id INTEGER NOT NULL,
    venue_name VARCHAR(255),
    team_id INTEGER,
    team_name VARCHAR(255),
    season INTEGER NOT NULL,
    year_range VARCHAR(20),
    n_pa INTEGER,
    
    index_runs INTEGER,
    index_hr INTEGER,
    index_woba INTEGER,
    index_wobacon INTEGER,
    index_xwobacon INTEGER,
    index_xbacon INTEGER,
    index_obp INTEGER,
    index_so INTEGER,
    index_bb INTEGER,
    index_bacon INTEGER,
    index_hits INTEGER,
    index_1b INTEGER,
    index_2b INTEGER,
    index_3b INTEGER,
    index_hardhit INTEGER,
    
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(venue_id, season)
);
```

---

## 8. RISK ASSESSMENT

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Savant changes Custom Leaderboard URL | Low | High | Pin selection parameters; monitor for 404s |
| Savant adds Cloudflare | Medium | Medium | Keep Playwright fallback ready |
| CSV column names change | Low | Medium | Validate headers; alert on mismatch |
| Park Factors / Bat Tracking page structure changes | Low | Medium | Both use simple regex; easy to adapt |
| Probable Pitchers HTML structure changes | Medium | Low | Non-critical feature; degrade gracefully |
| Rate limiting on frequent requests | Low | Low | Add 1-2s delays; cache for 6-12 hours |

---

## 9. IMPLEMENTATION ROADMAP (UPDATED)

### Phase A: Core Ingestion (2 days)

1. Create `backend/fantasy_baseball/savant_ingestion.py`
   - `SavantIngestionClient` with all 5 data sources
   - `parse_savant_csv()` with header normalization
   - `_extract_data_array()` for HTML-extracted sources
   - Unit tests with mock responses

2. Database migrations
   - `statcast_leaderboard` table (comprehensive schema above)
   - `statcast_park_factors` table

### Phase B: Proxy Engine (3 days)

1. Create `backend/fantasy_baseball/savant_proxy_engine.py`
   - Batter proxy with shrinkage + translation
   - Pitcher proxy (use real ERA/WHIP/K9 when available)
   - Park factor adjustment layer
   - Population-prior fallback

2. Integrate into `get_or_create_projection()`

### Phase C: Daily Pipeline (1 day)

1. Admin endpoint: `POST /api/admin/ingest-savant`
2. Cron/scheduler configuration
3. Monitoring and alerting

### Phase D: Advanced Features (3 days)

1. Bat tracking tiebreakers in waiver recommendations
2. Probable pitcher scraping for lineup confirmation
3. Park factor visualization on dashboard

---

## 10. REFERENCES

- **Custom Leaderboard:** https://baseballsavant.mlb.com/leaderboard/custom
- **Park Factors:** https://baseballsavant.mlb.com/leaderboard/statcast-park-factors
- **Bat Tracking:** https://baseballsavant.mlb.com/leaderboard/bat-tracking
- **Probable Pitchers:** https://baseballsavant.mlb.com/probable-pitchers
- **Exit Velocity & Barrels:** https://baseballsavant.mlb.com/leaderboard/statcast
- **Phase 9 Research:** `reports/2026-04-24-phase9-statcast-bayesian-proxy-research.md`
- **v1.0 Report:** `reports/2026-04-24-savant-pipeline-architecture-report.md`

---

*Report compiled by Kimi CLI v1.17.0 | Live endpoints tested: 8 | Data sources reverse-engineered: 5 | Total fields catalogued: 250+ | Players sampled: 1,445 | Parks sampled: 30*
