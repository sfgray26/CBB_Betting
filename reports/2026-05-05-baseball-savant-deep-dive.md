# Baseball Savant & FanGraphs Data Sourcing Deep Dive
## Corrected Architecture for Statcast + Pitching+ Metrics

**Date:** 2026-05-05  
**Author:** Kimi CLI (Deep Intelligence Unit)  
**Classification:** P0-Critical — Blocks Epic 2 (Statcast Integration)

---

## EXECUTIVE SUMMARY

The current codebase has a **fundamental data source mismatch**:

1. **`stuff_plus` and `location_plus` are FanGraphs metrics**, NOT Baseball Savant metrics.
2. The committed scraper (`backfill_statcast_pitcher_advanced.py`) attempts to fetch them from a **non-existent Savant endpoint** (`/leaderboard/pitching`), resulting in a 404.
3. **Production DB:** `stuff_plus` and `location_plus` are 100% NULL (0/546 rows) because the pipeline is pointed at the wrong source.
4. **Sprint speed** IS a Savant metric and CAN be fetched from Savant's real CSV endpoint.

**Recommendation:** Split the data model. Keep Savant data in `statcast_*` tables. Create a new `fangraphs_pitcher_metrics` table for Stuff+/Location+/Pitching+. Use `pybaseball.fg_pitching_data()` (already in requirements.txt) to source FanGraphs data.

---

## PART 1: BASEBALL SAVANT — WHAT'S ACTUALLY AVAILABLE

### 1.1 Savant CSV Endpoint Inventory

Baseball Savant exposes **dozens** of leaderboards with CSV download buttons. The URL pattern is:

```
https://baseballsavant.mlb.com/leaderboard/{metric}?year={YYYY}&...&csv=true
```

**Verified working endpoints (2026 season):**

| Category | Endpoint Path | Metrics Returned | Fantasy Relevance |
|----------|--------------|------------------|-------------------|
| **Running** | `/leaderboard/sprint_speed` | `sprint_speed`, `bolts`, `hp_to_1b` | ✅ High — stolen base prediction |
| | `/leaderboard/running_splits_90_ft` | `90ft_splits` | Medium — baserunning |
| **Batting** | `/statcast_leaderboard` (batting) | `exit_velocity`, `barrels`, `hard_hit_pct`, `sweet_spot_pct` | ✅ High — power/skill indicators |
| | `/leaderboard/expected_statistics` | `xba`, `xslg`, `xwoba`, `xwoBACON` | ✅ High — luck/regression |
| | `/leaderboard/exit_velocity_barrels` | `avg_hit_speed`, `max_hit_speed`, `barrels`, `brl_pa` | ✅ High — power |
| | `/leaderboard/home_runs` | `distance`, `ev`, `launch_angle` | Low — novelty |
| **Pitching** | `/leaderboard/pitch_arsenal_stats` | `pitch_type`, `velo`, `spin`, `movement`, `usage_pct` | ✅ High — pitcher arsenal |
| | `/leaderboard/pitch_tempo` | `tempo`, `timer_infractions` | Medium — pace |
| | `/leaderboard/expected_statistics` (pitcher) | `xera`, `xba_allowed`, `xslg_allowed` | ✅ High — pitcher skill |
| | `/leaderboard/arm_angle` | `arm_angle`, `arm_slot` | Medium — deception |
| | `/leaderboard/active_spin` | `active_spin_pct` | Medium — spin quality |
| **Fielding** | `/leaderboard/outs_above_average` | `oaa`, `runs_prevented` | ✅ High — defensive value |
| | `/leaderboard/arm_strength` | `arm_strength`, `exchange`, `pop_time` | Medium — catcher/IF |
| | `/leaderboard/outfield_jump` | `jump`, `reaction`, `burst` | Medium — OF defense |
| | `/leaderboard/catcher_blocking` | `blocking_runs` | Low — catcher-specific |
| | `/leaderboard/catcher_framing` | `framing_runs` | Medium — catcher value |
| **Bat Tracking** | `/leaderboard/bat_tracking` (new 2024) | `bat_speed`, `swing_length`, `attack_angle`, `squared_up_pct` | ✅ High — contact quality |

### 1.2 What Baseball Savant Does NOT Have

**Critical finding:** Baseball Savant does **NOT** publish:
- `Stuff+` / `Location+` / `Pitching+`
- Any endpoint at `/leaderboard/pitching`
- Pitch-level command metrics (these are FanGraphs models)

Savant has **raw pitch physics** (velocity, spin, movement, release point), but the **processed quality scores** (Stuff+, Location+, Pitching+) are proprietary models built by FanGraphs.

### 1.3 Savant CSV Format Notes

From manual inspection and `baseballr` documentation:

```python
# Example: Sprint Speed CSV
import pandas as pd
url = "https://baseballsavant.mlb.com/leaderboard/sprint_speed?year=2026&position=&team=&min=0&csv=true"
df = pd.read_csv(url)
# Columns: player_id, player_name, team, sprint_speed, bolts, hp_to_1b, ...
# player_id is INTEGER (mlbam_id)
```

**Important quirks:**
- Some CSVs have a **BOM** (Byte Order Mark) — use `encoding='utf-8-sig'`
- `player_id` is numeric but should be cast to `str` before DB insert (matches our VARCHAR schema)
- Missing values are blank strings, not "N/A" — pandas reads them as NaN
- Some leaderboards (e.g., bat tracking) are **season-to-date only** — no historical year filter

---

## PART 2: FANGRAPHS STUFF+ / LOCATION+ / PITCHING+

### 2.1 What These Metrics Are

From FanGraphs documentation (`library.fangraphs.com`):

| Metric | What It Measures | Data Source | Stabilization |
|--------|-----------------|-------------|---------------|
| **Stuff+** | Physical pitch quality (velocity, movement, spin, release point, seam-shifted wake) | Pitch-level Statcast physics | ~80 pitches |
| **Location+** | Pitch location quality, count- and pitch-type-adjusted | Pitch-level zone location | ~400 pitches |
| **Pitching+** | Combined model: stuff + location + count + batter handedness | Pitch-level everything | ~250 pitches |

**Scale:** 100 = league average. 10 points ≈ 1 standard deviation at pitch level.

### 2.2 How to Source Them

**pybaseball already supports this.**

```python
from pybaseball import fg_pitching_data

# Get ALL pitchers (qual=0 means no minimum IP filter)
df = fg_pitching_data(2026, qual=0)

# Stuff+, Location+, Pitching+ are in the DataFrame:
# Columns include: "Name", "Team", "Stuff+", "Location+", "Pitching+", "playerid"
# playerid = FanGraphs ID (NOT mlbam_id)
```

**This is the canonical source.** The `savant-extras` library tried to build a scraper for this but **removed it in v0.4.2** because `pybaseball.fg_pitching_data()` already provides identical data.

### 2.3 Why the Current Scraper Failed

The committed scraper tried:
```
https://baseballsavant.mlb.com/leaderboard/pitching?year=2026&position=P&team=&min=0&csv=true
```

**This URL does not exist.** Savant has no `/leaderboard/pitching` path. The 404 is expected.

Even if the scraper hit a real Savant pitching endpoint (like `/leaderboard/pitch_arsenal_stats`), it would get **raw physics** (velo, spin, movement) — not the **processed Stuff+ scores**.

---

## PART 3: CORRECTED DATA ARCHITECTURE

### 3.1 Current Schema (Wrong)

```sql
-- statcast_pitcher_metrics (current, misleading name)
stuff_plus DOUBLE PRECISION,      -- NULL for all 546 rows
location_plus DOUBLE PRECISION,  -- NULL for all 546 rows
```

**Problem:** These columns are in a "statcast" table but the data comes from FanGraphs.

### 3.2 Proposed Schema (Correct)

**Option A: Minimal Change (Rename Columns)**
```sql
-- statcast_pitcher_metrics: keep only TRUE Statcast metrics
ALTER TABLE statcast_pitcher_metrics
    DROP COLUMN stuff_plus,
    DROP COLUMN location_plus;

-- Add to a new FanGraphs table
CREATE TABLE fangraphs_pitcher_metrics (
    id BIGSERIAL PRIMARY KEY,
    bdl_player_id INTEGER REFERENCES player_id_mapping(bdl_id),
    season INTEGER NOT NULL,
    fangraphs_id INTEGER,           -- FanGraphs player ID
    stuff_plus FLOAT,
    location_plus FLOAT,
    pitching_plus FLOAT,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (bdl_player_id, season)
);
```

**Option B: Generic External Metrics Table**
```sql
CREATE TABLE external_pitcher_metrics (
    id BIGSERIAL PRIMARY KEY,
    bdl_player_id INTEGER REFERENCES player_id_mapping(bdl_id),
    season INTEGER NOT NULL,
    source VARCHAR(20) NOT NULL,     -- 'fangraphs', 'statcast', 'baseball_ref'
    metric_name VARCHAR(30) NOT NULL, -- 'stuff_plus', 'location_plus', etc.
    metric_value FLOAT,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (bdl_player_id, season, source, metric_name)
);
```

**Recommendation:** Option A is cleaner for the current use case. Option B is more extensible if we add Baseball-Reference or other sources later.

### 3.3 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                            │
├──────────────────────┬──────────────────────┬───────────────────┤
│  Baseball Savant     │  FanGraphs           │  Yahoo API        │
│  (Statcast)          │  (Pitching+)         │  (Ownership)      │
├──────────────────────┼──────────────────────┼───────────────────┤
│  • sprint_speed      │  • stuff_plus        │  • percent_owned  │
│  • exit_velocity     │  • location_plus     │  • add/drop rates │
│  • barrels           │  • pitching_plus     │                   │
│  • oaa               │                      │                   │
│  • bat_speed (new)   │                      │                   │
└──────────────────────┴──────────────────────┴───────────────────┘
           │                      │                      │
           ▼                      ▼                      ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌─────────────┐
│ statcast_batter_     │  │ fangraphs_pitcher_   │  │ player_     │
│ metrics              │  │ metrics              │  │ market_     │
├──────────────────────┤  ├──────────────────────┤  │ signals     │
│  sprint_speed        │  │  stuff_plus          │  ├─────────────┤
│  bolts               │  │  location_plus       │  │ yahoo_owned │
│                      │  │  pitching_plus       │  │ ownership_  │
│                      │  │                      │  │ velocity    │
└──────────────────────┘  └──────────────────────┘  └─────────────┘
```

---

## PART 4: IMPLEMENTATION PLAN

### 4.1 Immediate Fix (P0)

**PR 2.0 — Correct the Data Source Mismatch**

1. **Drop bad columns** from `statcast_pitcher_metrics`
   ```sql
   ALTER TABLE statcast_pitcher_metrics
       DROP COLUMN IF EXISTS stuff_plus,
       DROP COLUMN IF EXISTS location_plus;
   ```

2. **Create `fangraphs_pitcher_metrics` table**
   ```sql
   CREATE TABLE IF NOT EXISTS fangraphs_pitcher_metrics (
       id BIGSERIAL PRIMARY KEY,
       bdl_player_id INTEGER REFERENCES player_id_mapping(bdl_id),
       season INTEGER NOT NULL,
       fangraphs_id INTEGER,
       stuff_plus FLOAT,
       location_plus FLOAT,
       pitching_plus FLOAT,
       updated_at TIMESTAMPTZ DEFAULT NOW(),
       UNIQUE (bdl_player_id, season)
   );
   CREATE INDEX idx_fg_pitcher_bdl_season ON fangraphs_pitcher_metrics(bdl_player_id, season);
   ```

3. **Write `backend/ingestion/fangraphs_scraper.py`**
   ```python
   from pybaseball import fg_pitching_data
   import pandas as pd
   
   def fetch_pitcher_quality(year: int = 2026) -> pd.DataFrame:
       """
       Fetch Stuff+/Location+/Pitching+ from FanGraphs via pybaseball.
       Returns DataFrame with: fangraphs_id, name, team, stuff_plus, location_plus, pitching_plus
       """
       df = fg_pitching_data(year, qual=0)
       # Select only the columns we need
       return df[["Name", "Team", "playerid", "Stuff+", "Location+", "Pitching+"]].rename(
           columns={
               "playerid": "fangraphs_id",
               "Stuff+": "stuff_plus",
               "Location+": "location_plus",
               "Pitching+": "pitching_plus",
           }
       )
   ```

4. **Update feature flags**
   ```sql
   -- Disable the old broken flag
   UPDATE feature_flags SET enabled = false WHERE flag_name = 'statcast_stuff_plus_enabled';
   UPDATE feature_flags SET enabled = false WHERE flag_name = 'statcast_location_plus_enabled';
   
   -- Add new correct flags
   INSERT INTO feature_flags (flag_name, enabled, description) VALUES
       ('fangraphs_pitching_plus_enabled', false, 'Enable FanGraphs Stuff+/Location+/Pitching+ in scoring'),
       ('statcast_sprint_speed_enabled', false, 'Enable Baseball Savant sprint_speed in scoring')
   ON CONFLICT (flag_name) DO NOTHING;
   ```

### 4.2 Sprint Speed Backfill (P1)

**PR 2.1 — Savant Sprint Speed**

Use the **real** Savant endpoint:
```
https://baseballsavant.mlb.com/leaderboard/sprint_speed?year=2026&position=&team=&min=0&csv=true
```

```python
def fetch_sprint_speed(year: int = 2026) -> pd.DataFrame:
    url = f"https://baseballsavant.mlb.com/leaderboard/sprint_speed?year={year}&position=&team=&min=0&csv=true"
    response = requests.get(url, headers=_BROWSER_HEADERS)
    if response.status_code != 200:
        logger.warning("Savant sprint_speed fetch failed: %s", response.status_code)
        return pd.DataFrame()
    
    df = pd.read_csv(io.StringIO(response.text), encoding="utf-8-sig")
    return df[["player_id", "player_name", "sprint_speed"]].rename(
        columns={"player_id": "mlbam_id"}
    )
```

### 4.3 Other High-Value Savant Metrics (P2)

Priority order for future Epic 2.x PRs:

1. **Exit Velocity / Barrels** (`/statcast_leaderboard` batting) — power indicator
2. **Expected Stats** (`/leaderboard/expected_statistics`) — xBA, xSLG, xwOBA — luck/regression
3. **Pitch Arsenal** (`/leaderboard/pitch_arsenal_stats`) — velocity, spin, movement by pitch type
4. **Outs Above Average** (`/leaderboard/outs_above_average`) — defensive value
5. **Bat Tracking** (`/leaderboard/bat_tracking`) — bat speed, attack angle (2024+)

---

## PART 5: ID MAPPING CHALLENGE

### 5.1 The Problem

FanGraphs uses `fangraphs_id` (e.g., `19291` for Jacob deGrom).  
Baseball Savant uses `mlbam_id` (e.g., `592866` for deGrom).  
Our DB uses `bdl_player_id` (from Ball Don't Lie / player_id_mapping).

**We need a three-way mapping:**
```
bdl_player_id ↔ mlbam_id ↔ fangraphs_id
```

### 5.2 Solution

`pybaseball` provides a player ID lookup:

```python
from pybaseball import playerid_lookup

# Lookup by name
lookup = playerid_lookup("deGrom", "Jacob")
# Returns: mlbam_id, retro_id, bbref_id, fangraphs_id, ...
```

**Recommended approach:**
1. Add `fangraphs_id` column to `player_id_mapping` table
2. Backfill using `pybaseball.playerid_lookup()` for all known players
3. Use `fangraphs_id` as the join key for FanGraphs data
4. Use `mlbam_id` as the join key for Savant data

---

## PART 6: RISK ASSESSMENT

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| FanGraphs changes pybaseball API | Medium | High | Pin pybaseball version in requirements.txt |
| FanGraphs adds rate limiting | Medium | Medium | Add caching, respect robots.txt, use browser headers |
| Savant CSV format changes | Low | Medium | Validate columns on parse, fail gracefully |
| pybaseball `fg_pitching_data` breaks | Low | High | Fallback to direct FanGraphs leaderboard scrape |
| ID mapping failures (name mismatches) | High | Medium | Log unmatched players, manual review queue |

---

## PART 7: CORRECTED EPIC 2 BACKLOG

| PR | Title | Data Source | Table | Status |
|----|-------|-------------|-------|--------|
| **2.0** | Fix data source mismatch | N/A | `fangraphs_pitcher_metrics` | **BLOCKED — needs Claude approval** |
| **2.1** | Savant sprint_speed scraper | Savant CSV | `statcast_batter_metrics` | Ready for Codex |
| **2.2** | Sprint speed ingestion hook | Savant CSV | `statcast_batter_metrics` | Ready for Codex |
| **2.3** | Sprint speed validation | N/A | N/A | Ready for Codex |
| **2.4** | Sprint speed backfill | Savant CSV | `statcast_batter_metrics` | Ready for Codex |
| **2.5** | FanGraphs Stuff+/Location+ scraper | pybaseball | `fangraphs_pitcher_metrics` | Ready for Codex (after 2.0) |
| **2.6** | Stuff+/Location+ ingestion hook | pybaseball | `fangraphs_pitcher_metrics` | Ready for Codex (after 2.0) |
| **2.7** | Exit velocity / barrels scraper | Savant CSV | `statcast_batter_metrics` | Future |
| **2.8** | Expected stats scraper | Savant CSV | `statcast_batter_metrics` | Future |
| **2.9** | Pitch arsenal scraper | Savant CSV | `statcast_pitcher_metrics` | Future |
| **2.10** | Outs Above Average scraper | Savant CSV | `statcast_fielder_metrics` | Future |

---

## APPENDIX A: Verified Savant CSV URLs

```python
SAVANT_ENDPOINTS = {
    "sprint_speed": "https://baseballsavant.mlb.com/leaderboard/sprint_speed?year={year}&position=&team=&min=0&csv=true",
    "exit_velocity_barrels": "https://baseballsavant.mlb.com/leaderboard/exit_velocity_barrels?year={year}&position=&team=&min=0&csv=true",
    "expected_statistics_batting": "https://baseballsavant.mlb.com/leaderboard/expected_statistics?year={year}&position=&team=&min=0&player_type=batter&csv=true",
    "expected_statistics_pitching": "https://baseballsavant.mlb.com/leaderboard/expected_statistics?year={year}&position=&team=&min=0&player_type=pitcher&csv=true",
    "pitch_arsenal": "https://baseballsavant.mlb.com/leaderboard/pitch_arsenal_stats?year={year}&position=&team=&min=0&csv=true",
    "outs_above_average": "https://baseballsavant.mlb.com/leaderboard/outs_above_average?year={year}&position=&team=&min=0&csv=true",
    "arm_strength": "https://baseballsavant.mlb.com/leaderboard/arm_strength?year={year}&position=&team=&min=0&csv=true",
    "outfield_jump": "https://baseballsavant.mlb.com/leaderboard/outfield_jump?year={year}&position=&team=&min=0&csv=true",
    "bat_tracking": "https://baseballsavant.mlb.com/leaderboard/bat_tracking?year={year}&position=&team=&min=0&csv=true",
    "pitch_tempo": "https://baseballsavant.mlb.com/leaderboard/pitch_tempo?year={year}&position=&team=&min=0&csv=true",
}
```

**Note:** The endpoint `/leaderboard/pitching` does **NOT** exist. Do not use it.

---

## APPENDIX B: pybaseball FanGraphs Functions

```python
from pybaseball import (
    fg_pitching_data,      # Stuff+, Location+, Pitching+
    fg_batting_data,       # Advanced batting stats
    statcast_outs_above_average,
    statcast_outfielder_jump,
    statcast_sprint_speed,  # May exist in newer pybaseball versions
    playerid_lookup,        # ID mapping
)
```

Reference: https://github.com/jldbc/pybaseball

---

## APPENDIX C: FanGraphs Stuff+ Primer

From `library.fangraphs.com/pitching/stuff-location-and-pitching-primer/`:

> "Stuff+ looks only at the physical characteristics of a pitch. Important features include release point, velocity, vertical and horizontal movement, and spin rate."
>
> "Location+ is a count- and pitch type-adjusted judge of a pitcher's ability to put pitches in the right place. No velocity, movement, or any other physical characteristics are included."
>
> "Pitching+ is a third model that uses the physical characteristics, location, and count of each pitch. Batter handedness is also included."

**Scale:** 100 = league average

**Stabilization points:**
- Stuff+: ~80 pitches
- Location+: ~400 pitches  
- Pitching+: ~250 pitches

**Year-to-year stickiness:** Stuff+ is stickier than Location+, driving most of Pitching+'s season-to-season correlation.

---

## RECOMMENDATION

1. **Claude approves PR 2.0** (schema fix) — this is architectural
2. **Codex implements PR 2.1–2.4** (sprint_speed) — well-specified, Savant-only
3. **Codex implements PR 2.5–2.6** (Stuff+/Location+) — well-specified, FanGraphs-only
4. **Gemini deploys** when each PR batch is ready

Do NOT attempt to source Stuff+/Location+ from Baseball Savant. It is mathematically impossible — those metrics do not exist there.
