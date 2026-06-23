# Research: MLB Data Enhancements

This document contains research findings for the four requested MLB data enhancement tasks. It is prepared for Claude's review prior to implementation.

## Task 1: Platoon Split Data Integration (P2)
**Goal:** Research optimal sources for L/R split data (FanGraphs, Baseball-Reference, Statcast APIs) and document recommendations with sample data structures.

### Source Options:

1. **FanGraphs Undocumented API (Recommended for Aggregated Splits)**
   - **Endpoint:** `POST https://www.fangraphs.com/api/leaders/splits/splits-leaders`
   - **Method:** POST a JSON payload defining the split conditions.
   - **Split Codes (`strSplitArr`):**
     - vs. LHP = `[1]`
     - vs. RHP = `[2]`
   - **Sample Payload:**
     ```json
     {
       "strPlayerId": "all",
       "strSplitArr": [1],
       "strGroup": "season",
       "strPosition": "B",
       "strType": "1",
       "strStartDate": "2024-03-01",
       "strEndDate": "2024-11-01",
       "strSplitTeams": false,
       "dctFilters": [],
       "strStatType": "player",
       "strAutoPt": true,
       "arrPlayerId": [],
       "strSplitArrPitch": [],
       "arrWxTemperature": null,
       "arrWxPressure": null,
       "arrWxAirDensity": null,
       "arrWxElevation": null,
       "arrWxWindSpeed": null
     }
     ```
   - **Sample Response:**
     ```json
     {
       "data": [
         { "PlayerName": "Aaron Judge", "Team": "Yankees", "G": 158, "PA": 704, "wRC+": 218, "playerid": 15640 }
       ]
     }
     ```

2. **MLB Stats API (Direct Endpoint)**
   - **Endpoint:** `GET https://statsapi.mlb.com/api/v1/stats`
   - **Parameters:** `stats=statSplits`, `sitCodes=vl` (vs LHP) or `vr` (vs RHP)
   - **Example:** `https://statsapi.mlb.com/api/v1/stats?stats=statSplits&group=pitching&gameType=R&season=2026&playerPool=ALL&sitCodes=vl`

3. **PyBaseball (Statcast Raw Data)**
   - Extract raw pitch-level data and aggregate locally by checking pitcher throwing hand (`p_throws`) and batter stance (`stand`).

**Recommendation:** The FanGraphs undocumented API is ideal for standardized, aggregated metric splits (e.g., wRC+ vs LHP). The MLB Stats API is best if we already rely heavily on MLBAM player IDs and need situational box-stats dynamically.

---

## Task 2: Park Factor Quantification Research (P3)
**Goal:** Research park factor methodologies and document how to convert binary +/- to quantified factors (e.g., "+18% run environment").

### Methodology 1: FanGraphs Park Factor Formula
FanGraphs calculates a mathematically regressed park factor that adjusts for league context and sample size.

1. **Raw Park Factor:** `(H * T) / (((T - 1) * R) + H)`
   - `H` = Home Runs per Game
   - `R` = Road Runs per Game
   - `T` = Teams in League
2. **Intermediate PF:** `(Raw_PF + 1) / 2`
   - Dilutes the factor by half, as players only play 50% of games at home.
3. **Regressed PF:** `((Intermediate_PF - 1) * 0.9) + 1`
   - Regresses towards 1.00 (league average) based on sample size (0.9 used for a 5-year sample).

### Methodology 2: Baseball Savant (Statcast)
Savant provides granular park factors (e.g., wOBA, HR, Singles).
- **Endpoint:** `https://baseballsavant.mlb.com/leaderboard/statcast-park-factors?type=year&year=2024&batSide=&stat=index_wOBA&condition=All`
- **Data Extraction:** Savant doesn't have a clean JSON API. The JSON array is embedded in the HTML response under `var data = [...];`. We must regex this block out, parse it via `json.loads()`, and retrieve the quantified factor (e.g., `105` = 5% above average run environment).

**Recommendation:** For quantified adjustments, extract Baseball Savant's park factors (e.g., via the `savant_extras` python package or regex parsing). A factor of `118` implies a +18% boost to the run environment.

---

## Task 3: Opponent Scouting Data Sources (P2)
**Goal:** Research sources for opponent roster visibility, batting orders, probable pitchers.

### Source: MLB Stats API (Recommended)

1. **Probable Pitchers:**
   - **Endpoint:** `/schedule`
   - **Hydration:** `hydrate=probablePitcher(note)`
   - **Example:** `https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=2024-06-10&hydrate=probablePitcher`
   - **Data Path:** `dates[].games[].teams.away.probablePitcher`

2. **Batting Orders (Lineups) - All Games on Date:**
   - **Endpoint:** `/schedule`
   - **Hydration:** `hydrate=lineups`
   - **Example:** `https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=2024-06-10&hydrate=lineups`

3. **Batting Orders - Specific Game:**
   - **Endpoint:** `/game/{gamePk}/boxscore`
   - **Data Path:** Look at `teams.away.batters` and check the `players` object. The `battingOrder` field indicates the spot (100 = 1st, 200 = 2nd... 900 = 9th). Substitute players have intermediate numbers (e.g., 101).

**Recommendation:** Use the MLB Stats API's schedule endpoint with hydration `hydrate=probablePitcher,lineups` to fetch the slate of games, starters, and confirmed lineups efficiently in a single daily request.

---

## Task 4: Sample Size Warning System Design (P2)
**Goal:** Research thresholds for "Small Sample" warnings across different stat types.

### Stabilization Points
Stabilization in fantasy baseball is defined as the point where a metric's signal (skill) outweighs noise (variance) — typically a 0.70 correlation with a future sample of the same size. 

Below are the recommended thresholds to trigger a "Small Sample" warning:

#### Hitters (Plate Appearances / At Bats)
*   **K% (Strikeout Rate):** Stabilizes at **60 PA** (~2 weeks)
*   **BB% (Walk Rate):** Stabilizes at **120 PA** (~1 month)
*   **ISO (Power):** Stabilizes at **160 AB** (~5-6 weeks)
*   **HR Rate:** Stabilizes at **170 PA** (~6 weeks)
*   **OBP:** Stabilizes at **460 PA** (~75% of a season)
*   **Batting Average (AVG):** Stabilizes at **910 AB** (~1.5 seasons)
*   **BABIP:** Stabilizes at **820 BIP** (~2 seasons)
*   **Statcast EV/Hard Hit%:** Stabilizes at **50 BIP** (~2-3 weeks)

#### Pitchers (Batters Faced / Innings Pitched)
*   **K% (Strikeout Rate):** Stabilizes at **70 BF** (~17 IP / 3 starts)
*   **BB% (Walk Rate):** Stabilizes at **170 BF** (~40 IP / 7-8 starts)
*   **GB% / FB%:** Stabilizes at **70 BIP** (~25 IP)
*   **ERA / AVG:** Stabilizes at **630 BF** (~150 IP / full season)
*   **Pitching+ / Stuff+:** Stabilizes at **300-400 pitches** (~3-4 starts)

### System Design Recommendation:
- Implement a tiered warning system:
  - **Red Warning (Highly Unstable):** < 50% of the stabilization point. (e.g., < 30 PA for K%)
  - **Yellow Warning (Approaching Stability):** 50%-99% of the stabilization point.
  - **No Warning (Stabilized):** >= 100% of the stabilization point.
- For heavily variance-dependent stats like ERA and AVG, apply the "Halfway Rule" (regressing current metrics 50% towards league average before projecting future performance) instead of waiting 1.5 seasons for stability.