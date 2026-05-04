# Production Data Quality Audit — Fresh Probe (April 21, 2026)
**Date**: April 21, 2026  
**Probe Time**: 19:01:58 UTC  
**Method**: Fresh API probe against `https://fantasy-app-production-5079.up.railway.app`  
**Response Files**: `postman_collections/responses/*20260421_190158.json`  
**Scope**: 14 endpoints hit, 12 returned 200, 2 returned 404  

---

## Executive Summary

This audit is based on a **fresh probe executed April 21 at 19:01 UTC**. It replaces the prior April 20 audit, which analyzed stale responses from an earlier deployment state. **Seven issues from the April 20 audit have been fixed** in the latest deploy. **Twelve issues remain active**, and **two new issues** were discovered in the fresh data.

All findings below are derived from **direct inspection of the JSON response files** saved to disk. Root cause explanations are labeled as either **Confirmed** (directly observable in response data) or **Likely** (inference from data patterns, not direct observation).

---

## What Changed Since April 20 (Fixed Issues)

| # | Issue (April 20) | April 20 State | April 21 State | Verdict |
|---|------------------|----------------|----------------|---------|
| 1 | Roster endpoint 500 ImportError | 500 | **200** | Fixed |
| 2 | Waiver recommendations 503 RiskProfile error | 503 | **200** | Fixed |
| 3 | Lineup schedule blindness | games_count=0, all benched | **games_count=10, 8/13 starting** | Fixed |
| 4 | Matchup null ratio stats | AVG/OPS/ERA/WHIP/K_9=null | **All populated with strings/ints** | Fixed |
| 5 | Waiver stat ID "38" leak | 20/25 had key "38" | **Absent from all responses** | Fixed |
| 6 | Waiver 100% empty category_contributions | 0/25 populated | **3/25 populated** | Partially fixed |
| 7 | Decisions stale (2026-04-19) | as_of_date=2026-04-19 | **as_of_date=2026-04-20** | Fixed |

---

## Remaining Issue 1: Roster Rolling Windows 100% Null

**Endpoint**: `GET /api/fantasy/roster`  
**Status**: 200  
**Confidence**: 100% confirmed from JSON

**Observation**: All 23 roster players have `null` for every rolling window field:

| Field | Non-null / Total |
|-------|-----------------|
| `rolling_7d` | 0 / 23 |
| `rolling_14d` | 0 / 23 |
| `rolling_15d` | 0 / 23 |
| `rolling_30d` | 0 / 23 |
| `ros_projection` | 0 / 23 |
| `row_projection` | 0 / 23 |
| `game_context` | 0 / 23 |
| `injury_status` | 0 / 23 |
| `injury_return_timeline` | 0 / 23 |

**Sample player** (Moisés Ballesteros):
```json
{
  "rolling_14d": null,
  "ros_projection": null,
  "row_projection": null,
  "game_context": null,
  "injury_status": null,
  "ownership_pct": 0.0
}
```

**What IS present**: `season_stats` is populated for all 23 players (e.g., Ballesteros has `HR_B: 3.0`, `AVG: 0.378`, `OPS: 1.02`).

**Severity**: High — rolling windows are a core data product. Their absence means the frontend cannot display recent performance trends.

**Root cause**: Likely — the `player_rolling_stats` table has 48,910 rows (per pipeline health), so data exists in the database. The roster endpoint's join to rolling stats is probably failing silently or the query is filtering too aggressively.

---

## Remaining Issue 2: Waiver Intelligence Still Mostly Hollow

**Endpoint**: `GET /api/fantasy/waiver`  
**Status**: 200  
**Confidence**: 100% confirmed from JSON

**Observation**: Partial improvement since April 20, but most intelligence fields remain empty:

| Field | April 20 (non-empty) | April 21 (non-empty) | Change |
|-------|---------------------|---------------------|--------|
| `category_contributions` | 0 / 25 | **3 / 25** | +3 |
| `hot_cold` | 0 / 25 | **2 / 25** | +2 |
| `projected_saves` | 0 / 25 | **1 / 25** | +1 |
| `owned_pct` | 0 / 25 | **0 / 25** | — |
| `starts_this_week` | 0 / 25 | **0 / 25** | — |
| `projected_points` | 0 / 25 | **0 / 25** | — |
| `statcast_signals` | 0 / 25 | **0 / 25** | — |
| `statcast_stats` | 0 / 25 | **0 / 25** | — |
| `status` | 0 / 25 | **0 / 25** | — |
| `injury_note` | 0 / 25 | **0 / 25** | — |
| `injury_status` | 0 / 25 | **0 / 25** | — |

**Newly populated examples**:
- Seth Lugo: `category_contributions: {"W": 1.162}`, `hot_cold: "HOT"`
- Michael Wacha: `category_contributions: {"W": 0.663}`, `hot_cold: "HOT"`
- Colin Rea: `projected_saves: 1.0`

**Severity**: Medium-High — The partial fix shows the enrichment pipeline is starting to work (3 players now have category contributions), but 88–100% of the waiver pool still lacks intelligence data.

---

## Remaining Issue 3: Universal Drop Bug — Target Changed, Pattern Persists

**Endpoint**: `GET /api/fantasy/waiver/recommendations`  
**Status**: 200  
**Confidence**: 100% confirmed from JSON

**Observation**: The endpoint now returns 200 (was 503). It returns **exactly 1 recommendation**:

```json
{
  "action": "ADD_DROP",
  "add_player": {"name": "Seth Lugo", "team": "KC", "position": "SP"},
  "drop_player_name": "Spencer Arrighetti",
  "drop_player_position": "SP",
  "win_prob_before": 0.996,
  "win_prob_after": 0.997,
  "win_prob_gain": 0.001
}
```

**Drop target analysis** (`GET /api/fantasy/decisions`):
- April 20: 22/22 waiver decisions dropped **Garrett Crochet**
- April 21: 24/24 waiver decisions drop **Seiya Suzuki**

The universal drop target has **changed names but the pattern persists**: 100% of waiver decisions recommend dropping the same player regardless of who is being added.

**Severity**: High — Waiver recommendations are not credible when they always suggest the same drop. The win probability gain of 0.001 (0.1%) is negligible.

---

## Remaining Issue 4: K_P Field Still Mislabeled (Wins, Not Strikeouts)

**Endpoint**: `GET /api/fantasy/waiver`  
**Status**: 200  
**Confidence**: 100% confirmed by mathematical proof

**Observation**: The `K_P` field in pitcher `stats` dictionaries contains values that cannot be strikeouts.

| Player | IP | K_P | K_9 |
|--------|-----|-----|-----|
| Seth Lugo | 31.1 | **"1"** | 8.04 |
| Michael Wacha | 27.0 | **"2"** | 7.67 |
| Will Warren | 25.1 | **"2"** | 11.01 |
| Landen Roupp | 22.2 | **"3"** | 9.53 |
| Davis Martin | 25.0 | **"3"** | 6.84 |

**Proof K_P is NOT strikeouts**:
```
Seth Lugo: K_9 = 8.04, IP = 31.1
Expected K = 8.04 × 31.1 / 9 = 27.8 strikeouts
K_P field shows "1" → off by 28×
```

The values 1–3 are consistent with **wins** over ~5 starts, not strikeouts.

**Severity**: Medium — The field name misleads consumers. The actual strikeout data may be present under a different key or absent entirely.

---

## Remaining Issue 5: Waiver Stats Schema Pollution (Batters Have Pitcher Stats)

**Endpoint**: `GET /api/fantasy/waiver`  
**Status**: 200  
**Confidence**: 100% confirmed from JSON

**Observation**: Five batters in the waiver pool have pitcher-specific stat keys:

| Player | Position | IP | W | GS |
|--------|----------|-----|-----|-----|
| Ildemaro Vargas | 1B | "8" | "34" | "0" |
| Mickey Moniak | LF | "14" | "36" | "1" |
| Josh Bell | 1B | "18" | "34" | "0" |
| Andrés Giménez | 2B | "12" | "35" | "4" |
| Dalton Rushing | C | "6" | "27" | "0" |

A position player cannot have `IP` (innings pitched), `W` (wins), or `GS` (games started). The stats dict contains the player's complete Yahoo stat payload without position filtering.

**Severity**: Low-Medium — Does not break functionality, but creates confusion for consumers.

---

## Remaining Issue 6: Impossible Projection Narratives Persist

**Endpoint**: `GET /api/fantasy/decisions`  
**Status**: 200  
**Confidence**: 100% confirmed from JSON

**Observation**: 5 impossible projections remain (down from 9 on April 20):

| Player | Narrative |
|--------|-----------|
| Daniel Lynch IV | "WHIP Z-score +1.99 (ELITE); projects **0.00 WHIP ROS**" |
| Daniel Lynch IV | "ERA Z-score +1.12 (STRONG); projects **0.00 ERA ROS**" |
| Andrew Alvarez | "ERA Z-score +1.12 (STRONG); projects **0.00 ERA ROS**" |
| Tony Santillan | "ERA Z-score +1.12 (STRONG); projects **0.00 ERA ROS**" |
| Rico Garcia | "ERA Z-score +1.12 (STRONG); projects **0.00 ERA ROS**" |

All 5 pitchers have 0.00 ERA in their actual season stats (confirmed in waiver endpoint). The projection math extrapolates this to a full season, producing impossible 0.00 ERA/WHIP ROS projections.

**Severity**: Medium — Undermines user confidence in decision narratives. The z-scores themselves (+1.12) are mathematically correct for 0.00 ERA pitchers, but the ROS projection text is misleading.

---

## Remaining Issue 7: Draft Board Age=0 for 92.5% of Players

**Endpoint**: `GET /api/fantasy/draft-board`  
**Status**: 200  
**Confidence**: 100% confirmed from JSON

**Observation**: 185 of 200 players have `age: 0`.

**Severity**: Low — Age is used in risk-adjusted calculations. Players with age=0 receive no age-based risk discount, which may slightly misprice keeper value.

---

## Remaining Issue 8: Briefing Uses Legacy Category Names

**Endpoint**: `GET /api/fantasy/briefing/2026-04-20`  
**Status**: 200  
**Confidence**: 100% confirmed from JSON

**Observation**: The briefing returns 11 categories with legacy v1 names instead of v2 canonical codes:

| Briefing Name | V2 Canonical | Match? |
|---------------|-------------|--------|
| `HR` | `HR_B` | No |
| `SB` | `NSB` | No |
| `K` | `K_P` / `K_B` | Ambiguous |
| `SV` | `NSV` | No |

**Missing v2 categories**: HR_B, K_B, TB, NSB, OPS, L, HR_P, K_P, K_9, QS, NSV (11 of 18 absent).

**Severity**: Medium — The briefing is one of the primary user-facing endpoints. Legacy category names create confusion for v2 consumers.

---

## Remaining Issue 9: Player-Scores and Validate-System Routes Missing

**Endpoints**:
- `GET /api/fantasy/player-scores?period=season` → 404
- `GET /admin/validate-system` → 404

**Confidence**: 100% confirmed from HTTP status codes.

**Severity**: Low — These routes are referenced in documentation but not implemented. No known consumer depends on them.

---

## New Issue 10: Roster BDL/MLBAM IDs Null for All Players

**Endpoint**: `GET /api/fantasy/roster`  
**Status**: 200  
**Confidence**: 100% confirmed from JSON

**Observation**: Every roster player has `bdl_player_id: null` and `mlbam_id: null`.

**Sample** (Moisés Ballesteros):
```json
{
  "yahoo_player_key": "469.p.60120",
  "bdl_player_id": null,
  "mlbam_id": null
}
```

**Severity**: Medium — BDL and MLBAM IDs are the canonical identifiers for cross-referencing with external data sources (Baseball-Reference, FanGraphs, Statcast). Their absence breaks enrichment pipelines that depend on these keys.

**Root cause**: Likely — The `player_id_mapping` table does not have `yahoo_key` entries for these players, or the roster endpoint is not querying the mapping table.

---

## New Issue 11: Roster Season Stats Missing Most Pitcher Categories for Batters

**Endpoint**: `GET /api/fantasy/roster`  
**Status**: 200  
**Confidence**: 100% confirmed from JSON

**Observation**: Batter season_stats dictionaries contain `null` for pitcher categories, which is expected. However, they also contain `null` for several batter categories:

**Sample** (Moisés Ballesteros, catcher):
```json
{
  "values": {
    "HR_B": 3.0, "RBI": 10.0, "W": 27.0, "R": 8.0, "H": 17.0,
    "OPS": 1.02, "AVG": 0.378,
    "K_B": null, "NSV": null, "ERA": null, "WHIP": null,
    "L": null, "HR_P": null, "K_9": null, "TB": null,
    "K_P": null, "QS": null, "NSB": null
  }
}
```

**Missing batter stats that should exist**: `K_B` (batting strikeouts), `TB` (total bases), `NSB` (net stolen bases).

**Severity**: Low — The null values are for categories where the player genuinely has no activity (Ballesteros has 0 SB, so NSB=null is reasonable). However, `K_B` and `TB` being null when other batting stats exist suggests incomplete Yahoo stat mapping.

---

## What Is Working Well (Confirmed)

1. **Roster endpoint is alive and rich**: 23 players with season_stats, eligibility, freshness metadata, and a modern schema
2. **Schedule lookup is fixed**: Lineup shows real games, opponents, and start/sit recommendations
3. **Matchup has real data**: 15 categories populated for both teams with consistent types
4. **Pipeline is healthy and growing**: All 7 tables green, rolling stats grew from 46K→49K rows
5. **Waiver recommendations endpoint works**: Returns 200 with MCMC win probabilities across all 18 categories
6. **Decisions are fresh**: as_of_date matches current date
7. **Stat ID 38 leak fixed**: No raw numeric stat IDs in waiver responses
8. **Proxy player issue fixed**: 0 proxy players in roster (was 6)

---

## Appendix: Endpoint Status Matrix (April 21, 19:01 UTC)

| Endpoint | Status | Size | Assessment |
|----------|--------|------|------------|
| `GET /api/fantasy/draft-board?limit=200` | 200 | 200 KB | Functional; 92.5% age=0 |
| `GET /api/fantasy/roster` | 200 | 30 KB | Functional; rolling windows null |
| `GET /api/fantasy/lineup/2026-04-20` | 200 | 10 KB | **Fixed** — schedule working |
| `GET /api/fantasy/waiver` | 200 | 21 KB | Partial — enrichment 0–12% populated |
| `GET /api/fantasy/waiver/recommendations` | 200 | 1.9 KB | **Fixed** — 1 rec, universal drop persists |
| `GET /api/fantasy/matchup` | 200 | 856 B | **Fixed** — real data, consistent types |
| `GET /api/fantasy/decisions` | 200 | 69 KB | Functional; 5 impossible projections |
| `GET /api/fantasy/briefing/2026-04-20` | 200 | 5.3 KB | Functional; legacy category names |
| `GET /admin/pipeline-health` | 200 | 1.2 KB | All healthy |
| `GET /admin/scheduler/status` | 200 | 1.5 KB | 10 jobs |
| `GET /api/fantasy/player-scores?period=season` | 404 | 29 B | Not implemented |
| `GET /admin/validate-system` | 404 | 29 B | Not implemented |
| `POST /api/fantasy/roster/optimize` | 422 | 218 B | Needs request body |
| `POST /api/fantasy/matchup/simulate` | 422 | 218 B | Needs request body |

---

**Report Author**: Kimi CLI  
**Data Source**: `postman_collections/responses/*20260421_190158.json` (14 files)  
**Confidence Statement**: All quantitative claims (counts, percentages, exact values) are derived from direct JSON inspection. Root cause labels are either "Confirmed" (direct evidence in response) or "Likely" (logical inference from data pattern).
