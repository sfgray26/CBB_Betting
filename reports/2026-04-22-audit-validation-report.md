# Audit Validation Report — v5 Probe Critique & Correction
**Date**: April 22, 2026  
**Validation Probe**: 10:27 UTC (fresh responses, old files deleted first)  
**Auditor**: Kimi CLI (self-critique)  

---

## 1. Methodology Corrections

### A. POST Endpoints Called Without Request Bodies

**Flaw in v5 probe**: The original `api_probe.py` sent empty POST requests to `/api/fantasy/roster/optimize` and `/api/fantasy/matchup/simulate`. Both endpoints require JSON payloads:

- `roster/optimize` expects `RosterOptimizeRequest` → `{"target_date": "YYYY-MM-DD"}`
- `matchup/simulate` expects `MatchupSimulateRequest` → `{"my_roster": [...], "opponent_roster": [...], "n_sims": int, "week": str}`

**Result**: The 422 errors were **false positives caused by the probe, not application bugs**.

**Corrected behavior** (verified at 10:27 UTC):
- `POST /api/fantasy/roster/optimize` with `{"target_date": "2026-04-22"}` → **200** (4,317 bytes)
- `POST /api/fantasy/matchup/simulate` with `{"my_roster": [], "opponent_roster": [], "n_sims": 100, "week": "2026-W17"}` → **200** (150 bytes)

### B. No Server-Side Traceback Available

**Flaw in v5 probe**: The API only returns a 96-byte JSON error body (`{"detail": "Unexpected error: ..."}`). I falsely claimed the error was in a "secondary sorting key" without proof. Production APIs do not expose stack traces to clients.

**Correction**: I performed a **code-level root cause analysis** by reading the source code of the waiver recommendations endpoint, the `waiver_edge_detector.py` module, and the relevant Pydantic schemas. The exact line of failure was identified through static analysis, not runtime traceback.

### C. No Cache Confusion

All prior response files in `postman_collections/responses/` were deleted with `Remove-Item * -Recurse -Force` before the 10:27 probe. Every byte in the analysis below comes from files timestamped `20260422_102700`.

---

## 2. The 503 Traceback Analysis

### Verified: The 503 Is a Real, Reproducible Bug

**Endpoint**: `GET /api/fantasy/waiver/recommendations`  
**Status**: 503 (both 09:01 and 10:27 probes)  
**Error message**:
```json
{"detail": "Unexpected error: '>' not supported between instances of 'float' and 'tuple'"}
```

### Root Cause (>99% Confidence)

The error originates in `backend/routers/fantasy.py`, inside the `/api/fantasy/waiver/recommendations` endpoint, at approximately **line 2275**:

```python
drop_score_adj = max(
    _drop_candidate_value(drop_candidate),
    drop_candidate["z_score"] + statcast_need_score_boost(drop_signals),
)
```

**Why it fails**:
1. `_drop_candidate_value` is imported from `backend/services/waiver_edge_detector.py`.
2. In `waiver_edge_detector.py` at **line 115**, the function signature is:
   ```python
   def drop_candidate_value(player: dict) -> tuple:
   ```
   It returns a **5-element tuple**: `(primary_score, neg_tier, adp, neg_owned_pct, name_hash)` for deterministic ranking.
3. The second argument to `max()` is a **float**: `drop_candidate["z_score"] + statcast_need_score_boost(...)`.
4. `max(tuple, float)` attempts a direct comparison between incompatible types, raising exactly the error observed in production.

**Why the `_safe_need_score` fix did not help**:
- The `_safe_need_score()` helper (added by Claude at lines 2381-2392) guards the **final** `sorted()` call at line 2397.
- The tuple-vs-float comparison crashes **before** `_safe_need_score` is ever reached, inside the recommendation-building loop.

**Fix required**: Extract the scalar primary score from the tuple before passing it to `max()`, e.g.:
```python
drop_score_adj = max(
    drop_candidate_value(drop_candidate)[0],  # scalar primary_score
    drop_candidate["z_score"] + statcast_need_score_boost(drop_signals),
)
```

---

## 3. Verified False Positives

These items were reported as "broken" in v5 but are **confirmed to be probe artifacts or misinterpretations**:

| # | v5 Claim | Reality | Evidence |
|---|----------|---------|----------|
| 1 | `POST /api/fantasy/roster/optimize` returns 422 | **Returns 200** with valid payload | 10:27 probe: 200, 4,317 bytes, 13 starters + 5 bench assignments returned |
| 2 | `POST /api/fantasy/matchup/simulate` returns 422 | **Returns 200** with valid payload | 10:27 probe: 200, 150 bytes, `{"win_prob": 0.5, ...}` |
| 3 | 422 endpoints are "persistent errors" | **False** — they require POST bodies | Pydantic `MatchupSimulateRequest` and `RosterOptimizeRequest` schemas confirm required fields |

---

## 4. Verified True Negatives (Confirmed Real Bugs)

These are **hard-confirmed** through direct inspection of fresh JSON response files and/or source code static analysis. Each claim is reproducible.

### 4.1 Waiver Recommendations 503 (TypeError)
- **Status**: Confirmed real bug
- **Evidence**: 503 at 09:01 and 10:27 with identical error message
- **Root cause**: `max(tuple, float)` in `backend/routers/fantasy.py` ~line 2275
- **Confidence**: >99%

### 4.2 Roster — ROS/ROW Projections 100% Absent
- **Status**: Confirmed real data gap
- **Evidence**: All 23 players have `ros_projection: null` and `row_projection: null` in the raw JSON
- **JSON path verified**: `players[N].ros_projection` → `NoneType` for N=0..22
- **Confidence**: 100%

### 4.3 Roster — Game Context 100% Absent
- **Status**: Confirmed real data gap
- **Evidence**: All 23 players have `game_context: null` in the raw JSON
- **Confidence**: 100%

### 4.4 Roster — Rolling 15-Day Window 100% Absent
- **Status**: Confirmed real data gap
- **Evidence**: All 23 players have `rolling_15d: null` in the raw JSON
- **Confidence**: 100%

### 4.5 Roster — Ownership Percentage Universally Zero
- **Status**: Confirmed real data gap
- **Evidence**: All 23 players have `ownership_pct: 0.0` (including stars like Pete Alonso and Blake Snell)
- **Confidence**: 100%

### 4.6 Roster — Freshness `fetched_at` 100% Null
- **Status**: Confirmed real data gap
- **Evidence**: All 23 players have `freshness.fetched_at: null`
- **Confidence**: 100%

### 4.7 Blake Snell — Completely Null Season Stats
- **Status**: Confirmed real data gap
- **Evidence**: Blake Snell (`eligible_positions: ["SP", "P", "IL"]`) has all 18 `season_stats.values` fields set to `null`
- **Confidence**: 100%

### 4.8 Waiver Intelligence Layer Empty
- **Status**: Confirmed real data gap
- **Evidence** (25 available players):
  - `owned_pct`: 25/25 = 0.0
  - `starts_this_week`: 24/25 = 0
  - `statcast_signals`: 25/25 = `[]`
  - `hot_cold`: 24/25 = null
  - `status`: 25/25 = null
  - `injury_note`: 25/25 = null
  - `injury_status`: 25/25 = null
  - `statcast_stats`: 25/25 = null
  - `category_contributions`: 23/25 = `{}`
- **Confidence**: 100%

### 4.9 Decisions Missing Core Fields
- **Status**: Confirmed real data gap
- **Evidence** (14 decisions):
  - `value_gain`: 14/14 null
  - `risk_narrative`: 14/14 null
  - `track_record_narrative`: 14/14 null
  - `as_of_date`: all `2026-04-21` (not updated to 04-22)
  - 0 waiver decisions (all are lineup)
- **Confidence**: 100%

### 4.10 Draft Board Age=0 for 92.5%
- **Status**: Confirmed real data gap
- **Evidence**: 185 of 200 players have `age: 0`
- **Confidence**: 100%

### 4.11 Persistent 404 Endpoints
- **Status**: Confirmed real (not probe artifacts)
- **Evidence**:
  - `GET /api/fantasy/player-scores?period=season` → 404 `{"detail": "Not Found"}`
  - `GET /admin/validate-system` → 404 `{"detail": "Not Found"}`
- **Confidence**: 100%

### 4.12 Three Pitchers with ERA=0.00
- **Status**: Confirmed in data (mathematically possible, noteworthy)
- **Evidence**:
  - Louis Varland: ERA 0.00, IP 13.0
  - Tony Santillan: ERA 0.00, IP 10.0
  - Huascar Brazobán: ERA 0.00, IP 10.2
- **Confidence**: 100%

### 4.13 Lineup — Pitcher `lineup_status` and `eligible_positions` Still Null
- **Status**: Confirmed real data gap
- **Evidence**: All 10 pitchers have `lineup_status: null` and `eligible_positions: null`
- **Confidence**: 100%

---

## 5. Summary Table

| Finding | v5 Reported | Validated Status | Category |
|---------|-------------|------------------|----------|
| Waiver recommendations 503 | Broken | **Real bug** | True Negative |
| Roster/optimize 422 | Broken | **False positive** (missing payload) | False Positive |
| Matchup/simulate 422 | Broken | **False positive** (missing payload) | False Positive |
| ROS/ROW projections null | Missing | **Real data gap** | True Negative |
| game_context null | Missing | **Real data gap** | True Negative |
| rolling_15d null | Missing | **Real data gap** | True Negative |
| ownership_pct 0.0 | Missing | **Real data gap** | True Negative |
| Blake Snell stats null | Missing | **Real data gap** | True Negative |
| Waiver intel empty | Missing | **Real data gap** | True Negative |
| Decisions value_gain null | Missing | **Real data gap** | True Negative |
| Draft board age=0 | Missing | **Real data gap** | True Negative |
| Player-scores 404 | Broken | **Real** (endpoint missing) | True Negative |
| Validate-system 404 | Broken | **Real** (endpoint missing) | True Negative |
| Lineup pitcher status null | Broken | **Real data gap** | True Negative |

---

**Methodology**: All claims verified against fresh JSON response files from the 10:27 UTC probe, supplemented by static source code analysis where runtime tracebacks were unavailable. No claim relies on inference beyond what the code and response files directly reveal.
