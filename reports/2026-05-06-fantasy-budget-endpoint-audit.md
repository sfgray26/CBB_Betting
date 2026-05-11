# Fantasy Budget Endpoint Audit

> **Date:** 2026-05-06 | **Analyst:** Kimi CLI (Deep Intelligence Unit)
> **Scope:** `/api/fantasy/budget`, `/api/fantasy/matchup`, `/api/fantasy/lineup/*`
> **Status:** Partially wired — acquisitions and IL are live; IP is mock

---

## Executive Summary

The backend has a **dedicated budget endpoint** (`/api/fantasy/budget`) and budget data is **embedded in the matchup scoreboard** (`/api/fantasy/matchup`). However, **innings pitched (IP) accumulation is completely mocked** across all endpoints, while **acquisitions used** and **IL slots used** are wired to live Yahoo data.

| Field | `/api/fantasy/budget` | `/api/fantasy/matchup` | `/api/fantasy/lineup/*` | Status |
|---|---|---|---|---|
| `acquisitions_used` | ✅ Live Yahoo | ❌ Hardcoded `5` | ❌ Not returned | Wired |
| `acquisitions_remaining` | ✅ Computed | ✅ Computed | ❌ Not returned | Wired |
| `acquisition_limit` | ✅ Hardcoded `8` | ✅ Hardcoded `8` | ❌ Not returned | Config |
| `il_used` | ✅ Live Yahoo | ❌ Hardcoded `1` | ❌ Not returned | Wired |
| `il_total` | ✅ Hardcoded `3` | ✅ Hardcoded `3` | ❌ Not returned | Config |
| `ip_accumulated` | ❌ **Mock `0.0`** | ❌ **Mock `45.0`** | ❌ Not returned | **BLOCKED** |
| `ip_minimum` | ❌ `90.0` | ❌ `90.0` | ❌ Not returned | **Inconsistent** |
| `ip_pace` | ✅ Computed | ✅ Computed | ❌ Not returned | Depends on IP |

**Recommendation:** Build the context strip using `/api/fantasy/budget` as the single source of truth. Use placeholders for `ip_accumulated` and `ip_pace` until IP wiring is complete. Do not source budget data from `/api/fantasy/matchup` (it uses stale hardcoded values).

---

## 1. Endpoint Audit

### 1.1 `/api/fantasy/budget` — Dedicated Budget Endpoint

**File:** `backend/routers/fantasy.py:5282`

This is the **correct source** for the context strip. It returns a `BudgetResponse` shape:

```json
{
  "budget": {
    "acquisitions_used": <int>,
    "acquisitions_remaining": <int>,
    "acquisition_limit": <int>,
    "acquisition_warning": <bool>,
    "il_used": <int>,
    "il_total": <int>,
    "ip_accumulated": <float>,
    "ip_minimum": <float>,
    "ip_pace": "BEHIND|ON_TRACK|AHEAD",
    "as_of": "2026-05-06T..."
  },
  "freshness": { ... }
}
```

**How values are populated:**

| Field | Source | Live? |
|---|---|---|
| `acquisitions_used` | `yahoo.get_transactions(t_type="add")` → `count_weekly_acquisitions()` | ✅ Live |
| `il_used` | `yahoo.get_roster()` — counts players with `selected_position in ["IL", "IL60"]` | ✅ Live |
| `ip_accumulated` | `ip_accumulated = 0.0` — **hardcoded mock** | ❌ Mock |
| `ip_minimum` | `ip_minimum = 90.0` — hardcoded | ⚠️ Config |
| `days_remaining` | `days_remaining = 6` — approximate | ⚠️ Approximate |
| `season_days_elapsed` | `season_days_elapsed = 1` — approximate | ⚠️ Approximate |

**Code excerpt (lines 5338–5339):**
```python
# 3. IP tracking - still mock (requires matchup week logic + stat aggregation)
ip_accumulated = 0.0  # TODO: Wire to player_rolling_stats or Yahoo matchup stats
```

### 1.2 `/api/fantasy/matchup` — Matchup Scoreboard (Embeds Budget)

**File:** `backend/routers/fantasy.py:5198`

The matchup endpoint calls `assemble_matchup_scoreboard()` with **hardcoded budget values**:

```python
assemble_matchup_scoreboard(
    ...,
    ip_accumulated=45.0,      # ← hardcoded
    ip_minimum=90.0,          # ← hardcoded
    acquisitions_used=5,      # ← hardcoded
    il_used=1,                # ← hardcoded
    ...
)
```

These values are **not live** and should not be used for the context strip. They are MVP placeholders from the original scoreboard implementation.

### 1.3 `/api/fantasy/lineup/*` — Lineup Endpoints

**Files:** `backend/routers/fantasy.py:1019`, `2710`, `3945`, `3975`, `4074`, `4096`

None of the lineup endpoints return budget data. The response shapes are:

- `GET /api/fantasy/lineup/{lineup_date}` → `DailyLineupResponse` (players, scores, game times)
- `POST /api/fantasy/lineup` → save confirmation
- `GET /api/fantasy/lineup/elite-optimize/{lineup_date}` → optimized lineup
- `POST /api/fantasy/lineup/analyze-scarcity` → scarcity analysis
- `POST /api/fantasy/lineup/compare-scoring` → scoring comparison

**No budget fields in any lineup response.**

---

## 2. IP Data Sources (Fix Options)

### Option A: Yahoo Matchup Stats (Recommended — P0)

The Yahoo client already fetches matchup stats including IP:

**File:** `backend/fantasy_baseball/yahoo_client_resilient.py:1169`

```python
def get_matchup_stats(self, week=None, my_team_key=None) -> dict:
    """Returns my_stats and opp_stats keyed by canonical code."""
```

The canonical mapping includes:
```python
yahoo_to_canonical = {
    ...
    "50": "IP",   # ← Innings Pitched (line 1261)
    ...
}
```

**Usage in `/api/fantasy/budget`:**
```python
matchup = client.get_matchup_stats(week=current_week)
ip_accumulated = matchup.get("my_stats", {}).get("IP", 0.0)
```

**Pros:** Single API call, Yahoo is the source of truth for H2H matchup stats.  
**Cons:** Requires knowing the current matchup week; returns 0.0 before the matchup starts.

### Option B: Aggregate from `mlb_player_stats` (P1 — fallback)

The database has `mlb_player_stats.innings_pitched` as a string (e.g., `"6.2"`):

```sql
SELECT
    SUM(
        CAST(SPLIT_PART(innings_pitched, '.', 1) AS NUMERIC) +
        CAST(SPLIT_PART(innings_pitched, '.', 2) AS NUMERIC) / 3.0
    ) as total_ip
FROM mlb_player_stats
WHERE game_date >= :matchup_week_start
  AND game_date <= :matchup_week_end
  AND bdl_player_id IN (:my_team_bdl_ids);
```

**Pros:** Works even if Yahoo API is down; aggregates across all pitchers on roster.  
**Cons:** Requires mapping Yahoo roster players → `bdl_player_id` (only ~18% coverage today per K-NEXT-2); lags by ~12 hours (box stats ingest at 2 AM ET).

### Option C: `player_rolling_stats` (P2 — season-long)

The `player_rolling_stats` table has `ip` as a float. Aggregate the 7-day window for rostered pitchers.

---

## 3. `ip_minimum` Inconsistency

| Location | Value | Context |
|---|---|---|
| `/api/fantasy/budget` | `90.0` | Comment says "Yahoo H2H standard" |
| `scoreboard_orchestrator.py:267` | `18.0` | Default in `compute_budget_state()` |
| Yahoo actual H2H | **None** | Yahoo H2H categories leagues have **no IP minimum** |
| Yahoo Roto | `900.0–1400.0` | Season-long minimum (league-dependent) |

**Problem:** `90.0` appears to be a weekly IP target, but `compute_budget_state()` treats it as a **season pace** benchmark (dividing by 182 days). A team with 0 IP on day 1 of the season is correctly flagged "BEHIND" relative to a 90 IP season target, but that's nonsensical — 90 IP is roughly one week's worth of pitching, not a season target.

**Fix:** Decide whether the context strip tracks:
- **Weekly matchup IP** (target ~14–21 IP per week) → use `days_total=7`
- **Season-long IP pace** (target ~1200 IP) → use `ip_minimum=1200.0`, `days_total=182`

---

## 4. Context Strip Implementation Plan

### Phase 1: Placeholders (Today)

Use `/api/fantasy/budget` as the source. Show placeholders for mocked fields:

```typescript
// Context strip component
const budget = await fetch("/api/fantasy/budget").then(r => r.json());

return (
  <BudgetStrip>
    <Badge color={budget.acquisition_warning ? "red" : "green"}>
      Adds: {budget.acquisitions_used}/{budget.acquisition_limit}
    </Badge>
    <Badge>
      IL: {budget.il_used}/{budget.il_total}
    </Badge>
    <Badge color={budget.ip_pace === "BEHIND" ? "amber" : "green"}>
      IP: {budget.ip_accumulated ?? "—"} / {budget.ip_minimum}
      {/* TODO: Remove "—" placeholder when IP is wired */}
    </Badge>
  </BudgetStrip>
);
```

### Phase 2: Wire IP (Next Sprint)

1. In `/api/fantasy/budget`, replace:
   ```python
   ip_accumulated = 0.0  # TODO
   ```
   with:
   ```python
   try:
       matchup = client.get_matchup_stats(week=current_week)
       ip_accumulated = matchup.get("my_stats", {}).get("IP", 0.0)
   except (YahooAuthError, YahooAPIError):
       ip_accumulated = 0.0
   ```

2. Fix `ip_minimum` semantics:
   - For weekly tracking: `ip_minimum = 14.0`, `days_total = 7`
   - For season tracking: `ip_minimum = 1200.0`, `days_total = 182`

3. Remove hardcoded values from `/api/fantasy/matchup` and call `/api/fantasy/budget` internally if budget data is needed there.

---

## Appendix: Schema Reference

### `ConstraintBudget` (`backend/contracts.py:220`)

```python
class ConstraintBudget(BaseModel):
    acquisitions_used: int
    acquisitions_remaining: int
    acquisition_limit: int
    acquisition_warning: bool
    il_used: int
    il_total: int
    ip_accumulated: float
    ip_minimum: float
    ip_pace: IPPaceFlag  # Enum: BEHIND, ON_TRACK, AHEAD
    as_of: datetime
```

### `BudgetResponse` (`backend/contracts.py:314`)

```python
class BudgetResponse(BaseModel):
    budget: ConstraintBudget
```
