# K-19: In-Season Projection Pipeline Architecture Spec

**Date:** April 1, 2026  
**Analyst:** Kimi CLI (Deep Intelligence Unit)  
**Status:** Architecture decisions for Claude Code implementation

---

## 1. FanGraphs RoS Downloads

**Public URL Pattern:** `https://www.fangraphs.com/projections.aspx?pos=all&stats={type}&type={system}&team=0&lg=all&players=0`

| System | Type Param | Reliability | Notes |
|--------|-----------|-------------|-------|
| Steamer RoS | `steamerr` | High | Updates daily ~3 AM ET |
| ZiPS RoS | `zipsdc` | High | Combines ZiPS + Depth Charts |
| ATC RoS | `atc` | Very High | Ensemble of all systems |
| THE BAT RoS | `thebat` | High | Power-focused |
| Depth Charts | `depthcharts` | Medium | Playing time estimates |

**Column Names (Steamer RoS Batting):**
`Name, Team, G, PA, AB, H, 2B, 3B, HR, R, RBI, BB, SO, HBP, SF, AVG, OBP, SLG, OPS, wOBA, wRC+, BsR, Off, Def, WAR, ADP`

**Column Names (Steamer RoS Pitching):**
`Name, Team, W, L, ERA, G, GS, IP, H, ER, HR, BB, SO, WHIP, K/9, BB/9, K/BB, H/9, HR/9, AVG, BABIP, LOB%, GB%, HR/FB, FIP, xFIP, WAR, ADP`

**Rate Limiting:** FanGraphs uses Cloudflare. No documented rate limit, but be polite (1 req/sec). Use `cloudscraper` library as fallback if `requests` gets 403.

**Confidence:** HIGH — URLs are public, columns match existing `projections_loader.py` expectations.

**Blocker:** None. No API key required for export pages.

---

## 2. Ensemble Blender Design

### Recommended Weight Schedule (Decaying) — CORRECTED

**CRITICAL DOMAIN CORRECTION:** Statcast provides underlying performance metrics (xwOBA, Barrel%, xERA), NOT counting stat projections. You cannot mathematically average a player's xBA with their Steamer projected Home Runs. 

**Use only for projection systems that output counting stats:**

| Week | Pre-Season | RoS Ensemble (Counting Stats) | Statcast Trend Modifier | Z-Score Adj |
|------|-----------|------------------------------|------------------------|-------------|
| 1-2 | 40% | 55% (ATC 30%, Steamer 15%, ZiPS 10%) | Use as boost only | 5% |
| 3-4 | 20% | 65% (ATC 35%, Steamer 15%, ZiPS 15%) | Boost if Barrel% elite | 15% |
| 5-8 | 10% | 65% (ATC 35%, Steamer 15%, ZiPS 15%) | Boost if xwOBA > .380 | 25% |
| 9+ | 5% | 60% (ATC 30%, Steamer 15%, ZiPS 15%) | Significant weight | 35% |

**How to use Statcast correctly:**
- If projection says 25 HR but Barrel% is 95th percentile → boost to 28-30 HR
- If projection says .250 AVG but xBA is .280 → boost to .265-.270 AVG
- Use xERA vs ERA difference to adjust pitcher W/K projections

### Minimal Schema Change

Add columns to `PlayerDailyMetric` (no new table):

```python
# Add to backend/models.py PlayerDailyMetric class
ensemble_woba = Column(Float)           # Blended projection
ensemble_woba_uncertainty = Column(Float)  # Std dev across systems
prior_weight_applied = Column(Float)    # What % pre-season was used
data_freshness = Column(DateTime)       # Last ensemble update
source_weights = Column(JSON)           # {"atc": 0.30, "steamer": 0.20, ...}
```

**Why this table:** `PlayerDailyMetric` already stores daily player data. Adding ensemble projections here keeps the daily grain consistent and avoids JOIN complexity.

**Confidence:** HIGH — Simple additive schema change, no migration conflicts.

---

## 3. Yahoo ADP/Injury Polling

### Endpoints

Use existing `yahoo_client_resilient.py` base class — it handles OAuth refresh automatically.

| Data | Method | Endpoint Pattern | Pagination |
|------|--------|------------------|------------|
| Free Agents (with ADP) | `get_free_agents()` | `/league/{league_id}/players;status=A;start={start};count={count}` | `start` param, 25 per page |
| Injury Status | `get_free_agents()` (same) | Injury data in `status` field: `DTD`, `IL10`, `IL60` | Same as above |
| Player Metadata | Player resource | `/player/{player_key}/metadata` | N/A |

### Implementation

Extend existing `YahooFantasyClient`:

```python
def get_all_free_agents_with_adp(self, league_id: str) -> list[dict]:
    """Poll all free agents with ADP and injury status."""
    all_players = []
    start = 0
    count = 25
    while True:
        batch = self.get_free_agents(position="", start=start, count=count)
        if not batch:
            break
        all_players.extend(batch)
        if len(batch) < count:
            break
        start += count
    return all_players
```

**Storage:** Cache in new `yahoo_player_cache` table (not `player_board`):

```python
class YahooPlayerCache(Base):
    __tablename__ = "yahoo_player_cache"
    id = Column(Integer, primary_key=True)
    player_key = Column(String, index=True)  # Yahoo's player key
    player_name = Column(String)
    adp = Column(Float)
    injury_status = Column(String)  # ACTIVE, DTD, IL10, etc.
    injury_note = Column(Text)
    percent_owned = Column(Integer)
    percent_started = Column(Integer)
    updated_at = Column(DateTime, default=datetime.utcnow)
```

**Confidence:** HIGH — Extends existing proven pattern.

---

## 4. Statcast UTC Bug Fix Scope

### Confirmed Issue

**File:** `backend/fantasy_baseball/statcast_ingestion.py`  
**Line 726:** `target_date = date.today() - timedelta(days=1)`

**Problem:** `date.today()` returns local server date. On UTC servers during EDT evening hours (6 PM - midnight ET = 10 PM - 4 AM UTC), "yesterday" ET is actually "today" UTC, causing duplicate processing.

### Fix (Exact)

```python
# Line 726 BEFORE:
if target_date is None:
    target_date = date.today() - timedelta(days=1)

# Line 726 AFTER:
if target_date is None:
    from zoneinfo import ZoneInfo
    et_now = datetime.now(ZoneInfo("America/New_York"))
    target_date = (et_now - timedelta(days=1)).date()
```

### Other Date Anchors in File

| Line | Current | Issue? | Fix Needed |
|------|---------|--------|------------|
| 528 | `end_date = date.today()` | YES | Use ET anchor for consistency |
| 726 | `date.today() - timedelta(days=1)` | YES | ET anchor (above) |
| 801 | Comment only | No | N/A |

**Confidence:** HIGH — Two locations need ET anchor fix.

---

## 5. Lock ID Confirmation

### Current Lock IDs (HANDOFF.md §4)

```python
LOCK_IDS = {
    "mlb_odds":    100_001,  # Used
    "statcast":    100_002,  # Used
    "rolling_z":   100_003,  # Used
    "cbb_ratings": 100_004,  # Used
    "clv":         100_005,  # Used
    "cleanup":     100_006,  # Used
    "waiver_scan": 100_007,  # Used
    "mlb_brief":   100_008,  # Used
    "openclaw_perf":  100_009,  # Used
    "openclaw_sweep": 100_010,  # Used
    "valuation_cache": 100_011,  # Used
}
```

### Proposed New Lock IDs

| Lock ID | Job Name | Conflict? |
|---------|----------|-----------|
| 100_012 | `fangraphs_ros` | ✅ Available |
| 100_013 | `yahoo_adp_injury` | ✅ Available |
| 100_014 | `ensemble_update` | ✅ Available |
| 100_015 | `projection_freshness_check` | ✅ Available |

**Next available after these:** 100_016

**Confidence:** HIGH — No conflicts with existing locks.

---

## Summary for Claude Code

| # | Question | Answer | Confidence |
|---|----------|--------|------------|
| 1 | FanGraphs URLs | Public, `projections.aspx?type={steamerr,zipsdc,atc,thebat,depthcharts}`, no auth | HIGH |
| 2 | Ensemble schema | Add 5 columns to `PlayerDailyMetric`, decaying weights (see table) | HIGH |
| 3 | Yahoo ADP | Extend `YahooFantasyClient`, use `get_free_agents()` pagination, cache to `yahoo_player_cache` table | HIGH |
| 4 | Statcast UTC fix | Replace `date.today()` with ET anchor at lines 528 and 726 | HIGH |
| 5 | Lock IDs | 100_012-100_015 are available, no conflicts | HIGH |

**Blockers:** None. All architectural decisions are ready for implementation.

---

## Errata: Domain Knowledge Corrections (Post-Review)

The following corrections were identified during expert review by an elite fantasy baseball manager:

### 1. Statcast in Ensemble Blender (Section 2)

**Original Error:** Proposed blending Statcast (15% weight) with projection systems.

**Correction:** Statcast provides **underlying metrics** (xwOBA, Barrel%, xERA), NOT counting stat projections. You cannot average xBA with projected HRs.

**Correct Usage:** Use Statcast as a **trend modifier** — boost projections if underlying metrics are elite.

### 2. PROBABLE Status (Not in this report, but related)

**Note:** In baseball, `PROBABLE` means "Probable Pitcher" (scheduled to start today), NOT an injury designation. Map to Green/"STARTING" badge, NOT Yellow/Warning.

### 3. Negative Stats Validation (Not in this report, but related)

**Note:** Net Stolen Bases (NSB) CAN be negative (0 SB - 1 CS = -1). Only clamp stats where negative is mathematically impossible (GS, HR, RBI).

---

*Spec complete. Ready for Claude Code implementation. All 5 architectural questions answered with high confidence.*
