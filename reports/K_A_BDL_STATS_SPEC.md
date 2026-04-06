# K-A Research: BDL /mlb/v1/stats Endpoint Specification

**Date:** 2026-04-01  
**Researcher:** Kimi CLI  
**Status:** Complete — Awaiting Schema Verification  
**Research embargo:** NO production code until specs approved by Claude.

---

## 1. Executive Summary

BDL offers a `/mlb/v1/stats` endpoint for per-game player statistics that is **functionally equivalent** to the existing `/ncaab/v1/player_season_stats` pattern. However, **schema verification is REQUIRED** before implementation.

The endpoint appears to support:
- **Per-game box scores** (not just season aggregates)
- **Pagination** via cursor (same pattern as other BDL endpoints)
- **Filtering** by `player_ids[]`, `team_ids[]`, and `dates[]`

**Critical Dependency:** This spec is linked to `K_B_IDENTITY_RESOLUTION_SPEC.md` — the presence/absence of mlbam_id in BDL stats responses determines the complexity of identity resolution.

---

## 2. Endpoint Specification

### 2.1 HTTP Contract

```
GET /mlb/v1/stats
Authorization: Bearer {BALLDONTLIE_API_KEY}
Content-Type: application/json
```

**Note:** Authorization uses bare key (no "Bearer" prefix) — same as other BDL endpoints.

### 2.2 Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `player_ids[]` | int[] | No | Filter by BDL player.id(s) |
| `team_ids[]` | int[] | No | Filter by BDL team.id(s) |
| `dates[]` | string[] | No | Filter by date(s) in ISO 8601 format (YYYY-MM-DD) |
| `season` | int | No | Season year (defaults to current) |
| `cursor` | string | No | Pagination cursor |
| `per_page` | int | No | Items per page (default: 25, max: 100) |

### 2.3 Response Structure (PENDING VERIFICATION)

Based on BDL pattern consistency from `SCHEMA_DISCOVERY.md` and `/ncaab/v1/player_season_stats`:

```json
{
  "data": [
    {
      "id": 12345,
      "player": { /* MLBPlayer object */ },
      "team": { /* MLBTeam object */ },
      "game": { /* MLBGame stub or game_id */ },
      "date": "2026-03-30",
      "ab": 4,
      "r": 1,
      "h": 2,
      "double": 1,
      "triple": 0,
      "hr": 0,
      "rbi": 1,
      "bb": 1,
      "so": 0,
      "sb": 0,
      "cs": 0,
      "avg": ".285",
      "obp": ".342",
      "slg": ".412",
      "ops": ".754",
      "ip": null,
      "h_allowed": null,
      "r_allowed": null,
      "er": null,
      "bb_allowed": null,
      "k": null,
      "whip": null,
      "era": null
    }
  ],
  "meta": {
    "next_cursor": "abc123",
    "per_page": 25
  }
}
```

### 2.4 Critical Unknowns (TO BE VERIFIED)

1. **Does `player` object include `mlbam_id`?** — Determines identity resolution complexity (see K-B spec)
2. **What is the exact field naming convention?** (e.g., `double` vs `doubles`, `k` vs `strikeouts`)
3. **Are pitching and batting stats unified or separate endpoints?**
4. **What date range can be queried?** (single day only? historical range?)
5. **Are rate stats (avg, obp, slg) returned as strings or floats?**

---

## 3. Implementation Pattern (Proposed)

### 3.1 New Contract: `MLBPlayerStats`

```python
# backend/data_contracts/mlb_player_stats.py

from typing import Optional
from pydantic import BaseModel, ConfigDict
from backend.data_contracts.mlb_player import MLBPlayer
from backend.data_contracts.mlb_team import MLBTeam

class MLBPlayerStats(BaseModel):
    """Per-game player statistics from BDL /mlb/v1/stats endpoint."""
    model_config = ConfigDict(strict=True)
    
    id: int                      # BDL stats record ID
    player: MLBPlayer            # Nested player object (BDL id, name, etc.)
    team: MLBTeam                # Team at time of game
    game_id: Optional[int] = None  # BDL game.id
    date: str                    # ISO 8601 date
    
    # Batting stats (nullable for pitchers who don't bat)
    ab: Optional[int] = None
    r: Optional[int] = None
    h: Optional[int] = None
    double: Optional[int] = None  # Verify field name in schema probe
    triple: Optional[int] = None
    hr: Optional[int] = None
    rbi: Optional[int] = None
    bb: Optional[int] = None
    so: Optional[int] = None
    sb: Optional[int] = None
    cs: Optional[int] = None
    avg: Optional[str] = None     # Often returned as string: ".285"
    obp: Optional[str] = None
    slg: Optional[str] = None
    ops: Optional[str] = None
    
    # Pitching stats (nullable for position players)
    ip: Optional[str] = None      # Innings pitched (e.g., "6.2")
    h_allowed: Optional[int] = None
    r_allowed: Optional[int] = None
    er: Optional[int] = None      # Earned runs
    bb_allowed: Optional[int] = None
    k: Optional[int] = None       # Strikeouts (pitching)
    whip: Optional[str] = None
    era: Optional[str] = None
    
    # Two-way players (like Ohtani) may have BOTH sets populated
```

### 3.2 BDL Client Method (Proposed)

```python
# backend/services/balldontlie.py

from typing import Any, Dict, List, Optional
from backend.data_contracts.mlb_player_stats import MLBPlayerStats
from pydantic import ValidationError

def get_mlb_stats(
    self,
    player_ids: Optional[List[int]] = None,
    team_ids: Optional[List[int]] = None,
    dates: Optional[List[str]] = None,
    season: int = 2026,
    per_page: int = 100,
) -> List[MLBPlayerStats]:
    """
    Fetch per-game player statistics from BDL.
    
    Args:
        player_ids: BDL player.id values (NOT Yahoo IDs, NOT mlbam IDs)
        team_ids: BDL team.id values
        dates: ISO 8601 dates (YYYY-MM-DD)
        season: MLB season year
        per_page: Pagination page size
    
    Returns:
        List of validated MLBPlayerStats objects
        
    Note:
        Identity resolution to Yahoo/mlbam is handled separately by
        PlayerIDResolver (see K_B_IDENTITY_RESOLUTION_SPEC.md).
    """
    params: Dict[str, Any] = {"season": season, "per_page": per_page}
    if player_ids:
        params["player_ids[]"] = player_ids
    if team_ids:
        params["team_ids[]"] = team_ids
    if dates:
        params["dates[]"] = dates
    
    results = []
    for raw in self._mlb_paginate("/stats", params):
        try:
            results.append(MLBPlayerStats.model_validate(raw))
        except ValidationError as e:
            logger.warning("MLBPlayerStats validation failed: %s", e)
    return results
```

### 3.3 Pagination Helper

```python
def _mlb_paginate(
    self,
    path: str,
    params: Optional[Dict] = None,
    max_pages: int = 20,
) -> List[Dict]:
    """Fetch all pages from MLB endpoint using cursor pagination."""
    params = dict(params or {})
    params.setdefault("per_page", 100)
    results: List[Dict] = []
    cursor: Optional[str] = None
    page = 0
    
    while page < max_pages:
        if cursor is not None:
            params["cursor"] = cursor
        try:
            data = self._mlb_get(path, params)
            results.extend(data.get("data", []))
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
            page += 1
            time.sleep(0.1)
        except Exception as exc:
            logger.error("_mlb_paginate(%s) page=%d failed: %s", path, page, exc)
            break
    
    return results
```

---

## 4. Required Schema Verification Steps

**Before writing production code:**

### 4.1 Execute Live Probe

```bash
# Single date query
curl -H "Authorization: $BALLDONTLIE_API_KEY" \
     "https://api.balldontlie.io/mlb/v1/stats?dates[]=2026-03-30&per_page=5"

# Player-specific query (use known BDL player.id)
curl -H "Authorization: $BALLDONTLIE_API_KEY" \
     "https://api.balldontlie.io/mlb/v1/stats?player_ids[]=208&per_page=5"
```

### 4.2 Verify Response Contains

- [ ] `data` array with stat records
- [ ] `meta.next_cursor` for pagination
- [ ] `player` object with BDL `id` field
- [ ] **CRITICAL:** Any external ID field? (mlbam_id, yahoo_id, etc.)
- [ ] Stat field naming convention
- [ ] Whether batting and pitching stats are unified
- [ ] Date format in response

### 4.3 Document Findings

Save schema discovery to: `reports/BDL_MLB_STATS_SCHEMA_DISCOVERY.md`

---

## 5. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| BDL stats don't include mlbam ID | High | **CRITICAL** | Implement name-based resolution (K-B Strategy B) |
| Stats endpoint returns season aggregates only | Low | High | Adjust contract to handle aggregates vs per-game |
| Field naming inconsistent with FanGraphs | Medium | Medium | Map BDL fields to canonical internal names |
| Pagination cursor depth limits | Low | Low | Implement date windowing for bulk ingestion |
| Two-way player stats (Ohtani) split across rows | Medium | Medium | Detect and merge by player+game key |

---

## 6. Integration with Rolling Windows

Once the stats endpoint is integrated:

```python
# Pseudocode for rolling window calculation
# NOTE: Identity resolution is handled by PlayerIDResolver (see K-B spec)

bdl_stats = client.get_mlb_stats(
    dates=date_range  # Last 14 days
)

# Convert to internal format with resolved mlbam ID
for stat in bdl_stats:
    # Use PlayerIDResolver to map BDL player → mlbam ID
    mlbam_id = resolver.resolve_bdl_to_mlbam(
        bdl_player_id=stat.player.id,
        name=stat.player.full_name
    )
    if mlbam_id:
        rolling_window[mlbam_id].append(stat)
```

---

## 7. Recommendations

1. **Immediate:** Execute schema verification probe (Section 4)
2. **Document findings:** Save to `reports/BDL_MLB_STATS_SCHEMA_DISCOVERY.md`
3. **Based on verification results:**
   - If mlbam ID present → simpler resolution path (see K-B Strategy A)
   - If only BDL internal ID → requires name-based resolution (K-B Strategy B)
4. **After verification:** Claude implements `MLBPlayerStats` contract + BDL client method

---

## References

- BDL API Docs: https://www.balldontlie.io/
- Existing pattern: `backend/services/balldontlie.py::get_player_season_stats()` (NCAAB)
- `reports/SCHEMA_DISCOVERY.md` — BDL MLB schema verification methodology
- `reports/K_B_IDENTITY_RESOLUTION_SPEC.md` — Player identity resolution (corrected)
- `tests/fixtures/bdl_mlb_*.json` — Verified BDL response patterns
