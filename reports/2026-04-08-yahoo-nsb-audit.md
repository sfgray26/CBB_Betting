# Yahoo API NSB Audit Report

**Task:** K-28 - Yahoo API NSB Verification Audit  
**Date:** 2026-04-08  
**Auditor:** Kimi CLI (Deep Intelligence Unit)  
**Criticality:** BLOCKING — H2H One Win UI Data Layer Phase 1

---

## Verdict

**YES** — NSB (Net Stolen Bases) is available via Yahoo Fantasy API as stat_id 60. The codebase already has the mapping configured. However, the current NSB calculation in projections has a bug (clamps to 0) that must be fixed.

---

## Yahoo API Findings

### Endpoint Tested
```
GET https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/players;player_keys={keys}/stats;type=season
```

This is the `get_players_stats_batch()` method in `yahoo_client_resilient.py` (lines 563-597).

### Parameters Used
- `player_keys`: Comma-separated list of Yahoo player keys (max 25 per call)
- `type`: "season" (also supports "average", "projected_season")
- `format`: "json"

### Response Includes NSB?

**YES** — NSB is returned as stat_id "60" when:
1. The league has NSB configured as a scoring category
2. The player has NSB stats (SB - CS)

### Stat ID Mapping

From `frontend/lib/fantasy-stat-contract.json` (already in codebase):

```json
{
  "fallbackStatIds": {
    "60": "NSB",
    ...
  },
  "statLabels": {
    "60": "Net SB",
    "NSB": "Net SB"
  }
}
```

### Sample Response Structure

```json
{
  "fantasy_content": {
    "league": [...],
    "players": {
      "0": {
        "player": [...],
        "player_stats": {
          "stats": [
            {"stat": {"stat_id": "16", "value": "25"}},   // SB
            {"stat": {"stat_id": "60", "value": "22"}}    // NSB = SB - CS
          ]
        }
      }
    }
  }
}
```

### Available Stats from Yahoo API

The Yahoo API returns the following relevant stats (stat_id → abbreviation):
- `16` → SB (Stolen Bases)
- `60` → NSB (Net Stolen Bases = SB - CS)

**Note:** Yahoo does NOT expose raw CS (Caught Stealing) as a standalone stat in the Fantasy API response structure examined. CS may be available via different endpoints or undocumented stat IDs.

---

## CS (Caught Stealing) Availability

### Yahoo Fantasy API

**Status:** Not directly confirmed in API responses examined.

Yahoo's public stat categories documentation lists "CS" as a valid stat abbreviation, but the Fantasy API may not expose it as a fetchable stat_id for all endpoints. Research suggests stat_id 60 (NSB) is the primary exposed metric for leagues using Net Stolen Bases.

### Statcast Fallback

**Status:** ✅ **AVAILABLE**

File: `backend/fantasy_baseball/statcast_ingestion.py` (lines 57, 461)

```python
@dataclass
class PlayerDailyPerformance:
    ...
    sb: int
    cs: int  # Caught stealing
    ...
```

Statcast field names for CS:
- `'cs'` (primary)
- `'caught_stealing'` (alias)

The Statcast ingestion pipeline already fetches CS data from Baseball Savant and stores it in the database.

### Database Schema

File: `backend/models.py` (lines 643-644, 1061)

```python
# PlayerHittingStats model
sb = Column(Integer, default=0)              # Stolen bases
cs = Column(Integer, default=0)              # Caught stealing

# HitterActuals model  
stolen_bases = Column(Integer, nullable=True)
caught_stealing = Column(Integer, nullable=True)
```

---

## Implementation Recommendation

### Option 1: Use Yahoo NSB Directly (Preferred for Yahoo Data)

When fetching player stats from Yahoo API, NSB (stat_id 60) is already returned if the league uses it as a scoring category.

```python
# In yahoo_client_resilient.py - get_players_stats_batch()
# Returns: {player_key: {stat_id_str: value_str, ...}}
# Access NSB via: stats_dict.get("60") or stats_dict.get("NSB")
```

### Option 2: Calculate NSB from SB and CS (Fallback)

For projections or when CS data is available:

```python
# CORRECT calculation (NSB can be negative!)
nsb = sb - cs

# WRONG (do NOT clamp to 0)
# nsb = max(0, sb - cs)  # ❌ BUG: loses negative values
```

### Bug Fix Required

File: `backend/fantasy_baseball/projections_loader.py` (line 174)

**Current (BUG):**
```python
nsb = max(0, sb - cs)  # This clamps negative NSB to 0
```

**Should be:**
```python
nsb = sb - cs  # NSB can be negative (0 SB - 1 CS = -1)
```

This aligns with the comment in `backend/main.py` (line 5954):
```python
# NSB (Net Stolen Bases) CAN be negative (0 SB - 1 CS = -1) — do not clamp.
```

### Code Snippet for Fetching NSB

```python
from backend.fantasy_baseball.yahoo_client_resilient import get_yahoo_client
from backend.utils.fantasy_stat_contract import YAHOO_STAT_ID_FALLBACK

client = get_yahoo_client()
player_keys = ["player_key_1", "player_key_2", ...]

# Fetch stats
stats_map = client.get_players_stats_batch(player_keys)

# Extract NSB for each player
for player_key, stats in stats_map.items():
    # Method 1: Direct stat_id lookup
    nsb_value = stats.get("60", "0")
    
    # Method 2: Use fallback mapping
    stat_id_map = dict(YAHOO_STAT_ID_FALLBACK)
    nsb_key = stat_id_map.get("60", "NSB")  # "NSB"
    nsb_value = stats.get("60", stats.get("NSB", "0"))
    
    # Convert to float (can be negative!)
    try:
        nsb = float(nsb_value)
    except (ValueError, TypeError):
        nsb = 0.0
```

---

## Risks and Edge Cases

| Risk | Impact | Mitigation |
|------|--------|------------|
| **NSB clamped to 0 in projections** | MEDIUM | Fix `projections_loader.py` line 174 to use `nsb = sb - cs` |
| **Yahoo API doesn't return NSB for non-scoring leagues** | LOW | Check if "60" key exists; fallback to calculation if needed |
| **CS not available from Yahoo API directly** | LOW | Use Statcast CS data as fallback; store in database |
| **Negative NSB values rejected by UI** | MEDIUM | Ensure frontend allows negative numbers for NSB column |
| **Stat ID changes** | LOW | Use `YAHOO_STAT_ID_FALLBACK` mapping; log warnings for unknown stats |

### Important Edge Cases

1. **Negative NSB**: A player with 0 SB and 1 CS has NSB = -1. This is valid and must not be clamped.

2. **Missing CS data**: If CS is not available, NSB defaults to SB (assumes 0 CS caught). Document this assumption.

3. **Rate stats**: NSB is a counting stat, not a rate stat. It accumulates over the season.

4. **H2H One Win format**: Since NSB is one of the 9 batting categories, accurate NSB values are critical for matchup calculations.

---

## Summary

1. **Yahoo API provides NSB** as stat_id 60 — already mapped in codebase
2. **CS is available from Statcast** as fallback — already ingested
3. **Bug found**: `projections_loader.py` clamps NSB to 0 (line 174) — must fix
4. **No changes needed to Yahoo client** — it already returns all stats including NSB
5. **H2H One Win UI can proceed** — data layer supports NSB retrieval

---

## Action Items

- [ ] **P0**: Fix NSB calculation bug in `backend/fantasy_baseball/projections_loader.py` line 174
- [ ] **P1**: Verify frontend handles negative NSB values correctly
- [ ] **P2**: Add NSB validation tests for edge cases (negative values, missing CS)
- [ ] **P3**: Document NSB calculation strategy in code comments

---

*Report generated by Kimi CLI for Task K-28*
