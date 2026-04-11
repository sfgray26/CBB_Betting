# K-14: Yahoo Stat ID "57" Confirmation Report

**Date:** April 1, 2026  
**Analyst:** Kimi CLI (Deep Intelligence Unit)  
**Scope:** Determine correct mapping for Yahoo stat IDs 57 and 85

---

## Executive Summary

**Finding:** Stat ID 57 and 85 mappings are **deliberately excluded** from the codebase due to observed data inconsistencies. Web research cannot conclusively confirm the mappings — a live Yahoo API call is required.

**Recommendation:** Claude Code should execute a test query against the live league endpoint to verify stat names before adding mappings.

---

## Current State (Verified from Code)

### Backend (`backend/main.py` lines 5465-5466)
```python
# NOTE: "57" (BB?) and "85" (OBP?) deliberately excluded — not in category_tracker
# and observed to map to wrong stats in this league. Let get_league_settings() handle them.
```

### Frontend (`frontend/lib/constants.ts` line 31)
```typescript
// Pitching — numeric IDs (57=BB and 85=OBP excluded: unconfirmed for this league)
```

### category_tracker.py (YAHOO_STAT_MAP)
Stat ID 57 is **absent** from the confirmed mappings. Only these are mapped:
- Batting: 60, 7, 12, 13, 16, 3, 55
- Pitching: 50, 23, 32, 42, 26, 27, 83

---

## Web Research Findings

### Yahoo Fantasy Documentation
The [Yahoo Fantasy Sports API documentation](https://developer.yahoo.com/fantasysports/guide/) shows sample XML with stat IDs but does not provide a comprehensive stat ID reference table.

### Sample from API Docs
```xml
<stat>
  <stat_id>60</stat_id>   <!-- Hits -->
  <value>13/31</value>
</stat>
<stat>
  <stat_id>7</stat_id>    <!-- Runs -->
  <value>9</value>
</stat>
```

### Third-Party Sources (Unconfirmed)
Community documentation and forum posts suggest:
- **57** = "BB" (Walks — pitcher category)
- **85** = "OBP" (On-Base Percentage — batter category)

**However**, the code comments explicitly state these "observed to map to wrong stats in this league," suggesting the Treemendous league uses non-standard category configurations.

---

## Root Cause of Uncertainty

Yahoo Fantasy allows league commissioners to customize stat categories. The numeric IDs may vary based on:
1. **League-specific configuration** (standard vs. custom categories)
2. **Year-over-year changes** in Yahoo's API
3. **Position type** (batter vs. pitcher stats with same abbreviation)

The current exclusion is **intentional and correct** — adding mappings without verification would risk displaying wrong category names.

---

## Recommended Verification Method

Claude Code should execute this test against the live API:

```python
from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient

client = YahooFantasyClient()
settings = client.get_league_settings()

# Extract stat_categories from settings
stat_cats = (
    settings
    .get("settings", [{}])[0]
    .get("stat_categories", {})
    .get("stats", [])
)

# Print all stat IDs and display names
for entry in stat_cats:
    if isinstance(entry, dict):
        s = entry.get("stat", {})
        sid = s.get("stat_id")
        name = s.get("display_name") or s.get("abbreviation")
        pos_type = s.get("position_type")  # 'B' for batter, 'P' for pitcher
        print(f"Stat {sid}: {name} (type: {pos_type})")
```

**Expected output will show:**
- Stat 57 → "BB" or "Walks" (if pitcher walks)
- Stat 85 → "OBP" or "On-Base %" (if batter OBP)

---

## Decision Matrix

| Scenario | Stat 57 Mapping | Stat 85 Mapping | Action |
|----------|-----------------|-----------------|--------|
| **A** (Most likely) | BB (Pitcher Walks) | OBP (Batter On-Base %) | Add to both `_YAHOO_STAT_FALLBACK` and `STAT_LABELS` |
| **B** | Not in league | Not in league | Keep excluded; update comment to "not used in this league" |
| **C** | Different stat | Different stat | Map to correct names; document anomaly |

---

## Implementation Guidance (Post-Verification)

If Scenario A confirmed:

### Backend (`backend/main.py` line ~5464)
```python
_YAHOO_STAT_FALLBACK: dict[str, str] = {
    # ... existing mappings ...
    "57": "BB",    # Walks (pitching)
    "85": "OBP",   # On-Base % (batting)
}
```

### Frontend (`frontend/lib/constants.ts`)
```typescript
export const STAT_LABELS: Record<string, string> = {
  // ... existing mappings ...
  '57': 'Walks (P)',     // Pitcher walks
  '85': 'On-Base %',     // Batter OBP
}
```

---

## Conclusion

**Confidence in "57=BB, 85=OBP":** 75% (industry standard, but contradicted by code comments)

**Recommendation:** Do NOT add mappings without live API verification. The current exclusion is safer than potentially wrong labels.

**Escalation to Claude Code:** Execute the verification script above and update mappings based on actual API response.

---

*Report complete: Web research inconclusive; live API verification required for definitive answer.*
