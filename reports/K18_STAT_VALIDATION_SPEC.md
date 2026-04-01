# K-18: Impossible Stats Backend Validation Spec

**Date:** April 1, 2026  
**Analyst:** Kimi CLI (Deep Intelligence Unit)  
**Scope:** Design backend fix for impossible stat values (e.g., -1 GS)

---

## Problem Statement

The Yahoo Fantasy API occasionally returns impossible negative values for counting statistics (e.g., `-1` for Games Started). These values propagate through the system:

1. `_extract_team_stats()` (main.py ~5577) extracts raw values without validation
2. `MatchupTeamOut.stats: dict` accepts any values without schema validation
3. Frontend displays raw `-1` to users, appearing as a bug

**Note:** ARCH-003 F6 adds frontend masking (shows "—" for negatives), but backend validation prevents bad data from entering the system.

---

## Root Cause Location

### File: `backend/main.py`

**Function:** `_extract_team_stats()` (lines 5537-5585)

```python
# Line 5577 - raw value extraction with no validation
val = stat.get("value", "")
if key:
    stats_dict[key] = val  # Stored as-is
```

**Issue:** Yahoo returns string values like `"-1"` which are stored directly.

---

## Proposed Fix

### Option A: In-Function Sanitization (Recommended)

Add validation logic directly in `_extract_team_stats()`:

```python
def _extract_team_stats(team_entry) -> tuple[str, str, dict]:
    # ... existing flattening logic ...
    
    # Build stats dict with validation
    stats_dict: dict = {}
    for s in stats_raw:
        if isinstance(s, dict):
            stat = s.get("stat", {})
            if isinstance(stat, dict):
                sid = str(stat.get("stat_id", ""))
                key = stat_id_map.get(sid, sid)
                val = stat.get("value", "")
                
                # NEW: Sanitize impossible values
                val = _sanitize_stat_value(val, key)
                
                if key:
                    stats_dict[key] = val
    
    # ... rest of function ...
```

Add helper function near `_extract_team_stats`:

```python
def _sanitize_stat_value(val, stat_name: str):
    """
    Sanitize stat values from Yahoo API.
    
    Rules:
    - Counting stats (GS, W, SV, K, HR, RBI, R, SB, etc.) cannot be negative → clamp to 0
    - Ratio stats (ERA, WHIP, AVG, OBP, OPS) can be 0 but not negative → clamp to 0
    - String values (e.g., ".000") pass through
    - Empty/None values pass through
    """
    if val is None or val == "":
        return val
    
    # List of counting stats that cannot be negative
    COUNTING_STATS = {
        "GS", "W", "SV", "K", "HR", "RBI", "R", "SB", "NSV", "H",
        "62", "23", "32", "28", "12", "13", "7", "16", "83", "60"
    }
    
    # List of ratio stats that cannot be negative
    RATIO_STATS = {
        "ERA", "WHIP", "AVG", "OBP", "OPS", "K/BB", "K9",
        "26", "27", "3", "55", "85", "38"
    }
    
    # Only sanitize if this is a known stat category
    if stat_name not in COUNTING_STATS and stat_name not in RATIO_STATS:
        return val
    
    # Try to parse as number
    try:
        num_val = float(val)
        if num_val < 0:
            logger.warning(f"Clamping impossible negative value for {stat_name}: {val} → 0")
            return "0" if stat_name in COUNTING_STATS else "0.000"
    except (ValueError, TypeError):
        # Not a number, pass through
        pass
    
    return val
```

### Option B: Pydantic Validator (Alternative)

Add a validator to `MatchupTeamOut`:

```python
class MatchupTeamOut(BaseModel):
    team_key: str
    team_name: str
    stats: dict
    
    @field_validator("stats", mode="before")
    @classmethod
    def sanitize_stats(cls, v):
        """Clamp negative values for counting stats."""
        if not isinstance(v, dict):
            return v
        
        COUNTING_STATS = {"GS", "W", "SV", "K", "HR", "RBI", "R", "SB", "62", "23", "32"}
        result = {}
        
        for key, val in v.items():
            if key in COUNTING_STATS:
                try:
                    num = float(val)
                    if num < 0:
                        val = "0"
                except (ValueError, TypeError):
                    pass
            result[key] = val
        
        return result
```

**Trade-offs:**
- Option A: Earlier validation, logs source of bad data, more control
- Option B: Schema-level enforcement, automatic for all MatchupTeamOut creation

**Recommendation:** Option A — validates at data ingestion, adds logging for diagnostics.

---

## Implementation Details

### Lines to Modify in `backend/main.py`

1. **Add helper function** after line 5536 (before `_extract_team_stats`):
   - Insert `_sanitize_stat_value()` function

2. **Modify line 5577-5579** in `_extract_team_stats`:
   ```python
   # CURRENT:
   val = stat.get("value", "")
   if key:
       stats_dict[key] = val
   
   # NEW:
   val = stat.get("value", "")
   val = _sanitize_stat_value(val, key)  # ADD THIS LINE
   if key:
       stats_dict[key] = val
   ```

### Test Case

```python
# Test data simulating Yahoo API response with bad data
test_stats = [
    {"stat": {"stat_id": "62", "value": "-1"}},  # GS = -1 (impossible)
    {"stat": {"stat_id": "23", "value": "5"}},   # W = 5 (valid)
    {"stat": {"stat_id": "26", "value": "-0.5"}}, # ERA = -0.5 (impossible)
]

# Expected after sanitization
assert stats_dict["GS"] == "0"      # Was "-1"
assert stats_dict["W"] == "5"       # Unchanged
assert stats_dict["ERA"] == "0.000" # Was "-0.5"
```

---

## Logging Recommendation

Add a warning log when sanitization occurs:

```python
logger.warning(f"Yahoo API returned impossible negative value: {stat_name}={val}, clamping to 0")
```

This allows monitoring of how frequently Yahoo returns bad data without affecting user experience.

---

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `backend/main.py` | After 5536 | Add `_sanitize_stat_value()` function |
| `backend/main.py` | 5577-5579 | Call sanitizer before storing stat |

---

## Verification

After implementation:

1. **Backend test:** Query `/api/fantasy/matchup` when Yahoo has -1 GS → should return 0
2. **Log check:** Verify warning appears in Railway logs
3. **Frontend display:** Matchup page shows "0" or "0.000" instead of "-1"

---

## Edge Cases Considered

| Case | Handling |
|------|----------|
| String values (".000") | Pass through unchanged |
| Empty/None values | Pass through unchanged |
| Decimal values ("3.5") | Pass through if non-negative |
| Unknown stat names | Pass through (no validation) |
| Very large negative ("-999") | Clamp to 0, log warning |

---

## Effort Estimate

- Implementation: **5 minutes** (copy-paste spec, add function call)
- Testing: **10 minutes** (verify with live matchup endpoint)
- Total: **<15 minutes**

---

*Spec complete: Backend validation will sanitize -1 GS to 0 before it reaches the frontend.*
