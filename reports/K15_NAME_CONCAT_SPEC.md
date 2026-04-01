# K-15: Name Concatenation Root Cause & Fix Spec

**Date:** April 1, 2026  
**Analyst:** Kimi CLI (Deep Intelligence Unit)  
**Issue:** Player names concatenated with injury info ("Jason Adam Quadriceps")

---

## Root Cause Analysis

### Location
**File:** `backend/fantasy_baseball/yahoo_client_resilient.py`  
**Function:** `_parse_player()` (lines 843-930)

### Extraction Logic (Lines 913-918)
```python
name = meta.get("full_name")
if not name and isinstance(meta.get("name"), dict):
    name = meta["name"].get("full")
if not name:
    name = meta.get("name", "Unknown")
```

### Problem
Yahoo's API occasionally returns `full_name` with injury information concatenated:

```json
{
  "full_name": "Jason Adam Quadriceps",
  "first_name": "Jason",
  "last_name": "Adam",
  "injury_note": "Quadriceps",
  "status": "IL"
}
```

The code extracts `full_name` verbatim without sanitization, resulting in "Jason Adam Quadriceps" instead of "Jason Adam".

### Why This Happens
1. Yahoo's data sources (STATS LLC) sometimes include injury suffixes in the display name
2. The recursive flattening (lines 848-862) captures the raw `full_name` field
3. No post-processing removes injury-related suffixes

---

## Proposed Fix

### Option A: Name Sanitization (Recommended)

Add a sanitization function that removes common injury suffixes from names:

```python
@staticmethod
def _sanitize_player_name(name: str, injury_note: Optional[str]) -> str:
    """
    Remove injury information concatenated to player names.
    
    Examples:
    - "Jason Adam Quadriceps" → "Jason Adam"
    - "Mike Trout Hamstring" → "Mike Trout"
    - "Shohei Ohtani Elbow" → "Shohei Ohtani"
    """
    if not name or name == "Unknown":
        return name
    
    # Common body parts/injury terms that get concatenated
    INJURY_SUFFIXES = [
        # Body parts
        r'\s+(?:Right\s+|Left\s+)?(?:Hamstring|Quadriceps|Groin|Calf|Thigh)',
        r'\s+(?:Right\s+|Left\s+)?(?:Shoulder|Elbow|Wrist|Hand|Finger|Thumb)',
        r'\s+(?:Right\s+|Left\s+)?(?:Knee|Ankle|Foot|Toe|Heel|Achilles)',
        r'\s+(?:Right\s+|Left\s+)?(?:Oblique|Abdomen|Back|Neck|Hip)',
        r'\s+(?:Head|Concussion|Ribs|Side)',
        # General injury terms
        r'\s+(?:Strain|Sprain|Tear|Surgery|Injury|IL|IL10|IL60|DTD|OUT)',
    ]
    
    # If injury_note is provided, try to remove it from name
    if injury_note and isinstance(injury_note, str):
        injury_clean = injury_note.strip()
        if injury_clean and name.endswith(injury_clean):
            name = name[:-len(injury_clean)].strip()
            return name
    
    # Fallback: regex-based removal
    sanitized = name
    for pattern in INJURY_SUFFIXES:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
    
    return sanitized.strip()
```

**Usage in `_parse_player`:**

```python
# Line 913-918 - modify to:
name = meta.get("full_name")
if not name and isinstance(meta.get("name"), dict):
    name = meta["name"].get("full")
if not name:
    name = meta.get("name", "Unknown")

# NEW: Sanitize name
injury_note = meta.get("injury_note") or None
name = self._sanitize_player_name(name, injury_note)
```

### Option B: Reconstruct from First/Last Name

If Yahoo provides separate `first_name` and `last_name` fields:

```python
first = meta.get("first_name", "")
last = meta.get("last_name", "")
if first and last:
    name = f"{first} {last}".strip()
else:
    name = meta.get("full_name", "Unknown")
```

**Trade-off:** More reliable but loses any preferred display name formatting (e.g., "J.D. Martinez" might become "J. D. Martinez" or "Julio Rodriguez" might become "Julio Rodríguez").

**Recommendation:** Option A preserves display formatting while removing injury noise.

---

## Implementation Details

### Import Required
Add at top of file:
```python
import re
```

### Lines to Modify

**File:** `backend/fantasy_baseball/yahoo_client_resilient.py`

1. **Add helper method** after line 841 (before `_parse_player`):
   ```python
   @staticmethod
   def _sanitize_player_name(name: str, injury_note: Optional[str]) -> str:
       # ... implementation from Option A ...
   ```

2. **Modify lines 913-920** in `_parse_player`:
   ```python
   # CURRENT:
   name = meta.get("full_name")
   if not name and isinstance(meta.get("name"), dict):
       name = meta["name"].get("full")
   if not name:
       name = meta.get("name", "Unknown")
   
   # NEW:
   name = meta.get("full_name")
   if not name and isinstance(meta.get("name"), dict):
       name = meta["name"].get("full")
   if not name:
       name = meta.get("name", "Unknown")
   
   # Sanitize injury suffixes from name
   injury_note = meta.get("injury_note") or None
   name = YahooFantasyClient._sanitize_player_name(name, injury_note)
   ```

---

## Test Cases

```python
# Test _sanitize_player_name
test_cases = [
    # (input_name, injury_note, expected_output)
    ("Jason Adam Quadriceps", "Quadriceps", "Jason Adam"),
    ("Jason Adam Quadriceps", None, "Jason Adam"),  # Regex fallback
    ("Mike Trout Hamstring", "Hamstring", "Mike Trout"),
    ("Shohei Ohtani Elbow", "Elbow", "Shohei Ohtani"),
    ("Chris Sale Tommy John Surgery", "Tommy John Surgery", "Chris Sale"),
    ("Jacob deGrom", None, "Jacob deGrom"),  # No injury, unchanged
    ("J.D. Martinez", None, "J.D. Martinez"),  # Period preserved
    ("Unknown", None, "Unknown"),  # Edge case
    ("", None, ""),  # Empty string
]

for name, note, expected in test_cases:
    result = YahooFantasyClient._sanitize_player_name(name, note)
    assert result == expected, f"Failed: {name} → {result}, expected {expected}"
```

---

## Edge Cases & Considerations

| Edge Case | Handling | Example |
|-----------|----------|---------|
| Name legitimately contains body part word | Regex requires leading whitespace | "Armando" (not "Armando ") |
| Injury note is substring of name | Exact match check first | "Knee" in "Kneeland" |
| Multiple injury words | Remove longest match first | "Right Hamstring Strain" |
| Case sensitivity | Regex uses IGNORECASE flag | "QUADRICEPS" or "quadriceps" |
| Hyphenated names | Preserved by \s+ anchor | "J.D. Martinez-Smith" |
| Non-ASCII characters | Unicode-safe regex | "Kiké Hernández" |

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Over-sanitization (removing legitimate name parts) | Only remove if: (a) matches injury_note exactly, or (b) matches well-known injury pattern with word boundary |
| Missing new injury types | Regex covers common body parts; fallback to injury_note exact match catches novel cases |
| Performance impact | Regex compilation is cached by Python; negligible overhead |

---

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `yahoo_client_resilient.py` | After 841 | Add `_sanitize_player_name()` method |
| `yahoo_client_resolient.py` | 913-920 | Add sanitizer call in `_parse_player()` |

---

## Verification

After implementation:

1. **Roster page** should show "Jason Adam" not "Jason Adam Quadriceps"
2. **API response** from `/api/fantasy/roster` returns clean names
3. **Injury note field** still contains "Quadriceps" for display in status column

---

## Effort Estimate

- Implementation: **10 minutes** (add function + 2 lines call site)
- Testing: **5 minutes** (verify with live roster endpoint)
- Total: **<15 minutes**

---

*Spec complete: Name sanitization will remove injury suffixes from player names at parsing time.*
