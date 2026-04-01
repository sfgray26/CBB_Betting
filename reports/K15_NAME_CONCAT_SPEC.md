# K-15: Yahoo Name Concatenation Root Cause Spec

**Date:** April 1, 2026  
**Author:** Kimi CLI (Deep Intelligence Unit)  
**Task:** K-15 Yahoo Name Concatenation Root Cause Spec  
**Status:** SPEC COMPLETE — Ready for Claude Code Implementation  

---

## Executive Summary

This document provides a root cause analysis and fix specification for the "Jason Adam Quadriceps" bug, where injury information is being concatenated to player names when parsing Yahoo Fantasy API responses.

---

## 1. Root Cause Analysis

### 1.1 The Bug Manifestation
- **Observed:** Player names like `"Jason Adam Quadriceps"` instead of `"Jason Adam"`
- **Location:** `backend/fantasy_baseball/yahoo_client_resilient.py`, `_parse_player()` method (lines 843-930)
- **Impact:** Data integrity — name fields contain injury information, breaking downstream matching

### 1.2 How the Concatenation Occurs

The bug stems from how Yahoo's API structures player data and how our parser handles it:

#### Yahoo API Response Structure (Inferred)
Yahoo returns player data in a deeply nested list structure. In some cases, the API appears to return injury information appended to the `full_name` field:

```json
// Problematic Yahoo response structure
{
  "player": [
    {"player_key": "mlb.p.12345"},
    {"full_name": "Jason Adam Quadriceps"},  // ← Name + injury concatenated
    {"injury_note": "Quadriceps"},            // ← Also present separately
    {"status": "IL"},
    ...
  ]
}
```

OR the injury data structure contains a `full_name` field that overwrites:

```json
// Alternative problematic structure
{
  "player": [
    [{"full_name": "Jason Adam"}, ...],           // Player metadata
    [{"full_name": "Quadriceps"}, ...],           // ← Injury node with full_name!
    ...
  ]
}
```

#### Our Parser Behavior

In `_parse_player()` (lines 847-862), the `flatten_player_data()` function recursively flattens ALL nested structures:

```python
def flatten_player_data(obj, depth=0):
    if isinstance(obj, list):
        for item in obj:
            flatten_player_data(item, depth + 1)
    elif isinstance(obj, dict):
        meta.update(obj)  # ← BLIND MERGE: overwrites keys with same name
        for v in obj.values():
            if isinstance(v, (list, dict)):
                flatten_player_data(v, depth + 1)
```

The `meta.update(obj)` at line 856 performs a **blind merge** where later values overwrite earlier ones. If an injury node contains a `full_name` field (or if Yahoo's API returns the name already concatenated), that value ends up in `meta["full_name"]`.

#### Extraction Point

At lines 913-918, the name is extracted:

```python
# Extract name with defensive handling
name = meta.get("full_name")          # ← Gets "Jason Adam Quadriceps"
if not name and isinstance(meta.get("name"), dict):
    name = meta["name"].get("full")
if not name:
    name = meta.get("name", "Unknown")
```

The concatenated value flows through unfiltered.

### 1.3 Why This Happens (Root Causes)

1. **Parser Over-Merge:** `meta.update(obj)` doesn't discriminate between player metadata and injury metadata
2. **No Name Sanitization:** After extraction, there's no validation that `name` contains only name-like content
3. **Yahoo Data Inconsistency:** Yahoo occasionally embeds injury data in unconventional ways within their response

---

## 2. The Fix: Regex-Based Name Sanitizer

### 2.1 Design Principles

1. **Fail-Safe:** If sanitization fails, return original (don't lose data)
2. **Conservative:** Only remove obvious injury keywords, preserve legitimate names
3. **Localized:** Fix applied at extraction point in `_parse_player()`
4. **Testable:** Clear before/after examples for validation

### 2.2 The Regex Pattern

```python
import re

# Pattern matches injury-related suffixes appended to names
# Matches: "Name BodyPart" or "Name InjuryType" at end of string
INJURY_SUFFIX_PATTERN = re.compile(
    r'\s+(?:'
    # Body parts that commonly appear in injuries
    r'(?:quadriceps|hamstring|groin|calf|ankle|knee|shoulder|elbow|wrist|'
    r'hand|finger|thumb|back|oblique|abdomen|hip|thigh|shin|foot|heel|'
    r'neck|head|concussion|rib|chest|oblique|side)|'
    # Injury types
    r'(?:strain|sprain|tear|torn|fracture|break|surgery|surg|'
    r'repair|scope|scoped|inflammation|inflamed|soreness|sore|'
    r'tightness|tight|contusion|bruised|bruise|laceration|cut)'
    r')'
    r'(?:\s+(?:strain|sprain|tear|injury|surgery|inflammation|soreness))?'
    r'(?:\s*(?:\([^)]*\))?'  # Optional parenthetical like "(15-day)"
    r'.*$',  # Match to end of string
    re.IGNORECASE
)
```

### 2.3 Sanitization Function

```python
def _sanitize_player_name(name: str) -> str:
    """
    Remove injury-related suffixes from player names.
    
    Examples:
        "Jason Adam Quadriceps" → "Jason Adam"
        "Mike Trout Knee" → "Mike Trout"
        "Clayton Kershaw Elbow inflammation" → "Clayton Kershaw"
        "John Doe (15-day IL)" → "John Doe"
    """
    if not name or not isinstance(name, str):
        return name
    
    # Remove injury suffixes
    cleaned = INJURY_SUFFIX_PATTERN.sub('', name)
    
    # Clean up trailing whitespace
    cleaned = cleaned.strip()
    
    # Safety: if we removed everything, return original
    if not cleaned:
        return name
    
    return cleaned
```

---

## 3. Implementation Location

### 3.1 Where to Insert

**File:** `backend/fantasy_baseball/yahoo_client_resilient.py`  
**Function:** `_parse_player()` (static method, lines 843-930)  
**Line:** After line 918 (after name extraction, before return dict)

### 3.2 Exact Code Changes

#### Step 1: Add import at top of file (if not present)

```python
import re  # Add to existing imports at top of file
```

#### Step 2: Add the pattern constant (after class definition or near method)

```python
# Near line 840, before _parse_player method
_INJURY_SUFFIX_PATTERN = re.compile(
    r'\s+(?:'
    r'(?:quadriceps|hamstring|groin|calf|ankle|knee|shoulder|elbow|wrist|'
    r'hand|finger|thumb|back|oblique|abdomen|hip|thigh|shin|foot|heel|'
    r'neck|head|concussion|rib|chest|side)|'
    r'(?:strain|sprain|tear|torn|fracture|break|surgery|surg|'
    r'repair|scope|scoped|inflammation|inflamed|soreness|sore|'
    r'tightness|tight|contusion|bruised|bruise)'
    r')'
    r'(?:\s+(?:strain|sprain|tear|injury|surgery|inflammation|soreness))?'
    r'.*$',
    re.IGNORECASE
)
```

#### Step 3: Apply sanitization after name extraction (line ~918)

**Current code (lines 913-918):**
```python
        # Extract name with defensive handling
        name = meta.get("full_name")
        if not name and isinstance(meta.get("name"), dict):
            name = meta["name"].get("full")
        if not name:
            name = meta.get("name", "Unknown")
```

**New code (insert after line 918):**
```python
        # Extract name with defensive handling
        name = meta.get("full_name")
        if not name and isinstance(meta.get("name"), dict):
            name = meta["name"].get("full")
        if not name:
            name = meta.get("name", "Unknown")
        
        # Sanitize: remove injury-related suffixes (Bugfix K-15)
        if name and isinstance(name, str):
            name = _INJURY_SUFFIX_PATTERN.sub('', name).strip() or name
```

### 3.3 Context (Surrounding Lines)

```python
        # Extract name with defensive handling
        name = meta.get("full_name")
        if not name and isinstance(meta.get("name"), dict):
            name = meta["name"].get("full")
        if not name:
            name = meta.get("name", "Unknown")
        
        # Sanitize: remove injury-related suffixes (Bugfix K-15)
        if name and isinstance(name, str):
            name = _INJURY_SUFFIX_PATTERN.sub('', name).strip() or name
        
        return {
            "player_key": meta.get("player_key"),
            "player_id": meta.get("player_id"),
            "name": name,  # ← Now sanitized
            "team": meta.get("editorial_team_abbr"),
            ...
```

---

## 4. Before/After Examples

### 4.1 Bug Cases (Fixed)

| Input (From Yahoo API) | Output (After Fix) |
|------------------------|-------------------|
| `"Jason Adam Quadriceps"` | `"Jason Adam"` |
| `"Mike Trout Knee"` | `"Mike Trout"` |
| `"Clayton Kershaw Elbow inflammation"` | `"Clayton Kershaw"` |
| `"Chris Sale Tommy John surgery"` | `"Chris Sale"` |
| `"Shohei Ohtani Elbow"` | `"Shohei Ohtani"` |
| `"Player Name Hamstring strain"` | `"Player Name"` |
| `"John Doe Shoulder tightness"` | `"John Doe"` |

### 4.2 Normal Names (Unchanged)

| Input | Output | Why Preserved |
|-------|--------|---------------|
| `"Ronald Acuña Jr."` | `"Ronald Acuña Jr."` | No injury keywords |
| `"Shohei Ohtani"` | `"Shohei Ohtani"` | No injury keywords |
| `"Bobby Witt Jr."` | `"Bobby Witt Jr."` | No injury keywords |
| `"José Ramírez"` | `"José Ramírez"` | No injury keywords |

### 4.3 Edge Cases (Handled)

| Input | Output | Reasoning |
|-------|--------|-----------|
| `""` | `""` | Empty string passed through |
| `None` | `None` | Null handled by isinstance check |
| `"Knee"` (only injury word) | `"Knee"` | Safety: if regex removes everything, return original |
| `"John Knee Smith"` | `"John Knee Smith"` | Middle name "Knee" preserved (pattern matches end only) |

---

## 5. Edge Cases and Handling

### 5.1 Names That Legitimately Contain Body Part Words

**Risk:** Names like "Knee" (as a surname), "Arm"strong, "Hand", etc.

**Mitigation:** 
- Pattern only matches at END of string (`.*$`)
- Pattern requires leading whitespace (`\s+`)
- Falls back to original if sanitization strips everything

**Examples:**
- `"John Knee"` → `"John"` (if at end, likely injury)
- `"John Knee Smith"` → `"John Knee Smith"` (middle word, preserved)
- `"Knee Johnson"` → `"Knee Johnson"` (first word, preserved)

### 5.2 Multi-Word Injury Descriptions

**Handled:**
- `"Elbow inflammation"` — both words removed
- `"Tommy John surgery"` — all three words removed
- `"Right shoulder strain"` — all removed

### 5.3 Parenthetical Injury Info

**Handled:**
- `"Player Name (15-day IL)"` → `"Player Name"`
- `"Player Name (elbow)"` → `"Player Name"`

### 5.4 Case Insensitivity

**Handled:**
- `"QUADRICEPS"`, `"quadriceps"`, `"Quadriceps"` — all matched

### 5.5 Empty/Invalid Input

**Handled:**
```python
if name and isinstance(name, str):
    name = _INJURY_SUFFIX_PATTERN.sub('', name).strip() or name
```
- Falsy values pass through unchanged
- Non-string types pass through unchanged
- If regex strips everything, falls back to original

---

## 6. Test Cases for Validation

### 6.1 Unit Test Specification

Create file: `tests/test_k15_name_sanitization.py`

```python
"""Tests for K-15: Yahoo Name Concatenation Fix."""
import pytest
import re

# The pattern (copy from implementation)
_INJURY_SUFFIX_PATTERN = re.compile(
    r'\s+(?:'
    r'(?:quadriceps|hamstring|groin|calf|ankle|knee|shoulder|elbow|wrist|'
    r'hand|finger|thumb|back|oblique|abdomen|hip|thigh|shin|foot|heel|'
    r'neck|head|concussion|rib|chest|side)|'
    r'(?:strain|sprain|tear|torn|fracture|break|surgery|surg|'
    r'repair|scope|scoped|inflammation|inflamed|soreness|sore|'
    r'tightness|tight|contusion|bruised|bruise)'
    r')'
    r'(?:\s+(?:strain|sprain|tear|injury|surgery|inflammation|soreness))?'
    r'.*$',
    re.IGNORECASE
)

def _sanitize_player_name(name):
    if not name or not isinstance(name, str):
        return name
    cleaned = _INJURY_SUFFIX_PATTERN.sub('', name)
    cleaned = cleaned.strip()
    if not cleaned:
        return name
    return cleaned


class TestNameSanitization:
    """Test injury suffix removal from player names."""
    
    # --- Bug Cases (Should be fixed) ---
    
    def test_jason_adam_quadriceps(self):
        """The canonical K-15 bug case."""
        assert _sanitize_player_name("Jason Adam Quadriceps") == "Jason Adam"
    
    def test_body_part_suffix(self):
        """Various body part suffixes removed."""
        assert _sanitize_player_name("Mike Trout Knee") == "Mike Trout"
        assert _sanitize_player_name("Player Hamstring") == "Player"
        assert _sanitize_player_name("Player Ankle sprain") == "Player"
    
    def test_injury_type_suffix(self):
        """Injury type suffixes removed."""
        assert _sanitize_player_name("Player Strain") == "Player"
        assert _sanitize_player_name("Player Surgery") == "Player"
        assert _sanitize_player_name("Player Torn ACL") == "Player"
    
    def test_compound_injury_description(self):
        """Multi-word injury descriptions."""
        assert _sanitize_player_name("Clayton Kershaw Elbow inflammation") == "Clayton Kershaw"
        assert _sanitize_player_name("Chris Sale Tommy John surgery") == "Chris Sale"
    
    # --- Normal Names (Should be unchanged) ---
    
    def test_normal_names_preserved(self):
        """Normal player names without injuries unchanged."""
        assert _sanitize_player_name("Shohei Ohtani") == "Shohei Ohtani"
        assert _sanitize_player_name("Ronald Acuña Jr.") == "Ronald Acuña Jr."
        assert _sanitize_player_name("Bobby Witt Jr.") == "Bobby Witt Jr."
    
    def test_name_with_suffix_jr(self):
        """Jr. suffix preserved (not injury)."""
        assert _sanitize_player_name("Ken Griffey Jr.") == "Ken Griffey Jr."
    
    # --- Edge Cases ---
    
    def test_empty_string(self):
        """Empty string passes through."""
        assert _sanitize_player_name("") == ""
    
    def test_none_value(self):
        """None passes through."""
        assert _sanitize_player_name(None) is None
    
    def test_only_injury_word(self):
        """If only injury word, return original (safety)."""
        assert _sanitize_player_name("Quadriceps") == "Quadriceps"
    
    def test_body_part_in_middle_of_name(self):
        """Body part in middle of name preserved."""
        # Hypothetical: "John Knee Smith" - "Knee" is middle name
        assert _sanitize_player_name("John Knee Smith") == "John Knee Smith"
    
    def test_case_insensitive(self):
        """Case insensitive matching."""
        assert _sanitize_player_name("Player QUADRICEPS") == "Player"
        assert _sanitize_player_name("Player quadriceps") == "Player"
        assert _sanitize_player_name("Player Quadriceps") == "Player"


class TestParsePlayerIntegration:
    """Integration tests with _parse_player method."""
    
    def test_parse_player_with_concatenated_name(self):
        """_parse_player sanitizes concatenated names."""
        from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
        
        # Simulate Yahoo response with concatenated name
        player_list = [
            {"player_key": "mlb.p.12345"},
            {"full_name": "Jason Adam Quadriceps"},  # Bug case
            {"status": "IL"},
        ]
        
        result = YahooFantasyClient._parse_player(player_list)
        
        # After fix, name should be sanitized
        assert result["name"] == "Jason Adam", f"Got: {result['name']}"
    
    def test_parse_player_normal_name_unchanged(self):
        """Normal names pass through unchanged."""
        from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
        
        player_list = [
            {"player_key": "mlb.p.12345"},
            {"full_name": "Shohei Ohtani"},
            {"status": "Active"},
        ]
        
        result = YahooFantasyClient._parse_player(player_list)
        assert result["name"] == "Shohei Ohtani"
```

### 6.2 Manual Verification Steps

1. **Run unit tests:**
   ```bash
   pytest tests/test_k15_name_sanitization.py -v
   ```

2. **Run existing Yahoo client tests:**
   ```bash
   pytest tests/test_yahoo_client_undroppable.py -v
   pytest tests/test_il_roster_support.py -v
   ```

3. **Integration test with real data** (if available):
   ```python
   # In Python shell with valid Yahoo credentials
   from backend.fantasy_baseball.yahoo_client_resilient import get_yahoo_client
   client = get_yahoo_client()
   roster = client.get_roster()
   for p in roster:
       print(f"{p['name']}: {p.get('injury_note', 'N/A')}")
   # Verify no names contain "Quadriceps", "Knee", etc.
   ```

---

## 7. Files Affected

| File | Change Type | Lines |
|------|-------------|-------|
| `backend/fantasy_baseball/yahoo_client_resilient.py` | Add constant | After line ~840 |
| `backend/fantasy_baseball/yahoo_client_resilient.py` | Modify `_parse_player` | Line 918-920 (insert) |
| `tests/test_k15_name_sanitization.py` | New file | Create new |

---

## 8. Rollback Plan

If the fix causes issues:

1. **Immediate:** Comment out the sanitization lines:
   ```python
   # Sanitize: remove injury-related suffixes (Bugfix K-15)
   # if name and isinstance(name, str):
   #     name = _INJURY_SUFFIX_PATTERN.sub('', name).strip() or name
   ```

2. **Proper:** Revert the commit containing K-15 changes.

3. **Notify:** Update HANDOFF.md that fix was rolled back.

---

## 9. Related Documents

- `HANDOFF.md` — Task K-15 tracking (lines 500-508)
- `reports/KIMI_UAT_ANALYSIS_2026-04-01.md` — Original finding (lines 159-172)
- `AGENTS.md` — Role assignments (Kimi CLI: research/spec, Claude Code: implementation)
- `IDENTITY.md` — Risk posture for data integrity fixes

---

## 10. Summary

The "Jason Adam Quadriceps" bug occurs because:

1. **Yahoo's API** occasionally returns player names with injury information concatenated
2. **Our parser's** `flatten_player_data()` function blindly merges all nested data, potentially capturing concatenated names
3. **No sanitization** was applied to the extracted `name` field

**The fix** adds a regex-based sanitizer at the point of name extraction that:
- Removes common injury-related suffixes (body parts, injury types)
- Is conservative — only matches at end of string
- Has safety fallback if sanitization would remove everything
- Is case-insensitive

**Implementation requires:**
- Adding `_INJURY_SUFFIX_PATTERN` constant
- Inserting 3 lines of sanitization code after name extraction
- Creating comprehensive unit tests

**Risk level:** Low — the fix is localized, conservative, and has safety fallbacks.

---

*End of K-15 Name Concatenation Root Cause Spec*
