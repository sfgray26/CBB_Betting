# K-18: Matchup Impossible-Stats Backend Spec

**Date:** 2026-04-01  
**Agent:** Kimi CLI (Deep Intelligence Unit)  
**Task:** Design exact backend fix for -1 Games Started (GS) bug and other impossible stat values  
**Status:** SPECIFICATION COMPLETE — Ready for Claude implementation  

---

## 1. Executive Summary

The "-1 Games Started" bug is a **Yahoo API data issue** where Yahoo returns `-1` as a placeholder value for stats that have no accumulated data (typically at the start of a scoring week). The frontend currently masks this with an em-dash (F6 fix), but the backend should sanitize this data at the source.

---

## 2. Root Cause Analysis

### 2.1 Data Flow Trace

```
Yahoo API (scoreboard endpoint)
    ↓
_get() → get_scoreboard() returns raw XML/JSON
    ↓
_extract_team_stats() at main.py:5537-5585
    ↓
    Line 5577: val = stat.get("value", "")
    Line 5579: stats_dict[key] = val  ← NO VALIDATION
    ↓
MatchupResponse schema (main.py:5652-5665)
    ↓
Frontend matchup page
```

### 2.2 Where -1 Originates

**Source:** Yahoo Fantasy API returns `-1` as a **string value** for accumulated stats when:
- The scoring period has just started
- No games have been played yet in the current week
- Yahoo's internal null representation for counting stats

**Not a calculation bug:** The backend does not calculate these values — they are extracted verbatim from Yahoo's `team_stats` block.

### 2.3 Example Yahoo Response Snippet

```json
{
  "stat": {
    "stat_id": "62",
    "value": "-1"    ← Games Started (GS) before any pitcher has started
  }
}
```

---

## 3. Stat ID Mapping Reference

From `_YAHOO_STAT_FALLBACK` (main.py:5457-5467):

| Stat ID | Abbr | Type | Category |
|---------|------|------|----------|
| 3 | AVG | ratio | batting |
| 7 | R | counting | batting |
| 8 | H | counting | batting |
| 12 | HR | counting | batting |
| 13 | RBI | counting | batting |
| 16 | SB | counting | batting |
| 21 | IP | counting (decimal) | pitching |
| 23 | W | counting | pitching |
| 26 | ERA | ratio | pitching |
| 27 | WHIP | ratio | pitching |
| 28 | K | counting | pitching |
| 29 | QS | counting | pitching |
| 32 | SV | counting | pitching |
| 38 | K/BB | ratio | pitching |
| 50 | IP | counting (decimal) | pitching |
| 55 | OPS | ratio | batting |
| 60 | H | counting | batting |
| 62 | GS | counting | pitching |
| 83 | NSV | counting | pitching |

---

## 4. Validation Design

### 4.1 Where to Add Validation

**Recommended Location:** `backend/main.py` inside `_extract_team_stats()` function (lines 5570-5580)

**Rationale:**
- Single point of entry for all Yahoo stat values
- Keeps validation close to data source
- Avoids schema changes that might affect other endpoints
- Consistent with existing pattern (stat_id_map lookup)

**Not recommended:**
- Pydantic validator: Would require explicit field definitions (current schema uses `dict`)
- Endpoint level: Would duplicate validation across multiple places

### 4.2 Guard Conditions by Stat Type

#### Counting Stats (clamp negative → 0)
- R, H, HR, RBI, SB, W, K, QS, SV, NSV, GS, HLD, BB
- IP (Innings Pitched) — can be 0, never negative

#### Ratio Stats (clamp negative → None → displayed as "—")
- AVG, OPS, ERA, WHIP, K/BB, K9, OBP
- These require actual data to calculate; -1 indicates insufficient sample

### 4.3 Exact Validation Code

```python
# Stat categories that are counting stats (cannot be negative)
_COUNTING_STATS = {
    'R', 'H', 'HR', 'RBI', 'SB',  # batting
    'W', 'K', 'QS', 'SV', 'NSV', 'GS', 'HLD', 'BB',  # pitching
    'IP', '21', '50',  # IP has multiple IDs
    '7', '8', '12', '13', '16',  # batting by ID
    '23', '28', '29', '32', '62', '83',  # pitching by ID
}

# Ratio stats that cannot be negative (but can be 0)
_RATIO_STATS = {
    'AVG', 'OPS', 'ERA', 'WHIP', 'K/BB', 'K9', 'OBP',
    '3', '26', '27', '38', '55', '85',
}

def _sanitize_stat_value(key: str, raw_value: str) -> str | int | float | None:
    """
    Sanitize Yahoo stat values:
    - Counting stats: -1 → 0 (int)
    - Ratio stats: -1 → None (displayed as '—')
    - Valid values: converted to appropriate type
    """
    if raw_value is None or raw_value == '':
        return None
    
    # Try to parse as number
    try:
        num_val = float(raw_value)
    except (ValueError, TypeError):
        # Non-numeric strings pass through (rare, but possible)
        return raw_value
    
    # Check for negative values
    if num_val < 0:
        if key in _COUNTING_STATS:
            return 0
        elif key in _RATIO_STATS:
            return None
        # Unknown stats: pass through as-is (conservative)
        return raw_value
    
    # Non-negative: return appropriate type
    if key in _COUNTING_STATS:
        # Counting stats are integers (except IP which keeps decimals)
        if key in ('IP', '21', '50'):
            return num_val  # Keep decimal for IP
        return int(num_val)
    elif key in _RATIO_STATS:
        return num_val
    
    # Unknown stat: return as-is
    return raw_value
```

### 4.4 Integration Point in `_extract_team_stats()`

Replace lines 5570-5580 in `backend/main.py`:

```python
    # Build stats dict
    stats_dict: dict = {}
    for s in stats_raw:
        if isinstance(s, dict):
            stat = s.get("stat", {})
            if isinstance(stat, dict):
                sid = str(stat.get("stat_id", ""))
                key = stat_id_map.get(sid, sid)
                val = stat.get("value", "")
                if key:
                    # K-18: Sanitize impossible stat values from Yahoo
                    stats_dict[key] = _sanitize_stat_value(key, val)
```

---

## 5. Test Case Specification

### 5.1 Unit Test Example

```python
def test_extract_team_stats_sanitizes_negative_values():
    """K-18: -1 GS from Yahoo should become 0, not display as -1."""
    from backend.main import _extract_team_stats, _sanitize_stat_value
    
    # Direct function tests
    assert _sanitize_stat_value('GS', '-1') == 0
    assert _sanitize_stat_value('GS', '5') == 5
    assert _sanitize_stat_value('ERA', '-1') is None
    assert _sanitize_stat_value('ERA', '3.50') == 3.50
    assert _sanitize_stat_value('IP', '-1') == 0
    assert _sanitize_stat_value('W', '-1') == 0
    
    # Integration test with mock Yahoo response
    mock_team_entry = {
        "team_key": "428.l.12345.t.1",
        "name": "Test Team",
        "team_stats": {
            "stats": [
                {"stat": {"stat_id": "62", "value": "-1"}},  # GS
                {"stat": {"stat_id": "23", "value": "-1"}},  # W
                {"stat": {"stat_id": "28", "value": "15"}},  # K
                {"stat": {"stat_id": "26", "value": "-1"}},  # ERA
                {"stat": {"stat_id": "21", "value": "-1"}},  # IP
            ]
        }
    }
    
    team_key, team_name, stats = _extract_team_stats(mock_team_entry)
    
    assert stats.get('GS') == 0, f"GS should be 0, got {stats.get('GS')}"
    assert stats.get('W') == 0, f"W should be 0, got {stats.get('W')}"
    assert stats.get('K') == 15, f"K should be 15, got {stats.get('K')}"
    assert stats.get('ERA') is None, f"ERA should be None, got {stats.get('ERA')}"
    assert stats.get('IP') == 0, f"IP should be 0, got {stats.get('IP')}"
```

### 5.2 Expected Input/Output Table

| Input (from Yahoo) | Stat | Output (sanitized) | Display |
|-------------------|------|-------------------|---------|
| `"-1"` | GS (counting) | `0` | `0` |
| `"-1"` | W (counting) | `0` | `0` |
| `"-1"` | SV (counting) | `0` | `0` |
| `"-1"` | K (counting) | `0` | `0` |
| `"-1"` | ERA (ratio) | `None` | `—` |
| `"-1"` | WHIP (ratio) | `None` | `—` |
| `"-1"` | AVG (ratio) | `None` | `—` |
| `"15"` | K (counting) | `15` | `15` |
| `"3.50"` | ERA (ratio) | `3.50` | `3.500` |

---

## 6. Implementation Checklist

- [ ] Add `_COUNTING_STATS` and `_RATIO_STATS` sets at module level (near `_YAHOO_STAT_FALLBACK`)
- [ ] Add `_sanitize_stat_value()` helper function
- [ ] Modify `_extract_team_stats()` to use the sanitizer
- [ ] Add unit test in `tests/test_matchup_preseason.py` or new `tests/test_stat_validation.py`
- [ ] Verify frontend still displays correctly (em-dash for None, 0 for zero)
- [ ] Test with actual Yahoo API response (staging or production)

---

## 7. Files to Modify

| File | Lines | Change |
|------|-------|--------|
| `backend/main.py` | 5455-5467 | Add `_COUNTING_STATS`, `_RATIO_STATS`, `_sanitize_stat_value` after `_YAHOO_STAT_FALLBACK` |
| `backend/main.py` | 5577-5579 | Modify value assignment to call `_sanitize_stat_value(key, val)` |
| `tests/test_matchup_preseason.py` | new | Add test cases for stat sanitization |

---

## 8. Edge Cases Considered

1. **Empty string values:** Handled → returns `None`
2. **None values:** Handled → returns `None`
3. **Non-numeric strings:** Pass through unchanged (conservative)
4. **IP decimals:** Preserved (e.g., `6.1` innings → `6.1`)
5. **Unknown stat IDs:** Pass through unchanged (forward compatibility)
6. **Zero values:** Preserved as `0` (not confused with `-1`)
7. **Positive floats:** Preserved (e.g., `3.50` ERA)

---

## 9. Related Code References

- **Frontend masking (F6):** `frontend/app/(dashboard)/fantasy/matchup/page.tsx:22-23`
- **Schema definition:** `backend/schemas.py:518-521` (MatchupTeamOut.stats: dict)
- **Fallback stat map:** `backend/main.py:5457-5467` (_YAHOO_STAT_FALLBACK)
- **Extraction function:** `backend/main.py:5537-5585` (_extract_team_stats)
- **Endpoint:** `backend/main.py:5470-5665` (get_fantasy_matchup)

---

## 10. Verification Plan

After implementation, verify with:

```bash
# Run specific test
pytest tests/test_matchup_preseason.py -v -k sanitize

# Test matchup endpoint manually (if Yahoo configured)
curl -H "X-API-Key: $API_KEY" \
  https://your-domain.com/api/fantasy/matchup | jq '.my_team.stats.GS'
# Should return 0 (not -1) at start of week
```

---

**END OF SPECIFICATION**
