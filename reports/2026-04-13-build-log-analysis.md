# Build Log Analysis Report

> **Analysis Date:** April 13, 2026  
> **Log File:** `reports/logs.1776113724548.json`  
> **Analyst:** Kimi CLI

---

## Executive Summary

**Status:** ⚠️ **ISSUES FOUND**  
The build logs contain **Pydantic validation errors** in the MLB odds polling system. The errors are causing betting line data to fail validation when BDL API returns NULL values for spread/total odds.

---

## 🔴 Critical Issues Found

### Issue 1: Pydantic Validation Errors - MLB Betting Odds

**Location:** `backend/services/balldontlie.py` - `get_mlb_odds()` function  
**Error Count:** 4 validation errors per affected game  
**Affected Game ID:** 5058010 (and potentially others)

#### Error Details

```
backend.services.balldontlie - ERROR - get_mlb_odds(game_id=5058010) page=0 failed: 
4 validation errors for BDLResponse[MLBBettingOdd]

1. data.5.spread_home_value
   Value error, spread/total value must be a string, got NoneType: None
   [type=value_error, input_value=None, input_type=NoneType]

2. data.5.spread_home_odds
   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]

3. data.5.spread_away_value
   Value error, spread/total value must be a string, got NoneType: None
   [type=value_error, input_value=None, input_type=NoneType]

4. data.5.spread_away_odds
   (implied from context)
```

#### Root Cause

The BDL API is returning `NULL` values for spread betting lines when:
- No spread is available for the game
- The game is too close to start time
- The sportsbook hasn't set lines yet
- The game is a moneyline-only event

The Pydantic model `MLBBettingOdd` expects:
- `spread_home_value`: `str` (but receives `None`)
- `spread_home_odds`: `int` (but receives `None`)
- `spread_away_value`: `str` (but receives `None`)
- `spread_away_odds`: `int` (but receives `None`)

---

## 📊 Impact Analysis

### Immediate Impact

| Aspect | Effect | Severity |
|--------|--------|----------|
| **Odds Data** | Complete odds record dropped for affected games | HIGH |
| **Betting Lines** | No spread data available for users | HIGH |
| **Pipeline** | Individual game failures, pipeline continues | MEDIUM |
| **User Experience** | Missing betting information | MEDIUM |

### Affected Games

From the log sample:
- **Game ID 5058010**: Failed entirely (all spread data lost)
- Other games may be affected but not visible in this log segment

---

## 🔧 Recommended Fixes

### Fix 1: Make Spread Fields Optional (Immediate - 30 min)

**File:** `backend/schemas.py` or betting odds schema

```python
# CURRENT (causing errors):
class MLBBettingOdd(BaseModel):
    spread_home_value: str
    spread_home_odds: int
    spread_away_value: str
    spread_away_odds: int

# FIXED (allows NULL):
class MLBBettingOdd(BaseModel):
    spread_home_value: Optional[str] = None
    spread_home_odds: Optional[int] = None
    spread_away_value: Optional[str] = None
    spread_away_odds: Optional[int] = None
```

### Fix 2: Add Pre-Validation Handling (Recommended - 1 hour)

```python
@validator('spread_home_value', 'spread_away_value', pre=True)
def handle_null_spread_values(cls, v):
    """Convert None to empty string for spread values."""
    if v is None:
        return ""
    return v

@validator('spread_home_odds', 'spread_away_odds', pre=True)
def handle_null_spread_odds(cls, v):
    """Convert None to 0 for spread odds."""
    if v is None:
        return 0
    return v
```

### Fix 3: Graceful Degradation (Long-term - 2 hours)

```python
def get_mlb_odds(game_id: int):
    try:
        response = fetch_odds_from_bdl(game_id)
        odds = MLBBettingOdd.parse_obj(response)
        return odds
    except ValidationError as e:
        # Log the error but don't fail
        logger.warning(f"Odds validation failed for game {game_id}: {e}")
        # Return partial data (moneyline only)
        return extract_moneyline_only(response)
```

---

## 📋 Additional Observations

### Normal Operations (Healthy)

The following services are operating correctly:

| Service | Status | Notes |
|---------|--------|-------|
| Async Job Queue Processor | ✅ Healthy | Running every 5 seconds |
| MLB Odds Poll | ✅ Healthy | Executing every 5 minutes |
| BallDontLie API Client | ✅ Healthy | Session created, authenticated |
| Job Locking | ✅ Healthy | Lock 100001 acquired/released correctly |

### Performance Metrics

From the logs:
- **MLB Odds Poll**: 8 games processed, 7 with odds
- **Execution Time**: 914ms for full odds poll
- **Success Rate**: 7/8 games (87.5%)

---

## 🎯 Action Items

### Immediate (Today)

1. **Update Pydantic Schema** (30 min)
   - Make spread fields Optional
   - Deploy to production
   - Monitor for resolution

### Short-term (This Week)

2. **Add Validation Tests** (1 hour)
   - Test NULL spread values
   - Test partial odds data
   - Test complete odds data

3. **Implement Fallback Logic** (2 hours)
   - Return moneyline when spread unavailable
   - Add "lines not available" messaging
   - Log when games lack full odds

### Long-term (Next Sprint)

4. **Odds Quality Dashboard** (4 hours)
   - Track odds availability by game
   - Monitor validation error rates
   - Alert on persistent issues

---

## 📈 Validation Error Frequency

Based on this log sample (10-minute window):

| Metric | Count | Rate |
|--------|-------|------|
| Total Odds Requests | 1 | - |
| Validation Errors | 4 | 100% of affected game |
| Games Affected | 1 | 12.5% (1 of 8 games) |
| **Pipeline Success** | **7/8** | **87.5%** |

---

## Conclusion

The build logs reveal a **known issue** with Pydantic validation on betting odds data. The BDL API occasionally returns NULL for spread lines, which the current schema doesn't handle gracefully.

**Recommendation:** Implement Fix 1 (Optional fields) immediately to prevent data loss. Consider Fix 2 (pre-validation) for more robust handling.

---

*Analysis based on 10-minute log sample from 2026-04-13 20:19-20:27 UTC*
