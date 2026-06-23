# LOOP ITERATION 2 - SCOPE AND PLAN

**Date**: 2026-06-23  
**Triggered By**: Loop Iteration 1 Phase 1 diagnostic findings  
**Status**: Awaiting user approval

---

## CRITICAL FINDING FROM ITERATION 1

### Discovery Summary
The health endpoint diagnostic revealed that the optimize endpoint failure is **NOT** caused by Yahoo authentication issues (as originally hypothesized), but by a **data validation bug** in roster processing.

### Root Cause Identified

**Error**: `AttributeError: 'bool' object has no attribute 'upper'`  
**Location**: `backend/routers/fantasy.py:4782` in `_is_il_designated()` function  
**Code**: `status = (player.get("status") or "").upper().strip()`  
**Issue**: `player.get("status")` returns boolean instead of expected string

**Error Log**:
```
File "/app/backend/routers/fantasy.py", line 4563, in optimize_roster
    if _is_il_designated(p):
File "/app/backend/routers/fantasy.py", line 4782, in _is_il_designated
    status = (player.get("status") or "").upper().strip()
AttributeError: 'bool' object has no attribute 'upper'
```

### Impact Assessment

**Severity**: P0 - Complete endpoint failure
- Optimize endpoint crashes BEFORE reaching Yahoo client code
- No graceful error handling for this path
- Affects ALL roster optimization requests
- Would fail even if Yahoo credentials were properly configured

**Data Issue**: Malformed roster player objects
- Expected: `status` as string ("IL", "playing", "probable", etc.)
- Actual: `status` as boolean (True/False)
- Source: Unknown - possible Yahoo API format change or data corruption

---

## LOOP ITERATION 2 OBJECTIVE

Fix AttributeError in `_is_il_designated()` helper function and add defensive data validation to prevent similar crashes.

---

## SURGICAL EXECUTION PLAN

### Change 1: Fix `_is_il_designated()` Data Handling

**File**: `backend/routers/fantasy.py`  
**Location**: Line 4782  
**Scope**: 1 line change + 3 lines of defensive type checking

**BEFORE**:
```python
def _is_il_designated(player: dict) -> bool:
    """Return True if Yahoo status indicates an active IL designation."""
    status = (player.get("status") or "").upper().strip()
    return status in ("IL", "IL60", "D60", "D10", "D15", "O", "DND")
```

**AFTER**:
```python
def _is_il_designated(player: dict) -> bool:
    """Return True if Yahoo status indicates an active IL designation.
    
    Defensive: handles status as string, boolean, or None.
    Logs warning for unexpected types to aid data debugging.
    """
    raw_status = player.get("status")
    
    # Defensive: handle boolean status values (data corruption/API change)
    if isinstance(raw_status, bool):
        logger.warning(
            "Player %s has boolean status=%s - expected string. "
            "Treating as NOT IL-designated. Data may be corrupted.",
            player.get("name", "unknown"), raw_status
        )
        return False
    
    # Defensive: handle None or missing status
    if raw_status is None:
        return False
    
    # Expected case: status is string
    status = str(raw_status).upper().strip()
    return status in ("IL", "IL60", "D60", "D10", "D15", "O", "DND")
```

**Rationale**:
- Makes function robust to boolean, None, and string values
- Adds logging for data debugging (helps identify source of corruption)
- Returns safe default (False) for unexpected types
- Preserves original logic for valid string values

**Rollback**: If tests fail → revert to original and investigate data source

---

### Change 2: Add Roster Data Validation Wrapper

**File**: `backend/routers/fantasy.py`  
**Location**: After line 4495 (after `raw_players = client.get_roster(team_key=team_key)`)  
**Scope**: ~10 lines of validation code

**ADD**:
```python
    # Validate roster player data structure before processing
    # This catches data corruption early and provides actionable error messages
    for i, p in enumerate(raw_players):
        if not isinstance(p, dict):
            logger.error("Roster player at index %d is not a dict: %s", i, type(p))
            raise HTTPException(
                status_code=500,
                detail={
                    "error_code": "ROSTER_DATA_CORRUPTED",
                    "message": f"Roster player at index {i} has invalid type",
                    "index": i,
                    "type": str(type(p)),
                    "recovery_hint": "Check Yahoo API response format and player data structure"
                }
            )
        
        # Log warning for unexpected status types (helps debug data source)
        status_val = p.get("status")
        if status_val is not None and not isinstance(status_val, (str, bool)):
            logger.warning(
                "Player %s has unexpected status type=%s (value=%s)",
                p.get("name", "unknown"), type(status_val), status_val
            )
```

**Rationale**:
- Catches data corruption BEFORE it reaches optimize logic
- Provides structured error response for debugging
- Logs warnings for unexpected types (aids investigation)
- Minimal performance impact (single pass over roster)

**Rollback**: If validation blocks legitimate data → relax type checking

---

### Change 3: Add Data Validation Tests

**File**: `tests/test_roster_optimize_api.py`  
**Location**: After line 572  
**Scope**: ~40 lines (3 new test methods)

**ADD**:
```python
def test_optimize_handles_boolean_status_gracefully(self, fantasy_client):
    """optimize endpoint should handle boolean status values without crashing."""
    mock_roster = [
        {
            "player_key": "469.l.72586.p.111",
            "name": "Test Player",
            "team": "NYY",
            "positions": ["1B"],
            "selected_position": "1B",
            "status": True,  # Boolean instead of string - data corruption scenario
        }
    ]

    mock_client = MagicMock()
    mock_client.get_roster.return_value = mock_roster

    with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
        response = fantasy_client.post("/api/fantasy/roster/optimize", json={})

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    # Player with boolean status should not cause crash


def test_optimize_handles_none_status_gracefully(self, fantasy_client):
    """optimize endpoint should handle None status values without crashing."""
    mock_roster = [
        {
            "player_key": "469.l.72586.p.222",
            "name": "No Status Player",
            "team": "BOS",
            "positions": ["C"],
            "selected_position": "C",
            "status": None,  # Missing/None status
        }
    ]

    mock_client = MagicMock()
    mock_client.get_roster.return_value = mock_roster

    with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
        response = fantasy_client.post("/api/fantasy/roster/optimize", json={})

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_optimize_returns_structured_error_for_non_dict_player(self, fantasy_client):
    """optimize endpoint should return structured error for corrupted roster data."""
    mock_roster = [
        "not_a_dict",  # Completely malformed - should trigger validation error
        {
            "player_key": "469.l.72586.p.333",
            "name": "Valid Player",
            "team": "BAL",
            "positions": ["SS"],
            "selected_position": "SS",
            "status": "playing",
        }
    ]

    mock_client = MagicMock()
    mock_client.get_roster.return_value = mock_roster

    with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
        response = fantasy_client.post("/api/fantasy/roster/optimize", json={})

    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert data["detail"]["error_code"] == "ROSTER_DATA_CORRUPTED"
    assert "index" in data["detail"]
    assert data["detail"]["index"] == 0
```

**Rationale**:
- Tests the boolean status fix (Change 1)
- Tests None status handling (edge case)
- Tests data validation wrapper (Change 2)
- Ensures structured error responses

**Rollback**: If tests fail → fix implementation logic

---

## VALIDATION PLAN

### Test Execution

```bash
# Run new tests
venv/Scripts/python -m pytest tests/test_roster_optimize_api.py::TestRosterOptimizeEndpoint::test_optimize_handles_boolean_status_gracefully -v
venv/Scripts/python -m pytest tests/test_roster_optimize_api.py::TestRosterOptimizeEndpoint::test_optimize_handles_none_status_gracefully -v
venv/Scripts/python -m pytest tests/test_roster_optimize_api.py::TestRosterOptimizeEndpoint::test_optimize_returns_structured_error_for_non_dict_player -v

# Run regression tests
venv/Scripts/python -m pytest tests/test_roster_optimize_api.py -v
```

### Railway Deployment Test

```bash
# Deploy changes
railway redeploy --yes

# Test optimize endpoint
curl -X POST https://fantasy-app-production-5079.up.railway.app/api/fantasy/roster/optimize \
  -H "Content-Type: application/json" -d '{}'

# Check logs for data validation warnings
railway logs --lines 20
```

---

## SUCCESS CRITERIA

1. ✓ Optimize endpoint returns 200 (not 500) when processing roster with boolean status
2. ✓ Tests pass for boolean, None, and malformed data scenarios  
3. ✓ Railway deployment shows no AttributeError in logs
4. ✓ Data validation logs help identify source of boolean status values
5. ✓ No regression in existing test suite

---

## ROLLBACK CRITERIA

1. Optimize endpoint still fails with AttributeError → revert and investigate deeper
2. Tests fail after implementation → revert and fix logic
3. Railway deployment breaks other endpoints → revert validation wrapper
4. Performance regression from validation → optimize or remove validation

---

## BLOCKERS / RISKS

**Unknown**: Source of boolean status values
- Could be Yahoo API format change
- Could be data migration issue
- Could be client-side data corruption

**Mitigation**: 
- Added logging to help identify source
- Validation wrapper provides early detection
- Structured errors aid debugging

**Data Investigation** (post-fix):
- Check Yahoo API documentation for recent changes
- Audit database for malformed player records
- Add monitoring for data type violations

---

## FILES TO MODIFY

**Total**: 3 files (within constraint)

1. `backend/routers/fantasy.py` - fix `_is_il_designated()` (~10 lines)
2. `backend/routers/fantasy.py` - add data validation wrapper (~10 lines)
3. `tests/test_roster_optimize_api.py` - add validation tests (~40 lines)

**Total Lines**: ~60 lines across 3 files

---

## CONFIDENCE ASSESSMENT

**CONFIDENCE**: HIGH

**Reasoning**:
- Root cause clearly identified from Railway logs
- Surgical fix addresses exact error location
- Defensive programming prevents similar crashes
- Tests cover the specific failure scenario
- Validation wrapper provides early detection

**Remaining Risks**:
- Don't know WHY status is boolean (data source investigation needed post-fix)
- May discover additional data corruption issues during validation
- Performance impact of validation (should be minimal)

---

## APPROVAL REQUIRED

**User Decision Needed**:
- [ ] Approve Loop Iteration 2 as scoped above
- [ ] Modify scope (specify changes)
- [ ] Request additional investigation before proceeding
- [ ] Halt iteration and investigate data source first

---

**STATUS**: Awaiting user approval to proceed
