# LOOP ITERATION 1 (REVISED)
**Date**: 2026-06-23  
**Objective**: Fix 503 error on `/api/fantasy/roster/optimize` endpoint

---

## Phase 1: INGEST & AUDIT ✓ COMPLETED

### Complete System Flow Analysis

**Backend → Frontend Error Chain**:
1. `backend/routers/fantasy.py:4485-4490` raises `HTTPException(status_code=503, detail="...")`
2. `frontend/lib/api.ts:71-84` catches non-200 response, extracts `detail` field
3. Throws `Error("503: Yahoo not configured -- set YAHOO_REFRESH_TOKEN")`
4. `page.tsx:944` `onError` catches: `setMoveError("Optimize failed: Failed to fetch")`
5. User sees: "Optimize failed: Failed to fetch" (generic, not actionable)

**Root Cause**: Yahoo authentication failure (one of two paths):
- Path A: `get_yahoo_client()` init fails → missing YAHOO_CLIENT_ID/SECRET or YAHOO_REFRESH_TOKEN
- Path B: `client.get_roster()` fails → token refresh failed (invalid_grant, expired, etc.)

**Schema Constraints**:
- `RosterOptimizeResponse` does NOT have `error_code` field
- `freshness` field is REQUIRED (non-optional)
- Cannot return `freshness=None` without breaking schema validation

**Frontend Expectations**:
- Expects successful response (200 OK) with `success: boolean` field
- OR HTTP error (non-200) with `detail` field
- `apiFetch` throws Error on any non-200 status

---

## Phase 2: PLAN (REVISED)

### Updated Surgical Approach

**Constraint Discovered**: Cannot change HTTP 503 → 200 with `success=False` without:
1. Adding optional `error_code` field to `RosterOptimizeResponse` schema
2. Making `freshness` field optional in schema
3. Updating frontend to handle `success=False` responses

**Better Approach**: Keep HTTP 503, but make `detail` field **actionable** and **structured**

### Change 1: Enhanced Error Messages (backend/routers/fantasy.py)

**Lines 4485-4500** - Replace generic error details with structured, actionable messages:

```python
# Current (line 4487-4490):
raise HTTPException(
    status_code=503,
    detail="Yahoo not configured -- set YAHOO_REFRESH_TOKEN",
) from exc

# Enhanced:
raise HTTPException(
    status_code=503,
    detail={
        "error_code": "YAHOO_NOT_CONFIGURED",
        "message": "Yahoo Fantasy API credentials not configured",
        "recovery_hint": "Set YAHOO_REFRESH_TOKEN environment variable and restart the server",
        "documentation_url": "https://github.com/your-repo/wiki/yahoo-setup"
    }
) from exc
```

Similarly for line 4498 (get_roster failure):
```python
raise HTTPException(
    status_code=503,
    detail={
        "error_code": "YAHOO_AUTH_EXPIRED",
        "message": f"Yahoo authentication failed: {str(exc)}",
        "recovery_hint": "Refresh token may be expired. Re-run OAuth flow: python -m backend.fantasy_baseball.yahoo_client_resilient --auth",
    }
) from exc
```

### Change 2: Add Yahoo Health Check Endpoint (backend/routers/fantasy.py)

**Location**: After line 5376 (after `get_job_status` endpoint)

```python
@router.get("/api/fantasy/yahoo-health")
async def yahoo_health():
    """
    Health check for Yahoo Fantasy API connectivity.
    
    Returns structured status for frontend diagnostics and proactive UI disabling.
    """
    from backend.fantasy_baseball.yahoo_client_resilient import _client, _client_lock
    
    health_status = {
        "status": "unknown",  # healthy | degraded | down
        "circuit_state": None,  # closed | open | half_open
        "last_success_at": None,
        "error": None,
        "recovery_hint": None,
    }
    
    # Check client singleton status
    if _client is None:
        health_status.update({
            "status": "down",
            "error": "Yahoo client not initialized",
            "recovery_hint": "Check YAHOO_CLIENT_ID, YAHOO_CLIENT_SECRET, YAHOO_REFRESH_TOKEN environment variables"
        })
        return health_status
    
    # Check circuit breaker
    try:
        if hasattr(_client, 'circuit'):
            cb_stats = _client.circuit.get_stats()
            health_status["circuit_state"] = cb_stats["state"]
            if cb_stats["state"] == "open":
                health_status["status"] = "degraded"
                health_status["error"] = "Circuit breaker is OPEN after repeated failures"
                health_status["recovery_hint"] = "Wait 5 minutes for circuit recovery or investigate API failures"
        
        # Try a lightweight API call (get league metadata)
        league_meta = _client.get_league()
        health_status["status"] = "healthy"
        health_status["last_success_at"] = datetime.now(ZoneInfo("America/New_York")).isoformat()
        
    except YahooAuthError as exc:
        health_status["status"] = "down"
        health_status["error"] = f"Authentication failed: {str(exc)}"
        health_status["recovery_hint"] = "Re-run OAuth flow: python -m backend.fantasy_baseball.yahoo_client_resilient --auth"
    except Exception as exc:
        health_status["status"] = "degraded"
        health_status["error"] = f"API check failed: {str(exc)}"
    
    return health_status
```

### Change 3: Add Error-Path Tests (tests/test_roster_optimize_api.py)

**Location**: After line 572

```python
def test_optimize_returns_503_on_yahoo_client_init_failure(self, fantasy_client):
    """Endpoint returns HTTP 503 with structured detail when get_yahoo_client() fails."""
    from backend.fantasy_baseball.yahoo_client_resilient import YahooAuthError
    
    with patch("backend.routers.fantasy.get_yahoo_client", side_effect=YahooAuthError("YAHOO_CLIENT_ID missing")):
        response = fantasy_client.post("/api/fantasy/roster/optimize", json={})
    
    assert response.status_code == 503
    data = response.json()
    assert "detail" in data
    assert isinstance(data["detail"], dict)  # Structured error response
    assert data["detail"]["error_code"] in ("YAHOO_NOT_CONFIGURED", "YAHOO_AUTH_EXPIRED")
    assert "recovery_hint" in data["detail"]


def test_optimize_returns_503_on_get_roster_auth_failure(self, fantasy_client):
    """Endpoint returns HTTP 503 with structured detail when get_roster() auth fails."""
    from backend.fantasy_baseball.yahoo_client_resilient import YahooAuthError
    
    mock_client = MagicMock()
    mock_client.get_roster.side_effect = YahooAuthError("Token refresh failed: 401 — invalid_grant")
    
    with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
        response = fantasy_client.post("/api/fantasy/roster/optimize", json={})
    
    assert response.status_code == 503
    data = response.json()
    assert data["detail"]["error_code"] == "YAHOO_AUTH_EXPIRED"
    assert "Token refresh failed" in data["detail"]["message"]
    assert "recovery_hint" in data["detail"]
```

---

## Phase 3: EXECUTE (BLOCKED - AWAITING USER CHECKPOINT)

### Files to Modify (3 max, within constraint):
1. `backend/routers/fantasy.py` (lines 4485-4500, plus new endpoint)
2. `tests/test_roster_optimize_api.py` (add 2 tests)

### Rollback Criteria:
- If tests fail → revert fantasy.py changes
- If health check endpoint breaks existing routes → revert endpoint addition

---

## Phase 4: VALIDATION (PLANNED)

```bash
# Run new tests
venv/Scripts/python -m pytest tests/test_roster_optimize_api.py::TestRosterOptimizeEndpoint::test_optimize_returns_503_on_yahoo_client_init_failure -v
venv/Scripts/python -m pytest tests/test_roster_optimize_api.py::TestRosterOptimizeEndpoint::test_optimize_returns_503_on_get_roster_auth_failure -v

# Run regression tests
venv/Scripts/python -m pytest tests/test_roster_optimize_api.py -v

# Test health endpoint
curl http://localhost:8000/api/fantasy/yahoo-health
```

---

## Phase 5: REPORT & REFINE (IN PROGRESS)

### Current Confidence: MEDIUM (upgraded from LOW)

**Why Upgraded**:
- Complete frontend error chain mapped
- Schema constraints identified
- Surgical plan honors existing HTTP 503 semantics (no breaking changes)
- Structured error details improve UX without frontend changes

**Remaining Risks**:
- Don't have Railway production logs to confirm which 503 path is active
- Haven't validated that FastAPI supports dict `detail` field (need to verify)
- Frontend `apiFetch` expects `detail` as string OR object - should work but untested

### Recommendation Before Execution:

**SMALL PRE-STEP**: Add `/api/fantasy/yahoo-health` endpoint ONLY first (no optimize changes)
1. Implement health check endpoint independently
2. Test it locally
3. If Railway access available, deploy and check production health status
4. Use health check output to confirm exact error path
5. THEN implement structured 503 errors based on confirmed diagnosis

### NEXT ITERATION:
If confidence remains MEDIUM after health check, proceed with full 3-file change.
If health check reveals unexpected state, pivot to smaller scope.

---

**ITERATION 1 STATUS**: READY FOR USER CHECKPOINT
**RECOMMENDATION**: Approve health-check-only first step, then full plan
