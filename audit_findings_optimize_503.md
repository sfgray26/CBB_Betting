# Audit Findings: /api/fantasy/roster/optimize 503 Error

**Date**: 2026-06-23  
**Module**: Roster Optimization Endpoint  
**Severity**: P0 - Blocks highest-value fantasy manager action

## Endpoint Structure

**Location**: `backend/routers/fantasy.py:4429-4762`  
**Method**: POST  
**Route**: `/api/fantasy/roster/optimize`

## Error Flow Analysis

### 503 Error Sources

The endpoint can return 503 from TWO distinct code paths:

1. **Line 4485-4490**: Yahoo client initialization failure
   ```python
   try:
       client = get_yahoo_client()
   except YahooAuthError as exc:
       raise HTTPException(
           status_code=503,
           detail="Yahoo not configured -- set YAHOO_REFRESH_TOKEN",
       ) from exc
   ```

2. **Line 4497-4498**: Yahoo roster fetch failure
   ```python
   try:
       raw_players = client.get_roster(team_key=team_key)
   except YahooAuthError as exc:
       raise HTTPException(status_code=503, detail=str(exc)) from exc
   ```

### Yahoo Client Analysis

**Location**: `backend/fantasy_baseball/yahoo_client_resilient.py`

**Initialization Requirements** (lines 186-191):
- `YAHOO_CLIENT_ID` env var must be set
- `YAHOO_CLIENT_SECRET` env var must be set
- If missing → raises `YahooAuthError("YAHOO_CLIENT_ID and YAHOO_CLIENT_SECRET must be set in .env")`

**Token Refresh Flow** (lines 228-252):
- Requires `YAHOO_REFRESH_TOKEN` env var
- If missing → raises `YahooAuthError("No refresh token stored. Run: python -m backend.fantasy_baseball.yahoo_client_resilient --auth")`
- If refresh fails → raises `YahooAuthError(f"Token refresh failed: {response.status_code} — {response.text}")`

**Circuit Breaker** (lines 173, 1990-1995):
- `failure_threshold=3`
- `recovery_timeout=300` (5 minutes)
- When OPEN → raises `CircuitOpenError` which becomes `YahooAPIError(..., 503)`
- BUT: caught at line 4499-4500 → returns 502, not 503

### Test Coverage Analysis

**Test File**: `tests/test_roster_optimize_api.py`

**Test Count**: 18 tests  
**Test Pattern**: All tests mock `get_yahoo_client()` successfully

**Critical Gap**: NO test validates behavior when:
- Yahoo client initialization fails
- `get_yahoo_client()` raises `YahooAuthError`
- Circuit breaker is OPEN

**Test Example** (lines 42-104):
```python
mock_client = MagicMock()
mock_client.get_roster.return_value = mock_roster

with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
    response = fantasy_client.post("/api/fantasy/roster/optimize", json={...})
```

This pattern assumes `get_yahoo_client()` never raises exceptions.

## Root Cause Hypothesis

**Primary Hypothesis**: Yahoo OAuth credentials are misconfigured or expired on Railway production

**Evidence**:
1. 503 error maps directly to `YahooAuthError` exception handling
2. The frontend shows "Optimize failed: Failed to fetch" → 503 triggers fetch failure
3. No graceful degradation or retry logic exists
4. Tests never validate error paths

**Secondary Possibility**: Circuit breaker is OPEN from repeated Yahoo API failures
- This would return 502 (per line 4499-4500), not 503
- But worth investigating as contributory factor

## Data Flow Dependencies

The endpoint depends on:
1. Yahoo OAuth credentials (client_id, client_secret, refresh_token)
2. Yahoo API availability (fantasy sports v2 API)
3. Database `PlayerIDMapping` for Yahoo→BDL ID resolution
4. Database `PlayerScore` for rolling 14-day stats (fallback to projection system)
5. OR-Tools constraint solver (line 4669-4674)

## Breaking Points

**P0 - Current Blocker**:
- Yahoo client initialization fails → 503, entire endpoint dead

**P1 - Cascading Issues**:
- PlayerScores missing/stale → fallback to projections, but inconsistency in scoring
- No upstream status indicator (frontend has no way to know if Yahoo is down)

**P2 - Missing Features**:
- No cache layer for roster data (every call hits Yahoo API)
- No offline/fallback mode
- No circuit breaker status exposed to frontend

## Regression Risk

**HIGH** - The optimize endpoint is integrated with:
- Frontend: "Optimize Lineup" button on My Roster page
- War Room module (may rely on optimize logic)
- Weekly Preview projections
- Constraint budget calculations

## Recommendations

### Immediate Fix (Tier 1)
1. **Add graceful Yahoo auth failure handling**:
   - Return structured error with recovery instructions
   - Include frontend-ready error codes (e.g., `YAHOO_AUTH_EXPIRED`, `YAHOO_NOT_CONFIGURED`)

2. **Add health check endpoint** for Yahoo connectivity:
   - `GET /api/fantasy/yahoo-health` → returns {status: "healthy" | "degraded", last_success, circuit_state}
   - Frontend can poll this to disable "Optimize" button proactively

3. **Hardening test coverage**:
   - Add test for `YahooAuthError` on client init
   - Add test for `YahooAuthError` on get_roster
   - Add test for circuit breaker OPEN state

### Medium-Term (Tier 2)
4. **Add stale cache fallback**:
   - Cache last successful roster fetch with 24-hour TTL
   - Serve stale roster with warning if Yahoo API is down
   - Prevents total outage during Yahoo blips

5. **Add circuit breaker 半-open state logging**:
   - Emit structured logs when circuit transitions
   - Create Railway alert for repeated circuit openings

## Next Steps

**Loop Iteration 1** should focus on:
- Validating which specific 503 path is being hit (client init vs roster fetch)
- Adding structured error responses
- Creating Yahoo health check endpoint
- Adding error-path tests

**CONFIDENCE LEVEL**: HIGH - Root cause is clearly Yahoo auth, audit trail is complete
