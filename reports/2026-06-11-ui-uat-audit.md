# UI UAT Audit: 2026-06-11 07:30 ET

## Environment
- **URL:** https://observant-benevolence-production.up.railway.app/
- **Viewport:** 1920×1080
- **Browser:** Chrome (via DevTools MCP)
- **Auth Header:** `API_KEY_USER1`
- **Frontend API Base:** `https://fantasy-app-production-5079.up.railway.app` (observed in network tab)

## Executive Summary

**Overall Status: ❌ FAILED — P0 blocking issues prevent any Fantasy Baseball functionality from loading.**

The production frontend loads and the login form accepts `API_KEY_USER1`, but every protected API call returns `401 Invalid API key`. Additionally, the frontend is configured to call a separate backend domain (`fantasy-app-production-5079.up.railway.app`) rather than the same origin, and the newly wired `/api/fantasy/global-freshness` endpoint returns `404 Not Found` on the deployed backend.

## Screenshots

| Page | Status | Screenshot Path |
|------|--------|-----------------|
| Landing / Login | ⚠️ degraded | `reports/uat/2026-06-11-landing.png` |
| Dashboard | ❌ failed | `reports/uat/2026-06-11-dashboard.png` |
| Budget | ❌ failed | `reports/uat/2026-06-11-budget.png` |
| War Room / Matchup | ❌ failed | `reports/uat/2026-06-11-matchup.png` |
| Roster | ❌ failed | `reports/uat/2026-06-11-roster.png` |
| Waiver Wire | ❌ failed | `reports/uat/2026-06-11-waiver.png` |
| Streaming Station | ❌ failed | `reports/uat/2026-06-11-streaming.png` |
| 404 Error Page | ✅ pass | `reports/uat/2026-06-11-404.png` |

## Console Errors

| Page | Severity | Message | Count |
|------|----------|---------|-------|
| All | error | Failed to load resource: the server responded with a status of 401 () | many |
| All | error | Failed to load resource: the server responded with a status of 404 () | several |
| Dashboard / Roster | error | Uncaught Error: Minified React error #419 | 9+ |
| Dashboard / Roster | error | Error: 401: Invalid API key | many |
| Dashboard / Roster | error | ErrorBoundary caught error: Error: 401: Invalid API key | many |

## API Health

Direct fetch results using `API_KEY_USER1` against `https://fantasy-app-production-5079.up.railway.app`:

| Endpoint | Status | Latency | Notes |
|----------|--------|---------|-------|
| `GET /api/fantasy/budget` | 401 | 256ms | Invalid API key |
| `GET /api/fantasy/matchup` | 401 | 440ms | Invalid API key |
| `GET /api/fantasy/lineup/current` | 401 | 408ms | Invalid API key |
| `GET /api/fantasy/global-freshness` | 404 | 134ms | **Endpoint does not exist on deployed backend** |
| `GET /api/dashboard` | 401 | 645ms | Invalid API key |
| `GET /api/fantasy/roster` | 401 | 270ms | Invalid API key |
| `GET /health` (same origin) | 404 | — | Frontend Next.js catches route |

## Page-by-Page Results

### Phase 1: Landing / Login
- ✅ Page loads without 5xx
- ✅ Layout renders (logo, API key form, footer)
- ⚠️ Landing text includes: "Could not reach the backend at https://fantasy-app-production-5079.up.railway.app"
- ❌ Favicon status not explicitly verified

### Phase 2: Budget Panel
- ❌ Budget panel shows "Failed to load budget" / "401: Invalid API key"
- ❌ No budget numbers render
- ✅ Retry button is present

### Phase 3: Matchup View (War Room)
- ❌ Shows "401: Invalid API key"
- ❌ No matchup categories render
- ⚠️ Scoreboard endpoint returned 200 on one request, but matchup/roster/budget all 401

### Phase 4: Lineup / Roster View
- ❌ Roster page stuck on "Loading roster…" indefinitely
- ❌ Does not surface the 401 error to the user
- ❌ No player cards render

### Phase 5: Waiver Wire
- ❌ Shows "Failed to load waiver wire" / "401: Invalid API key"
- ❌ No waiver targets render

### Phase 6: Streaming Station
- ❌ Shows "401: Invalid API key"
- ❌ No streaming data render

### Phase 7: Error State Resilience
- ✅ Non-existent route `/nonexistent-route` renders styled Next.js 404 page (not raw JSON)
- ❌ Error boundaries catch 401s but still flood console with repeated identical errors

## Findings

### P0 (Blocking)
1. **Invalid API key in production.** Every protected endpoint returns `401 Invalid API key` for the documented key `API_KEY_USER1`. No Fantasy Baseball data can load.
2. **Missing `/api/fantasy/global-freshness` endpoint.** The frontend was just updated to call this endpoint, but the deployed backend returns `404 Not Found`. This will break the new FreshnessBadge on all pages.
3. **Cross-origin API configuration.** Frontend at `observant-benevolence-production.up.railway.app` calls `fantasy-app-production-5079.up.railway.app`. This is a CORS/cookie/auth risk and contradicts the production target documented in the skill.

### P1 (Degraded)
4. **Roster page does not handle 401 gracefully.** It remains on "Loading roster…" instead of showing an error message with a retry action.
5. **Console flooded with React error #419.** Appears repeatedly on Dashboard and Roster pages, likely caused by hydration/ErrorBoundary issues when API calls fail.
6. **Dashboard widgets all fail independently.** Each widget calls its own endpoint and shows "Failed to load. The widget will retry automatically." with no actionable user feedback.

### P2 (Polish)
7. **Landing page pre-emptively reports backend unreachable.** The login form shows "Could not reach the backend" even though the backend is reachable (it just rejects the API key).

## Recommendations

1. **Fix the API key issue.** Verify `API_KEY_USER1` exists and is active in the production backend, or update the audit documentation with the correct key.
2. **Deploy the backend `/api/fantasy/global-freshness` endpoint** before the new FreshnessBadge frontend changes go live; otherwise all Fantasy pages will show additional 404 errors.
3. **Align `NEXT_PUBLIC_API_URL` with the production target.** Either serve API from the same origin (`observant-benevolence-production.up.railway.app`) or document why a separate backend domain is required.
4. **Improve Roster loading state.** Surface API errors instead of leaving the spinner running indefinitely.
5. **Reduce console noise.** Suppress repeated identical 401 errors or batch them so ErrorBoundary does not log the same failure dozens of times.
6. **Add a global auth error banner** so users see a clear "Invalid API key" message instead of every widget failing silently.
