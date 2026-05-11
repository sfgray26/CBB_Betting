# UI UAT Audit: 2026-05-07 10:43

## Environment
- URL: https://observant-benevolence-production.up.railway.app/
- Backend: https://fantasy-app-production-5079.up.railway.app/
- Viewport: 1920×1080
- Browser: Chrome (via DevTools MCP)
- Auth: X-API-Key header (resolved from API_KEY_USER1 env var)

---

## Screenshots

| Page | Status | Screenshot Path |
|------|--------|-----------------|
| Landing (Login) | ⚠️ | `reports/uat/2026-05-07-landing.png` |
| Dashboard | ❌ | `reports/uat/2026-05-07-dashboard.png` |
| War Room (Matchup) | ⚠️ | `reports/uat/2026-05-07-war-room.png` |
| My Roster | ⚠️ | `reports/uat/2026-05-07-roster.png` |
| Waiver Wire | ⚠️ | `reports/uat/2026-05-07-waiver.png` |
| Streaming | ❌ | `reports/uat/2026-05-07-streaming.png` |
| 404 Error | ✅ | `reports/uat/2026-05-07-404.png` |

---

## Console Errors

| Page | Severity | Message | Count |
|------|----------|---------|-------|
| War Room | error | CORS policy blocked POST to `/api/fantasy/matchup/simulate` | 2 |
| War Room | error | Failed to load resource: net::ERR_FAILED (simulate) | 2 |
| Landing | warning | A form field element should have an id or name attribute | 1 |
| Landing | error | Failed to load resource: favicon.ico 404 | 1 |

---

## API Health

| Endpoint | Status | Latency | Notes |
|----------|--------|---------|-------|
| `GET /api/fantasy/budget` | 200 | 576ms | Returns correct schema. `ip_accumulated=0` (mocked). No UI integration. |
| `GET /api/fantasy/matchup` | 200 | 3,182ms | Slow. Returns live matchup data (8-7). No hardcoded budget values. |
| `GET /api/fantasy/lineup/current` | 422 | 200ms | **FAILS** — returns `lineup_date must be YYYY-MM-DD` even with valid date param. |
| `GET /api/fantasy/waiver?sort=need_score` | 200 | 164ms | Fast. Returns 25 player waiver targets with need scores. |
| `POST /api/fantasy/matchup/simulate` | CORS | — | **BLOCKED** — No `Access-Control-Allow-Origin` header on response. |
| `GET /api/dashboard` | 200 | <500ms | Returns data but frontend does not render it. |
| `GET /admin/portfolio/status` | 200 | <500ms | Healthy. |
| `GET /health` | 200 | <500ms | Healthy. |

---

## Findings

### P0 (Blocking)

1. **CORS Block on Matchup Simulate** (`POST /api/fantasy/matchup/simulate`)
   - The War Room page attempts to call the simulate endpoint but gets blocked by CORS.
   - Error: `No 'Access-Control-Allow-Origin' header is present on the requested resource.`
   - Impact: Simulation feature is completely non-functional in production.

2. **Streaming Page Infinite Loading**
   - The Streaming Station page shows "Loading waiver data..." indefinitely.
   - The underlying `GET /api/fantasy/waiver` API returns 200 in ~164ms.
   - Impact: Streaming feature is completely non-functional despite backend being healthy.

3. **Dashboard Empty Despite API Data**
   - `GET /api/dashboard` returns 200 with valid JSON (waiver targets, injury flags, lineup stats).
   - The main content area renders completely empty — no panels, no cards, no data.
   - Impact: Dashboard is non-functional; users cannot view lineup gaps, injuries, or waiver targets.

4. **Lineup Endpoint Always Returns 422**
   - `GET /api/fantasy/lineup/current?lineup_date=2026-05-07` returns `{"detail":"lineup_date must be YYYY-MM-DD"}`.
   - Tested with query param, without param, and with POST body — all return 422.
   - Impact: Lineup optimizer/current lineup API is completely broken.

### P1 (Degraded)

5. **Budget API Not Integrated into UI**
   - The `/api/fantasy/budget` endpoint returns correct data (acquisitions, IL, IP pace).
   - No page in the frontend calls this endpoint; there is no budget panel visible anywhere.
   - Latency is 576ms–3,059ms (exceeds the 500ms target for live Yahoo data).

6. **Matchup API Latency Too High**
   - `GET /api/fantasy/matchup` takes ~3,182ms.
   - This exceeds the 2,000ms acceptance threshold.
   - The UI renders a blank state while waiting; no loading skeleton is visible.

7. **My Roster & Waiver Wire Placeholders**
   - `/war-room/roster` shows "COMING NEXT — Slot-by-slot roster view"
   - `/war-room/waiver` shows "COMING NEXT — Filter-driven free agent rankings"
   - These are acknowledged work-in-progress but represent incomplete feature delivery.

8. **Favicon 404**
   - `GET /favicon.ico` returns 404 on every page load.
   - Minor but affects polish and browser tab UX.

### P2 (Polish)

9. **Form Accessibility Warning**
   - Login page API key input lacks an `id` or `name` attribute.
   - Triggers a DevTools accessibility issue warning.

10. **IP Data Mocked**
    - `ip_accumulated` returns `0.0` with `ip_pace: "BEHIND"`.
    - UI does not display this value anywhere, so no user-facing impact.

---

## Recommendations

1. **Fix CORS on `/api/fantasy/matchup/simulate`** — Add the frontend origin (`observant-benevolence-production.up.railway.app`) to the backend CORS allowlist, or ensure the existing CORS middleware covers POST requests to this route.

2. **Fix Streaming Page Rendering** — The waiver API returns data correctly; debug the React component responsible for rendering the streaming station. Check for unhandled promise rejections or incorrect state management.

3. **Fix Dashboard Rendering** — The `/api/dashboard` endpoint returns rich data. The frontend page component (`dashboard/page.tsx`) appears to not map the response into visible UI components. Verify the data shape matches the component's expectations.

4. **Fix `/api/fantasy/lineup/current` 422** — Inspect the backend route handler. The query parameter `lineup_date` may be parsed from a different source (e.g., request body for POST, or a custom header) rather than the GET query string. Update documentation or fix parsing logic.

5. **Add Budget Panel to War Room or Sidebar** — The budget API is healthy. Integrate a budget strip into the War Room header or sidebar to show acquisitions remaining, IL usage, and IP pace.

6. **Optimize Matchup API Latency** — 3+ seconds is too slow for an interactive matchup view. Consider caching, parallelizing Yahoo API calls, or adding a loading skeleton while data fetches.

7. **Add Favicon** — Place a `favicon.ico` or `favicon.svg` in the Next.js `public/` directory.

---

## Known Issues Watchlist — Verification

| Issue | Status | Notes |
|-------|--------|-------|
| IP Data Mocked (`ip_accumulated=0.0`) | ✅ No UI impact | Budget API not exposed in UI |
| Matchup Budget Confusion | ✅ Not applicable | No budget strip shown on matchup page |
| Savant Pitch Quality Flat | ⚠️ Unverified | Not exposed in any rendered UI |
| Cache Stale Window | ⚠️ Unverified | Leaderboard data not visible in current UI |
