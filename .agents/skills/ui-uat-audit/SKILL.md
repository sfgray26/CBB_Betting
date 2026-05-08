---
name: ui-uat-audit
description: Perform a repeatable UI/UX acceptance test audit on the live production deployment using Chrome DevTools MCP. Screenshots DOM, checks console errors, verifies API responses, and compares against expected states.
---

# UI UAT Audit Agent

## When to Activate

- "Run the UAT audit"
- "Check the UI for issues"
- "Audit the frontend"
- "Is the production UI working?"
- "Screenshot and verify the app"
- `/skill:ui-uat-audit` — explicit invocation

## Production Target

| Property | Value |
|----------|-------|
| **Base URL** | `https://observant-benevolence-production.up.railway.app/` |
| **Auth Header** | `API_KEY_USER1` (required for protected endpoints) |
| **Tech Stack** | FastAPI backend serving API + frontend |
| **DevTools MCP** | `chrome-devtools` — use for DOM inspection, screenshots, console monitoring |

## Pre-Audit Setup

1. **Ensure Chrome DevTools MCP is connected**: Run `kimi mcp test chrome-devtools` if unsure.
2. **Open the production URL** in a controlled Chrome tab via DevTools MCP.
3. **Set viewport**: Use a standard desktop viewport (1920×1080) unless mobile audit is requested.
4. **Clear console** before each page load to capture fresh errors.

---

## Audit Workflow

### Phase 1: Landing / Health Check

**Actions:**
1. Navigate to `https://observant-benevolence-production.up.railway.app/`
2. **Screenshot** the landing state
3. Check **console for errors** (JS exceptions, 404s, CSP violations)
4. Check **Network tab** for failing API calls
5. Verify page **title** and **meta description** are present

**Pass Criteria:**
- [ ] Page loads without 5xx errors
- [ ] No critical console errors (red)
- [ ] Core layout renders (header/nav, main content area, footer if applicable)
- [ ] Favicon loads

### Phase 2: Fantasy Baseball — Budget Panel

**Endpoint under test:** `GET /api/fantasy/budget` (live Yahoo data)

**Actions:**
1. Navigate to the budget view (or trigger API call with `API_KEY_USER1` header)
2. **Screenshot** the budget panel
3. Verify the API response shape matches:
   ```json
   {
     "budget": {
       "acquisitions_used": int,
       "acquisitions_remaining": int,
       "acquisition_limit": int,
       "acquisition_warning": bool,
       "il_used": int,
       "il_total": int,
       "ip_accumulated": float,
       "ip_minimum": float,
       "ip_pace": str,
       "as_of": str
     }
   }
   ```
4. Check that `ip_accumulated` is displayed (even if mocked at `0.0` — verify the UI handles it gracefully)

**Pass Criteria:**
- [ ] Budget numbers render (not blank or "undefined")
- [ ] `acquisitions_used` ≤ `acquisition_limit`
- [ ] `il_used` ≤ `il_total`
- [ ] Datestamp (`as_of`) is visible and recent
- [ ] No console errors on budget panel load

### Phase 3: Fantasy Baseball — Matchup View

**Endpoint under test:** `GET /api/fantasy/matchup`

**Actions:**
1. Navigate to matchup view
2. **Screenshot**
3. Verify matchup data renders (teams, categories, scores)
4. Check for **hardcoded values** — the `/api/fantasy/matchup` endpoint currently returns hardcoded `acquisitions_used=5`, `il_used=1`, `ip_accumulated=45.0`. The UI should NOT display these as if they were live budget data.

**Pass Criteria:**
- [ ] Matchup categories render
- [ ] If budget strip is shown on matchup page, it must come from `/api/fantasy/budget` (live), not hardcoded values
- [ ] No duplicate/conflicting budget numbers between pages

### Phase 4: Fantasy Baseball — Lineup Views

**Endpoints:** `/api/fantasy/lineup/*`

**Actions:**
1. Navigate to lineup optimizer / current lineup view
2. **Screenshot**
3. Verify player cards render with:
   - Player name
   - Position eligibility
   - Projected stats (if shown)
   - Injury status indicators
4. Check that **bench** vs **starting** lineup is visually distinct

**Pass Criteria:**
- [ ] All roster slots populate
- [ ] Empty slots show "Empty" or placeholder (not blank)
- [ ] Injury icons/flags render for players on IL
- [ ] No duplicate players across slots

### Phase 5: API Contract Verification

**Actions:**
1. Use DevTools Network tab or a direct `fetch` call to hit these endpoints with `API_KEY_USER1`:
   - `GET /api/fantasy/budget`
   - `GET /api/fantasy/matchup`
   - `GET /api/fantasy/lineup/current`
2. Verify **HTTP 200** and valid JSON
3. Verify **response time** < 2000ms (budget should be < 500ms since it's live Yahoo data)

**Pass Criteria:**
- [ ] All endpoints return 200
- [ ] All responses are valid JSON
- [ ] Response times are acceptable (< 2s)
- [ ] No CORS errors in console

### Phase 6: Error State Resilience

**Actions:**
1. If possible, trigger an error state (e.g., navigate to a non-existent route)
2. Verify the UI shows a graceful error page (not a raw FastAPI 404 JSON dump)
3. Check that error boundaries catch JS exceptions

**Pass Criteria:**
- [ ] 404 pages are styled (not raw JSON)
- [ ] No uncaught exceptions bubble to console

---

## Reporting

Produce a structured markdown report:

```markdown
# UI UAT Audit: YYYY-MM-DD HH:MM

## Environment
- URL: https://observant-benevolence-production.up.railway.app/
- Viewport: 1920×1080
- Browser: Chrome (via DevTools MCP)

## Screenshots
| Page | Status | Screenshot Path |
|------|--------|-----------------|
| Landing | ✅/❌ | `reports/uat/YYYY-MM-DD-landing.png` |
| Budget | ✅/❌ | `reports/uat/YYYY-MM-DD-budget.png` |
| Matchup | ✅/❌ | `reports/uat/YYYY-MM-DD-matchup.png` |
| Lineup | ✅/❌ | `reports/uat/YYYY-MM-DD-lineup.png` |

## Console Errors
| Page | Severity | Message | Count |
|------|----------|---------|-------|

## API Health
| Endpoint | Status | Latency | Notes |
|----------|--------|---------|-------|

## Findings
### P0 (Blocking)
- ...

### P1 (Degraded)
- ...

### P2 (Polish)
- ...

## Recommendations
1. ...
```

**Save report to:** `reports/YYYY-MM-DD-ui-uat-audit.md`
**Save screenshots to:** `reports/uat/YYYY-MM-DD-<page>.png` (create `reports/uat/` if needed)

---

## Post-Audit Actions

1. **Summarize critical findings** in `HANDOFF.md` under a new `## K-N UI UAT FINDINGS` section
2. **Flag P0 issues** for immediate Claude Code attention
3. **Attach screenshots** to the report for visual regression tracking
4. If this is a **scheduled audit** (e.g., post-deploy), compare against the previous audit screenshot to detect visual regressions

---

## Known Issues Watchlist

These are existing issues to verify on every audit:

1. **IP Data Mocked** — `ip_accumulated` is hardcoded to `0.0` in `/api/fantasy/budget`. UI should handle `0.0` gracefully (show "0.0" or "N/A", not crash).
2. **Matchup Budget Confusion** — `/api/fantasy/matchup` returns hardcoded budget values. If the UI shows budget on the matchup page, confirm it's calling `/api/fantasy/budget` separately.
3. **Savant Pitch Quality Flat** — All 550 pitcher scores = 100.0 due to NULL `ip`. If exposed in UI, verify it doesn't show misleading confidence.
4. **Cache Stale Window** — `statcast_loader.py` has 6-hour TTL. If leaderboard data looks stale, this may be why.

---

## Tips for Effective Auditing

- **Screenshot before interacting** — captures initial load state
- **Watch the Network tab** during navigation — catch slow or failing requests
- **Check both light and dark modes** if the UI supports them
- **Resize viewport** to 375×667 briefly to verify responsive breakpoints
- **Use DevTools `captureScreenshot` with fullPage=true** for full-page captures
- **Log every console error** — even warnings can indicate degradation
