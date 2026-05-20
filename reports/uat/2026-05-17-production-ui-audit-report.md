# Production UI Audit Report — Fantasy Baseball Pages
**Date:** 2026-05-17 12:10 PM ET  
**Environment:** Production (`https://observant-benevolence-production.up.railway.app/`)  
**User:** API_KEY_USER1 (Team 7 — Lindor Truffles)  
**Branch:** `stable/cbb-prod` (local HEAD `20349c5`, production `d319beb`)  
**Auditor:** Kimi CLI (Chrome DevTools MCP)

---

## Pages Audited

| # | Page | URL | Screenshot | Status |
|---|------|-----|------------|--------|
| 1 | Dashboard | `/dashboard` | `2026-05-17-dashboard.png` | ✅ Pass |
| 2 | War Room | `/war-room` | `2026-05-17-war-room.png` | ✅ Pass |
| 3 | My Roster | `/war-room/roster` | `2026-05-17-roster.png` | ⚠️ Notes |
| 4 | Waiver Wire | `/war-room/waiver` | `2026-05-17-waiver.png` | ⚠️ Notes |
| 5 | Streaming Station | `/war-room/streaming` | `2026-05-17-streaming.png` | ✅ Pass |
| 6 | Budget | `/war-room/budget` | `2026-05-17-budget.png` | ✅ Pass |

**Console Errors:** 0 across all pages  
**Crashes:** 0  
**Design System v2 Compliance:** All pages using `ds-*` tokens correctly

---

## Page-by-Page Findings

### 1. Dashboard (`/dashboard`) — ✅ PASS
- **Lineup status:** 9/9 filled, 21 healthy / 3 injured
- **Waiver targets:** Real need scores rendering (Max Meyer 18.75, Nick Martinez 17.49, etc.) — **FIX CONFIRMED**
- **Player trends:** Δ values visible (0.5, 0.1, -1.0, etc.)
- **Budget panel:** $100 FAAB showing
- **Two-start pitchers:** Listed
- **Design System:** `bg-bg-surface`, `border-border-subtle`, `text-accent-gold` all applied correctly
- **No console errors**

### 2. War Room (`/war-room`) — ✅ PASS
- **Matchup header:** Week 8, Lindor Truffles 4-12 trailing, projected 7-10, 12% win probability
- **Category battlefield:** All 18 categories rendering with status labels (SAFE, LEAD, BUBBLE, BEHIND, LOST)
- **Action hints:** Present for each category
- **Filter chips:** All/bubbles/hitting/pitching + sort dropdown functional
- **No console errors**

### 3. My Roster (`/war-room/roster`) — ⚠️ NOTES
- **Header:** "MY ROSTER · Team 7 · 24 players" — correct
- **Matchup bar:** 4W · 12L · 2T, 7% win prob — correct
- **Category grid:** All 18 categories with W/L/T status and opponent values
- **Season stats:** 4W · 10L, 16 active, rate stats = median
- **Constraints:** 0/8 weekly moves, 0.0/18 IP (BEHIND), 3/3 IL slots
- **Player table:** 24 players rendered with correct positions, stats, and move dropdowns
- **⚠️ Issue:** All "Move player" buttons are **disabled**. User must select a slot from the dropdown first, then the button enables. This is **functional but potentially confusing UX** — users may think the feature is broken. Consider adding helper text or enabling by default with validation on click.
- **⚠️ Issue:** All ownership values show "— owned" (0% or null). **Backend fix committed but not deployed.** Deploy `20349c5` to resolve.
- **No console errors**

### 4. Waiver Wire (`/war-room/waiver`) — ⚠️ NOTES
- **Header:** "WAIVER WIRE · THIS WEEK · VS HIGH&TIGHTYWHITEY'S"
- **Category cards:** All 18 categories with W/L/T status
- **Sort buttons:** MATCH SCORE / OVERALL VALUE functional
- **Position filters:** All, SP, RP, OF, 1B, 2B, 3B, SS, C — all present
- **Player list:** 50 top available players rendered
- **⚠️ Issue:** Ownership shows "— owned" for nearly all players. Only Carson Benge (3%), Brandon Nimmo (26%), Trevor Larnach (10%), Jeff McNeil (4%) show values. **Backend `_enrich_ownership_batch` fix committed but not deployed.**
- **⚠️ Issue:** HOT badge appears on ~15/50 players (Justin Crawford, Landen Roupp, Max Meyer, Nick Martinez, Alex Vesia, Grant Taylor, Michael McGreevy, Daulton Varsho, Zebby Matthews, Troy Johnston, Ben Brown, Erik Sabrowski, Matt Brash, Dylan Lee, etc.). With the rankPercentile gating fix (top 20% only), this should be ~10 players. **Fix committed but not deployed.**
- **Match scores:** Rendering correctly (8.82, 8.01, 7.59, etc.)
- **Season values:** z-scores rendering with +/- signs (+15.6z, +19.8z, etc.)
- **No console errors**

### 5. Streaming Station (`/war-room/streaming`) — ✅ PASS
- **Category deficits:** All 18 categories with +/- values (K -16.0, TB +16.0, R +8.0, H +8.0, etc.)
- **⚠️ Previously crashed** with `TypeError: Cannot read properties of undefined (reading 'toFixed')` — **NULL-GUARD FIX CONFIRMED WORKING**
- **Top available:** 50 players with NEED scores (Justin Crawford 8.8, Landen Roupp 8.0, Max Meyer 7.6, etc.)
- **No console errors**

### 6. Budget (`/war-room/budget`) — ✅ PASS
- **Header:** "CONSTRAINT BUDGET" with timestamp
- **Acquisitions:** 0 / 8
- **IL Slots:** 3 / 3 (Full)
- **Innings Pitched:** BEHIND — 0.0 IP accumulated, min 18 IP
- **Remaining:** 8 acquisitions this season
- **Minimal but functional page**
- **No console errors**

---

## Verified Fixes (Working in Production)

| Fix | Status | Evidence |
|-----|--------|----------|
| Dashboard waiver target need scores | ✅ Working | Max Meyer 18.75, Nick Martinez 17.49 visible |
| Dashboard player trend deltas | ✅ Working | Δ values shown for all players |
| Streaming Station null-guard crash | ✅ Working | All 18 deficit chips render without error |
| Design System v2 token migration | ✅ Working | All pages use `ds-*` / `bg-*` / `text-*` tokens |

## Fixes Committed But NOT Deployed

| Fix | Commit | Impact |
|-----|--------|--------|
| `_enrich_ownership_batch()` backend | `20349c5` | All ownership shows "— owned" (0%) |
| HOT badge rankPercentile gating (≥80) | `20349c5` | ~15/50 waiver players incorrectly tagged HOT |
| Projection TTL cache | `20349c5` | Waiver latency ~1.4s (acceptable but improvable) |

## Outstanding Issues (Pre-Existing)

| Issue | Severity | Notes |
|-------|----------|-------|
| Team totals show "–" for OPS and K/9 | Medium | Backend aggregation missing those categories |
| Player streaks all 0.0 avg/7d | Medium | Backend pipeline issue — not a frontend bug |
| Roster "Move player" buttons disabled by default | Low | Functional but confusing UX; select slot first |
| IP Pace "BEHIND 0.0 / 18 IP" | Low | Accurate for early week; will populate as games play |

---

## Recommendations

1. **DEPLOY `20349c5` IMMEDIATELY** — This resolves the two most visible user-facing bugs: ownership 0% and HOT badge inflation.
2. **Roster UX:** Consider adding helper text near the move dropdowns (e.g., "Select a slot to enable move") or auto-selecting the first valid position.
3. **Post-deploy re-audit:** Re-check Waiver Wire ownership values and HOT badge count after deployment.

---

## Audit Artifacts

- Screenshots: `reports/uat/2026-05-17-*.png` (6 files)
- Console log: 0 errors, 0 warnings
- Network: All API calls returning 200
