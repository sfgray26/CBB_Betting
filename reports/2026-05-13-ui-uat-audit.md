# UI UAT Audit: 2026-05-13 23:06 ET

## Environment
- **URL:** https://observant-benevolence-production.up.railway.app/
- **API Base:** https://fantasy-app-production-5079.up.railway.app
- **Viewport:** 1920×1080
- **Browser:** Chrome (via DevTools MCP)
- **Production Deploy:** `d319beb` (local HEAD is `20349c5` — deploy needed)

---

## Screenshots

| Page | Status | Screenshot Path |
|------|--------|-----------------|
| Landing / Dashboard | ✅ | `reports/uat/2026-05-13-dashboard.png` |
| War Room (Matchup) | ✅ | `reports/uat/2026-05-13-war-room.png` |
| Roster | ✅ | `reports/uat/2026-05-13-roster.png` |
| Waiver Wire | ✅ | `reports/uat/2026-05-13-waiver.png` |
| Streaming | ✅ | `reports/uat/2026-05-13-streaming.png` |
| Budget | ✅ | `reports/uat/2026-05-13-budget.png` |
| 404 Error | ✅ | `reports/uat/2026-05-13-404.png` |

---

## Console Errors

| Page | Severity | Message | Count |
|------|----------|---------|-------|
| All pages | — | No console errors detected | 0 |

**Result:** Zero JavaScript errors, warnings, or CSP violations across all audited pages.

---

## API Health

| Endpoint | Status | Latency | Notes |
|----------|--------|---------|-------|
| `GET /api/fantasy/budget` | 200 | 212ms | Returns `budget` + `freshness` |
| `GET /api/fantasy/matchup` | 200 | 98ms | Returns `week`, `my_team`, `opponent` |
| `GET /api/fantasy/roster` | 200 | 848ms | Returns `team_key`, `players`, `count` |
| `GET /api/fantasy/waiver?sort=need_score` | 200 | 1,380ms | Returns `top_available`, `two_start_pitchers` |
| `GET /api/fantasy/coverage` | 200 | 2,125ms | Returns `fangraphs_coverage_pct`, `players` |

**All endpoints return 200 with valid JSON.**

---

## Page-by-Page Findings

### Dashboard (`/dashboard`)
- **Lineup status:** 9/9 slots filled, 21 healthy, 3 injured
- **Lineup gaps:** None detected ✅
- **Injury alerts:** 3 players on IL (Roman Anthony/Wrist, Garrett Crochet/Shoulder, Edwin Díaz/Elbow) with "Move to IL slot immediately" actions
- **Waiver targets:** Now showing **real need scores** (19.75, 18.75, 18.61, 17.49, 14.24) — previously showed 0.00
- **Ownership:** All show "— owned" because production deploy `d319beb` lacks the ownership batch enrichment fix
- **Player trends:** Hot/Cold streaks all show `0.0 avg/7d` — data pipeline issue
- **Budget panel:** 0/8 acquisitions, 3/3 IL full, BEHIND IP pace (0.0/18)
- **Two-start pitchers:** Mitch Keller vs COL on 2026-05-13 shown

### War Room (`/war-room`)
- **Matchup header:** LINDOR TRUFFLES vs HIGH&TIGHTYWHITEY'S, score 4-13 (TRAILING)
- **Projected:** 9-9, 25% win probability
- **Category battlefield:** All 18 categories render with BATTING / PITCHING grouping
- **Status labels:** SAFE, LEAD, BUBBLE, BEHIND, LOST all display correctly
- **Comparison bars:** Present for counting stats; ratio cats show win% only
- **Controls:** ALL / BUBBLES / HITTING / PITCHING filters work; sort dropdown works
- **Re-run simulation button:** Functional

### Roster (`/war-room/roster`)
- **24 players loaded** with full stat grids
- **View toggles:** Season / 7D / 14D / 30D / RoS Proj all present
- **Sort modes:** Default / A–Z / RoS Value present
- **Position filters:** All / SP / RP / OF / 1B / 2B / 3B / SS / C present
- **Matchup strip:** Shows 3W-13L-2T, 0% win prob with category grid
- **Category totals:** OPS (.795) and K/9 (9.8) display correctly
- **Budget panel:** 0/8 weekly moves, BEHIND IP pace, 3/3 IL slots
- **Move controls:** Dropdowns populate with eligible positions + BN/IL/IL60
- **Move buttons:** Disabled until slot selected (expected behavior)
- **Injury badges:** Show for IL players (Wrist, Shoulder, Elbow, Ankle)
- **Player cards:** All show stats, team, positions, ownership (when deployed)

### Waiver Wire (`/war-room/waiver`)
- **25 top available** players loaded
- **Match scores:** Real values (19.75, 18.75, 18.61, etc.) — NOT 0.00
- **Ownership:** All show "— owned" (production lacks deploy)
- **HOT badges:** Every pitcher tagged HOT — the gating fix (rankPercentile ≥ 80) is in local HEAD but not deployed
- **Statcast signals:** LOW_INJURY_RISK shown for most players
- **Market signals:** BUY_LOW, BREAKOUT shown for some hitters
- **Sort controls:** MATCH SCORE / PROJECTED work
- **Position filters:** All 9 filters present

### Streaming (`/war-room/streaming`)
- **25 top available** loaded without crash
- **Need scores:** Display correctly (19.8, 18.8, 18.6, etc.)
- **No null-guard crashes** — the `.toFixed()` fix is working

### Budget (`/war-room/budget`)
- **Constraint Budget** header renders
- **Acquisitions:** 0/8 used, 8 remaining
- **IL Slots:** 3/3 Full
- **IP Pace:** BEHIND, 0.0/18 IP accumulated
- **Updated timestamp:** Shows "11:04 PM"

### 404 Error (`/nonexistent-page-12345`)
- **Styled 404 page** renders (Next.js default)
- **Not a raw FastAPI JSON dump** ✅

---

## Findings

### P0 (Blocking)
*None identified.*

### P1 (Degraded)
1. **Ownership 0% across all players** — `yahoo_client_resilient._enrich_ownership_batch()` fix is committed in local HEAD (`20349c5`) but production is on `d319beb`. Next deploy will resolve.
2. **HOT badge inflation on Waiver Wire** — Every pitcher shows HOT. The `rankPercentile ≥ 80` gating fix is in local HEAD but not deployed.
3. **Player streaks all 0.0 avg/7d** — Hot/Cold streak section on Dashboard shows `0.0 avg/7d` for all players. The `last_7_avg` field may not be computed correctly in the backend pipeline.
4. **Waiver endpoint latency ~1.4s** — Within acceptable range but on the high side. The projection name-map TTL cache (Phase 3b fix) should improve this after deploy.
5. **Coverage endpoint latency ~2.1s** — Exceeds 2s threshold. This is an admin/audit endpoint, not user-facing.

### P2 (Polish)
1. **Move player buttons disabled by default** — This is expected behavior (user must select a slot from dropdown first), but UAT flagged it as confusing. Consider adding helper text or enabling the button with a default action.
2. **IP accumulated = 0.0** — Budget shows 0.0 IP. Per Known Issues Watchlist, this is a known data issue. The Phase 3c fix (`get_matchup_stats` key fix) is in local HEAD.
3. **RoS projection tab** — Some players show "RoS projection not available" when RoS view is selected. Coverage gap — not all players have FanGraphs RoS projections.

---

## Design System v2 Verification

| Component | Status | Notes |
|-----------|--------|-------|
| Surface colors (bg-base, bg-surface, bg-elevated) | ✅ | All cards use new tokens |
| Text hierarchy (primary, secondary, tertiary) | ✅ | Clear distinction across pages |
| Status gradient (safe/lead/bubble/behind/lost) | ✅ | Consistent on Battlefield, MatchupStrip, alerts |
| Gold accent reservation | ✅ | CTAs, headers, key metrics only |
| Category color identity | ⚠️ | Dots present in Battlefield; not yet on Roster/Waiver cards |

---

## Recommendations

1. **Deploy immediately** — Local HEAD (`20349c5`) contains ownership fix, HOT badge gating, IP key fix, and projection cache. Production (`d319beb`) is 7+ commits behind.
2. **Investigate streak pipeline** — `last_7_avg` returning 0.0 for all players suggests the streak computation in `backend/services/snapshot_engine.py` or `dashboard_service.py` is not populating correctly.
3. **Add move button helper text** — On roster page, add a subtle tooltip or label explaining "Select a slot first" to reduce user confusion.
4. **Monitor waiver latency post-deploy** — After the projection cache deploys, re-measure `/api/fantasy/waiver` latency. Target: <800ms.

---

## Deploy Blocker Summary

| Fix | Commit | Effect if not deployed |
|-----|--------|------------------------|
| Ownership enrichment | `425f9d6` | All players show "— owned" |
| HOT badge gating | Part of design system v2 | Every waiver pitcher tagged HOT |
| IP key fix | `d0976b4` | IP accumulated always reads as 0.0 |
| Need score mapping | `bd180a4` | Need scores computed incorrectly |
| Projection cache | Pending | Waiver endpoint slow (~1.4s) |
| Matchup cache | Part of `20349c5` | Repeated page loads quota-heavy |
