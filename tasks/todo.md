# CBB Edge — Task Tracker
*Updated: 2026-03-20 | Architect: Claude Sonnet 4.6*

---

## System Status

| Phase | Status | Notes |
|-------|--------|-------|
| Frontend Phase 0 — Foundation | ✅ DONE | Scaffold, auth, layout, design system |
| Frontend Phase 1 — Core Analytics | ✅ DONE | /performance, /clv, /bet-history, /calibration, /alerts |
| Frontend Phase 2 — Trading | ✅ DONE | /today, /live-slate, /odds-monitor |
| Frontend Phase 3 — Tournament | ✅ DONE | /bracket (10k MC sims) |
| Frontend Phase 4 — Mobile & PWA | ✅ DONE | Viewport, manifest, drawer, responsive grids |
| Frontend Phase 5 — Polish | ✅ DONE (Mar 20) | error.tsx + loading.tsx on /bracket, /today, /fantasy |
| Fantasy Phase 1 — Draft Assistant | ✅ DONE (Mar 20) | Live Draft tab, snake order, roster panel |
| Fantasy DB Migration (v7) | ✅ DONE (Mar 20) | Railway — Gemini confirmed |

---

## GUARDIAN FREEZE 🔒 (until Apr 7)

**DO NOT TOUCH:**
- `backend/betting_model.py`
- `backend/services/analysis.py`
- Any CBB model services

**Post-Apr 7 (see HANDOFF.md Section 5):**
- [ ] V9.2: `sd_mult` 1.0→0.80, `ha` 2.419→2.85, `SNR_KELLY_FLOOR` 0.50→0.75
- [ ] Wire Haslametrics as 3rd rating source (scraper built, 12 tests pass)
- [ ] Add `pricing_engine` field to Prediction (K-14)
- [ ] Bump `model_version` to 'v9.2', confirm BET rate 3%→8-12%

---

## EMAC-074 — Admin Risk Dashboard (ACTIVE — Mar 20)

**Priority:** HIGH — tournament R64 live, need consolidated ops view.
**Swimlane:** Claude (frontend + types)
**Backend endpoints:** All exist — no new backend work needed.

### Tasks
- [x] Add `/admin` route to sidebar (new "Admin" section, ShieldAlert icon)
- [x] Build `frontend/app/(dashboard)/admin/page.tsx`:
  - Portfolio panel: drawdown gauge, bankroll, exposure, positions
  - Ratings panel: KenPom/BartTorvik/EvanMiya status + model_health badge
  - Scheduler panel: all jobs + next_run times
  - Odds Monitor panel: active status, games tracked, quota
- [x] Add types: `SchedulerStatus`, `SchedulerJob`, `RatingsStatus` to `lib/types.ts`
- [x] Add API methods: `schedulerStatus()`, `ratingsStatus()` to `lib/api.ts`
- [x] `admin/error.tsx` + `admin/loading.tsx`
- [x] TypeScript clean — 0 errors
- [x] Update HANDOFF.md

**Backend endpoints to wire:**
- `GET /admin/portfolio/status` — already in lib/api.ts as `portfolioStatusFull()`
- `GET /admin/scheduler/status` — needs new API client method
- `GET /admin/ratings/status` — needs new API client method
- `GET /health` — unauthenticated

---

## EMAC-075 — Fantasy Season Ops (Mar 27 — season opener)

**Priority:** MEDIUM — build before March 27.
**Swimlane:** Claude (frontend) + Gemini (research: SP confirmation sources)

### Tasks
- [ ] Add sidebar items: "Daily Lineup", "Waiver Wire" under Fantasy Baseball
- [ ] Build `frontend/app/(dashboard)/fantasy/lineup/page.tsx`:
  - Today's games with implied run environment (from Odds API)
  - Batters ranked by team implied runs × park factor
  - SPs ranked by opponent implied runs (lower = better)
  - "Bench" / "Start" visual for each roster slot
- [ ] Build `frontend/app/(dashboard)/fantasy/waiver/page.tsx`:
  - Category deficit tracker (H2H matchup this week)
  - Top available players ranked by need score
  - 2-start pitcher tracker
- [ ] New API types + client methods for lineup + waiver endpoints
- [ ] error.tsx + loading.tsx for both pages

**Backend endpoints (already exist):**
- `GET /api/fantasy/lineup/{date}` — saved lineup
- `POST /api/fantasy/lineup` — save lineup

---

## Pending Manual Actions

| Item | Owner | Status |
|------|-------|--------|
| Push `v0.8.0-cbb-stable` tag | User | `git push origin v0.8.0-cbb-stable` |
| Set `RAILWAY_TOKEN` in GitHub Secrets | User | Settings → Secrets → Actions |
| Set `NEXT_PUBLIC_API_URL` in Railway frontend | User | Railway dashboard |

---

## Done Archive

- Fantasy Draft Assistant (Live Draft tab, snake order, roster panel) — Mar 20
- Fantasy DB Migration v7 on Railway — Mar 20
- Frontend Phase 5 (error.tsx + loading.tsx on /bracket, /today) — Mar 20
- Frontend Phase 4 Mobile & PWA — Mar 20
- Frontend Phases 2+3 (trading + bracket pages) — Mar 19
- Frontend Phase 1 (all 5 analytics pages) — Mar 18
- Frontend Phase 0 scaffold — Mar 18
- Railway CORS fix — Mar 18
- Monte Carlo bracket simulator — Mar 16
- Discord morning brief + EOD results — Mar 16
- Team mapping hardening (29 St variants, 78 tests) — Mar 16
- Duplicate bet cleanup endpoint — Mar 16
- V9.1 model (fatigue, sharp money, conf HCA, recency) — Mar 11-12
