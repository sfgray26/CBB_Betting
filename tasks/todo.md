# CBB Edge — Task Tracker
*Updated: 2026-03-20 | Architect: Claude Sonnet 4.6*

---

## System Status

| Phase | Status | Notes |
|-------|--------|-------|
| Frontend Phases 0-5 | DONE | All pages validated and live on main |
| Fantasy Phase 1 — Draft Assistant | DONE (Mar 20) | Live Draft tab, snake order, roster panel |
| Fantasy DB Migration v7 | DONE (Mar 20) | Railway — Gemini confirmed |
| Admin Risk Dashboard (EMAC-074) | DONE (Mar 20) | /admin — 4-panel 2x2 ops view |
| EMAC-075 Frontend | DONE (Mar 20) | /fantasy/lineup + /fantasy/waiver pages live |
| EMAC-075 Backend | DONE (Mar 20) | Built by Claude. GET /lineup + GET /waiver live. |
| V9.2 Recalibration | LOCKED — Apr 7 | Guardian freeze active |

---

## GUARDIAN FREEZE (until Apr 7)

**DO NOT TOUCH:**
- `backend/betting_model.py`
- `backend/services/analysis.py`
- Any CBB model services

**Post-Apr 7 backlog (see HANDOFF.md Section 5 + 6):**
- [ ] V9.2: `sd_mult` 1.0 to 0.80, `ha` 2.419 to 2.85, `SNR_KELLY_FLOOR` 0.50 to 0.75
- [ ] Wire Haslametrics as 3rd rating source (scraper built at `backend/services/haslametrics.py`, 12 tests pass)
- [ ] Add `pricing_engine` field to Prediction table (K-14)
- [ ] Bump `model_version` to 'v9.2', confirm BET rate 3% to 8-12%

---

## EMAC-075 — Fantasy Season Ops (COMPLETE)

**Frontend:** DONE — `/fantasy/lineup` + `/fantasy/waiver` pages built and pushed (Mar 20)
**Backend:** DONE — built by Claude after Gemini introduced errors (see lessons below)

- [x] `GET /api/fantasy/lineup/{date}` — wired to `DailyLineupOptimizer.build_daily_report()`, maps batter/pitcher rankings to response schema, top 9 batters = START, top 2 SPs = START
- [x] `GET /api/fantasy/waiver` — MVP stub returning valid `WaiverWireResponse` (empty lists)
- [x] Schemas: `LineupPlayerOut`, `StartingPitcherOut`, `DailyLineupResponse`, `WaiverPlayerOut`, `CategoryDeficitOut`, `WaiverWireResponse` in `backend/schemas.py`
- [x] Route conflict resolved: old saved-lineup endpoint renamed to `/api/fantasy/saved-lineup/{date}`
- [x] Both endpoints return 200 and pass `py_compile`

**Lessons (added to HANDOFF.md hive wisdom):**
- Gemini created a duplicate `GET /api/fantasy/lineup/{date}` route — check for existing routes before adding any
- Gemini used nonexistent dict key `report.get("games_found")` — always read the function return type first
- Gemini tested against production without committing — meaningless results

---

## Pending Manual Actions

| Item | Owner | Action |
|------|-------|--------|
| Push `v0.8.0-cbb-stable` tag | User | `git push origin v0.8.0-cbb-stable` |
| Confirm `NEXT_PUBLIC_API_URL` in Railway frontend | User | Railway dashboard |

---

## Apr 7+ Backlog

| Item | Spec | Priority |
|------|------|----------|
| V9.2 params (sd_mult, ha, SNR_KELLY_FLOOR) | `reports/K12_RECALIBRATION_SPEC_V92.md` | High |
| Wire Haslametrics 3rd source | `docs/THIRD_RATING_SOURCE.md` | High |
| pricing_engine field (K-14) | `reports/K13_POSSESSION_SIM_AUDIT.md` | Medium |
| Fantasy Baseball Yahoo OAuth wiring | `docs/MLB_FANTASY_ROADMAP.md` | Low |
| Oracle validation (spread vs consensus) | K-15 spec TBD | Low |

---

## Done Archive

| Description | Date |
|-------------|------|
| Fantasy Season Ops — full stack (/fantasy/lineup, /fantasy/waiver + backend endpoints) | Mar 20 |
| Admin Risk Dashboard (/admin — 4-panel) | Mar 20 |
| Fantasy Draft Assistant (Live Draft tab, snake order, roster panel) | Mar 20 |
| Fantasy DB Migration v7 on Railway | Mar 20 |
| Frontend Phase 5 (error.tsx + loading.tsx on /bracket, /today, /fantasy) | Mar 20 |
| Frontend Phase 4 Mobile and PWA (viewport, manifest, drawer, responsive grids) | Mar 20 |
| Frontend Phase 3 Tournament (/bracket 10k MC sims) | Mar 19 |
| Frontend Phase 2 Trading (/today, /live-slate, /odds-monitor) | Mar 19 |
| Frontend Phase 1 Core Analytics (5 pages, OpenClaw validated) | Mar 18 |
| Frontend Phase 0 scaffold + auth + layout | Mar 18 |
| Railway CORS fix | Mar 18 |
| Monte Carlo bracket simulator (521 lines, 13_Tournament_Bracket.py) | Mar 16 |
| Discord morning brief + EOD results jobs fixed | Mar 16 |
| Team mapping hardening (29 St variants, 78 tests) | Mar 16 |
| Duplicate bet cleanup endpoint | Mar 16 |
| V9.1 confidence engine (SNR + integrity scalars) | Mar 12 |
| New services: fatigue, sharp_money, conference_hca, recency_weight, openclaw_lite | Mar 11-12 |
| BartTorvik 2-source mode confirmed (365 teams, no auth needed) | Mar 10 |
| Prediction dedup fix (run_tier NULL filter) | Mar 12 |
