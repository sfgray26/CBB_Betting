# CBB Edge — Task Tracker

> Updated: 2026-03-18
> Active workstream: **Frontend Migration (Phase 1 → Phase 2)**

---

## Frontend Migration — Phase 1 ✅ COMPLETE

### Claude Code (Architect & Engineer)
- [x] Phase 0: Next.js 15 scaffold, design system, TanStack Query, API client
- [x] Phase 0: Auth middleware + login page
- [x] Phase 0: Dashboard shell (sidebar, header, layout)
- [x] Phase 0: Fix Railway CORS (`allow_origins=["*"]`, `allow_credentials=False`)
- [x] Phase 1: `/performance` page — full rewrite with correct API field paths
- [x] Phase 1: `/clv` page — fixed against api_ground_truth.md spec
- [x] Phase 1: `/bet-history` page — fixed `timestamp` field, removed `clv_grade`
- [x] Phase 1: `/calibration` page — `calibration_buckets`, `bin` field, null-guard brier_score
- [x] Phase 1: `/alerts` page — uppercase severity, added live_alerts section
- [x] Update `lib/types.ts` — all 5 interfaces rewritten from api_ground_truth.md
- [x] Fix sidebar — `drawdown_pct` field (was `current_drawdown_pct`)

### Kimi CLI (API Ground Truth)
- [x] K-FRONTEND-1: Produce `reports/api_ground_truth.md` from backend Python source
  - Output: `reports/api_ground_truth.md` — 9 endpoints documented with full JSON shapes

### OpenClaw (Validation)
- [ ] OC-1: Validate `/performance` page against checklist
- [ ] OC-2: Validate `/clv` page against checklist
- [ ] OC-3: Validate `/bet-history` page against checklist
- [ ] OC-4: Validate `/calibration` page against checklist
- [ ] OC-5: Validate `/alerts` page against checklist

**OpenClaw instructions:** Run the 7-point checklist from `FRONTEND_MIGRATION.md`.
Read `reports/api_ground_truth.md` + each page file. Report PASS or issues with line numbers.

---

## Frontend Migration — Phase 2 ⏳ PLANNED

### Scope: Live Slate & Odds Monitor
- [ ] `/predictions` page — today's picks with verdict badges, edge display
- [ ] Predictions detail drawer/modal — full_analysis breakdown
- [ ] Live odds ticker — line movement from odds monitor endpoint
- [ ] Admin panel — run-analysis button, portfolio halt/resume

### Prerequisite: OpenClaw validation of Phase 1 pages (all must PASS)

---

## Betting Model — GUARDIAN FREEZE 🔒

**DO NOT TOUCH until Apr 7, 2026:**
- `backend/betting_model.py`
- `backend/services/analysis.py`
- Any CBB model services

Post-Apr 7 planned:
- [ ] V9.2 recalibration (spec: `reports/K12_RECALIBRATION_SPEC_V92.md`)
- [ ] EvanMiya replacement (G-R7: 3rd rating source via Gemini)
- [ ] CLV attribution deep dive (K-11 output pending)

---

## Known Issues / Blockers

| Issue | Owner | Status |
|-------|-------|--------|
| V9.1 calibration mismatch (SNR stacking) | Claude Code + Kimi | Deferred to Apr 7 |
| EvanMiya down → 2-source mode | Gemini (G-R7) | In progress |
| OpenClaw Phase 1 validation | OpenClaw | Ready to run |

---

## Done Archive

- BDL odds 400 fix (`dates[]` array notation) — Mar 18
- Bracket auto-load on first visit — Mar 18
- market_ml wired through bracket chain — Mar 18
- Fuzzy match rewrite (bracket odds) — Mar 18
- Frontend scaffold (Phase 0) — Mar 18
- Railway CORS fix — Mar 18
- Performance page type fixes — Mar 18
- Phase 1 all 4 pages fixed — Mar 18
