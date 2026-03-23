# CBB Edge — Task Tracker
*Updated: 2026-03-23 | Architect: Claude Sonnet 4.6 | Mission: EMAC-077 "DATA SUPERIORITY"*

> **Canonical source:** `HANDOFF.md` — full specs, ADRs, exit criteria for each EPIC.
> This file is the status board. HANDOFF.md has the implementation detail.

---

## System Status

| Subsystem | Status | Notes |
|-----------|--------|-------|
| V9.1 CBB Model | FROZEN until Apr 7 | Guardian active |
| Frontend (Next.js) Phases 0-5 | DONE | All pages live on main |
| Fantasy Draft + Yahoo OAuth | DONE (Mar 20) | Live draft, value-board, sync endpoint |
| Fantasy Lineup / Waiver endpoints | DONE (Mar 20) | `/fantasy/lineup`, `/fantasy/waiver` live |
| Admin Risk Dashboard | DONE (Mar 20) | `/admin` — 4-panel ops view |
| MLB+PGA Expansion Blueprint | REVIEWED (Mar 23) | Critical bugs identified — see `reports/EXPANSION_ARCHITECTURE_MLB_PGA_BLUEPRINT.md`. No implementation tickets open yet. |
| EPIC-1: Time-Series Schema | **NOT STARTED** | First priority |
| EPIC-2: Ingestion Orchestrator | NOT STARTED | Blocked on EPIC-1 exit criteria |
| EPIC-3: Edge Generation Engine | NOT STARTED | Blocked on EPIC-2 |
| EPIC-4: Waiver Edge Detector | NOT STARTED | Blocked on EPIC-3 |
| EPIC-5: Sport Polling Switch | NOT STARTED | CBB wind-down, MLB ramp-up |
| EPIC-6: Discord Router | NOT STARTED | Blocked on EPIC-4+5 |

---

## GUARDIAN FREEZE (until Apr 7)

**DO NOT TOUCH:**
- `backend/betting_model.py`
- `backend/services/analysis.py`
- Any CBB model services

---

## Active Priority Queue

### 1. EPIC-1 — Time-Series Schema (DO FIRST)
**Spec:** `HANDOFF.md §2` | **No prerequisites**

| Task | File | Done? |
|------|------|-------|
| Write `upgrade()` + `downgrade()` for `player_daily_metrics` | `scripts/migrate_v8_post_draft.py` | [ ] |
| Write `upgrade()` + `downgrade()` for `projection_snapshots` | same | [ ] |
| Add `pricing_engine` column to `predictions` (K-14) | same | [ ] |
| Add SQLAlchemy ORM models (`PlayerDailyMetric`, `ProjectionSnapshot`) | `backend/models.py` | [ ] |
| Write 7 required tests | `tests/test_schema_v8.py` | [ ] |
| Dry-run test locally | — | [ ] |
| Run migration on Railway | — | [ ] |
| Verify schema on Railway DB | — | [ ] |

**Exit gate:** All 8 criteria in `HANDOFF.md §2.5` must be TRUE before EPIC-2 starts.

---

### 2. EPIC-2 — Ingestion Orchestrator (after EPIC-1)
**Spec:** `HANDOFF.md §3` | **Prerequisite: EPIC-1 exit criteria**

Key deliverables:
- `backend/services/daily_ingestion.py` — `DailyIngestionOrchestrator` with advisory lock wrapper (ADR-001)
- `backend/services/clv.py` — add `compute_daily_clv_attribution()` (additive only)
- `/admin/ingestion/status` endpoint
- `tests/test_ingestion_orchestrator.py`
- Mount under `ENABLE_INGESTION_ORCHESTRATOR=true` env var

---

### 3. EPIC-3+: Edge Engine, Waiver Detector, Discord Router
**Spec:** `HANDOFF.md §4–6` | Sequential, each blocked on prior EPIC

---

## Post-Apr 7 Backlog (GUARDIAN-LOCKED)

| Item | Spec | Priority |
|------|------|----------|
| V9.2 params (sd_mult 1.0→0.80, ha 2.419→2.85, SNR_KELLY_FLOOR 0.50→0.75) | `reports/K12_RECALIBRATION_SPEC_V92.md` | HIGH |
| Wire Haslametrics as 3rd rating source | `docs/THIRD_RATING_SOURCE.md` | HIGH |
| `pricing_engine` field on Prediction (K-14) — done in EPIC-1 | — | COMPLETE |
| Oracle validation (spread vs consensus ±5pt flag) | `reports/K15_ORACLE_VALIDATION_SPEC.md` (TBD) | MEDIUM |
| Fantasy Baseball Yahoo OAuth wiring | `docs/MLB_FANTASY_ROADMAP.md` | LOW |

---

## MLB+PGA Expansion (Future — Not Yet Scheduled)

Blueprint reviewed Mar 23. **6 critical bugs identified** — must be fixed before any implementation:
1. `sport_type` enum references — replace with VARCHAR migrations
2. `calculate_stuff_plus` variable shadowing (`break_z`)
3. `mlb_player_stats` UNION ALL — mismatched columns
4. `pga_shots` RANGE partition on VARCHAR — use LIST or drop partition
5. `can_request()` priority bypass — clv_critical skips all budget guards
6. CBB budget (800/mo) incompatible with live system (~11,610/mo) without EPIC-5 first

No implementation tickets open until EMAC-077 EPIC-1 is complete and tournament is past Apr 7.

---

## Pending Manual Actions

| Item | Owner | Action |
|------|-------|--------|
| Push `v0.8.0-cbb-stable` tag | User | `git push origin v0.8.0-cbb-stable` |
| Confirm `NEXT_PUBLIC_API_URL` in Railway frontend | User | Railway dashboard |
| Set `ENABLE_INGESTION_ORCHESTRATOR=false` in Railway | User | Before EPIC-2 is deployed |

---

## Done Archive

| Description | Date |
|-------------|------|
| MLB+PGA Expansion Blueprint reviewed, critical bugs documented | Mar 23 |
| Fantasy Season Ops — full stack (/fantasy/lineup, /fantasy/waiver + backend endpoints) | Mar 20 |
| Admin Risk Dashboard (/admin — 4-panel) | Mar 20 |
| Fantasy Draft Assistant (Live Draft tab, snake order, roster panel) | Mar 20 |
| Fantasy DB Migration v7 on Railway | Mar 20 |
| Frontend Phase 5 (error.tsx + loading.tsx) | Mar 20 |
| Frontend Phase 4 Mobile and PWA (viewport, manifest, drawer, responsive grids) | Mar 20 |
| Frontend Phase 3 Tournament (/bracket 10k MC sims) | Mar 19 |
| Frontend Phase 2 Trading (/today, /live-slate, /odds-monitor) | Mar 19 |
| Frontend Phase 1 Core Analytics (5 pages, OpenClaw validated) | Mar 18 |
| Frontend Phase 0 scaffold + auth + layout | Mar 18 |
| Railway CORS fix | Mar 18 |
| Monte Carlo bracket simulator | Mar 16 |
| Discord morning brief + EOD results jobs fixed | Mar 16 |
| Team mapping hardening (29 St variants, 78 tests) | Mar 16 |
| V9.1 confidence engine (SNR + integrity scalars) | Mar 12 |
