# HANDOFF.md — MLB Platform Operational State

> **Date:** April 15, 2026 | **Author:** Claude Code (Master Architect)
> **Status:** NSB PIPELINE LIVE. v27 DEPLOYED. AWAITING NEXT ROLLING_WINDOWS + PLAYER_SCORES RUN.
>
> **Current Focus:** Close weather/park factor integration gap, investigate `decision_results` volume,
> and commit uncommitted v27 changes.
>
> **Full audit:** `reports/2026-04-15-comprehensive-application-audit.md`
> **Historical context:** `HANDOFF_ARCHIVE.md`

---

## NSB PIPELINE (P27)

**Problem:** NSB (Net Stolen Bases = SB - CS) was consumed downstream but never computed in the scoring pipeline.

**Solution deployed:**
- `scripts/migrate_v27_nsb.py` — adds `w_caught_stealing`, `w_net_stolen_bases` (rolling stats) and `z_nsb` (scores)
- `rolling_window_engine.py` — aggregates CS from BDL `mlb_player_stats`; NSB = SB - CS
- `scoring_engine.py` — `z_nsb` drives `composite_z`; `z_sb` excluded from composite to avoid double-counting
- `daily_ingestion.py` — upsert write paths extended for new columns
- `tests/test_nsb_pipeline.py` — 15 regression tests

**Verdict:** Columns added in prod on Apr 14. Data will populate on the next 4 AM ET `rolling_windows` + `player_scores` job.

---

## P27 FOLLOW-UP — TASKS C + D (April 14)

### Task C — NSB rollout diagnostic endpoints
- `GET /admin/diagnose-scoring/nsb-rollout` — fill-rate verdict
- `GET /admin/diagnose-scoring/nsb-leaders` — top/bottom NSB players
- `GET /admin/diagnose-scoring/nsb-player` — per-player window detail

**Tests:** `tests/test_admin_scoring_diagnostics.py` — 18/18 pass.

### Task D — explainability layer narrates NSB over SB
- `explainability_layer.py` prefers `z_nsb` and labels it `"Net basestealing (NSB Z-score)"`
- Falls back to `z_sb` only when `z_nsb is None`

**Tests:** `tests/test_explainability_layer.py` — 40/40 pass.

**Full suite:** 1884 passed, 3 skipped.

---

## APRIL 15 AUDIT FINDINGS (Kimi CLI)

### Live Production State

`GET /admin/pipeline-health` → **`overall_healthy: true`**

| Table | Rows | Latest Date | Status |
|-------|------|-------------|--------|
| `player_rolling_stats` | **30,667** | 2026-04-13 | ✅ Healthy |
| `player_scores` | **30,580** | 2026-04-13 | ✅ Healthy |
| `statcast_performances` | **6,971** | 2026-04-13 | ✅ Healthy |
| `simulation_results` | **10,236** | 2026-04-13 | ✅ Healthy |
| `mlb_player_stats` | **6,801** | 2026-04-13 | ✅ Healthy |
| `probable_pitchers` | **0** | — | ⚠️ Empty (upstream) |

### Critical Wins
- Statcast zero-metric rate: **42.4% → 5.0%** post-aggregation fix
- NSB pipeline live in prod
- Scheduler stable; 10 jobs active
- Tests green on Railway (1859 passed)

### Critical Gaps
1. **Weather/park factors:** Code exists (`weather_fetcher.py`, `park_weather.py`, `ballpark_factors.py`) but **not integrated** into DB, scoring engine, or ingestion pipeline.
2. **Probable pitchers:** Empty (0 rows). Upstream MLB Stats API returned 0.
3. **Decision results:** Suspiciously low volume (**26 rows**). Needs investigation.
4. **Uncommitted changes:** `scoring_engine.py`, `daily_ingestion.py`, `main.py`, `tests/test_nsb_pipeline.py` must be committed.
5. **Data ingestion logs:** Table exists but has 0 rows.

---

## CURRENT SESSION STATE (April 14, 2026)

### Gemini DevOps Report (April 14 — complete)

| Task | Result |
|------|--------|
| Deploy latest code to Railway | ✅ Healthy startup |
| `/admin/backfill-cs-from-statcast` | ✅ 0 updates (source CS gap) |
| Probable pitchers ingestion | ✅ 0 records (upstream lag) |
| Statcast quality audit (pre-fix) | ✅ 42.4% zero-metric — bug confirmed |
| Fuzzy linker | ✅ 0 linked, 362 orphans accepted |

### Live Production DB State (April 14 baseline)

| Table | Rows | Notes |
|-------|------|-------|
| `mlb_player_stats` | ~6,500+ | Growing via BDL |
| `statcast_performances` | 13,653 | Pre-fix raw count |
| `simulation_results` | ~9,400+ | Fresh |
| `player_rolling_stats` | ~28,000+ | OK |
| `player_scores` | ~28,000+ | OK |
| `decision_results` | 26 | Low — investigate |

### Work Completed This Session (April 14)
4 commits fixed Statcast aggregation (`ef80ecc`..`8fcff87`):
1. `is_pitcher` flag for type-scoped upserts
2. `_aggregate_to_daily()` pre-aggregation (core fix)
3. Type-scoped upserts protecting two-way players
4. End-to-end integration test

### Architect Review Queue
- **P1:** `/admin/diagnose-era` needs `::numeric` cast in prod
- **P2:** `player_id_mapping` dedup (60K vs ~2K expected)
- **P3:** `decision_results = 26` — verify expected or silent drop

---

## DATABASE STATE (April 15, 2026) — LIVE

| Table | Expected | Actual | Status | Notes |
|-------|----------|--------|--------|-------|
| `player_id_mapping` | ~2,000 | **60,000** (dupes) | POPULATED | FU-3: dedup deferred |
| `position_eligibility` | ~750 | **~2,376** | OK | 362 permanent orphans |
| `mlb_player_stats` | ~13,500 | **6,801** | OK | Growing daily via BDL |
| `probable_pitchers` | ~30/day | **0** | EMPTY | MLB Stats API lag |
| `statcast_performances` | ~20,000 | **6,971** | ✅ OK | 5.0% zero-metric rate |
| `player_projections` | varies | 0 | EMPTY | No projection pipeline |
| `player_rolling_stats` | ~25,000 | **30,667** | ✅ OK | Populated |
| `player_scores` | ~25,000 | **30,580** | ✅ OK | Populated |
| `simulation_results` | ~8,500+ | **10,236** | ✅ OK | Fresh through 2026-04-13 |
| `decision_results` | varies | **26** | 🔴 LOW | Investigate decision job |
| `data_ingestion_logs` | should have entries | 0 | EMPTY | Not implemented |

---

## NEXT STEPS (Priority Order)

### Immediate

| Priority | Action | Owner | ETA |
|----------|--------|-------|-----|
| P0 | Commit v27 changes (`scoring_engine.py`, `daily_ingestion.py`, `main.py`, `tests/test_nsb_pipeline.py`) | Claude | 15 min |
| P0 | Investigate `decision_results = 26` — trace decision optimization job | Claude | 2 hrs |
| P1 | Integrate park factors into data pipeline (table + scoring engine + lineup optimizer) | Claude | 1 week |
| P1 | Resolve probable pitchers source or implement alternative | Claude | 2-3 days |
| P1 | Build data ingestion logging infrastructure | Claude | 2 days |
| P2 | Dedupe `player_id_mapping` (60K → ~2K) | Claude/Gemini | 1 day |
| P2 | Integrate live weather forecasts (requires `OPENWEATHER_API_KEY`) | Claude | 1 week |

### Parallel Tracks

| Track | Owner | Status |
|-------|-------|--------|
| K-39..K-43 Infra Hardening | Kimi CLI | Research pending |
| K-44 UI/UX Design System | Claude | Ready to start |

---

**Last Updated:** April 15, 2026
