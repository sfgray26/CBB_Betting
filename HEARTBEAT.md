# HEARTBEAT.md — CBB Edge Operational Loops

> Maintained by: Claude Code (Architect)
> OpenClaw configuration & optimization: Kimi CLI
> For risk policy see `IDENTITY.md`. For agent roles see `AGENTS.md`.

---

## HEARTBEAT: Integrity Sweep

**Trigger:** After Pass 1 slate pre-scoring in `run_nightly_analysis()`. Runs on all games where `_prescore_edge >= MIN_BET_EDGE`.

**Owner:** OpenClaw (execution) | Kimi (configuration & optimization)

**Code:** `backend/services/analysis.py` — `_integrity_sweep()` + `_ddgs_and_check()`

**Action:**
1. Collect BET-tier game keys from Pass 1 output.
2. For each game: DDGS search `"{away} {home} injury suspension lineup"`.
3. `asyncio.gather` with `Semaphore(8)` — max 8 concurrent workers.
4. `perform_sanity_check()` via `scout.py` (qwen2.5:3b local).
5. Normalize to: `CONFIRMED / CAUTION / VOLATILE / ABORT / RED FLAG`.
6. Return `Dict[game_key, verdict]` to Pass 2 loop.

**Thresholds:**
- `>20%` of BET games VOLATILE → log `SYSTEM_RISK_ELEVATED`
- Any ABORT/RED FLAG → surface in morning briefing as priority
- DDGS rate-limited → reduce concurrency to 2, retry with 2s delay

**Target runtime:** <30s for slates ≤8 BET-tier games at 8× concurrency.

---

## HEARTBEAT: Nightly Health Check

**Trigger:** 4:30 AM ET daily (30 min after nightly snapshot job).

**Owner:** OpenClaw (execution) | Kimi (threshold tuning)

**Action:**
1. `GET /api/performance/model-accuracy?days=30` — flag if `margin_mae > 3 pts`.
2. `GET /admin/portfolio/status` — drawdown zones:
   - 0–10%: GREEN
   - 10–15%: YELLOW — add to HANDOFF.md
   - >15%: RED — circuit breaker active
3. `GET /api/predictions/today` — log BET/CONSIDER/PASS counts.
4. Write daily summary to `memory/YYYY-MM-DD.md`.

**Escalation:**
- MAE >3 pts for 7 consecutive days → queue recalibration in HANDOFF.md
- Drawdown >15% → PRIORITY flag in HANDOFF.md, notify operator
- Any pytest failures → halt; block next nightly run

---

## HEARTBEAT: Weekly Calibration Review

**Trigger:** Monday 6 AM ET (day after Sunday auto-recalibration at 5 AM ET).

**Owner:** OpenClaw (execution) | Kimi (interpretation & recommendations)

**Action:**
1. `GET /admin/scheduler/status` — confirm `weekly_recalibration` ran.
2. Read new `home_advantage` and `sd_multiplier` from `model_parameters` table.
3. Compare against `IDENTITY.md` baselines. Flag if shifted >15%.
4. Update HANDOFF.md calibration history with new values and date.

**Escalation:**
- Skip (<30 bets) → log INFO, no action
- Shift >15% → HANDOFF.md "Architect Review Required"
- Job failed → CRITICAL in HANDOFF.md, notify Gemini CLI

---

## STATUS TRACKER

| Loop | Status | Last Run | Notes |
|------|--------|----------|-------|
| Integrity Sweep | LIVE | 2026-03-07 | Async implemented. 0 BET games triggered since V9 launch — correct. |
| Nightly Health Check | LIVE | 2026-03-08 | K-7 thresholds active. Sentinel@05:00 ET. Morning briefing@07:00 ET. |
| Weekly Calibration Review | DEFINED | — | Recalibration running (Sunday 5 AM). Review loop not yet active. |
| O-6 Integrity Spot-Check | ✅ COMPLETE | 2026-03-07 | All 133 predictions have null integrity_verdict — correct (0 BET-tier games = sweep not triggered). |
| O-8 Pre-Tournament Baseline | READY | — | Script created. Run March 16 ~9 PM ET. See `reports/k6-o8-baseline-spec.md`. |
| OpenClaw Notification System | v2.1 | 2026-03-07 | Discord errors fixed. File logging fallback added. See `.openclaw/TROUBLESHOOTING.md`. |
| O-9 Tiered Escalation | LIVE | 2026-03-07 | coordinator.py live. Logs ESCALATION_FLAGGED on units>=1.5, neutral_site, VOLATILE. |
| V9 Production Verification | OK | 2026-03-07 | 474/474 tests. Railway live. All env var parsing clean. |
| Tournament Seed Data | LIVE | 2026-03-07 | BALLDONTLIE_API_KEY set in Railway. Seed-spread scalars active. |
