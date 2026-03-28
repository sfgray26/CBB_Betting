# HEARTBEAT.md — Dual-System Operational Loops

> Maintained by: Claude Code (Master Architect)
> Covers: CBB Betting Analyzer (V9.1) + Fantasy Baseball Platform (Season Live)
> For risk policy see `IDENTITY.md`. For agent roles see `AGENTS.md`.
> Last consolidated: March 28, 2026

---

## SYSTEM 1 — CBB BETTING ANALYZER

### HEARTBEAT: Integrity Sweep

**Trigger:** After Pass 1 slate pre-scoring in `run_nightly_analysis()`. Runs on all games where `_prescore_edge >= MIN_BET_EDGE`.

**Owner:** OpenClaw (execution) | Claude (configuration)

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

### HEARTBEAT: Sharp Money Monitor

**Trigger:** After each odds poll (every 5 min during active hours: 12 PM–11 PM ET).

**Owner:** Claude (integration) | OpenClaw (execution)

**Code:** `backend/services/sharp_money.py` — `SharpMoneyDetector.detect_from_history()`

**Action:**
1. Poll line history from `odds_monitor.py` in-memory buffer or DB.
2. Detect patterns:
   - **Steam:** ≥1.5 pt move in <30 min
   - **Opener Gap:** ≥2.0 pt divergence from open
   - **RLM:** Line moves against public % (if data available)
3. Log high-confidence signals (≥0.7) to Discord.
4. Store all signals in `Prediction.full_analysis["sharp_signal"]`.

**Escalation:**
- Steam + Opener Gap aligned → `HIGH_CONFIDENCE_SHARP` tag
- Sharp opposes model → Auto-reduce edge by 0.8% × confidence
- Sharp aligns with model → Auto-boost edge by 0.5% × confidence

---

### HEARTBEAT: Nightly Health Check

**Trigger:** 4:30 AM ET daily (30 min after nightly snapshot job).

**Owner:** OpenClaw (execution)

**Action:**
1. `GET /api/performance/model-accuracy?days=30` — flag if `margin_mae > 3 pts`.
2. `GET /admin/portfolio/status` — drawdown zones:
   - 0–10%: GREEN
   - 10–15%: YELLOW — add to HANDOFF.md
   - >15%: RED — circuit breaker active
3. `GET /api/predictions/today` — log BET/CONSIDER/PASS counts.
4. `GET /admin/ratings/status` — verify 2+ sources active.
5. Write daily summary to `memory/YYYY-MM-DD.md`.

**Escalation:**
- MAE >3 pts for 7 consecutive days → queue recalibration in HANDOFF.md
- Drawdown >15% → PRIORITY flag in HANDOFF.md, notify operator
- Rating sources <2 → CRITICAL alert, model degraded

---

### HEARTBEAT: Weekly Calibration Review

**Trigger:** Monday 6 AM ET (day after Sunday auto-recalibration at 5 AM ET).

**Owner:** OpenClaw (execution) | Claude (interpretation)

**Action:**
1. `GET /admin/scheduler/status` — confirm `weekly_recalibration` ran.
2. Read new `home_advantage` and `sd_multiplier` from `model_parameters` table.
3. Compare against `IDENTITY.md` baselines. Flag if shifted >15%.
4. Update HANDOFF.md calibration history with new values and date.

**Escalation:**
- Skip (<30 bets) → log INFO, no action
- Shift >15% → HANDOFF.md "Architect Review Required"
- Job failed → CRITICAL in HANDOFF.md, notify Gemini CLI

> **NOTE:** V9.1 manual recalibration (EMAC-068) is BLOCKED until post-Apr 7. Weekly auto-recalibration continues, but results are for monitoring only. Do not apply parameter changes until Claude unblocks EMAC-068.

---

## SYSTEM 2 — FANTASY BASEBALL PLATFORM

### HEARTBEAT: Daily Lineup Decision Window

**Trigger:** 9:00 AM ET daily (MLB lineups confirmed by ~10:30 AM ET on game days).

**Owner:** Claude (architecture) | Kimi (research delegation when needed)

**Code:** `backend/fantasy_baseball/daily_lineup_optimizer.py` — `solve_lineup()` + `flag_pitcher_starts()`

**Action:**
1. Fetch MLB slate via Odds API — date in **America/New_York** timezone (not UTC).
2. `flag_pitcher_starts(roster, game_date)` — confirm SP starts via MLB Stats API.
3. `solve_lineup(roster, projections, game_date)` — ILP (OR-Tools) or greedy fallback.
4. Surface gaps + warnings to dashboard via `DashboardService`.

**Key constraint:** All `game_date` defaults use `datetime.now(ZoneInfo("America/New_York"))` — never `datetime.utcnow()`. West Coast games (9pm+ EDT) were previously dropped by UTC date mismatch — fixed March 28, 2026.

**Escalation:**
- OR-Tools not installed → greedy fallback runs silently; log WARNING
- MLB Stats API down → pitcher starts fallback to Odds API slate only
- >3 unfilled lineup slots → surface as CRITICAL gap in dashboard

---

### HEARTBEAT: Waiver Wire Window

**Trigger:** Saturdays 10 AM ET (Yahoo waiver processing day).

**Owner:** Claude (service layer) | WaiverEdgeDetector (execution)

**Code:** `backend/services/waiver_edge_detector.py` — `get_top_moves()`

**Action:**
1. `get_top_moves(my_roster, opponent_roster, n_candidates=10)` — score FA vs category deficits.
2. Surface top 5 as `WaiverTarget` objects via `DashboardService._get_waiver_targets()`.
3. High-impact moves (`win_prob_gain >= 0.05`) → Discord priority-2 alert via `OpenClawAutonomousLoop`.

**Escalation:**
- Yahoo API down → waiver targets panel shows empty; log warning
- WaiverEdgeDetector throws → `DashboardService._get_waiver_targets()` returns `[]`

---

### HEARTBEAT: Morning Brief (Fantasy)

**Trigger:** 7:00 AM ET daily via `scripts/openclaw_scheduler.py --morning-brief`.

**Owner:** OpenClaw (execution) via APScheduler

**Code:** `backend/services/openclaw_briefs_improved.py` — `generate_and_send_morning_brief_improved()`
Alias: `generate_and_send_morning_brief` (canonical name used by all callers)

**Action:**
1. Compile roster health + probable pitchers + waiver targets.
2. Generate narrative via `scout.py` (qwen2.5:3b).
3. Send to Discord.

**Escalation:**
- Brief generation fails → `OpenClawAutonomousLoop.run_morning_workflow()` logs warning and continues (does not abort move evaluation)
- Discord send fails → file log fallback in `.openclaw/TROUBLESHOOTING.md`

---

## STATUS TRACKER

| Loop | System | Status | Last Updated | Notes |
|------|--------|--------|--------------|-------|
| Integrity Sweep | CBB | ✅ LIVE | Mar 11, 2026 | Async implemented. 0 BET games since V9 launch — correct. |
| Sharp Money Monitor | CBB | ✅ LIVE | Mar 11, 2026 | Steam, opener gap, RLM. Edge auto-adjustment active. |
| Nightly Health Check | CBB | ✅ LIVE | Mar 11, 2026 | Thresholds active. 2-source pipeline confirmed. |
| Weekly Calibration Review | CBB | ✅ DEFINED | Mar 11, 2026 | Auto-recalibration runs; manual EMAC-068 blocked until Apr 7. |
| Daily Lineup Window | Fantasy | ✅ LIVE | Mar 28, 2026 | UTC→ET fix deployed. OR-Tools pending deploy. |
| Waiver Wire Window | Fantasy | ✅ LIVE | Mar 28, 2026 | `get_top_moves()` wired to dashboard. |
| Morning Brief | Fantasy | ✅ LIVE | Mar 28, 2026 | `openclaw_briefs_improved.py` is canonical. |
| Dashboard SSE Stream | Fantasy | ❌ PENDING | — | Approved in roadmap. Next implementation task. |
| MCMC Matchup Simulator | Fantasy | ❌ SCAFFOLDED | — | `mcmc_simulator.py` exists, not calibrated. B5 roadmap. |
