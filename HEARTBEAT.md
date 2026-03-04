# HEARTBEAT.md — CBB Edge Operational Loops

> Maintained by Claude Code (Architect). Executed by OpenClaw.
> For risk policy and scalar values, see `IDENTITY.md`.
> For agent role definitions, see `AGENTS.md`.

---

## HEARTBEAT: Integrity Sweep

**Trigger:** Immediately after Pass 1 slate pre-scoring in `run_nightly_analysis()`.
Pre-scoring produces `_prescore_edges` dict. Sweep runs on all games where `_prescore_edge >= MIN_BET_EDGE`.

**Owner:** Integrity Execution Unit (OpenClaw via `backend/services/scout.py`)

**Action:**
1. Collect all BET-tier game keys from Pass 1 output (edge ≥ threshold).
2. For each game: build a search query (`"{away} {home} injury suspension lineup"`).
3. Execute `DDGS().text(query, max_results=5)` for all games via `asyncio.gather` (max 8 workers).
4. For each result set: call `perform_sanity_check(context, model_line, our_line)`.
5. Normalize verdict to contract string: CONFIRMED / CAUTION / VOLATILE / ABORT / RED FLAG.
6. Return `Dict[game_key, integrity_verdict]` to Pass 2 loop.

**Concurrency:**
```python
async def _integrity_sweep(bet_tier_games: list) -> dict:
    semaphore = asyncio.Semaphore(8)
    async def _one(game):
        async with semaphore:
            return await _ddgs_and_check(game)
    results = await asyncio.gather(*[_one(g) for g in bet_tier_games], return_exceptions=True)
    return {g["key"]: (r if not isinstance(r, Exception) else "Sanity check unavailable")
            for g, r in zip(bet_tier_games, results)}
```

**Fallback (sync context):**
If event loop not running, fall back to sequential calls with 1 s inter-call delay. Log WARNING.

**Escalation:**
- > 20% of BET games return VOLATILE → log `SYSTEM_RISK_ELEVATED`. Surface in Morning Briefing.
- Any ABORT/RED FLAG → surface in Morning Briefing as 🛑 priority item.
- `RateLimitError` from DDGS → reduce concurrency to 2, retry with 2 s delay.

**Expected Runtime:** < 30 s for slates ≤ 8 BET-tier games at 8× concurrency.
**Current State:** Implemented sync in `analysis.py`. Async refactor is EMAC-004 OpenClaw task.

---

## HEARTBEAT: Nightly Health Check

**Trigger:** After `run_nightly_analysis()` completes (or at 4:30 AM ET — 30 min after nightly snapshot job).

**Owner:** Performance Sentinel (OpenClaw)

**Action:**
1. `GET /api/performance/model-accuracy?days=30` — extract `margin_mae`. Flag if > 3 pts.
2. `GET /admin/portfolio/status` — extract `current_drawdown_pct`. Flag zones:
   - 0–10%: GREEN — log INFO only
   - 10–15%: YELLOW — WARNING, include in Morning Briefing
   - >15%: RED — circuit breaker active; verify dashboard shows error banner
3. `GET /api/predictions/today` — log verdict distribution (BET/CONSIDER/PASS counts).
4. Verify pytest status if any code was changed since last run: `pytest tests/ -q --tb=line`
5. Write summary to `memory/YYYY-MM-DD.md` (today's date).

**Escalation:**
- MAE > 3 pts for 7 consecutive days → queue manual recalibration task in HANDOFF.md
- Drawdown 10–15% → add to HANDOFF.md under "Operator Awareness"
- Drawdown >15% → add PRIORITY flag to HANDOFF.md; notify operator
- Any pytest failures → halt; do NOT allow next nightly run until fixed

**Expected Runtime:** < 60 s (all HTTP calls + pytest).

---

## HEARTBEAT: Weekly Calibration Review

**Trigger:** Monday 6 AM ET (day after Sunday auto-recalibration at 5 AM ET).

**Owner:** Performance Sentinel (OpenClaw)

**Action:**
1. `GET /admin/scheduler/status` — confirm `weekly_recalibration` job ran successfully.
2. Query last recalibration log entry from `model_parameters` table.
3. Compare new `home_advantage` and `sd_multiplier` values against `IDENTITY.md` baselines.
4. If parameters shifted by > 15% from defaults → flag for Architect review.
5. Update HANDOFF.md "Calibration History" section with new values and date.

**Escalation:**
- Recalibration skipped (< 30 bets) → log INFO, no action
- Parameter shift > 15% → add to HANDOFF.md as "Architect Review Required"
- Recalibration job failed → add to HANDOFF.md as CRITICAL, notify Gemini CLI

---

## Status Tracker

```json
{
  "integrity_sweep": {
    "status": "async_implemented",
    "async_target": "EMAC-004",
    "last_run": null
  },
  "nightly_health_check": {
    "status": "sentinel_active",
    "last_run": null
  },
  "weekly_calibration_review": {
    "status": "sentinel_active",
    "last_run": null
  }
}
```
