# Calibration Memory — CBB Edge Analyzer

> Maintained by: Claude Code (Master Architect)
> Warning: Do NOT change ha or sd_mult until V9 recalibration runs (need 50 settled V9 bets).

---

## Review: 2026-03-07 (A-27)

### Current Parameter Values

| Parameter | Default (code) | DB-calibrated value | Source |
|-----------|---------------|---------------------|--------|
| `home_advantage` | 3.09 pts | **2.419 pts** | V8-era recalibration, stored in model_parameters table |
| `sd_multiplier` | 0.85 | **1.0** | V8-era recalibration, stored in model_parameters table |
| `base_sd` | 11.0 pts | 11.0 pts (not recalibrated) | betting_model.py constructor default |
| `MIN_BET_EDGE` | 2.5% | 1.8% (EMAC-068 pre-GUARDIAN) | get_float_env("MIN_BET_EDGE", "1.8") |
| `weight_kenpom` | 0.342 | 0.342 | No drift observed |
| `weight_barttorvik` | 0.333 | 0.333 | No drift observed |
| `weight_evanmiya` | 0.325 | 0.325 | No drift observed |

### Recalibration System State

- **Auto-recalibration schedule:** Sunday 5:00 AM ET (APScheduler job `weekly_recalibration`)
- **Minimum bets threshold:** 30 settled bets with linked predictions (`MIN_BETS_FOR_RECALIBRATION`, default 30)
- **V9-specific threshold:** 50 settled V9 bets required before trusting V9-era recalibration signal (per K-3 audit)
- **V9 recal status: PENDING** — 0 BET-tier games since V9 launch (correct conservatism per K-3). V9 bets have not yet accumulated to the 50-bet threshold.

### SD Multiplier Notes

- sd_mult=1.0 widens the distribution relative to the 0.85 default. This compresses edges (makes BET verdicts harder to achieve), which is intentional conservatism during the V9 ramp-up period.
- The 1.0 value was set during V8-era recalibration. It should NOT be reduced until V9 bets provide a clean probability-calibration signal.
- Min-delta guard (0.03) prevents flip-flopping at the noise boundary (lesson from EMAC-031).

### Home Advantage Notes

- ha=2.419 is meaningfully below the 3.09 code default, suggesting the model had been over-crediting home teams during V8 calibration.
- The recalibrator uses a 25% correction rate per run with a +/-0.50 cap, so this reflects multiple runs of consistent signal.
- Safety bounds: [1.5, 5.5] — current value is well within range.

### MAE Drift Assessment

- No per-source MAE data available for dynamic weight adjustment (requires 7+ PerformanceSnapshot rows with populated kenpom_mae/barttorvik_mae/evanmiya_mae columns).
- Source weights (KP=0.342, BT=0.333, EM=0.325) are at their defaults — no drift observed.
- HEARTBEAT escalation threshold: MAE > 3 pts for 7 consecutive days triggers recalibration queue.

### Action Taken

**None.** Parameters frozen pending V9 recalibration.

### Next Review

- After 50 V9 bets settle, OR
- March 25, 2026 (whichever comes first)
- Trigger: check Railway logs for `"Recalibration complete"` or manually run `GET /admin/recalibrate?dry_run=true`

---

## Parameter History

| Date | ha | sd_mult | Trigger | Notes |
|------|----|---------|---------|-------|
| V8-era (pre-V9) | 3.09 -> 2.419 | 0.85 -> 1.0 | Auto recalibration | Multiple Sunday runs converged here |
| 2026-03-07 | 2.419 (frozen) | 1.0 (frozen) | A-27 review | Frozen pending V9 50-bet threshold |
