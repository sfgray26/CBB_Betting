# Task: V9.2 Model Recalibration

**STATUS: GUARDIAN — DO NOT TOUCH UNTIL APRIL 7, 2026**
**Assigned to:** Claude Code (Master Architect)
**Estimated:** 1 session
**Priority:** HIGH (post-tournament)

## ⛔ GUARDIAN LOCK

This task is locked until April 7 (NCAA Championship Day).
Do NOT modify `betting_model.py`, `analysis.py`, or any backend service files
until the tournament is complete. Any changes risk destabilizing the live model.

## Goal

Recalibrate V9.1 model to V9.2 — unblock the over-conservative bet rate.

## Root Cause (from HANDOFF)

V9.1 stacks: SNR scalar (~0.70) × integrity scalar (~0.85) × fractional Kelly (÷2.0)
= effective divisor ~3.37x (intended ~2.0x). MIN_BET_EDGE fix was Phase 1; these are Phase 2.

## Changes Required

| File | Parameter | Old | New |
|------|-----------|-----|-----|
| `betting_model.py` | `sd_mult` | 1.0 | 0.80 |
| `betting_model.py` | `ha` (home advantage) | 2.419 | 2.85 |
| `betting_model.py` | `SNR_KELLY_FLOOR` | 0.50 | 0.75 |
| `betting_model.py` | `model_version` | 'v9.1' | 'v9.2' |

## References

- `reports/K12_RECALIBRATION_SPEC_V92.md` — full justification + math
- `reports/K13_POSSESSION_SIM_AUDIT.md` — push-aware Kelly guidance

## Success Criteria

- [ ] All parameters updated
- [ ] `pytest tests/ -q` — passes (target: 683/686+)
- [ ] BET rate improves from ~3% toward 8-12% in backtesting
- [ ] Model version bumped to `v9.2`

## Wire Haslametrics (same session)

- `backend/services/haslametrics.py` already built (12 tests pass)
- Add to `backend/services/ratings.py` with 32.5% weight (replaces EvanMiya)
- See `docs/THIRD_RATING_SOURCE.md`
