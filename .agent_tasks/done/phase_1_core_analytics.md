# Task: Phase 1 — Core Analytics Pages ✅ DONE

**STATUS: DONE**
**Completed by:** Claude Code (Master Architect)
**Completed:** March 18-19, 2026
**Branch:** claude/fix-clv-null-safety-fPcKB

## Pages Built

- `/performance` — ROI, win rate, bankroll curve, rolling windows
- `/clv` — CLV distribution, scatter, top/bottom 10
- `/bet-history` — sortable table with all bet logs
- `/calibration` — probability calibration chart + Brier score
- `/alerts` — live + historical alerts with acknowledge

## Validation

- OpenClaw claw-validated: ALL 5 pages PASS
- Null-safety: all API fields guarded
- Decimal display: all roi/win_rate/clv fields × 100

## Lessons

- API returns roi/win_rate/clv as decimals (0.043) — must × 100 for display
- Empty state: `{ message: "...", total_bets: 0 }` — no `overall` key when no data
- `parseISO(null)` crashes — guard `created_at` before parsing
