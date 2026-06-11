# Wave 3B — Bulk Lineup Commit

**Date:** 2026-05-20
**Severity:** P3 MEDIUM — UX: no way to apply optimization results in one click

---

## Problem

After clicking [OPTIMIZE LINEUP], users had to click [Apply] 14 times individually.
No batch endpoint existed. Tedious and error-prone.

---

## Solution

### Backend: `POST /api/fantasy/roster/bulk-apply`

**File:** `backend/routers/fantasy.py` (new endpoint after `move_roster_player`)
**Schema:** `backend/contracts.py` (`BulkRosterMove`, `BulkRosterMoveRequest`, `BulkRosterMoveResponse`)

Key design:
- **Phase 1 validate**: checks all positions + all player_keys against current roster before any Yahoo call. Returns 400 with `{errors:[...]}` if anything fails.
- **Atomic execution**: builds the complete lineup with ALL moves applied, then calls `client.set_lineup()` exactly once. Yahoo processes it as a single transaction — partial execution is a Yahoo-side decision, not ours.
- **Partial failure reporting**: cross-references Yahoo's `applied[]` list against requested moves. Returns `applied_count`, `failed_count`, `errors[]` so the frontend can surface failures specifically.

```python
POST /api/fantasy/roster/bulk-apply
{
  "moves": [
    {"player_key": "469.l.72586.p.12345", "target_position": "1B"},
    {"player_key": "469.l.72586.p.67890", "target_position": "SP"}
  ]
}
→ 200: {"applied_count": 2, "failed_count": 0, "errors": []}
→ 400: {"detail": {"errors": ["Invalid position 'INVALID' for player ..."]}}
```

### Frontend: `OptimizePanel` — Apply All button

**Files:** `frontend/app/(dashboard)/war-room/roster/page.tsx`, `frontend/lib/api.ts`, `frontend/lib/types.ts`

Changes:
1. `OptimizePanel` gets `onApplyAll` + `isApplyingAll` props
2. "Apply All N Moves" button appears below the assignment grid
3. Clicking it shows an inline confirmation listing all proposed moves (scrollable, max-height 40)
4. Confirm → calls `endpoints.bulkApplyMoves(moves)` via `bulkApplyMutation`
5. Cancel → dismisses without API call
6. Success: shows "Applied N moves successfully" toast + dismisses the optimize panel
7. Partial failure: shows both success count and error details
8. Individual [Apply] buttons unchanged and still functional

---

## Files Changed

| File | Change |
|------|--------|
| `backend/contracts.py` | Added `BulkRosterMove`, `BulkRosterMoveRequest`, `BulkRosterMoveResponse` |
| `backend/routers/fantasy.py` | New `bulk_apply_roster_moves` endpoint; import new schemas |
| `frontend/lib/types.ts` | Added `BulkRosterMove`, `BulkRosterMoveResponse` interfaces |
| `frontend/lib/api.ts` | Added `bulkApplyMoves` function |
| `frontend/app/(dashboard)/war-room/roster/page.tsx` | Updated `OptimizePanel` + added `bulkApplyMutation` + `handleApplyAll` |
| `tests/test_roster_move_api.py` | 7 new `TestBulkRosterMoveEndpoint` tests |

**Test result:** 14/14 passing (7 existing + 7 new).
**TypeScript:** `npx tsc --noEmit` — zero errors.
