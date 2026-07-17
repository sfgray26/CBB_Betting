# UAT_DEV_TRIAGE.md — Fantasy Baseball UAT (2026-07-17) Codebase Triage

> Prepared by: Senior React/Node triage pass, 2026-07-17
> Source: UAT report (Lindor Truffles / Week 17 vs. High&TightyWhitey's)
> Scope: "Critical Bugs" + "High Severity" items only. Sprint 1 implements the 3 CRITICAL fixes.

---

## CRITICAL 1 — Optimize Lineup: single "Apply" executes all 18 moves + false confirmation

**Bug Description:**
Clicking the per-player "Apply" (Carson Benge) in the Optimize Lineup panel executed all 18
optimizer-recommended moves, while the confirmation banner read "Moved Pete Alonso from 1B to 1B"
(a no-op message about a different player).

**Root Cause Hypothesis:**
The frontend wiring is *superficially* correct
(`OptimizePanel` row → `onApplyMove(assignment.player_key, assignment.assigned_slot)` →
`handleMove` → `POST /api/fantasy/roster/move`), so the dangerous behavior lives one layer down:

1. **Backend submits the ENTIRE 22-player lineup on a "single" move.** `move_roster_player()`
   builds `lineup` from *every* player returned by `client.get_roster()` and calls
   `client.set_lineup(lineup)` (fantasy.py ~4392–4448). `get_roster()` reads through a
   **5-minute in-memory cache** (`YahooAPICache`, yahoo_client_resilient.py:206). If the cached
   positions have drifted from Yahoo's live state (e.g. cache populated before/around optimizer
   experimentation), the "single move" PUT force-writes every player's stale position — silently
   executing what looks like the full 18-move bulk plan. Yahoo accepts partial player lists
   (proven by `set_lineup`'s own per-player fallback path, yahoo_client_resilient.py:1407–1426),
   so scoping the payload is safe.
2. **Confirmation toast trusts the backend message blindly.** `moveMutation.onSuccess` renders
   `data.message` verbatim (roster/page.tsx:1027) with no check that `data.player_key` /
   `data.success` match the clicked player, so a mismatched/no-op backend message
   ("Moved Pete Alonso from 1B to 1B") is shown as a success banner.

**Files to Modify:**
- `backend/routers/fantasy.py` (`move_roster_player`, lines ~4378–4448)
- `frontend/app/(dashboard)/war-room/roster/page.tsx` (`OptimizePanel` props wiring ~1253,
  `moveMutation.onSuccess` ~1025–1044, new dedicated `handleApplyOptimizerMove`)
- `tests/test_roster_move_swap_logic.py` (2 assertions that enforce the old full-lineup contract)

**Proposed Fix:**
1. Backend: after validation, build the `set_lineup` payload with **only** the moved player
   (plus the displaced occupant when a swap is required). Never include untouched players.
2. Backend: fetch the roster with `bypass_cache=True` so occupant/swap detection is based on
   live Yahoo state, not the 5-minute cache.
3. Frontend: give `OptimizePanel` a dedicated `onApplyMove` handler (separate from the
   PlayerCard dropdown path) that passes exactly `(assignment.player_key, assignment.assigned_slot)`.
4. Frontend: in `moveMutation.onSuccess`, build the banner from the **mutation variables**
   (clicked player's name from the roster cache + requested slot); only show success when
   `data.success === true && data.player_key === variables.playerId`, otherwise show an error.
5. Update the two swap-logic test assertions that require untouched players to appear in the
   `set_lineup` payload (contract change: payload is now scoped to moved + swapped players).

---

## CRITICAL 2 — War Room: 422 "0 opponent players" but "6–0 LEADING" still rendered

**Bug Description:**
War Room shows the simulate error "422: Roster data unavailable — Yahoo returned 22 my players,
0 opponent players…" while directly below it the matchup scoreboard confidently renders
"6–0 LEADING".

**Root Cause Hypothesis:**
Two independent data paths render side by side with no consistency gate:
- `POST /api/fantasy/matchup/simulate` 422s (fantasy.py:7538) → error banner via
  `simulateMutation.isError` (war-room/page.tsx:243–248).
- `GET /api/fantasy/matchup` succeeds but returns an opponent with empty/zero stats
  (`opp_entry = ("", "Unknown", {})` fallback, fantasy.py:6282–6284). `MatchupHeader.computeScore`
  (matchup-header.tsx:15–27) then counts a "win" for every category where my value beats the
  opponent's missing/zero value → fabricated "6–0 LEADING" at full visual prominence.
Neither component knows the other failed, and there is no "opponent data present?" guard.

**Files to Modify:**
- `frontend/app/(dashboard)/war-room/page.tsx` (derive `opponentDataUnavailable`, pass prop,
  suppress `CategoryBattlefield`, clear stale `simulateData` on error)
- `frontend/components/war-room/matchup-header.tsx` (conditional fallback UI instead of scoreboard)

**Proposed Fix:**
1. In `WarRoomPage`, compute `opponentDataUnavailable = simulateMutation.isError ||
   !matchup.data.opponent.team_key || every opponent stat is null/empty`.
2. Pass it to `MatchupHeader`; when true, render "Matchup data unavailable — opponent data
   failed to load from Yahoo. Score hidden to avoid showing a misleading record." in place of
   the W/L numbers, LEADING/TRAILING label, and projected strip.
3. Render a matching fallback in place of `CategoryBattlefield` (it derives per-category W/L
   chips from the same opponent stats).
4. Clear `simulateData` on simulate error so a stale projection strip can't survive a failed
   re-run.

---

## CRITICAL 3 — Strikeouts (K): Win on Roster page, Loss on Waiver page (same 0-vs-2)

**Bug Description:**
Identical live batting-K stat (0 vs 2) shows green **Win** on My Roster and red **Loss** on
Waiver Wire; overall records disagree (6W-0L-12T vs 5W-1L-12T).

**Root Cause Hypothesis:**
Two different evaluation paths with two different direction tables:
- **Roster page** evaluates client-side in `buildMatchupRows` (roster/page.tsx:537–570) using
  canonical keys + `LOWER_IS_BETTER` from `lib/types.ts:434`, which correctly includes `K_B`
  → 0 < 2 = **Win**.
- **Waiver page** renders `d.winning` computed **server-side** in the waiver endpoint
  (fantasy.py:2045–2056) via `compare_category(cat, …)`. `cat` there is the Yahoo display name
  `"K(B)"` (from `_B_RENAME`, fantasy.py:1994). `CATEGORY_DIRECTIONS`
  (category_comparator.py:22–52) has no `"K(B)"` entry and *defaults unknown categories to
  higher-is-better* (category_comparator.py:164) → 0 vs 2 = **Loss**.
- The waiver page additionally hard-codes its own direction list for highlighting
  (`['ERA','WHIP','L','HRA'].includes(label)`, waiver/page.tsx:365) — a third, divergent copy.

**Files to Modify:**
- `frontend/lib/types.ts` (add shared `evaluateCategoryOutcome()` + Yahoo-variant key aliases)
- `frontend/app/(dashboard)/war-room/roster/page.tsx` (`buildMatchupRows` → shared evaluator)
- `frontend/app/(dashboard)/war-room/waiver/page.tsx` (`CategoryDeficitsBar` → shared evaluator
  for outcome, W/L/T counts, and ahead-highlight)
- `backend/services/category_comparator.py` (add canonical/variant direction entries: `K_B`,
  `HR_P` lower; `K_P`, `HR_B`, `H`, `TB`, `NSB`, `NSV` higher; `"K(B)"` lower)
- `backend/routers/fantasy.py` (waiver `category_deficits` build ~2045: canonicalize the
  category key before `compare_category` and emit canonical keys in `CategoryDeficitOut`)

**Proposed Fix:**
1. Create one exported `evaluateCategoryOutcome(category, myVal, oppVal): 'W'|'L'|'T'|null` in
   `lib/types.ts` (normalizes Yahoo-variant keys like `K(B)`→`K_B`, then applies
   `LOWER_IS_BETTER`).
2. Roster `buildMatchupRows` and Waiver `CategoryDeficitsBar` both import and use that exact
   function (waiver recomputes outcome from `my_total`/`opponent_total` instead of trusting
   `d.winning`; W/L/T header counts derive from the same result).
3. Backend: canonicalize waiver deficit keys (`K(B)`→`K_B`, `K(P)`→`K_P`, `HRA`→`HR_P`, …)
   before comparison and extend `CATEGORY_DIRECTIONS` so server-side `winning` agrees —
   keeping need-score and any other consumer of `category_deficits` consistent too.

---

## HIGH 1 — Dashboard: "Move to IL slot immediately" for players already in IL slots

**Bug Description:**
Dashboard injury alerts tell the user to IL-slot Edwin Díaz, Garrett Crochet, Michael Soroka,
yet all three already occupy IL slots (IL 3/3 full).

**Root Cause Hypothesis:**
`_get_injury_flags` (dashboard_service.py:752–838) fires `action = "Move to IL slot immediately"`
from injury *status* without (previously) checking `selected_position`. **Note:** the working
tree already contains an uncommitted partial fix (adds `already_in_il_slot` → severity "info" /
"Monitor status") — but it still emits a flag, so the alert still renders as if action is needed.
UAT expectation: no actionable alert for already-correct placement.

**Files to Modify:**
- `backend/services/dashboard_service.py` (`_get_injury_flags`, ~787–838)
- Possibly `frontend/app/(dashboard)/dashboard/_components/dashboard-client.tsx` if "info"
  flags are rendered in the same alert list (verify rendering/filtering during Sprint 2)

**Proposed Fix:**
1. Keep the uncommitted `already_in_il_slot` detection, but exclude those players from the
   actionable alerts entirely (either drop the flag or add an `informational` partition the UI
   renders separately/collapsed).
2. Regression test: injured player with `selected_position in (IL, IL10, IL15, IL60)` → no
   "Move to IL" action; injured player in an active slot → critical alert retained.

---

## HIGH 2 — Dashboard "19 healthy · 3 injured" undercounts impaired players

**Bug Description:**
Kyle Harrison (real 15-Day IL, occupying an active SP slot, no alert anywhere), plus
Day-to-Day Geraldo Perdomo and Juan Soto, are all counted as "healthy".

**Root Cause Hypothesis:**
In `_get_injury_flags` (dashboard_service.py:787–800): `status` comes from Yahoo and is then
**overwritten by the injury overlay** (`if overlay and overlay.status: status = overlay.status`,
line 795–796). Perdomo's overlay ETA is "EXPIRED — CHECK STATUS" elsewhere, implying overlay
records can blank/stale-out `status`; a bool `True` from Yahoo is coerced to `""` (line 790–791).
If the final `status` is `""`, `is_injured` is False unless the player sits in an IL slot —
Harrison is in an *active SP slot*, so if his Yahoo/overlay status resolves to empty he is
silently counted healthy and gets no flag. DTD *is* in `injury_statuses`, so DTD players
counting as healthy proves their status is being blanked upstream.

**Files to Modify:**
- `backend/services/dashboard_service.py` (`_get_injury_flags` status resolution, ~787–836)
- `backend/fantasy_baseball/` injury overlay loader (`load_injury_overlays_for_yahoo_players`)
  — verify expired-ETA overlays don't blank a live Yahoo status

**Proposed Fix:**
1. Never let an overlay *remove* an injury designation: resolve status as
   `overlay.status or yahoo_status` (overlay may only refine, not erase), and treat expired-ETA
   overlays as "unknown — keep Yahoo status + flag CHECK STATUS".
2. Add explicit handling so any player whose real designation is IL10/IL15/IL60/DTD/OUT is
   counted injured and (if in an active slot) flagged — Harrison's case (IL player in active
   slot) should produce the existing "ROSTER EMERGENCY" style warning (cf. line ~361–378).
3. Regression tests for: DTD counted injured; IL-status player in active slot counted injured +
   critical alert; expired overlay ETA does not blank status.

---

## HIGH 3 — Optimize panel note: "Data from 2026-07-17, not requested 2026-07-17"

**Bug Description:**
The Optimize Lineup panel always renders a self-comparing, meaningless date note.

**Root Cause Hypothesis:**
`optimize_roster` (fantasy.py:5235, 5274, 5579–5583): `actual_data_date` starts as the
`target_date` **string**, but when scores exist it becomes `max(as_of_dates)` — a
`datetime.date` **object** from `PlayerScore.as_of_date`. The guard
`actual_data_date != target_date` then compares `date(2026,7,17) != "2026-07-17"` → always
True, and both sides stringify identically → "Data from 2026-07-17, not requested 2026-07-17".

**Files to Modify:**
- `backend/routers/fantasy.py` (`optimize_roster`, ~5235, ~5274, ~5577–5583)

**Proposed Fix:**
1. Normalize both sides to ISO strings before comparison/rendering
   (`actual.isoformat() if isinstance(actual, date) else str(actual)`).
2. Only append the note when the normalized dates genuinely differ.
3. Unit test: same-date case renders no note; different-date case renders the note once with
   two different dates.

---

## Sprint Plan (per UAT roadmap)

- **Sprint 1 (this pass):** CRITICAL 1–3 fixes above + focused test updates
  (`test_roster_move_swap_logic.py`, comparator tests if needed) + `py_compile` /
  `pytest` subset / `tsc --noEmit` verification.
- **Sprint 2 (awaiting review):** HIGH 1–3 + UX quick wins (inline errors, IL dropdown
  eligibility, sort redundancy, projection freshness label, team-name consistency, stat labels).
- **Sprint 3:** player search, React #419 hydration errors, optimizer panel staleness, mobile
  audit, STALE badge redesign.

---

## SPRINT 1 IMPLEMENTATION STATUS (2026-07-17) — COMPLETE, AWAITING REVIEW

| Bug | Fix applied | Verification |
|-----|-------------|--------------|
| CRITICAL 1 — Apply button | Backend `set_lineup` payload scoped to moved + swapped players only; roster fetch `bypass_cache=True`; frontend dedicated `handleApplyOptimizerMove`; toast built from mutation variables, gated on `data.success && data.player_key === variables.playerId` | 121 roster-move/comparator tests pass |
| CRITICAL 2 — Fabricated score | `opponentDataUnavailable` gate in War Room page (simulate error ∨ empty team_key ∨ no usable opponent stats); `MatchupHeader` renders "Matchup data unavailable" fallback; `CategoryBattlefield` replaced with fallback panel; stale `simulateData` cleared on error | `tsc --noEmit` clean; vitest matchup-strip 4/4 |
| CRITICAL 3 — K W/L flip | Shared `evaluateCategoryOutcome()` in `lib/types.ts` used by BOTH Roster `buildMatchupRows` and Waiver `CategoryDeficitsBar`; backend `CATEGORY_DIRECTIONS` extended (`K(B)`/`K_B` = lower, etc.); waiver `category_deficits` keys canonicalized before comparison | comparator + consistency suites pass |

**Test totals:** 152 backend tests passed (roster move API, swap logic, category comparator,
category consistency, waiver gates, IL support, matchup API, category tracker).
`frontend: tsc --noEmit` clean; vitest `matchup-strip.test.tsx` 4/4.
Known pre-existing failure (untouched): `streaming-recommendations.test.tsx` — missing
`@testing-library/react` dependency in frontend node_modules.

**Pre-existing uncommitted work preserved:** `dashboard_service.py` IL-slot alert partial fix
and waiver OF-filter handling were already in the working tree and were left intact.

**STOP** — per instructions, Sprint 2 (HIGH 1–3 + UX quick wins) will not start until review.
