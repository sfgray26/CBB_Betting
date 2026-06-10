# Design Spec: Availability Guard + Roster Constraint Awareness
**Date:** 2026-06-10  
**Branch:** `stable/cbb-prod`  
**Priority:** P0 (Tasks 1–3), P1 (Task 4)

---

## Overview

Four independent augmentations to the War Room suite that prevent unsafe roster decisions and clarify contextual labels. All changes extend existing data flows rather than adding new endpoints.

---

## Task 1: Real-Time Availability Guard

### Problem
Waiver wire recommendation cards (`AddPanel`) show no indication when the suggested add player is injured or on IL. The `PlayerRow` component shows an `injury_status` badge but uses red for all statuses including DTD — which over-alarms.

### Data Source
BDL GOAT MLB injury feed. Already ingested into `IngestedInjury` table and loaded via `load_injury_overlays_for_yahoo_players()` in both `get_waiver()` and `get_waiver_recommendations()`. BDL provides `status: "DTD" | "10-Day-IL" | "15-Day-IL" | "60-Day-IL"` — not game-day lineup confirmations. Badge framing is therefore conservative.

### Backend Changes

**`backend/schemas.py` → `WaiverPlayerOut`**
- Add `availability_note: Optional[str] = None`

**`backend/routers/fantasy.py`**

In both `get_waiver()` and `get_waiver_recommendations()`, after applying the injury overlay to each player, compute `availability_note`:
- overlay status contains "IL" (case-insensitive) → `"On IL — check IL slot availability"`
- overlay status is "DTD" → `"DTD — confirm before adding"`
- no overlay → `None`

### Frontend Changes

**`frontend/app/(dashboard)/war-room/waiver/page.tsx`**

1. `PlayerRow` — Color-split `injury_status` badge:
   - Status contains "IL" → existing red (`bg-status-lost/10 text-status-lost`)
   - Status is "DTD" → amber (`bg-status-bubble/10 text-status-bubble`)

2. `AddPanel` in `RecommendationCard` — If `fa.availability_note` is set, render a warning line beneath the player name:
   ```tsx
   {fa.availability_note && (
     <p className="text-[10px] text-status-bubble font-semibold">⚠ {fa.availability_note}</p>
   )}
   ```

---

## Task 2: Roster Constraint Awareness

### Problem
The recommendations engine generates "ADD_DROP" moves for players who are on the waiver wire but currently on IL, even when the user has zero IL slots available. The current code adds a text hint to `rationale` (line 3213) but the recommendation tier remains unchanged and no structured constraint is surfaced.

### Backend Changes

**`backend/schemas.py` → `RosterMoveRecommendation`**
- Add `constraint_warning: Optional[str] = None`

**`backend/routers/fantasy.py` → `get_waiver_recommendations()`**

Before the recommendation loop, compute:
```python
from backend.services.waiver_edge_detector import il_capacity_info as _il_cap
_il_slots_available = _il_cap(my_roster)["available"] if my_roster else 0
```

In the loop body, after `fa` is selected, before appending the recommendation:
```python
_constraint = None
_fa_injury = (fa.injury_status or "").upper()
_IL_KEYWORDS = ("IL", "DL", "60-DAY", "15-DAY", "10-DAY")
if any(kw in _fa_injury for kw in _IL_KEYWORDS) and _il_slots_available == 0:
    _constraint = "IL slots full — move an injured player to IL first"
elif _faab_balance is not None and _faab_balance < 1:
    _constraint = "FAAB budget exhausted — free agents only"
```

Pass `constraint_warning=_constraint` to the `RosterMoveRecommendation` constructor.

### Frontend Changes

**`frontend/app/(dashboard)/war-room/waiver/page.tsx` → `RecommendationCard`**

Before the two-panel ADD/DROP row, if `rec.constraint_warning` is set:
```tsx
{rec.constraint_warning && (
  <div className="px-3 pt-3 pb-0">
    <div className="flex items-center gap-1.5 text-[11px] text-status-bubble font-semibold">
      <WarnIcon className="h-3.5 w-3.5 flex-shrink-0" />
      {rec.constraint_warning}
    </div>
  </div>
)}
```

---

## Task 3: Dashboard IL Crisis Detection

### Problem
`_get_lineup_gaps()` in `dashboard_service.py` filters "active" players as those whose `selected_position` is not an IL slot. This produces an empty gaps list (→ "No lineup gaps detected") when 3+ players are in active/bench slots but have confirmed injury statuses. The "Lineup Gaps: none" message is a false positive.

### Backend Changes

**`backend/services/dashboard_service.py` → `LineupGap` dataclass**
- Add `action_url: Optional[str] = None`

**`_get_lineup_gaps()` — post-processing block (append after existing gap detection):**
```python
_IL_CONFIRMED = {"il", "il10", "il60", "15-day-il", "60-day-il", "out"}
_SAFE_SELECTED_POS = {"IL", "IL10", "IL60", "NA", "DL"}
crisis_players = [
    p for p in roster
    if p.get("selected_position") not in _SAFE_SELECTED_POS
    and (p.get("injury_status") or "").strip().lower() in _IL_CONFIRMED
]
if len(crisis_players) >= 3:
    names = ", ".join(p["name"] for p in crisis_players[:3])
    gaps.append(LineupGap(
        position="ROSTER",
        severity="critical",
        message=(
            f"ROSTER EMERGENCY: {len(crisis_players)} injured players in active slots "
            f"({names}+) — move to IL slots now"
        ),
        suggested_add=None,
        action_url="/war-room/roster",
    ))
```

### Frontend Changes

**`frontend/app/(dashboard)/dashboard/_components/dashboard-client.tsx` → `LineupGapsWidget`**

1. Add `action_url?: string` to the `LineupGap` TypeScript type in `frontend/lib/types.ts`.
2. In `LineupGapsWidget`, for gaps where `gap.position === "ROSTER"` and `gap.severity === "critical"`, render an emergency-styled block instead of the dot + text list item:
   ```tsx
   {gap.position === 'ROSTER' ? (
     <li key={i} className="flex items-start gap-2 p-2 bg-status-lost/5 border border-status-lost/20 rounded">
       <AlertCircle className="mt-0.5 h-4 w-4 text-status-lost shrink-0" />
       <div className="flex-1 min-w-0">
         <p className="text-status-lost text-sm font-bold">{gap.message}</p>
         {gap.action_url && (
           <Link href={gap.action_url} className="inline-flex items-center gap-1 text-xs text-accent-gold mt-1 hover:underline">
             Go to Roster <ArrowRight className="h-3 w-3" />
           </Link>
         )}
       </div>
     </li>
   ) : (
     /* existing dot + message rendering */
   )}
   ```

---

## Task 4: Win Probability Context Labels

### Problem
War Room shows live current-week win probability from MCMC simulation. Weekly Preview shows projected next-week win probability. There is no visual context indicating which week or whether the data is live vs. projected, creating a confusing discrepancy when numbers differ.

### Backend Changes
None. `MatchupResponse.week` (War Room) and `MatchupPreviewResponse.week_number` (Preview) already return the week number.

### Frontend Changes

**`frontend/app/(dashboard)/war-room/page.tsx`**

After the "War Room" span in the page header row, insert:
```tsx
{matchup.data && (
  <span className="text-[10px] px-2 py-1 bg-accent-gold/10 text-accent-gold border border-accent-gold/30 rounded font-bold tracking-wider uppercase">
    Week {matchup.data.week} · IN-FLIGHT
  </span>
)}
```

**`frontend/app/(dashboard)/war-room/preview/page.tsx`**

After the "Weekly Preview" span in the header, insert:
```tsx
{data.week_number > 0 && (
  <span className="text-[10px] px-2 py-1 bg-text-muted/10 text-text-muted border border-text-muted/30 rounded font-bold tracking-wider uppercase">
    Week {data.week_number} · PREVIEW
  </span>
)}
```

---

## File Change Summary

| File | Change |
|------|--------|
| `backend/schemas.py` | Add `availability_note` to `WaiverPlayerOut`; add `constraint_warning` to `RosterMoveRecommendation` |
| `backend/routers/fantasy.py` | Compute `availability_note` in waiver assembly; compute `constraint_warning` in recommendations loop |
| `backend/services/dashboard_service.py` | Add `action_url` to `LineupGap`; add IL crisis post-processor in `_get_lineup_gaps()` |
| `frontend/lib/types.ts` | Add `availability_note`, `constraint_warning`, `action_url` optional fields |
| `frontend/app/(dashboard)/war-room/waiver/page.tsx` | Color-split injury badge; add `availability_note` to `AddPanel`; add `constraint_warning` strip to `RecommendationCard` |
| `frontend/app/(dashboard)/dashboard/_components/dashboard-client.tsx` | Emergency-styled roster gap rendering with action link |
| `frontend/app/(dashboard)/war-room/page.tsx` | Week + IN-FLIGHT badge in header |
| `frontend/app/(dashboard)/war-room/preview/page.tsx` | Week + PREVIEW badge in header |

## Tests Required

- `tests/test_waiver_recommendations.py` — add case: FA player with IL status + IL full → `constraint_warning` set
- `tests/test_dashboard_service.py` — add case: roster with 3 IL-status active players → crisis gap appended
- Syntax checks: `backend/schemas.py`, `backend/routers/fantasy.py`, `backend/services/dashboard_service.py`
