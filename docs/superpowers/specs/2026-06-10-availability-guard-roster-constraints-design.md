# Design Spec: Availability Guard + Roster Constraint Awareness
**Date:** 2026-06-10 (updated post-review)  
**Branch:** `stable/cbb-prod`  
**Priority:** P0 (Tasks 1–3), P1 (Task 4)

---

## Overview

Four independent augmentations to the War Room suite that prevent unsafe roster decisions and clarify contextual labels. All changes extend existing data flows rather than adding new endpoints.

---

## Task 1: Real-Time Availability Guard

### Problem
Waiver wire recommendation cards (`AddPanel`) show no indication when the suggested add player is injured or on IL. The `PlayerRow` component shows an `injury_status` badge but uses red for all statuses including DTD — which over-alarms.

**Caballero pattern (root cause):** BDL injury feed only reflects formal IL/DTD designations (updated ~daily). Manager day-off announcements (e.g., "Caballero won't start June 10") are not captured. Players with no formal injury status but a confirmed day off still ranked as top streamers.

### Data Sources

**Tier 1 — BDL injury feed (automatic, daily):** Provides `status: "DTD" | "10-Day-IL" | "15-Day-IL" | "60-Day-IL"`. Already ingested via `load_injury_overlays_for_yahoo_players()`.

**Tier 2 — Daily availability blacklist (admin-seeded):** New `DailyAvailabilityOverride` DB table allows manual entry for confirmed day-offs and lineup scratches. Populated via admin API for high-value cases. Future: MLB lineup API feed can write to this table automatically.

### New Model — `DailyAvailabilityOverride`

**`backend/models.py`**
```python
class DailyAvailabilityOverride(Base):
    __tablename__ = "daily_availability_overrides"
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_key = Column(String(64), nullable=False)   # Yahoo player_key
    player_name = Column(String(128), nullable=False)
    game_date = Column(Date, nullable=False)           # ET local date
    status = Column(String(32), nullable=False)        # "OUT" | "DAY_OFF"
    note = Column(String(256), nullable=True)
    source = Column(String(32), default="admin")       # "admin" | "mlb_lineup_api" (future)
    created_at = Column(DateTime, nullable=False, default=func.now())
    __table_args__ = (UniqueConstraint("player_key", "game_date", name="uq_override_player_date"),)
```

**Alembic migration required** — new table, no column changes to existing tables.

### Admin Endpoints

**`backend/routers/fantasy.py`** (or a new `admin.py` router)

```
POST /api/admin/availability-override
Body: {player_key, player_name, status, note}
→ Upserts for today's ET date

DELETE /api/admin/availability-override/{player_key}
→ Removes today's entry for that player
```

Both endpoints require `verify_api_key` (same auth as all fantasy endpoints).

### Backend Changes

**`backend/schemas.py` → `WaiverPlayerOut`**
- Add `availability_note: Optional[str] = None`

**`backend/routers/fantasy.py`**

Pre-load today's blacklist once per request (before `_score_fa` closure):
```python
from zoneinfo import ZoneInfo as _ZI
_today_et = datetime.now(_ZI("America/New_York")).date()
_blacklist_keys: set[str] = set()
try:
    from backend.models import DailyAvailabilityOverride as _DAO
    _blacklist_keys = {
        r.player_key
        for r in db.query(_DAO.player_key)
            .filter(_DAO.game_date == _today_et, _DAO.status.in_(["OUT", "DAY_OFF"]))
            .all()
    }
except Exception:
    pass  # non-fatal
```

In `_score_fa()` (recommendations path) and the equivalent assembly in `get_waiver()`, after computing `_injury_status` from the overlay, derive `availability_note`:
```python
_pkey = p.get("player_key") or ""
if _pkey and _pkey in _blacklist_keys:
    _avail_note = "NOT AVAILABLE TODAY — day off confirmed"
    need_score = 0.0   # suppress from top-N ranking
elif _injury_status and any(kw in _injury_status.upper() for kw in ("IL", "DL", "60-DAY", "15-DAY", "10-DAY")):
    _avail_note = "On IL — check IL slot availability"
elif _injury_status and "DTD" in _injury_status.upper():
    _avail_note = "DTD — confirm before adding"
else:
    _avail_note = None
```

Pass `availability_note=_avail_note` to `WaiverPlayerOut`.

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

### IL Slot Counting Clarification
`il_capacity_info(roster)` returns `{"used": used, "total": total, "available": max(0, total - used)}`.  
`available` = currently **empty** IL slots (total minus occupied). The spec's logic is correct — no adjustment needed.

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

`AlertCircle` is already imported at line 14 — no new import needed.

1. Add `action_url?: string` to the `LineupGap` TypeScript type in `frontend/lib/types.ts`.
2. In `LineupGapsWidget`, for gaps where `gap.position === "ROSTER"`, render an emergency-styled block:
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

After the "Weekly Preview" span in the header, insert (using `accent-primary` — the project's blue accent, #2563eb — to distinguish projected from live):
```tsx
{data.week_number > 0 && (
  <span className="text-[10px] px-2 py-1 bg-accent-primary/10 text-accent-primary border border-accent-primary/30 rounded font-bold tracking-wider uppercase">
    Week {data.week_number} · PREVIEW
  </span>
)}
```

Note: `accent-blue` does not exist in the project's Tailwind config. `accent-primary` (#2563eb) is the correct blue token.

---

## File Change Summary

| File | Change |
|------|--------|
| `backend/models.py` | Add `DailyAvailabilityOverride` model |
| `backend/alembic/versions/` | New migration — create `daily_availability_overrides` table |
| `backend/schemas.py` | Add `availability_note` to `WaiverPlayerOut`; add `constraint_warning` to `RosterMoveRecommendation` |
| `backend/routers/fantasy.py` | Admin endpoints; blacklist pre-load; `availability_note` in waiver assembly; `constraint_warning` in recommendations loop |
| `backend/services/dashboard_service.py` | Add `action_url` to `LineupGap`; add IL crisis post-processor in `_get_lineup_gaps()` |
| `frontend/lib/types.ts` | Add `availability_note`, `constraint_warning`, `action_url` optional fields; add `DailyAvailabilityOverride` type |
| `frontend/app/(dashboard)/war-room/waiver/page.tsx` | Color-split injury badge; `availability_note` in `AddPanel`; `constraint_warning` strip in `RecommendationCard` |
| `frontend/app/(dashboard)/dashboard/_components/dashboard-client.tsx` | Emergency-styled roster gap with action link |
| `frontend/app/(dashboard)/war-room/page.tsx` | Week + IN-FLIGHT badge in header |
| `frontend/app/(dashboard)/war-room/preview/page.tsx` | Week + PREVIEW badge (accent-primary blue) in header |

## Tests Required

- `tests/test_waiver_recommendations.py` — case: FA with IL status + IL full → `constraint_warning` set; case: blacklisted player_key → `availability_note = "NOT AVAILABLE TODAY"` and `need_score = 0.0`
- `tests/test_dashboard_service.py` — case: roster with 3+ IL-status active players → crisis gap appended with `action_url`
- Syntax checks: `backend/models.py`, `backend/schemas.py`, `backend/routers/fantasy.py`, `backend/services/dashboard_service.py`
