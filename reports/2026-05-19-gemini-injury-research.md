# Injury Status API Research Report

**Date:** 2026-05-19
**Subject:** Garrett Crochet Injury Status Discrepancy
**Focus:** Why does the API return a boolean `true` instead of the string `'IL'`?

## Executive Summary
The discrepancy where Garrett Crochet (and potentially others) displays a boolean `true` instead of `'IL'` stems from inconsistent type coercion across the backend API layer. The Yahoo API occasionally returns `True` (boolean) to indicate an active injury flag when a player is injured but not explicitly assigned to an IL slot. While some backend schemas handle this coercion, others—specifically standard Python dataclasses and certain Pydantic models—allow the boolean to pass through to the frontend.

## 1. Frontend Expectations
The frontend strongly types injury statuses as strings. In `frontend/lib/types.ts`:
```typescript
export interface InjuryFlag {
  player_id: string
  name: string
  status: "IL" | "IL10" | "IL60" | "DTD" | "OUT"
  injury_note?: string | null
  // ...
}
```
The frontend expects `status` to be a precise string enum. Furthermore, components like `WaiverPlayerOut` check `player.injury_status`, and Roster pages map the status through a `STATUS_LABELS` dictionary. When the backend sends an uncoerced boolean (or a stringified boolean like `"True"`), the frontend either fails to render the badge or renders the literal fallback text.

## 2. Backend Reality: The Yahoo API Quirk
As documented in `backend/data_contracts/yahoo_player.py`:
```python
        Three independent signals from live capture:
          1. status=True (Yahoo injury flag)
          2. injury_note is not None (body part string present)
          3. "IL" in positions (IL roster slot marker)

        Example: Crochet has status=True, injury_note=None, no "IL" in positions.
```
For Garrett Crochet, the Yahoo API literally returns `status: True` (boolean). 

## 3. Root Cause Analysis
The boolean `true` leaks to the frontend through several unpatched holes in the API layer:

### A. Bypassed Pydantic Validation via `dataclasses.asdict()`
In `backend/services/dashboard_service.py`, `InjuryFlag` is defined as a standard `@dataclass`, not a Pydantic model. 
When the `dashboard.injury_flags` list is serialized in `backend/main.py`:
```python
"injury_flags": [asdict(i) for i in dashboard.injury_flags],
```
The `asdict()` function performs no type coercion. If `injury_note` or `status` received a boolean `True` from Yahoo, it remains a Python `bool`, which FastAPI serializes as `true` in the JSON response, violating the frontend's string expectation.

### B. Missing Validators in Pydantic Models
While `LineupPlayerOut` and `CanonicalPlayerRow` have explicit `@field_validator` methods to coerce booleans (e.g., `coerce_injury_status_to_string`), several other models lack this protection:
- **`WaiverPlayerOut`**: Defines `status: Optional[str] = None` and `injury_status: Optional[str] = None` with no boolean coercion.
- **`StartingPitcherOut`**: Defines `status: str = "UNKNOWN"` and `injury_status: Optional[str] = None` with no boolean coercion.
- **`DropPlayerOut`**: Defines `status: Optional[str] = None` with no boolean coercion.

### C. Mapper Field Mismatches
In `backend/routers/fantasy.py` for waivers, the API populates `injury_status=p.get("injury_status")`. However, the Yahoo parser `_parse_player` only extracts `"status"` and `"injury_note"`. Thus, `injury_status` evaluates to `None`, causing the frontend's explicit `player.injury_status` check to fail silently, while the raw boolean `True` leaks through the `status` field instead.

## 4. Recommended Fixes (Do Not Modify Files Per Instructions)
To resolve this discrepancy across the board:
1. **Unify Validation**: Move the `coerce_injury_status_to_string` validator to a shared base class or utility function and apply it to `WaiverPlayerOut`, `StartingPitcherOut`, and `DropPlayerOut`.
2. **Fix Dataclass Serialization**: Convert `InjuryFlag` in `dashboard_service.py` to a Pydantic `BaseModel` so that strong typing and coercion are enforced before the JSON response is built.
3. **Correct Field Mappings**: In `player_mapper.py` and `fantasy.py`, ensure that the backend consistently coalesces Yahoo's `status` and `injury_note` into the `injury_status` field that the frontend expects.
