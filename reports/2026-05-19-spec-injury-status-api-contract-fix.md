# Spec Memo: Injury Status API Contract Fix (Boolean → String)

**Date:** 2026-05-19  
**Author:** Kimi CLI (Research)  
**Status:** Ready for Claude Code implementation review  
**Priority:** P1 (Degraded — contract violation breaks frontend UI rendering)

---

## 1. Problem Statement

UAT audit (K-NEXT-4, 2026-05-13) identified that **Garrett Crochet's `injury_status` is emitted as JSON boolean `true` instead of string `"IL"`**. This is an API contract violation: the frontend `STATUS_COLORS` and `STATUS_LABELS` maps expect string keys (`"IL"`, `"playing"`, `"probable"`, etc.) and render nothing (or crash) when given a boolean.

### Impact
- Players on IL display without the red injury badge
- Roster page cannot apply `STATUS_COLORS['IL']` styling → falls through to default/unstyled
- Downstream consumers (mobile app, Discord alerts, OpenClaw briefs) that rely on typed contracts may fail validation

---

## 2. Root Cause Analysis

### 2.1 Yahoo API Behavior
Yahoo's `team/{team_key}/roster/players` endpoint returns injury metadata in a nested structure. In some response shapes, `injury_status` or `injury_note` arrives as a bare JSON boolean (`true`) rather than a string. This appears to happen when Yahoo's internal flag is a simple "has injury" toggle rather than a typed status string.

### 2.2 Current Code Paths (Three Layers — Inconsistent!)

| Layer | File | Logic | Boolean `true` maps to... |
|-------|------|-------|---------------------------|
| **Mapper** | `backend/services/player_mapper.py:251-258` | `_raw_status = "IL" if _raw_status else None` | `"IL"` ✅ |
| **Contract schema** | `backend/contracts.py:371-377` | `return "IL" if v else "Active"` | `"IL"` ✅ |
| **Router schema** | `backend/schemas.py:569-575` | `return "Active" if v else "Inactive"` | `"Active"` ❌ |

**The conflict:** `schemas.py` `RosterPlayerOut` validator assumes `true = Active`, which is semantically wrong for an injured player. If any code path serializes through `RosterPlayerOut` instead of `CanonicalPlayerRow`, the boolean leaks through as `"Active"` — or worse, if validation is bypassed, the raw boolean `true` leaks into JSON.

### 2.3 Leak Path
The roster endpoint (`GET /api/fantasy/roster`) is decorated with:
```python
response_model=CanonicalRosterResponse  # uses CanonicalPlayerRow
```

However, `RosterPlayerOut` is still imported and referenced in `schemas.py` as a legacy response shape. If any internal helper, cached response, or alternate endpoint (e.g., dashboard roster panel, lineup optimizer preview) inadvertently constructs a `RosterPlayerOut` instead of `CanonicalPlayerRow`, the wrong validator fires.

The UAT observed raw `true` in the JSON — this suggests **Pydantic validation is being bypassed entirely** for that field, likely because the value is being injected into a plain dict after model construction or because an intermediate `dict()` conversion strips the validator.

---

## 3. Proposed Fix

### 3.1 Align All Validators to the Same Semantics

**Rule:** `injury_status` is a **string field** representing the player's roster slot / availability state. It is NEVER a boolean in the canonical contract.

| Raw Yahoo value | Canonical value | Rationale |
|-----------------|-----------------|-----------|
| `true` (bool) | `"IL"` | Boolean flag from Yahoo means "on injured list" |
| `false` (bool) | `null` | Boolean false means "no injury status" |
| `"IL"` | `"IL"` | Pass through |
| `"DTD"` | `"DTD"` | Pass through |
| `"NA"` | `"minors"` | Normalize Yahoo's not-available code |
| `null` / missing | `null` | No injury information |

### 3.2 Code Changes

#### A. `backend/schemas.py` — Fix `RosterPlayerOut` validator
```python
@field_validator("status", "injury_status", mode="before")
@classmethod
def coerce_status_to_string(cls, v):
    """Coerce boolean injury flags to strings. 
    
    Yahoo returns boolean true for IL-flagged players; false means no injury.
    """
    if isinstance(v, bool):
        return "IL" if v else None
    return v
```

> **Note:** Remove `"status"` from this validator if `status` has its own coercion logic, or apply the same semantics. Currently `status` and `injury_status` share a validator but have different semantic meanings — `status` is roster state (Active/BN/IL) while `injury_status` is injury detail. They should NOT share a validator.

#### B. `backend/contracts.py` — Keep current validator (already correct)
```python
@field_validator("injury_status", "injury_return_timeline", mode="before")
def coerce_injury_fields_to_string(cls, v):
    if isinstance(v, bool):
        return "IL" if v else None  # "Active" was wrong — false means null, not Active
    return v
```

> **Note:** Change `"Active"` to `None` for the `else` branch. A boolean `false` from Yahoo means "no injury", not "actively healthy". The frontend distinguishes `null` (no data) from `"Active"` (confirmed healthy).

#### C. `backend/services/player_mapper.py` — Keep current logic (already correct)
```python
if isinstance(_raw_status, bool):
    _raw_status = "IL" if _raw_status else None
injury_status = _raw_note or _raw_status
```

#### D. Add a post-serialization guard
In `backend/routers/fantasy.py`, after constructing the roster response, add a defensive pass that asserts no boolean values exist in `injury_status` fields:
```python
for player in response.players:
    if isinstance(player.injury_status, bool):
        logger.error("BOOLEAN LEAK: injury_status for %s is bool", player.player_name)
        player = player.model_copy(update={"injury_status": "IL" if player.injury_status else None})
```

This is a runtime safety net during the transition period.

### 3.3 Remove Legacy Ambiguity

**Delete** `RosterPlayerOut` from `schemas.py` if it is no longer the canonical roster shape. All roster endpoints should use `CanonicalPlayerRow` (from `contracts.py`) exclusively. If `RosterPlayerOut` is still needed for backward compatibility:
1. Add a deprecation comment
2. Make it a thin wrapper around `CanonicalPlayerRow`
3. Do NOT duplicate validators — import from contracts

---

## 4. Testing Requirements

### Unit tests
```python
def test_injury_status_boolean_true_becomes_il():
    row = CanonicalPlayerRow(
        player_name="Garrett Crochet",
        team="BOS",
        eligible_positions=["SP"],
        status="IL",
        injury_status=True,  # Yahoo raw
        freshness=...,
    )
    assert row.injury_status == "IL"

def test_injury_status_boolean_false_becomes_none():
    row = CanonicalPlayerRow(
        player_name="Shohei Ohtani",
        team="LAD",
        eligible_positions=["DH"],
        status="Active",
        injury_status=False,
        freshness=...,
    )
    assert row.injury_status is None

def test_json_serialization_never_emits_boolean():
    row = CanonicalPlayerRow(..., injury_status=True, ...)
    json_str = row.model_dump_json()
    assert '"injury_status": true' not in json_str
    assert '"injury_status": "IL"' in json_str
```

### Integration test
```python
def test_roster_endpoint_no_boolean_injury_status(client):
    resp = client.get("/api/fantasy/roster", headers={"x-api-key": ...})
    for player in resp.json()["players"]:
        status = player.get("injury_status")
        assert status is None or isinstance(status, str), \
            f"{player['player_name']}: injury_status is {type(status).__name__} = {status}"
```

---

## 5. Acceptance Criteria

- [ ] `GET /api/fantasy/roster` returns `"injury_status": "IL"` for all IL players (never `true`)
- [ ] `GET /api/fantasy/roster` returns `"injury_status": null` for healthy players (never `false`)
- [ ] `GET /api/fantasy/waiver` applies the same coercion for `injury_status` on waiver players
- [ ] All three schema locations (`contracts.py`, `schemas.py`, `player_mapper.py`) use identical boolean semantics
- [ ] Unit tests for boolean coercion pass
- [ ] Integration test for roster endpoint passes
- [ ] Frontend UAT re-run: IL badge renders correctly for all injured players

---

## 6. Related Files

| File | Role |
|------|------|
| `backend/services/player_mapper.py:251-258` | Mapper coercion logic (already correct) |
| `backend/contracts.py:371-377` | `CanonicalPlayerRow` validator (mostly correct, fix "Active" → `None`) |
| `backend/schemas.py:569-575` | `RosterPlayerOut` validator (wrong — fix `true` → `"IL"`, `false` → `None`) |
| `backend/schemas.py:317-323` | `LineupPlayerOut` validator (verify same semantics) |
| `frontend/app/(dashboard)/war-room/roster/page.tsx:39-53` | Frontend status color maps |
| `reports/2026-05-13-ui-uat-audit.md` | Original UAT finding |
