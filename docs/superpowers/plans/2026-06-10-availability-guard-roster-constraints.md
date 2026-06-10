# Availability Guard + Roster Constraint Awareness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent unsafe roster decisions in the War Room by surfacing availability warnings, IL/FAAB constraints, and an emergency IL-crisis alert on the dashboard, plus add week-context badges to War Room and Weekly Preview headers.

**Architecture:** All changes extend existing data flows inline — no new endpoints except the admin blacklist API. Backend adds fields to existing Pydantic schemas and extends two waiver assembly closures (`_to_waiver_player` and `_score_fa`). Dashboard service gets a post-processing block after `_get_lineup_gaps`. Frontend adds optional fields to TypeScript interfaces and injects UI components into existing cards/headers.

**Tech Stack:** Python/FastAPI backend, SQLAlchemy (sync), Next.js/React frontend, Tailwind CSS, pytest (source-inspection + unit patterns).

**Test runner:** `venv/Scripts/python -m pytest tests/<file>.py::<test> -v`  
**Syntax check:** `venv/Scripts/python -m py_compile <file>`

---

## File Map

| File | Task(s) | Change |
|------|---------|--------|
| `backend/models.py` | T1 | Add `DailyAvailabilityOverride` model at end of file |
| `backend/schemas.py` | T1 | Add `availability_note` to `WaiverPlayerOut`; add `constraint_warning` to `RosterMoveRecommendation` |
| `backend/routers/admin.py` | T4 | Add POST + DELETE `/api/admin/availability-override` endpoints |
| `backend/routers/fantasy.py` | T2, T3, T5 | Blacklist pre-load + `_avail_note` in both waiver closures; `_constraint` check before recommendations append |
| `backend/services/dashboard_service.py` | T6 | Add `action_url` to `LineupGap`; IL crisis block in `_get_lineup_gaps` |
| `frontend/lib/types.ts` | T7 | Add optional fields to `WaiverAvailablePlayer`, `WaiverRecommendation`, `LineupGap` |
| `frontend/app/(dashboard)/war-room/waiver/page.tsx` | T8 | Color-split badge; `availability_note` in `AddPanel`; `constraint_warning` strip |
| `frontend/app/(dashboard)/dashboard/_components/dashboard-client.tsx` | T9 | ROSTER emergency gap rendering |
| `frontend/app/(dashboard)/war-room/page.tsx` | T10 | IN-FLIGHT week badge |
| `frontend/app/(dashboard)/war-room/preview/page.tsx` | T10 | PREVIEW week badge |
| `tests/test_availability_guard.py` | T1–T5 | New test file covering model, schemas, blacklist, and constraint |
| `tests/test_dashboard_il_crisis.py` | T6 | New test file for crisis detection |

---

## Task 1: Model + Schema Foundation

**Files:**
- Modify: `backend/models.py:2337` (append after last line)
- Modify: `backend/schemas.py:435` (after `closer_role` in `WaiverPlayerOut`)
- Modify: `backend/schemas.py:513` (after `roster_context` in `RosterMoveRecommendation`)
- Create: `tests/test_availability_guard.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_availability_guard.py`:

```python
"""Tests for availability guard + roster constraint features."""
import pytest


def test_daily_availability_override_model_importable():
    """DailyAvailabilityOverride model must exist and have required columns."""
    from backend.models import DailyAvailabilityOverride
    assert hasattr(DailyAvailabilityOverride, "__tablename__")
    assert DailyAvailabilityOverride.__tablename__ == "daily_availability_overrides"
    cols = {c.name for c in DailyAvailabilityOverride.__table__.columns}
    assert "player_key" in cols
    assert "game_date" in cols
    assert "status" in cols


def test_waiver_player_out_has_availability_note():
    """WaiverPlayerOut must accept availability_note without error."""
    from backend.schemas import WaiverPlayerOut
    p = WaiverPlayerOut(
        player_id="123.p.456",
        name="Test Player",
        team="NYY",
        position="OF",
        availability_note="DTD — confirm before adding",
    )
    assert p.availability_note == "DTD — confirm before adding"


def test_waiver_player_out_availability_note_defaults_none():
    """availability_note defaults to None for players with no overlay."""
    from backend.schemas import WaiverPlayerOut
    p = WaiverPlayerOut(
        player_id="123.p.456",
        name="Test Player",
        team="NYY",
        position="OF",
    )
    assert p.availability_note is None


def test_roster_move_recommendation_has_constraint_warning():
    """RosterMoveRecommendation must accept constraint_warning."""
    from backend.schemas import RosterMoveRecommendation, WaiverPlayerOut
    add_p = WaiverPlayerOut(player_id="x", name="Add", team="T", position="SP")
    rec = RosterMoveRecommendation(
        action="ADD_DROP",
        add_player=add_p,
        drop_player_name="Drop Guy",
        drop_player_position="OF",
        rationale="test",
        category_targets=[],
        need_score=5.0,
        confidence=0.7,
        constraint_warning="IL slots full — move an injured player to IL first",
    )
    assert rec.constraint_warning == "IL slots full — move an injured player to IL first"


def test_roster_move_recommendation_constraint_warning_defaults_none():
    """constraint_warning defaults to None for unconstrained moves."""
    from backend.schemas import RosterMoveRecommendation, WaiverPlayerOut
    add_p = WaiverPlayerOut(player_id="x", name="Add", team="T", position="SP")
    rec = RosterMoveRecommendation(
        action="ADD_DROP",
        add_player=add_p,
        drop_player_name="Drop Guy",
        drop_player_position="OF",
        rationale="test",
        category_targets=[],
        need_score=5.0,
        confidence=0.7,
    )
    assert rec.constraint_warning is None
```

- [ ] **Step 2: Run tests to verify they fail**

```
venv/Scripts/python -m pytest tests/test_availability_guard.py -v
```

Expected: all 5 tests FAIL with `ImportError` or `ValidationError`.

- [ ] **Step 3: Add `DailyAvailabilityOverride` to `backend/models.py`**

Append after the last line (2337) of `backend/models.py`:

```python


class DailyAvailabilityOverride(Base):
    """Admin-seeded daily availability overrides (day-offs, game-day scratches).

    Populated via POST /api/admin/availability-override.
    Future: MLB lineup API feed can write source="mlb_lineup_api" entries.
    """
    __tablename__ = "daily_availability_overrides"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_key = Column(String(64), nullable=False)
    player_name = Column(String(128), nullable=False)
    game_date = Column(Date, nullable=False)
    status = Column(String(32), nullable=False)   # "OUT" | "DAY_OFF"
    note = Column(String(256), nullable=True)
    source = Column(String(32), default="admin")  # "admin" | "mlb_lineup_api"
    created_at = Column(DateTime, default=_now_et)

    __table_args__ = (
        UniqueConstraint("player_key", "game_date", name="uq_override_player_date"),
        Index("idx_dao_game_date", "game_date"),
    )
```

- [ ] **Step 4: Add `availability_note` to `WaiverPlayerOut` in `backend/schemas.py`**

In `WaiverPlayerOut`, after the `closer_role` field (line 435), add:

```python
    availability_note: Optional[str] = None  # "NOT AVAILABLE TODAY" | "DTD — confirm" | "On IL" | None
```

- [ ] **Step 5: Add `constraint_warning` to `RosterMoveRecommendation` in `backend/schemas.py`**

In `RosterMoveRecommendation`, after the `roster_context` field (line 513), add:

```python
    constraint_warning: Optional[str] = None  # "IL slots full — ..." | "FAAB exhausted" | None
```

- [ ] **Step 6: Run syntax checks**

```
venv/Scripts/python -m py_compile backend/models.py
venv/Scripts/python -m py_compile backend/schemas.py
```

Both must exit 0.

- [ ] **Step 7: Run tests to verify they pass**

```
venv/Scripts/python -m pytest tests/test_availability_guard.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 8: Commit**

```
git add backend/models.py backend/schemas.py tests/test_availability_guard.py
git commit -m "feat(backend): add DailyAvailabilityOverride model + availability_note + constraint_warning schema fields"
```

---

## Task 2: Availability Note in Main Waiver Endpoint (`_to_waiver_player`)

**Files:**
- Modify: `backend/routers/fantasy.py:2017` (before `_to_waiver_player` definition)
- Modify: `backend/routers/fantasy.py:2122` (inside `_to_waiver_player`, after injury overlay)
- Modify: `backend/routers/fantasy.py:2224` (inside `_to_waiver_player` constructor call)
- Test: `tests/test_availability_guard.py` (add test)

- [ ] **Step 1: Add test for blacklist + availability_note in source**

Append to `tests/test_availability_guard.py`:

```python
def test_to_waiver_player_blacklist_preload_present():
    """get_fantasy_waiver_recommendations must pre-load _blacklist_keys before _to_waiver_player."""
    import pathlib
    src = (pathlib.Path(__file__).parent.parent / "backend" / "routers" / "fantasy.py").read_text()
    assert "_blacklist_keys" in src, (
        "get_fantasy_waiver_recommendations must define _blacklist_keys before _to_waiver_player"
    )
    assert "NOT AVAILABLE TODAY" in src, (
        "_to_waiver_player must emit 'NOT AVAILABLE TODAY' for blacklisted players"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```
venv/Scripts/python -m pytest tests/test_availability_guard.py::test_to_waiver_player_blacklist_preload_present -v
```

Expected: FAIL — `_blacklist_keys` not yet in source.

- [ ] **Step 3: Add blacklist pre-load before `_to_waiver_player` in `backend/routers/fantasy.py`**

After line 2017 (the blank line before `def _to_waiver_player`), insert:

```python
        # Daily availability blacklist: admin-seeded player_keys confirmed OUT or on day-off.
        _blacklist_keys: set = set()
        try:
            from backend.models import DailyAvailabilityOverride as _DAO
            from zoneinfo import ZoneInfo as _ZI
            _bl_today = datetime.now(_ZI("America/New_York")).date()
            _blacklist_keys = {
                r.player_key
                for r in db.query(_DAO.player_key)
                    .filter(_DAO.game_date == _bl_today, _DAO.status.in_(["OUT", "DAY_OFF"]))
                    .all()
            }
        except Exception:
            pass  # non-fatal: empty blacklist on any error

```

- [ ] **Step 4: Add `_avail_note` computation inside `_to_waiver_player` after overlay (line 2122)**

After the line `_injury_timeline = getattr(_overlay, "return_timeline", None)` (line 2122), insert:

```python
            _pkey = p.get("player_key") or ""
            if _pkey and _pkey in _blacklist_keys:
                _avail_note: Optional[str] = "NOT AVAILABLE TODAY — day off confirmed"
                need_score = 0.0
            elif _injury_status and any(kw in (_injury_status or "").upper() for kw in ("IL", "DL", "60-DAY", "15-DAY", "10-DAY")):
                _avail_note = "On IL — check IL slot availability"
            elif _injury_status and "DTD" in (_injury_status or "").upper():
                _avail_note = "DTD — confirm before adding"
            else:
                _avail_note = None
```

- [ ] **Step 5: Pass `availability_note` to `WaiverPlayerOut` constructor in `_to_waiver_player`**

In the `WaiverPlayerOut(...)` constructor call (ending at line 2225), add before the closing `)`:

```python
                availability_note=_avail_note,
```

The constructor's closing lines should now look like:
```python
                park_factor=round(_get_park_factor(p.get("team") or "", "run"), 3),
                closer_role=_closer_role,
                availability_note=_avail_note,
            )
```

- [ ] **Step 6: Run syntax check**

```
venv/Scripts/python -m py_compile backend/routers/fantasy.py
```

Must exit 0.

- [ ] **Step 7: Run tests**

```
venv/Scripts/python -m pytest tests/test_availability_guard.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 8: Commit**

```
git add backend/routers/fantasy.py tests/test_availability_guard.py
git commit -m "feat(backend): add availability blacklist + availability_note to main waiver endpoint"
```

---

## Task 3: Availability Note in Recommendations Endpoint (`_score_fa`)

**Files:**
- Modify: `backend/routers/fantasy.py:2810` (before `_score_fa` definition)
- Modify: `backend/routers/fantasy.py:2842` (inside `_score_fa`, after injury overlay)
- Modify: `backend/routers/fantasy.py:2937` (inside `_score_fa` constructor call)

- [ ] **Step 1: Add blacklist pre-load before `_score_fa` in `backend/routers/fantasy.py`**

After line 2810 (the blank line after the pitcher quality lookup block, before `def _score_fa`), insert:

```python
        # Daily availability blacklist for recommendations endpoint (mirrors main waiver path).
        _rec_blacklist_keys: set = set()
        try:
            from backend.models import DailyAvailabilityOverride as _DAO_rec
            from zoneinfo import ZoneInfo as _ZI_rec
            _rec_bl_today = datetime.now(_ZI_rec("America/New_York")).date()
            _rec_blacklist_keys = {
                r.player_key
                for r in db.query(_DAO_rec.player_key)
                    .filter(_DAO_rec.game_date == _rec_bl_today, _DAO_rec.status.in_(["OUT", "DAY_OFF"]))
                    .all()
            }
        except Exception:
            pass  # non-fatal

```

- [ ] **Step 2: Add `_avail_note` computation inside `_score_fa` after overlay (line 2842)**

After the line `_injury_timeline = getattr(_overlay, "return_timeline", None)` (line 2842), insert:

```python
            _pkey_rec = p.get("player_key") or ""
            if _pkey_rec and _pkey_rec in _rec_blacklist_keys:
                _avail_note_rec: Optional[str] = "NOT AVAILABLE TODAY — day off confirmed"
                need_score = 0.0
            elif _injury_status and any(kw in (_injury_status or "").upper() for kw in ("IL", "DL", "60-DAY", "15-DAY", "10-DAY")):
                _avail_note_rec = "On IL — check IL slot availability"
            elif _injury_status and "DTD" in (_injury_status or "").upper():
                _avail_note_rec = "DTD — confirm before adding"
            else:
                _avail_note_rec = None
```

- [ ] **Step 3: Pass `availability_note` to `WaiverPlayerOut` constructor in `_score_fa`**

In the `WaiverPlayerOut(...)` constructor call in `_score_fa` (ending at line ~2937), add before the closing `)`:

```python
                availability_note=_avail_note_rec,
```

The constructor's closing lines should look like:
```python
                park_factor=round(_get_park_factor_rec(p.get("team") or "", "run"), 3),
                availability_note=_avail_note_rec,
            )
```

- [ ] **Step 4: Run syntax check + tests**

```
venv/Scripts/python -m py_compile backend/routers/fantasy.py
venv/Scripts/python -m pytest tests/test_availability_guard.py -v
```

Both must pass.

- [ ] **Step 5: Commit**

```
git add backend/routers/fantasy.py
git commit -m "feat(backend): add availability blacklist + availability_note to recommendations endpoint"
```

---

## Task 4: Admin Endpoints for Daily Blacklist

**Files:**
- Modify: `backend/routers/admin.py` (append two new endpoints near end of file)
- Test: `tests/test_availability_guard.py` (add test)

- [ ] **Step 1: Add test for admin route existence**

Append to `tests/test_availability_guard.py`:

```python
def test_admin_availability_override_routes_exist():
    """POST and DELETE /api/admin/availability-override must exist in admin router source."""
    import pathlib
    src = (pathlib.Path(__file__).parent.parent / "backend" / "routers" / "admin.py").read_text()
    assert "/api/admin/availability-override" in src, (
        "admin.py must define POST /api/admin/availability-override"
    )
    assert "DailyAvailabilityOverride" in src, (
        "admin.py must import and use DailyAvailabilityOverride"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```
venv/Scripts/python -m pytest tests/test_availability_guard.py::test_admin_availability_override_routes_exist -v
```

Expected: FAIL.

- [ ] **Step 3: Add admin endpoints to `backend/routers/admin.py`**

Find the end of `backend/routers/admin.py` and append:

```python


# ──────────────────────────────────────────────────────────────────────────────
# Daily Availability Override (admin-seeded blacklist for game-day scratches)
# ──────────────────────────────────────────────────────────────────────────────

from pydantic import BaseModel as _PydanticBase


class _AvailabilityOverrideIn(_PydanticBase):
    player_key: str
    player_name: str
    status: str = "OUT"          # "OUT" | "DAY_OFF"
    note: Optional[str] = None


@router.post("/api/admin/availability-override", tags=["admin"])
async def create_availability_override(
    payload: _AvailabilityOverrideIn,
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """Upsert a day-off or scratch override for today (ET). Suppresses player from waiver top rankings."""
    from backend.models import DailyAvailabilityOverride
    from zoneinfo import ZoneInfo
    today_et = datetime.now(ZoneInfo("America/New_York")).date()
    existing = (
        db.query(DailyAvailabilityOverride)
        .filter(
            DailyAvailabilityOverride.player_key == payload.player_key,
            DailyAvailabilityOverride.game_date == today_et,
        )
        .first()
    )
    if existing:
        existing.status = payload.status
        existing.note = payload.note
        existing.player_name = payload.player_name
    else:
        db.add(DailyAvailabilityOverride(
            player_key=payload.player_key,
            player_name=payload.player_name,
            game_date=today_et,
            status=payload.status,
            note=payload.note,
            source="admin",
        ))
    db.commit()
    return {"ok": True, "player_key": payload.player_key, "game_date": str(today_et), "status": payload.status}


@router.delete("/api/admin/availability-override/{player_key}", tags=["admin"])
async def delete_availability_override(
    player_key: str,
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """Remove today's availability override for a player."""
    from backend.models import DailyAvailabilityOverride
    from zoneinfo import ZoneInfo
    today_et = datetime.now(ZoneInfo("America/New_York")).date()
    deleted = (
        db.query(DailyAvailabilityOverride)
        .filter(
            DailyAvailabilityOverride.player_key == player_key,
            DailyAvailabilityOverride.game_date == today_et,
        )
        .delete()
    )
    db.commit()
    return {"ok": True, "deleted": deleted > 0, "player_key": player_key}
```

- [ ] **Step 4: Run syntax check**

```
venv/Scripts/python -m py_compile backend/routers/admin.py
```

Must exit 0.

- [ ] **Step 5: Run tests**

```
venv/Scripts/python -m pytest tests/test_availability_guard.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 6: Commit**

```
git add backend/routers/admin.py tests/test_availability_guard.py
git commit -m "feat(backend): add POST/DELETE /api/admin/availability-override endpoints"
```

---

## Task 5: Constraint Warning in Recommendations Loop

**Files:**
- Modify: `backend/routers/fantasy.py` — before `for fa in scored_fas[:15]:` loop (line ~3099) and inside loop before `recommendations.append`

- [ ] **Step 1: Add tests**

Append to `tests/test_availability_guard.py`:

```python
def test_constraint_warning_il_full_present_in_source():
    """Recommendations loop must check IL capacity and set constraint_warning."""
    import pathlib
    src = (pathlib.Path(__file__).parent.parent / "backend" / "routers" / "fantasy.py").read_text()
    assert "_il_slots_available" in src, (
        "fantasy.py must compute _il_slots_available before the recommendations loop"
    )
    assert "IL slots full" in src, (
        "fantasy.py must set constraint_warning 'IL slots full' when no IL capacity"
    )
    assert "constraint_warning=_constraint" in src, (
        "RosterMoveRecommendation must receive constraint_warning=_constraint"
    )


def test_constraint_warning_faab_present_in_source():
    """Recommendations loop must check FAAB balance."""
    import pathlib
    src = (pathlib.Path(__file__).parent.parent / "backend" / "routers" / "fantasy.py").read_text()
    assert "FAAB budget exhausted" in src, (
        "fantasy.py must set constraint_warning for zero FAAB"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```
venv/Scripts/python -m pytest tests/test_availability_guard.py::test_constraint_warning_il_full_present_in_source tests/test_availability_guard.py::test_constraint_warning_faab_present_in_source -v
```

Expected: FAIL.

- [ ] **Step 3: Add FAAB fetch + IL pre-load before recommendations loop**

In `get_waiver_recommendations()`, after `scored_fas = sorted(...)` is built (around line 2939–2943) and before `for fa in scored_fas[:15]:` (line 3099), add (inside the outer `try` block):

```python
        # Constraint pre-computations for the recommendation loop.
        from backend.services.waiver_edge_detector import il_capacity_info as _il_cap
        _il_slots_available = _il_cap(my_roster)["available"] if my_roster else 0

        # FAAB balance for budget constraint check (non-fatal).
        _rec_faab_balance: Optional[float] = None
        try:
            _rec_faab_balance = client.get_faab_balance()
        except Exception:
            pass
```

- [ ] **Step 4: Add `_constraint` computation inside the loop, before `recommendations.append`**

In the loop body, just before the `roster_context = { ... }` dict (around line 3316), insert:

```python
            _constraint: Optional[str] = None
            _fa_injury_upper = (fa.injury_status or "").upper()
            _IL_KW = ("IL", "DL", "60-DAY", "15-DAY", "10-DAY")
            if any(kw in _fa_injury_upper for kw in _IL_KW) and _il_slots_available == 0:
                _constraint = "IL slots full — move an injured player to IL first"
            elif _rec_faab_balance is not None and _rec_faab_balance < 1:
                _constraint = "FAAB budget exhausted — free agents only"
```

- [ ] **Step 5: Pass `constraint_warning` to `RosterMoveRecommendation` constructor**

In the `recommendations.append(RosterMoveRecommendation(...))` call (around line 3322), add `constraint_warning=_constraint` before the closing `))`:

```python
            recommendations.append(RosterMoveRecommendation(
                action="ADD_DROP",
                add_player=fa,
                drop_player_name=drop_candidate["name"],
                drop_player_position=drop_candidate["positions"][0] if drop_candidate["positions"] else "?",
                rationale=rationale,
                category_targets=[
                    k for k, v in (fa.category_contributions or {}).items()
                    if isinstance(v, (int, float)) and v > 0
                ],
                need_score=round(gain, 3),
                confidence=confidence,
                statcast_signals=fa_signals,
                regression_delta=fa_reg_delta,
                win_prob_before=_mcmc.get("win_prob_before", 0.0),
                win_prob_after=_mcmc.get("win_prob_after", 0.0),
                win_prob_gain=_mcmc.get("win_prob_gain", 0.0),
                category_win_probs=_mcmc.get("category_win_probs_after", {}),
                mcmc_enabled=_mcmc.get("mcmc_enabled", False),
                drop_player=drop_out,
                category_deltas=category_deltas,
                alternative_drops=alternative_drops,
                positional_impact=positional_impact,
                roster_context=roster_context,
                constraint_warning=_constraint,
            ))
```

- [ ] **Step 6: Run syntax check + tests**

```
venv/Scripts/python -m py_compile backend/routers/fantasy.py
venv/Scripts/python -m pytest tests/test_availability_guard.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 7: Commit**

```
git add backend/routers/fantasy.py tests/test_availability_guard.py
git commit -m "feat(backend): add constraint_warning to recommendations — IL slots + FAAB checks"
```

---

## Task 6: IL Crisis Detection in Dashboard Service

**Files:**
- Modify: `backend/services/dashboard_service.py:47` (`LineupGap` dataclass)
- Modify: `backend/services/dashboard_service.py:341` (before `return gaps` in `_get_lineup_gaps`)
- Create: `tests/test_dashboard_il_crisis.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_dashboard_il_crisis.py`:

```python
"""Tests for IL crisis detection in _get_lineup_gaps."""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import asdict


def _make_roster_player(name: str, selected_position: str, injury_status: str | None = None) -> dict:
    return {
        "name": name,
        "selected_position": selected_position,
        "positions": [selected_position],
        "injury_status": injury_status,
    }


def test_lineup_gap_has_action_url_field():
    """LineupGap dataclass must have an action_url field."""
    from backend.services.dashboard_service import LineupGap
    gap = LineupGap(
        position="ROSTER",
        severity="critical",
        message="test",
        action_url="/war-room/roster",
    )
    assert gap.action_url == "/war-room/roster"
    d = asdict(gap)
    assert "action_url" in d


def test_lineup_gap_action_url_defaults_none():
    """action_url defaults to None for normal gaps."""
    from backend.services.dashboard_service import LineupGap
    gap = LineupGap(position="OF", severity="warning", message="No eligible player")
    assert gap.action_url is None


@pytest.mark.asyncio
async def test_il_crisis_appended_when_three_injured_active():
    """_get_lineup_gaps must append a ROSTER EMERGENCY gap when 3+ injured active players."""
    from backend.services.dashboard_service import DashboardService

    # 3 injured players in active/bench slots
    roster = [
        _make_roster_player("Player A", "BN", injury_status="il"),
        _make_roster_player("Player B", "BN", injury_status="il10"),
        _make_roster_player("Player C", "OF", injury_status="out"),
        _make_roster_player("Healthy D", "1B"),
        _make_roster_player("Healthy E", "SS"),
    ]

    service = DashboardService.__new__(DashboardService)
    service.reliability_engine = MagicMock()
    service.reliability_engine.validate_yahoo_roster = MagicMock(
        return_value=MagicMock(is_valid=True, errors=[])
    )

    mock_client = MagicMock()
    mock_client.get_roster = MagicMock(return_value=roster)
    service._get_yahoo_client = MagicMock(return_value=mock_client)

    with patch("backend.services.dashboard_service.SessionLocal") as mock_sl:
        mock_db = MagicMock()
        mock_sl.return_value = mock_db   # SessionLocal() called directly, not as context manager
        service._detect_pitcher_swap_gaps = MagicMock(return_value=[])

        gaps, _, _ = await service._get_lineup_gaps("user1", None)

    crisis_gaps = [g for g in gaps if g.position == "ROSTER"]
    assert len(crisis_gaps) == 1, f"Expected 1 ROSTER crisis gap, got {len(crisis_gaps)}: {gaps}"
    assert crisis_gaps[0].severity == "critical"
    assert "ROSTER EMERGENCY" in crisis_gaps[0].message
    assert crisis_gaps[0].action_url == "/war-room/roster"


@pytest.mark.asyncio
async def test_no_il_crisis_when_fewer_than_three_injured():
    """No ROSTER EMERGENCY gap when fewer than 3 injured active players."""
    from backend.services.dashboard_service import DashboardService

    roster = [
        _make_roster_player("Player A", "BN", injury_status="il"),
        _make_roster_player("Healthy B", "1B"),
        _make_roster_player("Healthy C", "SS"),
    ]

    service = DashboardService.__new__(DashboardService)
    service.reliability_engine = MagicMock()
    service.reliability_engine.validate_yahoo_roster = MagicMock(
        return_value=MagicMock(is_valid=True, errors=[])
    )
    mock_client = MagicMock()
    mock_client.get_roster = MagicMock(return_value=roster)
    service._get_yahoo_client = MagicMock(return_value=mock_client)

    with patch("backend.services.dashboard_service.SessionLocal") as mock_sl:
        mock_db = MagicMock()
        mock_sl.return_value = mock_db
        service._detect_pitcher_swap_gaps = MagicMock(return_value=[])

        gaps, _, _ = await service._get_lineup_gaps("user1", None)

    crisis_gaps = [g for g in gaps if g.position == "ROSTER"]
    assert len(crisis_gaps) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```
venv/Scripts/python -m pytest tests/test_dashboard_il_crisis.py -v
```

Expected: `test_lineup_gap_has_action_url_field` and `test_il_crisis_appended_when_three_injured_active` FAIL (field doesn't exist yet).

- [ ] **Step 3: Add `action_url` to `LineupGap` dataclass**

In `backend/services/dashboard_service.py`, the `LineupGap` dataclass is at lines 41–48. After the `suggested_add` field (line 47), add:

```python
    action_url: Optional[str] = None
```

Full dataclass should look like:
```python
@dataclass
class LineupGap:
    """Identifies an unfilled or sub-optimally filled lineup slot."""
    position: str
    severity: str  # "critical", "warning", "info", "optimization"
    message: str
    suggested_add: Optional[str] = None
    action_url: Optional[str] = None
```

- [ ] **Step 4: Add IL crisis post-processor to `_get_lineup_gaps`**

In `_get_lineup_gaps`, after the Phase 2 try/except block (after the `logger.warning("_get_lineup_gaps Phase 2...")` line, around line 341), insert before `return gaps, filled_count, len(required_positions)`:

```python
            # Phase 3: IL crisis detection.
            # If 3+ rostered players have confirmed injury status but are NOT in IL slots,
            # the "no gaps" verdict is a false positive. Override with ROSTER EMERGENCY.
            _IL_CONFIRMED_STATUS = {"il", "il10", "il60", "15-day-il", "60-day-il", "out"}
            _SAFE_IL_POSITIONS = {"IL", "IL10", "IL60", "NA", "DL"}
            try:
                crisis_players = [
                    p for p in roster
                    if p.get("selected_position") not in _SAFE_IL_POSITIONS
                    and (p.get("injury_status") or "").strip().lower() in _IL_CONFIRMED_STATUS
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
            except Exception as _ph3_err:
                logger.warning("_get_lineup_gaps Phase 3 (IL crisis) failed: %s", _ph3_err)
```

- [ ] **Step 5: Run syntax check**

```
venv/Scripts/python -m py_compile backend/services/dashboard_service.py
```

Must exit 0.

- [ ] **Step 6: Run tests**

```
venv/Scripts/python -m pytest tests/test_dashboard_il_crisis.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 7: Run full test suite to check for regressions**

```
venv/Scripts/python -m pytest tests/ -q --tb=short
```

Expected: no new failures beyond pre-existing baseline.

- [ ] **Step 8: Commit**

```
git add backend/services/dashboard_service.py tests/test_dashboard_il_crisis.py
git commit -m "feat(backend): add IL crisis detection to _get_lineup_gaps — ROSTER EMERGENCY gap when 3+ injured active"
```

---

## Task 7: Frontend TypeScript Type Updates

**Files:**
- Modify: `frontend/lib/types.ts:210` (after `LineupGap.suggested_add`)
- Modify: `frontend/lib/types.ts:550` (after `WaiverAvailablePlayer.closer_role`)
- Modify: `frontend/lib/types.ts:614` (after `WaiverRecommendation.roster_context`)

- [ ] **Step 1: Add `action_url` to `LineupGap` interface**

In `frontend/lib/types.ts`, the `LineupGap` interface (lines 205–210):

```typescript
export interface LineupGap {
  position: string
  severity: "critical" | "warning" | "optimization" | "info"
  message: string
  suggested_add?: string | null
  action_url?: string | null   // add this line
}
```

- [ ] **Step 2: Add `availability_note` to `WaiverAvailablePlayer` interface**

In `frontend/lib/types.ts`, the `WaiverAvailablePlayer` interface (lines 516–550), after `closer_role`:

```typescript
  closer_role?: 'CLOSER' | 'NO_SAVE_ROLE' | null
  availability_note?: string | null   // add this line
}
```

- [ ] **Step 3: Add `constraint_warning` to `WaiverRecommendation` interface**

In `frontend/lib/types.ts`, the `WaiverRecommendation` interface (lines 593–615), after `roster_context`:

```typescript
  roster_context: WaiverRosterContext
  constraint_warning?: string | null   // add this line
}
```

- [ ] **Step 4: Commit**

```
git add frontend/lib/types.ts
git commit -m "feat(frontend): add availability_note, constraint_warning, action_url to TypeScript types"
```

---

## Task 8: Frontend Waiver Page Changes

**Files:**
- Modify: `frontend/app/(dashboard)/war-room/waiver/page.tsx`
  - `PlayerRow` component (~line 164): color-split `injury_status` badge
  - `AddPanel` component (~line 392): add `availability_note` warning
  - `RecommendationCard` component (~line 479): add `constraint_warning` strip

- [ ] **Step 1: Color-split `injury_status` badge in `PlayerRow`**

Current code in `PlayerRow` (around line 206–210):
```tsx
          {player.injury_status && (
            <span className="text-[10px] px-1.5 py-0.5 bg-status-lost/10 text-status-lost border border-status-lost/30 rounded font-semibold">
              {player.injury_status}
            </span>
          )}
```

Replace with:
```tsx
          {player.injury_status && (() => {
            const _inj = player.injury_status.toUpperCase()
            const isDtd = _inj === 'DTD' || _inj === 'D2D'
            return (
              <span className={cn(
                'text-[10px] px-1.5 py-0.5 rounded font-semibold border',
                isDtd
                  ? 'bg-status-bubble/10 text-status-bubble border-status-bubble/30'
                  : 'bg-status-lost/10 text-status-lost border-status-lost/30',
              )}>
                {player.injury_status}
              </span>
            )
          })()}
```

- [ ] **Step 2: Add `availability_note` warning in `AddPanel`**

Current `AddPanel` component (around lines 392–413) ends with the statcast signals. After the `fa.z_score` / `fa.starts_this_week` div, add:

```tsx
      {fa.availability_note && (
        <p className="text-[10px] text-status-bubble font-semibold mt-1">⚠ {fa.availability_note}</p>
      )}
```

The updated `AddPanel` return should look like:
```tsx
    <div className="flex-1 min-w-0 space-y-1">
      <p className="text-[10px] font-bold tracking-widest uppercase text-status-safe">ADD</p>
      <p className="text-sm font-semibold text-text-primary truncate">{fa.name}</p>
      <p className="text-[10px] text-text-muted">{fa.position} · {fa.team}</p>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[10px] text-text-secondary">
        {fa.z_score !== undefined && (
          <span>Z <span className="text-text-primary font-mono">{fa.z_score >= 0 ? '+' : ''}{fa.z_score.toFixed(2)}</span></span>
        )}
        {(fa.starts_this_week ?? 0) > 0 && (
          <span className="text-status-safe">{fa.starts_this_week}-start</span>
        )}
        {rec.statcast_signals.map((sig) => (
          <span key={sig} className="text-accent-gold">[{sig}]</span>
        ))}
      </div>
      {fa.availability_note && (
        <p className="text-[10px] text-status-bubble font-semibold mt-1">⚠ {fa.availability_note}</p>
      )}
    </div>
```

- [ ] **Step 3: Add `constraint_warning` strip to `RecommendationCard`**

In `RecommendationCard` (around line 479), the outer `div` starts with `<div className="bg-bg-surface border...">`. The two-panel row starts with `<div className="p-3 flex...">`. Insert the constraint strip **before** the two-panel row:

```tsx
      {/* Constraint warning strip */}
      {rec.constraint_warning && (
        <div className="px-3 pt-3 pb-0">
          <div className="flex items-center gap-1.5 text-[11px] text-status-bubble font-semibold">
            <WarnIcon className="h-3.5 w-3.5 flex-shrink-0" />
            {rec.constraint_warning}
          </div>
        </div>
      )}
      {/* Two-panel row */}
      <div className="p-3 flex flex-col sm:flex-row gap-3">
```

(`WarnIcon` is already imported at the top of the file as `AlertTriangle as WarnIcon`.)

- [ ] **Step 4: Commit**

```
git add "frontend/app/(dashboard)/war-room/waiver/page.tsx"
git commit -m "feat(frontend): availability badge color-split + availability_note in AddPanel + constraint_warning strip"
```

---

## Task 9: Frontend Dashboard IL Crisis Rendering

**Files:**
- Modify: `frontend/app/(dashboard)/dashboard/_components/dashboard-client.tsx`
  - `LineupGapsWidget` (~line 229): add ROSTER emergency rendering

- [ ] **Step 1: Update `LineupGapsWidget` to render ROSTER crisis gap**

In `LineupGapsWidget`, the `regularGaps` list is iterated with:
```tsx
{regularGaps.map((gap, i) => (
  <li key={i} className="flex items-start gap-2">
    <span className={`mt-0.5 h-2 w-2 rounded-full shrink-0 ${severityDotClass(gap.severity)}`} />
    <div className="flex-1 min-w-0">
      <p className="text-text-secondary text-sm font-medium">{gap.position}</p>
      <p className="text-text-tertiary text-xs">{gap.message}</p>
      {gap.suggested_add && (
        <p className="text-accent-gold text-xs mt-0.5">Add: {gap.suggested_add}</p>
      )}
    </div>
  </li>
))}
```

Replace with:
```tsx
{regularGaps.map((gap, i) =>
  gap.position === 'ROSTER' ? (
    <li key={i} className="flex items-start gap-2 p-2 bg-status-lost/5 border border-status-lost/20 rounded">
      <AlertCircle className="mt-0.5 h-4 w-4 text-status-lost shrink-0" />
      <div className="flex-1 min-w-0">
        <p className="text-status-lost text-sm font-bold">{gap.message}</p>
        {gap.action_url && (
          <Link
            href={gap.action_url}
            className="inline-flex items-center gap-1 text-xs text-accent-gold mt-1 hover:underline"
          >
            Go to Roster <ArrowRight className="h-3 w-3" />
          </Link>
        )}
      </div>
    </li>
  ) : (
    <li key={i} className="flex items-start gap-2">
      <span className={`mt-0.5 h-2 w-2 rounded-full shrink-0 ${severityDotClass(gap.severity)}`} />
      <div className="flex-1 min-w-0">
        <p className="text-text-secondary text-sm font-medium">{gap.position}</p>
        <p className="text-text-tertiary text-xs">{gap.message}</p>
        {gap.suggested_add && (
          <p className="text-accent-gold text-xs mt-0.5">Add: {gap.suggested_add}</p>
        )}
      </div>
    </li>
  )
)}
```

(`AlertCircle` is already imported at line 14. `Link` is already imported. `ArrowRight` is already imported.)

- [ ] **Step 2: Commit**

```
git add "frontend/app/(dashboard)/dashboard/_components/dashboard-client.tsx"
git commit -m "feat(frontend): ROSTER EMERGENCY crisis gap rendering in LineupGapsWidget"
```

---

## Task 10: Week Context Badges (War Room + Weekly Preview)

**Files:**
- Modify: `frontend/app/(dashboard)/war-room/page.tsx` (~line 176)
- Modify: `frontend/app/(dashboard)/war-room/preview/page.tsx` (~line 183)

- [ ] **Step 1: Add IN-FLIGHT badge to War Room header**

In `frontend/app/(dashboard)/war-room/page.tsx`, find the page header row (around line 175):

```tsx
        <div className="flex items-center gap-3 mb-2">
          <Swords className="h-6 w-6 text-accent-gold" />
          <span className="text-lg font-bold tracking-widest uppercase text-accent-gold">War Room</span>
```

Insert after the `War Room` span:

```tsx
          {matchup.data && matchup.data.week > 0 && (
            <span className="text-[10px] px-2 py-1 bg-accent-gold/10 text-accent-gold border border-accent-gold/30 rounded font-bold tracking-wider uppercase">
              Week {matchup.data.week} · IN-FLIGHT
            </span>
          )}
```

- [ ] **Step 2: Add PREVIEW badge to Weekly Preview header**

In `frontend/app/(dashboard)/war-room/preview/page.tsx`, find the header (around line 183):

```tsx
        <div className="flex items-center gap-3 mb-6">
          <Eye className="h-6 w-6 text-accent-gold" />
          <span className="text-lg font-bold tracking-widest uppercase text-accent-gold">Weekly Preview</span>
          {data.week_number > 0 && (
            <span className="text-xs font-semibold tracking-widest text-text-muted uppercase">
              Week {data.week_number}
            </span>
          )}
```

Replace the existing `Week {data.week_number}` span with:

```tsx
          {data.week_number > 0 && (
            <span className="text-[10px] px-2 py-1 bg-accent-primary/10 text-accent-primary border border-accent-primary/30 rounded font-bold tracking-wider uppercase">
              Week {data.week_number} · PREVIEW
            </span>
          )}
```

(`accent-primary` is `#2563eb` — defined in `tailwind.config.*`. `accent-blue` does not exist in this project.)

- [ ] **Step 3: Commit**

```
git add "frontend/app/(dashboard)/war-room/page.tsx" "frontend/app/(dashboard)/war-room/preview/page.tsx"
git commit -m "feat(frontend): add Week N·IN-FLIGHT badge to War Room + Week N·PREVIEW badge to Weekly Preview"
```

---

## Final Verification

- [ ] **Run full test suite**

```
venv/Scripts/python -m pytest tests/ -q --tb=short
```

Expected: no new failures beyond pre-existing baseline (4 known pre-existing failures in `test_row_projector.py`).

- [ ] **Run all syntax checks**

```
venv/Scripts/python -m py_compile backend/models.py
venv/Scripts/python -m py_compile backend/schemas.py
venv/Scripts/python -m py_compile backend/routers/fantasy.py
venv/Scripts/python -m py_compile backend/routers/admin.py
venv/Scripts/python -m py_compile backend/services/dashboard_service.py
```

All must exit 0.

- [ ] **Targeted test suites**

```
venv/Scripts/python -m pytest tests/test_availability_guard.py tests/test_dashboard_il_crisis.py -v
```

Expected: all tests PASS.
