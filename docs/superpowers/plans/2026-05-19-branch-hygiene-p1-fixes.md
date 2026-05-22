# Branch Hygiene + P1 Code Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve local `stable/cbb-prod` branch divergence (13 commits ahead of origin) and fix 5 remaining P1 UI bugs from the K-NEXT-4 UAT audit.

**Architecture:** Phase 1 is pure git — no file edits. Phase 2 fixes 5 independent code bugs across backend (Python) and frontend (TypeScript). Each Phase 2 task is self-contained and can be done in any order after Phase 1.

**Tech Stack:** Python 3.11 / FastAPI / Pydantic v2 / Next.js / TypeScript / pytest / Windows dev (`venv/Scripts/python`)

---

## Background: Branch State

```
main (local + origin)        → 4a6a3c0  (production)
origin/stable/cbb-prod       → 4a6a3c0  (same as main)
stable/cbb-prod (local only) → 004972e  [ahead 13, behind 4]
```

The 13 local-only commits on `stable/cbb-prod` include valuable UI redesign work (Yahoo-style roster view, Design System v3, light theme) and backend fixes (resilient roster count, SB override) that were **never pushed or deployed**. The 4 commits on `main` (including acquisitions dict-walk fix) are NOT on local `stable/cbb-prod`.

---

## Phase 1: Branch Hygiene

---

### Task 1: Audit the 13 local-only commits

**Files:** Git history only — no file edits.

- [ ] **Step 1: List the 13 divergent commits with file stats**

```bash
git log stable/cbb-prod --not main --oneline --stat
```

Expected output: 13 commits with file change counts. Key commits to note:
- `ee05542` — Fix 6 failed pipeline jobs (2026-05-18) → `fangraphs_loader.py`, `savant_ingestion.py`, `daily_ingestion.py`
- `7da68b4` / `0c1d1f4` — SB override + utcnow fix → `projection_assembly_service.py`, `fantasy.py`
- `8e331c4` — resilient roster count → `yahoo_client_resilient.py`
- `6807d4a` — Design System v3 + P0 scoreboard fix → many frontend + test files
- `d24243c` — light theme conversion → frontend files
- `aea55bd`, `948cefd` — Yahoo-style roster + waiver swap bar → frontend files
- `873c709` — test fixes post-handedness migration → `tests/conftest.py`, test files
- `f8341cc` — JSX closing tag fix → `roster/page.tsx`
- `004972e` — flake8 F-violations → `.claude/worktrees/` lint files (deletable stubs)

- [ ] **Step 2: Check what `92beb1c` ("port stable/cbb-prod fixes") already brought to main**

```bash
git show 92beb1c --stat --oneline
```

Look at which files were touched. If `projection_assembly_service.py` and `yahoo_client_resilient.py` appear here, their fixes are already on main.

- [ ] **Step 3: Check for actual content differences in key backend files**

```bash
git diff main stable/cbb-prod -- backend/fantasy_baseball/yahoo_client_resilient.py | head -60
git diff main stable/cbb-prod -- backend/fantasy_baseball/projection_assembly_service.py | head -40
git diff main stable/cbb-prod -- backend/routers/fantasy.py | head -60
```

If the diffs are empty (already ported), those backend files don't need cherry-picking. If diffs exist, note which file/function changed.

- [ ] **Step 4: Commit your findings as a note (optional but recommended)**

Write a brief note on what's valuable vs already-on-main. No file changes needed.

---

### Task 2: Resolve branch divergence

**Files:** Git operations only — no file edits.

The recommended strategy is: **merge `stable/cbb-prod` → `main`** (brings frontend UI changes to the deployable branch), then reset local `stable/cbb-prod` to track `origin`.

- [ ] **Step 1: Create a safety backup branch**

```bash
git branch backup/stable-cbb-prod-20260519 stable/cbb-prod
```

This preserves the 13 commits before any destructive operation.

- [ ] **Step 2: From main, merge local stable/cbb-prod**

```bash
git checkout main
git merge stable/cbb-prod --no-ff -m "merge: bring stable/cbb-prod UI + backend fixes into main"
```

Expected: merge succeeds or shows conflicts. If conflicts appear, see Step 3. If no conflicts, skip to Step 4.

- [ ] **Step 3 (if conflicts): Resolve conflicts**

For each conflicted file, keep the most recent correct version. General rules:
- `backend/routers/fantasy.py`: Keep the `main` version of acquisitions dict-walk fix (`4a6a3c0`); keep `stable/cbb-prod` version for SB override and roster count fixes.
- Frontend files: Keep `stable/cbb-prod` (newer UI) unless main has newer fixes.
- Test files: Accept both changes — don't discard passing tests.

After resolving:
```bash
git add .
git commit -m "merge: resolve stable/cbb-prod conflicts — prefer newer UI, keep main bug fixes"
```

- [ ] **Step 4: Reset local stable/cbb-prod to track origin**

```bash
git branch -f stable/cbb-prod origin/stable/cbb-prod
```

Now `stable/cbb-prod` local = `origin/stable/cbb-prod` = `main` (before the merge above). After pushing main, all three will align.

- [ ] **Step 5: Verify branch state**

```bash
git log --oneline -5 main
git branch -vv | grep stable
```

Expected: `stable/cbb-prod` shows `[origin/stable/cbb-prod]` with no ahead/behind divergence.

---

### Task 3: Validate test suite after merge

**Files:** No changes — run tests only.

- [ ] **Step 1: Syntax-check modified Python files**

```bash
venv/Scripts/python -m py_compile backend/routers/fantasy.py
venv/Scripts/python -m py_compile backend/fantasy_baseball/yahoo_client_resilient.py
venv/Scripts/python -m py_compile backend/fantasy_baseball/projection_assembly_service.py
venv/Scripts/python -m py_compile backend/services/daily_ingestion.py
```

Each must exit with no output (= pass).

- [ ] **Step 2: Run full test suite**

```bash
venv/Scripts/python -m pytest tests/ -q --tb=short 2>&1 | tail -30
```

Expected: All tests pass (baseline was `92beb1c` "clean suite on main"). If failures appear, fix them before proceeding to Phase 2.

- [ ] **Step 3: Commit test results as phase checkpoint**

If all tests pass:
```bash
git add -A
git commit -m "chore: phase 1 complete — branch divergence resolved, tests passing"
```

---

## Phase 2: P1 Code Fixes

---

### Task 4: Fix waiver target Need score: 0.00 on dashboard

**Root cause:** `dashboard_service._get_waiver_targets` builds `CategoryDeficitOut` objects using Yahoo stat IDs (e.g., `"55"` for OPS, `"57"` for K/9) as the `category` field. But `compute_need_score` translates via `_CANONICAL_TO_BOARD` which expects canonical codes like `"OPS"`, `"K_9"`. Yahoo stat ID `"55"` doesn't match `"OPS"` → `needs_dict` is empty → falls back to `player_z_score` which is often 0.0.

**Files:**
- Modify: `backend/services/dashboard_service.py:508-516` (the `CategoryDeficitOut` building loop)
- Test: `tests/test_dashboard_service.py` (create if missing)

- [ ] **Step 1: Write the failing test**

Create `tests/test_dashboard_service.py` if it doesn't exist. Add:

```python
# tests/test_dashboard_service.py
import pytest
from unittest.mock import patch, MagicMock
from backend.services.dashboard_service import DashboardService


def test_waiver_target_need_score_uses_canonical_category_codes():
    """Need score must not be 0.0 when category_deficits are populated with Yahoo stat IDs."""
    # Verify that compute_need_score receives canonical codes, not Yahoo stat IDs
    from backend.fantasy_baseball.category_aware_scorer import compute_need_score
    from backend.schemas import CategoryDeficitOut

    # Simulate what dashboard_service builds with Yahoo stat IDs (the bug)
    bad_deficits = [
        CategoryDeficitOut(category="55", deficit=0.05, winning=False),  # Yahoo OPS stat ID
        CategoryDeficitOut(category="57", deficit=2.0, winning=False),   # Yahoo K/9 stat ID
    ]
    cat_scores = {"ops": 1.5, "k9": 2.0}  # board-key cat scores
    score_with_bad_deficits = compute_need_score(cat_scores, 1.0, bad_deficits, 10)

    # With canonical codes (the fix)
    good_deficits = [
        CategoryDeficitOut(category="OPS", deficit=0.05, winning=False),
        CategoryDeficitOut(category="K_9", deficit=2.0, winning=False),
    ]
    score_with_good_deficits = compute_need_score(cat_scores, 1.0, good_deficits, 10)

    # Bug: Yahoo stat ID keys produce 0 category contribution (falls back to z_score=1.0)
    # Fix: canonical codes produce a higher score reflecting category need
    assert score_with_bad_deficits == pytest.approx(1.0, abs=0.5), "bad path should fall back to z_score"
    assert score_with_good_deficits > score_with_bad_deficits, "canonical codes must produce higher need score"
```

- [ ] **Step 2: Run the test to verify it fails (or passes, confirming buggy behavior)**

```bash
venv/Scripts/python -m pytest tests/test_dashboard_service.py::test_waiver_target_need_score_uses_canonical_category_codes -v
```

Expected: Test reveals the bug (bad_deficits score ≈ good_deficits score, both 1.0, since Yahoo IDs don't match).

- [ ] **Step 3: Read the current buggy code**

Open `backend/services/dashboard_service.py` around line 483–516. The problematic loop:

```python
for sid, my_val in my_stats.items():
    opp_val = opp_stats.get(sid, 0)
    try:
        deficit = float(opp_val or 0) - float(my_val or 0)
        category_deficits.append(CategoryDeficitOut(
            category=sid, deficit=deficit, winning=deficit <= 0
        ))
    except (TypeError, ValueError):
        pass
```

Here `sid` is a Yahoo stat ID like `"55"`, `"57"`. `CategoryDeficitOut.category` receives `"55"` instead of `"OPS"`.

- [ ] **Step 4: Fix — translate Yahoo stat IDs to canonical codes before building CategoryDeficitOut**

At the top of `_get_waiver_targets` (before the `for sid, my_val` loop), add the import and translation. The full replacement of the inner loop in `dashboard_service.py`:

```python
        from backend.stat_contract import CONTRACT as _CONTRACT
        _yahoo_index = _CONTRACT.yahoo_id_index  # {stat_id_str: canonical_code}

        for sid, my_val in my_stats.items():
            opp_val = opp_stats.get(sid, 0)
            try:
                deficit = float(opp_val or 0) - float(my_val or 0)
                # Translate Yahoo stat ID → canonical code so compute_need_score
                # can match against cat_scores board keys.
                canon = _yahoo_index.get(str(sid), sid)
                category_deficits.append(CategoryDeficitOut(
                    category=canon, deficit=deficit, winning=deficit <= 0
                ))
            except (TypeError, ValueError):
                pass
```

Location: `backend/services/dashboard_service.py`, replace the loop body starting at the `deficit = float(opp_val or 0) - float(my_val or 0)` line.

- [ ] **Step 5: Run the test to verify it passes**

```bash
venv/Scripts/python -m pytest tests/test_dashboard_service.py::test_waiver_target_need_score_uses_canonical_category_codes -v
```

Expected: PASS

- [ ] **Step 6: Syntax check**

```bash
venv/Scripts/python -m py_compile backend/services/dashboard_service.py
```

Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add backend/services/dashboard_service.py tests/test_dashboard_service.py
git commit -m "fix(dashboard): translate Yahoo stat IDs to canonical codes for need_score computation"
```

---

### Task 5: Fix injury status returning boolean `true` instead of string `"IL"`

**Root cause:** `CanonicalPlayerRow` in `contracts.py` declares `injury_status: Optional[str]` with no validator. Yahoo API sometimes returns `injury_status=True` (bool). Python passes it as-is; FastAPI serializes to JSON `true` (bool). The fix is a `@field_validator` on `CanonicalPlayerRow`, matching the pattern already used on `RosterPlayerOut` in `schemas.py`.

Also: `player_mapper.py:238` uses `yahoo_player.get("injury_note") or yahoo_player.get("injury_status")` — if `injury_note` is `True` (truthy bool), it assigns the boolean directly.

**Files:**
- Modify: `backend/contracts.py:342-373` (add validator to `CanonicalPlayerRow`)
- Modify: `backend/services/player_mapper.py:237-238` (coerce in mapper)
- Test: `tests/test_contracts.py` or `tests/test_player_mapper.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contracts.py (add to existing or create)
from backend.contracts import CanonicalPlayerRow, FreshnessMetadata
from datetime import datetime
from zoneinfo import ZoneInfo


def _make_minimal_canonical_row(**overrides) -> dict:
    base = dict(
        player_name="Garrett Crochet",
        team="CWS",
        eligible_positions=["SP"],
        status="IL",
        freshness=FreshnessMetadata(
            primary_source="yahoo",
            fetched_at=None,
            computed_at=datetime.now(ZoneInfo("America/New_York")),
            staleness_threshold_minutes=60,
            is_stale=False,
        ),
    )
    base.update(overrides)
    return base


def test_canonical_row_coerces_boolean_injury_status_true():
    row = CanonicalPlayerRow(**_make_minimal_canonical_row(injury_status=True))
    assert isinstance(row.injury_status, str), "injury_status must be a string, not bool"
    assert row.injury_status != "True", "should not stringify boolean as 'True'"


def test_canonical_row_coerces_boolean_injury_status_false():
    row = CanonicalPlayerRow(**_make_minimal_canonical_row(injury_status=False))
    assert isinstance(row.injury_status, str)


def test_canonical_row_passes_string_injury_status():
    row = CanonicalPlayerRow(**_make_minimal_canonical_row(injury_status="IL"))
    assert row.injury_status == "IL"


def test_canonical_row_allows_none_injury_status():
    row = CanonicalPlayerRow(**_make_minimal_canonical_row(injury_status=None))
    assert row.injury_status is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
venv/Scripts/python -m pytest tests/test_contracts.py::test_canonical_row_coerces_boolean_injury_status_true -v
```

Expected: FAIL — `CanonicalPlayerRow` has no validator, `injury_status` stays as Python `True`.

- [ ] **Step 3: Add validator to `CanonicalPlayerRow` in `contracts.py`**

The class currently ends at line ~372 with:
```python
    class Config:
        frozen = True
```

Add the validator BEFORE `class Config`. The `CanonicalPlayerRow` class uses `frozen=True` but does not use `model_config = ConfigDict(...)`. Pydantic v2 validators work with `@field_validator`.

Insert after `mlbam_id: Optional[int] = None` (line ~369):

```python
    @field_validator("injury_status", "injury_return_timeline", mode="before")
    @classmethod
    def coerce_injury_fields_to_string(cls, v):
        """Yahoo API sometimes returns boolean injury flags; coerce to string."""
        if isinstance(v, bool):
            return "IL" if v else "Active"
        return v
```

Also add the import at the top of `contracts.py` if not already present:
```python
from pydantic import BaseModel, field_validator
```

Check the existing import line in `contracts.py` and add `field_validator` to it.

- [ ] **Step 4: Fix the mapper to avoid passing boolean injury_note**

In `backend/services/player_mapper.py`, line ~238, the current code:
```python
injury_status = yahoo_player.get("injury_note") or yahoo_player.get("injury_status")
```

This assigns boolean `True` if `injury_note=True`. Replace with:

```python
_raw_note = yahoo_player.get("injury_note")
_raw_status = yahoo_player.get("injury_status")
# Coerce booleans: True → "IL", False → None (not an injury note)
if isinstance(_raw_note, bool):
    _raw_note = "IL" if _raw_note else None
if isinstance(_raw_status, bool):
    _raw_status = "IL" if _raw_status else None
injury_status = _raw_note or _raw_status
```

- [ ] **Step 5: Run all injury tests**

```bash
venv/Scripts/python -m pytest tests/test_contracts.py -k "injury" -v
```

Expected: All PASS.

- [ ] **Step 6: Syntax check both files**

```bash
venv/Scripts/python -m py_compile backend/contracts.py
venv/Scripts/python -m py_compile backend/services/player_mapper.py
```

- [ ] **Step 7: Commit**

```bash
git add backend/contracts.py backend/services/player_mapper.py tests/test_contracts.py
git commit -m "fix(contracts): coerce boolean injury_status to string in CanonicalPlayerRow + player_mapper"
```

---

### Task 6: Fix roster "Move player" buttons (universally disabled)

**Root cause:** Two issues:
1. The move button `disabled={!selectedSlot || isMoving}` is grayed out by default because `selectedSlot = ''` (no slot pre-selected). Users see all buttons as gray/disabled before interacting.
2. `handleMoveClick` contains `if (!selectedSlot || !player.yahoo_player_key) return` — if `yahoo_player_key` is empty string `""` (falsy), clicks silently no-op even after selecting a slot.

**Fix:**
1. Pre-populate `selectedSlot` with the first valid move target (first option that isn't the player's current slot).
2. Filter `current_slot` out of `moveOptions` so there's always a meaningful choice.
3. Add a visible error state when `yahoo_player_key` is missing instead of silently no-oping.

**Files:**
- Modify: `frontend/app/(dashboard)/war-room/roster/page.tsx:670-758`

- [ ] **Step 1: Read the PlayerRow component**

Read lines 655–760 of `frontend/app/(dashboard)/war-room/roster/page.tsx` to understand the current `moveOptions` and `handleMoveClick` logic.

Key current code (around line 670–685):
```tsx
const eligible = player.eligible_positions ?? []
const isPitcher = eligible.some((p) => ['SP', 'RP', 'P'].includes(p))
const displayCats = isPitcher ? PITCHER_DISPLAY : BATTER_DISPLAY
const statValues = getStatWindow(player, viewMode)

// Move dropdown: eligible positions + universal slots, deduplicated
const moveOptions = Array.from(new Set([...eligible, ...UNIVERSAL_SLOTS]))

const handleMoveClick = () => {
  if (!selectedSlot || !player.yahoo_player_key) return
  onMove(player.yahoo_player_key, selectedSlot)
  setSelectedSlot('')
}
```

- [ ] **Step 2: Update `moveOptions` to exclude current slot, and auto-select first valid target**

Replace the `moveOptions` computation and `useState` initialization:

```tsx
// Before (current):
const [selectedSlot, setSelectedSlot] = useState('')
// ...
const moveOptions = Array.from(new Set([...eligible, ...UNIVERSAL_SLOTS]))
```

```tsx
// After (fixed):
const currentSlot = player.current_slot?.toUpperCase() ?? ''

// Exclude the player's current slot so every option represents a real move.
const moveOptions = Array.from(
  new Set([...eligible, ...UNIVERSAL_SLOTS])
).filter((s) => s.toUpperCase() !== currentSlot)

// Auto-select first available target so button is enabled by default.
const [selectedSlot, setSelectedSlot] = useState(() => moveOptions[0] ?? '')
```

**Note:** `useState` with an initializer function runs once on mount. This pre-selects the first valid target. The `filter` ensures we never pre-select the current slot.

- [ ] **Step 3: Add error feedback when yahoo_player_key is missing**

Replace `handleMoveClick`:

```tsx
// Before:
const handleMoveClick = () => {
  if (!selectedSlot || !player.yahoo_player_key) return
  onMove(player.yahoo_player_key, selectedSlot)
  setSelectedSlot('')
}

// After:
const handleMoveClick = () => {
  if (!selectedSlot) return
  if (!player.yahoo_player_key) {
    console.warn('[Roster] Move skipped: yahoo_player_key missing for', player.player_name)
    return
  }
  onMove(player.yahoo_player_key, selectedSlot)
  setSelectedSlot(moveOptions[0] ?? '')
}
```

The `setSelectedSlot` after success resets to the first option (not `''`) so the button remains enabled for another move.

- [ ] **Step 4: TypeScript build check**

```bash
cd frontend && npm run build 2>&1 | tail -20
```

Expected: no TypeScript errors. If there are type errors from `moveOptions[0]`, cast: `moveOptions[0] ?? '' as string`.

- [ ] **Step 5: Commit**

```bash
git add frontend/app/(dashboard)/war-room/roster/page.tsx
git commit -m "fix(roster): pre-select move target and filter current slot from dropdown"
```

---

### Task 7: Fix team totals showing "–" for OPS and K/9

**Root cause:** `_map_rolling_to_category_stats` in `player_mapper.py` maps `w_ops → OPS` and `w_k_per_9 → K_9`. These fields are `nullable=True` in `PlayerRollingStats` — set to `None` when there are insufficient AB or IP in the rolling window. When `None`, the frontend `opsValues.filter(v != null && v > 0)` produces empty arrays → "–".

Additionally, `w_obp` and `w_slg` are computed before `w_ops` and may be non-null even when `w_ops` isn't stored (edge case in some DB rows). Similarly, `w_strikeouts_pit` and `w_ip` are often non-null even when `w_k_per_9` isn't stored.

**Fix:** In `_map_rolling_to_category_stats`, if `OPS` is still `None` after the main mapping loop, compute it from `w_obp + w_slg` as a fallback. Same for `K_9`: compute from `9 * w_strikeouts_pit / w_ip`.

**Files:**
- Modify: `backend/services/player_mapper.py:52-73` (`_map_rolling_to_category_stats`)
- Test: `tests/test_player_mapper.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_player_mapper.py (add to existing or create)
from backend.services.player_mapper import _map_rolling_to_category_stats
from unittest.mock import MagicMock


def _make_rolling_stats(**fields):
    """Create a mock PlayerRollingStats with only the given fields set."""
    mock = MagicMock()
    # Default all fields to None
    for attr in ['w_runs', 'w_hits', 'w_home_runs', 'w_rbi', 'w_strikeouts_bat',
                 'w_tb', 'w_avg', 'w_ops', 'w_net_stolen_bases',
                 'w_strikeouts_pit', 'w_era', 'w_whip', 'w_k_per_9', 'w_qs',
                 'w_obp', 'w_slg', 'w_ip']:
        setattr(mock, attr, None)
    for k, v in fields.items():
        setattr(mock, k, v)
    return mock


def test_ops_fallback_from_obp_plus_slg():
    """If w_ops is None but w_obp and w_slg are available, compute OPS."""
    stats = _make_rolling_stats(w_obp=0.350, w_slg=0.480)  # w_ops is None
    result = _map_rolling_to_category_stats(stats)
    assert result is not None
    assert result.values.get('OPS') == pytest.approx(0.830), "OPS should be w_obp + w_slg"


def test_k9_fallback_from_strikeouts_and_ip():
    """If w_k_per_9 is None but w_strikeouts_pit and w_ip are available, compute K/9."""
    stats = _make_rolling_stats(w_strikeouts_pit=27.0, w_ip=30.0)  # w_k_per_9 is None
    result = _map_rolling_to_category_stats(stats)
    assert result is not None
    assert result.values.get('K_9') == pytest.approx(8.1), "K_9 should be 9 * 27 / 30"


def test_ops_uses_stored_value_if_available():
    """If w_ops is stored, use it directly without recomputation."""
    stats = _make_rolling_stats(w_ops=0.900, w_obp=0.400, w_slg=0.500)
    result = _map_rolling_to_category_stats(stats)
    assert result.values.get('OPS') == pytest.approx(0.900)


import pytest
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
venv/Scripts/python -m pytest tests/test_player_mapper.py::test_ops_fallback_from_obp_plus_slg tests/test_player_mapper.py::test_k9_fallback_from_strikeouts_and_ip -v
```

Expected: FAIL — currently no fallback computation exists.

- [ ] **Step 3: Add fallback computation in `_map_rolling_to_category_stats`**

In `backend/services/player_mapper.py`, after line ~71 (after the two mapping loops), add:

```python
    # Fallback: compute OPS from component parts if w_ops was not stored
    if values.get('OPS') is None:
        _w_obp = getattr(rolling_stats, 'w_obp', None)
        _w_slg = getattr(rolling_stats, 'w_slg', None)
        if _w_obp is not None and _w_slg is not None:
            values['OPS'] = round(_w_obp + _w_slg, 4)

    # Fallback: compute K/9 from raw counts if w_k_per_9 was not stored
    if values.get('K_9') is None:
        _w_k = getattr(rolling_stats, 'w_strikeouts_pit', None)
        _w_ip = getattr(rolling_stats, 'w_ip', None)
        if _w_k is not None and _w_ip is not None and _w_ip > 0:
            values['K_9'] = round(9.0 * _w_k / _w_ip, 2)

    return CategoryStats(values=values)
```

The `return CategoryStats(values=values)` at line ~73 stays at the end — move it after the two new blocks.

- [ ] **Step 4: Run all player_mapper tests**

```bash
venv/Scripts/python -m pytest tests/test_player_mapper.py -v
```

Expected: All PASS.

- [ ] **Step 5: Syntax check**

```bash
venv/Scripts/python -m py_compile backend/services/player_mapper.py
```

- [ ] **Step 6: Commit**

```bash
git add backend/services/player_mapper.py tests/test_player_mapper.py
git commit -m "fix(mapper): fallback OPS from OBP+SLG and K/9 from K+IP when w_ops/w_k_per_9 null"
```

---

### Task 8: Expand budget page beyond 3-line placeholder

**Current state:** `/war-room/budget` page shows only: acquisitions bar, IL slots, IP pace. This is a valid layout but feels sparse for a dedicated "budget" page.

**Root cause:** The `/api/fantasy/budget` endpoint only returns 9 fields. The `BudgetPanel` component only renders those 9. No transaction history, no season context, no acquisition pace projection.

**Fix:**
1. Backend: Add `weeks_remaining_in_season`, `acquisitions_this_season` (from Yahoo transaction count), and `week_label` to the budget response.
2. Frontend: Add a "Season Acquisition Pace" card below `BudgetPanel` using the new fields.

**Files:**
- Modify: `backend/routers/fantasy.py:5902-6016` (`get_constraint_budget` endpoint)
- Modify: `frontend/lib/types.ts:579-601` (`BudgetData` interface)
- Modify: `frontend/app/(dashboard)/war-room/budget/page.tsx:48-91` (add new panel)

- [ ] **Step 1: Add new fields to the budget endpoint**

In `backend/routers/fantasy.py`, inside `get_constraint_budget` after the existing data collection (around line 5994), add:

```python
    # 4. Season acquisition context
    from datetime import date as _date
    _MLB_OPENING_DATE_2026 = _date(2026, 3, 20)
    _SEASON_END_DATE_2026 = _date(2026, 9, 28)
    today = now_et.date()
    total_season_weeks = max(1, (_SEASON_END_DATE_2026 - _MLB_OPENING_DATE_2026).days // 7)
    weeks_elapsed = max(0, (today - _MLB_OPENING_DATE_2026).days // 7)
    weeks_remaining = max(0, total_season_weeks - weeks_elapsed)
    week_label = f"Week {weeks_elapsed + 1} of {total_season_weeks}"

    # Count season-wide acquisitions (all add transactions, not just this week)
    acquisitions_this_season = 0
    try:
        all_transactions = client.get_transactions(t_type="add")
        season_start = datetime(_MLB_OPENING_DATE_2026.year, _MLB_OPENING_DATE_2026.month,
                                _MLB_OPENING_DATE_2026.day, tzinfo=ZoneInfo("America/New_York"))
        acquisitions_this_season = count_weekly_acquisitions(
            all_transactions, team_key, season_start, now_et
        )
    except Exception:
        acquisitions_this_season = acquisitions_used  # best-effort fallback
```

Then add these fields to the returned dict:

```python
    return {
        "budget": {
            # ... existing fields ...
            "weeks_remaining": weeks_remaining,
            "week_label": week_label,
            "acquisitions_this_season": acquisitions_this_season,
        },
        "freshness": { ... }
    }
```

- [ ] **Step 2: Update TypeScript `BudgetData` interface**

In `frontend/lib/types.ts`, add to `BudgetData`:

```typescript
export interface BudgetData {
  // ... existing fields ...
  weeks_remaining: number
  week_label: string
  acquisitions_this_season: number
}
```

- [ ] **Step 3: Add "Season Pace" section to budget page**

In `frontend/app/(dashboard)/war-room/budget/page.tsx`, after `<BudgetPanel budget={budget} />` (line 77), add:

```tsx
      {/* Season Acquisition Pace */}
      <div className="bg-bg-surface border border-border-subtle rounded-lg px-4 py-3 space-y-2">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-bold tracking-widest uppercase text-accent-gold">
            Season Pace
          </span>
          <span className="text-[10px] text-text-muted ml-auto">{budget.week_label}</span>
        </div>
        <div className="flex items-center justify-between text-xs">
          <span className="text-text-secondary">Acquisitions this season</span>
          <span className="text-text-primary font-mono tabular-nums font-semibold">
            {budget.acquisitions_this_season}
          </span>
        </div>
        <div className="flex items-center justify-between text-xs">
          <span className="text-text-secondary">Weeks remaining</span>
          <span className="text-text-primary font-mono tabular-nums font-semibold">
            {budget.weeks_remaining}
          </span>
        </div>
        {budget.weeks_remaining > 0 && (
          <div className="flex items-center justify-between text-xs">
            <span className="text-text-secondary">Avg adds/week (season)</span>
            <span className="text-text-primary font-mono tabular-nums font-semibold">
              {(budget.acquisitions_this_season / Math.max(1, 25 - budget.weeks_remaining)).toFixed(1)}
            </span>
          </div>
        )}
      </div>
```

- [ ] **Step 4: Syntax-check backend and build frontend**

```bash
venv/Scripts/python -m py_compile backend/routers/fantasy.py
cd frontend && npm run build 2>&1 | tail -20
```

Both must complete without errors.

- [ ] **Step 5: Commit**

```bash
git add backend/routers/fantasy.py frontend/lib/types.ts frontend/app/(dashboard)/war-room/budget/page.tsx
git commit -m "feat(budget): add season acquisition pace panel with week label and season totals"
```

---

### Task 9: Full test suite pass + HANDOFF.md update

**Files:**
- Modify: `HANDOFF.md`
- Run: full pytest suite

- [ ] **Step 1: Run full test suite**

```bash
venv/Scripts/python -m pytest tests/ -q --tb=short 2>&1 | tail -30
```

Expected: All tests pass. If any fail, fix them before continuing.

- [ ] **Step 2: Syntax-check all modified Python files**

```bash
venv/Scripts/python -m py_compile backend/services/dashboard_service.py
venv/Scripts/python -m py_compile backend/contracts.py
venv/Scripts/python -m py_compile backend/services/player_mapper.py
venv/Scripts/python -m py_compile backend/routers/fantasy.py
```

All must exit silently.

- [ ] **Step 3: Update HANDOFF.md**

Update the mission status section. Mark the following as complete:
- Branch divergence resolved
- P1 #2 (waiver need_score 0.00) — FIXED
- P1 #5 (injury status boolean) — FIXED
- P1 #3 (Move buttons disabled) — FIXED
- P1 #3b (team totals OPS/K_9) — FIXED
- P1 #4 (budget page sparse) — FIXED

Mark as **still needing deploy:**
- All 5 fixes above require a Railway deploy (route to Gemini CLI)

Mark as **not yet done:**
- P1 #1 (Ownership 0%) — already fixed in code commits `425f9d6`, `27304f8` — needs deploy only
- Phase 3: Design System v2 (Kimi K-NEXT-5 spec) — queue after P1 deploy

- [ ] **Step 4: Final commit**

```bash
git add HANDOFF.md
git commit -m "docs(handoff): mark Phase 1+2 P1 fixes complete, route deploy to Gemini CLI"
```

---

## Self-Review: Spec Coverage Check

| Delegation Item | Task | Status |
|----------------|------|--------|
| Branch divergence: inspect 13 commits | Task 1 | ✅ |
| Branch divergence: resolve strategy | Task 2 | ✅ |
| Validate test suite after merge | Task 3 | ✅ |
| P1 #1: Ownership 0% (deploy-only) | Not in plan | ℹ️ Deploy-only, no code change needed |
| P1 #2: Dashboard waiver need_score 0.00 | Task 4 | ✅ |
| P1 #3: Roster Move buttons disabled | Task 6 | ✅ |
| P1 #4: Team totals OPS/K_9 = "–" | Task 7 | ✅ |
| P1 #5: Budget page sparse | Task 8 | ✅ |
| P1 #6: Injury status boolean true | Task 5 | ✅ |
| HANDOFF.md update | Task 9 | ✅ |
| Do NOT deploy to Railway | Not in plan | ✅ (Gemini CLI's job) |
| Do NOT modify betting_model.py | Not in plan | ✅ |

## Validation Checklist (from Delegation Bundle)

Before declaring complete:
- [ ] `pytest tests/` — all tests pass
- [ ] Dashboard waiver targets show non-zero need scores
- [ ] Roster "Move" buttons have a pre-selected slot (not universally grayed)
- [ ] Team totals include OPS and K/9 for players with sufficient rolling data
- [ ] Budget page shows season pace panel (not just 3 lines)
- [ ] Injury status returns string `"IL"` / `"Active"` (not boolean)
- [ ] `py_compile` clean on all modified `.py` files
- [ ] `npm run build` clean in `frontend/` (TSX modified)

## Escalation After This Plan

- **Railway deploy** → Route to Gemini CLI with deploy bundle once `pytest` is green
- **Design System v2** → Kimi K-NEXT-5 spec already in `docs/DESIGN_SYSTEM_V2.md`
- **P1 #1 Ownership 0% deploy** → Route to Gemini CLI with commits `425f9d6`, `27304f8`
