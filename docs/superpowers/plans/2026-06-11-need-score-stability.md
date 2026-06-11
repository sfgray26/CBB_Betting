# Need Score Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the 57% need_score swing on the waiver wire (Caballero pattern) by fixing the stale projection cache, normalising `n_cats` across both waiver endpoints, and surfacing a confidence interval + volatility flag so future instability is immediately visible.

**Architecture:** Root Cause A — `reset_board_cache()` never cleared `_projection_cache`, so post-ingestion projection updates were invisible until pod restart. Root Cause B — the two waiver endpoints used different `n_cats` denominators (total category count vs translated-dict length), producing different scores for the same player. Monitoring layer — per-player score history in a module-level dict detects >20% swings within a session and sets a `need_score_volatile` flag on the API response; a `need_score_ci` field conveys projection confidence based on fusion source.

**Tech Stack:** Python/FastAPI backend (`player_board.py`, `schemas.py`, `fantasy.py`), Next.js/React frontend (`types.ts`, `waiver/page.tsx`), pytest.

**Test runner:** `venv/Scripts/python -m pytest tests/<file>.py::<test> -v`  
**Syntax check:** `venv/Scripts/python -m py_compile <file>`

---

## File Map

| File | Task | Change |
|------|------|--------|
| `backend/fantasy_baseball/player_board.py` | T1 | `reset_board_cache()` clears `_projection_cache` |
| `backend/schemas.py` | T2 | Add `need_score_ci`, `need_score_volatile`, `projection_source` to `WaiverPlayerOut`; add `scored_at` + `scoring_model_version` to `WaiverWireResponse` |
| `backend/routers/fantasy.py` | T3 | Fix `n_cats` in recommendations endpoint to match main waiver |
| `backend/routers/fantasy.py` | T4 | Add `_NEED_SCORE_HISTORY` dict; wire CI + volatility into `_to_waiver_player` and `_score_fa`; populate `scored_at` / `scoring_model_version` in `WaiverWireResponse` |
| `frontend/lib/types.ts` | T5 | Add 3 optional fields to `WaiverAvailablePlayer` |
| `frontend/app/(dashboard)/war-room/waiver/page.tsx` | T5 | `NeedBar` shows `±CI`; `PlayerRow` shows Volatile badge |
| `tests/test_need_score_stability.py` | T1–T4 | New test file |

---

## Task 1: Fix `reset_board_cache` — clear `_projection_cache`

**Files:**
- Modify: `backend/fantasy_baseball/player_board.py:864`
- Create: `tests/test_need_score_stability.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_need_score_stability.py`:

```python
"""Tests for need_score stability fixes."""


def test_reset_board_cache_clears_projection_cache():
    """reset_board_cache must clear _projection_cache so post-ingestion projections are served."""
    import backend.fantasy_baseball.player_board as pb

    # Seed the projection cache with a fake stale entry
    pb._projection_cache["test_key_12345"] = {"name": "Fake Player", "z_score": 99.9}
    assert "test_key_12345" in pb._projection_cache

    # Calling reset_board_cache must clear it
    pb.reset_board_cache()

    assert "test_key_12345" not in pb._projection_cache, (
        "reset_board_cache() must call _projection_cache.clear() so "
        "post-ingestion DB updates are not masked by in-process cached projections"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```
venv/Scripts/python -m pytest tests/test_need_score_stability.py::test_reset_board_cache_clears_projection_cache -v
```

Expected: FAIL — cache entry survives reset.

- [ ] **Step 3: Implement the fix**

In `backend/fantasy_baseball/player_board.py`, `reset_board_cache()` starts at line 864. Add one line after `_BOARD = None`:

```python
def reset_board_cache() -> None:
    """Clear the in-memory board cache so the next get_board() call reloads from disk.

    Called by /admin/board/refresh when new Steamer CSVs are dropped to data/projections/.
    Safe to call at any time -- next request rebuilds the board automatically.
    """
    global _BOARD
    _BOARD = None
    _projection_cache.clear()  # invalidate per-player cache after ingestion updates DB
    try:
        from backend.fantasy_baseball.projections_loader import load_full_board
        load_full_board.cache_clear()
    except Exception:
        pass
```

- [ ] **Step 4: Syntax check + run test**

```
venv/Scripts/python -m py_compile backend/fantasy_baseball/player_board.py
venv/Scripts/python -m pytest tests/test_need_score_stability.py::test_reset_board_cache_clears_projection_cache -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```
git add backend/fantasy_baseball/player_board.py tests/test_need_score_stability.py
git commit -m "fix(backend): clear _projection_cache in reset_board_cache — was serving stale projections after ingestion"
```

---

## Task 2: Schema additions — CI, volatile flag, source, response metadata

**Files:**
- Modify: `backend/schemas.py:463` (after `availability_note` in `WaiverPlayerOut`)
- Modify: `backend/schemas.py:513` (after `data_as_of` in `WaiverWireResponse`)
- Test: `tests/test_need_score_stability.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_need_score_stability.py`:

```python
def test_waiver_player_out_has_stability_fields():
    """WaiverPlayerOut must accept need_score_ci, need_score_volatile, projection_source."""
    from backend.schemas import WaiverPlayerOut
    p = WaiverPlayerOut(
        player_id="123.p.456",
        name="José Caballero",
        team="TB",
        position="SS",
        need_score=12.59,
        need_score_ci=1.89,
        need_score_volatile=True,
        projection_source="steamer+statcast",
    )
    assert p.need_score_ci == 1.89
    assert p.need_score_volatile is True
    assert p.projection_source == "steamer+statcast"


def test_waiver_player_out_stability_fields_default_safe():
    """Stability fields default to None/False so existing callers are unaffected."""
    from backend.schemas import WaiverPlayerOut
    p = WaiverPlayerOut(player_id="x", name="X", team="T", position="OF")
    assert p.need_score_ci is None
    assert p.need_score_volatile is False
    assert p.projection_source is None


def test_waiver_wire_response_has_metadata():
    """WaiverWireResponse must expose scored_at and scoring_model_version."""
    from backend.schemas import WaiverWireResponse
    from datetime import date, datetime
    r = WaiverWireResponse(
        week_end=date.today(),
        matchup_opponent="Opp",
        category_deficits=[],
        top_available=[],
        two_start_pitchers=[],
        scored_at=datetime(2026, 6, 11, 10, 0, 0),
        scoring_model_version="2.1",
    )
    assert r.scoring_model_version == "2.1"
    assert r.scored_at is not None
```

- [ ] **Step 2: Run tests to confirm they fail**

```
venv/Scripts/python -m pytest tests/test_need_score_stability.py -v
```

Expected: 3 new tests FAIL with `ValidationError` / `TypeError`.

- [ ] **Step 3: Add fields to `WaiverPlayerOut`**

In `backend/schemas.py`, after the `availability_note` line (currently the last field before `@field_validator`, around line 463), add:

```python
    need_score_ci: Optional[float] = None          # ±confidence interval; None = unknown
    need_score_volatile: bool = False               # True when score swung >20% vs prior run
    projection_source: Optional[str] = None        # "steamer+statcast" | "steamer" | "draft_board" | "proxy"
```

- [ ] **Step 4: Add fields to `WaiverWireResponse`**

In `backend/schemas.py`, `WaiverWireResponse` ends with `data_as_of` (around line 513). Add after it:

```python
    scoring_model_version: str = "2.1"             # bumped when need_score formula changes
    scored_at: Optional[datetime] = None           # when this scoring run completed (ET)
```

Note: `datetime` is already imported in `schemas.py` via `from datetime import ...`. If not, add `from datetime import datetime` at the top of the class section or at file top.

- [ ] **Step 5: Syntax check + run tests**

```
venv/Scripts/python -m py_compile backend/schemas.py
venv/Scripts/python -m pytest tests/test_need_score_stability.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

```
git add backend/schemas.py tests/test_need_score_stability.py
git commit -m "feat(backend): add need_score_ci/volatile/projection_source to WaiverPlayerOut + scored_at/model_version to WaiverWireResponse"
```

---

## Task 3: Fix `n_cats` inconsistency in recommendations endpoint

**Files:**
- Modify: `backend/routers/fantasy.py:2909`
- Test: `tests/test_need_score_stability.py` (append)

Root Cause B: the main waiver endpoint uses `n_cats = max(1, len(category_deficits))` (all categories, consistent denominator) while the recommendations endpoint used `n_cats = max(1, len(_need_vector.needs))` (translated dict, can be smaller due to key collisions, inflates scores).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_need_score_stability.py`:

```python
def test_n_cats_formula_consistent_in_source():
    """Both waiver endpoints must use len(category_deficits) for n_cats, not _need_vector.needs."""
    import pathlib
    src = (pathlib.Path(__file__).parent.parent / "backend" / "routers" / "fantasy.py").read_text()

    # The buggy line: n_cats = max(1, len(_need_vector.needs))
    assert "max(1, len(_need_vector.needs))" not in src, (
        "Recommendations endpoint must use len(category_deficits) for n_cats, "
        "not len(_need_vector.needs) — the translated dict can be smaller due to "
        "_CANONICAL_TO_BOARD key collisions, inflating need_score vs main waiver endpoint"
    )
```

- [ ] **Step 2: Run test to confirm it fails**

```
venv/Scripts/python -m pytest tests/test_need_score_stability.py::test_n_cats_formula_consistent_in_source -v
```

Expected: FAIL — the bad formula is still in source.

- [ ] **Step 3: Fix the `n_cats` line**

In `backend/routers/fantasy.py`, find line 2909 (inside `_score_fa`, the recommendations closure):

```python
                        n_cats = max(1, len(_need_vector.needs))
```

Change to:

```python
                        n_cats = max(1, len(category_deficits))
```

This aligns with how the main waiver endpoint computes it at line 2141:
```python
                    n_cats = max(1, len(category_deficits))
```

- [ ] **Step 4: Syntax check + run tests**

```
venv/Scripts/python -m py_compile backend/routers/fantasy.py
venv/Scripts/python -m pytest tests/test_need_score_stability.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```
git add backend/routers/fantasy.py tests/test_need_score_stability.py
git commit -m "fix(backend): align n_cats denominator — use len(category_deficits) in recommendations endpoint (was len(_need_vector.needs))"
```

---

## Task 4: Wire volatility detection + CI into both waiver closures

**Files:**
- Modify: `backend/routers/fantasy.py:120` (module-level dict)
- Modify: `backend/routers/fantasy.py:2224` (inside `_to_waiver_player` after `apply_injury_penalty`)
- Modify: `backend/routers/fantasy.py:2261` (`WaiverPlayerOut` constructor in `_to_waiver_player`)
- Modify: `backend/routers/fantasy.py:2982` (inside `_score_fa` after `apply_injury_penalty`)
- Modify: `backend/routers/fantasy.py:3009` (`WaiverPlayerOut` constructor in `_score_fa`)
- Modify: `backend/routers/fantasy.py` (`WaiverWireResponse` return — search for `data_as_of=`)
- Test: `tests/test_need_score_stability.py` (append)

- [ ] **Step 1: Write tests**

Append to `tests/test_need_score_stability.py`:

```python
def test_need_score_history_and_volatility_logic_present():
    """fantasy.py must define _NEED_SCORE_HISTORY and reference need_score_volatile."""
    import pathlib
    src = (pathlib.Path(__file__).parent.parent / "backend" / "routers" / "fantasy.py").read_text()
    assert "_NEED_SCORE_HISTORY" in src, (
        "fantasy.py must define module-level _NEED_SCORE_HISTORY dict for per-player score tracking"
    )
    assert "need_score_volatile" in src, (
        "WaiverPlayerOut constructors must pass need_score_volatile="
    )
    assert "need_score_ci" in src, (
        "WaiverPlayerOut constructors must pass need_score_ci="
    )
    assert "projection_source" in src, (
        "WaiverPlayerOut constructors must pass projection_source="
    )
    assert "scored_at" in src, (
        "WaiverWireResponse must be returned with scored_at= populated"
    )
```

- [ ] **Step 2: Run test to confirm it fails**

```
venv/Scripts/python -m pytest tests/test_need_score_stability.py::test_need_score_history_and_volatility_logic_present -v
```

Expected: FAIL.

- [ ] **Step 3: Add `_NEED_SCORE_HISTORY` module-level dict**

In `backend/routers/fantasy.py`, after the existing `_LEAGUE_SETTINGS_CACHE` block (around line 122), add:

```python
# Per-player need_score history for volatility detection.
# Keyed by player_key → (score: float, ts: datetime).
# Only entries within the last 60 minutes trigger a volatile flag.
_NEED_SCORE_HISTORY: dict = {}
```

- [ ] **Step 4: Add CI + volatility block after `apply_injury_penalty` in `_to_waiver_player`**

In `_to_waiver_player` (main waiver endpoint), find the lines after `apply_injury_penalty` call (around line 2224):

```python
            need_score, _penalty_note = apply_injury_penalty(need_score, _overlay)
            if _penalty_note and "HIGH_INJURY_RISK" not in _sc_sigs:
                _sc_sigs.append("HIGH_INJURY_RISK")
```

After this block (before the injury suppression and small-sample logic), insert:

```python
            # ── Stability metadata ──────────────────────────────────────────
            _pkey_stab = p.get("player_key") or ""
            _volatile = False
            _now_stab = datetime.now(ZoneInfo("America/New_York"))
            if _pkey_stab and _pkey_stab in _NEED_SCORE_HISTORY:
                _prev_score, _prev_ts = _NEED_SCORE_HISTORY[_pkey_stab]
                _age_min = (_now_stab - _prev_ts).total_seconds() / 60.0
                if _age_min < 60.0 and _prev_score > 0.1:
                    _delta_pct = abs(need_score - _prev_score) / _prev_score
                    if _delta_pct > 0.20:
                        _volatile = True
                        logger.warning(
                            "[waiver_volatility] %s: %.2f→%.2f (%.0f%% swing in %.0fm)",
                            name, _prev_score, need_score, _delta_pct * 100.0, _age_min,
                        )
            if _pkey_stab:
                _NEED_SCORE_HISTORY[_pkey_stab] = (need_score, _now_stab)

            # CI factor: steamer+statcast=15%, steamer=25%, proxy/draft=40%
            _proj_src = board_player.get("fusion_source") or (
                "proxy" if board_player.get("is_proxy") else "draft_board"
            )
            _ci_factors = {"steamer+statcast": 0.15, "steamer": 0.25}
            _need_ci: Optional[float] = round(need_score * _ci_factors.get(_proj_src, 0.40), 2) if need_score > 0 else None
            logger.debug(
                "[waiver_score] %s need=%.3f ci=±%.2f src=%s volatile=%s n_defs=%d",
                name, need_score, _need_ci or 0.0, _proj_src, _volatile, len(category_deficits),
            )
```

- [ ] **Step 5: Add the 3 new fields to the `WaiverPlayerOut` constructor in `_to_waiver_player`**

The constructor currently ends around line 2290 with:
```python
                closer_role=_closer_role,
                availability_note=_avail_note,
            )
```

Change to:
```python
                closer_role=_closer_role,
                availability_note=_avail_note,
                need_score_ci=_need_ci,
                need_score_volatile=_volatile,
                projection_source=_proj_src,
            )
```

- [ ] **Step 6: Add CI + volatility block after `apply_injury_penalty` in `_score_fa`**

In `_score_fa` (recommendations endpoint), find the lines after `apply_injury_penalty` call (around line 2982):

```python
            need_score, _penalty_note = apply_injury_penalty(need_score, _overlay)
            if _penalty_note and "HIGH_INJURY_RISK" not in _sc_sigs:
                _sc_sigs.append("HIGH_INJURY_RISK")
```

After this block (before the `_hc` hot/cold block), insert:

```python
            # ── Stability metadata ──────────────────────────────────────────
            _pkey_stab_rec = p.get("player_key") or ""
            _volatile_rec = False
            _now_stab_rec = datetime.now(ZoneInfo("America/New_York"))
            if _pkey_stab_rec and _pkey_stab_rec in _NEED_SCORE_HISTORY:
                _prev_score_rec, _prev_ts_rec = _NEED_SCORE_HISTORY[_pkey_stab_rec]
                _age_min_rec = (_now_stab_rec - _prev_ts_rec).total_seconds() / 60.0
                if _age_min_rec < 60.0 and _prev_score_rec > 0.1:
                    _delta_pct_rec = abs(need_score - _prev_score_rec) / _prev_score_rec
                    if _delta_pct_rec > 0.20:
                        _volatile_rec = True
                        logger.warning(
                            "[waiver_volatility_rec] %s: %.2f→%.2f (%.0f%% swing in %.0fm)",
                            name, _prev_score_rec, need_score, _delta_pct_rec * 100.0, _age_min_rec,
                        )
            if _pkey_stab_rec:
                _NEED_SCORE_HISTORY[_pkey_stab_rec] = (need_score, _now_stab_rec)

            _proj_src_rec = bp.get("fusion_source") or (
                "proxy" if bp.get("is_proxy") else "draft_board"
            )
            _ci_factors_rec = {"steamer+statcast": 0.15, "steamer": 0.25}
            _need_ci_rec: Optional[float] = round(need_score * _ci_factors_rec.get(_proj_src_rec, 0.40), 2) if need_score > 0 else None
            logger.debug(
                "[waiver_score_rec] %s need=%.3f ci=±%.2f src=%s volatile=%s",
                name, need_score, _need_ci_rec or 0.0, _proj_src_rec, _volatile_rec,
            )
```

- [ ] **Step 7: Add the 3 new fields to the `WaiverPlayerOut` constructor in `_score_fa`**

The constructor currently ends around line 3029 with:
```python
                park_factor=round(_get_park_factor_rec(p.get("team") or "", "run"), 3),
                availability_note=_avail_note_rec,
            )
```

Change to:
```python
                park_factor=round(_get_park_factor_rec(p.get("team") or "", "run"), 3),
                availability_note=_avail_note_rec,
                need_score_ci=_need_ci_rec,
                need_score_volatile=_volatile_rec,
                projection_source=_proj_src_rec,
            )
```

- [ ] **Step 8: Populate `scored_at` and `scoring_model_version` in the `WaiverWireResponse` return**

In `get_fantasy_waiver_recommendations()`, find the final `return WaiverWireResponse(...)` call (search for `data_as_of=`). It currently has:

```python
        data_as_of=_data_as_of,
```

Add two lines after it (before closing `)`):

```python
        data_as_of=_data_as_of,
        scored_at=datetime.now(ZoneInfo("America/New_York")),
        scoring_model_version="2.1",
```

`ZoneInfo` is already imported in fantasy.py. `datetime` is also already imported.

- [ ] **Step 9: Syntax check + run tests**

```
venv/Scripts/python -m py_compile backend/routers/fantasy.py
venv/Scripts/python -m pytest tests/test_need_score_stability.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 10: Commit**

```
git add backend/routers/fantasy.py tests/test_need_score_stability.py
git commit -m "feat(backend): add need_score volatility detection + CI + projection_source to both waiver closures"
```

---

## Task 5: Frontend — volatile badge + ±CI in NeedBar + TypeScript types

**Files:**
- Modify: `frontend/lib/types.ts` (after `availability_note` in `WaiverAvailablePlayer`)
- Modify: `frontend/app/(dashboard)/war-room/waiver/page.tsx` (`NeedBar` props + `PlayerRow` badge area)

- [ ] **Step 1: Add 3 optional fields to `WaiverAvailablePlayer` and 2 to `WaiverResponse` in `frontend/lib/types.ts`**

**`WaiverAvailablePlayer`** — find the interface (currently ends with `availability_note?: string | null`). Add after `availability_note`:

```typescript
  need_score_ci?: number | null
  need_score_volatile?: boolean | null
  projection_source?: string | null
```

**`WaiverResponse`** — find the interface (currently ends with `data_as_of?: string | null`). Add after `data_as_of`:

```typescript
  scoring_model_version?: string | null
  scored_at?: string | null
```

- [ ] **Step 2: Update `NeedBar` to accept and display `ci` prop**

In `frontend/app/(dashboard)/war-room/waiver/page.tsx`, the current `NeedBar` signature is:

```tsx
function NeedBar({ score, contributions }: { score: number; contributions?: Record<string, number> }) {
```

Change to:

```tsx
function NeedBar({ score, contributions, ci }: { score: number; contributions?: Record<string, number>; ci?: number | null }) {
```

And update the return to show `±CI` after the score tooltip:

```tsx
function NeedBar({ score, contributions, ci }: { score: number; contributions?: Record<string, number>; ci?: number | null }) {
  const pct = Math.min(100, Math.max(0, score * 10))
  const color = score >= 7.0 ? 'bg-status-safe' : score >= 4.0 ? 'bg-status-bubble' : 'bg-text-muted'
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-bg-inset rounded-full overflow-hidden">
        <div className={cn('h-full rounded-full transition-all duration-700 ease-out', color)} style={{ width: `${pct}%` }} />
      </div>
      <Tooltip content={<NeedScoreTooltipContent score={score} contributions={contributions} />}>
        <span className="text-xs text-text-primary tabular-nums w-8 text-right cursor-help underline decoration-dotted">
          {score.toFixed(2)}
        </span>
      </Tooltip>
      {ci != null && ci > 0 && (
        <span className="text-[9px] text-text-muted tabular-nums flex-shrink-0">±{ci.toFixed(1)}</span>
      )}
    </div>
  )
}
```

- [ ] **Step 3: Pass `ci` prop in `PlayerRow`**

In `PlayerRow`, find the `NeedBar` usage (around line 298):

```tsx
          <NeedBar score={player.need_score} contributions={player.category_contributions} />
```

Change to:

```tsx
          <NeedBar score={player.need_score} contributions={player.category_contributions} ci={player.need_score_ci} />
```

- [ ] **Step 4: Add Volatile badge in `PlayerRow`**

In `PlayerRow`, the identity section shows badges (PREMIUM, STRONG, 2-Start, HotCold, momentum, injury). After the `injury_status` badge block, add:

```tsx
          {player.need_score_volatile && (
            <span className="text-[10px] px-1.5 py-0.5 bg-status-bubble/10 text-status-bubble border border-status-bubble/30 rounded font-semibold">
              ⚠ Volatile
            </span>
          )}
```

The exact location: after the `{player.injury_status && (...)}` block and before the end of the flex-wrap identity row.

- [ ] **Step 5: Commit**

```
git add frontend/lib/types.ts "frontend/app/(dashboard)/war-room/waiver/page.tsx"
git commit -m "feat(frontend): show ±CI on NeedBar + Volatile badge in PlayerRow"
```

---

## Final Verification

- [ ] **Run all stability tests**

```
venv/Scripts/python -m pytest tests/test_need_score_stability.py -v
```

Expected: 6/6 PASS.

- [ ] **Run full suite (no regressions)**

```
venv/Scripts/python -m pytest tests/ -q --tb=short
```

Expected: 4 pre-existing failures in `test_row_projector*.py` only. Zero new failures.

- [ ] **All syntax checks**

```
venv/Scripts/python -m py_compile backend/fantasy_baseball/player_board.py backend/schemas.py backend/routers/fantasy.py
```

All must exit 0.
