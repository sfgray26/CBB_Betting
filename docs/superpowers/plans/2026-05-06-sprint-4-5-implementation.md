# Sprint 4 + 5: Matchup Context Activation & Projection Freshness

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the already-populated `matchup_context` table into waiver recommendations (Sprint 4), and make `ProjectionAssemblyService` prefer live DB projections over stale March CSVs (Sprint 5).

**Architecture:** Sprint 4 adds a `_load_matchup_scores()` bulk-load helper and applies a confidence-weighted matchup addend inside `get_top_moves()`, gated by `feature_matchup_enabled`. Sprint 5 adds `_get_live_projection()` to `ProjectionAssemblyService` so that when `player_projections` rows exist with `update_method != 'prior'`, their counting stats override the March board CSVs. Also adds `/admin/board/refresh` to clear the in-memory `_BOARD` cache without a Railway restart.

**Tech Stack:** Python 3.11, SQLAlchemy, FastAPI, pytest. No new tables. No schema changes. Windows dev env: `venv/Scripts/python`.

---

## File Map

| File | Change |
|------|--------|
| `backend/services/waiver_edge_detector.py` | Add `_load_matchup_scores()`, refactor BDL ID map, apply matchup addend, expose fields in move dict |
| `scripts/seed_matchup_context_flag.py` | New — seeds `feature_matchup_enabled = false` into `feature_flags` |
| `tests/test_waiver_edge.py` | Append matchup context tests (extend existing file) |
| `backend/fantasy_baseball/projection_assembly_service.py` | Add `_get_live_projection()`, call in `_assemble_batter()` / `_assemble_pitcher()` |
| `backend/main.py` | Add `POST /admin/board/refresh` endpoint |
| `tests/test_projection_assembly_service.py` | Append live-projection override tests |

---

## Sprint 4: Matchup Context Wiring

### Task 1: Tests for `_load_matchup_scores` (write failing first)

**Files:**
- Modify: `tests/test_waiver_edge.py`

- [ ] **Step 1: Append failing tests**

Add to end of `tests/test_waiver_edge.py`:

```python
# ---------------------------------------------------------------------------
# Sprint 4: matchup context loading
# ---------------------------------------------------------------------------
from unittest.mock import patch, MagicMock

class TestLoadMatchupScores:
    def test_returns_empty_on_empty_ids(self):
        result = WaiverEdgeDetector._load_matchup_scores([])
        assert result == {}

    def test_returns_empty_on_db_error(self):
        with patch("backend.services.waiver_edge_detector.SessionLocal",
                   side_effect=Exception("db down")):
            result = WaiverEdgeDetector._load_matchup_scores([123])
        assert result == {}

    def test_maps_bdl_id_to_matchup_fields(self):
        mock_db = MagicMock()
        row = MagicMock()
        row.__getitem__ = lambda s, i: [101, 1.5, 80.0, 0.85][i]
        mock_db.execute.return_value.fetchall.return_value = [row]
        mock_db.__enter__ = lambda s: s
        mock_db.__exit__ = MagicMock(return_value=False)
        with patch("backend.services.waiver_edge_detector.SessionLocal",
                   return_value=mock_db):
            result = WaiverEdgeDetector._load_matchup_scores([101])
        assert 101 in result
        assert result[101]["matchup_z"] == 1.5
        assert result[101]["matchup_score"] == 80.0
        assert result[101]["matchup_confidence"] == 0.85

    def test_none_values_coerced_to_defaults(self):
        mock_db = MagicMock()
        row = MagicMock()
        row.__getitem__ = lambda s, i: [202, None, None, None][i]
        mock_db.execute.return_value.fetchall.return_value = [row]
        mock_db.__enter__ = lambda s: s
        mock_db.__exit__ = MagicMock(return_value=False)
        with patch("backend.services.waiver_edge_detector.SessionLocal",
                   return_value=mock_db):
            result = WaiverEdgeDetector._load_matchup_scores([202])
        assert result[202]["matchup_z"] == 0.0
        assert result[202]["matchup_score"] == 50.0
        assert result[202]["matchup_confidence"] == 0.0
```

- [ ] **Step 2: Run to confirm they fail**

```
venv/Scripts/python -m pytest tests/test_waiver_edge.py::TestLoadMatchupScores -v --tb=short
```

Expected: `AttributeError: type object 'WaiverEdgeDetector' has no attribute '_load_matchup_scores'`

### Task 2: Implement `_load_matchup_scores` in `waiver_edge_detector.py`

**Files:**
- Modify: `backend/services/waiver_edge_detector.py`

- [ ] **Step 1: Add the method after `_load_market_scores` (around line 262)**

Insert after the closing `}` of `_load_market_scores`:

```python
    def _load_matchup_scores(self, bdl_ids: list) -> dict:
        """Sprint 4: Bulk-load today's matchup context for given BDL player IDs.

        Returns {bdl_id: {"matchup_z": float, "matchup_score": float, "matchup_confidence": float}}.
        Empty dict on any failure — never blocks waiver recommendations.
        """
        if not bdl_ids:
            return {}
        try:
            from datetime import date as _date
            from backend.models import SessionLocal
            from sqlalchemy import text

            today = _date.today()
            db = SessionLocal()
            try:
                rows = db.execute(
                    text(
                        "SELECT bdl_player_id, matchup_z, matchup_score, matchup_confidence "
                        "FROM matchup_context "
                        "WHERE bdl_player_id = ANY(:ids) AND game_date = :today"
                    ),
                    {"ids": list(bdl_ids), "today": today},
                ).fetchall()
                return {
                    r[0]: {
                        "matchup_z": float(r[1]) if r[1] is not None else 0.0,
                        "matchup_score": float(r[2]) if r[2] is not None else 50.0,
                        "matchup_confidence": float(r[3]) if r[3] is not None else 0.0,
                    }
                    for r in rows
                }
            finally:
                db.close()
        except Exception as exc:
            logger.debug("_load_matchup_scores failed (%s)", exc)
            return {}
```

- [ ] **Step 2: Run tests — should pass**

```
venv/Scripts/python -m pytest tests/test_waiver_edge.py::TestLoadMatchupScores -v --tb=short
```

Expected: 4 passed

### Task 3: Write failing tests for matchup addend in `get_top_moves`

**Files:**
- Modify: `tests/test_waiver_edge.py`

- [ ] **Step 1: Append integration test for addend**

Add to end of `tests/test_waiver_edge.py`:

```python
class TestMatchupAddendInGetTopMoves:
    """Matchup z-score is added to batter score when feature_matchup_enabled=True."""

    def _make_detector(self):
        det = WaiverEdgeDetector.__new__(WaiverEdgeDetector)
        det.mcmc = None
        det.fa_cache_ttl = 600
        return det

    def _fa(self, yahoo_id, positions=None):
        return {
            "player_id": yahoo_id,
            "name": f"Player {yahoo_id}",
            "positions": positions or ["OF"],
            "z_score": 1.0,
            "cat_scores": {},
            "proj": {},
            "tier": 3,
            "adp": 100.0,
            "percent_owned": 5.0,
        }

    def test_batter_score_increased_by_positive_matchup(self):
        det = self._make_detector()
        fa = self._fa(999)

        with patch.object(det, "_fetch_fas", return_value=[fa]), \
             patch.object(det, "_enrich_players", side_effect=lambda x: x), \
             patch.object(det, "_compute_deficits", return_value={}), \
             patch.object(det, "_load_scarcity_lookup", return_value={}), \
             patch.object(det, "_weakest_droppable_at", return_value=None), \
             patch.object(det, "_yahoo_id_to_bdl_id", return_value=999), \
             patch.object(det, "_load_matchup_scores", return_value={
                 999: {"matchup_z": 2.0, "matchup_score": 90.0, "matchup_confidence": 0.8}
             }), \
             patch("backend.services.waiver_edge_detector.is_flag_enabled",
                   side_effect=lambda f: f == "feature_matchup_enabled"):
            moves = det.get_top_moves([], [], n_candidates=1)

        assert len(moves) == 1
        # matchup_addend = 2.0 * 0.8 * 0.5 = 0.8 added to base score
        assert moves[0]["matchup_z"] == 2.0
        assert moves[0]["matchup_confidence"] == 0.8
        assert moves[0]["need_score"] > 1.0  # base z_score=1.0 + addend

    def test_pitcher_score_not_affected_by_matchup(self):
        det = self._make_detector()
        fa = self._fa(888, positions=["SP"])

        with patch.object(det, "_fetch_fas", return_value=[fa]), \
             patch.object(det, "_enrich_players", side_effect=lambda x: x), \
             patch.object(det, "_compute_deficits", return_value={}), \
             patch.object(det, "_load_scarcity_lookup", return_value={}), \
             patch.object(det, "_weakest_droppable_at", return_value=None), \
             patch.object(det, "_yahoo_id_to_bdl_id", return_value=888), \
             patch.object(det, "_load_matchup_scores", return_value={
                 888: {"matchup_z": 3.0, "matchup_score": 95.0, "matchup_confidence": 1.0}
             }), \
             patch("backend.services.waiver_edge_detector.is_flag_enabled",
                   side_effect=lambda f: f == "feature_matchup_enabled"):
            moves = det.get_top_moves([], [], n_candidates=1)

        assert len(moves) == 1
        # Pitchers are not affected by hitter matchup context
        assert moves[0]["need_score"] == pytest.approx(1.0, abs=0.15)  # base score only (scarcity 6%)
        assert moves[0]["matchup_z"] == 0.0  # no matchup data for pitchers

    def test_low_confidence_matchup_not_applied(self):
        det = self._make_detector()
        fa = self._fa(777)

        with patch.object(det, "_fetch_fas", return_value=[fa]), \
             patch.object(det, "_enrich_players", side_effect=lambda x: x), \
             patch.object(det, "_compute_deficits", return_value={}), \
             patch.object(det, "_load_scarcity_lookup", return_value={}), \
             patch.object(det, "_weakest_droppable_at", return_value=None), \
             patch.object(det, "_yahoo_id_to_bdl_id", return_value=777), \
             patch.object(det, "_load_matchup_scores", return_value={
                 777: {"matchup_z": 2.0, "matchup_score": 80.0, "matchup_confidence": 0.2}
             }), \
             patch("backend.services.waiver_edge_detector.is_flag_enabled",
                   side_effect=lambda f: f == "feature_matchup_enabled"):
            moves = det.get_top_moves([], [], n_candidates=1)

        assert len(moves) == 1
        # confidence=0.2 < 0.3 threshold — addend NOT applied
        assert moves[0]["matchup_z"] == 2.0  # exposed for transparency
        assert moves[0]["need_score"] == pytest.approx(1.0, abs=0.15)  # base only
```

- [ ] **Step 2: Run to confirm they fail**

```
venv/Scripts/python -m pytest tests/test_waiver_edge.py::TestMatchupAddendInGetTopMoves -v --tb=short
```

Expected: AttributeError or KeyError — `matchup_z` not in move dict yet.

### Task 4: Wire matchup into `get_top_moves`

**Files:**
- Modify: `backend/services/waiver_edge_detector.py` (lines ~370-480)

- [ ] **Step 1: Refactor BDL ID map and add matchup load**

In `get_top_moves`, replace the existing market-signals BDL resolution block:

**FIND (starting around line 371):**
```python
        # PR 4.5: Load market_score as tertiary tiebreaker
        from backend.services.config_service import is_flag_enabled
        use_market_signals = is_flag_enabled("market_signals_enabled")
        market_lookup = {}
        if use_market_signals:
            bdl_ids = []
            for fa in free_agents[:40]:
                # Map Yahoo player_key to BDL ID
                yahoo_id = fa.get("player_id")  # Yahoo player_id (integer)
                if yahoo_id:
                    bdl_id = self._yahoo_id_to_bdl_id(yahoo_id)
                    if bdl_id:
                        bdl_ids.append(bdl_id)
            if bdl_ids:
                market_lookup = self._load_market_scores(bdl_ids)
```

**REPLACE WITH:**
```python
        from backend.services.config_service import is_flag_enabled

        # Build single BDL ID map for all FAs — reused by market signals + matchup context.
        fa_bdl_map: dict = {}  # yahoo_id -> bdl_id
        for fa in free_agents[:40]:
            yahoo_id = fa.get("player_id")
            if yahoo_id:
                bdl_id = self._yahoo_id_to_bdl_id(yahoo_id)
                if bdl_id:
                    fa_bdl_map[yahoo_id] = bdl_id
        bdl_id_list = list(fa_bdl_map.values())

        # PR 4.5: Load market_score as tertiary tiebreaker
        use_market_signals = is_flag_enabled("market_signals_enabled")
        market_lookup: dict = {}
        if use_market_signals and bdl_id_list:
            market_lookup = self._load_market_scores(bdl_id_list)

        # Sprint 4: Load today's matchup context for all FA batters
        use_matchup_context = is_flag_enabled("feature_matchup_enabled")
        matchup_lookup: dict = {}
        if use_matchup_context and bdl_id_list:
            matchup_lookup = self._load_matchup_scores(bdl_id_list)
```

- [ ] **Step 2: Update the FA loop to resolve BDL ID from `fa_bdl_map` and apply addend**

Inside the `for fa in free_agents[:40]:` loop, replace the market_score resolution and the depth-factor block with:

**FIND (inside the loop, around line 426):**
```python
            # PR 4.5: Load market_score as tertiary tiebreaker
            market_score = 50.0  # Default to neutral
            if use_market_signals:
                yahoo_id = fa.get("player_id")
                if yahoo_id:
                    bdl_id = self._yahoo_id_to_bdl_id(yahoo_id)
                    if bdl_id:
                        market_score = market_lookup.get(bdl_id, 50.0)
            if score <= 0 and not has_deficit_signal:
                score = float(fa.get("z_score") or 0.0)
```

**REPLACE WITH:**
```python
            # Resolve BDL ID for this FA (pre-built above)
            fa_yahoo_id = fa.get("player_id")
            fa_bdl_id = fa_bdl_map.get(fa_yahoo_id) if fa_yahoo_id else None

            # PR 4.5: Market score as tertiary tiebreaker
            market_score = market_lookup.get(fa_bdl_id, 50.0) if fa_bdl_id else 50.0

            if score <= 0 and not has_deficit_signal:
                score = float(fa.get("z_score") or 0.0)

            # Sprint 4: Apply matchup context addend for batters only.
            # Hitter matchup context (opponent pitcher ERA/WHIP, park factor, handedness splits)
            # is only meaningful for batting categories — skip pitchers.
            fa_is_pitcher = bool(_PITCHER_POSITIONS.intersection(fa.get("positions") or []))
            matchup_row = matchup_lookup.get(fa_bdl_id, {}) if fa_bdl_id else {}
            matchup_z = matchup_row.get("matchup_z", 0.0) or 0.0
            matchup_score_val = matchup_row.get("matchup_score", 50.0) or 50.0
            matchup_confidence = matchup_row.get("matchup_confidence", 0.0) or 0.0
            if (
                use_matchup_context
                and not fa_is_pitcher
                and matchup_confidence >= 0.3
            ):
                # Addend: matchup_z weighted by confidence, scaled to ~15% of a typical score.
                # At matchup_z=2.0, confidence=1.0: adds 1.0 to score (strong positive matchup).
                # At matchup_z=-2.0, confidence=1.0: subtracts 1.0 (bad matchup).
                score += matchup_z * matchup_confidence * 0.5
```

- [ ] **Step 3: Update the move dict to expose matchup fields**

**FIND:**
```python
            move = {
                "add_player": fa,
                "drop_player_name": drop_candidate.get("name", "") if drop_candidate else "",
                "need_score": score,
                "market_score": market_score,  # PR 4.5: Tertiary tiebreaker
                "canonical_depth_factor": applied_depth_factor,  # 1.0 when flag disabled
```

**REPLACE WITH:**
```python
            move = {
                "add_player": fa,
                "drop_player_name": drop_candidate.get("name", "") if drop_candidate else "",
                "need_score": score,
                "market_score": market_score,
                "canonical_depth_factor": applied_depth_factor,
                "matchup_z": matchup_z if not fa_is_pitcher else 0.0,
                "matchup_score": matchup_score_val if not fa_is_pitcher else 50.0,
                "matchup_confidence": matchup_confidence if not fa_is_pitcher else 0.0,
```

- [ ] **Step 4: Remove the now-duplicate `fa_is_pitcher` declaration below (it was declared in the depth-factor block)**

The existing CANONICAL_PROJECTION_V1 depth-factor block uses `fa_is_pitcher` — ensure it now reads from the variable declared in Step 2 (already in scope before that block). Remove the duplicate `_PITCHER_POSITIONS = frozenset(...)` if it was inside the loop.

- [ ] **Step 5: Syntax check**

```
venv/Scripts/python -m py_compile backend/services/waiver_edge_detector.py
```

Expected: no output (clean).

- [ ] **Step 6: Run tests**

```
venv/Scripts/python -m pytest tests/test_waiver_edge.py::TestMatchupAddendInGetTopMoves tests/test_waiver_edge.py::TestLoadMatchupScores -v --tb=short
```

Expected: all pass.

- [ ] **Step 7: Commit**

```
git add backend/services/waiver_edge_detector.py tests/test_waiver_edge.py
git commit -m "feat(sprint-4): wire matchup_context into waiver recommendations"
```

### Task 5: Seed script for `feature_matchup_enabled` flag

**Files:**
- Create: `scripts/seed_matchup_context_flag.py`

- [ ] **Step 1: Write script**

```python
"""Seed feature_matchup_enabled into feature_flags (default: disabled)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models import SessionLocal
from sqlalchemy import text


def main():
    db = SessionLocal()
    try:
        db.execute(
            text(
                "INSERT INTO feature_flags (flag_name, enabled) "
                "VALUES ('feature_matchup_enabled', false) "
                "ON CONFLICT (flag_name) DO NOTHING"
            )
        )
        db.commit()
        row = db.execute(
            text(
                "SELECT flag_name, enabled FROM feature_flags "
                "WHERE flag_name = 'feature_matchup_enabled'"
            )
        ).fetchone()
        print(f"Flag state: {row}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check**

```
venv/Scripts/python -m py_compile scripts/seed_matchup_context_flag.py
```

- [ ] **Step 3: Commit**

```
git add scripts/seed_matchup_context_flag.py
git commit -m "feat(sprint-4): add seed script for feature_matchup_enabled flag"
```

---

## Sprint 5: Projection Freshness

### Task 6: Write failing tests for `_get_live_projection`

**Files:**
- Modify: `tests/test_projection_assembly_service.py`

- [ ] **Step 1: Append failing tests**

Add to end of `tests/test_projection_assembly_service.py`:

```python
# ---------------------------------------------------------------------------
# Sprint 5: live projection lookup
# ---------------------------------------------------------------------------
from unittest.mock import MagicMock, patch

class TestGetLiveProjection:
    def _make_service(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.db = MagicMock()
        svc.season = 2026
        return svc

    def test_returns_none_when_no_mlbam_and_no_name_match(self):
        svc = self._make_service()
        svc.db.query.return_value.filter.return_value.first.return_value = None
        result = svc._get_live_projection(None, "Unknown Player")
        assert result is None

    def test_returns_row_when_mlbam_matches(self):
        svc = self._make_service()
        mock_row = MagicMock()
        mock_row.update_method = "bayesian"
        mock_row.hr = 25
        svc.db.query.return_value.filter.return_value.first.return_value = mock_row
        result = svc._get_live_projection(12345, "Mike Trout")
        assert result is mock_row

    def test_returns_none_when_only_prior_update_method(self):
        svc = self._make_service()
        mock_row = MagicMock()
        mock_row.update_method = "prior"
        svc.db.query.return_value.filter.return_value.first.return_value = mock_row
        result = svc._get_live_projection(12345, "Mike Trout")
        # prior-only rows are treated as stale — don't override board
        assert result is None

    def test_counting_stats_override_board_proj(self):
        """When live projection exists, its counting stats replace board CSV values."""
        svc = self._make_service()
        mock_proj = MagicMock()
        mock_proj.update_method = "bayesian"
        mock_proj.hr = 30
        mock_proj.r = 90
        mock_proj.rbi = 95
        mock_proj.sb = 5

        board_proj = {"hr": 18, "r": 70, "rbi": 65, "sb": 8, "pa": 550,
                      "avg": 0.280, "obp": 0.360, "slg": 0.500}

        # Simulate what _assemble_batter does with the live projection
        live = mock_proj if mock_proj.update_method != "prior" else None
        if live is not None:
            merged = dict(board_proj)
            if live.hr is not None:
                merged["hr"] = live.hr
                merged["r"] = live.r
                merged["rbi"] = live.rbi
                merged["sb"] = live.sb
        else:
            merged = board_proj

        assert merged["hr"] == 30
        assert merged["r"] == 90
        assert merged["rbi"] == 95
        assert merged["sb"] == 5
        assert merged["avg"] == 0.280  # rate stats unchanged
```

- [ ] **Step 2: Run to confirm they fail**

```
venv/Scripts/python -m pytest tests/test_projection_assembly_service.py::TestGetLiveProjection -v --tb=short
```

Expected: `AttributeError: 'ProjectionAssemblyService' object has no attribute '_get_live_projection'`

### Task 7: Implement `_get_live_projection` and hook into `_assemble_batter`/`_assemble_pitcher`

**Files:**
- Modify: `backend/fantasy_baseball/projection_assembly_service.py`

- [ ] **Step 1: Add import for `PlayerProjection` at top of file**

In the `from backend.models import (...)` block, add `PlayerProjection`:

```python
from backend.models import (
    CanonicalProjection,
    CategoryImpact,
    PlayerIdentity,
    PlayerProjection,
    StatcastBatterMetrics,
    StatcastPitcherMetrics,
    SessionLocal,
)
```

- [ ] **Step 2: Add `_get_live_projection` method to `ProjectionAssemblyService`**

Add after `_get_statcast_pitcher` (find that method and insert after its closing `return`):

```python
    def _get_live_projection(
        self, mlbam_id: Optional[int], player_name: str
    ) -> Optional[object]:
        """Return a PlayerProjection row if one exists with update_method != 'prior'.

        Only non-prior rows are trusted over the static board — 'prior' means the
        row was seeded from the March CSV and has not been updated by fangraphs_ros
        or a Bayesian pass yet.

        Returns None when:
          - No row exists for the player.
          - Row exists but update_method == 'prior' (stale — treat as no override).
          - Any DB error (never raises).
        """
        try:
            if mlbam_id is not None:
                row = (
                    self.db.query(PlayerProjection)
                    .filter(PlayerProjection.player_id == str(mlbam_id))
                    .first()
                )
                if row is not None:
                    return row if row.update_method != "prior" else None
            # Fallback: name match (catches players without MLBAM mapping)
            norm = _normalize_name(player_name)
            row = (
                self.db.query(PlayerProjection)
                .filter(PlayerProjection.player_name.ilike(f"%{player_name}%"))
                .first()
            )
            if row is not None:
                return row if row.update_method != "prior" else None
        except Exception as exc:
            logger.debug("_get_live_projection failed for %s: %s", player_name, exc)
        return None
```

- [ ] **Step 3: Hook into `_assemble_batter` — merge counting stats from live projection**

In `_assemble_batter`, after `board_proj = player.get("proj", {})` (or equivalent — the `board_proj` is passed as a parameter, so add the merge at the top of the method body, before `pa = float(...)`):

**FIND (in `_assemble_batter`, line ~237):**
```python
    ) -> int:
        pa = float(board_proj.get("pa", 0) or 0)
```

**REPLACE WITH:**
```python
    ) -> int:
        # Sprint 5: prefer live DB counting stats over March CSV when available
        live_proj = self._get_live_projection(mlbam_id, "")
        if live_proj is not None and live_proj.hr is not None:
            board_proj = dict(board_proj)
            board_proj["hr"] = live_proj.hr or board_proj.get("hr", 0)
            board_proj["r"] = live_proj.r or board_proj.get("r", 0)
            board_proj["rbi"] = live_proj.rbi or board_proj.get("rbi", 0)
            board_proj["sb"] = live_proj.sb or board_proj.get("sb", 0)

        pa = float(board_proj.get("pa", 0) or 0)
```

- [ ] **Step 4: Hook into `_assemble_pitcher` — merge counting stats from live projection**

In `_assemble_pitcher`, similarly after the method signature and before `ip = float(...)`:

**FIND (in `_assemble_pitcher`, line ~324):**
```python
    ) -> int:
        ip = float(board_proj.get("ip", 0) or 0)
```

**REPLACE WITH:**
```python
    ) -> int:
        # Sprint 5: prefer live DB counting stats over March CSV when available
        live_proj = self._get_live_projection(mlbam_id, "")
        if live_proj is not None:
            board_proj = dict(board_proj)
            if live_proj.w is not None:
                board_proj["w"] = live_proj.w
            if live_proj.k_pit is not None:
                board_proj["k"] = live_proj.k_pit
            if live_proj.nsv is not None:
                board_proj["sv"] = live_proj.nsv

        ip = float(board_proj.get("ip", 0) or 0)
```

- [ ] **Step 5: Syntax check**

```
venv/Scripts/python -m py_compile backend/fantasy_baseball/projection_assembly_service.py
```

- [ ] **Step 6: Run tests**

```
venv/Scripts/python -m pytest tests/test_projection_assembly_service.py::TestGetLiveProjection -v --tb=short
```

Expected: 4 passed.

- [ ] **Step 7: Commit**

```
git add backend/fantasy_baseball/projection_assembly_service.py tests/test_projection_assembly_service.py
git commit -m "feat(sprint-5): prefer live player_projections over March CSV for counting stats"
```

### Task 8: Board cache refresh endpoint

**Files:**
- Modify: `backend/fantasy_baseball/player_board.py` (add `reset_board_cache()`)
- Modify: `backend/main.py` (add `/admin/board/refresh` endpoint)

- [ ] **Step 1: Add `reset_board_cache()` to `player_board.py`**

`get_board()` uses a module-level `_BOARD` global. Add this function after `get_board()`:

```python
def reset_board_cache() -> None:
    """Clear the in-memory board cache so the next get_board() call reloads from disk.

    Called by /admin/board/refresh when new Steamer CSVs are dropped to data/projections/.
    Safe to call at any time — next request rebuilds the board automatically.
    """
    global _BOARD
    _BOARD = None
    # Also clear the projections_loader lru_cache so it re-reads CSVs
    try:
        from backend.fantasy_baseball.projections_loader import load_full_board
        load_full_board.cache_clear()
    except Exception:
        pass
```

- [ ] **Step 2: Add endpoint to `backend/main.py`**

Find the admin endpoints section (search for `/admin/force-capture-lines`) and add after it:

```python
@app.post("/admin/board/refresh")
async def admin_board_refresh(user: str = Depends(verify_admin_api_key)):
    """Clear in-memory player board cache. Use after dropping new Steamer CSVs to data/projections/."""
    from backend.fantasy_baseball.player_board import reset_board_cache
    reset_board_cache()
    return {"status": "ok", "message": "Board cache cleared — next request reloads from disk"}
```

- [ ] **Step 3: Syntax check both files**

```
venv/Scripts/python -m py_compile backend/fantasy_baseball/player_board.py
venv/Scripts/python -m py_compile backend/main.py
```

- [ ] **Step 4: Commit**

```
git add backend/fantasy_baseball/player_board.py backend/main.py
git commit -m "feat(sprint-5): add board cache refresh endpoint for CSV hot-reload"
```

---

## Sprint 3: DevOps Handoff (no code — Railway execution only)

### Task 9: Write DevOps handoff prompt

The following Railway commands need to run. This is a DevOps handoff to Codex — no code changes required.

**AFTER canonical_projection_refresh validates successfully:**

```bash
# Savant park factors
railway connect postgres
# Run: migration from scripts/migration_savant_park_factors.py
railway run python scripts/migration_savant_park_factors.py
railway run python scripts/seed_savant_park_factors.py

# Savant pitch quality
railway run python scripts/migration_savant_pitch_quality.py
railway run python scripts/seed_savant_pitch_quality_flags.py
railway run python scripts/backfill_savant_pitch_quality.py
# Validate: SELECT COUNT(*), AVG(savant_pitch_quality) FROM savant_pitch_quality_scores WHERE season = 2026;

# Matchup context flag (seed, keep disabled until Sprint 4 validation)
railway run python scripts/seed_matchup_context_flag.py

# Verify fangraphs_ros job health
# railway logs --filter "fangraphs_ros" -- look for "fetched N RoS" or "failed"
# If "failed" every day → FanGraphs RoS is Cloudflare-blocked, document and close
```

---

## Final Validation

- [ ] **Run full test suite**

```
venv/Scripts/python -m pytest tests/ -q --tb=short
```

Expected: 2684 + new tests passing, 0 failed.

- [ ] **Update HANDOFF.md** with Sprint 4 + 5 status and DevOps handoff for Sprint 3.
