"""
Tests for LineupConstraintSolver — natural-position scarcity bonus (K1, Session K).

Verifies:
1. ILP solver places the natural-C player at C, not Util, when both C-eligible players
   have equal scores (bonus breaks the tie).
2. Greedy solver does the same when OR-Tools is unavailable.
3. Baseline: solver assigns 9 unique players when roster has ≥9 eligible.
"""
import pytest
from unittest.mock import patch

from backend.fantasy_baseball.elite_lineup_scorer import EliteScore
from backend.fantasy_baseball.lineup_constraint_solver import (
    LineupConstraintSolver,
    PositionSlot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score(total: float) -> EliteScore:
    return EliteScore(
        total_score=total,
        environment_score=0.0,
        matchup_multiplier=1.0,
        platoon_multiplier=1.0,
        form_adjusted_woba=0.0,
        regression_boost=0.0,
        lineup_spot_bonus=0.0,
        confidence=1.0,
        data_quality="test",
        reasoning="test",
    )


def _player(pid: str, name: str) -> dict:
    return {"player_id": pid, "name": name}


def _build_full_roster():
    """
    9-player roster that fills all slots.
    PlayerA (C-natural) and PlayerB (C+SS) both score 10.0.
    Without the bonus the ILP has a free choice; with it, A→C and B→SS.
    """
    players = [
        _player("A", "Catcher Nat"),       # eligibility: ["C"]
        _player("B", "Catcher SS Flex"),   # eligibility: ["C", "SS"]
        _player("2B", "Second"),            # eligibility: ["2B"]
        _player("3B", "Third"),             # eligibility: ["3B"]
        _player("1B", "First"),             # eligibility: ["1B"]
        _player("OF1", "Outfield1"),        # eligibility: ["OF"]
        _player("OF2", "Outfield2"),        # eligibility: ["OF"]
        _player("OF3", "Outfield3"),        # eligibility: ["OF"]
        _player("DH", "Designated"),        # eligibility: ["DH"] → Util only
    ]
    scores = {
        "A":   _score(10.0),
        "B":   _score(10.0),   # equal — bonus must decide
        "2B":  _score(9.0),
        "3B":  _score(8.0),
        "1B":  _score(7.0),
        "OF1": _score(6.0),
        "OF2": _score(5.0),
        "OF3": _score(4.0),
        "DH":  _score(3.0),
    }
    eligibility = {
        "A":   ["C"],
        "B":   ["C", "SS"],
        "2B":  ["2B"],
        "3B":  ["3B"],
        "1B":  ["1B"],
        "OF1": ["OF"],
        "OF2": ["OF"],
        "OF3": ["OF"],
        "DH":  ["DH"],
    }
    return players, scores, eligibility


# ---------------------------------------------------------------------------
# ILP tests
# ---------------------------------------------------------------------------

def test_ilp_natural_pos_bonus_breaks_tie():
    """
    PlayerA (C-only, score=10) and PlayerB (C+SS, score=10) are tied.
    The natural-C bonus (+90 for A at C vs +90 for B at C) should place A at C
    and B at SS (combined bonus 90+80=170 beats B→C, A→Util at 90+0=90).
    """
    players, scores, eligibility = _build_full_roster()
    solver = LineupConstraintSolver()

    if not solver.use_ortools:
        pytest.skip("OR-Tools not available — ILP test skipped")

    result = solver.solve(players, scores, eligibility)

    slot_map = {a.slot: a.player_id for a in result.assignments}
    assert slot_map.get(PositionSlot.CATCHER) == "A", (
        f"Expected natural-C player 'A' at C slot, got {slot_map.get(PositionSlot.CATCHER)!r}"
    )
    assert slot_map.get(PositionSlot.SHORTSTOP) == "B", (
        f"Expected flex player 'B' at SS slot, got {slot_map.get(PositionSlot.SHORTSTOP)!r}"
    )


def test_ilp_baseline_assigns_nine_unique_players():
    """Solver fills all 9 slots with distinct players from a full roster."""
    players, scores, eligibility = _build_full_roster()
    solver = LineupConstraintSolver()

    if not solver.use_ortools:
        pytest.skip("OR-Tools not available — ILP test skipped")

    result = solver.solve(players, scores, eligibility)

    assigned_ids = [a.player_id for a in result.assignments if a.player_id]
    assert len(assigned_ids) == 9
    assert len(set(assigned_ids)) == 9, "Duplicate player assigned to multiple slots"


# ---------------------------------------------------------------------------
# Greedy tests
# ---------------------------------------------------------------------------

def test_greedy_natural_pos_bonus_breaks_tie():
    """Same tie-breaking scenario but with OR-Tools patched out → greedy path."""
    players, scores, eligibility = _build_full_roster()

    with patch("backend.fantasy_baseball.lineup_constraint_solver.ORTOOLS_AVAILABLE", False):
        solver = LineupConstraintSolver()
        assert not solver.use_ortools
        result = solver.solve(players, scores, eligibility)

    slot_map = {a.slot: a.player_id for a in result.assignments}
    assert slot_map.get(PositionSlot.CATCHER) == "A", (
        f"Expected natural-C player 'A' at C slot (greedy), got {slot_map.get(PositionSlot.CATCHER)!r}"
    )
    assert slot_map.get(PositionSlot.SHORTSTOP) == "B", (
        f"Expected flex player 'B' at SS slot (greedy), got {slot_map.get(PositionSlot.SHORTSTOP)!r}"
    )


def test_greedy_baseline_assigns_nine_unique_players():
    """Greedy solver fills all 9 slots with distinct players."""
    players, scores, eligibility = _build_full_roster()

    with patch("backend.fantasy_baseball.lineup_constraint_solver.ORTOOLS_AVAILABLE", False):
        solver = LineupConstraintSolver()
        result = solver.solve(players, scores, eligibility)

    assigned_ids = [a.player_id for a in result.assignments if a.player_id]
    assert len(assigned_ids) == 9
    assert len(set(assigned_ids)) == 9
