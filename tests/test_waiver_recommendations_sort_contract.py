"""Regression tests for the /api/fantasy/waiver/recommendations sort-safety contract.

Production probe 2026-04-22 returned a 503 with:
  'detail': "Unexpected error: '>' not supported between instances of 'float' and 'tuple'"

Root cause: a tuple (score, tiebreak) leaked into RosterMoveRecommendation.need_score,
which sorted() then compared against float values, raising TypeError.

Two guards were added:
  1. Pydantic field_validator on need_score: coerces tuple -> float (first element).
  2. _safe_need_score() helper at the sort site: extra belt-and-suspenders in case a
     non-numeric value bypasses model construction.

These tests verify both guards.
"""

import pathlib

import pytest


def test_need_score_validator_coerces_tuple():
    """RosterMoveRecommendation.need_score validates tuple -> float (first element)."""
    from backend.schemas import RosterMoveRecommendation
    rec = RosterMoveRecommendation(
        action="ADD",
        add_player=None,
        drop_player_name=None,
        drop_player_position=None,
        rationale="test",
        category_targets=[],
        need_score=(0.8, "tiebreak"),
        confidence=0.5,
    )
    assert rec.need_score == pytest.approx(0.8)
    assert isinstance(rec.need_score, float)


def test_need_score_validator_coerces_empty_tuple():
    from backend.schemas import RosterMoveRecommendation
    rec = RosterMoveRecommendation(
        action="ADD",
        add_player=None,
        drop_player_name=None,
        drop_player_position=None,
        rationale="test",
        category_targets=[],
        need_score=(),
        confidence=0.5,
    )
    assert rec.need_score == 0.0


def test_need_score_validator_coerces_string():
    from backend.schemas import RosterMoveRecommendation
    rec = RosterMoveRecommendation(
        action="ADD",
        add_player=None,
        drop_player_name=None,
        drop_player_position=None,
        rationale="test",
        category_targets=[],
        need_score="abc",
        confidence=0.5,
    )
    assert rec.need_score == 0.0


def test_need_score_validator_coerces_none():
    from backend.schemas import RosterMoveRecommendation
    rec = RosterMoveRecommendation(
        action="ADD",
        add_player=None,
        drop_player_name=None,
        drop_player_position=None,
        rationale="test",
        category_targets=[],
        need_score=None,
        confidence=0.5,
    )
    assert rec.need_score == 0.0


def test_need_score_validator_preserves_float():
    from backend.schemas import RosterMoveRecommendation
    rec = RosterMoveRecommendation(
        action="ADD",
        add_player=None,
        drop_player_name=None,
        drop_player_position=None,
        rationale="test",
        category_targets=[],
        need_score=1.234,
        confidence=0.5,
    )
    assert rec.need_score == pytest.approx(1.234)


def test_mixed_need_score_list_sorts_without_crash():
    """Sorted list with float, int, and tuple need_scores must not raise."""
    from backend.schemas import RosterMoveRecommendation

    def make_rec(score):
        return RosterMoveRecommendation(
            action="ADD",
            add_player=None,
            drop_player_name=None,
            drop_player_position=None,
            rationale="test",
            category_targets=[],
            need_score=score,
            confidence=0.5,
        )

    recs = [make_rec(v) for v in [(2.0, "x"), 1.5, 0, None, "bad", (0.9, "y"), 3.1]]
    # Sorting by need_score must not raise after the validator coerces all values.
    sorted_recs = sorted(recs, key=lambda r: r.need_score, reverse=True)
    scores = [r.need_score for r in sorted_recs]
    # Verify descending order
    assert scores == sorted(scores, reverse=True)


def test_router_has_safe_need_score_helper():
    """The fantasy router must contain the _safe_need_score defensive sort helper."""
    src = (
        pathlib.Path(__file__).parent.parent
        / "backend"
        / "routers"
        / "fantasy.py"
    ).read_text(encoding="utf-8")
    assert "_safe_need_score" in src, (
        "_safe_need_score helper missing from fantasy.py — "
        "the defensive sort guard was removed."
    )
