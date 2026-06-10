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
