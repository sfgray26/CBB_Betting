"""Lineup schema contract tests — verifies that LineupPlayerOut and StartingPitcherOut
expose the fields required by the frontend / audit probe.

Production probe 2026-04-22 found all 23 players had lineup_status=null and all
10 pitchers had has_game=null after the batters/pitchers schema refactor. These
tests guard the schema field presence and default values.
"""

from backend.schemas import LineupPlayerOut, StartingPitcherOut


def test_lineup_player_out_has_lineup_status():
    """lineup_status must be a defined field (alias of status)."""
    p = LineupPlayerOut(
        player_id="test_id",
        name="Test Player",
        team="NYY",
        position="OF",
        status="START",
        lineup_status="START",
    )
    assert p.lineup_status == "START"


def test_lineup_player_out_lineup_status_defaults_none():
    """lineup_status defaults to None when not provided."""
    p = LineupPlayerOut(
        player_id="test_id",
        name="Test Player",
        team="NYY",
        position="OF",
    )
    assert p.lineup_status is None


def test_lineup_player_out_has_eligible_positions():
    """eligible_positions must be a defined Optional[List[str]] field."""
    p = LineupPlayerOut(
        player_id="test_id",
        name="Test Player",
        team="NYY",
        position="OF",
        eligible_positions=["OF", "Util"],
    )
    assert p.eligible_positions == ["OF", "Util"]


def test_lineup_player_out_eligible_positions_defaults_none():
    p = LineupPlayerOut(player_id="x", name="X", team="NYY", position="OF")
    assert p.eligible_positions is None


def test_lineup_player_out_has_game_time():
    from datetime import datetime, timezone
    t = datetime(2026, 4, 22, 18, 10, tzinfo=timezone.utc)
    p = LineupPlayerOut(player_id="x", name="X", team="NYY", position="OF", game_time=t)
    assert p.game_time == t


def test_starting_pitcher_out_has_game_defaults_false():
    """has_game must be defined and default to False (not null)."""
    p = StartingPitcherOut(
        player_id="sp_id",
        name="Test Pitcher",
        team="NYY",
    )
    assert p.has_game is False
    assert isinstance(p.has_game, bool)


def test_starting_pitcher_out_has_game_set_true():
    p = StartingPitcherOut(
        player_id="sp_id",
        name="Test Pitcher",
        team="NYY",
        has_game=True,
    )
    assert p.has_game is True


def test_starting_pitcher_out_is_two_start_defaults_false():
    p = StartingPitcherOut(player_id="x", name="X", team="NYY")
    assert p.is_two_start is False
    assert isinstance(p.is_two_start, bool)


def test_starting_pitcher_out_is_two_start_set_true():
    p = StartingPitcherOut(player_id="x", name="X", team="NYY", is_two_start=True)
    assert p.is_two_start is True


def test_starting_pitcher_out_game_time():
    from datetime import datetime, timezone
    t = datetime(2026, 4, 22, 19, 0, tzinfo=timezone.utc)
    p = StartingPitcherOut(player_id="x", name="X", team="NYY", game_time=t)
    assert p.game_time == t
