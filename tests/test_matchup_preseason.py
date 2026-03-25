"""
EMAC-082: Tests for matchup preseason message field.

Bug 5: MatchupResponse.message field was missing — matchup page showed blank
when scoreboard returned no data. Fix adds Optional[str] message field.

Run:
    pytest tests/test_matchup_preseason.py -v
"""

import pytest
from backend.schemas import MatchupResponse, MatchupTeamOut


def _stub_team(name: str = "N/A") -> MatchupTeamOut:
    return MatchupTeamOut(team_key="", team_name=name, stats={})


def test_matchup_stub_has_message():
    """Stub MatchupResponse carries the preseason message."""
    stub = MatchupResponse(
        my_team=_stub_team(),
        opponent=_stub_team(),
        message="No active matchup. Season starts March 28, 2026.",
    )
    assert stub.message == "No active matchup. Season starts March 28, 2026."


def test_matchup_schema_has_message_field():
    """MatchupResponse.message is Optional[str] and defaults to None."""
    stub = MatchupResponse(
        my_team=_stub_team("My Team"),
        opponent=_stub_team("Their Team"),
    )
    assert stub.message is None
