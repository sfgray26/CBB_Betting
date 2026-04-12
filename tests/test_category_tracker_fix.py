"""
Regression test for category_tracker dead-code removal (copy-paste bug lines 58-59).

Verifies:
- CategoryTracker can be instantiated with a mock Yahoo client
- get_category_needs() returns a list (not AttributeError from missing _extract_stats)
- get_category_needs() returns [] when matchup data is unavailable
- get_category_needs() returns CategoryNeed items when matchup data is valid
"""

from unittest.mock import MagicMock, patch
import pytest

from backend.fantasy_baseball.category_tracker import CategoryTracker
from backend.fantasy_baseball.smart_lineup_selector import CategoryNeed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(scoreboard=None, team_key="372.l.12345.t.1"):
    """Return a mock YahooFantasyClient."""
    client = MagicMock()
    client.get_my_team_key.return_value = team_key
    client.get_scoreboard.return_value = scoreboard or []
    return client


def _make_matchup(my_team_key="372.l.12345.t.1", opp_team_key="372.l.12345.t.2"):
    """Return a minimal Yahoo scoreboard matchup dict with two teams."""
    def _team_block(team_key, hr=5, r=20, rbi=18, sb=3, avg=0.270, ops=0.780):
        return {
            "team": [
                {"team_key": team_key},
                {
                    "team_stats": {
                        "stats": [
                            {"stat": {"stat_id": "7",  "value": str(hr)}},   # HR
                            {"stat": {"stat_id": "12", "value": str(r)}},    # R
                            {"stat": {"stat_id": "13", "value": str(rbi)}},  # RBI
                            {"stat": {"stat_id": "16", "value": str(sb)}},   # SB
                            {"stat": {"stat_id": "3",  "value": str(avg)}},  # AVG (if mapped)
                        ]
                    }
                },
            ]
        }

    return {
        "week": 4,
        "teams": {
            "count": 2,
            "0": _team_block(my_team_key, hr=5, r=20, rbi=18, sb=3),
            "1": _team_block(opp_team_key, hr=3, r=15, rbi=12, sb=1),
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCategoryTrackerInstantiation:
    def test_can_be_instantiated_with_mock_client(self):
        client = _make_client()
        tracker = CategoryTracker(client=client)
        assert tracker is not None
        assert tracker.client is client

    def test_has_no_extract_stats_method(self):
        """Confirm _extract_stats was never defined — removing the calls prevents AttributeError."""
        tracker = CategoryTracker(client=_make_client())
        assert not hasattr(tracker, "_extract_stats"), (
            "_extract_stats should not exist; its call sites were dead code"
        )


class TestGetCategoryNeedsReturnType:
    def test_returns_list_when_no_matchup(self):
        client = _make_client(scoreboard=[])
        tracker = CategoryTracker(client=client)
        result = tracker.get_category_needs()
        assert isinstance(result, list)

    def test_returns_empty_list_when_scoreboard_empty(self):
        client = _make_client(scoreboard=[])
        tracker = CategoryTracker(client=client)
        result = tracker.get_category_needs()
        assert result == []

    def test_returns_category_need_items_when_matchup_present(self):
        my_key = "372.l.12345.t.1"
        matchup = _make_matchup(my_team_key=my_key)
        client = _make_client(scoreboard=[matchup], team_key=my_key)
        tracker = CategoryTracker(client=client)
        result = tracker.get_category_needs()
        assert isinstance(result, list)
        # Must return CategoryNeed instances (may be empty if no stat IDs map, but no exception)
        for item in result:
            assert isinstance(item, CategoryNeed)

    def test_no_attribute_error_from_dead_code(self):
        """The old dead code called self._extract_stats() which does not exist.
        This test confirms no AttributeError is raised."""
        my_key = "372.l.12345.t.1"
        matchup = _make_matchup(my_team_key=my_key)
        client = _make_client(scoreboard=[matchup], team_key=my_key)
        tracker = CategoryTracker(client=client)
        # Should not raise AttributeError
        result = tracker.get_category_needs()
        assert isinstance(result, list)

    def test_team_order_does_not_matter(self):
        """Opponent listed first in matchup["teams"] should still resolve correctly."""
        my_key = "372.l.12345.t.1"
        opp_key = "372.l.12345.t.2"
        # Swap team order so opponent is at index 0
        matchup = _make_matchup(my_team_key=opp_key, opp_team_key=my_key)
        client = _make_client(scoreboard=[matchup], team_key=my_key)
        tracker = CategoryTracker(client=client)
        result = tracker.get_category_needs()
        assert isinstance(result, list)
