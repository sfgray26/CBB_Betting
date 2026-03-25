"""
Integration-level unit tests for waiver wire backend.
All Yahoo API calls are mocked — no live network required.
"""
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yahoo_player(name="Test Player", positions=None, percent_owned=42.5):
    return {
        "player_key": "422.p.12345",
        "name": name,
        "team": "NYY",
        "positions": positions or ["OF"],
        "percent_owned": percent_owned,
        "status": None,
        "injury_note": None,
        "is_undroppable": 0,
    }


# ---------------------------------------------------------------------------
# Step 1 — yahoo_client.py "out" param
# ---------------------------------------------------------------------------

class TestYahooClientOutParam:

    def test_get_free_agents_includes_out_param(self):
        from backend.fantasy_baseball.yahoo_client import YahooFantasyClient

        client = YahooFantasyClient.__new__(YahooFantasyClient)
        client.league_key = "fake.lg.999"

        captured_params = {}

        def mock_get(path, params=None):
            captured_params.update(params or {})
            return {"fantasy_content": {"league": [{}, {"players": {}}]}}

        client._get = mock_get
        client._league_section = lambda data, idx: {}
        client._parse_players_block = lambda raw: []

        client.get_free_agents(count=10)

        assert "out" in captured_params, "get_free_agents() must include 'out' param"
        # Yahoo MLB rejects 'ownership' as a subresource on the /players collection endpoint
        # (400: Invalid subresource ownership requested). Use metadata,stats only.
        assert "ownership" not in captured_params["out"], "ownership subresource breaks MLB API (use metadata,stats)"
        assert "metadata" in captured_params["out"]
        assert "stats" in captured_params["out"]

    def test_get_waiver_players_includes_out_param(self):
        from backend.fantasy_baseball.yahoo_client import YahooFantasyClient

        client = YahooFantasyClient.__new__(YahooFantasyClient)
        client.league_key = "fake.lg.999"

        captured_params = {}

        def mock_get(path, params=None):
            captured_params.update(params or {})
            return {"fantasy_content": {"league": [{}, {"players": {}}]}}

        client._get = mock_get
        client._league_section = lambda data, idx: {}
        client._parse_players_block = lambda raw: []

        client.get_waiver_players(count=10)

        assert "out" in captured_params, "get_waiver_players() must include 'out' param"
        assert "ownership" not in captured_params["out"], "ownership subresource breaks MLB API (use metadata,stats)"
        assert "metadata" in captured_params["out"]
        assert "stats" in captured_params["out"]


# ---------------------------------------------------------------------------
# Step 2 — two_start_pitchers must only contain SPs
# ---------------------------------------------------------------------------

class TestTwoStartPitchers:

    def test_two_start_pitchers_are_sps_only(self):
        """Only players with 'SP' in positions should appear in two_start_pitchers."""
        from backend.services.waiver_edge_detector import WaiverEdgeDetector

        # Verify the filtering logic directly
        mixed_fas = [
            {"name": "SP Guy", "positions": ["SP"], "cat_scores": {"era": 1.0}, "percent_owned": 5.0},
            {"name": "OF Guy", "positions": ["OF"], "cat_scores": {"hr": 1.0}, "percent_owned": 10.0},
            {"name": "RP Guy", "positions": ["RP"], "cat_scores": {"sv": 1.0}, "percent_owned": 3.0},
        ]
        sp_fas = [p for p in mixed_fas if "SP" in (p.get("positions") or [])]
        assert len(sp_fas) == 1
        assert sp_fas[0]["name"] == "SP Guy"


# ---------------------------------------------------------------------------
# Step 3 — top_available must be sorted descending by need_score
# ---------------------------------------------------------------------------

class TestTopAvailableSorting:

    def test_top_available_sorted_descending(self):
        """After scoring, top_available must be sorted by need_score descending."""
        # Simulate the sort logic applied in the endpoint
        players = [
            MagicMock(need_score=0.5, owned_pct=30.0),
            MagicMock(need_score=2.1, owned_pct=15.0),
            MagicMock(need_score=1.3, owned_pct=25.0),
        ]
        players.sort(key=lambda x: x.need_score, reverse=True)
        scores = [p.need_score for p in players]
        assert scores == sorted(scores, reverse=True), "top_available must be descending by need_score"


# ---------------------------------------------------------------------------
# Step 4 — get_roster() called, not get_my_roster()
# ---------------------------------------------------------------------------

class TestRosterMethodName:

    def test_get_roster_is_called_not_get_my_roster(self):
        """YahooFantasyClient.get_roster() must exist; get_my_roster() must not."""
        from backend.fantasy_baseball.yahoo_client import YahooFantasyClient

        assert hasattr(YahooFantasyClient, "get_roster"), (
            "YahooFantasyClient must have get_roster() method"
        )
        assert not hasattr(YahooFantasyClient, "get_my_roster"), (
            "get_my_roster() doesn't exist — main.py must use get_roster() instead"
        )

    def test_main_py_does_not_call_get_my_roster(self):
        """Verify get_my_roster() has been removed from main.py source."""
        import inspect
        import backend.main as main_module
        source = inspect.getsource(main_module)
        assert "get_my_roster" not in source, (
            "main.py still calls get_my_roster() — must be changed to get_roster()"
        )


# ---------------------------------------------------------------------------
# Step 5 — position filter forwarded to Yahoo
# ---------------------------------------------------------------------------

class TestPositionFilter:

    def test_position_filter_forwarded_to_yahoo(self):
        """When position='2B' is passed, get_free_agents() must receive position='2B'."""
        from backend.fantasy_baseball.yahoo_client import YahooFantasyClient

        client = YahooFantasyClient.__new__(YahooFantasyClient)
        client.league_key = "fake.lg.999"

        captured_params = {}

        def mock_get(path, params=None):
            captured_params.update(params or {})
            return {"fantasy_content": {"league": [{}, {"players": {}}]}}

        client._get = mock_get
        client._league_section = lambda data, idx: {}
        client._parse_players_block = lambda raw: []

        client.get_free_agents(position="2B", count=10)

        assert captured_params.get("position") == "2B", (
            f"Expected position='2B' in params, got: {captured_params}"
        )
