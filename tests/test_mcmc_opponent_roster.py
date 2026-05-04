"""Regression tests for Session W: opponent roster wired into MCMC simulator.

Bug: fantasy.py called _sim_move(opponent_roster=[]) — hardcoded empty list — so
win_prob_before == win_prob_after for every recommendation, causing win_prob_gain=0.0.

Fix: extract opponent_team_key from scoreboard matchup, call client.get_roster(opponent_team_key)
to build opponent_roster_scored, pass it to _sim_move. Falls back to [] on any exception.

These tests guard against silent regression of that fix.
"""
import pathlib
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SCOREBOARD = [{
    "teams": {
        "count": 2,
        "0": {
            "team": [
                {"team_key": "469.l.1.t.7", "name": "My Team"},
                {"team_stats": {"stats": []}},
            ]
        },
        "1": {
            "team": [
                {"team_key": "469.l.1.t.3", "name": "Rival Squad"},
                {"team_stats": {"stats": []}},
            ]
        },
    }
}]

_OPP_ROSTER = [
    {
        "name": "Opp Player",
        "player_key": "mlb.p.22222",
        "positions": ["OF"],
        "status": None,
        "selected_position": "OF",
        "is_undroppable": 0,
        "percent_owned": 60.0,
        "starts_this_week": 2,
    }
]


# ---------------------------------------------------------------------------
# Test 1: structural guard (fast, zero mocks)
# ---------------------------------------------------------------------------

def test_fantasy_router_contains_opponent_roster_extraction():
    """CI tripwire — fails immediately if Session W is reverted."""
    src = (
        pathlib.Path(__file__).parent.parent / "backend" / "routers" / "fantasy.py"
    ).read_text(encoding="utf-8")

    assert 'opponent_team_key = ""' in src, (
        "opponent_team_key must be initialized empty before scoreboard parse"
    )
    assert "client.get_roster(opponent_team_key)" in src, (
        "opponent roster must be fetched via client.get_roster(opponent_team_key)"
    )
    assert "opponent_roster=opponent_roster_scored" in src, (
        "opponent_roster_scored must be passed to _sim_move"
    )


# ---------------------------------------------------------------------------
# Test 2: happy path — get_roster called with opponent team key
# ---------------------------------------------------------------------------

def test_opponent_roster_fetched_when_scoreboard_has_opponent():
    """When scoreboard identifies opponent_team_key, client.get_roster is called
    with that key and the result is non-empty (Session W fix working)."""
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.auth import verify_api_key

    opp_roster_calls = []

    def roster_side_effect(*args):
        if args:
            opp_roster_calls.append(args[0])
            return _OPP_ROSTER
        return []  # my roster: empty is fine for this test

    mock_client = MagicMock()
    mock_client.get_my_team_key.return_value = "469.l.1.t.7"
    mock_client.get_scoreboard.return_value = _SCOREBOARD
    mock_client.get_roster.side_effect = roster_side_effect
    mock_client.get_free_agents.return_value = []

    async def _auth():
        return "test-user"

    app.dependency_overrides[verify_api_key] = _auth
    try:
        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client), \
             patch("backend.fantasy_baseball.statcast_loader.build_statcast_signals", return_value=([], 0.0)), \
             patch("backend.fantasy_baseball.statcast_loader.statcast_need_score_boost", return_value=0.0), \
             patch("backend.fantasy_baseball.pybaseball_loader.load_pybaseball_batters", return_value={}), \
             patch("backend.fantasy_baseball.pybaseball_loader.load_pybaseball_pitchers", return_value={}):
            client = TestClient(app)
            resp = client.get("/api/fantasy/waiver/recommendations")

        assert resp.status_code == 200

        # Core assertion: get_roster was called with opponent team key
        assert "469.l.1.t.3" in opp_roster_calls, (
            "get_roster must be called with opponent_team_key='469.l.1.t.3'. "
            f"Actual calls with args: {opp_roster_calls}"
        )

        # Sanity: matchup_opponent is correctly extracted from same scoreboard pass
        data = resp.json()
        assert data.get("matchup_opponent") == "Rival Squad"

    finally:
        app.dependency_overrides.pop(verify_api_key, None)


# ---------------------------------------------------------------------------
# Test 3: fallback path — no crash when get_roster raises for opponent
# ---------------------------------------------------------------------------

def test_opponent_roster_falls_back_to_empty_on_exception():
    """When client.get_roster(opponent_team_key) raises, opponent_roster_scored
    silently falls back to [] and the route still returns HTTP 200."""
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.auth import verify_api_key

    def roster_side_effect(*args):
        if args:
            raise RuntimeError("Simulated Yahoo API failure fetching opponent roster")
        return []

    mock_client = MagicMock()
    mock_client.get_my_team_key.return_value = "469.l.1.t.7"
    mock_client.get_scoreboard.return_value = _SCOREBOARD
    mock_client.get_roster.side_effect = roster_side_effect
    mock_client.get_free_agents.return_value = []

    async def _auth():
        return "test-user"

    app.dependency_overrides[verify_api_key] = _auth
    try:
        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client), \
             patch("backend.fantasy_baseball.statcast_loader.build_statcast_signals", return_value=([], 0.0)), \
             patch("backend.fantasy_baseball.statcast_loader.statcast_need_score_boost", return_value=0.0), \
             patch("backend.fantasy_baseball.pybaseball_loader.load_pybaseball_batters", return_value={}), \
             patch("backend.fantasy_baseball.pybaseball_loader.load_pybaseball_pitchers", return_value={}):
            client = TestClient(app)
            resp = client.get("/api/fantasy/waiver/recommendations")

        # Route must survive get_roster exception without crashing
        assert resp.status_code == 200, (
            f"Route must return 200 even when get_roster raises. Got: {resp.status_code}"
        )
        data = resp.json()
        assert "recommendations" in data

    finally:
        app.dependency_overrides.pop(verify_api_key, None)
