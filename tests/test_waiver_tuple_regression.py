"""
Regression test for P0 tuple leak in waiver recommendations endpoint.

Issue: fantasy.py line 2280 crashed with:
  "'>' not supported between instances of 'float' and 'tuple'"

Root cause: drop_candidate["z_score"] was leaking as tuple from board projection
data instead of being a float.

Fix: Added defensive type coercion to handle both tuple and float z_scores.
"""
import pytest
from unittest.mock import MagicMock, patch


async def mock_verify_api_key():
    """Mock API key verification for testing."""
    return "test-user"


def test_waiver_recommendations_tuple_z_score_coercion():
    """
    Verify waiver endpoint handles tuple-typed z_score without crashing.
    
    Simulates the April 22 regression where z_score leaked as tuple,
    causing max() comparison to fail with "'>' not supported between
    instances of 'float' and 'tuple'" error.
    """
    from backend.routers.fantasy import router
    from backend.auth import verify_api_key
    from fastapi.testclient import TestClient
    from backend.main import app
    
    client = TestClient(app)
    
    # Mock Yahoo client
    mock_client = MagicMock()
    
    # Roster with tuple z_score leak (simulating the bug)
    mock_roster = [
        {
            "name": "Seiya Suzuki",
            "player_key": "mlb.p.12345",
            "positions": ["OF", "LF"],
            "status": None,
            "selected_position": "OF",
            "is_undroppable": 0,
            "percent_owned": 78.5,
        }
    ]
    
    # Free agents
    mock_fas = [
        {
            "name": "Test Player",
            "player_key": "mlb.p.99999",
            "positions": ["OF"],
            "team": "TEST",
            "percent_owned": 12.5,
        }
    ]
    
    # Scoreboard (minimal valid structure)
    mock_scoreboard = [{
        "teams": {
            "count": 2,
            "0": {
                "team": [
                    {"team_key": "mlb.l.12345.t.1", "name": "My Team"},
                    {"team_stats": {"stats": []}},
                ]
            },
            "1": {
                "team": [
                    {"team_key": "mlb.l.12345.t.2", "name": "Opponent"},
                    {"team_stats": {"stats": []}},
                ]
            },
        }
    }]
    
    mock_client.get_roster.return_value = mock_roster
    mock_client.get_free_agents.return_value = mock_fas
    mock_client.get_scoreboard.return_value = mock_scoreboard
    mock_client.get_my_team_key.return_value = "mlb.l.12345.t.1"
    
    # Mock projection that returns tuple z_score (the bug)
    def mock_projection(yahoo_player):
        name = yahoo_player.get("name", "")
        if "Seiya" in name:
            # Simulate tuple leak
            return {
                "id": "seiya_suzuki",
                "name": "Seiya Suzuki",
                "positions": ["OF", "LF"],
                "team": "CHC",
                "type": "batter",
                "tier": 5,
                "rank": 150,
                "adp": 150.0,
                "z_score": (2.5, -5, 150.0, -78.5, 12345),  # TUPLE LEAK
                "cat_scores": {"hr": 0.8, "rbi": 0.6},
                "proj": {},
                "is_keeper": False,
                "keeper_round": None,
                "is_proxy": False,
            }
        else:
            # Normal FA projection
            return {
                "id": "test_player",
                "name": "Test Player",
                "positions": ["OF"],
                "team": "TEST",
                "type": "batter",
                "tier": 8,
                "rank": 300,
                "adp": 300.0,
                "z_score": 1.2,  # Normal float
                "cat_scores": {"hr": 0.5, "rbi": 0.4},
                "proj": {},
                "is_keeper": False,
                "keeper_round": None,
                "is_proxy": True,
            }
    
    with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client), \
         patch("backend.fantasy_baseball.player_board.get_or_create_projection", side_effect=mock_projection), \
         patch("backend.fantasy_baseball.statcast_loader.build_statcast_signals", return_value=([], 0.0)), \
         patch("backend.fantasy_baseball.statcast_loader.statcast_need_score_boost", return_value=0.0), \
         patch("backend.fantasy_baseball.pybaseball_loader.load_pybaseball_batters", return_value={}), \
         patch("backend.fantasy_baseball.pybaseball_loader.load_pybaseball_pitchers", return_value={}), \
         patch("backend.services.waiver_edge_detector.drop_candidate_value") as mock_drop_value:
        
        # Override auth dependency
        app.dependency_overrides[verify_api_key] = mock_verify_api_key
        
        try:
            # drop_candidate_value returns tuple (as designed)
            mock_drop_value.return_value = (2.5, -5, 150.0, -78.5, 12345)
            
            # The endpoint should NOT crash despite tuple z_score
            response = client.get(
                "/api/fantasy/waiver/recommendations",
            )
            
            # Should return 200, not 500
            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
            
            # Verify response structure
            data = response.json()
            assert "recommendations" in data
            assert isinstance(data["recommendations"], list)
        
        finally:
            # Clean up dependency override
            app.dependency_overrides.pop(verify_api_key, None)


def test_waiver_recommendations_normal_z_score():
    """Verify endpoint works with normal float z_score (control test)."""
    from backend.auth import verify_api_key
    from fastapi.testclient import TestClient
    from backend.main import app
    
    client = TestClient(app)
    
    mock_client = MagicMock()
    
    mock_roster = [
        {
            "name": "Normal Player",
            "player_key": "mlb.p.11111",
            "positions": ["OF"],
            "status": None,
            "selected_position": "OF",
            "is_undroppable": 0,
            "percent_owned": 65.0,
        }
    ]
    
    mock_fas = [
        {
            "name": "FA Player",
            "player_key": "mlb.p.22222",
            "positions": ["OF"],
            "team": "TEST",
            "percent_owned": 8.0,
        }
    ]
    
    mock_scoreboard = [{
        "teams": {
            "count": 2,
            "0": {"team": [{"team_key": "mlb.l.12345.t.1", "name": "My Team"}, {"team_stats": {"stats": []}}]},
            "1": {"team": [{"team_key": "mlb.l.12345.t.2", "name": "Opp"}, {"team_stats": {"stats": []}}]},
        }
    }]
    
    mock_client.get_roster.return_value = mock_roster
    mock_client.get_free_agents.return_value = mock_fas
    mock_client.get_scoreboard.return_value = mock_scoreboard
    mock_client.get_my_team_key.return_value = "mlb.l.12345.t.1"
    
    def normal_projection(yahoo_player):
        return {
            "id": yahoo_player.get("name", "").lower().replace(" ", "_"),
            "name": yahoo_player.get("name", "Unknown"),
            "positions": yahoo_player.get("positions", []),
            "team": yahoo_player.get("team", ""),
            "type": "batter",
            "tier": 6,
            "rank": 200,
            "adp": 200.0,
            "z_score": 1.5,  # Normal float — no tuple
            "cat_scores": {"hr": 0.6, "rbi": 0.5},
            "proj": {},
            "is_keeper": False,
            "keeper_round": None,
            "is_proxy": False,
        }
    
    with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client), \
         patch("backend.fantasy_baseball.player_board.get_or_create_projection", side_effect=normal_projection), \
         patch("backend.fantasy_baseball.statcast_loader.build_statcast_signals", return_value=([], 0.0)), \
         patch("backend.fantasy_baseball.statcast_loader.statcast_need_score_boost", return_value=0.0), \
         patch("backend.fantasy_baseball.pybaseball_loader.load_pybaseball_batters", return_value={}), \
         patch("backend.fantasy_baseball.pybaseball_loader.load_pybaseball_pitchers", return_value={}), \
         patch("backend.services.waiver_edge_detector.drop_candidate_value", return_value=(1.5, -6, 200.0, -65.0, 11111)):
        
        # Override auth dependency
        app.dependency_overrides[verify_api_key] = mock_verify_api_key
        
        try:
            response = client.get(
                "/api/fantasy/waiver/recommendations",
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "recommendations" in data
        
        finally:
            # Clean up dependency override
            app.dependency_overrides.pop(verify_api_key, None)

