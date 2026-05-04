"""
Regression test for Phase 2 ID translation fix (Yahoo ID → MLBAM ID).

Validates that proxy players with Yahoo IDs correctly fetch projections
from the PlayerProjection table using the PlayerIDMapping translation layer.
"""
import pytest
from unittest.mock import MagicMock, patch
from backend.fantasy_baseball.player_board import get_or_create_projection


def test_yahoo_id_translates_to_mlbam_id_and_fetches_projection():
    """
    Test that a free agent with Yahoo ID successfully:
    1. Translates Yahoo ID (7590) → MLBAM ID (545361)
    2. Queries PlayerProjection with MLBAM ID
    3. Returns populated cat_scores from DB

    NOTE: Uses a player NOT on the board to force proxy creation.
    """
    yahoo_player = {
        "name": "Free Agent Call-Up Not On Board",
        "player_key": "469.p.7590",  # Yahoo ID = 7590
        "positions": ["OF"],
        "editorial_team_abbr": "ATL",
        "percent_owned": 99
    }
    
    # Mock DB session
    mock_db = MagicMock()
    
    # Mock PlayerIDMapping query result (Yahoo ID 7590 → MLBAM ID 545361)
    mock_id_mapping = MagicMock()
    mock_id_mapping.yahoo_id = "7590"
    mock_id_mapping.mlbam_id = 545361
    mock_id_mapping.bdl_id = None
    
    # Mock PlayerProjection query result (MLBAM ID 545361 has real projection)
    mock_projection = MagicMock()
    mock_projection.player_id = "545361"
    mock_projection.cat_scores = {
        "R": 1.8,
        "HR": 2.1,
        "RBI": 1.9,
        "SB": 1.5,
        "AVG": 1.2
    }
    
    # Configure query mocks to return correct results based on filter
    def query_side_effect(model):
        mock_query = MagicMock()
        if model.__name__ == "PlayerIDMapping":
            mock_query.filter.return_value.first.return_value = mock_id_mapping
        elif model.__name__ == "PlayerProjection":
            mock_query.filter.return_value.first.return_value = mock_projection
        return mock_query
    
    mock_db.query.side_effect = query_side_effect
    
    # Patch get_db at the import location (backend.models.get_db)
    with patch("backend.models.get_db") as mock_get_db:
        mock_get_db.return_value = iter([mock_db])
        
        result = get_or_create_projection(yahoo_player)
    
    # Assertions
    assert result["name"] == "Free Agent Call-Up Not On Board"
    assert result["team"] == "ATL"  # Extracted from editorial_team_abbr
    assert result["is_proxy"] is True
    assert result["cat_scores"] == {
        "R": 1.8,
        "HR": 2.1,
        "RBI": 1.9,
        "SB": 1.5,
        "AVG": 1.2
    }
    assert result["z_score"] == sum(mock_projection.cat_scores.values())  # 8.5
    
    # Verify queries were called correctly
    assert mock_db.query.call_count >= 2  # At least PlayerIDMapping + PlayerProjection


def test_yahoo_id_not_found_in_mapping_returns_empty_projection():
    """
    Test that a player with Yahoo ID not found in PlayerIDMapping
    returns empty cat_scores (graceful degradation).
    """
    yahoo_player = {
        "name": "Unknown Rookie",
        "player_key": "469.p.9999",  # Not in mapping table
        "positions": ["3B"],
        "team": "MIA"
    }
    
    mock_db = MagicMock()
    
    # Mock PlayerIDMapping query returns None (no mapping found)
    mock_db.query.return_value.filter.return_value.first.return_value = None
    
    with patch("backend.models.get_db") as mock_get_db:
        mock_get_db.return_value = iter([mock_db])
        
        result = get_or_create_projection(yahoo_player)
    
    # Assertions
    assert result["name"] == "Unknown Rookie"
    assert result["team"] == "MIA"
    assert result["is_proxy"] is True
    assert result["cat_scores"] == {}  # Empty because no mapping found
    assert result["z_score"] == 0.0  # NO synthetic baselines


def test_mlbam_id_not_in_projection_table_returns_empty():
    """
    Test that a player with valid Yahoo ID → MLBAM ID mapping,
    but no projection in PlayerProjection table, returns empty cat_scores.
    """
    yahoo_player = {
        "name": "Recent Call-Up",
        "player_key": "469.p.8888",
        "positions": ["SP"],
        "editorial_team_abbr": "SEA"
    }
    
    mock_db = MagicMock()
    
    # Mock PlayerIDMapping found (Yahoo ID → MLBAM ID)
    mock_id_mapping = MagicMock()
    mock_id_mapping.mlbam_id = 999999
    mock_id_mapping.bdl_id = None
    
    def query_side_effect(model):
        mock_query = MagicMock()
        if model.__name__ == "PlayerIDMapping":
            mock_query.filter.return_value.first.return_value = mock_id_mapping
        elif model.__name__ == "PlayerProjection":
            mock_query.filter.return_value.first.return_value = None  # Not in projection table
        return mock_query
    
    mock_db.query.side_effect = query_side_effect
    
    with patch("backend.models.get_db") as mock_get_db:
        mock_get_db.return_value = iter([mock_db])
        
        result = get_or_create_projection(yahoo_player)
    
    # Assertions
    assert result["name"] == "Recent Call-Up"
    assert result["team"] == "SEA"
    assert result["is_proxy"] is True
    assert result["cat_scores"] == {}  # Empty because no projection data
    assert result["z_score"] == 0.0  # NO synthetic baselines


def test_bdl_id_fallback_when_mlbam_id_null():
    """
    Test that when PlayerIDMapping has bdl_id but mlbam_id is None,
    the system falls back to using bdl_id for PlayerProjection query.
    """
    yahoo_player = {
        "name": "Player With BDL ID Only",
        "player_key": "469.p.5555",
        "positions": ["2B"],
        "editorial_team_abbr": "CHC"
    }
    
    mock_db = MagicMock()
    
    # Mock PlayerIDMapping with bdl_id but no mlbam_id
    mock_id_mapping = MagicMock()
    mock_id_mapping.yahoo_id = "5555"
    mock_id_mapping.mlbam_id = None
    mock_id_mapping.bdl_id = 700000
    
    # Mock PlayerProjection with bdl_id
    mock_projection = MagicMock()
    mock_projection.player_id = "700000"
    mock_projection.cat_scores = {
        "R": 0.8,
        "HR": 0.5,
        "RBI": 0.7,
        "SB": 0.2,
        "AVG": 0.6
    }
    
    def query_side_effect(model):
        mock_query = MagicMock()
        if model.__name__ == "PlayerIDMapping":
            mock_query.filter.return_value.first.return_value = mock_id_mapping
        elif model.__name__ == "PlayerProjection":
            mock_query.filter.return_value.first.return_value = mock_projection
        return mock_query
    
    mock_db.query.side_effect = query_side_effect
    
    with patch("backend.models.get_db") as mock_get_db:
        mock_get_db.return_value = iter([mock_db])
        
        result = get_or_create_projection(yahoo_player)
    
    # Assertions
    assert result["name"] == "Player With BDL ID Only"
    assert result["team"] == "CHC"
    assert result["cat_scores"] == mock_projection.cat_scores
    assert result["z_score"] == sum(mock_projection.cat_scores.values())  # 2.8
