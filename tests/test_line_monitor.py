
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from backend.services.line_monitor import check_line_movements
from backend.models import BetLog, Game, Prediction

@pytest.fixture
def mock_db():
    with patch("backend.services.line_monitor.SessionLocal") as mock:
        yield mock()

@pytest.fixture
def mock_odds_client():
    with patch("backend.services.line_monitor.OddsAPIClient") as mock_cls:
        mock_cls.quota_is_low.return_value = False  # don't skip due to quota
        yield mock_cls()

@pytest.fixture
def mock_model():
    with patch("backend.services.line_monitor.CBBEdgeModel") as mock:
        yield mock()

@pytest.fixture
def mock_reanalysis():
    with patch("backend.services.line_monitor.ReanalysisEngine") as mock:
        yield mock

@pytest.fixture
def mock_alert():
    with patch("backend.services.line_monitor.send_line_movement_alert") as mock:
        yield mock

def test_check_line_movements_no_bets(mock_db, mock_odds_client):
    mock_db.query().filter().all.return_value = []
    mock_odds_client.get_todays_games.return_value = []
    
    result = check_line_movements()
    assert result["bets_checked"] == 0
    assert result["significant_moves"] == 0

def test_check_line_movements_significant_move(
    mock_db, mock_odds_client, mock_model, mock_reanalysis, mock_alert
):
    # Setup mock game and bet
    game = Game(id=1, external_id="game1", home_team="Duke", away_team="Kansas", is_neutral=False)
    bet = BetLog(id=1, game=game, pick="Duke -4.5", prediction_id=10)
    mock_db.query(BetLog).filter().all.return_value = [bet]
    
    # Mock current odds (Duke moved to -6.5, which is -2.0 pts delta)
    mock_odds_client.get_todays_games.return_value = [
        {"game_id": "game1", "sharp_consensus_spread": -6.5}
    ]
    
    # Mock prediction for re-analysis
    pred = Prediction(id=10, verdict="Bet 1.0u Duke -4.5", full_analysis={
        "inputs": {
            "ratings": {},
            "odds": {"spread": -4.5},
            "injuries": [],
            "home_style": {},
            "away_style": {}
        }
    })
    mock_db.query(Prediction).filter().first.return_value = pred
    
    # Mock re-analysis result
    mock_engine = mock_reanalysis.return_value
    mock_analysis = MagicMock()
    mock_analysis.edge_conservative = 0.03 # 3% edge
    mock_engine.reanalyze.return_value = mock_analysis
    
    with patch("backend.services.line_monitor.get_float_env", return_value=2.5):
        result = check_line_movements()
        
    assert result["significant_moves"] == 1
    assert result["abandonments"] == 0
    mock_alert.assert_called_once()
    # Duke -4.5 -> -6.5 is -2.0 move
    args, kwargs = mock_alert.call_args
    assert kwargs["delta"] == -2.0
    assert kwargs["abandoned"] == False

def test_check_line_movements_abandon(
    mock_db, mock_odds_client, mock_model, mock_reanalysis, mock_alert
):
    # Setup mock game and bet
    game = Game(id=1, external_id="game1", home_team="Duke", away_team="Kansas", is_neutral=False)
    bet = BetLog(id=1, game=game, pick="Duke -4.5", prediction_id=10)
    mock_db.query(BetLog).filter().all.return_value = [bet]
    
    # Mock current odds (Duke moved to -3.0, which is +1.5 pts delta - bad move)
    mock_odds_client.get_todays_games.return_value = [
        {"game_id": "game1", "sharp_consensus_spread": -3.0}
    ]
    
    # Mock prediction for re-analysis
    pred = Prediction(id=10, verdict="Bet 1.0u Duke -4.5", full_analysis={
        "inputs": {
            "ratings": {},
            "odds": {"spread": -4.5},
            "injuries": [],
            "home_style": {},
            "away_style": {}
        }
    })
    mock_db.query(Prediction).filter().first.return_value = pred
    
    # Mock re-analysis result (Edge dropped to 1%)
    mock_engine = mock_reanalysis.return_value
    mock_analysis = MagicMock()
    mock_analysis.edge_conservative = 0.01 
    mock_engine.reanalyze.return_value = mock_analysis
    
    with patch("backend.services.line_monitor.get_float_env", return_value=2.5):
        result = check_line_movements()
        
    assert result["significant_moves"] == 1
    assert result["abandonments"] == 1
    mock_alert.assert_called_once()
    args, kwargs = mock_alert.call_args
    assert kwargs["abandoned"] == True
    assert kwargs["new_edge"] == 0.01
