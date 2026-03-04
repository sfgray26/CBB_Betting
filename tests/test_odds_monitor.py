"""
Tests for OddsMonitor Level 5 Real-Time Pulse — Verdict Flips.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from backend.services.odds_monitor import OddsMonitor, LineMovement
from backend.betting_model import GameAnalysis

def test_odds_monitor_verdict_flip():
    """
    Test that a significant spread move triggers reanalysis 
    and correctly flags a VERDICT_FLIP event.
    """
    # 1. Setup
    monitor = OddsMonitor(api_key="test")
    
    # Mock analysis result (flips to a Bet)
    mock_analysis = MagicMock(spec=GameAnalysis)
    mock_analysis.verdict = "Bet 1.0u @ -110"
    mock_analysis.edge_conservative = 0.05
    
    # Mock ReanalysisEngine — original verdict was PASS so a flip to BET is detectable
    mock_engine = MagicMock()
    mock_engine.reanalyze.return_value = mock_analysis
    mock_engine._ctx.original_verdict = "PASS"
    
    # Pre-populate monitor history and cache
    game_id = "test_game_1"
    game_key = "Away@Home"
    monitor._reanalysis_cache[game_key] = mock_engine
    
    # Mock callback
    callback = MagicMock()
    monitor.on_significant_move(callback)
    
    # 2. Mock API response
    # First snapshot
    game_v1 = {
        "game_id": game_id,
        "home_team": "Home",
        "away_team": "Away",
        "best_spread": -3.5,
        "best_spread_odds": -110,
        "best_total": 145.0,
        "commence_time": datetime.utcnow().isoformat()
    }
    
    # Second snapshot (significant move -2.0 points)
    game_v2 = {
        "game_id": game_id,
        "home_team": "Home",
        "away_team": "Away",
        "best_spread": -5.5,
        "best_spread_odds": -110,
        "best_total": 145.0,
        "commence_time": datetime.utcnow().isoformat()
    }
    
    # 3. Run first poll
    with patch.object(monitor._client, 'get_todays_games', return_value=[game_v1]):
        monitor.poll()
    
    # 4. Run second poll (this should trigger the logic)
    with patch.object(monitor._client, 'get_todays_games', return_value=[game_v2]):
        result = monitor.poll()
    
    # 5. Assertions
    assert result["significant_movements"] == 1
    
    # Check if callback was called with VERDICT_FLIP
    assert callback.called
    movement = callback.call_args[0][0]
    assert isinstance(movement, LineMovement)
    assert movement.event_type == "VERDICT_FLIP"
    assert movement.fresh_analysis == mock_analysis
    assert movement.new_value == -5.5
    
    # Verify engine was called with the NEW spread
    mock_engine.reanalyze.assert_called_with(new_spread=-5.5)

if __name__ == "__main__":
    pytest.main([__file__])
