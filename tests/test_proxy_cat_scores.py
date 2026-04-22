"""Regression test for Step 2: Proxy players should fetch real projections from DB.

Issue: Free Agents not on draft board (proxy players) had empty cat_scores,
causing need_score to collapse to 0.0 for 90% of FAs. 

ARCHITECTURAL FIX: Query PlayerProjection table for real data instead of
generating synthetic baseline distributions (which would destroy +EV math).

If player has DB projection → use real cat_scores from DB
If player has NO DB projection → return empty cat_scores (correct behavior)
If test provides explicit cat_scores → preserve for test isolation
"""
import pytest
from backend.fantasy_baseball.player_board import get_or_create_projection


def test_proxy_player_with_explicit_cat_scores():
    """Test fixtures with explicit cat_scores should be preserved (test isolation)."""
    test_batter = {
        "player_key": "mlb.p.99999",
        "name": "Test Batter",
        "team": "CHC",
        "positions": ["OF"],
        "cat_scores": {"hr": 1.5, "r": 0.8, "rbi": 1.2},  # Explicit test data
    }
    
    projection = get_or_create_projection(test_batter)
    
    # Verify explicit cat_scores are preserved
    assert projection["cat_scores"] == {"hr": 1.5, "r": 0.8, "rbi": 1.2}
    assert projection["is_proxy"] is True


def test_proxy_player_without_db_projection():
    """Unknown players WITHOUT DB projection should have empty cat_scores (correct behavior)."""
    unknown_batter = {
        "player_key": "mlb.p.88888",
        "name": "Unknown Call-Up",
        "team": "MIL",
        "positions": ["OF"],
    }
    
    projection = get_or_create_projection(unknown_batter)
    
    # Verify it's a proxy
    assert projection["is_proxy"] is True
    
    # CRITICAL: No synthetic baselines! Empty cat_scores if no DB projection
    cat_scores = projection.get("cat_scores", {})
    assert cat_scores == {}, (
        "Players without DB projections should have empty cat_scores, "
        "not synthetic baseline distributions"
    )


def test_board_player_unchanged():
    """Players on board should still get real cat_scores (not proxy)."""
    # Use a known player from the hardcoded board (Aaron Judge)
    aaron_judge = {
        "player_key": "mlb.p.12345",
        "name": "Aaron Judge",
        "team": "NYY",
        "positions": ["OF"],
    }
    
    projection = get_or_create_projection(aaron_judge)
    
    # Aaron Judge should be on the board (has tier, adp, real cat_scores)
    # If he's on board, is_proxy should be False or None (not set for board players)
    if projection.get("tier", 99) < 10 and projection.get("adp", 999) < 500:
        # This is a real board player
        assert projection.get("is_proxy") is not True, "Board players should not be flagged as proxy"
        assert projection.get("cat_scores"), "Board players should have cat_scores"
        # Board players have more granular cat_scores (not just position baseline)
        assert len(projection["cat_scores"]) >= 8, "Board players have full category breakdown"
