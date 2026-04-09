"""
Test endpoint to verify Yahoo API parsing logic is working.
"""
from fastapi import APIRouter
import os

router = APIRouter()

@router.get("/test-yahoo-parsing")
async def test_yahoo_parsing():
    """Test endpoint to verify Yahoo API parsing logic."""
    try:
        # Import here to avoid circular imports
        from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient

        client = YahooFantasyClient()

        # Test get_league_rosters
        rosters = client.get_league_rosters(client.league_key)

        return {
            "status": "success",
            "roster_count": len(rosters),
            "first_3_rosters": rosters[:3] if len(rosters) > 0 else [],
            "message": f"Found {len(rosters)} total players"
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
