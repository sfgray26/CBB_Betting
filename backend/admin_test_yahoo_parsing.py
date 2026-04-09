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
        # Test basic parsing logic
        roster_wrapper = {
            'coverage_type': 'all',
            'date': '2026-04-09',
            'is_prescoring': False,
            'is_editable': True,
            '0': {
                'player_key': '469.p.12345',
                'name': 'Test Player'
            },
            'outs_pitched': 100
        }

        has_players_key = "players" in roster_wrapper
        players_raw = roster_wrapper.get("players", {})
        players_raw_has_data = len(players_raw) > 0

        player_entries = []
        for key, value in roster_wrapper.items():
            if key.isdigit() and isinstance(value, dict):
                player_entries.append(value)

        return {
            "status": "success",
            "test_results": {
                "has_players_key": has_players_key,
                "players_raw_has_data": players_raw_has_data,
                "player_entries_count": len(player_entries),
                "player_entries": player_entries
            },
            "message": "Parsing logic test completed successfully"
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
