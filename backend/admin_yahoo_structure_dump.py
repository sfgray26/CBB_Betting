"""
Debug endpoint to dump the full Yahoo API roster response structure.
"""
from fastapi import APIRouter
import json
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/debug-yahoo-structure")
async def debug_yahoo_structure():
    """Dump the full Yahoo API roster response to understand the structure."""
    try:
        from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient

        client = YahooFantasyClient()

        # Make the raw API call
        import requests
        url = f"{client._base_url}league/{client.league_key}/teams/roster"
        headers = {"Authorization": f"Bearer {client._access_token}"}

        response = requests.get(url, params={'format': 'json'}, headers=headers, timeout=10)

        if response.status_code != 200:
            return {
                "status": "error",
                "error": f"Yahoo API call failed: {response.status_code}",
                "response": response.text[:500]
            }

        data = response.json()

        # Return the full structure for debugging
        return {
            "status": "success",
            "response_structure": json.dumps(data, indent=2)[:5000],  # First 5000 chars
            "full_response": data
        }

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
