"""
Admin endpoint to debug Yahoo API authentication and response structure.
TEMPORARY - Remove after Yahoo API debugging is complete.
"""

from fastapi import APIRouter
import requests
import os
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/debug-yahoo-api")
async def debug_yahoo_api():
    """
    Debug Yahoo API authentication and response structure.
    REMOVE AFTER DEBUGGING COMPLETE!
    """
    try:
        access_token = os.getenv('YAHOO_ACCESS_TOKEN')
        refresh_token = os.getenv('YAHOO_REFRESH_TOKEN')
        client_id = os.getenv('YAHOO_CLIENT_ID')
        client_secret = os.getenv('YAHOO_CLIENT_SECRET')

        results = {}
        headers = {'Authorization': f'Bearer {access_token}'}

        # TEST 1: User endpoint
        try:
            r = requests.get('https://fantasysports.yahooapis.com/fantasy/v2/users;use_login=1',
                            headers=headers, params={'format': 'json'}, timeout=10)
            results["user_endpoint"] = {
                "status_code": r.status_code,
                "success": r.status_code == 200
            }
            if r.status_code == 200:
                data = r.json()
                results["user_endpoint"]["authenticated"] = True
            else:
                results["user_endpoint"]["error"] = r.text[:200]
        except Exception as e:
            results["user_endpoint"] = {"error": str(e)}

        # TEST 2: League endpoint
        try:
            r = requests.get('https://fantasysports.yahooapis.com/fantasy/v2/league/469.l.72586',
                                headers=headers, params={'format': 'json'}, timeout=10)
            results["league_endpoint"] = {
                "status_code": r.status_code,
                "success": r.status_code == 200
            }
            if r.status_code == 200:
                data = r.json()
                results["league_endpoint"]["league_name"] = data.get('fantasy_content', {}).get('league', {}).get('name', 'N/A')
                results["league_endpoint"]["has_teams"] = 'team' in str(data).lower()
            else:
                results["league_endpoint"]["error"] = r.text[:200]
        except Exception as e:
            results["league_endpoint"] = {"error": str(e)}

        # TEST 3: League rosters endpoint
        try:
            r = requests.get('https://fantasysports.yahooapis.com/fantasy/v2/league/469.l.72586/teams/roster',
                                headers=headers, params={'format': 'json'}, timeout=10)
            results["rosters_endpoint"] = {
                "status_code": r.status_code,
                "success": r.status_code == 200
            }
            if r.status_code == 200:
                data = r.json()
                response_str = str(data)

                # Analyze response structure
                fantasy_content = data.get('fantasy_content', {})
                league_data = fantasy_content.get('league', {})
                teams_data = league_data.get('teams', {})

                if isinstance(teams_data, dict):
                    team_count = len(teams_data.get('team', []))
                else:
                    team_count = "unknown"

                results["rosters_endpoint"]["team_count"] = team_count
                results["rosters_endpoint"]["has_players"] = 'player' in response_str.lower()
                results["rosters_endpoint"]["response_preview"] = response_str[:500]
            else:
                results["rosters_endpoint"]["error"] = r.text[:200]
        except Exception as e:
            results["rosters_endpoint"] = {"error": str(e)}

        # TEST 4: Token refresh if 401s
        if any("401" in str(results.get(key, {})) for key in ["user_endpoint", "league_endpoint", "rosters_endpoint"]):
            try:
                r = requests.post('https://api.login.yahoo.com/oauth2/get_token', data={
                    'client_id': client_id,
                    'client_secret': client_secret,
                    'refresh_token': refresh_token,
                    'grant_type': 'refresh_token'
                }, timeout=10)
                results["token_refresh"] = {
                    "status_code": r.status_code,
                    "success": r.status_code == 200
                }
                if r.status_code == 200:
                    new_token = r.json().get('access_token')
                    results["token_refresh"]["new_token_preview"] = f"{new_token[:20]}..."
                else:
                    results["token_refresh"]["error"] = r.text[:200]
            except Exception as e:
                results["token_refresh"] = {"error": str(e)}

        results["overall_status"] = "SUCCESS" if all(
            results.get(key, {}).get("success", False) for key in ["user_endpoint", "league_endpoint", "rosters_endpoint"]
        ) else "NEEDS_INVESTIGATION"

        return results

    except Exception as e:
        logger.exception("Yahoo API debug failed")
        return {"status": "error", "error": str(e)}
