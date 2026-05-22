#!/usr/bin/env python3
"""
Diagnostic script for scoreboard showing all zeros issue.
Tests Yahoo API connectivity and data retrieval.
"""
import os
import sys
import json
import logging

# Add the project root to path
sys.path.insert(0, '/mnt/c/Users/sfgra/repos/Fixed/cbb-edge')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_yahoo_connection():
    """Test basic Yahoo API connectivity."""
    try:
        from backend.fantasy_baseball.yahoo_client_resilient import get_yahoo_client, YahooAuthError, YahooAPIError
        
        logger.info("=== Testing Yahoo API Connection ===")
        client = get_yahoo_client()
        logger.info("✓ Yahoo client initialized successfully")
        
        # Test league key
        logger.info(f"  League key: {client.league_key}")
        logger.info(f"  League ID: {client.league_id}")
        
        return client
    except YahooAuthError as e:
        logger.error(f"✗ Yahoo authentication error: {e}")
        return None
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        return None

def test_get_league(client):
    """Test fetching league data."""
    try:
        logger.info("\n=== Testing Get League ===")
        league = client.get_league()
        logger.info(f"✓ League data retrieved")
        logger.info(f"  League name: {league.get('name', 'N/A')}")
        return True
    except Exception as e:
        logger.error(f"✗ Error fetching league: {e}")
        return False

def test_get_scoreboard(client):
    """Test fetching scoreboard data."""
    try:
        logger.info("\n=== Testing Get Scoreboard ===")
        scoreboard = client.get_scoreboard()
        logger.info(f"✓ Scoreboard retrieved")
        logger.info(f"  Number of matchups: {len(scoreboard)}")
        
        if scoreboard:
            logger.info(f"  First matchup keys: {list(scoreboard[0].keys())}")
            if 'teams' in scoreboard[0]:
                teams = scoreboard[0]['teams']
                logger.info(f"  Teams data type: {type(teams)}")
                if isinstance(teams, dict):
                    logger.info(f"  Teams keys: {list(teams.keys())}")
        
        return scoreboard
    except Exception as e:
        logger.error(f"✗ Error fetching scoreboard: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_get_matchup_stats(client):
    """Test fetching matchup stats."""
    try:
        logger.info("\n=== Testing Get Matchup Stats ===")
        matchup_stats = client.get_matchup_stats()
        
        logger.info(f"✓ Matchup stats retrieved")
        logger.info(f"  Opponent name: {matchup_stats.get('opponent_name', 'N/A')}")
        
        my_stats = matchup_stats.get('my_stats', {})
        opp_stats = matchup_stats.get('opp_stats', {})
        
        logger.info(f"  My stats count: {len(my_stats)}")
        logger.info(f"  Opp stats count: {len(opp_stats)}")
        
        if my_stats:
            logger.info(f"  My stats sample: {json.dumps(my_stats, indent=2)[:500]}")
        else:
            logger.warning("  ⚠ My stats is EMPTY - this is the problem!")
            
        if opp_stats:
            logger.info(f"  Opp stats sample: {json.dumps(opp_stats, indent=2)[:500]}")
        else:
            logger.warning("  ⚠ Opp stats is EMPTY - this is the problem!")
        
        # Check for all zeros
        all_zero = all(v == 0 or v == 0.0 for v in my_stats.values() if isinstance(v, (int, float)))
        if all_zero and len(my_stats) > 0:
            logger.warning("  ⚠ All my stats are zero!")
        
        return matchup_stats
    except Exception as e:
        logger.error(f"✗ Error fetching matchup stats: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_raw_scoreboard_response(client):
    """Test raw scoreboard response from Yahoo."""
    try:
        logger.info("\n=== Testing Raw Scoreboard Response ===")
        
        # Get raw response
        path = f"league/{client.league_key}/scoreboard"
        data = client._get(path)
        
        logger.info(f"  Raw response keys: {list(data.keys())}")
        
        if 'fantasy_content' in data:
            fc = data['fantasy_content']
            logger.info(f"  Fantasy content keys: {list(fc.keys())}")
            
            if 'league' in fc:
                league = fc['league']
                logger.info(f"  League data type: {type(league)}")
                if isinstance(league, list) and len(league) > 1:
                    logger.info(f"  League[1] keys: {list(league[1].keys()) if isinstance(league[1], dict) else 'N/A'}")
        
        return data
    except Exception as e:
        logger.error(f"✗ Error fetching raw scoreboard: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    logger.info("Starting Scoreboard Diagnostics")
    logger.info("=" * 50)
    
    # Test connection
    client = test_yahoo_connection()
    if not client:
        logger.error("\n❌ CRITICAL: Cannot connect to Yahoo API")
        logger.error("Possible causes:")
        logger.error("  - YAHOO_CLIENT_ID or YAHOO_CLIENT_SECRET not set")
        logger.error("  - YAHOO_REFRESH_TOKEN expired or invalid")
        logger.error("  - Network connectivity issues")
        return 1
    
    # Test league
    if not test_get_league(client):
        logger.warning("\n⚠️ Could not fetch league data")
    
    # Test raw response
    raw_data = test_raw_scoreboard_response(client)
    
    # Test scoreboard
    scoreboard = test_get_scoreboard(client)
    if not scoreboard:
        logger.error("\n❌ Could not fetch scoreboard")
    
    # Test matchup stats
    matchup_stats = test_get_matchup_stats(client)
    if not matchup_stats:
        logger.error("\n❌ Could not fetch matchup stats")
    
    # Final diagnosis
    logger.info("\n" + "=" * 50)
    logger.info("DIAGNOSIS SUMMARY")
    logger.info("=" * 50)
    
    if matchup_stats:
        my_stats = matchup_stats.get('my_stats', {})
        opp_stats = matchup_stats.get('opp_stats', {})
        
        if not my_stats and not opp_stats:
            logger.error("❌ BOTH my_stats AND opp_stats are empty!")
            logger.error("   This means Yahoo API returned no stat data.")
            logger.error("   Possible causes:")
            logger.error("     - Week number is incorrect (no data for that week)")
            logger.error("     - Season hasn't started yet")
            logger.error("     - League is not active")
            logger.error("     - Yahoo API response structure changed")
        elif len(my_stats) == 0:
            logger.error("❌ my_stats is empty but opp_stats has data")
            logger.error("   Possible causes:")
            logger.error("     - Cannot find user's team in scoreboard")
            logger.error("     - Team key mismatch")
        elif all(v == 0 or v == 0.0 for v in my_stats.values() if isinstance(v, (int, float))):
            logger.error("❌ All stats are zero!")
            logger.error("   Possible causes:")
            logger.error("     - Yahoo has no stats recorded for this week yet")
            logger.error("     - Stat ID mapping is incorrect")
            logger.error("     - Yahoo API returned zeros")
        else:
            logger.info("✓ Stats appear to have valid data")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
