# scripts/map_teams.py
import os
import logging
import inspect
from dotenv import load_dotenv

# 1. Load environment variables from .env in the root directory
load_dotenv()

from thefuzz import process  # Ensure you've run: pip install thefuzz
from backend.services.odds import OddsAPIClient
from backend.services.ratings import get_ratings_service

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_mapping_exercise():
    """
    Automated script to find mismatches between The Odds API and KenPom.
    Outputs a dictionary to be pasted into TEAM_NAME_MAPPING in ratings.py.
    """
    # 1. Initialize Clients
    try:
        odds_client = OddsAPIClient()
        ratings_service = get_ratings_service()
    except ValueError as e:
        print(f"❌ Error: {e}")
        return

    print("🔄 Fetching fresh data from KenPom and The Odds API...")
    
    # Fetch KenPom teams (the "Official" names we want to match to)
    all_ratings = ratings_service.get_all_ratings(use_cache=False)
    kenpom_teams = list(all_ratings['kenpom'].keys())
    
    if not kenpom_teams:
        print("❌ Could not fetch KenPom ratings. Check your KENPOM_API_KEY.")
        return

    # Fetch current games from The Odds API
    odds_games = odds_client.get_todays_games()
    odds_teams = set()
    for game in odds_games:
        odds_teams.add(game['home_team'])
        odds_teams.add(game['away_team'])

    # 2. Compare and Map
    new_mappings = {}
    print(f"🔍 Analyzing {len(odds_teams)} teams from The Odds API...")
    print("-" * 50)
    
    # Check if normalize_name requires valid_choices
    sig = inspect.signature(ratings_service.normalize_name)
    needs_valid_choices = 'valid_choices' in sig.parameters

    for team in sorted(list(odds_teams)):
        # Call normalize_name based on its actual signature to avoid TypeErrors
        if needs_valid_choices:
            normalized = ratings_service.normalize_name(team, valid_choices=kenpom_teams)
        else:
            normalized = ratings_service.normalize_name(team)
        
        # If the normalized name is already in KenPom, skip it
        if normalized in kenpom_teams:
            continue
            
        # Fuzzy match if not already mapped correctly
        match_result = process.extractOne(team, kenpom_teams)
        
        if match_result:
            match_name, score = match_result
            if score >= 90:
                new_mappings[team] = match_name
                print(f"✅ Suggestion: '{team}' -> '{match_name}' (Score: {score})")
            elif score >= 70:
                print(f"⚠️  Review needed: '{team}' (Best guess: '{match_name}', Score: {score})")
            else:
                print(f"❓ No clear match for: '{team}' (Closest: '{match_name}', Score: {score})")

    # 3. Output results
    print("-" * 50)
    if new_mappings:
        print("\n🚀 COPY AND PASTE THESE INTO TEAM_NAME_MAPPING (ratings.py):")
        for k, v in new_mappings.items():
            print(f'    "{k}": "{v}",')
    else:
        print("\n🙌 No new mappings needed for today's games!")

if __name__ == "__main__":
    run_mapping_exercise()