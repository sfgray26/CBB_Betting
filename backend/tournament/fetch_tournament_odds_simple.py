#!/usr/bin/env python3
"""
Fetch live moneyline odds for NCAA tournament R64 games and write them
into data/bracket_2026.json as `market_ml` fields.

Simple version using requests only (no httpx dependency).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# R64 pairings
R64_PAIRS: List[Tuple[int, int]] = [
    (1, 16), (2, 15), (3, 14), (4, 13),
    (5, 12), (6, 11), (7, 10), (8, 9),
]

API_KEY = os.getenv("THE_ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4"


def fetch_odds() -> List[Dict]:
    """Fetch CBB odds from The Odds API using requests."""
    import requests
    
    if not API_KEY:
        logger.error("THE_ODDS_API_KEY not set")
        return []
    
    url = f"{BASE_URL}/sports/basketball_ncaab/odds"
    params = {
        "apiKey": API_KEY,
        "markets": "h2h",
        "oddsFormat": "american",
    }
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"Fetched {len(data)} games from The Odds API")
        return data
    except Exception as e:
        logger.error(f"Failed to fetch odds: {e}")
        return []


def _build_seed_lookup(bracket: Dict) -> Dict[str, Dict]:
    """Return {team_name_lower: team_dict} for all 64 bracket teams."""
    lookup = {}
    for region_teams in bracket.values():
        if not isinstance(region_teams, list):
            continue
        for team in region_teams:
            key = team["name"].lower().strip()
            lookup[key] = team
    return lookup


def _fuzzy_match(api_name: str, bracket_names: List[str]) -> Optional[str]:
    """Simple fuzzy match for team names."""
    api_lower = api_name.lower().strip()
    
    # Exact match
    if api_lower in bracket_names:
        return api_lower
    
    # Substring match
    for bn in bracket_names:
        bn_words = set(bn.split())
        api_words = set(api_lower.split())
        common = bn_words & api_words
        if len(common) >= 2 or (len(bn_words) == 1 and bn_words == api_words):
            return bn
    
    return None


def fetch_and_write_odds(bracket_path: str) -> int:
    """Fetch live moneylines and write them into bracket JSON."""
    bracket_path = Path(bracket_path)
    with open(bracket_path) as f:
        bracket = json.load(f)
    
    games = fetch_odds()
    if not games:
        logger.warning("No games returned from Odds API")
        return 0
    
    seed_lookup = _build_seed_lookup(bracket)
    bracket_names = list(seed_lookup.keys())
    
    updated = 0
    
    for game in games:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        
        home_key = _fuzzy_match(home, bracket_names)
        away_key = _fuzzy_match(away, bracket_names)
        
        if home_key is None or away_key is None:
            logger.debug(f"No bracket match: {home} vs {away}")
            continue
        
        home_team = seed_lookup[home_key]
        away_team = seed_lookup[away_key]
        
        # Check if R64 matchup
        pair = tuple(sorted([home_team["seed"], away_team["seed"]]))
        if pair not in [tuple(sorted(p)) for p in R64_PAIRS]:
            logger.debug(f"Skipping non-R64: {home_team['name']} vs {away_team['name']}")
            continue
        
        # Extract moneylines
        home_ml: Optional[int] = None
        away_ml: Optional[int] = None
        
        bookmakers = game.get("bookmakers", [])
        preferred = ["pinnacle", "draftkings", "betmgm", "fanduel"]
        ordered = sorted(
            bookmakers,
            key=lambda b: preferred.index(b.get("key", "").lower())
            if b.get("key", "").lower() in preferred else 99,
        )
        
        for book in ordered:
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    if outcome["name"].lower().strip() == home.lower().strip():
                        home_ml = outcome.get("price")
                    elif outcome["name"].lower().strip() == away.lower().strip():
                        away_ml = outcome.get("price")
            if home_ml is not None and away_ml is not None:
                break
        
        if home_ml is None or away_ml is None:
            logger.warning(f"Incomplete lines for {home} vs {away}")
            continue
        
        home_team["market_ml"] = int(home_ml)
        away_team["market_ml"] = int(away_ml)
        updated += 2
        
        logger.info(f"Updated: {home_team['name']} ({home_ml:+d}) vs {away_team['name']} ({away_ml:+d})")
    
    with open(bracket_path, "w") as f:
        json.dump(bracket, f, indent=2)
    
    logger.info(f"Wrote {updated} market_ml values to {bracket_path}")
    return updated


def main():
    parser = argparse.ArgumentParser(description="Fetch tournament moneylines")
    parser.add_argument(
        "--bracket",
        default=str(PROJECT_ROOT / "data" / "bracket_2026.json"),
        help="Path to bracket JSON file",
    )
    args = parser.parse_args()
    
    if not API_KEY:
        logger.error("THE_ODDS_API_KEY not set — cannot fetch odds")
        sys.exit(1)
    
    n = fetch_and_write_odds(args.bracket)
    logger.info(f"Done. {n} teams updated with live moneylines.")


if __name__ == "__main__":
    main()
