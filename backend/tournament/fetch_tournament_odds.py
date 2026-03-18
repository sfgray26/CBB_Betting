"""
Fetch live moneyline odds for NCAA tournament R64 games and write them
into data/bracket_2026.json as `market_ml` fields.

Usage:
    python -m backend.tournament.fetch_tournament_odds
    # or with a custom bracket file:
    python -m backend.tournament.fetch_tournament_odds --bracket data/bracket_2026.json

The script calls The Odds API (basketball_ncaab, h2h markets), fuzzy-matches
team names from the bracket JSON to API game entries, and writes:
    team["market_ml"] = <American moneyline integer>

Only R64 matchups are populated (seeds 1-16 per region).  Later-round lines
aren't available until pairings are set, and predict_game() handles None
gracefully (skips market blend when market_ml is None).

Requires: THE_ODDS_API_KEY in environment (or .env file).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup — allow running from project root without install
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from backend.services.odds import OddsAPIClient
from backend.services.team_mapping import TeamMapper

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# R64 pairing (standard NCAA bracket structure)
# Seed pairs that meet in R64: 1v16, 2v15, 3v14, 4v13, 5v12, 6v11, 7v10, 8v9
# ---------------------------------------------------------------------------
R64_PAIRS: List[Tuple[int, int]] = [
    (1, 16), (2, 15), (3, 14), (4, 13),
    (5, 12), (6, 11), (7, 10), (8, 9),
]


def _build_seed_lookup(bracket: Dict) -> Dict[str, Dict]:
    """
    Return {team_name_lower: team_dict} for all 64 bracket teams.
    """
    lookup = {}
    for region_teams in bracket.values():
        if not isinstance(region_teams, list):
            continue
        for team in region_teams:
            key = team["name"].lower().strip()
            lookup[key] = team
    return lookup


def _fuzzy_match(api_name: str, bracket_names: List[str]) -> Optional[str]:
    """
    Simple fuzzy match: first try exact (case-insensitive), then
    check if any bracket name is a substring of the API name or vice-versa.
    Returns the matched bracket name or None.
    """
    api_lower = api_name.lower().strip()
    for bn in bracket_names:
        if bn == api_lower:
            return bn
    # substring match — handles "St. John's" vs "St Johns" etc.
    for bn in bracket_names:
        bn_words = set(bn.split())
        api_words = set(api_lower.split())
        # require at least 2 common words (avoids false positives on short names)
        common = bn_words & api_words
        if len(common) >= 2 or (len(bn_words) == 1 and bn_words == api_words):
            return bn
    # try TeamMapper if available
    try:
        mapper = TeamMapper()
        mapped = mapper.normalize(api_name)
        mapped_lower = mapped.lower().strip()
        if mapped_lower in bracket_names:
            return mapped_lower
    except Exception:
        pass
    return None


def fetch_and_write_odds(bracket_path: str) -> int:
    """
    Fetch live moneylines and write them into bracket JSON.
    Returns the number of teams updated.
    """
    bracket_path = Path(bracket_path)
    with open(bracket_path) as f:
        bracket = json.load(f)

    client = OddsAPIClient()
    games: List[Dict] = client.get_cbb_odds(markets="h2h")
    if not games:
        logger.warning("No games returned from Odds API — is it tournament time?")
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
            logger.debug("No bracket match: %s vs %s", home, away)
            continue

        home_team = seed_lookup[home_key]
        away_team = seed_lookup[away_key]

        # Check if this is an R64 matchup
        pair = tuple(sorted([home_team["seed"], away_team["seed"]]))
        if pair not in [tuple(sorted(p)) for p in R64_PAIRS]:
            logger.debug(
                "Skipping non-R64 matchup: %s (#%d) vs %s (#%d)",
                home_team["name"], home_team["seed"],
                away_team["name"], away_team["seed"],
            )
            continue

        # Extract best moneylines from bookmakers (prefer Pinnacle, then DraftKings)
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
                break  # found preferred book, stop

        if home_ml is None or away_ml is None:
            logger.warning("Incomplete lines for %s vs %s", home, away)
            continue

        home_team["market_ml"] = int(home_ml)
        away_team["market_ml"] = int(away_ml)
        updated += 2

        logger.info(
            "Updated: %s (%+d) vs %s (%+d)",
            home_team["name"], home_ml, away_team["name"], away_ml,
        )

    with open(bracket_path, "w") as f:
        json.dump(bracket, f, indent=2)

    logger.info("Wrote %d market_ml values to %s", updated, bracket_path)
    return updated


def main():
    parser = argparse.ArgumentParser(description="Fetch tournament moneylines from The Odds API")
    parser.add_argument(
        "--bracket",
        default=str(PROJECT_ROOT / "data" / "bracket_2026.json"),
        help="Path to bracket JSON file (default: data/bracket_2026.json)",
    )
    args = parser.parse_args()

    api_key = os.getenv("THE_ODDS_API_KEY")
    if not api_key:
        logger.error("THE_ODDS_API_KEY not set — cannot fetch odds")
        sys.exit(1)

    n = fetch_and_write_odds(args.bracket)
    logger.info("Done. %d teams updated with live moneylines.", n)


if __name__ == "__main__":
    main()
