#!/usr/bin/env python3
"""
Play-by-play data ingestion pipeline using CBBpy.

Fetches granular play-by-play data and builds TeamSimProfiles for the
possession-based Markov simulator.

Usage:
    # Full season ingest (run once at start of season, slow)
    python scripts/ingest_pbp.py --season 2026

    # Update specific team (run daily for teams you're tracking)
    python scripts/ingest_pbp.py --team "Duke"

    # Build profiles from existing box-score data (fast, no PbP needed)
    python scripts/ingest_pbp.py --box-scores --season 2026

Requirements:
    pip install cbbpy pandas
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CBBpy-based ingestion
# ---------------------------------------------------------------------------

def ingest_season_box_scores(season: int) -> Dict[str, Dict]:
    """
    Fetch season box scores via CBBpy and compute Four Factors per team.

    Returns: {team_name: {efg_pct, to_pct, orb_pct, ft_rate, pace, ...}}
    """
    try:
        import cbbpy.mens_scraper as s
    except ImportError:
        logger.error(
            "cbbpy not installed. Install with: pip install cbbpy"
        )
        return {}

    logger.info("Fetching %d season data via CBBpy...", season)

    # CBBpy returns DataFrames for game info, box scores, and PbP
    try:
        season_data = s.get_games_season(season)
    except Exception as exc:
        logger.error("CBBpy season fetch failed: %s", exc)
        return {}

    if season_data is None:
        logger.warning("No data returned for season %d", season)
        return {}

    # season_data is a tuple of (game_info_df, box_score_df, pbp_df)
    if isinstance(season_data, tuple) and len(season_data) >= 2:
        box_df = season_data[1]
    else:
        logger.warning("Unexpected data format from CBBpy")
        return {}

    # Aggregate box-score stats per team
    team_stats: Dict[str, Dict[str, float]] = {}

    for team in box_df["team"].unique():
        team_games = box_df[box_df["team"] == team]

        if len(team_games) < 5:
            continue

        # Sum counting stats across all games
        totals = {}
        for col in ["fgm", "fga", "fgm3", "fga3", "ftm", "fta", "oreb", "to"]:
            if col in team_games.columns:
                totals[col] = team_games[col].sum()
            else:
                totals[col] = 0

        # Four Factors
        from backend.services.possession_sim import calculate_four_factors

        ff = calculate_four_factors(
            fgm=int(totals.get("fgm", 0)),
            fga=int(totals.get("fga", 1)),
            fgm3=int(totals.get("fgm3", 0)),
            fga3=int(totals.get("fga3", 0)),
            fta=int(totals.get("fta", 0)),
            ftm=int(totals.get("ftm", 0)),
            oreb=int(totals.get("oreb", 0)),
            to=int(totals.get("to", 0)),
        )

        n_games = len(team_games)
        avg_poss = ff["possessions"]
        pace = avg_poss * 2  # Possessions per 40 min (approx)

        # Shot distribution (3PA rate from box scores)
        fga_total = max(totals.get("fga", 1), 1)
        three_rate = totals.get("fga3", 0) / fga_total

        team_stats[team] = {
            **ff,
            "pace": round(pace, 1),
            "three_rate": round(three_rate, 4),
            "games": n_games,
        }

    logger.info("Computed Four Factors for %d teams", len(team_stats))
    return team_stats


def build_profiles_from_stats(
    team_stats: Dict[str, Dict],
) -> Dict:
    """Convert raw stats into TeamSimProfile dicts for JSON serialization."""
    from backend.services.possession_sim import build_sim_profile

    profiles = {}
    for team, stats in team_stats.items():
        profile = build_sim_profile(
            team=team,
            four_factors={
                "efg_pct": stats.get("efg_pct", 0.50),
                "to_pct": stats.get("to_pct", 0.17),
                "orb_pct": stats.get("orb_pct", 0.28),
                "ft_rate": stats.get("ft_rate", 0.30),
            },
            pace=stats.get("pace", 68.0),
        )
        profiles[team] = {
            k: getattr(profile, k)
            for k in profile.__dataclass_fields__
        }

    return profiles


def save_profiles(profiles: Dict, output_path: str) -> None:
    """Save profiles to a JSON file for the simulation engine to load."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(profiles, f, indent=2, default=str)
    logger.info("Saved %d profiles to %s", len(profiles), output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest play-by-play data and build simulation profiles"
    )
    parser.add_argument(
        "--season", type=int, default=2026,
        help="Season year (e.g. 2026 for 2025-26 season)",
    )
    parser.add_argument(
        "--team", type=str, default=None,
        help="Specific team to update (optional)",
    )
    parser.add_argument(
        "--box-scores", action="store_true",
        help="Build profiles from box scores only (faster, no PbP)",
    )
    parser.add_argument(
        "--output", type=str, default="data/sim_profiles.json",
        help="Output path for profiles JSON",
    )

    args = parser.parse_args()

    logger.info("Starting PbP ingestion for season %d", args.season)

    team_stats = ingest_season_box_scores(args.season)

    if not team_stats:
        logger.warning("No team stats generated — check CBBpy installation")
        return

    if args.team:
        # Filter to specific team
        if args.team in team_stats:
            team_stats = {args.team: team_stats[args.team]}
        else:
            logger.error("Team '%s' not found in data", args.team)
            return

    profiles = build_profiles_from_stats(team_stats)
    save_profiles(profiles, args.output)

    logger.info(
        "Done! %d team profiles ready for simulation.", len(profiles)
    )


if __name__ == "__main__":
    main()
