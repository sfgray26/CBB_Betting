"""
Selection Sunday helper: pull team ratings from DB and generate bracket JSON.

Run AFTER you've filled in team names and seeds. This script looks up each team's
KenPom and BartTorvik ratings from the database and computes the composite_rating.

Usage:
    # Step 1: Edit data/bracket_template_2026.json with team names and seeds
    # Step 2: Run this script to populate ratings from DB
    python scripts/build_bracket_from_db.py --input data/bracket_template_2026.json

    # Step 3: Run simulations
    python scripts/run_bracket_sims.py --bracket data/bracket_2026.json --quick

Output: data/bracket_2026.json (bracket_template with ratings filled in)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import Session
from backend.models import engine, TeamProfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# V9.1 rating weights (KenPom 51%, BartTorvik 49%) — matches analysis.py 2-source mode
WEIGHT_KP = 0.51
WEIGHT_BT = 0.49

SEASON_YEAR = 2026


def lookup_team_ratings(session: Session, team_name: str) -> dict:
    """
    Look up a team's KenPom and BartTorvik ratings from the TeamProfile table.

    Returns dict with kp_adj_em, bt_adj_em, pace, three_pt_rate, def_efg_pct.
    Returns zeros if team not found (will need manual override).
    """
    # Try KenPom profile
    kp_profile = (
        session.query(TeamProfile)
        .filter(
            TeamProfile.team_name == team_name,
            TeamProfile.season_year == SEASON_YEAR,
            TeamProfile.source == "kenpom",
        )
        .first()
    )

    # Try BartTorvik profile
    bt_profile = (
        session.query(TeamProfile)
        .filter(
            TeamProfile.team_name == team_name,
            TeamProfile.season_year == SEASON_YEAR,
            TeamProfile.source == "barttorvik",
        )
        .first()
    )

    if kp_profile is None and bt_profile is None:
        logger.warning("No ratings found for '%s' — will use zeros (manual override needed)", team_name)
        return {}

    kp_adj_em = kp_profile.adj_em if kp_profile else None
    bt_adj_em = bt_profile.adj_em if bt_profile else None

    # Compute composite rating (same as analysis.py 2-source mode)
    ratings = []
    weights = []
    if kp_adj_em is not None:
        ratings.append(kp_adj_em * WEIGHT_KP)
        weights.append(WEIGHT_KP)
    if bt_adj_em is not None:
        ratings.append(bt_adj_em * WEIGHT_BT)
        weights.append(WEIGHT_BT)

    if weights:
        total_weight = sum(weights)
        composite = sum(ratings) / total_weight * sum(weights)
    else:
        composite = 0.0

    # Style profile from BartTorvik (primary source for four factors)
    pace = bt_profile.pace if bt_profile and bt_profile.pace else 68.0
    three_pt_rate = bt_profile.three_par if bt_profile and bt_profile.three_par else 0.35
    def_efg_pct = bt_profile.def_efg_pct if bt_profile and bt_profile.def_efg_pct else 0.50

    return {
        "kp_adj_em": round(kp_adj_em, 2) if kp_adj_em else None,
        "bt_adj_em": round(bt_adj_em, 2) if bt_adj_em else None,
        "composite_rating": round(composite, 2),
        "pace": round(pace, 1),
        "three_pt_rate": round(three_pt_rate, 3),
        "def_efg_pct": round(def_efg_pct, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate bracket JSON with DB ratings")
    parser.add_argument(
        "--input", default="data/bracket_template_2026.json",
        help="Input bracket JSON with team names and seeds filled in"
    )
    parser.add_argument(
        "--output", default="data/bracket_2026.json",
        help="Output bracket JSON with ratings populated"
    )
    args = parser.parse_args()

    with open(args.input) as f:
        bracket = json.load(f)

    total_teams = 0
    found = 0
    missing = []

    with Session(engine) as session:
        for region, teams in bracket.items():
            if region.startswith("_"):
                continue

            for team in teams:
                name = team.get("name", "")
                if name in ("FILL_IN", "", None):
                    logger.warning("Region %s seed %d: team name not filled in", region, team.get("seed"))
                    continue

                total_teams += 1
                ratings = lookup_team_ratings(session, name)

                if ratings:
                    team.update(ratings)
                    found += 1
                    logger.info(
                        "%s (#%d %s): composite=%.1f kp=%.1f bt=%.1f",
                        name, team["seed"], region,
                        ratings.get("composite_rating", 0),
                        ratings.get("kp_adj_em") or 0,
                        ratings.get("bt_adj_em") or 0,
                    )
                else:
                    missing.append(f"#{team['seed']} {name} ({region})")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(bracket, f, indent=2)

    logger.info("\n--- Summary ---")
    logger.info("Teams processed: %d", total_teams)
    logger.info("Ratings found:   %d", found)
    logger.info("Missing:         %d", len(missing))

    if missing:
        logger.warning("These teams need manual composite_rating override:")
        for m in missing:
            logger.warning("  %s", m)

    print(f"\nBracket with ratings saved to: {output_path}")
    if missing:
        print(f"\nWARNING: {len(missing)} teams missing from DB — edit {output_path} manually:")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()
