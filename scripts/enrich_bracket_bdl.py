"""
Enrich bracket_2026.json with live BallDontLie data.

What this does:
  1. Fetches today's tournament odds -> patches market_ml on every team
  2. Fetches team season stats -> patches pace, three_pt_rate, def_efg_pct
  3. Fetches bracket API (GOAT) -> reports which games have been played / updates winners
  4. Writes enriched JSON back to data/bracket_2026.json

Usage:
    python scripts/enrich_bracket_bdl.py                  # dry-run, print only
    python scripts/enrich_bracket_bdl.py --write          # write to bracket_2026.json
    python scripts/enrich_bracket_bdl.py --date 2026-03-20 --write

Requires:
    BALLDONTLIE_API_KEY environment variable (or set in .env)
"""

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", encoding="utf-8")
except ImportError:
    pass

from backend.services.balldontlie import BallDontLieClient

BRACKET_PATH = Path(__file__).parent.parent / "data" / "bracket_2026.json"
REGIONS = ["east", "west", "south", "midwest"]


# ---------------------------------------------------------------------------
# Name normalization helpers
# ---------------------------------------------------------------------------

_MANUAL_NAME_MAP = {
    # BDL name -> bracket_2026.json name (add as needed)
    "Fla": "Florida",
    "Mich": "Michigan",
    "Mich St": "Michigan State",
    "Iowa St": "Iowa State",
    "Va": "Virginia",
    "Ala": "Alabama",
    "Ark": "Arkansas",
    "Gonz": "Gonzaga",
    "Pur": "Purdue",
    "Ariz": "Arizona",
    "Neb": "Nebraska",
    "Hou": "Houston",
    "Ill": "Illinois",
    "UConn": "UConn",
    "St John's": "St. John's",
    "Loyola Chi": "Loyola Chicago",
    "NCSU": "NC State",
    "Colo St": "Colorado State",
    "S Dakota St": "South Dakota State",
}


def normalize_name(bdl_name: str) -> str:
    return _MANUAL_NAME_MAP.get(bdl_name, bdl_name)


def find_team_in_bracket(bracket: dict, name: str) -> tuple:
    """Return (region, index) for team name, or (None, None)."""
    name_lc = name.lower()
    for region in REGIONS:
        for i, team in enumerate(bracket.get(region, [])):
            if team.get("name", "").lower() == name_lc:
                return region, i
    return None, None


# ---------------------------------------------------------------------------
# Step 1: Odds enrichment
# ---------------------------------------------------------------------------

def enrich_odds(client: BallDontLieClient, bracket: dict, target_date: str) -> int:
    """Fetch odds for target_date and patch market_ml. Returns patch count."""
    print(f"\n[Odds] Fetching odds for {target_date} ...")
    try:
        odds_records = client.get_odds_by_date(target_date)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return 0

    print(f"  Got {len(odds_records)} odds records")
    patched = 0

    # Group by game_id to match with team names
    game_odds: dict = {}
    for rec in odds_records:
        gid = rec.get("game_id") or rec.get("game", {}).get("id")
        if gid:
            game_odds.setdefault(gid, []).append(rec)

    # For each game, extract the best ML for home and away
    # We need team names from the games endpoint
    try:
        games = client.get_live_tournament_games(target_date)
    except Exception as exc:
        print(f"  ERROR fetching games: {exc}")
        return 0

    for game in games:
        home = game.get("home_team", {})
        away = game.get("visitor_team", {}) or game.get("away_team", {})
        home_name = normalize_name(home.get("name", "") or home.get("abbreviation", ""))
        away_name = normalize_name(away.get("name", "") or away.get("abbreviation", ""))
        gid = game.get("id")

        game_rec_list = game_odds.get(gid, [])
        if not game_rec_list:
            continue

        ml_data = client.extract_market_ml(game_rec_list, home_name, away_name)
        home_ml = ml_data.get("home_ml")
        away_ml = ml_data.get("away_ml")

        if home_ml is not None:
            region, idx = find_team_in_bracket(bracket, home_name)
            if region:
                bracket[region][idx]["market_ml"] = int(home_ml)
                patched += 1
                print(f"  {home_name}: market_ml={home_ml}")

        if away_ml is not None:
            region, idx = find_team_in_bracket(bracket, away_name)
            if region:
                bracket[region][idx]["market_ml"] = int(away_ml)
                patched += 1
                print(f"  {away_name}: market_ml={away_ml}")

    return patched


# ---------------------------------------------------------------------------
# Step 2: Team season stats enrichment
# ---------------------------------------------------------------------------

def enrich_team_stats(client: BallDontLieClient, bracket: dict) -> int:
    """Fetch season stats and patch pace/3pt_rate/def_efg_pct. Returns patch count."""
    print("\n[Stats] Fetching team season stats ...")
    try:
        stats_list = client.get_team_season_stats()
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return 0

    print(f"  Got stats for {len(stats_list)} teams")

    # Build lookup: BDL team_id -> stats row
    # Also build name lookup
    stats_by_name: dict = {}
    for row in stats_list:
        team = row.get("team", {})
        raw_name = team.get("name") or team.get("abbreviation") or ""
        norm = normalize_name(raw_name)
        stats_by_name[norm.lower()] = row

    patched = 0
    for region in REGIONS:
        for team in bracket.get(region, []):
            team_name = team.get("name", "")
            row = stats_by_name.get(team_name.lower())
            if not row:
                continue

            # Pace: BDL stores possessions if available; fall back to None
            pace = row.get("pace") or row.get("possessions")
            if pace and isinstance(pace, (int, float)) and 55 < pace < 85:
                team["pace"] = round(float(pace), 1)
                patched += 1

            # 3PT rate: fg3a / fga
            fg3a = row.get("fg3a") or row.get("fg3_pct_a")
            fga = row.get("fga")
            if fg3a and fga and fga > 0:
                three_rate = fg3a / fga
                if 0.15 < three_rate < 0.60:
                    team["three_pt_rate"] = round(three_rate, 3)
                    patched += 1
            elif row.get("fg3_pct"):
                # Some versions expose fg3_pct directly as attempt rate
                fg3_pct = float(row["fg3_pct"])
                if 0.15 < fg3_pct < 0.60:
                    team["three_pt_rate"] = round(fg3_pct, 3)
                    patched += 1

            # def_efg_pct: BDL doesn't have eFG% directly; skip if not available
            opp_fg_pct = row.get("opp_fg_pct")
            if opp_fg_pct and 0.35 < float(opp_fg_pct) < 0.65:
                team["def_efg_pct"] = round(float(opp_fg_pct), 3)
                patched += 1

    print(f"  Patched {patched} stat fields")
    return patched


# ---------------------------------------------------------------------------
# Step 3: Bracket results (mark played games)
# ---------------------------------------------------------------------------

def report_bracket_results(client: BallDontLieClient) -> None:
    """Fetch bracket API and print completed game results."""
    print("\n[Bracket] Fetching official tournament bracket ...")
    try:
        bracket_data = client.get_full_bracket()
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return

    for round_name, games in bracket_data.items():
        if not games:
            continue
        print(f"\n  {round_name.upper()} ({len(games)} games):")
        for g in games[:8]:  # preview first 8 per round
            home = g.get("home_team", {}).get("name", "?")
            away = (g.get("visitor_team") or g.get("away_team", {})).get("name", "?")
            home_score = g.get("home_team_score", "")
            away_score = g.get("visitor_team_score") or g.get("away_team_score") or ""
            status = g.get("status", "")
            if home_score and away_score:
                print(f"    {away} {away_score} @ {home} {home_score}  [{status}]")
            else:
                print(f"    {away} @ {home}  [{status}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Enrich bracket_2026.json with BDL data")
    parser.add_argument("--date", default=date.today().isoformat(),
                        help="Date to pull odds for (YYYY-MM-DD)")
    parser.add_argument("--write", action="store_true",
                        help="Write enriched JSON back to bracket_2026.json")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only run team stats enrichment (skip odds)")
    parser.add_argument("--odds-only", action="store_true",
                        help="Only run odds enrichment (skip stats)")
    parser.add_argument("--bracket-results", action="store_true",
                        help="Print completed tournament results from bracket API")
    args = parser.parse_args()

    # Load client
    try:
        client = BallDontLieClient()
        print(f"BallDontLie client initialized. Date target: {args.date}")
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Load bracket
    if not BRACKET_PATH.exists():
        print(f"ERROR: {BRACKET_PATH} not found")
        sys.exit(1)

    with open(BRACKET_PATH, encoding="utf-8") as f:
        bracket = json.load(f)

    total_patches = 0

    if args.bracket_results:
        report_bracket_results(client)
        return

    if not args.stats_only:
        total_patches += enrich_odds(client, bracket, args.date)

    if not args.odds_only:
        total_patches += enrich_team_stats(client, bracket)

    print(f"\nTotal fields patched: {total_patches}")

    if args.write and total_patches > 0:
        with open(BRACKET_PATH, "w", encoding="utf-8") as f:
            json.dump(bracket, f, indent=2, ensure_ascii=False)
        print(f"Written to {BRACKET_PATH}")
    elif args.write:
        print("Nothing to write (0 patches).")
    else:
        print("\n(Dry run — use --write to persist changes)")


if __name__ == "__main__":
    main()
