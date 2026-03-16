"""
Selection Sunday pipeline: run bracket simulations and generate all reports.

Usage:
    # Quick first-pass (10k sims, ~1-2 min)
    python scripts/run_bracket_sims.py --bracket data/bracket_2026.json --quick

    # Full run (50k sims, ~5 min, 4 workers)
    python scripts/run_bracket_sims.py --bracket data/bracket_2026.json --sims 50000 --workers 4

    # With futures odds for value bet detection
    python scripts/run_bracket_sims.py --bracket data/bracket_2026.json --futures data/futures_odds_2026.json

Bracket JSON format: see data/bracket_template_2026.json
Futures JSON format: see data/futures_odds_template.json

Outputs go to: outputs/tournament_2026/
    sim_results.json         - Full probability distributions
    championship_probs.csv   - Championship probabilities all 68 teams
    final_four_probs.csv     - F4/E8/S16 probabilities
    cinderella_rankings.csv  - Double-digit seed deep run probabilities
    upset_heatmap_r64.csv    - First round upset risk for all 32 matchups
    futures_value.csv        - Value bets vs market (only if --futures provided)
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.tournament.matchup_predictor import TournamentTeam
from backend.tournament.bracket_simulator import run_monte_carlo
from backend.tournament.cinderella_tracker import (
    cinderella_rankings,
    upset_heat_map,
    format_cinderella_table,
    format_upset_heat_map,
)
from backend.tournament.futures_analyzer import analyze_futures, format_futures_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/tournament_2026")


def load_bracket(path: str) -> dict:
    """
    Load bracket from JSON file and return {region: [TournamentTeam, ...]}.

    Expected JSON format:
    {
        "south": [
            {
                "seed": 1,
                "name": "Auburn",
                "composite_rating": 25.3,
                "kp_adj_em": 27.1,
                "bt_adj_em": 23.8,
                "pace": 70.1,
                "three_pt_rate": 0.38,
                "def_efg_pct": 0.48,
                "conference": "SEC",
                "tournament_exp": 0.75
            },
            ...  (16 teams per region)
        ],
        "east": [...],
        "west": [...],
        "midwest": [...]
    }
    """
    with open(path) as f:
        raw = json.load(f)

    bracket = {}
    total_teams = 0
    for region, teams in raw.items():
        if region.startswith("_"):
            continue  # Skip metadata keys like "_instructions"
        bracket[region] = [
            TournamentTeam(
                name=t["name"],
                seed=t["seed"],
                region=region,
                composite_rating=t.get("composite_rating", 0.0),
                kp_adj_em=t.get("kp_adj_em"),
                bt_adj_em=t.get("bt_adj_em"),
                pace=t.get("pace", 68.0),
                three_pt_rate=t.get("three_pt_rate", 0.35),
                def_efg_pct=t.get("def_efg_pct", 0.50),
                conference=t.get("conference", ""),
                tournament_exp=t.get("tournament_exp", 0.70),
            )
            for t in teams
        ]
        total_teams += len(bracket[region])

    logger.info(
        "Loaded bracket: %d teams across %d regions (%s)",
        total_teams, len(bracket), ", ".join(bracket.keys())
    )
    return bracket


def write_csv(rows: list, fieldnames: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s (%d rows)", path, len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run March Madness Monte Carlo bracket simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--bracket", required=True,
        help="Path to bracket JSON file (see data/bracket_template_2026.json)"
    )
    parser.add_argument(
        "--sims", type=int, default=50000,
        help="Number of simulations (default: 50000)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel worker processes (default: 4)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--futures",
        help="Path to futures odds JSON file for value bet detection"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 10k sims, 2 workers (first-pass analysis)"
    )
    parser.add_argument(
        "--output", default=str(OUTPUT_DIR),
        help="Output directory (default: outputs/tournament_2026)"
    )
    args = parser.parse_args()

    if args.quick:
        args.sims = 10000
        args.workers = 2
        logger.info("Quick mode: 10,000 sims, 2 workers")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # 1. Load bracket
    # -------------------------------------------------------------------
    bracket = load_bracket(args.bracket)
    all_teams = [
        {"name": t.name, "seed": t.seed, "region": t.region}
        for region_teams in bracket.values()
        for t in region_teams
    ]

    # -------------------------------------------------------------------
    # 2. Run Monte Carlo simulations
    # -------------------------------------------------------------------
    logger.info("Starting %d simulations across %d workers...", args.sims, args.workers)
    t0 = time.time()
    results = run_monte_carlo(
        bracket,
        n_sims=args.sims,
        n_workers=args.workers,
        base_seed=args.seed,
    )
    elapsed = time.time() - t0
    logger.info("Simulations complete in %.1fs (%.0f sims/sec)", elapsed, args.sims / elapsed)

    # -------------------------------------------------------------------
    # 3. Save full results JSON
    # -------------------------------------------------------------------
    sim_json = {
        "n_sims": results.n_sims,
        "elapsed_seconds": round(elapsed, 1),
        "avg_championship_margin": round(results.avg_championship_margin, 1),
        "avg_upsets_per_tournament": round(results.avg_upsets_per_tournament, 1),
        "championship": dict(sorted(results.championship.items(), key=lambda x: -x[1])),
        "final_four": dict(sorted(results.final_four.items(), key=lambda x: -x[1])),
        "elite_eight": dict(sorted(results.elite_eight.items(), key=lambda x: -x[1])),
        "sweet_sixteen": dict(sorted(results.sweet_sixteen.items(), key=lambda x: -x[1])),
        "round_of_32": dict(sorted(results.round_of_32.items(), key=lambda x: -x[1])),
    }
    json_path = output_dir / "sim_results.json"
    with open(json_path, "w") as f:
        json.dump(sim_json, f, indent=2)
    logger.info("Saved %s", json_path)

    # -------------------------------------------------------------------
    # 4. Championship probabilities CSV
    # -------------------------------------------------------------------
    champ_rows = [
        {
            "rank": i,
            "team": t,
            "championship_prob": round(p, 4),
            "championship_pct": f"{p * 100:.1f}%",
            "final_four_prob": round(results.final_four.get(t, 0), 4),
            "elite_eight_prob": round(results.elite_eight.get(t, 0), 4),
        }
        for i, (t, p) in enumerate(
            sorted(results.championship.items(), key=lambda x: -x[1]), 1
        )
    ]
    write_csv(
        champ_rows,
        ["rank", "team", "championship_prob", "championship_pct", "final_four_prob", "elite_eight_prob"],
        output_dir / "championship_probs.csv",
    )

    # -------------------------------------------------------------------
    # 5. Full milestone probabilities CSV
    # -------------------------------------------------------------------
    all_team_names = set(results.championship) | set(results.final_four) | set(results.sweet_sixteen)
    milestone_rows = [
        {
            "team": t,
            "p_round_of_32": round(results.round_of_32.get(t, 0), 4),
            "p_sweet_sixteen": round(results.sweet_sixteen.get(t, 0), 4),
            "p_elite_eight": round(results.elite_eight.get(t, 0), 4),
            "p_final_four": round(results.final_four.get(t, 0), 4),
            "p_runner_up": round(results.runner_up.get(t, 0), 4),
            "p_championship": round(results.championship.get(t, 0), 4),
        }
        for t in sorted(all_team_names, key=lambda x: -results.championship.get(x, 0))
    ]
    write_csv(
        milestone_rows,
        ["team", "p_round_of_32", "p_sweet_sixteen", "p_elite_eight",
         "p_final_four", "p_runner_up", "p_championship"],
        output_dir / "final_four_probs.csv",
    )

    # -------------------------------------------------------------------
    # 6. Cinderella rankings CSV
    # -------------------------------------------------------------------
    cinderellas = cinderella_rankings(results, all_teams)
    cind_rows = [
        {
            "rank": i,
            "team": c.team,
            "seed": c.seed,
            "region": c.region,
            "p_round_of_32": round(c.p_round_of_32, 4),
            "p_sweet_sixteen": round(c.p_sweet_sixteen, 4),
            "p_elite_eight": round(c.p_elite_eight, 4),
            "p_final_four": round(c.p_final_four, 4),
            "cinderella_score": round(c.cinderella_score, 2),
        }
        for i, c in enumerate(cinderellas, 1)
    ]
    write_csv(
        cind_rows,
        ["rank", "team", "seed", "region", "p_round_of_32", "p_sweet_sixteen",
         "p_elite_eight", "p_final_four", "cinderella_score"],
        output_dir / "cinderella_rankings.csv",
    )

    # -------------------------------------------------------------------
    # 7. Upset heat map CSV (R64)
    # -------------------------------------------------------------------
    upsets = upset_heat_map(bracket)
    upset_rows = [
        {
            "region": m.region,
            "favorite": m.favorite,
            "favorite_seed": m.favorite_seed,
            "underdog": m.underdog,
            "underdog_seed": m.underdog_seed,
            "upset_probability": m.upset_probability,
            "margin_estimate": m.margin_estimate,
            "risk_level": m.risk_level,
        }
        for m in upsets
    ]
    write_csv(
        upset_rows,
        ["region", "favorite", "favorite_seed", "underdog", "underdog_seed",
         "upset_probability", "margin_estimate", "risk_level"],
        output_dir / "upset_heatmap_r64.csv",
    )

    # -------------------------------------------------------------------
    # 8. Futures value bets (optional)
    # -------------------------------------------------------------------
    value_bets = []
    if args.futures:
        with open(args.futures) as f:
            futures_odds = json.load(f)
        value_bets = analyze_futures(results, futures_odds)
        fut_rows = [
            {
                "team": b.team,
                "market": b.market,
                "american_odds": b.american_odds,
                "fair_american_odds": b.fair_american_odds,
                "model_prob": round(b.model_prob, 4),
                "market_implied_prob": round(b.market_implied_prob, 4),
                "edge_pct": b.edge_pct,
                "ev_pct": b.ev_pct,
                "recommendation": b.recommendation,
            }
            for b in value_bets
        ]
        write_csv(
            fut_rows,
            ["team", "market", "american_odds", "fair_american_odds", "model_prob",
             "market_implied_prob", "edge_pct", "ev_pct", "recommendation"],
            output_dir / "futures_value.csv",
        )

    # -------------------------------------------------------------------
    # 9. Console summary
    # -------------------------------------------------------------------
    print()
    print("=" * 65)
    print(f"  MARCH MADNESS 2026 PROJECTIONS  ({args.sims:,} simulations)")
    print("=" * 65)

    print("\nCHAMPIONSHIP PROBABILITIES (Top 15):")
    sorted_champ = sorted(results.championship.items(), key=lambda x: -x[1])
    for i, (team, prob) in enumerate(sorted_champ[:15], 1):
        bar = "#" * int(prob * 200)
        print(f"  {i:2}. {team:<28} {prob * 100:5.1f}%  {bar}")

    print(f"\nTOP CINDERELLA CANDIDATES (seed >= 10):")
    if cinderellas:
        print(format_cinderella_table(cinderellas, top_n=6))
    else:
        print("  None found.")

    print(f"\nHIGH UPSET RISK MATCHUPS (R64):")
    high_risk = [m for m in upsets if m.risk_level == "HIGH"]
    for m in high_risk[:8]:
        print(
            f"  {m.region:<8} #{m.favorite_seed} {m.favorite:<22} vs "
            f"#{m.underdog_seed} {m.underdog:<22}  "
            f"{m.upset_probability * 100:.1f}% upset"
        )

    if value_bets:
        print(f"\nFUTURES VALUE BETS ({len(value_bets)} found):")
        print(format_futures_table(value_bets))

    print(f"\nAvg championship margin: {results.avg_championship_margin:.1f} pts")
    print(f"Avg upsets per tournament: {results.avg_upsets_per_tournament:.1f}")
    print(f"\nOutput files: {output_dir}/")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
