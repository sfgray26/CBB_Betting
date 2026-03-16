"""
Tournament futures value analyzer.

Identifies positive EV bets in championship and milestone futures markets
by comparing model-derived win probabilities against market-implied probabilities.

Thresholds (from BRACKET-001):
- EV >= 15%: BET
- EV >= 5%:  CONSIDER
- EV <  5%:  PASS (not returned)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from backend.tournament.bracket_simulator import SimulationResults

logger = logging.getLogger(__name__)

# Markets supported for futures analysis
SUPPORTED_MARKETS = ("championship", "final_four", "elite_eight", "sweet_sixteen")


@dataclass
class FuturesBet:
    """A potential futures value bet."""
    team: str
    market: str                  # "championship", "final_four", etc.
    model_prob: float            # Model-derived probability from simulation
    market_implied_prob: float   # Bookmaker's implied probability (raw, before vig)
    american_odds: int           # Market price in American format
    fair_american_odds: int      # Model's fair-value American odds
    edge_pct: float              # model_prob - market_implied_prob (percentage points)
    ev_pct: float                # Expected value as % of stake (positive = profitable)
    recommendation: str          # "BET", "CONSIDER", or "PASS"


def american_to_implied(odds: int) -> float:
    """Convert American odds to raw implied probability (does not remove vig)."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def implied_to_american(prob: float) -> int:
    """Convert probability to American odds representation."""
    prob = max(0.001, min(0.999, prob))
    if prob >= 0.5:
        return -int(round(prob / (1.0 - prob) * 100))
    else:
        return int(round((1.0 - prob) / prob * 100))


def calculate_ev(model_prob: float, american_odds: int) -> float:
    """
    Calculate expected value of a futures bet as fraction of stake.

    Returns:
        Positive = profitable (e.g., 0.15 means 15 cents per dollar wagered)
        Negative = losing bet
    """
    if american_odds > 0:
        profit_per_unit = american_odds / 100.0
    else:
        profit_per_unit = 100.0 / abs(american_odds)

    # EV = P(win) * profit - P(lose) * stake
    ev = (model_prob * profit_per_unit) - ((1.0 - model_prob) * 1.0)
    return ev  # per unit staked


def analyze_futures(
    sim_results: SimulationResults,
    market_odds: Dict[str, Dict[str, int]],
    min_ev_pct: float = 0.05,
) -> List[FuturesBet]:
    """
    Identify value bets in futures markets.

    Args:
        sim_results: Output from run_monte_carlo()
        market_odds: {team_name: {market: american_odds}}
            Example:
            {
                "Duke": {"championship": 1200, "final_four": 450},
                "Auburn": {"championship": 800, "final_four": 280},
            }
        min_ev_pct: Minimum EV to include a bet in results (default: 5%)

    Returns:
        List of FuturesBet objects sorted by EV descending (best value first)
    """
    value_bets: List[FuturesBet] = []

    for team, team_odds in market_odds.items():
        for market in SUPPORTED_MARKETS:
            if market not in team_odds:
                continue

            am_odds = team_odds[market]
            market_probs = getattr(sim_results, market, {})
            model_prob = market_probs.get(team, 0.0)

            if model_prob < 0.001:
                logger.debug("Skipping %s %s — model prob near zero (%.4f)", team, market, model_prob)
                continue

            market_implied = american_to_implied(am_odds)
            ev = calculate_ev(model_prob, am_odds)
            edge_pct = (model_prob - market_implied) * 100
            fair_odds = implied_to_american(model_prob)

            if ev < min_ev_pct:
                continue

            rec = "BET" if ev >= 0.15 else "CONSIDER"

            value_bets.append(FuturesBet(
                team=team,
                market=market,
                model_prob=model_prob,
                market_implied_prob=market_implied,
                american_odds=am_odds,
                fair_american_odds=fair_odds,
                edge_pct=round(edge_pct, 1),
                ev_pct=round(ev * 100, 1),
                recommendation=rec,
            ))

    value_bets.sort(key=lambda x: x.ev_pct, reverse=True)
    n_bet = sum(1 for b in value_bets if b.recommendation == "BET")
    logger.info(
        "Futures analysis complete: %d value bets (%d BET, %d CONSIDER)",
        len(value_bets), n_bet, len(value_bets) - n_bet,
    )
    return value_bets


def format_futures_table(bets: List[FuturesBet]) -> str:
    """Format futures value bets as a readable ASCII table for Discord/console output."""
    if not bets:
        return "No futures value bets found."

    header = f"{'Team':<25} {'Market':<14} {'Odds':>6} {'Fair':>6} {'Edge':>6} {'EV':>6} {'Rec':>8}"
    sep = "-" * len(header)
    rows = [header, sep]
    for b in bets:
        rows.append(
            f"{b.team:<25} {b.market:<14} {b.american_odds:>+6d} "
            f"{b.fair_american_odds:>+6d} {b.edge_pct:>+5.1f}% {b.ev_pct:>+5.1f}% "
            f"{b.recommendation:>8}"
        )
    return "\n".join(rows)
