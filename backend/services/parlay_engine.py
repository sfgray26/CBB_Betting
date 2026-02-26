"""
Cross-game parlay builder for CBB Edge Analyzer.

Constructs mathematically optimal parlays from a slate of +EV straight bets.
Parlays compound edge but introduce extreme variance — use conservative sizing.
"""

import itertools
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Conservative Kelly divisor for parlays (4x more conservative than singles)
PARLAY_KELLY_DIVISOR = 4.0

# Minimum edge threshold for parlay inclusion
MIN_EDGE_THRESHOLD = 0.01  # 1%

# Minimum recommended units below which a parlay ticket is dropped.
# Prevents recommending a parlay when the portfolio cap leaves no meaningful
# capacity (e.g. 0.03 units = $0.30 on a $1000 bankroll).
MIN_PARLAY_UNITS = 0.05


def _american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return (american_odds / 100.0) + 1.0
    else:
        return (100.0 / abs(american_odds)) + 1.0


def _calculate_parlay_metrics(
    bets: List[Dict],
    win_probs: List[float],
    decimal_odds: List[float]
) -> Dict:
    """
    Calculate joint probability, parlay odds, and edge for a parlay combination.

    Args:
        bets: List of bet dictionaries from the slate
        win_probs: Side-aware conservative win probabilities for each leg.
            Each value is market_prob + edge_conservative from
            full_analysis['calculations'], which equals our_cover_lower
            for whichever side (home or away) the model is betting.
        decimal_odds: List of decimal odds for each leg

    Returns:
        Dict with joint_prob, parlay_odds, edge, kelly_fraction
    """
    # Joint probability (assuming independence)
    joint_prob = 1.0
    for p in win_probs:
        joint_prob *= p

    # Parlay decimal odds (product of all leg odds)
    parlay_odds = 1.0
    for odds in decimal_odds:
        parlay_odds *= odds

    # Edge calculation
    # EV = joint_prob * (parlay_odds - 1) - (1 - joint_prob)
    # Edge = joint_prob - implied_prob, but for parlays we use EV directly
    parlay_payout = parlay_odds - 1.0  # Net payout per unit
    expected_value = joint_prob * parlay_payout - (1.0 - joint_prob)
    edge = expected_value  # This is the EV per unit risked

    # Kelly fraction for parlay
    # f = (p * b - q) / b  where b = net_payout, p = joint_prob, q = 1 - p
    if parlay_payout > 0:
        kelly_full = (joint_prob * parlay_payout - (1.0 - joint_prob)) / parlay_payout
    else:
        kelly_full = 0.0

    # Apply conservative divisor
    kelly_fractional = kelly_full / PARLAY_KELLY_DIVISOR
    kelly_fractional = max(0.0, min(kelly_fractional, 0.05))  # Cap at 5%

    return {
        "joint_prob": joint_prob,
        "parlay_odds": parlay_odds,
        "parlay_payout": parlay_payout,
        "expected_value": expected_value,
        "edge": edge,
        "kelly_full": kelly_full,
        "kelly_fractional": kelly_fractional,
        "recommended_units": round(kelly_fractional * 100, 2),  # Convert to units
    }


def build_optimal_parlays(
    slate_bets: List[Dict],
    max_legs: int = 3,
    max_parlays: int = 10,
    remaining_capacity_dollars: Optional[float] = None,
    bankroll: float = 1000.0,
) -> List[Dict]:
    """
    Build mathematically optimal cross-game parlays from a slate of +EV bets.

    Args:
        slate_bets: List of bet dictionaries with fields:
            - game_id: int
            - pick: str
            - edge_conservative: float
            - full_analysis: dict with 'calculations' containing:
                - market_prob: float   (market's implied prob for the bet side)
                - edge_conservative: float  (our_cover_lower − market_prob)
                - bet_odds: float (American odds for the bet side)
        max_legs: Maximum number of legs per parlay (default 3)
        max_parlays: Maximum number of parlay tickets to return (default 10)
        remaining_capacity_dollars: Dollars of portfolio capacity still available
            after straight bets have been sized.  When provided, parlay
            recommended_units are scaled down proportionally so the aggregate
            parlay dollar exposure never exceeds this figure.  Pass 0 to
            suppress all parlay recommendations (no capacity left).
        bankroll: Starting bankroll in dollars (default 1000).  Used only when
            remaining_capacity_dollars is provided to convert units ↔ dollars.

    Returns:
        List of parlay dictionaries sorted by expected value, each containing:
            - legs: List[Dict] - the individual bets in the parlay
            - num_legs: int
            - joint_prob: float
            - parlay_odds: float (decimal)
            - parlay_american_odds: float
            - expected_value: float
            - edge: float
            - kelly_full: float
            - kelly_fractional: float
            - recommended_units: float
            - leg_summary: str - human-readable summary
    """
    logger.info("Building parlays from %d slate bets (max_legs=%d)", len(slate_bets), max_legs)

    # Filter for strong edges only
    qualified_bets = [
        b for b in slate_bets
        if (b.get("edge_conservative") or 0.0) > MIN_EDGE_THRESHOLD
    ]

    if len(qualified_bets) < 2:
        logger.info("Not enough qualified bets for parlays (need 2+, have %d)", len(qualified_bets))
        return []

    logger.info("Qualified bets for parlays: %d (edge > %.1f%%)", len(qualified_bets), MIN_EDGE_THRESHOLD * 100)

    parlays = []

    # Generate all combinations from 2-leg up to max_legs
    for num_legs in range(2, max_legs + 1):
        for combo in itertools.combinations(qualified_bets, num_legs):
            # Validation: ensure no two bets from same game_id (cross-game only)
            game_ids = [b.get("game_id") for b in combo]
            if len(game_ids) != len(set(game_ids)):
                continue  # Skip if duplicate game_ids

            # Extract probabilities and odds
            win_probs = []
            decimal_odds = []
            american_odds = []

            for bet in combo:
                # Reconstruct the side-aware conservative win probability.
                # The model writes market_prob and edge_conservative relative
                # to whichever side has edge (home or away):
                #   our_cover_lower = market_prob + edge_conservative
                # Using lower_ci_prob directly would always reflect the home
                # team's cover probability, producing wrong joint_prob for
                # any leg where bet_side == "away".
                calcs = bet.get("full_analysis", {}).get("calculations", {})
                market_prob = calcs.get("market_prob")
                edge_cons = calcs.get("edge_conservative")

                if market_prob is None or edge_cons is None:
                    logger.warning(
                        "Missing market_prob/edge_conservative in calculations "
                        "for bet '%s', skipping combo",
                        bet.get("pick"),
                    )
                    break

                true_leg_prob = market_prob + edge_cons
                if not (0.0 < true_leg_prob < 1.0):
                    logger.warning(
                        "true_leg_prob %.4f out of (0, 1) for bet '%s', skipping combo",
                        true_leg_prob, bet.get("pick"),
                    )
                    break

                win_probs.append(true_leg_prob)

                bet_odds = calcs.get("bet_odds", -110)
                american_odds.append(bet_odds)
                decimal_odds.append(_american_to_decimal(bet_odds))
            else:
                # All bets in combo are valid
                metrics = _calculate_parlay_metrics(combo, win_probs, decimal_odds)

                # Hard-clamp recommended sizing to remaining portfolio capacity.
                # Applying the cap here (at generation time) means:
                #   - Every ticket in the candidate pool already reflects the
                #     real deployable dollar amount — no post-hoc rescaling.
                #   - Tickets whose clamped size falls below MIN_PARLAY_UNITS
                #     are dropped immediately so they never appear in output.
                kelly_frac = metrics["kelly_fractional"]
                if remaining_capacity_dollars is not None and bankroll > 0:
                    if remaining_capacity_dollars <= 0.0:
                        # Portfolio fully consumed — skip all parlay combos.
                        continue
                    bet_dollars = kelly_frac * bankroll
                    if bet_dollars > remaining_capacity_dollars:
                        bet_dollars = remaining_capacity_dollars
                    kelly_frac = bet_dollars / bankroll
                    recommended_units = round((bet_dollars / bankroll) * 100.0, 2)
                else:
                    recommended_units = metrics["recommended_units"]

                if recommended_units < MIN_PARLAY_UNITS:
                    continue  # Too small to recommend after clamping

                # Convert parlay decimal odds back to American for display
                if metrics["parlay_odds"] >= 2.0:
                    parlay_american = (metrics["parlay_odds"] - 1.0) * 100
                else:
                    parlay_american = -100.0 / (metrics["parlay_odds"] - 1.0)

                # Create leg summary
                leg_summary = " + ".join([b.get("pick", "Unknown") for b in combo])

                parlay_ticket = {
                    "legs": list(combo),
                    "num_legs": num_legs,
                    "joint_prob": metrics["joint_prob"],
                    "parlay_odds": metrics["parlay_odds"],
                    "parlay_american_odds": round(parlay_american, 0),
                    "expected_value": metrics["expected_value"],
                    "edge": metrics["edge"],
                    "kelly_full": metrics["kelly_full"],
                    "kelly_fractional": round(kelly_frac, 6),
                    "recommended_units": recommended_units,
                    "leg_summary": leg_summary,
                }

                parlays.append(parlay_ticket)

    # Sort by expected value (descending)
    parlays.sort(key=lambda p: p["expected_value"], reverse=True)

    # Select top parlays with leg-overlap prevention.
    # Once a game appears in an accepted parlay ticket no subsequent ticket
    # may re-use the same game_id — prevents compounding correlated exposure
    # across returned tickets (e.g., Duke spread + Duke/UNC total both in
    # separate parlays doubles our exposure to that single game).
    top_parlays: List[Dict] = []
    used_game_ids: set = set()

    for parlay in parlays:
        parlay_game_ids = {leg.get("game_id") for leg in parlay["legs"]}
        if parlay_game_ids & used_game_ids:
            # At least one game already appears in an accepted parlay — skip
            continue
        top_parlays.append(parlay)
        used_game_ids.update(parlay_game_ids)
        if len(top_parlays) >= max_parlays:
            break

    logger.info(
        "Generated %d parlays, returning top %d non-overlapping (best EV: %.4f)",
        len(parlays), len(top_parlays),
        top_parlays[0]["expected_value"] if top_parlays else 0.0
    )

    return top_parlays


def format_parlay_ticket(parlay: Dict) -> str:
    """
    Format a parlay ticket for human-readable display.

    Args:
        parlay: Parlay dictionary from build_optimal_parlays()

    Returns:
        Formatted string for display
    """
    lines = []
    lines.append(f"🎫 {parlay['num_legs']}-Leg Parlay @ {parlay['parlay_american_odds']:+.0f}")
    lines.append(f"   Legs: {parlay['leg_summary']}")
    lines.append(f"   Joint Prob: {parlay['joint_prob']:.2%}")
    lines.append(f"   Expected Value: {parlay['expected_value']:.4f} units")
    lines.append(f"   Edge: {parlay['edge']:.2%}")
    lines.append(f"   Kelly Rec: {parlay['recommended_units']:.2f} units (fractional @ 1/{PARLAY_KELLY_DIVISOR:.0f})")

    return "\n".join(lines)
