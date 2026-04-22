"""
Category-aware FA scoring engine for H2H fantasy baseball.

Scores a free agent against a team's category need vector.
The rate-stat protection gate prevents recommending players who damage
categories the team is already winning (ERA, WHIP, AVG, OPS, K/9).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

# Rate stats where adding a player with a bad z-score dilutes the team's lead.
# All keys are lowercase board keys matching PlayerProjection.cat_scores.
RATE_STAT_CATS: frozenset = frozenset({"avg", "ops", "era", "whip", "k9"})

# How far ahead the team must be (|deficit| > threshold) before the gate fires.
# deficit is expressed in z-score units (negative = team leading).
RATE_STAT_PROTECT_THRESHOLD: float = 0.5

# Multiplier applied to the absolute deficit when the gate fires.
# Produces a negative score: player_z * |deficit| * MULTIPLIER.
RATE_STAT_PENALTY_MULTIPLIER: float = 3.0


@dataclass(frozen=True)
class CategoryNeedVector:
    """Team's current per-category deficits against this week's opponent.

    needs: {category_key → deficit}
      Positive deficit  → team is losing that category (needs improvement)
      Negative deficit  → team is winning that category (has a lead)
    """
    needs: Dict[str, float]


@dataclass(frozen=True)
class PlayerCategoryImpactVector:
    """FA's projected impact per scoring category.

    impacts: {category_key → z_score}
      Positive z_score → player helps that category
      Negative z_score → player hurts that category
    """
    impacts: Dict[str, float]


def score_fa_against_needs(
    fa_impact: PlayerCategoryImpactVector,
    team_needs: CategoryNeedVector,
) -> float:
    """Compute a scalar score for a FA against the team's current needs.

    Scoring rules per category:

    Counting stats (not in RATE_STAT_CATS):
        contribution = player_z * max(0.0, deficit)
        - Team winning (deficit < 0): contribution = 0 (no reward for excess)
        - Team losing  (deficit > 0): contribution = player_z * deficit

    Rate stats (in RATE_STAT_CATS):
        Normal path  (deficit >= -THRESHOLD): contribution = player_z * max(0.0, deficit)
        Penalty path (deficit <  -THRESHOLD AND player_z < 0):
            contribution = player_z * |deficit| * RATE_STAT_PENALTY_MULTIPLIER
            This produces a NEGATIVE score, correctly flagging the FA as harmful.

    Returns the sum of contributions across all categories the FA has data for.
    """
    total = 0.0

    for cat, deficit in team_needs.needs.items():
        player_z = fa_impact.impacts.get(cat, 0.0)
        if player_z == 0.0 and cat not in fa_impact.impacts:
            continue

        if cat in RATE_STAT_CATS:
            if deficit < -RATE_STAT_PROTECT_THRESHOLD and player_z < 0.0:
                # Gate fires: player damages a category the team leads heavily
                total += player_z * abs(deficit) * RATE_STAT_PENALTY_MULTIPLIER
            else:
                total += player_z * max(0.0, deficit)
        else:
            total += player_z * max(0.0, deficit)

    return float(total)
