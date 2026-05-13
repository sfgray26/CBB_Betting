"""
Category-aware FA scoring engine for H2H fantasy baseball.

Scores a free agent against a team's category need vector.
The rate-stat protection gate prevents recommending players who damage
categories the team is already winning (ERA, WHIP, AVG, OBP, OPS, K/9).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from backend.services.config_service import get_threshold as _get_threshold

try:
    from backend.fantasy_baseball.team_context import TeamContext as _TeamContext
except ImportError:
    _TeamContext = None  # type: ignore

# Rate stats where adding a player with a bad z-score dilutes the team's lead.
# All keys are lowercase board keys matching PlayerProjection.cat_scores.
RATE_STAT_CATS: frozenset = frozenset({"avg", "obp", "ops", "era", "whip", "k9"})

# How far ahead the team must be (|deficit| > threshold) before the gate fires.
# deficit is expressed in z-score units (negative = team leading).
RATE_STAT_PROTECT_THRESHOLD: float = _get_threshold("scoring.rate_stat_protect", default=0.5)

# Multiplier applied to the absolute deficit when the gate fires.
# Produces a negative score: player_z * |deficit| * MULTIPLIER.
RATE_STAT_PENALTY_MULTIPLIER: float = 3.0

# Canonical scoring codes → board keys used in PlayerProjection.cat_scores.
# cat.lower() is correct for most codes; only non-trivial remappings are listed.
_CANONICAL_TO_BOARD: dict = {
    "HR_B": "hr",
    "HR_P": "hr_pit",
    "K_9":  "k9",
    "K_P":  "k_pit",
    "K_B":  "k_bat",
}

# League-average standard deviations for rate categories (2025 MLB season approx).
# Used to convert marginal rate deltas to z-score units in marginal_rate_impact().
RATE_STAT_LEAGUE_STD: dict[str, float] = {
    "avg": 0.025,
    "obp": 0.028,
    "ops": 0.055,
    "era": 0.60,
    "whip": 0.12,
    "k9": 0.80,
}

# Category-specific numerator/denominator field names for marginal math.
# Maps category key -> (numerator_attr, denominator_attr) on CanonicalProjection.
MARGINAL_RATE_FIELDS: dict[str, tuple[str, str]] = {
    "avg": ("proj_h", "proj_ab"),
    "obp": ("proj_h", "proj_pa"),
    "era": ("proj_er", "proj_ip"),
    "whip": ("proj_h", "proj_ip"),
}


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


def marginal_rate_impact(
    category: str,
    team_numerator: float,
    team_denominator: float,
    player_numerator: float,
    player_denominator: float,
    league_std: float,
) -> float:
    """Compute true marginal rate-stat impact of adding a player.

    Instead of using the player's standalone z-score, computes the actual
    change in the team's rate stat after adding the player.

    Formula:
        combined_rate = (team_numerator + player_numerator) / (team_denominator + player_denominator)
        current_rate  = team_numerator / team_denominator
        marginal_delta = combined_rate - current_rate
        marginal_z = marginal_delta / league_std

    For lower-is-better categories (ERA, WHIP), marginal_delta is negated so
    that a positive z-score means the player helps the team.

    Args:
        category: Category key (e.g. 'avg', 'era', 'whip', 'obp')
        team_numerator: Team's current projected numerator (e.g. projected_hits for AVG)
        team_denominator: Team's current projected denominator (e.g. projected_ab for AVG)
        player_numerator: Player's projected numerator contribution
        player_denominator: Player's projected denominator contribution
        league_std: League standard deviation for this rate stat (for z-score conversion)

    Returns:
        Marginal z-score impact. Positive = player helps, Negative = player hurts.
        Returns 0.0 if denominators are zero or league_std is zero.
    """
    if team_denominator <= 0 or player_denominator <= 0 or league_std <= 0:
        return 0.0

    current_rate = team_numerator / team_denominator
    combined_rate = (team_numerator + player_numerator) / (team_denominator + player_denominator)
    marginal_delta = combined_rate - current_rate

    # ERA and WHIP: lower is better, so flip sign
    if category in RATE_STAT_CATS and category in {"era", "whip"}:
        marginal_delta = -marginal_delta

    return marginal_delta / league_std


def compute_need_score(
    player_cat_scores: dict,
    player_z_score: float,
    category_deficits: list,
    n_cats: int,
    team_context: Optional[object] = None,
    marginal_stats: Optional[dict] = None,
) -> float:
    """Unified need_score computation for waiver and recommendations endpoints.

    Combines generic player quality (z_score) with matchup-specific category impact
    using a 40/60 blend. Applies rate-stat protection via score_fa_against_needs().

    Args:
        player_cat_scores: Dict of {category: z_score} from PlayerProjection.cat_scores
        player_z_score: Overall z_score for the player (generic quality metric)
        category_deficits: List of CategoryDeficitOut objects from matchup analysis
        n_cats: Number of categories in the scoring system (for normalization)
        team_context: Optional TeamContext with roster denominator context.
        marginal_stats: Optional raw numerator/denominator data keyed by category.

    Returns:
        Float need_score. If category_deficits is empty/None, returns player_z_score.
        If scoring fails, falls back to player_z_score.
    """
    from backend.schemas import CategoryDeficitOut

    # Fallback: no matchup data available or empty cat_scores
    if not category_deficits or not player_cat_scores:
        return float(player_z_score)

    # Filter to only numeric values (skip strings, None, etc.)
    valid_cat_scores: Dict[str, float] = {
        k: float(v) for k, v in player_cat_scores.items()
        if isinstance(v, (int, float))
    }
    if not valid_cat_scores:
        return float(player_z_score)

    # Build CategoryNeedVector from CategoryDeficit list
    needs_dict: Dict[str, float] = {}
    for cd in category_deficits:
        if isinstance(cd, CategoryDeficitOut):
            # Translate canonical code (e.g. "HR_B") to board key (e.g. "hr")
            # so it aligns with PlayerProjection.cat_scores keys.
            board_key = _CANONICAL_TO_BOARD.get(cd.category.upper(), cd.category.lower())
            needs_dict[board_key] = float(cd.deficit)

    if not needs_dict:
        return float(player_z_score)

    # Override rate-stat z-scores with true marginal impact when raw stats available
    if marginal_stats:
        for cat in list(valid_cat_scores.keys()):
            if cat in marginal_stats and cat in RATE_STAT_CATS:
                ms = marginal_stats[cat]
                try:
                    marginal_z = marginal_rate_impact(
                        category=cat,
                        team_numerator=float(ms["team_num"]),
                        team_denominator=float(ms["team_den"]),
                        player_numerator=float(ms["player_num"]),
                        player_denominator=float(ms["player_den"]),
                        league_std=RATE_STAT_LEAGUE_STD.get(cat, 0.05),
                    )
                    if marginal_z != 0.0:
                        valid_cat_scores[cat] = marginal_z
                except (KeyError, TypeError, ZeroDivisionError):
                    pass  # Fall back to existing z-score

    team_needs = CategoryNeedVector(needs=needs_dict)

    # Build PlayerCategoryImpactVector from cat_scores
    fa_impact = PlayerCategoryImpactVector(impacts=valid_cat_scores)

    try:
        # Compute matchup-specific score with rate-stat protection
        cat_score = score_fa_against_needs(fa_impact, team_needs)

        # Normalize per-category and blend: 40% generic z + 60% matchup-specific
        n_cats_safe = max(1, n_cats)
        blended_score = 0.4 * player_z_score + 0.6 * (cat_score / n_cats_safe)

        if team_context is not None and hasattr(team_context, "rate_pa_denominator"):
            # Apply roster-depth scaling to the matchup-specific component only.
            # Deeper rosters (>3600 PA) dilute each add's rate impact slightly;
            # shallower rosters (<3600 PA) amplify it. Capped [0.8, 1.2].
            pa_denom = float(getattr(team_context, "rate_pa_denominator", 0.0))
            ip_denom = float(getattr(team_context, "rate_ip_denominator", 0.0))

            if pa_denom > 0:
                depth_factor = max(0.8, min(1.2, 3600.0 / pa_denom))
            elif ip_denom > 0:
                depth_factor = max(0.8, min(1.2, 900.0 / ip_denom))
            else:
                depth_factor = 1.0

            # Re-decompose blended_score: scale only the 60% matchup component.
            matchup_component = blended_score - 0.4 * player_z_score
            blended_score = 0.4 * player_z_score + depth_factor * matchup_component

        return float(blended_score)
    except Exception:
        # Fallback to plain z_score if scorer fails
        return float(player_z_score)
