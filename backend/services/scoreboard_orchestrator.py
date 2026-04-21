"""
Scoreboard Orchestrator — Phase 4 Workstream A

Orchestrates L1-L4 components to build the Matchup Scoreboard.
Coordinates:
- L2: Yahoo current matchup data
- L3: ROW projections, player scores
- L4: Monte Carlo simulation
- L1: Category math, constraint helpers

This is the data assembly layer for GET /api/fantasy/scoreboard.
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from backend.stat_contract import (
    SCORING_CATEGORY_CODES,
    LOWER_IS_BETTER,
    BATTING_CODES,
    CONTRACT,
)
from backend.services.row_projector import ROWProjectionResult, compute_row_projection_from_canonical_rows
from backend.services.category_math import (
    compute_all_category_math,
    CategoryMathResult,
)
from backend.services.row_simulation_bridge import (
    prepare_simulation_inputs,
    prepare_h2h_monte_carlo_inputs,
    summarize_simulation_bundles,
)
from backend.services.constraint_helpers import classify_ip_pace
from backend.fantasy_baseball.h2h_monte_carlo import H2HOneWinSimulator
from backend.contracts import (
    MatchupScoreboardRow,
    MatchupScoreboardResponse,
    CategoryStats,
    CategoryStatusTag,
    ConstraintBudget,
    FreshnessMetadata,
    IPPaceFlag,
)


def _resolve_ratio_components(
    row: ROWProjectionResult,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return usable ratio components, falling back to synthetic denominators when needed."""
    finals = row.to_dict()
    numerators = {
        "AVG": finals.get("AVG", 0.0) * 450.0,
        "OPS": finals.get("TB", 0.0),
        "ERA": finals.get("ERA", 0.0) * 90.0 / 9.0,
        "WHIP": finals.get("WHIP", 0.0) * 90.0,
        "K_9": finals.get("K_9", 0.0) * 90.0 / 9.0,
    }
    denominators = {
        "AVG": 450.0,
        "OPS": 450.0,
        "ERA": 270.0,
        "WHIP": 270.0,
        "K_9": 270.0,
    }

    for code, numerator in (row.numerators or {}).items():
        denominator = (row.denominators or {}).get(code)
        if denominator and denominator > 0:
            numerators[code] = numerator
            denominators[code] = denominator

    return numerators, denominators


def _map_status_tag(win_prob: float) -> CategoryStatusTag:
    """Map Monte Carlo win probability to CategoryStatusTag."""
    if win_prob > 0.90:
        return CategoryStatusTag.LOCKED_WIN
    elif win_prob < 0.10:
        return CategoryStatusTag.LOCKED_LOSS
    elif win_prob > 0.65:
        return CategoryStatusTag.LEANING_WIN
    elif win_prob < 0.35:
        return CategoryStatusTag.LEANING_LOSS
    else:
        return CategoryStatusTag.BUBBLE


def _format_delta_to_flip(
    category: str,
    margin: float,
    delta_to_flip: float,
    is_batting: bool,
) -> str:
    """Human-readable delta-to-flip string."""
    if margin == 0:
        return "Tied"

    if is_batting:
        # Counting stats: "Need +3 HR" or "Leading by 5 R"
        if margin > 0:
            return f"Leading by {abs(margin):.0f}"
        else:
            return f"Need +{delta_to_flip:.0f} {CONTRACT.stats[category].short_label}"
    else:
        # Pitching counting stats similar
        if margin > 0:
            return f"Leading by {abs(margin):.0f}"
        else:
            return f"Need +{delta_to_flip:.0f} {CONTRACT.stats[category].short_label}"


def _project_row_from_player_scores(
    player_scores: List[Dict],
    games_remaining: Optional[Dict[str, int]] = None,
) -> ROWProjectionResult:
    """
    Project ROW from player score dicts.

    Extracts rolling_14d stats and delegates to compute_row_projection.
    """
    from backend.contracts import CanonicalPlayerRow
    from backend.services.row_projector import compute_row_projection

    # Extract rolling stats into the format expected by compute_row_projection
    rolling_by_player = {}
    season_by_player = {}

    for player in player_scores:
        player_key = str(player.get("yahoo_player_key") or player.get("bdl_player_id"))

        # Get rolling_14d dict if available
        rolling_by_player[player_key] = player.get("rolling_14d") or {}

        # Get season stats for blended rate
        season_by_player[player_key] = {
            "runs": player.get("runs", 0),
            "hits": player.get("hits", 0),
            "home_runs": player.get("home_runs", 0),
            "rbi": player.get("rbi", 0),
            "strikeouts_bat": player.get("strikeouts_bat", 0),
            "total_bases": player.get("total_bases", 0),
            "net_stolen_bases": player.get("net_stolen_bases", 0),
            "at_bats": player.get("at_bats", 0),
            "walks": player.get("walks", 0),
            "earned_runs": player.get("earned_runs", 0),
            "ip": player.get("ip", 0),
            "hits_allowed": player.get("hits_allowed", 0),
            "walks_allowed": player.get("walks_allowed", 0),
            "strikeouts_pit": player.get("strikeouts_pit", 0),
            "quality_starts": player.get("quality_starts", 0),
        }

    return compute_row_projection(
        rolling_stats_by_player=rolling_by_player,
        season_stats_by_player=season_by_player,
        games_remaining=games_remaining,
    )


def _build_ip_context(ip_accumulated: float, ip_minimum: float) -> str:
    """Build IP context string for ratio stats."""
    ip_remaining = ip_minimum - ip_accumulated
    if ip_remaining <= 0:
        return f"IP minimum met ({ip_accumulated:.1f}/{ip_minimum:.0f})"
    else:
        return f"{ip_accumulated:.1f}/{ip_minimum:.0f} IP, {ip_remaining:.1f} remaining"


def build_scoreboard_rows(
    my_current: Dict[str, float],
    opp_current: Dict[str, float],
    my_row: ROWProjectionResult,
    opp_row: ROWProjectionResult,
    category_math: Dict[str, CategoryMathResult],
    monte_carlo_result: Optional["H2HOneWinResult"] = None,
    ip_accumulated: float = 0.0,
    ip_minimum: float = 18.0,
    games_remaining: int = 0,
) -> List[MatchupScoreboardRow]:
    """Build 18 scoreboard rows from all data sources."""
    my_finals = my_row.to_dict()
    opp_finals = opp_row.to_dict()

    rows = []
    for cat in SCORING_CATEGORY_CODES:
        is_batting = cat in BATTING_CODES
        is_lower_better = cat in LOWER_IS_BETTER

        my_curr = my_current.get(cat, 0.0)
        opp_curr = opp_current.get(cat, 0.0)

        # Current margin (positive = winning, respects lower-is-better)
        if is_lower_better:
            curr_margin = opp_curr - my_curr
        else:
            curr_margin = my_curr - opp_curr

        # Projected values
        my_proj = my_finals.get(cat, 0.0)
        opp_proj = opp_finals.get(cat, 0.0)

        # Projected margin
        if is_lower_better:
            proj_margin = opp_proj - my_proj
        else:
            proj_margin = my_proj - opp_proj

        # Category math
        cat_math = category_math.get(cat)
        delta_to_flip_str = None
        if cat_math:
            delta_to_flip_str = _format_delta_to_flip(
                cat, curr_margin, cat_math.delta_to_flip, is_batting
            )

        # Status and flip probability from Monte Carlo
        status = None
        flip_prob = None
        if monte_carlo_result:
            cat_prob = monte_carlo_result.category_win_probs.get(cat, 0.5)
            status = _map_status_tag(cat_prob)
            flip_prob = cat_prob

        # IP context for pitching ratio stats
        ip_context = None
        if cat in ("ERA", "WHIP", "K_9"):
            ip_context = _build_ip_context(ip_accumulated, ip_minimum)

        # Games remaining for counting stats
        games_rem = None
        if not is_batting and cat not in ("ERA", "WHIP", "K_9"):
            # Pitching counting stats use IP remaining conceptually
            pass
        elif is_batting:
            games_rem = games_remaining

        row = MatchupScoreboardRow(
            category=cat,
            category_label=CONTRACT.stats[cat].short_label,
            is_lower_better=is_lower_better,
            is_batting=is_batting,
            my_current=my_curr,
            opp_current=opp_curr,
            current_margin=curr_margin,
            my_projected_final=my_proj,
            opp_projected_final=opp_proj,
            projected_margin=proj_margin,
            status=status,
            flip_probability=flip_prob,
            delta_to_flip=delta_to_flip_str,
            games_remaining=games_rem,
            ip_context=ip_context,
        )
        rows.append(row)

    return rows


def compute_budget_state(
    acquisitions_used: int,
    acquisition_limit: int = 8,
    il_used: int = 0,
    il_total: int = 3,
    ip_accumulated: float = 0.0,
    ip_minimum: float = 18.0,
    days_remaining: int = 7,
    season_days_elapsed: int = 90,
) -> ConstraintBudget:
    """Compute ConstraintBudget from raw values."""
    ip_pace = classify_ip_pace(
        ip_accumulated=ip_accumulated,
        ip_minimum=ip_minimum,
        days_elapsed=season_days_elapsed,
        days_total=182,  # Full MLB season (approximately)
    )

    return ConstraintBudget(
        acquisitions_used=acquisitions_used,
        acquisitions_remaining=acquisition_limit - acquisitions_used,
        acquisition_limit=acquisition_limit,
        acquisition_warning=acquisitions_used >= 6,
        il_used=il_used,
        il_total=il_total,
        ip_accumulated=ip_accumulated,
        ip_minimum=ip_minimum,
        ip_pace=IPPaceFlag[ip_pace.upper()],
        as_of=datetime.now(ZoneInfo("America/New_York")),
    )


def assemble_matchup_scoreboard(
    week: int,
    opponent_name: str,
    my_current_stats: Dict[str, float],
    opp_current_stats: Dict[str, float],
    my_player_scores: List[Dict],  # From player_scores table
    opp_player_scores: Optional[List[Dict]] = None,  # Optional opponent data
    ip_accumulated: float = 0.0,
    ip_minimum: float = 18.0,
    games_remaining: int = 0,
    days_remaining: int = 7,
    acquisitions_used: int = 0,
    il_used: int = 0,
    n_monte_carlo_sims: int = 1000,
    force_stale: bool = False,
) -> MatchupScoreboardResponse:
    """
    Assemble complete Matchup Scoreboard from all data sources.

    This is the main entry point for GET /api/fantasy/scoreboard.
    Orchestrates L1-L4 components to produce the full response.
    """
    now_et = datetime.now(ZoneInfo("America/New_York"))

    # Step 1: Compute ROW projections (L3)
    my_row = _project_row_from_player_scores(my_player_scores)
    # For opponent, use current stats as proxy if no player data available
    if opp_player_scores:
        opp_row = _project_row_from_player_scores(opp_player_scores)
    else:
        # Fallback: preserve current-state ratios/counts rather than zeroing them out.
        opp_row = ROWProjectionResult(
            **{k: opp_current_stats.get(k, 0.0) for k in SCORING_CATEGORY_CODES}
        )

    # Step 2: Compute category math (L1)
    my_finals = my_row.to_dict()
    opp_finals = opp_row.to_dict()

    # Use ratio stat components from projected rows for category math
    my_numerators, my_denominators = _resolve_ratio_components(my_row)
    opp_numerators, opp_denominators = _resolve_ratio_components(opp_row)

    category_math = compute_all_category_math(
        my_finals=my_finals,
        opp_finals=opp_finals,
        my_numerators=my_numerators,
        my_denominators=my_denominators,
        opp_numerators=opp_numerators,
        opp_denominators=opp_denominators,
    )

    # Step 3: Run Monte Carlo simulation (L4)
    monte_carlo_result = None
    overall_win_prob = None
    try:
        bundle = prepare_simulation_inputs(my_row, opp_row)
        my_finals_sim, opp_finals_sim = prepare_h2h_monte_carlo_inputs(bundle)
        sim = H2HOneWinSimulator()
        monte_carlo_result = sim.simulate_week_from_projections(
            my_finals_sim, opp_finals_sim, n_sims=n_monte_carlo_sims
        )
        overall_win_prob = monte_carlo_result.win_probability
    except Exception as e:
        # Monte Carlo is optional — scoreboard still useful without it
        pass

    # Step 4: Build scoreboard rows
    rows = build_scoreboard_rows(
        my_current_stats,
        opp_current_stats,
        my_row,
        opp_row,
        category_math,
        monte_carlo_result,
        ip_accumulated,
        ip_minimum,
        games_remaining,
    )

    # Step 5: Compute budget
    budget = compute_budget_state(
        acquisitions_used=acquisitions_used,
        il_used=il_used,
        ip_accumulated=ip_accumulated,
        ip_minimum=ip_minimum,
        days_remaining=days_remaining,
    )

    # Step 6: Count categories
    categories_won = sum(1 for r in rows if r.current_margin > 0)
    categories_lost = sum(1 for r in rows if r.current_margin < 0)
    categories_tied = sum(1 for r in rows if r.current_margin == 0)

    projected_won = sum(1 for r in rows if r.projected_margin is not None and r.projected_margin > 0)
    projected_lost = sum(1 for r in rows if r.projected_margin is not None and r.projected_margin < 0)
    projected_tied = sum(1 for r in rows if r.projected_margin is not None and r.projected_margin == 0)

    # Step 7: Freshness metadata
    freshness = FreshnessMetadata(
        primary_source="yahoo",
        fetched_at=None,  # TODO: track from Yahoo client
        computed_at=now_et,
        staleness_threshold_minutes=60,  # 1 hour
        is_stale=False,  # TODO: compute from fetched_at
    )

    return MatchupScoreboardResponse(
        week=week,
        opponent_name=opponent_name,
        categories_won=categories_won,
        categories_lost=categories_lost,
        categories_tied=categories_tied,
        projected_won=projected_won,
        projected_lost=projected_lost,
        projected_tied=projected_tied,
        overall_win_probability=overall_win_prob,
        rows=rows,
        budget=budget,
        freshness=freshness,
    )
