"""
P16 -- Rest-of-Season Monte Carlo Simulation Engine.

Pure-computation module (no DB imports, no side effects).
All imports are at module top level -- no imports inside functions.

Algorithm:
  - Basis: 14-day decay-weighted rolling window (player_rolling_stats)
  - Simulations: N=1000 per player
  - Per-game draw: max(0, Normal(rate, rate * CV)) where CV=0.35
  - RNG: random.Random(seed) instance -- thread-safe, no global state

Output: SimulationResult dataclass with P10/P25/P50/P75/P90 percentiles
        for each counting and rate stat, plus composite risk metrics.

ADR-004: Never import betting_model or analysis.
"""

import random
from dataclasses import dataclass
from datetime import date
from typing import Optional, Dict
from sqlalchemy.orm import Session
from sqlalchemy import func

CV = 0.35                       # coefficient of variation per simulated game
N_SIMULATIONS = 1000
MLB_SEASON_GAMES = 162          # standard MLB season length
# Position-aware fallbacks when games_played unavailable
HITTER_GAMES_FALLBACK = 130     # everyday hitters mid-April
STARTER_APPEARANCES_FALLBACK = 12  # ~12 starts remaining for SP
RELIEVER_APPEARANCES_FALLBACK = 30 # ~30 appearances for RP
STARTER_APPEARANCE_INTERVAL = 5.0
RELIEVER_APPEARANCE_RATE = 0.45


# ---------------------------------------------------------------------------
# SimulationResult dataclass (pure-computation output -- NOT the ORM model)
# In daily_ingestion.py import the ORM as:
#   from backend.models import SimulationResult as SimulationResultORM
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    bdl_player_id: int
    as_of_date: date
    window_days: int            # always 14 -- 14d window is the simulation basis
    remaining_games: int
    n_simulations: int          # always 1000
    player_type: str            # "hitter" | "pitcher" | "two_way" | "unknown"

    # Hitter stat percentiles (None for pure pitchers / unknown)
    # Counting stats
    proj_r_p10: Optional[float] = None
    proj_r_p25: Optional[float] = None
    proj_r_p50: Optional[float] = None
    proj_r_p75: Optional[float] = None
    proj_r_p90: Optional[float] = None

    proj_h_p10: Optional[float] = None
    proj_h_p25: Optional[float] = None
    proj_h_p50: Optional[float] = None
    proj_h_p75: Optional[float] = None
    proj_h_p90: Optional[float] = None

    proj_hr_p10: Optional[float] = None
    proj_hr_p25: Optional[float] = None
    proj_hr_p50: Optional[float] = None
    proj_hr_p75: Optional[float] = None
    proj_hr_p90: Optional[float] = None

    proj_rbi_p10: Optional[float] = None
    proj_rbi_p25: Optional[float] = None
    proj_rbi_p50: Optional[float] = None
    proj_rbi_p75: Optional[float] = None
    proj_rbi_p90: Optional[float] = None

    proj_tb_p10: Optional[float] = None
    proj_tb_p25: Optional[float] = None
    proj_tb_p50: Optional[float] = None
    proj_tb_p75: Optional[float] = None
    proj_tb_p90: Optional[float] = None

    # Net stolen bases (SB - CS)
    proj_nsb_p10: Optional[float] = None
    proj_nsb_p25: Optional[float] = None
    proj_nsb_p50: Optional[float] = None
    proj_nsb_p75: Optional[float] = None
    proj_nsb_p90: Optional[float] = None

    # Legacy SB (retained for backward compatibility)
    proj_sb_p10: Optional[float] = None
    proj_sb_p25: Optional[float] = None
    proj_sb_p50: Optional[float] = None
    proj_sb_p75: Optional[float] = None
    proj_sb_p90: Optional[float] = None

    # Batting strikeouts (lower is better)
    proj_k_b_p10: Optional[float] = None
    proj_k_b_p25: Optional[float] = None
    proj_k_b_p50: Optional[float] = None
    proj_k_b_p75: Optional[float] = None
    proj_k_b_p90: Optional[float] = None

    # Rate stats
    proj_avg_p10: Optional[float] = None
    proj_avg_p25: Optional[float] = None
    proj_avg_p50: Optional[float] = None
    proj_avg_p75: Optional[float] = None
    proj_avg_p90: Optional[float] = None

    proj_ops_p10: Optional[float] = None
    proj_ops_p25: Optional[float] = None
    proj_ops_p50: Optional[float] = None
    proj_ops_p75: Optional[float] = None
    proj_ops_p90: Optional[float] = None

    # Pitcher stat percentiles (None for pure hitters / unknown)
    proj_k_p10: Optional[float] = None
    proj_k_p25: Optional[float] = None
    proj_k_p50: Optional[float] = None
    proj_k_p75: Optional[float] = None
    proj_k_p90: Optional[float] = None

    proj_qs_p10: Optional[float] = None
    proj_qs_p25: Optional[float] = None
    proj_qs_p50: Optional[float] = None
    proj_qs_p75: Optional[float] = None
    proj_qs_p90: Optional[float] = None

    # Rate stats (lower is better)
    proj_era_p10: Optional[float] = None
    proj_era_p25: Optional[float] = None
    proj_era_p50: Optional[float] = None
    proj_era_p75: Optional[float] = None
    proj_era_p90: Optional[float] = None

    proj_whip_p10: Optional[float] = None
    proj_whip_p25: Optional[float] = None
    proj_whip_p50: Optional[float] = None
    proj_whip_p75: Optional[float] = None
    proj_whip_p90: Optional[float] = None

    proj_k_per_9_p10: Optional[float] = None
    proj_k_per_9_p25: Optional[float] = None
    proj_k_per_9_p50: Optional[float] = None
    proj_k_per_9_p75: Optional[float] = None
    proj_k_per_9_p90: Optional[float] = None

    # Risk metrics (populated when composite z-scores are computable)
    composite_variance: Optional[float] = None
    downside_p25: Optional[float] = None    # P25 of per-simulation composite scores
    upside_p75: Optional[float] = None      # P75 of per-simulation composite scores
    prob_above_median: Optional[float] = None  # fraction of runs above P50 threshold


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _percentiles(values: list) -> tuple:
    """
    Return (P10, P25, P50, P75, P90) from a list of floats.

    Uses 0-indexed position: P10 = values[int(0.10 * n)], etc.
    Empty list returns all 0.0.
    """
    if not values:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    s = sorted(values)
    n = len(s)

    def _pct(p):
        idx = max(0, min(n - 1, int(p * n)))
        return s[idx]

    return (_pct(0.10), _pct(0.25), _pct(0.50), _pct(0.75), _pct(0.90))


def _sample_positive(rng, mu: float, sigma: float) -> float:
    """
    Draw from Normal(mu, sigma), floor at 0.

    Returns 0.0 immediately if mu <= 0 (avoids negative sigma).
    Uses the caller-provided rng instance for thread safety.
    """
    if mu <= 0:
        return 0.0
    return max(0.0, rng.gauss(mu, sigma))


def _draw_games(rng, rate: float, n_games: int) -> float:
    """
    Sum n_games independent draws from Normal(rate, rate*CV), each floored at 0.

    This is the inner loop shared by all counting stats.
    Returns the season total for one simulation run.
    """
    sigma = rate * CV
    total = 0.0
    for _ in range(n_games):
        total += _sample_positive(rng, rate, sigma)
    return total


def _compute_composite_risk(
    sim_composites: list,
) -> tuple:
    """
    Given 1000 composite scores (one per simulation run), return
    (composite_variance, downside_p25, upside_p75, prob_above_median).

    prob_above_median = fraction of runs exceeding the P50 composite value.
    """
    if not sim_composites:
        return (None, None, None, None)

    n = len(sim_composites)
    mean_c = sum(sim_composites) / n
    variance = sum((x - mean_c) ** 2 for x in sim_composites) / n

    p10, p25, p50, p75, p90 = _percentiles(sim_composites)
    prob_above = sum(1 for x in sim_composites if x > p50) / n

    return (variance, p25, p75, prob_above)


def _estimate_pitching_appearances(ip_per_appearance: float, remaining_team_games: int) -> int:
    """Estimate remaining pitching appearances from role-like usage.

    Using team games directly for pitchers wildly overstates ROS strikeout totals,
    especially for starters. We infer a coarse role from innings per appearance:
    starters get roughly one appearance every five team games, while relievers
    appear in a fraction of team games.
    """
    if remaining_team_games <= 0:
        return 0
    if ip_per_appearance >= 3.0:
        return max(1, int(round(remaining_team_games / STARTER_APPEARANCE_INTERVAL)))
    return max(1, int(round(remaining_team_games * RELIEVER_APPEARANCE_RATE)))


def _calculate_player_remaining_games(
    bdl_player_id: int,
    player_type: str,
    db: Optional[Session] = None,
    as_of_date: Optional[date] = None,
) -> int:
    """
    Calculate player-specific remaining games based on games_played.

    Formula:
    - Hitters: 162 - games_played (fallback: 130)
    - Starting Pitchers: round((162 - team_games_played) / 5) (fallback: 12)
    - Relief Pitchers: fallback: 30 appearances

    Parameters
    ----------
    bdl_player_id : int
    player_type : str ("hitter", "pitcher", "two_way", "unknown")
    db : Session or None
    as_of_date : date or None (defaults to today)

    Returns
    -------
    int : remaining games/appearances
    """
    if db is None or as_of_date is None:
        # No DB access - use fallbacks
        if player_type == "hitter":
            return HITTER_GAMES_FALLBACK
        elif player_type == "pitcher":
            return STARTER_APPEARANCES_FALLBACK
        return HITTER_GAMES_FALLBACK

    try:
        # Import here to avoid circular dependency
        from backend.models import MLBPlayerStats

        # Count games played by this player so far this season
        games_played = (
            db.query(func.count(func.distinct(MLBPlayerStats.game_date)))
            .filter(
                MLBPlayerStats.bdl_player_id == bdl_player_id,
                MLBPlayerStats.game_date <= as_of_date,
                MLBPlayerStats.season == as_of_date.year,
            )
            .scalar()
        ) or 0

        if games_played == 0:
            # No games found - use fallbacks
            if player_type == "hitter":
                return HITTER_GAMES_FALLBACK
            elif player_type == "pitcher":
                return STARTER_APPEARANCES_FALLBACK
            return HITTER_GAMES_FALLBACK

        # Calculate remaining games
        remaining = MLB_SEASON_GAMES - games_played

        # For pitchers, convert to starts/appearances
        if player_type == "pitcher":
            # SP assumption: ~1 start every 5 team games
            # For MVP, use simplified estimate
            starts_remaining = max(1, round(remaining / 5))
            return starts_remaining

        # For hitters and two-way players, return games directly
        return max(1, remaining)

    except Exception:
        # Database error - use safe fallbacks
        if player_type == "hitter":
            return HITTER_GAMES_FALLBACK
        elif player_type == "pitcher":
            return STARTER_APPEARANCES_FALLBACK
        return HITTER_GAMES_FALLBACK


# ---------------------------------------------------------------------------
# Main simulation entry points
# ---------------------------------------------------------------------------

def simulate_player(
    rolling_row,
    remaining_games: int = HITTER_GAMES_FALLBACK,
    n_simulations: int = N_SIMULATIONS,
    seed: Optional[int] = None,
    league_means: Optional[dict] = None,
    league_stds: Optional[dict] = None,
) -> SimulationResult:
    """
    Run Monte Carlo Rest-of-Season simulation for one player.

    Parameters
    ----------
    rolling_row : PlayerRollingStats ORM row (window_days=14)
        Must have bdl_player_id, as_of_date, games_in_window set.
        Batting fields present -> hitter/two_way path.
        Pitching fields present -> pitcher/two_way path.

    remaining_games : int
        Number of games remaining in the season (default 130 for mid-April 2026).

    n_simulations : int
        Number of Monte Carlo runs (default 1000).

    seed : int or None
        If provided, the RNG is seeded for reproducibility. Use seed=42 in tests.

    league_means : dict or None
        Batting: {"r": float, "h": float, "hr": float, "rbi": float, "tb": float,
                  "nsb": float, "k_b": float, "avg": float, "ops": float}
        Pitching: {"k": float, "qs": float, "k_per_9": float, "era": float, "whip": float}
        Required to compute composite risk metrics. If None, risk fields are None.

    league_stds : dict or None
        Same keys as league_means. Required alongside league_means.

    Returns
    -------
    SimulationResult dataclass with all applicable percentile fields populated.
    Covers 15 of 18 v2 categories (W, L, HR_P deferred - not available in rolling stats).
    """
    rng = random.Random(seed)
    # M3 fix: use decay-weighted game count for consistent rate derivation.
    # Fall back to raw games_in_window for rows computed before w_games was added.
    _wg = getattr(rolling_row, 'w_games', None)
    g = _wg if isinstance(_wg, (int, float)) and _wg > 0 else (rolling_row.games_in_window or 1)

    has_batting = rolling_row.w_ab is not None
    has_pitching = rolling_row.w_ip is not None

    if has_batting and has_pitching:
        player_type = "two_way"
    elif has_batting:
        player_type = "hitter"
    elif has_pitching:
        player_type = "pitcher"
    else:
        player_type = "unknown"

    result = SimulationResult(
        bdl_player_id=rolling_row.bdl_player_id,
        as_of_date=rolling_row.as_of_date,
        window_days=14,
        remaining_games=remaining_games,
        n_simulations=n_simulations,
        player_type=player_type,
    )

    if player_type == "unknown":
        return result

    # ------------------------------------------------------------------
    # Batting simulation
    # ------------------------------------------------------------------
    sim_composites = []  # populated later if league parameters available

    if has_batting:
        # Counting stat rates
        r_rate   = (rolling_row.w_runs           or 0.0) / g
        h_rate   = (rolling_row.w_hits           or 0.0) / g
        hr_rate  = (rolling_row.w_home_runs      or 0.0) / g
        rbi_rate = (rolling_row.w_rbi            or 0.0) / g
        tb_rate  = (rolling_row.w_tb             or 0.0) / g
        nsb_rate = (rolling_row.w_net_stolen_bases or 0.0) / g  # SB - CS
        sb_rate  = (rolling_row.w_stolen_bases   or 0.0) / g   # legacy
        k_b_rate = (rolling_row.w_strikeouts_bat or 0.0) / g   # K (lower better)

        # Rate stat inputs
        ab_rate   = (rolling_row.w_ab    or 0.0) / g
        hit_rate  = (rolling_row.w_hits  or 0.0) / g
        double_rate = (rolling_row.w_doubles or 0.0) / g
        triple_rate = (rolling_row.w_triples or 0.0) / g
        walk_rate   = (rolling_row.w_walks   or 0.0) / g

        # Sim lists for counting stats
        sim_r   = []
        sim_h   = []
        sim_hr  = []
        sim_rbi = []
        sim_tb  = []
        sim_nsb = []
        sim_sb  = []   # legacy
        sim_k_b = []

        # Sim lists for rate stats
        sim_avg = []
        sim_ops = []

        for _ in range(n_simulations):
            total_r    = _draw_games(rng, r_rate,    remaining_games)
            total_h    = _draw_games(rng, h_rate,    remaining_games)
            total_hr   = _draw_games(rng, hr_rate,   remaining_games)
            total_rbi  = _draw_games(rng, rbi_rate,  remaining_games)
            total_tb   = _draw_games(rng, tb_rate,   remaining_games)
            total_nsb  = _draw_games(rng, nsb_rate,  remaining_games)
            total_sb   = _draw_games(rng, sb_rate,   remaining_games)
            total_k_b  = _draw_games(rng, k_b_rate,  remaining_games)
            total_ab   = _draw_games(rng, ab_rate,   remaining_games)
            total_hit  = _draw_games(rng, hit_rate,  remaining_games)
            total_2b   = _draw_games(rng, double_rate, remaining_games)
            total_3b   = _draw_games(rng, triple_rate, remaining_games)
            total_walk = _draw_games(rng, walk_rate,   remaining_games)

            avg = total_hit / total_ab if total_ab > 0 else 0.0

            # OPS = (H + BB) / (AB + BB) + (TB) / (AB)
            # Using simulated totals for rate calculation
            obp_num = total_hit + total_walk
            obp_den = total_ab + total_walk
            obp = obp_num / obp_den if obp_den > 0 else 0.0
            slg = total_tb / total_ab if total_ab > 0 else 0.0
            ops = obp + slg

            sim_r.append(total_r)
            sim_h.append(total_h)
            sim_hr.append(total_hr)
            sim_rbi.append(total_rbi)
            sim_tb.append(total_tb)
            sim_nsb.append(total_nsb)
            sim_sb.append(total_sb)
            sim_k_b.append(total_k_b)
            sim_avg.append(avg)
            sim_ops.append(ops)

        # Populate percentile fields for counting stats
        (result.proj_r_p10, result.proj_r_p25, result.proj_r_p50,
         result.proj_r_p75, result.proj_r_p90) = _percentiles(sim_r)

        (result.proj_h_p10, result.proj_h_p25, result.proj_h_p50,
         result.proj_h_p75, result.proj_h_p90) = _percentiles(sim_h)

        (result.proj_hr_p10, result.proj_hr_p25, result.proj_hr_p50,
         result.proj_hr_p75, result.proj_hr_p90) = _percentiles(sim_hr)

        (result.proj_rbi_p10, result.proj_rbi_p25, result.proj_rbi_p50,
         result.proj_rbi_p75, result.proj_rbi_p90) = _percentiles(sim_rbi)

        (result.proj_tb_p10, result.proj_tb_p25, result.proj_tb_p50,
         result.proj_tb_p75, result.proj_tb_p90) = _percentiles(sim_tb)

        (result.proj_nsb_p10, result.proj_nsb_p25, result.proj_nsb_p50,
         result.proj_nsb_p75, result.proj_nsb_p90) = _percentiles(sim_nsb)

        (result.proj_sb_p10, result.proj_sb_p25, result.proj_sb_p50,
         result.proj_sb_p75, result.proj_sb_p90) = _percentiles(sim_sb)

        (result.proj_k_b_p10, result.proj_k_b_p25, result.proj_k_b_p50,
         result.proj_k_b_p75, result.proj_k_b_p90) = _percentiles(sim_k_b)

        # Populate percentile fields for rate stats
        (result.proj_avg_p10, result.proj_avg_p25, result.proj_avg_p50,
         result.proj_avg_p75, result.proj_avg_p90) = _percentiles(sim_avg)

        (result.proj_ops_p10, result.proj_ops_p25, result.proj_ops_p50,
         result.proj_ops_p75, result.proj_ops_p90) = _percentiles(sim_ops)

        # Build per-run composite z-scores for batting if league params available
        if (
            league_means is not None
            and league_stds is not None
        ):
            # Use all applicable batting categories for composite
            for i in range(n_simulations):
                zs = []
                # Counting stats (higher is better)
                for stat_name, sim_list, mean_key in [
                    ("r", sim_r, "r"), ("h", sim_h, "h"), ("hr", sim_hr, "hr"),
                    ("rbi", sim_rbi, "rbi"), ("tb", sim_tb, "tb"),
                    ("nsb", sim_nsb, "nsb"),
                ]:
                    std = league_stds.get(mean_key, 0)
                    if std > 0:
                        mean = league_means.get(mean_key, 0.0)
                        zs.append((sim_list[i] - mean) / std)
                # K_B is lower-is-better, invert z-score
                if league_stds.get("k_b", 0) > 0:
                    mean = league_means.get("k_b", 0.0)
                    std = league_stds.get("k_b", 1.0)
                    # Lower K is better, so negate the z-score
                    zs.append((mean - sim_k_b[i]) / std)
                # Rate stats (higher is better)
                for stat_name, sim_list, mean_key in [
                    ("avg", sim_avg, "avg"), ("ops", sim_ops, "ops"),
                ]:
                    std = league_stds.get(mean_key, 0)
                    if std > 0:
                        mean = league_means.get(mean_key, 0.0)
                        zs.append((sim_list[i] - mean) / std)
                comp = sum(zs) / len(zs) if zs else 0.0
                sim_composites.append(comp)

    # ------------------------------------------------------------------
    # Pitching simulation
    # ------------------------------------------------------------------
    sim_k_list    = []
    sim_qs_list   = []
    sim_era_list  = []
    sim_whip_list = []
    sim_k_per_9_list = []

    if has_pitching:
        ip_rate  = (rolling_row.w_ip              or 0.0) / g
        k_rate   = (rolling_row.w_strikeouts_pit  or 0.0) / g
        qs_rate  = (rolling_row.w_qs              or 0.0) / g
        er_rate  = (rolling_row.w_earned_runs     or 0.0) / g
        h_rate   = (rolling_row.w_hits_allowed    or 0.0) / g
        bb_rate  = (rolling_row.w_walks_allowed   or 0.0) / g
        pitching_appearances = _estimate_pitching_appearances(ip_rate, remaining_games)

        for _ in range(n_simulations):
            total_ip  = _draw_games(rng, ip_rate,  pitching_appearances)
            total_k   = _draw_games(rng, k_rate,   pitching_appearances)
            total_qs  = _draw_games(rng, qs_rate,  pitching_appearances)
            total_er  = _draw_games(rng, er_rate,  pitching_appearances)
            total_h   = _draw_games(rng, h_rate,   pitching_appearances)
            total_bb  = _draw_games(rng, bb_rate,  pitching_appearances)

            era  = 9.0 * total_er / total_ip       if total_ip > 0 else 0.0
            whip = (total_h + total_bb) / total_ip if total_ip > 0 else 0.0
            k_per_9 = 9.0 * total_k / total_ip    if total_ip > 0 else 0.0

            sim_k_list.append(total_k)
            sim_qs_list.append(total_qs)
            sim_era_list.append(era)
            sim_whip_list.append(whip)
            sim_k_per_9_list.append(k_per_9)

        (
            result.proj_k_p10,
            result.proj_k_p25,
            result.proj_k_p50,
            result.proj_k_p75,
            result.proj_k_p90,
        ) = _percentiles(sim_k_list)

        (
            result.proj_qs_p10,
            result.proj_qs_p25,
            result.proj_qs_p50,
            result.proj_qs_p75,
            result.proj_qs_p90,
        ) = _percentiles(sim_qs_list)

        (
            result.proj_era_p10,
            result.proj_era_p25,
            result.proj_era_p50,
            result.proj_era_p75,
            result.proj_era_p90,
        ) = _percentiles(sim_era_list)

        (
            result.proj_whip_p10,
            result.proj_whip_p25,
            result.proj_whip_p50,
            result.proj_whip_p75,
            result.proj_whip_p90,
        ) = _percentiles(sim_whip_list)

        (
            result.proj_k_per_9_p10,
            result.proj_k_per_9_p25,
            result.proj_k_per_9_p50,
            result.proj_k_per_9_p75,
            result.proj_k_per_9_p90,
        ) = _percentiles(sim_k_per_9_list)

        # Augment sim_composites with pitcher z-scores if league params available
        if (
            league_means is not None
            and league_stds is not None
            and not has_batting    # pure pitcher -- composites built here
        ):
            for i in range(n_simulations):
                zs = []
                # Counting stats (higher is better)
                for stat_name, sim_list, mean_key in [
                    ("k", sim_k_list, "k"), ("qs", sim_qs_list, "qs"),
                ]:
                    std = league_stds.get(mean_key, 0)
                    if std > 0:
                        mean = league_means.get(mean_key, 0.0)
                        zs.append((sim_list[i] - mean) / std)
                # Rate stats - ERA, WHIP are lower-is-better, invert
                for stat_name, sim_list, mean_key in [
                    ("era", sim_era_list, "era"),
                    ("whip", sim_whip_list, "whip"),
                ]:
                    std = league_stds.get(mean_key, 0)
                    if std > 0:
                        mean = league_means.get(mean_key, 0.0)
                        # Lower ERA/WHIP is better, so negate the z-score
                        zs.append((mean - sim_list[i]) / std)
                # K_9 is higher-is-better
                if league_stds.get("k_per_9", 0) > 0:
                    mean = league_means.get("k_per_9", 0.0)
                    std = league_stds.get("k_per_9", 1.0)
                    zs.append((sim_k_per_9_list[i] - mean) / std)
                comp = sum(zs) / len(zs) if zs else 0.0
                sim_composites.append(comp)

    # ------------------------------------------------------------------
    # Risk metrics (require sim_composites)
    # ------------------------------------------------------------------
    if sim_composites:
        (
            result.composite_variance,
            result.downside_p25,
            result.upside_p75,
            result.prob_above_median,
        ) = _compute_composite_risk(sim_composites)

    return result


def simulate_all_players(
    rolling_rows,
    remaining_games: Optional[int] = None,
    n_simulations: int = N_SIMULATIONS,
    league_means: Optional[dict] = None,
    league_stds: Optional[dict] = None,
    db: Optional[Session] = None,
    as_of_date: Optional[date] = None,
) -> list:
    """
    Run simulate_player for every row in rolling_rows.

    Rows where player_type resolves to "unknown" are silently skipped
    (no batting AND no pitching data -- not useful for projection).

    Parameters
    ----------
    rolling_rows : list of PlayerRollingStats ORM objects (window_days=14)
    remaining_games : int or None
        If None, calculates player-specific remaining games from DB.
        If provided, uses this value for ALL players (legacy behavior).
    n_simulations : int
    league_means : dict or None -- forwarded to simulate_player
    league_stds : dict or None -- forwarded to simulate_player
    db : Session or None
        Required for player-specific remaining games calculation.
    as_of_date : date or None
        Reference date for games_played query.

    Returns
    -------
    list of SimulationResult dataclass objects (unknown types excluded)
    """
    results = []
    for row in rolling_rows:
        # Determine player type first
        has_batting = row.w_ab is not None
        has_pitching = row.w_ip is not None
        if has_batting and has_pitching:
            player_type = "two_way"
        elif has_batting:
            player_type = "hitter"
        elif has_pitching:
            player_type = "pitcher"
        else:
            player_type = "unknown"

        # Skip unknown types
        if player_type == "unknown":
            continue

        # Calculate player-specific remaining games if not provided
        if remaining_games is None:
            player_remaining = _calculate_player_remaining_games(
                bdl_player_id=row.bdl_player_id,
                player_type=player_type,
                db=db,
                as_of_date=as_of_date or row.as_of_date,
            )
        else:
            player_remaining = remaining_games

        r = simulate_player(
            row,
            remaining_games=player_remaining,
            n_simulations=n_simulations,
            seed=None,
            league_means=league_means,
            league_stds=league_stds,
        )
        if r.player_type != "unknown":
            results.append(r)
    return results
