"""
P13 ROW Projector -- Rest-of-Week projections for H2H fantasy matchups.

Pure computation module. Zero I/O. Stateless functions only.
Input:  roster of CanonicalPlayerRow with rolling_14d stats.
Output: dict of canonical_code -> projected team total for rest of week.

Algorithm outline
-----------------
1. For each player, compute daily rate from rolling_14d (fallback to season).
2. Blend rates: 60% rolling + 40% season (stabilizes early-season).
3. Multiply by games_remaining for counting stats.
4. For ratio stats, accumulate numerators/denominators across roster.
5. Compute team-level ratios from accumulated components.

Ratio stat aggregation
----------------------
- AVG  = sum(H) / sum(AB)
- OPS  = sum(H+BB)/sum(AB+BB) + sum(TB)/sum(AB)
- ERA  = 27 * sum(ER) / sum(IP_outs)
- WHIP = 3 * sum(H+BB) / sum(IP_outs)
- K/9  = 27 * sum(K) / sum(IP_outs)

Greenfield categories (no upstream data yet)
---------------------------------------------
W, L, HR_P, NSV return 0.0 placeholders until Yahoo/MLB API ingestion exists.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from backend.stat_contract import (
    BATTING_CODES,
    LOWER_IS_BETTER,
    PITCHING_CODES,
    SCORING_CATEGORY_CODES,
)
from datetime import date


# MLB Opening Day 2026 - season start reference point
# ⚠️ VERIFY from authoritative source (MLB.com, official schedule)
# Last verified: 2026-04-20 - confirm before deploying
_MLB_OPENING_DAY = date(2026, 3, 27)


def _days_into_season(as_of_date: Optional[date] = None) -> int:
    """
    Return days since MLB Opening Day (minimum 1).

    Used for season-rate normalization in ROW projections.
    Prevents early-season projection bias from hardcoded 100-day assumption.

    Args:
        as_of_date: Date to calculate from. PREFER explicit date over None.
                    Defaults to today only for backward compatibility.

    Returns:
        Days since opening day (1 = opening day, 100 = ~June 1)
    """
    if as_of_date is None:
        # Fallback for callers without explicit date context
        as_of_date = date.today()
    delta = (as_of_date - _MLB_OPENING_DAY).days + 1
    return max(1, delta)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ROWProjectionResult:
    """
    Rest-of-Week projection result for a team roster.

    All 18 scoring categories are projected. Greenfield categories
    (W, L, HR_P, NSV) return 0.0 until upstream data is available.
    """
    # Batting counting stats
    R: float = 0.0
    H: float = 0.0
    HR_B: float = 0.0
    RBI: float = 0.0
    K_B: float = 0.0
    TB: float = 0.0
    NSB: float = 0.0

    # Batting ratio stats
    AVG: float = 0.0
    OPS: float = 0.0

    # Pitching counting stats
    W: float = 0.0       # Greenfield: no data yet
    L: float = 0.0       # Greenfield: no data yet
    HR_P: float = 0.0    # Greenfield: no data yet
    K_P: float = 0.0
    QS: float = 0.0
    NSV: float = 0.0     # Greenfield: no data yet

    # Pitching ratio stats
    ERA: float = 0.0
    WHIP: float = 0.0
    K_9: float = 0.0

    # Ratio stat components for delta-to-flip math
    numerators: Dict[str, float] = None
    denominators: Dict[str, float] = None

    def __post_init__(self):
        if self.numerators is None:
            self.numerators = {}
        if self.denominators is None:
            self.denominators = {}

    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dict keyed by canonical codes."""
        return {
            "R": self.R, "H": self.H, "HR_B": self.HR_B, "RBI": self.RBI,
            "K_B": self.K_B, "TB": self.TB, "NSB": self.NSB,
            "AVG": self.AVG, "OPS": self.OPS,
            "W": self.W, "L": self.L, "HR_P": self.HR_P, "K_P": self.K_P,
            "QS": self.QS, "NSV": self.NSV,
            "ERA": self.ERA, "WHIP": self.WHIP, "K_9": self.K_9,
        }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Categories that require numerator/denominator accumulation
# (not simple sum of player values)
_RATIO_STATS = frozenset({"AVG", "OPS", "ERA", "WHIP", "K_9"})

# Greenfield categories: no upstream data source exists yet
_GREENFIELD_CATEGORIES = frozenset({"W", "L", "HR_P", "NSV"})

# Default blended rate weights
_ROLLING_WEIGHT = 0.60
_SEASON_WEIGHT = 0.40

# Standard rolling window size for daily rate computation
_STANDARD_WINDOW_DAYS = 14

# MLB average HBP and SF rates per AB (for OBP imputation)
# Yahoo Fantasy API does not provide HBP/SF; we impute from league averages
_MLB_HBP_PER_AB = 0.0067   # ~4 HBP per 600 AB
_MLB_SF_PER_AB = 0.0083    # ~5 SF per 600 AB


# ---------------------------------------------------------------------------
# Mapping: canonical_code -> (rolling_w_col, season_col)
# ---------------------------------------------------------------------------
#
# rolling_w_col: the decay-weighted rolling stat column from PlayerRollingStats
# season_col: the season-long stat name (may differ from rolling)
#
# For ratio stats, these map to the numerator/denominator components.

# Counting stats: direct mapping
_COUNTING_STAT_MAPPING: Dict[str, tuple[str, str]] = {
    # Batting
    "R": ("w_runs", "runs"),
    "H": ("w_hits", "hits"),
    "HR_B": ("w_home_runs", "home_runs"),
    "RBI": ("w_rbi", "rbi"),
    "K_B": ("w_strikeouts_bat", "strikeouts_bat"),
    "TB": ("w_tb", "total_bases"),
    "NSB": ("w_net_stolen_bases", "net_stolen_bases"),
    # Pitching
    "W": (None, "wins"),           # Greenfield
    "L": (None, "losses"),         # Greenfield
    "HR_P": (None, "home_runs_allowed"),  # Greenfield
    "K_P": ("w_strikeouts_pit", "strikeouts_pit"),
    "QS": ("w_qs", "quality_starts"),
    "NSV": (None, "net_saves"),    # Greenfield
}

# Ratio stat components: (numerator_rolling, denominator_rolling, numerator_season, denominator_season)
# Note: ERA/WHIP/K_9 use w_ip (decimal IP) and convert to outs internally (×3)
_RATIO_COMPONENTS: Dict[str, tuple[str, str, str, str]] = {
    "AVG": ("w_hits", "w_ab", "hits", "at_bats"),
    "OPS": ("w_ops_components", "w_ab", "ops_components", "at_bats"),  # Composite
    "ERA": ("w_earned_runs", "w_ip", "earned_runs", "ip"),
    "WHIP": ("w_whip_numer", "w_ip", "whip_numer", "ip"),
    "K_9": ("w_strikeouts_pit", "w_ip", "strikeouts_pit", "ip"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_row_projection(
    rolling_stats_by_player: Dict[str, Dict[str, float]],
    season_stats_by_player: Optional[Dict[str, Dict[str, float]]] = None,
    games_remaining: Optional[Dict[str, int]] = None,
    *,
    rolling_weight: float = _ROLLING_WEIGHT,
    season_weight: float = _SEASON_WEIGHT,
    window_days: int = _STANDARD_WINDOW_DAYS,
    as_of_date: Optional[date] = None,
) -> ROWProjectionResult:
    """
    Compute team-level Rest-of-Week (ROW) projections for all 18 scoring categories.

    Parameters
    ----------
    rolling_stats_by_player : dict
        {player_key: {stat_col: decay_weighted_value}}
        Expected columns include w_runs, w_hits, w_home_runs, w_rbi, w_strikeouts_bat,
        w_tb, w_net_stolen_bases, w_ab, w_earned_runs, w_ip (decimal IP, converted to outs internally),
        w_strikeouts_pit, w_qs, w_walks, w_hits_allowed, w_walks_allowed.
    season_stats_by_player : dict, optional
        {player_key: {stat_name: season_total}}
        Used for blended rate stabilization. If None, only rolling rates used.
    games_remaining : dict, optional
        {player_key: expected_games_remaining}
        If None, assumes 1 game for all players (hitter) or 0 (pitcher projection disabled).
    rolling_weight : float
        Weight for rolling rate in blended calculation (default 0.60).
    season_weight : float
        Weight for season rate in blended calculation (default 0.40).
    window_days : int
        Rolling window size for daily rate computation (default 14).
    as_of_date : date, optional
        Date to calculate season rate from. PREFER explicit date over None.
        Defaults to today only for backward compatibility.

    Returns
    -------
    ROWProjectionResult with all 18 categories projected.
    Greenfield categories (W, L, HR_P, NSV) return 0.0.
    """
    if season_stats_by_player is None:
        season_stats_by_player = {}
    if games_remaining is None:
        # MVP fix: Position-aware fallbacks instead of hardcoded 1
        # Hitters: Assume most games remaining in week (~7 days max)
        # SPs: Assume 1 start (conservative, can be improved with probable pitcher data)
        # RPs: Assume 2-3 appearances
        games_remaining = {}
        for player_key, stats in rolling_stats_by_player.items():
            # Detect player type from rolling stats
            has_batting = stats.get("w_ab", 0) > 0
            has_pitching = stats.get("w_ip", 0) > 0
            
            if has_batting and not has_pitching:
                # Pure hitter: games in week (TODO: use actual schedule)
                games_remaining[player_key] = 7
            elif has_pitching and not has_batting:
                # Pure pitcher: assume 1 start for SP, 3 appearances for RP
                # (TODO: use ProbablePitcherSnapshot for accurate 1-start vs 2-start)
                qs = stats.get("w_qs", 0)
                if qs > 0:  # Likely a starter (had QS recently)
                    games_remaining[player_key] = 1
                else:
                    games_remaining[player_key] = 3  # Reliever appearances
            else:
                # Two-way or unknown: conservative fallback
                games_remaining[player_key] = 3

    # Initialize accumulators
    result = ROWProjectionResult()

    # Compute days into season once for all rate calculations
    days = _days_into_season(as_of_date)

    # Ratio stat component accumulators
    sum_h = sum_ab = 0.0
    sum_tb = 0.0
    sum_bb = 0.0
    sum_obp_denom = 0.0  # AB + BB
    sum_er = 0.0
    sum_ip_outs = 0.0
    sum_h_allowed = 0.0
    sum_bb_allowed = 0.0
    sum_k_p = 0.0

    # ------------------------------------------------------------------
    # Step 1: Accumulate player projections
    # ------------------------------------------------------------------
    for player_key, rolling_stats in rolling_stats_by_player.items():
        gr = games_remaining.get(player_key, 0)
        if gr == 0:
            continue

        season_stats = season_stats_by_player.get(player_key, {})

        # --- Counting stats (direct projection) ---
        for code, (rolling_col, season_col) in _COUNTING_STAT_MAPPING.items():
            if code in _GREENFIELD_CATEGORIES:
                continue  # Skip greenfield categories
            if rolling_col is None:
                continue  # No rolling data available

            rolling_val = rolling_stats.get(rolling_col, 0.0)
            season_val = season_stats.get(season_col, 0.0) if season_col in season_stats else 0.0

            # Compute daily rates
            rolling_rate = rolling_val / window_days if rolling_val else 0.0
            season_rate = season_val / days if season_val else 0.0

            # Blended rate
            blended_rate = (rolling_weight * rolling_rate +
                          season_weight * season_rate)

            # Project for games remaining
            proj_value = blended_rate * gr
            setattr(result, code, getattr(result, code, 0.0) + proj_value)

        # --- Ratio stat component accumulation ---
        # AVG: H / AB
        h_daily = _blended_daily_rate(
            rolling_stats.get("w_hits", 0.0),
            season_stats.get("hits", 0.0) if season_stats else 0.0,
            window_days,
            days,
            rolling_weight,
            season_weight,
        )
        ab_daily = _blended_daily_rate(
            rolling_stats.get("w_ab", 0.0),
            season_stats.get("at_bats", 0.0) if season_stats else 0.0,
            window_days,
            days,
            rolling_weight,
            season_weight,
        )
        sum_h += h_daily * gr
        sum_ab += ab_daily * gr

        # TB (for SLG component of OPS)
        tb_daily = _blended_daily_rate(
            rolling_stats.get("w_tb", 0.0),
            season_stats.get("total_bases", 0.0) if season_stats else 0.0,
            window_days,
            days,
            rolling_weight,
            season_weight,
        )
        sum_tb += tb_daily * gr

        # BB (for OBP component of OPS)
        bb_daily = _blended_daily_rate(
            rolling_stats.get("w_walks", 0.0),
            season_stats.get("walks", 0.0) if season_stats else 0.0,
            window_days,
            days,
            rolling_weight,
            season_weight,
        )
        sum_bb += bb_daily * gr
        # OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
        # Impute HBP and SF from league averages
        sum_obp_denom += (ab_daily + bb_daily +
                          ab_daily * _MLB_HBP_PER_AB +
                          ab_daily * _MLB_SF_PER_AB) * gr

        # ERA: ER / IP_outs (convert w_ip to outs by multiplying by 3)
        er_daily = _blended_daily_rate(
            rolling_stats.get("w_earned_runs", 0.0),
            season_stats.get("earned_runs", 0.0) if season_stats else 0.0,
            window_days,
            days,
            rolling_weight,
            season_weight,
        )
        # Convert decimal IP to outs (1 IP = 3 outs)
        ip_daily = _blended_daily_rate(
            rolling_stats.get("w_ip", 0.0),
            season_stats.get("ip", 0.0) if season_stats else 0.0,
            window_days,
            days,
            rolling_weight,
            season_weight,
        )
        ip_outs_daily = ip_daily * 3.0
        sum_er += er_daily * gr
        sum_ip_outs += ip_outs_daily * gr

        # WHIP: (H + BB) / IP_outs
        h_allowed_daily = _blended_daily_rate(
            rolling_stats.get("w_hits_allowed", 0.0),
            season_stats.get("hits_allowed", 0.0) if season_stats else 0.0,
            window_days,
            days,
            rolling_weight,
            season_weight,
        )
        bb_allowed_daily = _blended_daily_rate(
            rolling_stats.get("w_walks_allowed", 0.0),
            season_stats.get("walks_allowed", 0.0) if season_stats else 0.0,
            window_days,
            days,
            rolling_weight,
            season_weight,
        )
        sum_h_allowed += h_allowed_daily * gr
        sum_bb_allowed += bb_allowed_daily * gr

        # K_9: K / IP_outs
        k_p_daily = _blended_daily_rate(
            rolling_stats.get("w_strikeouts_pit", 0.0),
            season_stats.get("strikeouts_pit", 0.0) if season_stats else 0.0,
            window_days,
            days,
            rolling_weight,
            season_weight,
        )
        sum_k_p += k_p_daily * gr

    # ------------------------------------------------------------------
    # Step 2: Compute team-level ratio stats
    # ------------------------------------------------------------------
    # AVG = sum(H) / sum(AB)
    if sum_ab > 0:
        result.AVG = sum_h / sum_ab

    # OPS = OBP + SLG = (sum(H+BB)/sum(AB+BB)) + (sum(TB)/sum(AB))
    # OBP with HBP imputation: (H + BB + HBP) / (AB + BB + HBP + SF)
    sum_hbp = sum_ab * _MLB_HBP_PER_AB
    sum_sf = sum_ab * _MLB_SF_PER_AB
    obp_numer = sum_h + sum_bb + sum_hbp
    obp_denom = sum_ab + sum_bb + sum_hbp + sum_sf

    if obp_denom > 0:
        obp = obp_numer / obp_denom
    else:
        obp = 0.0

    # SLG unchanged: TB / AB
    if sum_ab > 0:
        slg = sum_tb / sum_ab
    else:
        slg = 0.0

    result.OPS = obp + slg

    # ERA = 27 * sum(ER) / sum(IP_outs)
    if sum_ip_outs > 0:
        result.ERA = 27.0 * sum_er / sum_ip_outs

    # WHIP = 3 * sum(H + BB) / sum(IP_outs)
    if sum_ip_outs > 0:
        result.WHIP = 3.0 * (sum_h_allowed + sum_bb_allowed) / sum_ip_outs

    # K_9 = 27 * sum(K) / sum(IP_outs)
    if sum_ip_outs > 0:
        result.K_9 = 27.0 * sum_k_p / sum_ip_outs

    # Store ratio components
    result.numerators = {
        "AVG": sum_h,
        "OPS": sum_h + sum_bb + sum_tb,  # Simplified composite
        "ERA": sum_er,
        "WHIP": sum_h_allowed + sum_bb_allowed,
        "K_9": sum_k_p,
    }
    result.denominators = {
        "AVG": sum_ab,
        "OPS": sum_obp_denom + sum_ab,  # Simplified composite
        "ERA": sum_ip_outs,
        "WHIP": sum_ip_outs,
        "K_9": sum_ip_outs,
    }

    return result


def _blended_daily_rate(
    rolling_total: float,
    season_total: float,
    window_days: int,
    days_into_season: int,
    rolling_weight: float,
    season_weight: float,
) -> float:
    """
    Compute blended daily rate from rolling and season totals.

    Formula: (rolling_weight * rolling_total / window_days) +
             (season_weight * season_total / days_into_season)

    Args:
        rolling_total: Decay-weighted rolling total over window_days
        season_total: Season-to-date total
        window_days: Rolling window size (typically 14)
        days_into_season: Days since MLB Opening Day (dynamic, not hardcoded)
        rolling_weight: Weight for rolling rate (default 0.60)
        season_weight: Weight for season rate (default 0.40)
    """
    if rolling_total <= 0 and season_total <= 0:
        return 0.0

    rolling_rate = rolling_total / window_days if rolling_total > 0 else 0.0
    season_rate = season_total / days_into_season if season_total > 0 else 0.0

    return rolling_weight * rolling_rate + season_weight * season_rate


def compute_row_projection_from_canonical_rows(
    roster_rows: List["CanonicalPlayerRow"],
    games_remaining: Dict[str, int],
    season_stats_by_player: Optional[Dict[str, Dict[str, float]]] = None,
) -> ROWProjectionResult:
    """
    Convenience wrapper for CanonicalPlayerRow objects.

    Extracts rolling_14d stats from each row and delegates to compute_row_projection.

    Parameters
    ----------
    roster_rows : List[CanonicalPlayerRow]
        Active roster (not bench). Each row must have rolling_14d populated.
    games_remaining : dict
        {player_key: expected_games_remaining}
    season_stats_by_player : dict, optional
        {player_key: {stat_name: season_total}}

    Returns
    -------
    ROWProjectionResult with all 18 categories projected.
    """
    from backend.contracts import CanonicalPlayerRow

    # Extract rolling stats into the format expected by compute_row_projection
    rolling_by_player = {}
    for row in roster_rows:
        player_key = row.yahoo_player_key or str(row.bdl_player_id)
        if row.rolling_14d:
            rolling_by_player[player_key] = dict(row.rolling_14d)

    return compute_row_projection(
        rolling_by_player,
        season_stats_by_player,
        games_remaining,
    )


# ---------------------------------------------------------------------------
# Helper functions for games_remaining estimation
# ---------------------------------------------------------------------------

def estimate_hitter_games_remaining(
    current_day_of_week: int,
    days_remaining_in_week: int,
    player_status: str = "healthy",
) -> int:
    """
    Estimate games remaining for a hitter this week.

    Parameters
    ----------
    current_day_of_week : int
        0=Monday, 6=Sunday.
    days_remaining_in_week : int
        Days until Sunday (inclusive).
    player_status : str
        "healthy", "day_to_day", "out", "dl"

    Returns
    -------
    Expected games remaining (0-7).
    """
    if player_status in ("out", "dl"):
        return 0

    # Healthy hitters generally play all remaining days
    # TODO: Incorporate off-days from schedule
    return days_remaining_in_week


def estimate_pitcher_games_remaining(
    current_day_of_week: int,
    days_remaining_in_week: int,
    probable_starters: Dict[str, int],
    pitcher_team: str,
    rotation_slot: Optional[int] = None,
) -> int:
    """
    Estimate starts remaining for a pitcher this week.

    Parameters
    ----------
    current_day_of_week : int
        0=Monday, 6=Sunday.
    days_remaining_in_week : int
        Days until Sunday (inclusive).
    probable_starters : dict
        {team: bdl_player_id} for each probable starter.
    pitcher_team : str
        The pitcher's team abbreviation.
    rotation_slot : int, optional
        1-5 for rotation order. Used to project future starts.

    Returns
    -------
    Expected starts remaining (0-2 typically).
    """
    # TODO: Implement rotation math + probable pitcher feed integration
    # For now, return 1 if pitcher is listed as probable today
    probable_id = probable_starters.get(pitcher_team)
    if probable_id:
        return 1
    return 0
