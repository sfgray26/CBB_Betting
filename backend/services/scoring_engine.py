"""
P14 Scoring Engine -- League Z-scores per player per rolling window.

Pure computation module. Zero I/O. Stateless functions only.
Input:  list of PlayerRollingStats ORM rows for one window size.
Output: list of PlayerScoreResult dataclasses.

Algorithm outline
-----------------
1. For each scoring category collect all non-None values across the player pool.
2. If fewer than MIN_SAMPLE players have a value, skip the category (all Z = None).
3. Compute population std (ddof=0). If std == 0, skip (degenerate column).
4. Z = (val - mean) / std; negate Z for lower-is-better categories.
5. Cap Z at +/-Z_CAP to dampen outlier distortion.
6. composite_z = mean of all applicable non-None Z-scores for the player.
7. score_0_100 = percentile rank within the player's player_type cohort.
8. confidence = min(1.0, games_in_window / window_days).

Hard stops
----------
- No numpy / scipy / pandas imports.
- No position-level Z-scores (deferred -- position not yet in schema).
- No datetime.utcnow() usage.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Category configuration
# ---------------------------------------------------------------------------

# key -> (column_name_on_PlayerRollingStats, is_lower_better)
HITTER_CATEGORIES: dict[str, tuple[str, bool]] = {
    "z_hr":  ("w_home_runs",   False),
    "z_rbi": ("w_rbi",         False),
    "z_sb":  ("w_stolen_bases", False),
    "z_avg": ("w_avg",         False),
    "z_obp": ("w_obp",         False),
}

PITCHER_CATEGORIES: dict[str, tuple[str, bool]] = {
    "z_era":     ("w_era",         True),   # lower ERA is better -> negate Z
    "z_whip":    ("w_whip",        True),   # lower WHIP is better -> negate Z
    "z_k_per_9": ("w_k_per_9",     False),
}

# Combined for iteration convenience
_ALL_CATEGORIES: dict[str, tuple[str, bool]] = {
    **HITTER_CATEGORIES,
    **PITCHER_CATEGORIES,
}

# Minimum number of players with a non-null value before computing Z for a category
MIN_SAMPLE: int = 5

# Cap Z-scores at this absolute value to reduce outlier distortion
Z_CAP: float = 3.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlayerScoreResult:
    bdl_player_id: int
    as_of_date: date
    window_days: int
    player_type: str          # "hitter" | "pitcher" | "two_way"
    games_in_window: int

    # Per-category Z-scores (None if category not applicable or < MIN_SAMPLE)
    z_hr:      Optional[float] = None
    z_rbi:     Optional[float] = None
    z_sb:      Optional[float] = None
    z_avg:     Optional[float] = None
    z_obp:     Optional[float] = None
    z_era:     Optional[float] = None
    z_whip:    Optional[float] = None
    z_k_per_9: Optional[float] = None

    composite_z: float = 0.0   # mean of all applicable non-None Z-scores
    score_0_100: float = 50.0  # percentile rank 0-100 within player_type
    confidence:  float = 0.0   # games_in_window / window_days, capped at 1.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _population_std(values: list[float]) -> float:
    """Population standard deviation (ddof=0) for a list of floats."""
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    return math.sqrt(variance)


def _cap(z: float) -> float:
    """Clamp Z to [-Z_CAP, +Z_CAP]."""
    return max(-Z_CAP, min(Z_CAP, z))


def _detect_player_type(row) -> str:
    """
    Determine player type from a PlayerRollingStats row.

    Returns "two_way", "hitter", "pitcher", or "unknown".
    "unknown" rows are excluded from results.
    """
    has_batting = row.w_ab is not None
    has_pitching = row.w_ip is not None
    if has_batting and has_pitching:
        return "two_way"
    if has_batting:
        return "hitter"
    if has_pitching:
        return "pitcher"
    return "unknown"


def _applicable_z_keys(player_type: str) -> list[str]:
    """Return the Z-score field names applicable to a given player_type."""
    if player_type == "hitter":
        return list(HITTER_CATEGORIES.keys())
    if player_type == "pitcher":
        return list(PITCHER_CATEGORIES.keys())
    if player_type == "two_way":
        return list(HITTER_CATEGORIES.keys()) + list(PITCHER_CATEGORIES.keys())
    return []


def _percentile_rank(player_z: float, cohort_z_scores: list[float]) -> float:
    """
    Percentile rank: proportion of cohort at or below player_z, scaled to 0-100.
    Uses weak ordering (<=) consistent with scipy.stats.percentileofscore 'weak'.
    """
    n = len(cohort_z_scores)
    if n <= 1:
        return 50.0
    rank = sum(1 for c in cohort_z_scores if c <= player_z)
    return round(rank / n * 100.0, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_league_zscores(
    rolling_rows: list,
    as_of_date: date,
    window_days: int,
) -> list[PlayerScoreResult]:
    """
    Compute league Z-scores for all players in rolling_rows.

    Parameters
    ----------
    rolling_rows : list of PlayerRollingStats ORM objects for ONE window size.
    as_of_date   : the date these rolling stats represent.
    window_days  : 7, 14, or 30.

    Returns
    -------
    list[PlayerScoreResult] -- one entry per player with a known player_type.
    Players whose w_ab and w_ip are both None are excluded ("unknown" type).
    """
    if not rolling_rows:
        return []

    # ------------------------------------------------------------------
    # Step 1: Compute league-level stats per category
    # ------------------------------------------------------------------
    # category_z_lookup[z_key][bdl_player_id] = raw Z-score (pre-cap)
    category_z_lookup: dict[str, dict[int, float]] = {k: {} for k in _ALL_CATEGORIES}

    for z_key, (col_name, is_lower_better) in _ALL_CATEGORIES.items():
        # Collect (player_id, value) pairs where value is non-None
        pairs: list[tuple[int, float]] = []
        for row in rolling_rows:
            val = getattr(row, col_name, None)
            if val is not None:
                pairs.append((row.bdl_player_id, float(val)))

        if len(pairs) < MIN_SAMPLE:
            # Not enough data -- all Z for this category remain None
            continue

        values = [v for _, v in pairs]
        mean = sum(values) / len(values)
        std = _population_std(values)

        if std == 0.0:
            # Degenerate column (all identical) -- skip
            continue

        for player_id, val in pairs:
            z = (val - mean) / std
            if is_lower_better:
                z = -z
            category_z_lookup[z_key][player_id] = _cap(z)

    # ------------------------------------------------------------------
    # Step 2: Build per-player results
    # ------------------------------------------------------------------
    results: list[PlayerScoreResult] = []

    for row in rolling_rows:
        player_type = _detect_player_type(row)
        if player_type == "unknown":
            continue

        applicable_keys = _applicable_z_keys(player_type)
        pid = row.bdl_player_id

        result = PlayerScoreResult(
            bdl_player_id=pid,
            as_of_date=as_of_date,
            window_days=window_days,
            player_type=player_type,
            games_in_window=row.games_in_window,
        )

        # Assign per-category Z-scores
        for z_key in applicable_keys:
            z_val = category_z_lookup[z_key].get(pid)  # None if not computed
            setattr(result, z_key, z_val)

        # Step 3: composite_z = mean of all applicable non-None Z-scores
        non_none = [
            getattr(result, k)
            for k in applicable_keys
            if getattr(result, k) is not None
        ]
        if non_none:
            result.composite_z = sum(non_none) / len(non_none)
        else:
            # No category had enough peers — insufficient data for this player.
            # composite_z stays 0.0; log so operators can detect this at startup/season.
            result.composite_z = 0.0
            logger.debug(
                "scoring_engine: bdl_player_id=%s has no scoreable categories "
                "(window=%dd, games=%d) — composite_z defaults to 0.0",
                pid, window_days, row.games_in_window,
            )

        # Step 4: confidence = min(1.0, games_in_window / window_days)
        result.confidence = min(1.0, row.games_in_window / window_days)

        results.append(result)

    # ------------------------------------------------------------------
    # Step 5: score_0_100 -- percentile rank within player_type cohort
    # ------------------------------------------------------------------
    # Group composite_z by player_type for ranking
    type_groups: dict[str, list[float]] = {}
    for res in results:
        type_groups.setdefault(res.player_type, []).append(res.composite_z)

    for res in results:
        cohort = type_groups[res.player_type]
        res.score_0_100 = _percentile_rank(res.composite_z, cohort)

    return results
