"""
VORP Engine -- Value Over Replacement Player computation.

Pure computation module. Zero I/O. Stateless functions only.

VORP measures how much better a player performs relative to a freely-available
replacement at their scarcest eligible position. This makes position-scarce
players (C, SS) more valuable than position-abundant ones (OF, DH).

Formula: VORP = composite_z - replacement_z(scarcest_position)

Replacement levels from K-38 VORP Implementation Guide:
  C=-5.5, 1B=-3.0, 2B=-4.0, SS=-4.0, 3B=-3.5, OF=-2.5
  SP=-3.0, RP=-2.0, DH=-2.0, UTIL=-1.5

Multi-eligible players use the position with the LOWEST (most negative)
replacement level, because scarcity there makes their VORP highest.

Input:  composite_z (from scoring_engine) + position eligibility flags
Output: vorp value (float)
"""

from typing import Optional

# ---------------------------------------------------------------------------
# Replacement-level Z-scores by position (K-38 research)
# More negative = scarcer position = higher VORP for same composite_z
# ---------------------------------------------------------------------------

REPLACEMENT_LEVELS: dict[str, float] = {
    "C":    -5.5,
    "1B":   -3.0,
    "2B":   -4.0,
    "SS":   -4.0,
    "3B":   -3.5,
    "LF":   -2.5,
    "CF":   -2.5,
    "RF":   -2.5,
    "OF":   -2.5,
    "DH":   -2.0,
    "UTIL": -1.5,
    "SP":   -3.0,
    "RP":   -2.0,
}

# Position flag column names on PositionEligibility -> position key
_POS_FLAG_MAP: dict[str, str] = {
    "can_play_c":  "C",
    "can_play_1b": "1B",
    "can_play_2b": "2B",
    "can_play_3b": "3B",
    "can_play_ss": "SS",
    "can_play_lf": "LF",
    "can_play_cf": "CF",
    "can_play_rf": "RF",
    "can_play_of": "OF",
    "can_play_dh": "DH",
    "can_play_sp": "SP",
    "can_play_rp": "RP",
}


def get_eligible_positions(pos_row) -> list[str]:
    """
    Extract list of eligible positions from a PositionEligibility ORM row.

    Parameters
    ----------
    pos_row : PositionEligibility ORM object with can_play_* boolean flags.

    Returns
    -------
    list[str] -- e.g. ["C", "1B", "DH"]
    """
    positions = []
    for flag_col, pos_key in _POS_FLAG_MAP.items():
        if getattr(pos_row, flag_col, False):
            positions.append(pos_key)
    return positions


def scarcest_replacement_level(positions: list[str]) -> Optional[float]:
    """
    Return the lowest (most negative) replacement level among eligible positions.

    This is the position where the player's VORP is maximized -- the scarcest
    position they can fill.

    Returns None if no eligible positions have a defined replacement level.
    """
    if not positions:
        return None
    levels = [REPLACEMENT_LEVELS[p] for p in positions if p in REPLACEMENT_LEVELS]
    if not levels:
        return None
    return min(levels)


def compute_vorp(composite_z: float, positions: list[str]) -> Optional[float]:
    """
    Compute VORP for a single player.

    VORP = composite_z - replacement_z(scarcest_position)

    Parameters
    ----------
    composite_z : float -- the player's composite Z-score from scoring_engine.
    positions   : list[str] -- eligible positions (e.g. ["C", "1B", "DH"]).

    Returns
    -------
    float or None if no replacement level can be determined.
    """
    repl = scarcest_replacement_level(positions)
    if repl is None:
        return None
    return round(composite_z - repl, 4)


def compute_vorp_batch(
    player_scores: list,
    position_map: dict[int, list[str]],
) -> dict[int, float]:
    """
    Compute VORP for a batch of players.

    Parameters
    ----------
    player_scores : list of PlayerScoreResult (from scoring_engine).
                    Must have .bdl_player_id and .composite_z attributes.
    position_map  : dict mapping bdl_player_id -> list of position strings.
                    Built from PositionEligibility rows via get_eligible_positions().

    Returns
    -------
    dict[int, float] -- bdl_player_id -> VORP value. Players without position
                        data or composite_z are excluded.
    """
    results: dict[int, float] = {}
    for score in player_scores:
        pid = score.bdl_player_id
        positions = position_map.get(pid, [])
        if not positions:
            continue
        vorp = compute_vorp(score.composite_z, positions)
        if vorp is not None:
            results[pid] = vorp
    return results
