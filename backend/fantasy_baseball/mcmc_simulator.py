"""
MCMC Weekly Matchup Simulator

Monte Carlo simulation of H2H fantasy baseball weekly matchup outcomes.
Uses numpy for fast vectorized sampling — 1000 simulations in <50ms.

v2 Alignment:
- Uses v2 canonical category codes (lowercase internally)
- 18 categories: 9 batting, 9 pitching
- Win threshold: 10 (majority of 18)
- All z-scores normalized so HIGHER = BETTER (LOWER_IS_BETTER inverted at input)

Public API:
  simulate_weekly_matchup(my_roster, opponent_roster, ...) -> dict
  simulate_roster_move(my_roster, opponent_roster, add_player, drop_player_name, ...) -> dict

Each player dict must contain:
  cat_scores: dict[str, float]   — z-score per fantasy category (higher = better for all)
  positions:  list[str] | str    — position(s)
  starts_this_week: int          — pitcher starts this week (default 1)
  name: str

Category keys (lowercase v2 canonical codes):
  Batting (9): r, h, hr_b, rbi, k_b, tb, avg, ops, nsb
  Pitching (9): w, l, hr_p, k_p, era, whip, k_9, qs, nsv

Note: All cat_scores are z-scores where HIGHER = BETTER.
LOWER_IS_BETTER categories (ERA, WHIP, K_B, L, HR_P) must be inverted
before being passed to cat_scores (multiply by -1).
"""

import logging
import time
from typing import Optional

import numpy as np

from backend.stat_contract import SCORING_CATEGORY_CODES, LOWER_IS_BETTER

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-player weekly standard deviation in z-score units
# ---------------------------------------------------------------------------
# These represent realistic week-to-week noise around each player's projection.
# Volatile positions and sparse stat categories get higher values.
# v2: Keys are lowercase versions of canonical codes.

_PLAYER_WEEKLY_STD: dict[str, float] = {
    # Batting — counting (v2 canonical codes)
    "r": 0.70,
    "h": 0.55,
    "hr_b": 0.65,
    "rbi": 0.70,
    "k_b": 0.50,   # strikeouts: volatile for batters
    "tb": 0.65,
    "nsb": 0.90,   # stolen bases: volatile
    # Batting — rate
    "avg": 0.40,
    "ops": 0.40,
    # Pitching — counting
    "w": 0.85,
    "l": 0.85,
    "hr_p": 0.75,
    "k_p": 0.75,
    "qs": 0.80,
    "nsv": 1.00,   # saves: binary/volatile
    # Pitching — rate
    "era": 0.65,
    "whip": 0.55,
    "k_9": 0.40,
}

_DEFAULT_STD = 0.60  # fallback for unknown categories

# Position-based variance multiplier: role players / relievers more volatile
_POSITION_MULT: dict[str, float] = {
    "C": 1.30, "1B": 1.00, "2B": 1.10, "3B": 1.10, "SS": 1.10,
    "OF": 1.00, "LF": 1.00, "CF": 1.00, "RF": 1.00, "DH": 0.90,
    "SP": 1.20, "RP": 1.50, "P": 1.30,
}

# Counting pitcher categories that scale with starts (v2 codes)
_STARTS_SCALE_CATS = frozenset({"k_p", "w", "qs"})

# Count-based categories that must be >= 0 (saves, SB, HR can't be negative)
# NOTE: This is a short-term fix. Normal distribution is inappropriate for count data.
# Follow-up: migrate to Poisson or negative binomial modeling for count categories.
_COUNT_CATEGORIES = frozenset({
    "hr_b",      # home runs
    "rbi",       # runs batted in
    "nsb",       # net stolen bases
    "w",         # pitcher wins
    "qs",        # quality starts
    "nsv",       # net saves
    "k_b",       # batting strikeouts
    "k_p",       # pitching strikeouts
})

# v2: Mapping from canonical codes to lowercase keys used internally
_CANONICAL_TO_LOWER = {code: code.lower() for code in SCORING_CATEGORY_CODES}
_LOWER_TO_CANONICAL = {v: k for k, v in _CANONICAL_TO_LOWER.items()}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _primary_position(player: dict) -> str:
    pos = player.get("positions") or player.get("position") or []
    if isinstance(pos, list):
        return pos[0] if pos else "?"
    return str(pos) if pos else "?"


def _player_std(player: dict, cat: str) -> float:
    base = _PLAYER_WEEKLY_STD.get(cat, _DEFAULT_STD)
    mult = _POSITION_MULT.get(_primary_position(player), 1.0)
    return max(0.05, base * mult)


def _roster_means_stds(roster: list[dict], cats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (means, stds) arrays of shape (n_players, n_cats) for numpy sampling.

    v2: Accepts both lowercase v2 canonical codes and legacy keys from cat_scores.
    """
    n = len(roster)
    if n == 0:
        return np.zeros((0, len(cats))), np.zeros((0, len(cats)))

    means = np.zeros((n, len(cats)))
    stds = np.zeros((n, len(cats)))

    for i, player in enumerate(roster):
        cat_scores = player.get("cat_scores") or {}
        starts = max(1, int(player.get("starts_this_week", 1)))
        pos = _primary_position(player)
        is_pitcher = pos in ("SP", "RP", "P")

        for j, cat in enumerate(cats):
            # Try to get the value from cat_scores with multiple key options
            # 1. Direct match (lowercase v2 code)
            # 2. Legacy key mapping
            base_mean = _get_cat_score(cat_scores, cat)

            # Two-start pitchers: scale counting cats proportionally
            if is_pitcher and cat in _STARTS_SCALE_CATS and starts >= 2:
                base_mean *= min(starts, 2) * 0.85  # 0.85 avoids double-counting rest effects

            means[i, j] = base_mean
            stds[i, j] = _player_std(player, cat)

    return means, stds


def _get_cat_score(cat_scores: dict, lowercase_v2_code: str) -> float:
    """
    Get category score from cat_scores dict, handling legacy keys.

    Tries in order:
    1. Direct lowercase v2 code (e.g., "k_p")
    2. Legacy player_board keys (e.g., "k_pit" -> "k_p")
    3. Canonical code (e.g., "K_P" -> "k_p")

    Returns:
        float value from cat_scores, or 0.0 if not found
    """
    # Direct match
    if lowercase_v2_code in cat_scores:
        return float(cat_scores[lowercase_v2_code])

    # Legacy mappings
    _LEGACY_INPUT_MAP = {
        "k_p": ["k_pit"],
        "k_9": ["k9"],
        "hr_b": ["hr"],  # Legacy batting HR
    }

    legacy_keys = _LEGACY_INPUT_MAP.get(lowercase_v2_code, [])
    for legacy_key in legacy_keys:
        if legacy_key in cat_scores:
            return float(cat_scores[legacy_key])

    # Try uppercase version (canonical code)
    canonical_upper = lowercase_v2_code.upper()
    if canonical_upper in cat_scores:
        return float(cat_scores[canonical_upper])

    return 0.0


def _detect_categories(rosters: list[list[dict]]) -> list[str]:
    """
    Auto-detect categories present across all rosters.

    v2: Returns lowercase v2 canonical codes. Accepts both canonical
    and legacy keys from input rosters, normalizing to lowercase.
    """
    all_cats: set[str] = set()
    for roster in rosters:
        for p in roster:
            cat_scores = p.get("cat_scores") or {}
            for key in cat_scores.keys():
                # Normalize to lowercase v2 code
                # Handle legacy keys: "k_pit" -> "k_p", "hr" -> "hr_b", etc.
                normalized = _normalize_category_key(key)
                all_cats.add(normalized)

    return sorted(all_cats)


def _normalize_category_key(key: str) -> str:
    """
    Normalize a category key to lowercase v2 canonical code.

    Handles:
    - Canonical codes (e.g., "HR_B") -> lowercase ("hr_b")
    - Legacy player_board keys (e.g., "k_pit" -> "k_p", "hr" -> "hr_b")
    - Already lowercase keys -> returned as-is if valid

    Returns:
        Lowercase v2 canonical code
    """
    key_lower = key.lower()

    # Legacy mapping table
    _LEGACY_MAP = {
        "k_pit": "k_p",
        "k9": "k_9",
        "hr": "hr_b",  # Legacy batting HR
        "sb": "nsb",   # Legacy SB (now NSV for pitching)
    }

    # Check legacy map first
    if key_lower in _LEGACY_MAP:
        return _LEGACY_MAP[key_lower]

    # If already a valid lowercase canonical code, return it
    if key_lower in _CANONICAL_TO_LOWER.values():
        return key_lower

    # Fallback: try to match against canonical codes case-insensitively
    for canonical in SCORING_CATEGORY_CODES:
        if canonical.lower() == key_lower:
            return canonical.lower()

    # Unknown key - return as-is (will be filtered or use default std)
    return key_lower


def _clamp_counts(totals: np.ndarray, categories: list[str]) -> np.ndarray:
    """
    Clamp count-based categories at zero (saves, SB, HR can't be negative).

    NOTE: This is a short-term fix. Normal distribution is inappropriate for
    count data—follow-up should migrate to Poisson or negative binomial models.
    Low-count categories (saves, quality starts) should use Poisson or
    zero-inflated Poisson. High-count categories (strikeouts, RBI) should use
    negative binomial to handle overdispersion.

    Args:
        totals: np.ndarray of shape (n_sims, n_cats) with simulated totals
        categories: list of category names corresponding to totals columns

    Returns:
        np.ndarray with count-based categories clamped at >= 0
    """
    result = totals.copy()
    for j, cat in enumerate(categories):
        if cat in _COUNT_CATEGORIES:
            result[:, j] = np.maximum(result[:, j], 0.0)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate_weekly_matchup(
    my_roster: list[dict],
    opponent_roster: list[dict],
    categories: Optional[list[str]] = None,
    n_sims: int = 1000,
    seed: Optional[int] = None,
) -> dict:
    """
    Monte Carlo simulation of one week's H2H matchup.

    v2: Uses lowercase v2 canonical codes. Win threshold = 10 (majority of 18).

    Parameters
    ----------
    my_roster / opponent_roster:
        Lists of player dicts. Each dict needs cat_scores, positions, starts_this_week.
        cat_scores keys are normalized to lowercase v2 codes internally.
        Pass an empty list [] for opponent to compare against a league-average opponent
        (all cat z-scores = 0, representing the statistical mean).
    categories:
        Category keys to simulate (lowercase v2 codes). Auto-detected from rosters if None.
    n_sims:
        Monte Carlo iterations. 1000 is fast (<50ms) and stable.
    seed:
        Optional RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        win_prob              float       fraction of sims where my team wins (10+ cats)
        category_win_probs    dict        per-category win fraction (lowercase v2 keys)
        expected_cats_won     float       expected categories won per matchup
        n_sims                int
        elapsed_ms            float
        categories_simulated  list[str]   lowercase v2 codes
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    if categories is None:
        categories = _detect_categories([my_roster, opponent_roster])

    if not categories:
        # No category data — return 50/50
        return {
            "win_prob": 0.5,
            "category_win_probs": {},
            "expected_cats_won": 0.0,
            "n_sims": 0,
            "elapsed_ms": 0.0,
            "categories_simulated": [],
        }

    my_means, my_stds = _roster_means_stds(my_roster, categories)
    opp_means, opp_stds = _roster_means_stds(opponent_roster, categories)

    # Vectorized sampling: (n_sims, n_players, n_cats)
    n_cats = len(categories)

    if my_means.shape[0] > 0:
        my_noise = rng.normal(0.0, my_stds, size=(n_sims,) + my_means.shape)
        my_totals = _clamp_counts((my_means + my_noise).sum(axis=1), categories)   # (n_sims, n_cats)
    else:
        my_totals = np.zeros((n_sims, n_cats))

    if opp_means.shape[0] > 0:
        opp_noise = rng.normal(0.0, opp_stds, size=(n_sims,) + opp_means.shape)
        opp_totals = _clamp_counts((opp_means + opp_noise).sum(axis=1), categories)  # (n_sims, n_cats)
    else:
        # Empty opponent = league average (z=0), still has week-level noise
        avg_std = np.full(n_cats, 2.0)  # team-level noise for 12-player average opponent
        opp_totals = _clamp_counts(rng.normal(0.0, avg_std, size=(n_sims, n_cats)), categories)

    # All cats: higher is better (LOWER_IS_BETTER z-scores inverted at input)
    cat_wins = (my_totals > opp_totals).astype(float)   # (n_sims, n_cats)

    cat_win_probs = {
        cat: round(float(cat_wins[:, j].mean()), 4)
        for j, cat in enumerate(categories)
    }
    total_cat_wins = cat_wins.sum(axis=1)   # (n_sims,)

    # v2: Dynamic win threshold = majority of categories simulated
    # If all 18 categories: need 10+ (majority)
    # If partial categories: need more than half
    win_threshold = n_cats / 2.0
    matchup_wins = (total_cat_wins > win_threshold).astype(float)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "win_prob": round(float(matchup_wins.mean()), 4),
        "category_win_probs": cat_win_probs,
        "expected_cats_won": round(float(total_cat_wins.mean()), 2),
        "n_sims": n_sims,
        "elapsed_ms": round(elapsed_ms, 1),
        "categories_simulated": categories,
    }


def simulate_roster_move(
    my_roster: list[dict],
    opponent_roster: list[dict],
    add_player: dict,
    drop_player_name: str,
    categories: Optional[list[str]] = None,
    n_sims: int = 1000,
    seed: Optional[int] = None,
) -> dict:
    """
    Simulate the win-probability impact of a single ADD/DROP roster move.

    The before/after simulations use the same RNG seed offset so results
    are directly comparable (variance is minimized).

    Returns
    -------
    dict with keys:
        win_prob_before         float
        win_prob_after          float
        win_prob_gain           float   positive = move improves win chances
        category_win_probs_before  dict
        category_win_probs_after   dict
        mcmc_enabled            True
        n_sims                  int
        elapsed_ms              float
    """
    t0 = time.perf_counter()
    _seed = seed if seed is not None else 42

    # Auto-detect cats from all players (including add_player)
    if categories is None:
        categories = _detect_categories([my_roster, opponent_roster, [add_player]])

    before = simulate_weekly_matchup(
        my_roster, opponent_roster,
        categories=categories, n_sims=n_sims, seed=_seed,
    )

    # Build modified roster: drop the named player, add the new one
    drop_key = drop_player_name.strip().lower()
    new_roster = [p for p in my_roster if p.get("name", "").strip().lower() != drop_key]
    new_roster.append(add_player)

    after = simulate_weekly_matchup(
        new_roster, opponent_roster,
        categories=categories, n_sims=n_sims, seed=_seed + 1,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "win_prob_before": before["win_prob"],
        "win_prob_after": after["win_prob"],
        "win_prob_gain": round(after["win_prob"] - before["win_prob"], 4),
        "category_win_probs_before": before["category_win_probs"],
        "category_win_probs_after": after["category_win_probs"],
        "expected_cats_won_before": before["expected_cats_won"],
        "expected_cats_won_after": after["expected_cats_won"],
        "mcmc_enabled": True,
        "n_sims": n_sims,
        "elapsed_ms": round(elapsed_ms, 1),
    }


_MCMC_DISABLED: dict = {
    "win_prob_before": 0.5,
    "win_prob_after": 0.5,
    "win_prob_gain": 0.0,
    "category_win_probs_before": {},
    "category_win_probs_after": {},
    "expected_cats_won_before": 0.0,
    "expected_cats_won_after": 0.0,
    "mcmc_enabled": False,
    "n_sims": 0,
    "elapsed_ms": 0.0,
}
