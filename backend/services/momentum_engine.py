"""
P15 Momentum Engine -- pure computation module (no I/O, no DB access).

Computes delta-Z momentum signals by comparing 14-day vs 30-day composite
Z-scores from the player_scores table (populated by Phase 4 scoring engine).

Design principles:
  - Zero I/O: all functions are pure transforms. Callers own DB queries.
  - Stateless: no module-level mutable state.
  - Deterministic: given same inputs, always returns same outputs.
  - Signal thresholds use exact boundary semantics as specified:
      delta_z >  0.5  -> SURGING    (strictly greater than)
      delta_z >= 0.2  -> HOT        (includes 0.2, excludes 0.5)
      delta_z >  -0.2 -> STABLE     (excludes -0.2 and 0.2)
      delta_z >= -0.5 -> COLD       (includes -0.2 and -0.5, excludes above)
      else            -> COLLAPSING (strictly less than -0.5)

Consumed by:
  - daily_ingestion._compute_player_momentum() (lock 100_020, 5 AM ET)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

# ---------------------------------------------------------------------------
# Signal constants
# ---------------------------------------------------------------------------

SURGING    = "SURGING"
HOT        = "HOT"
STABLE     = "STABLE"
COLD       = "COLD"
COLLAPSING = "COLLAPSING"

# ---------------------------------------------------------------------------
# Threshold constants (boundary semantics defined in module docstring)
# ---------------------------------------------------------------------------

SURGING_THRESHOLD    =  0.5
HOT_THRESHOLD        =  0.2
COLD_THRESHOLD       = -0.2
COLLAPSING_THRESHOLD = -0.5


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

@dataclass
class MomentumResult:
    """
    Momentum computation output for a single player on a single date.

    Fields:
      bdl_player_id  -- BallDontLie integer player ID
      as_of_date     -- date the underlying scores were computed (yesterday)
      player_type    -- "hitter", "pitcher", or "two_way" (from 14d score row)
      delta_z        -- Z_14d - Z_30d (positive = improving)
      signal         -- one of SURGING / HOT / STABLE / COLD / COLLAPSING
      composite_z_14d -- composite_z from the 14-day score row
      composite_z_30d -- composite_z from the 30-day score row
      score_14d      -- score_0_100 from 14-day row
      score_30d      -- score_0_100 from 30-day row
      confidence_14d -- confidence from 14-day row
      confidence_30d -- confidence from 30-day row
      confidence     -- min(confidence_14d, confidence_30d)
    """
    bdl_player_id:   int
    as_of_date:      date
    player_type:     str
    delta_z:         float
    signal:          str
    composite_z_14d: float
    composite_z_30d: float
    score_14d:       float
    score_30d:       float
    confidence_14d:  float
    confidence_30d:  float
    confidence:      float


# ---------------------------------------------------------------------------
# Core pure functions
# ---------------------------------------------------------------------------

def classify_signal(delta_z: float) -> str:
    """
    Map a delta-Z value to one of the five momentum signal strings.

    Boundary semantics (see module docstring for rationale):
      >  0.5  -> SURGING
      >= 0.2  -> HOT
      >  -0.2 -> STABLE
      >= -0.5 -> COLD
      else    -> COLLAPSING
    """
    if delta_z > SURGING_THRESHOLD:
        return SURGING
    if delta_z >= HOT_THRESHOLD:
        return HOT
    if delta_z > COLD_THRESHOLD:
        return STABLE
    if delta_z >= COLLAPSING_THRESHOLD:
        return COLD
    return COLLAPSING


def compute_player_momentum(score_14d, score_30d) -> MomentumResult:
    """
    Compute a MomentumResult from two PlayerScore ORM rows (14d and 30d).

    Parameters
    ----------
    score_14d : PlayerScore ORM instance with window_days == 14
    score_30d : PlayerScore ORM instance with window_days == 30

    Both rows must share the same bdl_player_id and as_of_date.

    Returns
    -------
    MomentumResult with delta_z, signal, and blended confidence.
    """
    delta_z = score_14d.composite_z - score_30d.composite_z
    signal  = classify_signal(delta_z)
    conf    = min(score_14d.confidence, score_30d.confidence)

    return MomentumResult(
        bdl_player_id=score_14d.bdl_player_id,
        as_of_date=score_14d.as_of_date,
        player_type=score_14d.player_type,
        delta_z=delta_z,
        signal=signal,
        composite_z_14d=score_14d.composite_z,
        composite_z_30d=score_30d.composite_z,
        score_14d=score_14d.score_0_100,
        score_30d=score_30d.score_0_100,
        confidence_14d=score_14d.confidence,
        confidence_30d=score_30d.confidence,
        confidence=conf,
    )


def compute_all_momentum(scores_14d: list, scores_30d: list) -> list:
    """
    Pair 14d and 30d score rows by bdl_player_id and compute momentum for all matches.

    Players present in only one list are silently skipped (insufficient data).

    Parameters
    ----------
    scores_14d : list of PlayerScore ORM rows with window_days == 14
    scores_30d : list of PlayerScore ORM rows with window_days == 30

    Returns
    -------
    list[MomentumResult] -- one entry per player present in BOTH lists.
    """
    index_30d: dict = {row.bdl_player_id: row for row in scores_30d}

    results: list = []
    for row_14 in scores_14d:
        row_30 = index_30d.get(row_14.bdl_player_id)
        if row_30 is None:
            continue
        results.append(compute_player_momentum(row_14, row_30))

    return results
