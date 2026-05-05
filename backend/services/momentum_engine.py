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

from backend.services.config_service import get_threshold as _get_threshold

# ---------------------------------------------------------------------------
# Signal constants
# ---------------------------------------------------------------------------

SURGING    = "SURGING"
HOT        = "HOT"
STABLE     = "STABLE"
COLD       = "COLD"
COLLAPSING = "COLLAPSING"

# ---------------------------------------------------------------------------
# Threshold constants (DEPRECATED - replaced by percentile-based approach)
# ---------------------------------------------------------------------------
# These hardcoded thresholds are no longer used. classify_signal() now uses
# z-score-based, cohort-relative percentile thresholds computed from cohort_deltas.

SURGING_THRESHOLD: float    = _get_threshold("momentum.surging.delta_z",    default=0.5)   # DEPRECATED
HOT_THRESHOLD: float        = _get_threshold("momentum.hot.delta_z",         default=0.2)   # DEPRECATED
COLD_THRESHOLD: float       = _get_threshold("momentum.cold.delta_z",        default=-0.2)  # DEPRECATED
COLLAPSING_THRESHOLD: float = _get_threshold("momentum.collapsing.delta_z",  default=-0.5)  # DEPRECATED

# ---------------------------------------------------------------------------
# Percentile-based momentum configuration (documented constants)
# ---------------------------------------------------------------------------
# All thresholds are computed as percentiles of the cohort's delta_z distribution
# after normalizing to z-scores. This makes signals cohort-relative and accounts
# for volatility.

MOMENTUM_TOP_PCT_SURGING: float = _get_threshold("momentum.top_pct.surging", default=0.90)
MOMENTUM_TOP_PCT_HOT: float = _get_threshold("momentum.top_pct.hot", default=0.70)
MOMENTUM_BOT_PCT_COLD: float = _get_threshold("momentum.bot_pct.cold", default=0.30)
MOMENTUM_BOT_PCT_COLLAPSING: float = _get_threshold("momentum.bot_pct.collapsing", default=0.10)

# Level gate: bottom quartile (by definition) cannot be SURGING
LEVEL_GATE_PCT_BOTTOM_QUARTILE: float = _get_threshold(
    "momentum.level_gate.bottom_quartile", default=25.0
)


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

def classify_signal(
    delta_z: float,
    absolute_level: float | None = None,
    percentile_rank: float | None = None,
    cohort_z_scores: list[float] | None = None,
    cohort_deltas: list[float] | None = None,
) -> str:
    """
    Return momentum signal with absolute level gating and z-score-based,
    percentile-based thresholds.

    Level gating: Players in bottom quartile (by definition) cannot be "SURGING"
    even with large positive delta—they're improving from unplayable to bad.

    Z-score approach: All cohort_deltas are normalized to z-scores before computing
    percentiles. This makes thresholds cohort-relative and accounts for volatility.

    Parameters
    ----------
    delta_z : float
        Change in composite_z (14d - 30d)
    absolute_level : float, optional
        Current composite_z (14d). Used for level gating.
    percentile_rank : float, optional
        Current percentile rank (0-100) within cohort. Used for level gating.
    cohort_z_scores : list[float], optional
        All composite_z values in cohort for computing 25th percentile if percentile_rank not available.
    cohort_deltas : list[float], REQUIRED
        All delta_z values in cohort. REQUIRED—raises error if None.

    Returns
    -------
    str -- SURGING, HOT, STABLE, COLD, COLLAPSING

    Raises
    ------
    ValueError: If cohort_deltas is None (cannot compute cohort-relative thresholds)
    """
    # ------------------------------------------------------------------
    # REQUIREMENT: cohort_deltas is mandatory (no fallback to hardcoded thresholds)
    # ------------------------------------------------------------------
    if cohort_deltas is None:
        raise ValueError(
            "cohort_deltas is required for classify_signal. "
            "Cannot compute cohort-relative thresholds without cohort data."
        )

    # ------------------------------------------------------------------
    # Level gate: 25th percentile is by definition (quartile boundary)
    # This is not a tunable parameter—it's the mathematical definition of "bottom quartile"
    # ------------------------------------------------------------------
    if percentile_rank is not None and percentile_rank < LEVEL_GATE_PCT_BOTTOM_QUARTILE:
        # Bottom quartile by definition—cannot be surging regardless of delta
        return COLD
    if absolute_level is not None and cohort_z_scores is not None:
        # Compute 25th percentile from cohort if percentile_rank not available
        cohort_sorted = sorted(cohort_z_scores)
        p25_index = int(len(cohort_sorted) * 0.25)
        p25_threshold = cohort_sorted[p25_index]
        if absolute_level <= p25_threshold:
            return COLD

    # ------------------------------------------------------------------
    # Z-score approach: Normalize all deltas by cohort spread before percentile comparison
    # ------------------------------------------------------------------
    import math

    mean_delta = sum(cohort_deltas) / len(cohort_deltas)
    variance_delta = sum((d - mean_delta) ** 2 for d in cohort_deltas) / len(cohort_deltas)
    std_delta = math.sqrt(variance_delta) if variance_delta > 0 else 1.0

    # Normalize current delta to z-score
    z_score_delta = (delta_z - mean_delta) / std_delta if std_delta > 0 else 0.0

    # Normalize all cohort deltas to z-scores for percentile comparison
    z_scores = [(d - mean_delta) / std_delta for d in cohort_deltas]
    sorted_z_scores = sorted(z_scores)

    # Compute percentile thresholds from z-scores (cohort-relative)
    n = len(sorted_z_scores)
    idx_surging = min(n - 1, int(n * MOMENTUM_TOP_PCT_SURGING))  # 90th percentile
    idx_hot = min(n - 1, int(n * MOMENTUM_TOP_PCT_HOT))  # 70th percentile
    idx_cold = min(n - 1, int(n * MOMENTUM_BOT_PCT_COLD))  # 30th percentile
    idx_collapsing = min(n - 1, int(n * MOMENTUM_BOT_PCT_COLLAPSING))  # 10th percentile

    z_surging = sorted_z_scores[idx_surging]
    z_hot = sorted_z_scores[idx_hot]
    z_cold = sorted_z_scores[idx_cold]
    z_collapsing = sorted_z_scores[idx_collapsing]

    # Classify using z-score comparison
    if z_score_delta >= z_surging:
        return SURGING
    if z_score_delta >= z_hot:
        return HOT
    if z_score_delta >= z_cold:
        return STABLE
    if z_score_delta >= z_collapsing:
        return COLD
    return COLLAPSING


def compute_player_momentum(
    score_14d,
    score_30d,
    cohort_z_scores: list[float] | None = None,
    cohort_deltas: list[float] | None = None,
) -> MomentumResult:
    """
    Compute a MomentumResult from two PlayerScore ORM rows (14d and 30d).

    Parameters
    ----------
    score_14d : PlayerScore ORM instance with window_days == 14
    score_30d : PlayerScore ORM instance with window_days == 30
    cohort_z_scores : list[float], optional
        All composite_z values in cohort for level gating (25th percentile computation)
    cohort_deltas : list[float], REQUIRED
        All delta_z values in cohort for computing cohort-relative thresholds

    Both rows must share the same bdl_player_id and as_of_date.

    Returns
    -------
    MomentumResult with delta_z, signal, and blended confidence.

    Raises
    ------
    ValueError: If cohort_deltas is None (required for cohort-relative thresholds)
    """
    delta_z = score_14d.composite_z - score_30d.composite_z
    signal = classify_signal(
        delta_z=delta_z,
        absolute_level=score_14d.composite_z,
        percentile_rank=score_14d.score_0_100,  # score_0_100 is percentile rank (0-100)
        cohort_z_scores=cohort_z_scores,
        cohort_deltas=cohort_deltas,
    )
    conf = min(score_14d.confidence, score_30d.confidence)

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

    Computes cohort-relative thresholds using all delta_z values in the current batch.

    Parameters
    ----------
    scores_14d : list of PlayerScore ORM rows with window_days == 14
    scores_30d : list of PlayerScore ORM rows with window_days == 30

    Returns
    -------
    list[MomentumResult] -- one entry per player present in BOTH lists.
    """
    index_30d: dict = {row.bdl_player_id: row for row in scores_30d}

    # Build cohort data for z-score-based, cohort-relative thresholds
    # cohort_z_scores: all composite_z values from 14d scores (for level gating)
    # cohort_deltas: all delta_z values (14d - 30d) for volatility adjustment
    cohort_z_scores: list[float] = [row.composite_z for row in scores_14d if row.composite_z is not None]
    cohort_deltas: list[float] = []

    # First pass: compute all delta_z values for cohort_deltas
    index_14d: dict = {row.bdl_player_id: row for row in scores_14d}
    for row_30 in scores_30d:
        row_14 = index_14d.get(row_30.bdl_player_id)
        if row_14 is not None and row_14.composite_z is not None and row_30.composite_z is not None:
            delta_z = row_14.composite_z - row_30.composite_z
            cohort_deltas.append(delta_z)

    # Second pass: compute momentum with cohort context
    results: list = []
    for row_14 in scores_14d:
        row_30 = index_30d.get(row_14.bdl_player_id)
        if row_30 is None:
            continue
        results.append(
            compute_player_momentum(
                row_14,
                row_30,
                cohort_z_scores=cohort_z_scores,
                cohort_deltas=cohort_deltas,
            )
        )

    return results
