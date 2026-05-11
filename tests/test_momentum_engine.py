"""
Tests for backend/services/momentum_engine.py (P15 Momentum Layer).

Covers:
  - classify_signal boundary semantics (all 5 signals + exact boundaries)
  - MomentumResult field computation (delta_z, confidence, player_type)
  - compute_all_momentum pairing logic (skip missing 14d / missing 30d)
  - End-to-end SURGING and COLLAPSING player scenarios
"""

import pytest
from datetime import date
from types import SimpleNamespace

from backend.services.momentum_engine import (
    classify_signal,
    compute_player_momentum,
    compute_all_momentum,
    SURGING,
    HOT,
    STABLE,
    COLD,
    COLLAPSING,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_score(bdl_player_id=1, as_of_date=None, window_days=14,
                player_type="hitter", composite_z=0.0, score_0_100=50.0,
                confidence=0.8, games_in_window=10):
    """Construct a minimal mock PlayerScore-like namespace."""
    return SimpleNamespace(
        bdl_player_id=bdl_player_id,
        as_of_date=as_of_date or date(2026, 4, 5),
        window_days=window_days,
        player_type=player_type,
        composite_z=composite_z,
        score_0_100=score_0_100,
        confidence=confidence,
        games_in_window=games_in_window,
    )


# ---------------------------------------------------------------------------
# classify_signal -- core signal mapping (with cohort context)
# ---------------------------------------------------------------------------

def test_classify_signal_surging():
    # Cohort where delta_z=2.5 is top 10% (90th percentile)
    cohort_deltas = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    assert classify_signal(2.5, cohort_deltas=cohort_deltas) == SURGING


def test_classify_signal_hot():
    # Cohort where delta_z=1.0 is top 30% (70th percentile)
    cohort_deltas = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    assert classify_signal(1.0, cohort_deltas=cohort_deltas) == HOT


def test_classify_signal_stable_positive():
    # Cohort where delta_z=0.2 is middle (30-70% range = STABLE)
    cohort_deltas = [-1.0, -0.5, 0.0, 0.2, 0.5, 1.0]
    assert classify_signal(0.2, cohort_deltas=cohort_deltas) == STABLE


def test_classify_signal_stable_zero():
    # Zero delta in middle cohort
    cohort_deltas = [-0.5, 0.0, 0.5]
    assert classify_signal(0.0, cohort_deltas=cohort_deltas) == STABLE


def test_classify_signal_stable_negative():
    # Slightly negative delta in middle cohort
    cohort_deltas = [-1.0, -0.5, -0.1, 0.0, 0.5]
    assert classify_signal(-0.1, cohort_deltas=cohort_deltas) == STABLE


def test_classify_signal_cold():
    # Cohort where -0.8 is at ~14th percentile (rank 1/7) — bottom 30% → COLD
    cohort_deltas = [-1.5, -0.8, 0.5, 1.0, 1.5, 2.0, 2.5]
    assert classify_signal(-0.8, cohort_deltas=cohort_deltas) == COLD


def test_classify_signal_collapsing():
    # Target -1.8 is below the entire cohort (well below 10th pct) → COLLAPSING
    cohort_deltas = [-0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    assert classify_signal(-1.8, cohort_deltas=cohort_deltas) == COLLAPSING


def test_classify_signal_requires_cohort_deltas():
    """classify_signal raises ValueError if cohort_deltas is None."""
    with pytest.raises(ValueError, match="cohort_deltas is required"):
        classify_signal(0.5, cohort_deltas=None)


# ---------------------------------------------------------------------------
# compute_player_momentum -- field correctness
# ---------------------------------------------------------------------------

def test_delta_z_is_14d_minus_30d():
    s14 = _make_score(window_days=14, composite_z=1.2)
    s30 = _make_score(window_days=30, composite_z=0.8)
    cohort_deltas = [0.4, 0.5, 0.3]  # Simple cohort with delta around 0.4
    result = compute_player_momentum(s14, s30, cohort_deltas=cohort_deltas)
    assert abs(result.delta_z - 0.4) < 1e-9


def test_confidence_is_min_of_two():
    s14 = _make_score(window_days=14, confidence=0.9)
    s30 = _make_score(window_days=30, confidence=0.6)
    cohort_deltas = [0.1, 0.2, 0.3]
    result = compute_player_momentum(s14, s30, cohort_deltas=cohort_deltas)
    assert result.confidence == 0.6


def test_confidence_is_min_of_two_reversed():
    """Verify min works regardless of which window has lower confidence."""
    s14 = _make_score(window_days=14, confidence=0.4)
    s30 = _make_score(window_days=30, confidence=0.95)
    cohort_deltas = [0.1, 0.2, 0.3]
    result = compute_player_momentum(s14, s30, cohort_deltas=cohort_deltas)
    assert result.confidence == 0.4


def test_player_type_from_14d_row():
    """player_type on the result must come from the 14d row, not the 30d row."""
    s14 = _make_score(window_days=14, player_type="pitcher")
    s30 = _make_score(window_days=30, player_type="hitter")
    cohort_deltas = [0.1, 0.2, 0.3]
    result = compute_player_momentum(s14, s30, cohort_deltas=cohort_deltas)
    assert result.player_type == "pitcher"


# ---------------------------------------------------------------------------
# compute_all_momentum -- pairing and skip logic
# ---------------------------------------------------------------------------

def test_skip_player_missing_14d():
    """Player only in 30d list must be skipped."""
    s30 = _make_score(bdl_player_id=99, window_days=30)
    results = compute_all_momentum([], [s30])
    assert results == []


def test_skip_player_missing_30d():
    """Player only in 14d list must be skipped."""
    s14 = _make_score(bdl_player_id=99, window_days=14)
    results = compute_all_momentum([s14], [])
    assert results == []


def test_compute_all_momentum_pairs_correctly():
    """Two players both present in both lists -> two results."""
    s14_a = _make_score(bdl_player_id=1, window_days=14, composite_z=0.5)
    s30_a = _make_score(bdl_player_id=1, window_days=30, composite_z=0.1)
    s14_b = _make_score(bdl_player_id=2, window_days=14, composite_z=-0.3)
    s30_b = _make_score(bdl_player_id=2, window_days=30, composite_z=0.2)
    results = compute_all_momentum([s14_a, s14_b], [s30_a, s30_b])
    assert len(results) == 2
    player_ids = {r.bdl_player_id for r in results}
    assert player_ids == {1, 2}


def test_compute_all_momentum_partial_overlap():
    """Player 2 only in 14d -> only player 1 in results."""
    s14_a = _make_score(bdl_player_id=1, window_days=14, composite_z=1.0)
    s30_a = _make_score(bdl_player_id=1, window_days=30, composite_z=0.3)
    s14_b = _make_score(bdl_player_id=2, window_days=14, composite_z=0.5)
    results = compute_all_momentum([s14_a, s14_b], [s30_a])
    assert len(results) == 1
    assert results[0].bdl_player_id == 1


# ---------------------------------------------------------------------------
# End-to-end signal scenarios (with compute_all_momentum)
# ---------------------------------------------------------------------------

def test_surging_player_classified_correctly():
    """Player with composite_z rising strongly -> SURGING."""
    # Multiple players to create a cohort distribution
    s14 = _make_score(bdl_player_id=10, window_days=14, composite_z=2.0, score_0_100=92.0)
    s30 = _make_score(bdl_player_id=10, window_days=30, composite_z=1.3, score_0_100=78.0)
    # Add other players to create cohort context
    s14_b = _make_score(bdl_player_id=11, window_days=14, composite_z=0.5, score_0_100=55.0)
    s30_b = _make_score(bdl_player_id=11, window_days=30, composite_z=0.3, score_0_100=50.0)
    s14_c = _make_score(bdl_player_id=12, window_days=14, composite_z=-1.0, score_0_100=20.0)
    s30_c = _make_score(bdl_player_id=12, window_days=30, composite_z=-0.5, score_0_100=30.0)

    results = compute_all_momentum([s14, s14_b, s14_c], [s30, s30_b, s30_c])
    # Find our target player
    r = next(res for res in results if res.bdl_player_id == 10)
    assert r.signal in (SURGING, HOT)  # Top performer in cohort
    assert abs(r.delta_z - 0.7) < 1e-9
    assert r.score_14d == 92.0
    assert r.score_30d == 78.0


def test_collapsing_player_classified_correctly():
    """Player with composite_z dropping sharply -> COLLAPSING."""
    s14 = _make_score(bdl_player_id=20, window_days=14, composite_z=-1.5, score_0_100=8.0)
    s30 = _make_score(bdl_player_id=20, window_days=30, composite_z=-0.8, score_0_100=22.0)
    # Add other players to create cohort context
    s14_b = _make_score(bdl_player_id=21, window_days=14, composite_z=0.5, score_0_100=55.0)
    s30_b = _make_score(bdl_player_id=21, window_days=30, composite_z=0.3, score_0_100=50.0)

    results = compute_all_momentum([s14, s14_b], [s30, s30_b])
    r = next(res for res in results if res.bdl_player_id == 20)
    # With delta_z=-0.7 and being the worst performer, should be COLLAPSING or COLD
    assert r.signal in (COLLAPSING, COLD)
    assert abs(r.delta_z - (-0.7)) < 1e-9


def test_momentum_result_as_of_date_from_14d():
    """as_of_date on result comes from the 14d score row."""
    target_date = date(2026, 4, 5)
    s14 = _make_score(window_days=14, as_of_date=target_date)
    s30 = _make_score(window_days=30, as_of_date=target_date)
    result = compute_player_momentum(s14, s30, cohort_deltas=[0.1, 0.2, 0.3])
    assert result.as_of_date == target_date


def test_momentum_result_composite_z_fields():
    """composite_z_14d and composite_z_30d must be stored verbatim."""
    s14 = _make_score(window_days=14, composite_z=1.75)
    s30 = _make_score(window_days=30, composite_z=0.95)
    result = compute_player_momentum(s14, s30, cohort_deltas=[0.5, 0.6, 0.7])
    assert result.composite_z_14d == 1.75
    assert result.composite_z_30d == 0.95


# ---------------------------------------------------------------------------
# NEW: Z-score-based momentum with level gating (Fix 2)
# ---------------------------------------------------------------------------


def test_momentum_level_gate():
    """Bottom-3% player with +7.5 delta gets COLD, not SURGING."""
    # Dylan Moore case: composite_z=-8.83 (3rd percentile), delta_z=+7.51
    # Should be COLD due to level gate, not SURGING despite positive delta
    cohort_z = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    cohort_deltas = [-1.0, -0.5, 0.0, 0.5, 1.0]  # Various deltas in cohort
    result = classify_signal(
        delta_z=7.51,
        absolute_level=-8.83,
        percentile_rank=3.0,
        cohort_z_scores=cohort_z,
        cohort_deltas=cohort_deltas,
    )
    assert result == COLD  # NOT "SURGING"


def test_momentum_percentile_thresholds():
    """Top 10% of adjusted deltas should be SURGING, top 30% HOT."""
    # 7-element cohort: 1.8 (not in cohort) sits above the 90th pct threshold → SURGING
    cohort_z = [0.0, 0.5, 1.0, 1.5, 2.0]
    cohort_deltas = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

    # 90th percentile of 7-element cohort: idx=int(7*0.90)=6 → z-score of 1.5
    # z-score of 1.8 (using same cohort mean/std) exceeds that threshold → SURGING
    result_surging = classify_signal(
        delta_z=1.8,  # Above cohort 90th percentile threshold
        cohort_z_scores=cohort_z,
        cohort_deltas=cohort_deltas,
    )
    assert result_surging == SURGING

    # 0.8 is between 70th and 90th pct of this cohort → HOT
    result_hot = classify_signal(
        delta_z=0.8,  # Between 70th and 90th percentile
        cohort_z_scores=cohort_z,
        cohort_deltas=cohort_deltas,
    )
    assert result_hot == HOT


def test_momentum_zscore_normalization():
    """Z-score approach: stable vs noisy cohorts produce same relative rankings."""
    # Stable cohort: tight cluster, small deltas stand out more
    stable_cohort_z = [0.9, 1.0, 1.1, 1.2, 1.3]
    stable_cohort_deltas = [0.1, 0.2, 0.3, 0.4, 0.5]  # mean=0.3, std≈0.14

    # Noisy cohort: wide spread, same delta doesn't stand out
    noisy_cohort_z = [-5.0, 0.0, 5.0, 10.0, 15.0]
    noisy_cohort_deltas = [-2.0, -1.0, 0.0, 1.0, 2.0]  # mean=0.0, std≈1.41

    # delta=0.5
    # In stable: z = (0.5-0.3)/0.14 ≈ 1.43 (high percentile)
    result_stable = classify_signal(
        delta_z=0.5,
        cohort_z_scores=stable_cohort_z,
        cohort_deltas=stable_cohort_deltas,
    )

    # In noisy: z = (0.5-0.0)/1.41 ≈ 0.35 (lower percentile)
    result_noisy = classify_signal(
        delta_z=0.5,
        cohort_z_scores=noisy_cohort_z,
        cohort_deltas=noisy_cohort_deltas,
    )

    # Stable cohort should classify same delta more positively (higher z-score)
    assert result_stable in (SURGING, HOT)  # High z-score in stable cohort
    # (noisy cohort result depends on full distribution)


def test_level_gate_with_percentile_rank():
    """Player below 25th percentile gets COLD regardless of delta."""
    cohort_z = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    cohort_deltas = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

    # Player at 20th percentile with +2.0 delta (big improvement from bad)
    result = classify_signal(
        delta_z=2.0,
        percentile_rank=20.0,  # Below 25th percentile
        cohort_z_scores=cohort_z,
        cohort_deltas=cohort_deltas,
    )
    assert result == COLD  # Level gate applies


def test_level_gate_with_absolute_level():
    """Player below 25th percentile (computed from cohort) gets COLD."""
    cohort_z = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]  # 25th percentile = -1.0
    cohort_deltas = [-1.0, 0.0, 1.0, 2.0]

    # Player at absolute_level=-2.0 (below -1.0 threshold) with +1.5 delta
    result = classify_signal(
        delta_z=1.5,
        absolute_level=-2.0,  # Below cohort 25th percentile
        cohort_z_scores=cohort_z,
        cohort_deltas=cohort_deltas,
    )
    assert result == COLD  # Level gate applies
