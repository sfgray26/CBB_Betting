"""
Tests for P18 Backtesting Harness -- pure computation module.

All tests are DB-free.  They exercise:
  - compute_mae / compute_rmse edge cases
  - evaluate_player composite logic per player_type
  - evaluate_cohort filtering of unknown player_type
  - summarize regression detection
  - golden baseline round-trip (save then load)
"""

import json
import os
from datetime import date

import pytest

from backend.services.backtesting_harness import (
    BacktestInput,
    BacktestResult,
    BacktestSummary,
    compute_mae,
    compute_rmse,
    evaluate_cohort,
    evaluate_player,
    load_golden_baseline,
    save_golden_baseline,
    summarize,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TODAY = date(2026, 4, 6)
YESTERDAY = date(2026, 4, 5)


def _make_hitter_input(
    pid=1,
    proj_hr=30.0, proj_rbi=90.0, proj_sb=10.0, proj_avg=0.280,
    actual_hr=28.0, actual_rbi=85.0, actual_sb=9.0, actual_avg=0.265,
    games=14,
) -> BacktestInput:
    return BacktestInput(
        bdl_player_id=pid,
        as_of_date=YESTERDAY,
        player_type="hitter",
        proj_hr_p50=proj_hr,
        proj_rbi_p50=proj_rbi,
        proj_sb_p50=proj_sb,
        proj_avg_p50=proj_avg,
        proj_k_p50=None,
        proj_era_p50=None,
        proj_whip_p50=None,
        actual_hr=actual_hr,
        actual_rbi=actual_rbi,
        actual_sb=actual_sb,
        actual_avg=actual_avg,
        actual_k=None,
        actual_era=None,
        actual_whip=None,
        games_played=games,
    )


def _make_pitcher_input(
    pid=2,
    proj_k=200.0, proj_era=3.10, proj_whip=1.10,
    actual_k=188.0, actual_era=3.45, actual_whip=1.22,
    games=14,
) -> BacktestInput:
    return BacktestInput(
        bdl_player_id=pid,
        as_of_date=YESTERDAY,
        player_type="pitcher",
        proj_hr_p50=None,
        proj_rbi_p50=None,
        proj_sb_p50=None,
        proj_avg_p50=None,
        proj_k_p50=proj_k,
        proj_era_p50=proj_era,
        proj_whip_p50=proj_whip,
        actual_hr=None,
        actual_rbi=None,
        actual_sb=None,
        actual_avg=None,
        actual_k=actual_k,
        actual_era=actual_era,
        actual_whip=actual_whip,
        games_played=games,
    )


# ---------------------------------------------------------------------------
# Test: compute_mae
# ---------------------------------------------------------------------------

def test_compute_mae_basic():
    """MAE of (10, 8) must be 2.0."""
    assert compute_mae(10.0, 8.0) == pytest.approx(2.0)


def test_compute_mae_projected_none():
    """None projected -> None result."""
    assert compute_mae(None, 8.0) is None


def test_compute_mae_actual_none():
    """None actual -> None result."""
    assert compute_mae(10.0, None) is None


def test_compute_mae_none_inputs():
    """Both None -> None."""
    assert compute_mae(None, None) is None


def test_compute_mae_zero_error():
    """Perfect prediction -> 0.0."""
    assert compute_mae(3.14, 3.14) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test: compute_rmse
# ---------------------------------------------------------------------------

def test_compute_rmse_basic():
    """For a single pair RMSE == |projected - actual|."""
    assert compute_rmse(10.0, 8.0) == pytest.approx(2.0)
    assert compute_rmse(8.0, 10.0) == pytest.approx(2.0)


def test_compute_rmse_none_inputs():
    """Both None -> None."""
    assert compute_rmse(None, None) is None


# ---------------------------------------------------------------------------
# Test: evaluate_player (hitter)
# ---------------------------------------------------------------------------

def test_evaluate_player_hitter():
    """Hitter composite_mae = mean(mae_hr, mae_rbi, mae_sb, mae_avg)."""
    inp = _make_hitter_input(
        proj_hr=30.0, actual_hr=28.0,   # mae = 2.0
        proj_rbi=90.0, actual_rbi=85.0, # mae = 5.0
        proj_sb=10.0, actual_sb=9.0,    # mae = 1.0
        proj_avg=0.280, actual_avg=0.265, # mae = 0.015
    )
    res = evaluate_player(inp)

    assert res.mae_hr  == pytest.approx(2.0)
    assert res.mae_rbi == pytest.approx(5.0)
    assert res.mae_sb  == pytest.approx(1.0)
    assert res.mae_avg == pytest.approx(0.015)

    expected_composite = (2.0 + 5.0 + 1.0 + 0.015) / 4
    assert res.composite_mae == pytest.approx(expected_composite)

    # Pitcher fields must all be None for a hitter
    assert res.mae_k   is None
    assert res.mae_era is None
    assert res.mae_whip is None

    # direction_correct not set by evaluate_player
    assert res.direction_correct is None
    assert res.player_type == "hitter"
    assert res.games_played == 14


# ---------------------------------------------------------------------------
# Test: evaluate_player (pitcher)
# ---------------------------------------------------------------------------

def test_evaluate_player_pitcher():
    """Pitcher composite_mae = mean(mae_k, mae_era, mae_whip)."""
    inp = _make_pitcher_input(
        proj_k=200.0, actual_k=188.0,     # mae = 12.0
        proj_era=3.10, actual_era=3.45,    # mae = 0.35
        proj_whip=1.10, actual_whip=1.22,  # mae = 0.12
    )
    res = evaluate_player(inp)

    assert res.mae_k   == pytest.approx(12.0)
    assert res.mae_era == pytest.approx(0.35)
    assert res.mae_whip == pytest.approx(0.12)

    expected_composite = (12.0 + 0.35 + 0.12) / 3
    assert res.composite_mae == pytest.approx(expected_composite)

    # Hitter fields must all be None
    assert res.mae_hr  is None
    assert res.mae_rbi is None
    assert res.player_type == "pitcher"


# ---------------------------------------------------------------------------
# Test: evaluate_cohort
# ---------------------------------------------------------------------------

def test_evaluate_cohort_skips_unknown():
    """Players with player_type='unknown' must not appear in results."""
    unknown_inp = BacktestInput(
        bdl_player_id=99,
        as_of_date=YESTERDAY,
        player_type="unknown",
        proj_hr_p50=None, proj_rbi_p50=None, proj_sb_p50=None, proj_avg_p50=None,
        proj_k_p50=None, proj_era_p50=None, proj_whip_p50=None,
        actual_hr=None, actual_rbi=None, actual_sb=None, actual_avg=None,
        actual_k=None, actual_era=None, actual_whip=None,
        games_played=0,
    )
    hitter_inp = _make_hitter_input(pid=1)
    results = evaluate_cohort([hitter_inp, unknown_inp])

    assert len(results) == 1
    assert results[0].bdl_player_id == 1


def test_evaluate_cohort_none_projections():
    """Players where all projections are None still produce a result with None composite_mae."""
    inp = BacktestInput(
        bdl_player_id=5,
        as_of_date=YESTERDAY,
        player_type="hitter",
        proj_hr_p50=None, proj_rbi_p50=None, proj_sb_p50=None, proj_avg_p50=None,
        proj_k_p50=None, proj_era_p50=None, proj_whip_p50=None,
        actual_hr=10.0, actual_rbi=30.0, actual_sb=5.0, actual_avg=0.270,
        actual_k=None, actual_era=None, actual_whip=None,
        games_played=10,
    )
    results = evaluate_cohort([inp])
    assert len(results) == 1
    # All proj fields None -> all mae fields None -> composite_mae None
    assert results[0].composite_mae is None
    assert results[0].mae_hr is None


# ---------------------------------------------------------------------------
# Test: summarize
# ---------------------------------------------------------------------------

def test_summarize_no_regression():
    """When mean_composite_mae <= baseline * 1.20, regression_detected must be False."""
    results = [
        evaluate_player(_make_hitter_input(pid=1,
            proj_hr=30.0, actual_hr=29.0)),  # small error
        evaluate_player(_make_pitcher_input(pid=2,
            proj_k=200.0, actual_k=198.0)),
    ]
    # Force composite_mae to a known small value
    baseline = 10.0
    summary = summarize(results, date(2026, 3, 23), YESTERDAY, baseline_mean_mae=baseline)

    assert summary.n_players == 2
    assert summary.n_hitters == 1
    assert summary.n_pitchers == 1
    assert summary.regression_detected is False
    assert summary.mean_composite_mae is not None
    assert summary.baseline_mean_mae == pytest.approx(baseline)


def test_summarize_regression_detected():
    """When mean_composite_mae > baseline * 1.20, regression_detected must be True."""
    # Create a hitter with very large errors
    inp = _make_hitter_input(pid=1,
        proj_hr=30.0, actual_hr=0.0,
        proj_rbi=90.0, actual_rbi=0.0,
        proj_sb=10.0, actual_sb=0.0,
        proj_avg=0.300, actual_avg=0.0)
    results = [evaluate_player(inp)]

    # Set baseline low enough that the large composite_mae triggers regression
    baseline = 0.1
    summary = summarize(results, date(2026, 3, 23), YESTERDAY, baseline_mean_mae=baseline)

    assert summary.regression_detected is True
    assert summary.regression_delta is not None
    assert summary.regression_delta > 0.0


def test_summarize_empty_results():
    """Empty result list -> n_players=0, mean_composite_mae=None, no regression."""
    summary = summarize([], date(2026, 3, 23), YESTERDAY, baseline_mean_mae=5.0)

    assert summary.n_players == 0
    assert summary.mean_composite_mae is None
    assert summary.regression_detected is False


def test_summarize_no_baseline():
    """When baseline is None, regression_detected must always be False."""
    inp = _make_hitter_input(pid=1,
        proj_hr=30.0, actual_hr=0.0,
        proj_rbi=90.0, actual_rbi=0.0,
        proj_sb=10.0, actual_sb=0.0,
        proj_avg=0.300, actual_avg=0.0)
    results = [evaluate_player(inp)]
    summary = summarize(results, date(2026, 3, 23), YESTERDAY, baseline_mean_mae=None)

    assert summary.regression_detected is False
    assert summary.regression_delta is None


# ---------------------------------------------------------------------------
# Test: golden baseline round-trip
# ---------------------------------------------------------------------------

def test_golden_baseline_round_trip(tmp_path):
    """save_golden_baseline then load_golden_baseline returns the same value."""
    baseline_file = str(tmp_path / "baseline.json")

    # Build a summary with no regression
    results = [evaluate_player(_make_hitter_input(pid=1,
        proj_hr=30.0, actual_hr=29.0))]
    summary = summarize(results, date(2026, 3, 23), YESTERDAY, baseline_mean_mae=None)

    # Must not be regressed to allow save
    assert summary.regression_detected is False

    save_golden_baseline(summary, baseline_file)
    loaded = load_golden_baseline(baseline_file)

    assert "mean_composite_mae" in loaded
    if summary.mean_composite_mae is not None:
        assert loaded["mean_composite_mae"] == pytest.approx(summary.mean_composite_mae)
    else:
        assert loaded["mean_composite_mae"] is None


def test_golden_baseline_missing_file(tmp_path):
    """load_golden_baseline returns {'mean_composite_mae': None} when file absent."""
    missing = str(tmp_path / "does_not_exist.json")
    result = load_golden_baseline(missing)
    assert result == {"mean_composite_mae": None}


def test_golden_baseline_not_saved_on_regression(tmp_path):
    """save_golden_baseline must NOT write a file when regression_detected is True."""
    baseline_file = str(tmp_path / "baseline.json")

    # Manually construct a regressed summary
    summary = BacktestSummary(
        window_start=date(2026, 3, 23),
        window_end=YESTERDAY,
        n_players=1,
        n_hitters=1,
        n_pitchers=0,
        mean_composite_mae=99.0,
        mean_mae_hr=99.0,
        mean_mae_rbi=None,
        mean_mae_sb=None,
        mean_mae_avg=None,
        mean_mae_k=None,
        mean_mae_era=None,
        mean_mae_whip=None,
        baseline_mean_mae=1.0,
        regression_detected=True,
        regression_delta=98.0,
    )

    save_golden_baseline(summary, baseline_file)
    assert not os.path.exists(baseline_file), (
        "save_golden_baseline must not write a file when regression_detected=True"
    )
