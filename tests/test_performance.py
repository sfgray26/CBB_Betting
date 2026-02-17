"""Tests for performance.py stat calculations."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from backend.services.performance import (
    _mean, _median, _std, _safe_roi, _win_rate, _clv_status,
    calculate_summary_stats, calculate_clv_analysis, calculate_calibration,
)


# ---------------------------------------------------------------------------
# Pure math helpers
# ---------------------------------------------------------------------------

def test_mean_empty():
    assert _mean([]) is None

def test_mean_values():
    assert _mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

def test_median_odd():
    assert _median([1, 3, 5]) == pytest.approx(3.0)

def test_median_even():
    assert _median([1, 2, 3, 4]) == pytest.approx(2.5)

def test_std_single():
    assert _std([5.0]) is None

def test_std_known():
    # std([2, 4, 4, 4, 5, 5, 7, 9]) = 2.0 (population, not sample)
    # With sample std (n-1), it's approx 2.138
    result = _std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert result == pytest.approx(2.138, abs=0.01)

def test_safe_roi_zero_risked():
    assert _safe_roi(100.0, 0.0) == 0.0

def test_safe_roi_positive():
    assert _safe_roi(10.0, 100.0) == pytest.approx(0.1)

def test_win_rate_zero_total():
    assert _win_rate(5, 0) == 0.0

def test_win_rate():
    assert _win_rate(6, 10) == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# CLV status thresholds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("clv, expected", [
    (0.01,   "HEALTHY"),
    (0.005,  "WARNING"),   # exactly at threshold — >0.005 is strict, so 0.005 → WARNING
    (0.004,  "WARNING"),
    (-0.004, "WARNING"),
    (-0.005, "WARNING"),   # exactly at lower threshold
    (-0.006, "STOP"),
    (None,   "UNKNOWN"),
])
def test_clv_status(clv, expected):
    assert _clv_status(clv) == expected


# ---------------------------------------------------------------------------
# calculate_summary_stats with mocked DB
# ---------------------------------------------------------------------------

def _fake_bet(outcome, pl_dollars, bet_dollars=10.0, clv=None, bet_type="spread",
              conservative_edge=0.05, bankroll=1000.0):
    b = MagicMock()
    b.outcome = outcome
    b.profit_loss_dollars = pl_dollars
    b.bet_size_dollars = bet_dollars
    b.clv_prob = clv
    b.bet_type = bet_type
    b.conservative_edge = conservative_edge
    b.bankroll_at_bet = bankroll
    b.game = MagicMock()
    b.game.game_date = datetime.utcnow()
    b.model_prob = 0.58
    return b


def _mock_db_with_bets(bets):
    """Return a mock Session whose _settled_bets call returns bets."""
    db = MagicMock()
    query_mock = MagicMock()
    query_mock.join.return_value = query_mock
    query_mock.options.return_value = query_mock
    query_mock.filter.return_value = query_mock
    query_mock.order_by.return_value = query_mock
    query_mock.all.return_value = bets
    db.query.return_value = query_mock
    return db


def test_summary_no_bets():
    db = _mock_db_with_bets([])
    with patch("backend.services.performance._settled_bets", return_value=[]):
        result = calculate_summary_stats(db)
    assert result.get("total_bets", 0) == 0


def test_summary_basic():
    bets = [
        _fake_bet(1,  9.09, clv=0.01),
        _fake_bet(1,  9.09, clv=0.02),
        _fake_bet(0, -10.0, clv=-0.01),
        _fake_bet(0, -10.0, clv=0.005),
    ]
    with patch("backend.services.performance._settled_bets", return_value=bets):
        result = calculate_summary_stats(MagicMock())

    overall = result["overall"]
    assert overall["total_bets"] == 4
    assert overall["wins"] == 2
    assert overall["win_rate"] == pytest.approx(0.5)
    assert overall["total_profit_dollars"] == pytest.approx(9.09 + 9.09 - 10.0 - 10.0, abs=0.01)


def test_clv_analysis_no_data():
    with patch("backend.services.performance._settled_bets", return_value=[]):
        result = calculate_clv_analysis(MagicMock())
    assert result.get("bets_with_clv", 0) == 0


def test_clv_analysis_distribution():
    bets = [
        _fake_bet(1, 9.09, clv=0.04),   # strong_positive
        _fake_bet(0, -10.0, clv=-0.04), # strong_negative
        _fake_bet(1, 9.09, clv=0.005),  # neutral
    ]
    with patch("backend.services.performance._settled_bets", return_value=bets):
        result = calculate_clv_analysis(MagicMock())

    dist = result["distribution"]
    assert dist["strong_positive"] == 1
    assert dist["strong_negative"] == 1
    assert dist["neutral"] == 1


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _fake_calib_bet(outcome, model_prob):
    b = MagicMock()
    b.outcome = outcome
    b.model_prob = model_prob
    b.clv_prob = None
    b.profit_loss_dollars = 0
    b.bet_size_dollars = 10.0
    b.bet_type = "spread"
    b.conservative_edge = 0.05
    b.bankroll_at_bet = 1000.0
    b.game = MagicMock()
    b.game.game_date = datetime.utcnow()
    return b


def test_calibration_no_data():
    with patch("backend.services.performance._settled_bets", return_value=[]):
        result = calculate_calibration(MagicMock())
    assert result["calibration_buckets"] == []


def test_calibration_one_bucket():
    # 6 bets in 55-60% bin: 3 wins → actual rate = 0.5, predicted = 0.575, error = 0.075
    bets = [_fake_calib_bet(i % 2, 0.57) for i in range(6)]
    with patch("backend.services.performance._settled_bets", return_value=bets):
        result = calculate_calibration(MagicMock())

    assert len(result["calibration_buckets"]) == 1
    bucket = result["calibration_buckets"][0]
    assert bucket["bin"] == "55-60%"
    assert bucket["count"] == 6
    assert bucket["actual_win_rate"] == pytest.approx(0.5)
    assert result["brier_score"] is not None
