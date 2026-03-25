"""
EMAC-084: Tests for calculate_calibration() Brier score fix.

Verifies:
1. Brier score uses individual model_prob values, not bin midpoints
2. days filter is applied (cutoff passed to _settled_bets)
3. Empty bets return safe defaults
4. Response includes bets_with_prob + days fields
5. Precise Brier vs midpoint diverge measurably when probs vary within a bin

Run:
    pytest tests/test_calibration_brier.py -v
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from backend.services.performance import calculate_calibration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bet(model_prob, outcome, days_ago=5):
    b = MagicMock()
    b.model_prob = model_prob
    b.outcome = outcome
    b.timestamp = datetime.utcnow() - timedelta(days=days_ago)
    b.game_id = id(b)  # unique
    return b


def _mock_db():
    return MagicMock()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_calculate_calibration_returns_brier_score_when_bets_exist():
    """
    Given 10 bets with model_prob=0.65 and alternating outcomes 0/1,
    precise Brier = mean((0.65 - outcome)^2).
    """
    bets = [_make_bet(0.65, i % 2) for i in range(10)]
    # Precise: (0.65-0)^2 = 0.4225, (0.65-1)^2 = 0.1225
    expected_brier = (5 * 0.4225 + 5 * 0.1225) / 10  # = 0.2725

    with patch(
        "backend.services.performance._settled_bets", return_value=bets
    ):
        result = calculate_calibration(_mock_db(), days=90)

    assert result["brier_score"] == pytest.approx(expected_brier, abs=1e-4)
    assert len(result["calibration_buckets"]) == 1
    assert result["calibration_buckets"][0]["bin"] == "65-70%"
    assert result["calibration_buckets"][0]["count"] == 10


def test_calculate_calibration_respects_days_filter():
    """
    calculate_calibration(db, days=30) must pass a cutoff to _settled_bets
    approximately 30 days ago (within a 5-second window for test timing).
    """
    with patch(
        "backend.services.performance._settled_bets", return_value=[]
    ) as mock_settled:
        calculate_calibration(_mock_db(), days=30)

    mock_settled.assert_called_once()
    _db_arg, cutoff_kwarg = mock_settled.call_args[0][0], mock_settled.call_args[1].get("cutoff")
    if cutoff_kwarg is None:
        # Positional call: _settled_bets(db, cutoff=...)
        cutoff_kwarg = mock_settled.call_args[0][1] if len(mock_settled.call_args[0]) > 1 else None

    assert cutoff_kwarg is not None, "_settled_bets was not called with a cutoff"
    now = datetime.utcnow()
    expected_cutoff = now - timedelta(days=30)
    delta = abs((cutoff_kwarg - expected_cutoff).total_seconds())
    assert delta < 5, f"Cutoff {cutoff_kwarg} is not ~30 days ago (delta={delta}s)"


def test_calculate_calibration_empty_when_no_bets():
    """When no bets are returned, all fields are safely empty/None."""
    with patch("backend.services.performance._settled_bets", return_value=[]):
        result = calculate_calibration(_mock_db(), days=90)

    assert result["calibration_buckets"] == []
    assert result["brier_score"] is None
    assert result["mean_calibration_error"] is None
    assert result["is_well_calibrated"] is None
    assert result["bets_with_prob"] == 0
    assert result["days"] == 90


def test_calculate_calibration_response_includes_meta_fields():
    """Response always includes bets_with_prob and days regardless of data."""
    with patch("backend.services.performance._settled_bets", return_value=[]):
        result = calculate_calibration(_mock_db(), days=45)

    assert "bets_with_prob" in result
    assert "days" in result
    assert result["days"] == 45


def test_brier_score_uses_individual_model_prob_not_bin_midpoint():
    """
    Single bet with model_prob=0.651 (far from bin midpoint 0.675), outcome=1.
    Precise Brier:   (0.651 - 1)^2 = 0.121801
    Midpoint Brier:  (0.675 - 1)^2 = 0.105625
    Difference: ~0.016 — verifies the implementation uses the stored prob, not the midpoint.
    """
    prob, outcome = 0.651, 1
    bets = [_make_bet(prob, outcome)]

    precise_brier = (prob - outcome) ** 2          # 0.121801
    mid = 0.675
    midpoint_brier = (mid - outcome) ** 2          # 0.105625

    assert abs(precise_brier - midpoint_brier) > 0.01, "Test setup sanity check failed"

    with patch("backend.services.performance._settled_bets", return_value=bets):
        result = calculate_calibration(_mock_db(), days=90)

    assert result["brier_score"] == pytest.approx(precise_brier, abs=1e-4)
    assert result["brier_score"] != pytest.approx(midpoint_brier, abs=1e-4)
