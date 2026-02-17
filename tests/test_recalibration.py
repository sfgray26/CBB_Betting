"""Tests for recalibration.py — model parameter calibration service."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, call

from backend.services.recalibration import (
    _margin_bias,
    _home_advantage_bias,
    _overconfidence,
    _brier_score,
    load_current_params,
    run_recalibration,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rec(predicted_margin, actual_margin, model_prob=0.6, outcome=1, is_neutral=False):
    """Build a synthetic settled-bet record dict."""
    return {
        "predicted_margin": predicted_margin,
        "actual_margin":    actual_margin,
        "model_prob":       model_prob,
        "outcome":          outcome,
        "is_neutral":       is_neutral,
    }


# ---------------------------------------------------------------------------
# _margin_bias
# ---------------------------------------------------------------------------

def test_margin_bias_empty():
    assert _margin_bias([]) is None


def test_margin_bias_zero():
    # predictions exactly match actuals
    records = [_rec(5.0, 5.0), _rec(-3.0, -3.0)]
    assert _margin_bias(records) == pytest.approx(0.0)


def test_margin_bias_positive():
    # model always over-predicts by 2 pts
    records = [_rec(7.0, 5.0), _rec(4.0, 2.0)]
    assert _margin_bias(records) == pytest.approx(2.0)


def test_margin_bias_negative():
    # model always under-predicts by 1 pt
    records = [_rec(3.0, 4.0), _rec(1.0, 2.0)]
    assert _margin_bias(records) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# _home_advantage_bias
# ---------------------------------------------------------------------------

def test_ha_bias_no_home_games():
    # Only neutral-site games → returns None
    records = [_rec(5.0, 3.0, is_neutral=True)]
    assert _home_advantage_bias(records) is None


def test_ha_bias_no_neutral_games():
    # No neutral games → neutral_bias treated as 0
    records = [_rec(7.0, 5.0, is_neutral=False), _rec(3.0, 1.0, is_neutral=False)]
    # home_bias = mean([2, 2]) = 2.0; neutral_bias = 0 → result = 2.0
    assert _home_advantage_bias(records) == pytest.approx(2.0)


def test_ha_bias_isolates_component():
    # Home games: predicted-actual = +4 on average
    # Neutral games: predicted-actual = +1 on average
    # HA component = 4 - 1 = 3.0
    home_recs = [_rec(10.0, 6.0, is_neutral=False), _rec(8.0, 4.0, is_neutral=False)]   # errors: +4, +4
    neutral_recs = [_rec(5.0, 4.0, is_neutral=True), _rec(3.0, 2.0, is_neutral=True)]   # errors: +1, +1
    records = home_recs + neutral_recs
    result = _home_advantage_bias(records)
    assert result == pytest.approx(3.0)


def test_ha_bias_zero_when_equal():
    # Same error pattern in home and neutral → HA component = 0
    home_recs    = [_rec(6.0, 4.0, is_neutral=False)]   # error +2
    neutral_recs = [_rec(6.0, 4.0, is_neutral=True)]    # error +2
    assert _home_advantage_bias(home_recs + neutral_recs) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _overconfidence
# ---------------------------------------------------------------------------

def test_overconfidence_insufficient_valid():
    # Fewer than 10 valid probs → None
    records = [_rec(0, 0, model_prob=0.6, outcome=1) for _ in range(9)]
    assert _overconfidence(records) is None


def test_overconfidence_filters_boundary_probs():
    # model_prob=0.0 and 1.0 are excluded; 8 valid + 2 boundary = 10 total → only 8 valid → None
    records = [_rec(0, 0, model_prob=0.6, outcome=1) for _ in range(8)]
    records += [_rec(0, 0, model_prob=0.0, outcome=0), _rec(0, 0, model_prob=1.0, outcome=1)]
    assert _overconfidence(records) is None


def test_overconfidence_perfectly_calibrated():
    # model_prob = 0.6, outcomes split 60% wins → overconfidence = 0
    records = [_rec(0, 0, model_prob=0.6, outcome=1) for _ in range(6)]
    records += [_rec(0, 0, model_prob=0.6, outcome=0) for _ in range(4)]
    result = _overconfidence(records)
    # mean_prob = 0.6, mean_outcome = 0.6 → 0.0
    assert result == pytest.approx(0.0, abs=1e-9)


def test_overconfidence_over_confident():
    # model says 0.7 but actual win rate is 0.5
    records = [_rec(0, 0, model_prob=0.7, outcome=1) for _ in range(5)]
    records += [_rec(0, 0, model_prob=0.7, outcome=0) for _ in range(5)]
    result = _overconfidence(records)
    assert result == pytest.approx(0.2)


def test_overconfidence_under_confident():
    # model says 0.4 but actual win rate is 0.6
    records = [_rec(0, 0, model_prob=0.4, outcome=1) for _ in range(6)]
    records += [_rec(0, 0, model_prob=0.4, outcome=0) for _ in range(4)]
    result = _overconfidence(records)
    assert result == pytest.approx(-0.2)


# ---------------------------------------------------------------------------
# _brier_score
# ---------------------------------------------------------------------------

def test_brier_score_empty():
    assert _brier_score([]) is None


def test_brier_score_perfect():
    # Always prob=1 and wins, or prob=0 and loses → BS = 0
    records = [
        _rec(0, 0, model_prob=1.0, outcome=1),
        _rec(0, 0, model_prob=0.0, outcome=0),
    ]
    assert _brier_score(records) == pytest.approx(0.0)


def test_brier_score_worst():
    # Always wrong: prob=1 on losses, prob=0 on wins
    records = [
        _rec(0, 0, model_prob=1.0, outcome=0),
        _rec(0, 0, model_prob=0.0, outcome=1),
    ]
    assert _brier_score(records) == pytest.approx(1.0)


def test_brier_score_known():
    # (0.7 - 1)^2 = 0.09; (0.4 - 0)^2 = 0.16; mean = 0.125
    records = [
        _rec(0, 0, model_prob=0.7, outcome=1),
        _rec(0, 0, model_prob=0.4, outcome=0),
    ]
    assert _brier_score(records) == pytest.approx(0.125)


# ---------------------------------------------------------------------------
# load_current_params
# ---------------------------------------------------------------------------

def _make_param(name, value):
    p = MagicMock()
    p.parameter_name = name
    p.parameter_value = value
    p.effective_date = datetime.utcnow()
    return p


def _db_with_params(params_by_name):
    """Return a mock Session where querying ModelParameter gives the supplied values."""
    db = MagicMock()
    def side_effect(model):
        q = MagicMock()
        q.filter.return_value = q
        q.order_by.return_value = q
        # Return the mock param or None
        q.first.return_value = None  # default
        return q
    db.query.side_effect = side_effect

    # More precise: differentiate by parameter_name via filter
    q = MagicMock()
    q.order_by.return_value = q

    def make_filter_chain(name):
        inner = MagicMock()
        inner.order_by.return_value = inner
        inner.first.return_value = params_by_name.get(name)
        return inner

    q.filter.side_effect = lambda *a, **kw: make_filter_chain(
        # Extract the name from the filter argument — too hard to inspect, so use call count
        None
    )

    # Simpler approach: just mock the whole query chain per call order
    calls = [iter([
        params_by_name.get("home_advantage"),
        params_by_name.get("sd_multiplier"),
    ])]

    results_iter = iter([
        params_by_name.get("home_advantage"),
        params_by_name.get("sd_multiplier"),
    ])

    inner_q = MagicMock()
    inner_q.filter.return_value = inner_q
    inner_q.order_by.return_value = inner_q
    inner_q.first.side_effect = lambda: next(results_iter)
    db.query.return_value = inner_q
    return db


def test_load_current_params_env_fallback():
    """When no DB rows exist, returns env-var defaults."""
    db = MagicMock()
    q = MagicMock()
    q.filter.return_value = q
    q.order_by.return_value = q
    q.first.return_value = None
    db.query.return_value = q

    with patch.dict("os.environ", {"HOME_ADVANTAGE": "3.50", "SD_MULTIPLIER": "0.90"}):
        result = load_current_params(db)

    assert result["home_advantage"] == pytest.approx(3.50)
    assert result["sd_multiplier"] == pytest.approx(0.90)


def test_load_current_params_db_override():
    """DB values override env-var defaults."""
    ha_param = _make_param("home_advantage", 4.00)
    sd_param = _make_param("sd_multiplier", 0.95)

    results = iter([ha_param, sd_param])

    db = MagicMock()
    q = MagicMock()
    q.filter.return_value = q
    q.order_by.return_value = q
    q.first.side_effect = lambda: next(results)
    db.query.return_value = q

    result = load_current_params(db)
    assert result["home_advantage"] == pytest.approx(4.00)
    assert result["sd_multiplier"] == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# run_recalibration — insufficient data
# ---------------------------------------------------------------------------

def _mock_db_for_recal(records):
    """Mock a DB session so _fetch_settled_records returns `records`."""
    db = MagicMock()
    with patch("backend.services.recalibration._fetch_settled_records", return_value=records):
        yield db


def test_recalibration_insufficient_data():
    db = MagicMock()
    with patch("backend.services.recalibration._fetch_settled_records", return_value=[]):
        result = run_recalibration(db, min_bets=30)
    assert result["status"] == "insufficient_data"
    assert "bets_available" in result


def test_recalibration_insufficient_data_partial():
    # 10 records when 30 are required
    recs = [_rec(5.0, 5.0) for _ in range(10)]
    db = MagicMock()
    with patch("backend.services.recalibration._fetch_settled_records", return_value=recs):
        result = run_recalibration(db, min_bets=30)
    assert result["status"] == "insufficient_data"
    assert result["bets_available"] == 10
    assert result["min_required"] == 30


# ---------------------------------------------------------------------------
# run_recalibration — no changes (bias within thresholds)
# ---------------------------------------------------------------------------

def test_recalibration_no_changes_small_bias():
    """Bias below trigger thresholds → status='no_changes'."""
    # ha_bias threshold = 0.5, overconfidence threshold = 0.03
    # Create records with near-zero bias
    recs = []
    for _ in range(30):
        # Home game, error ≈ 0.1 (well below 0.5)
        recs.append(_rec(5.1, 5.0, model_prob=0.55, outcome=1, is_neutral=False))

    def _fake_load(db):
        return {"home_advantage": 3.09, "sd_multiplier": 0.85}

    db = MagicMock()
    with (
        patch("backend.services.recalibration._fetch_settled_records", return_value=recs),
        patch("backend.services.recalibration.load_current_params", side_effect=_fake_load),
    ):
        result = run_recalibration(db, min_bets=30)

    assert result["status"] == "no_changes"
    assert result["parameters_changed"] == 0


# ---------------------------------------------------------------------------
# run_recalibration — home_advantage correction
# ---------------------------------------------------------------------------

def test_recalibration_adjusts_home_advantage():
    """
    Home-advantage bias > 0.5 triggers a correction.
    If home games show +2 error and neutral games show 0, ha_bias=+2.
    Expected: home_advantage decreases (model over-predicted home teams).
    """
    recs = []
    # 20 home games: predicted 2 pts too high for home team
    for _ in range(20):
        recs.append(_rec(7.0, 5.0, model_prob=0.55, outcome=1, is_neutral=False))
    # 10 neutral games: predicted correctly
    for _ in range(10):
        recs.append(_rec(5.0, 5.0, model_prob=0.55, outcome=1, is_neutral=True))

    def _fake_load(db):
        return {"home_advantage": 3.09, "sd_multiplier": 0.85}

    db = MagicMock()
    with (
        patch("backend.services.recalibration._fetch_settled_records", return_value=recs),
        patch("backend.services.recalibration.load_current_params", side_effect=_fake_load),
    ):
        result = run_recalibration(db, min_bets=30, apply_changes=False)

    assert result["status"] == "ok"
    ha_changes = [c for c in result["changes"] if c["parameter"] == "home_advantage"]
    assert len(ha_changes) == 1
    # Over-prediction of home teams → HA should decrease
    assert ha_changes[0]["new"] < ha_changes[0]["old"]
    assert ha_changes[0]["applied"] is False  # dry-run


def test_recalibration_ha_increases_when_under_predicted():
    """
    Negative ha_bias → home_advantage increases.
    """
    recs = []
    # Home games: model under-predicted by 2 pts (actual > predicted)
    for _ in range(20):
        recs.append(_rec(3.0, 5.0, model_prob=0.55, outcome=1, is_neutral=False))
    # Neutral games: predicted correctly
    for _ in range(10):
        recs.append(_rec(5.0, 5.0, model_prob=0.55, outcome=1, is_neutral=True))

    def _fake_load(db):
        return {"home_advantage": 3.09, "sd_multiplier": 0.85}

    db = MagicMock()
    with (
        patch("backend.services.recalibration._fetch_settled_records", return_value=recs),
        patch("backend.services.recalibration.load_current_params", side_effect=_fake_load),
    ):
        result = run_recalibration(db, min_bets=30, apply_changes=False)

    ha_changes = [c for c in result["changes"] if c["parameter"] == "home_advantage"]
    assert ha_changes[0]["new"] > ha_changes[0]["old"]


def test_recalibration_ha_bounded():
    """Extreme bias is capped at _MAX_HA_ADJ_PER_RUN = 0.50 pts."""
    recs = []
    # Massive over-prediction: ha_bias = +10 (much larger than any correction cap)
    for _ in range(20):
        recs.append(_rec(15.0, 5.0, model_prob=0.55, outcome=1, is_neutral=False))
    for _ in range(10):
        recs.append(_rec(5.0, 5.0, model_prob=0.55, outcome=1, is_neutral=True))

    def _fake_load(db):
        return {"home_advantage": 3.09, "sd_multiplier": 0.85}

    db = MagicMock()
    with (
        patch("backend.services.recalibration._fetch_settled_records", return_value=recs),
        patch("backend.services.recalibration.load_current_params", side_effect=_fake_load),
    ):
        result = run_recalibration(db, min_bets=30, apply_changes=False)

    ha_changes = [c for c in result["changes"] if c["parameter"] == "home_advantage"]
    assert len(ha_changes) == 1
    old_val = ha_changes[0]["old"]
    new_val = ha_changes[0]["new"]
    # Change is bounded to 0.50 pts
    assert abs(new_val - old_val) <= 0.50 + 1e-9


def test_recalibration_ha_safety_floor():
    """home_advantage never drops below _HA_MIN = 1.5."""
    recs = []
    for _ in range(20):
        recs.append(_rec(15.0, 5.0, model_prob=0.55, outcome=1, is_neutral=False))
    for _ in range(10):
        recs.append(_rec(5.0, 5.0, model_prob=0.55, outcome=1, is_neutral=True))

    def _fake_load(db):
        return {"home_advantage": 1.6, "sd_multiplier": 0.85}  # near floor

    db = MagicMock()
    with (
        patch("backend.services.recalibration._fetch_settled_records", return_value=recs),
        patch("backend.services.recalibration.load_current_params", side_effect=_fake_load),
    ):
        result = run_recalibration(db, min_bets=30, apply_changes=False)

    ha_changes = [c for c in result["changes"] if c["parameter"] == "home_advantage"]
    if ha_changes:
        assert ha_changes[0]["new"] >= 1.5


# ---------------------------------------------------------------------------
# run_recalibration — sd_multiplier correction
# ---------------------------------------------------------------------------

def test_recalibration_adjusts_sd_multiplier_over_confident():
    """
    Overconfidence > 0.03 triggers an SD multiplier increase.
    Model prob=0.75 but only 50% win rate → overconfidence = 0.25.
    """
    recs = []
    # 15 wins at 0.75 prob, 15 losses at 0.75 prob → mean_outcome = 0.5
    for _ in range(15):
        recs.append(_rec(5.0, 5.0, model_prob=0.75, outcome=1, is_neutral=True))
    for _ in range(15):
        recs.append(_rec(5.0, 5.0, model_prob=0.75, outcome=0, is_neutral=True))

    def _fake_load(db):
        return {"home_advantage": 3.09, "sd_multiplier": 0.85}

    db = MagicMock()
    with (
        patch("backend.services.recalibration._fetch_settled_records", return_value=recs),
        patch("backend.services.recalibration.load_current_params", side_effect=_fake_load),
    ):
        result = run_recalibration(db, min_bets=30, apply_changes=False)

    sd_changes = [c for c in result["changes"] if c["parameter"] == "sd_multiplier"]
    assert len(sd_changes) == 1
    # Over-confident → SD should increase
    assert sd_changes[0]["new"] > sd_changes[0]["old"]


def test_recalibration_adjusts_sd_multiplier_under_confident():
    """
    Overconfidence < -0.03 → SD multiplier decreases (distribution too wide).
    Model prob=0.55 but actual win rate is 0.8.
    """
    recs = []
    for _ in range(24):
        recs.append(_rec(5.0, 5.0, model_prob=0.55, outcome=1, is_neutral=True))
    for _ in range(6):
        recs.append(_rec(5.0, 5.0, model_prob=0.55, outcome=0, is_neutral=True))

    def _fake_load(db):
        return {"home_advantage": 3.09, "sd_multiplier": 0.85}

    db = MagicMock()
    with (
        patch("backend.services.recalibration._fetch_settled_records", return_value=recs),
        patch("backend.services.recalibration.load_current_params", side_effect=_fake_load),
    ):
        result = run_recalibration(db, min_bets=30, apply_changes=False)

    sd_changes = [c for c in result["changes"] if c["parameter"] == "sd_multiplier"]
    assert len(sd_changes) == 1
    assert sd_changes[0]["new"] < sd_changes[0]["old"]


def test_recalibration_sd_bounded():
    """Extreme overconfidence is capped at _MAX_SD_MULT_ADJ_PER_RUN = 0.03."""
    recs = []
    # prob=0.99 but only 50% wins → massive overconfidence ~ 0.49
    for _ in range(15):
        recs.append(_rec(5.0, 5.0, model_prob=0.99, outcome=1, is_neutral=True))
    for _ in range(15):
        recs.append(_rec(5.0, 5.0, model_prob=0.99, outcome=0, is_neutral=True))

    # Note: model_prob=0.99 is filtered as < 1.0, so it IS valid
    def _fake_load(db):
        return {"home_advantage": 3.09, "sd_multiplier": 0.85}

    db = MagicMock()
    with (
        patch("backend.services.recalibration._fetch_settled_records", return_value=recs),
        patch("backend.services.recalibration.load_current_params", side_effect=_fake_load),
    ):
        result = run_recalibration(db, min_bets=30, apply_changes=False)

    sd_changes = [c for c in result["changes"] if c["parameter"] == "sd_multiplier"]
    if sd_changes:
        assert abs(sd_changes[0]["new"] - sd_changes[0]["old"]) <= 0.03 + 1e-9


def test_recalibration_sd_safety_ceiling():
    """sd_multiplier never exceeds _SD_MULT_MAX = 1.10."""
    recs = []
    for _ in range(15):
        recs.append(_rec(5.0, 5.0, model_prob=0.99, outcome=1, is_neutral=True))
    for _ in range(15):
        recs.append(_rec(5.0, 5.0, model_prob=0.99, outcome=0, is_neutral=True))

    def _fake_load(db):
        return {"home_advantage": 3.09, "sd_multiplier": 1.09}  # near ceiling

    db = MagicMock()
    with (
        patch("backend.services.recalibration._fetch_settled_records", return_value=recs),
        patch("backend.services.recalibration.load_current_params", side_effect=_fake_load),
    ):
        result = run_recalibration(db, min_bets=30, apply_changes=False)

    sd_changes = [c for c in result["changes"] if c["parameter"] == "sd_multiplier"]
    if sd_changes:
        assert sd_changes[0]["new"] <= 1.10


# ---------------------------------------------------------------------------
# run_recalibration — apply_changes=True writes to DB
# ---------------------------------------------------------------------------

def test_recalibration_writes_to_db_when_apply_true():
    """When apply_changes=True and bias triggers a change, ModelParameter is added."""
    recs = []
    for _ in range(20):
        recs.append(_rec(7.0, 5.0, model_prob=0.55, outcome=1, is_neutral=False))
    for _ in range(10):
        recs.append(_rec(5.0, 5.0, model_prob=0.55, outcome=1, is_neutral=True))

    def _fake_load(db):
        return {"home_advantage": 3.09, "sd_multiplier": 0.85}

    db = MagicMock()
    with (
        patch("backend.services.recalibration._fetch_settled_records", return_value=recs),
        patch("backend.services.recalibration.load_current_params", side_effect=_fake_load),
    ):
        result = run_recalibration(db, min_bets=30, apply_changes=True)

    # db.add should have been called at least once (for the HA parameter)
    db.add.assert_called()
    db.commit.assert_called_once()


def test_recalibration_dry_run_does_not_write():
    """When apply_changes=False, no DB writes occur."""
    recs = []
    for _ in range(20):
        recs.append(_rec(7.0, 5.0, model_prob=0.55, outcome=1, is_neutral=False))
    for _ in range(10):
        recs.append(_rec(5.0, 5.0, model_prob=0.55, outcome=1, is_neutral=True))

    def _fake_load(db):
        return {"home_advantage": 3.09, "sd_multiplier": 0.85}

    db = MagicMock()
    with (
        patch("backend.services.recalibration._fetch_settled_records", return_value=recs),
        patch("backend.services.recalibration.load_current_params", side_effect=_fake_load),
    ):
        result = run_recalibration(db, min_bets=30, apply_changes=False)

    db.add.assert_not_called()
    db.commit.assert_not_called()
    # changes still returned
    assert len(result["changes"]) > 0


# ---------------------------------------------------------------------------
# run_recalibration — response structure
# ---------------------------------------------------------------------------

def test_recalibration_response_keys():
    """Response always contains expected top-level keys."""
    recs = [_rec(5.0, 5.0) for _ in range(30)]

    def _fake_load(db):
        return {"home_advantage": 3.09, "sd_multiplier": 0.85}

    db = MagicMock()
    with (
        patch("backend.services.recalibration._fetch_settled_records", return_value=recs),
        patch("backend.services.recalibration.load_current_params", side_effect=_fake_load),
    ):
        result = run_recalibration(db, min_bets=30, apply_changes=False)

    for key in ("status", "bets_analyzed", "parameters_changed", "changes", "diagnostics", "timestamp"):
        assert key in result, f"Missing key: {key}"


def test_recalibration_diagnostics_present():
    """Diagnostics dict contains expected metrics."""
    recs = [_rec(5.0, 5.0) for _ in range(30)]

    def _fake_load(db):
        return {"home_advantage": 3.09, "sd_multiplier": 0.85}

    db = MagicMock()
    with (
        patch("backend.services.recalibration._fetch_settled_records", return_value=recs),
        patch("backend.services.recalibration.load_current_params", side_effect=_fake_load),
    ):
        result = run_recalibration(db, min_bets=30, apply_changes=False)

    diag = result["diagnostics"]
    for key in ("margin_bias", "home_advantage_bias", "overconfidence", "brier_score"):
        assert key in diag, f"Missing diagnostic: {key}"
