"""
Tests for backend/services/market_engine.py (PRs 4.3 / 4.4).

All tests are pure-function unit tests — no DB, no I/O.
"""
from __future__ import annotations

import pytest

from backend.services.market_engine import (
    MarketResult,
    BUY_LOW, SELL_HIGH, HOT_PICKUP, SLEEPER, FAIR,
    ACT_NOW, THIS_WEEK, MONITOR,
    classify_market_tag,
    compute_market_score,
    compute_ownership_deltas,
    compute_ownership_velocity,
    compute_add_drop_ratio,
)


# ---------------------------------------------------------------------------
# compute_ownership_velocity
# ---------------------------------------------------------------------------

def test_velocity_rising_ownership():
    # +14% over 7 days = +2.0/day
    assert compute_ownership_velocity(30.0, 16.0) == pytest.approx(2.0)


def test_velocity_falling_ownership():
    assert compute_ownership_velocity(10.0, 24.0) == pytest.approx(-2.0)


def test_velocity_none_inputs_return_zero():
    assert compute_ownership_velocity(None, 20.0) == pytest.approx(0.0)
    assert compute_ownership_velocity(20.0, None) == pytest.approx(0.0)
    assert compute_ownership_velocity(None, None) == pytest.approx(0.0)


def test_velocity_no_change():
    assert compute_ownership_velocity(50.0, 50.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_ownership_deltas
# ---------------------------------------------------------------------------

def test_deltas_both_present():
    d7, d30 = compute_ownership_deltas(40.0, 30.0, 15.0)
    assert d7  == pytest.approx(10.0)
    assert d30 == pytest.approx(25.0)


def test_deltas_missing_7d_returns_none():
    d7, d30 = compute_ownership_deltas(40.0, None, 15.0)
    assert d7 is None
    assert d30 == pytest.approx(25.0)


def test_deltas_missing_30d_returns_none():
    d7, d30 = compute_ownership_deltas(40.0, 30.0, None)
    assert d7  == pytest.approx(10.0)
    assert d30 is None


def test_deltas_all_none():
    d7, d30 = compute_ownership_deltas(None, None, None)
    assert d7  is None
    assert d30 is None


# ---------------------------------------------------------------------------
# compute_add_drop_ratio
# ---------------------------------------------------------------------------

def test_add_drop_ratio_normal():
    assert compute_add_drop_ratio(10.0, 5.0) == pytest.approx(2.0)


def test_add_drop_ratio_none_inputs():
    assert compute_add_drop_ratio(None, 5.0) is None
    assert compute_add_drop_ratio(10.0, None) is None


def test_add_drop_ratio_zero_drops():
    assert compute_add_drop_ratio(5.0, 0.0) is None


# ---------------------------------------------------------------------------
# classify_market_tag
# ---------------------------------------------------------------------------

def test_tag_buy_low_high_skill_stable_ownership():
    tag, urgency = classify_market_tag(
        skill_gap_percentile=0.90,
        ownership_velocity=0.5,   # stable
        owned_pct=25.0,
    )
    assert tag == BUY_LOW
    assert urgency == ACT_NOW


def test_tag_sell_high_low_skill_rising_fast():
    tag, urgency = classify_market_tag(
        skill_gap_percentile=0.10,
        ownership_velocity=5.0,   # rising fast
        owned_pct=60.0,
    )
    assert tag == SELL_HIGH
    assert urgency == THIS_WEEK


def test_tag_hot_pickup_high_velocity_decent_skill():
    tag, urgency = classify_market_tag(
        skill_gap_percentile=0.70,
        ownership_velocity=7.0,   # rising very fast
        owned_pct=20.0,
    )
    assert tag == HOT_PICKUP
    assert urgency == ACT_NOW


def test_tag_sleeper_low_owned_good_skill():
    tag, urgency = classify_market_tag(
        skill_gap_percentile=0.80,
        ownership_velocity=0.2,
        owned_pct=8.0,   # very low ownership
    )
    assert tag == SLEEPER
    assert urgency == THIS_WEEK


def test_tag_fair_middling_player():
    tag, urgency = classify_market_tag(
        skill_gap_percentile=0.50,
        ownership_velocity=1.0,
        owned_pct=45.0,
    )
    assert tag == FAIR
    assert urgency == MONITOR


def test_tag_buy_low_requires_stable_velocity():
    # High skill but ownership rising fast → should NOT be BUY_LOW (already discovered)
    tag, urgency = classify_market_tag(
        skill_gap_percentile=0.90,
        ownership_velocity=6.0,   # rising fast, >2.0 threshold
        owned_pct=25.0,
    )
    # With high velocity and high skill, it's HOT_PICKUP not BUY_LOW
    assert tag == HOT_PICKUP


def test_tag_sell_high_requires_rising_velocity():
    # Low skill but nobody dropping → just FAIR (not SELL_HIGH without buying pressure)
    tag, urgency = classify_market_tag(
        skill_gap_percentile=0.10,
        ownership_velocity=0.5,   # stable — nobody is buying or selling
        owned_pct=60.0,
    )
    assert tag == FAIR


# ---------------------------------------------------------------------------
# compute_market_score — core algorithm
# ---------------------------------------------------------------------------

def test_market_score_high_skill_low_awareness_is_above_50():
    result = compute_market_score(
        skill_gap=1.5,
        skill_gap_percentile=0.90,   # > 0.85 threshold
        ownership_velocity=0.5,   # low awareness
        owned_pct=20.0,
        confidence=1.0,
    )
    assert result.market_score > 65.0
    assert result.market_tag == BUY_LOW


def test_market_score_low_skill_high_awareness_is_below_50():
    result = compute_market_score(
        skill_gap=-1.5,
        skill_gap_percentile=0.10,
        ownership_velocity=4.0,   # high awareness — market already knows
        owned_pct=55.0,
        confidence=1.0,
    )
    assert result.market_score < 50.0


def test_market_score_neutral_player_near_50():
    result = compute_market_score(
        skill_gap=0.0,
        skill_gap_percentile=0.50,
        ownership_velocity=0.0,
        owned_pct=40.0,
        confidence=1.0,
    )
    assert result.market_score == pytest.approx(50.0)
    assert result.market_tag == FAIR


def test_market_score_bounded_0_to_100():
    # Extreme high skill, zero market awareness
    r1 = compute_market_score(5.0, 1.0, 0.0, 5.0, 1.0)
    assert r1.market_score <= 100.0

    # Extreme low skill, zero market awareness
    r2 = compute_market_score(-5.0, 0.0, 0.0, 90.0, 1.0)
    assert r2.market_score >= 0.0


def test_market_score_confidence_gate_dampens_contrarian():
    high_conf = compute_market_score(
        skill_gap=2.0,
        skill_gap_percentile=0.90,
        ownership_velocity=0.0,
        owned_pct=10.0,
        confidence=1.0,
    )
    low_conf = compute_market_score(
        skill_gap=2.0,
        skill_gap_percentile=0.90,
        ownership_velocity=0.0,
        owned_pct=10.0,
        confidence=0.3,   # below 0.5 gate
    )
    # Low confidence should dampen the contrarian signal → score closer to 50
    assert abs(low_conf.market_score - 50.0) < abs(high_conf.market_score - 50.0)


def test_market_score_confidence_gate_at_boundary():
    # Exactly at 0.5 should NOT trigger the gate (< 0.5 only)
    at_boundary = compute_market_score(1.0, 0.80, 0.0, 20.0, confidence=0.5)
    just_below  = compute_market_score(1.0, 0.80, 0.0, 20.0, confidence=0.49)
    # Below gate → smaller deviation from 50
    assert abs(just_below.market_score - 50.0) < abs(at_boundary.market_score - 50.0)


def test_market_score_high_market_awareness_dampens_contrarian():
    # Even with high skill, if market already knows (high velocity), score approaches 50
    low_awareness  = compute_market_score(2.0, 0.90, 0.0,  10.0, 1.0)
    high_awareness = compute_market_score(2.0, 0.90, 10.0, 10.0, 1.0)
    # High awareness → contrarian dampened → score closer to 50
    assert abs(high_awareness.market_score - 50.0) < abs(low_awareness.market_score - 50.0)


def test_market_score_returns_market_result_type():
    result = compute_market_score(0.0, 0.5, 0.0, 30.0, 0.8)
    assert isinstance(result, MarketResult)
    assert isinstance(result.market_score, float)
    assert result.market_tag in (BUY_LOW, SELL_HIGH, HOT_PICKUP, SLEEPER, FAIR)
    assert result.market_urgency in (ACT_NOW, THIS_WEEK, MONITOR)


def test_sell_high_scenario_end_to_end():
    result = compute_market_score(
        skill_gap=-1.0,
        skill_gap_percentile=0.10,   # weak player
        ownership_velocity=4.0,      # market hasn't noticed yet? Actually still buying
        owned_pct=70.0,
        confidence=0.9,
    )
    assert result.market_tag == SELL_HIGH
    assert result.market_urgency == THIS_WEEK


def test_hot_pickup_end_to_end():
    result = compute_market_score(
        skill_gap=1.0,
        skill_gap_percentile=0.72,   # above floor
        ownership_velocity=6.5,      # above HOT threshold (5.0)
        owned_pct=12.0,
        confidence=0.8,
    )
    assert result.market_tag == HOT_PICKUP
    assert result.market_urgency == ACT_NOW


def test_sleeper_end_to_end():
    result = compute_market_score(
        skill_gap=0.8,
        skill_gap_percentile=0.75,
        ownership_velocity=0.1,
        owned_pct=5.0,   # very low ownership
        confidence=0.9,
    )
    assert result.market_tag == SLEEPER
    assert result.market_urgency == THIS_WEEK
