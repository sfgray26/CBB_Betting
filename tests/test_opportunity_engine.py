"""
Tests for backend/services/opportunity_engine.py (PR 3.2 / 3.3).

All tests are pure-function unit tests — no DB required.
DB-facing functions (aggregate_player_opportunity, compute_opportunity_baselines)
are tested via mocks that verify the contract without hitting Postgres.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from backend.services.opportunity_engine import (
    OpportunityMetrics,
    compute_lineup_entropy,
    compute_opportunity_confidence,
    compute_opportunity_score,
    compute_opportunity_z,
    compute_role_certainty,
    compute_platoon_risk,
    compute_all_opportunity,
    aggregate_player_opportunity,
    _pitcher_opportunity_score,
)

TODAY = date(2026, 5, 5)


# ---------------------------------------------------------------------------
# compute_lineup_entropy
# ---------------------------------------------------------------------------

def test_entropy_stable_slot_returns_zero():
    assert compute_lineup_entropy([1, 1, 1, 1, 1]) == pytest.approx(0.0)


def test_entropy_uniform_nine_slots_returns_one():
    slots = list(range(1, 10))  # 9 distinct slots, one each
    result = compute_lineup_entropy(slots)
    assert result == pytest.approx(1.0, abs=1e-9)


def test_entropy_empty_returns_none():
    assert compute_lineup_entropy([]) is None


def test_entropy_two_slots_midpoint():
    slots = [1, 1, 2, 2]  # 50/50 split across 2 of 9 slots
    result = compute_lineup_entropy(slots)
    assert result is not None
    assert 0.0 < result < 1.0


def test_entropy_bounded_zero_to_one():
    import random
    random.seed(42)
    for _ in range(50):
        slots = [random.randint(1, 9) for _ in range(random.randint(1, 20))]
        result = compute_lineup_entropy(slots)
        if result is not None:
            assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# compute_platoon_risk
# ---------------------------------------------------------------------------

def test_platoon_risk_none_when_inputs_none():
    assert compute_platoon_risk(None, None) is None
    assert compute_platoon_risk(10, None) is None


def test_platoon_risk_zero_total_returns_none():
    assert compute_platoon_risk(0, 0) is None


def test_platoon_risk_everyday_player_is_zero():
    # 50% vs LHP / 50% vs RHP = no platoon
    result = compute_platoon_risk(50, 50)
    assert result == pytest.approx(0.0)


def test_platoon_risk_strict_platoon_is_one():
    # All PA vs one hand
    result = compute_platoon_risk(100, 0)
    assert result == pytest.approx(1.0)


def test_platoon_risk_moderate():
    # 70% vs RHP = moderate platoon
    result = compute_platoon_risk(30, 70)
    assert result is not None
    assert 0.3 < result < 0.6


# ---------------------------------------------------------------------------
# compute_role_certainty
# ---------------------------------------------------------------------------

def test_role_certainty_hitter_returns_none():
    assert compute_role_certainty(10, 5, 3, "hitter") is None


def test_role_certainty_zero_appearances_returns_zero():
    assert compute_role_certainty(0, None, None, "pitcher") == pytest.approx(0.0)
    assert compute_role_certainty(None, None, None, "pitcher") == pytest.approx(0.0)


def test_role_certainty_pure_closer_is_high():
    # 10 appearances, 8 saves = elite closer
    result = compute_role_certainty(10, 8, 2, "pitcher")
    assert result is not None
    assert result >= 0.8


def test_role_certainty_swingman_is_low():
    # 10 appearances, 0 saves, 0 holds = swingman
    result = compute_role_certainty(10, 0, 0, "pitcher")
    assert result is not None
    assert result <= 0.2


def test_role_certainty_blends_season_sv_pct():
    # Without season data: pure from recent
    r1 = compute_role_certainty(10, 5, 0, "pitcher", season_sv_pct=None)
    # With high season save pct: should be higher
    r2 = compute_role_certainty(10, 5, 0, "pitcher", season_sv_pct=0.9)
    assert r2 >= r1


def test_role_certainty_bounded():
    result = compute_role_certainty(5, 10, 10, "pitcher")  # saves > appearances
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# compute_opportunity_confidence
# ---------------------------------------------------------------------------

def test_confidence_zero_pa_returns_zero():
    assert compute_opportunity_confidence(0) == pytest.approx(0.0)
    assert compute_opportunity_confidence(None) == pytest.approx(0.0)


def test_confidence_min_pa_is_near_half():
    # At MIN_PA_CONFIDENCE (20), sigmoid should return ~0.5
    result = compute_opportunity_confidence(20)
    assert 0.45 <= result <= 0.55


def test_confidence_high_pa_is_near_one():
    result = compute_opportunity_confidence(200)
    assert result > 0.95


def test_confidence_monotonic():
    results = [compute_opportunity_confidence(n) for n in [5, 10, 20, 40, 80]]
    assert results == sorted(results)


# ---------------------------------------------------------------------------
# compute_opportunity_score (hitters)
# ---------------------------------------------------------------------------

def _make_hitter(pa_per_game=4.0, started_pct=1.0, entropy=None, platoon=None):
    m = OpportunityMetrics(
        bdl_player_id=1,
        as_of_date=TODAY,
        player_type="hitter",
        pa_per_game=pa_per_game,
        games_started_pct=started_pct,
        lineup_slot_entropy=entropy,
        platoon_risk_score=platoon,
    )
    return m


def test_hitter_score_everyday_leadoff_is_high():
    m = _make_hitter(pa_per_game=4.5, started_pct=1.0)
    score = compute_opportunity_score(m)
    assert score > 0.65


def test_hitter_score_bench_player_is_low():
    m = _make_hitter(pa_per_game=1.0, started_pct=0.2)
    score = compute_opportunity_score(m)
    assert score < 0.4


def test_hitter_score_no_data_returns_zero():
    m = OpportunityMetrics(bdl_player_id=1, as_of_date=TODAY, player_type="hitter")
    score = compute_opportunity_score(m)
    assert score == pytest.approx(0.0)


def test_hitter_score_bounded_zero_to_one():
    for pa in [0.5, 2.0, 3.5, 5.0, 7.0]:
        m = _make_hitter(pa_per_game=pa, started_pct=min(1.0, pa / 5.0))
        score = compute_opportunity_score(m)
        assert 0.0 <= score <= 1.0


def test_hitter_score_stable_slot_increases_score():
    m_stable = _make_hitter(entropy=0.0)
    m_chaotic = _make_hitter(entropy=0.9)
    assert compute_opportunity_score(m_stable) > compute_opportunity_score(m_chaotic)


def test_hitter_score_low_platoon_risk_is_better():
    m_everyday = _make_hitter(platoon=0.0)
    m_platoon = _make_hitter(platoon=0.9)
    assert compute_opportunity_score(m_everyday) > compute_opportunity_score(m_platoon)


# ---------------------------------------------------------------------------
# compute_opportunity_score (pitchers)
# ---------------------------------------------------------------------------

def _make_pitcher(appearances=10, role_certainty=0.8):
    m = OpportunityMetrics(
        bdl_player_id=2,
        as_of_date=TODAY,
        player_type="pitcher",
        appearances_14d=appearances,
        role_certainty_score=role_certainty,
    )
    return m


def test_pitcher_closer_daily_is_high():
    m = _make_pitcher(appearances=12, role_certainty=0.95)
    score = compute_opportunity_score(m)
    assert score > 0.7


def test_pitcher_swingman_no_saves_is_low():
    m = _make_pitcher(appearances=3, role_certainty=0.0)
    score = compute_opportunity_score(m)
    assert score < 0.3


def test_pitcher_score_bounded():
    for app in range(0, 15):
        m = _make_pitcher(appearances=app, role_certainty=app / 14.0)
        score = compute_opportunity_score(m)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# compute_opportunity_z
# ---------------------------------------------------------------------------

def test_opportunity_z_raises_on_empty_cohort():
    with pytest.raises(ValueError, match="cohort_scores is required"):
        compute_opportunity_z(0.5, [])


def test_opportunity_z_at_mean_is_zero():
    cohort = [0.1, 0.3, 0.5, 0.7, 0.9]
    mean = sum(cohort) / len(cohort)
    z = compute_opportunity_z(mean, cohort)
    assert z == pytest.approx(0.0, abs=1e-9)


def test_opportunity_z_high_score_is_positive():
    cohort = [0.1, 0.2, 0.3, 0.4, 0.5]
    z = compute_opportunity_z(0.9, cohort)
    assert z > 1.0


def test_opportunity_z_low_score_is_negative():
    cohort = [0.4, 0.5, 0.6, 0.7, 0.8]
    z = compute_opportunity_z(0.1, cohort)
    assert z < -1.0


def test_opportunity_z_single_member_cohort():
    # Single member cohort: std=0, z should be 0
    z = compute_opportunity_z(0.5, [0.5])
    assert z == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_all_opportunity (integration — mocked DB)
# ---------------------------------------------------------------------------

def _mock_db_no_data():
    """DB session that returns no rows for all queries."""
    db = MagicMock()
    result = MagicMock()
    result.fetchone.return_value = None
    result.fetchall.return_value = []
    db.execute.return_value = result
    return db


def test_compute_all_opportunity_empty_input():
    db = _mock_db_no_data()
    results = compute_all_opportunity([], TODAY, db)
    assert results == []


def test_compute_all_opportunity_returns_one_per_player():
    db = _mock_db_no_data()
    players = [(101, "hitter"), (102, "pitcher"), (103, "hitter")]
    results = compute_all_opportunity(players, TODAY, db)
    assert len(results) == 3


def test_compute_all_opportunity_no_db_mapping_graceful():
    db = _mock_db_no_data()
    players = [(999, "hitter")]
    results = compute_all_opportunity(players, TODAY, db)
    assert len(results) == 1
    assert results[0].opportunity_score == pytest.approx(0.0)
    assert results[0].opportunity_confidence == pytest.approx(0.0)


def test_compute_all_opportunity_z_scores_computed():
    """With varied scores, z-scores should not all be zero."""
    db = MagicMock()

    call_count = [0]

    def execute_side_effect(stmt, params=None):
        result = MagicMock()
        call_count[0] += 1
        # id_mapping: return a valid MLBAM id
        result.fetchone.return_value = ("123456", None)
        # statcast_performances: return some rows with varying PA
        pa_by_player = {
            101: [(date(2026, 5, 1), 4, 4, 0.0), (date(2026, 5, 2), 4, 4, 0.0)],
            102: [(date(2026, 5, 1), 1, 1, 0.0)],
        }
        result.fetchall.return_value = pa_by_player.get(101, [])
        return result

    db.execute.side_effect = execute_side_effect

    players = [(101, "hitter"), (102, "hitter"), (103, "hitter")]
    results = compute_all_opportunity(players, TODAY, db)
    assert len(results) == 3
    # At least some should have non-zero z (cohort variance)
    z_values = [r.opportunity_z for r in results]
    assert not all(z == 0.0 for z in z_values)


def test_compute_all_opportunity_confidence_in_range():
    db = _mock_db_no_data()
    players = [(i, "hitter") for i in range(1, 6)]
    results = compute_all_opportunity(players, TODAY, db)
    for r in results:
        assert 0.0 <= r.opportunity_confidence <= 1.0
