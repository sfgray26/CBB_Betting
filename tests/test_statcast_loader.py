"""
Tests for statcast_loader.py — Statcast enrichment for waiver recommendations.
"""

import time
import pytest
from backend.fantasy_baseball.statcast_loader import (
    get_statcast_batter,
    get_statcast_pitcher,
    build_statcast_signals,
    statcast_need_score_boost,
    cache_age_seconds,
    _batter_cache,
    _pitcher_cache,
)
import backend.fantasy_baseball.statcast_loader as _loader_mod
from backend.fantasy_baseball.advanced_metrics import StatcastBatter, StatcastPitcher


# ---------------------------------------------------------------------------
# Fixtures — inject known data without hitting filesystem
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def populate_caches():
    """Inject a small set of known test records into module caches."""
    _batter_cache.clear()
    _pitcher_cache.clear()
    # Set _loaded_at to "just now" so _ensure_loaded() sees a fresh cache
    # and doesn't overwrite our test data with the CSV files.
    _loader_mod._loaded_at = time.time()

    # Batter: strong BUY_LOW signal (xwoba_diff < -0.020)
    _batter_cache["unlucky hitter"] = StatcastBatter(
        name="Unlucky Hitter",
        barrel_pct=11.0,
        exit_velo_avg=92.5,
        hard_hit_pct=42.0,
        sweet_spot_pct=34.0,
        xwoba=0.370,
        xwoba_diff=-0.035,  # xwOBA >> wOBA → BUY_LOW
        regression_up=True,
    )
    # Batter: SELL_HIGH signal (xwoba_diff > 0.030)
    _batter_cache["lucky hitter"] = StatcastBatter(
        name="Lucky Hitter",
        barrel_pct=5.0,
        exit_velo_avg=86.0,
        hard_hit_pct=28.0,
        xwoba=0.295,
        xwoba_diff=0.040,   # xwOBA << wOBA → SELL_HIGH
        regression_down=True,
    )
    # Batter: breakout candidate (barrel% > 8, EV > 90, xwoba_diff < -0.015)
    _batter_cache["young breakout"] = StatcastBatter(
        name="Young Breakout",
        barrel_pct=9.5,
        exit_velo_avg=91.0,
        hard_hit_pct=40.0,
        xwoba_diff=-0.018,
        regression_up=True,
    )
    # Pitcher: BUY_LOW (xera_diff > 0.40 means ERA will rise → pitcher improving = buy)
    _pitcher_cache["unlucky pitcher"] = StatcastPitcher(
        name="Unlucky Pitcher",
        stuff_plus=115.0,
        fb_velo_avg=95.0,
        xera=3.80,
        xera_diff=0.45,     # ERA 3.35, xERA 3.80 → ERA should rise → wait, this is SELL_HIGH for ERA
        injury_risk_score=10.0,
    )
    # Pitcher: HIGH_INJURY_RISK
    _pitcher_cache["hurt pitcher"] = StatcastPitcher(
        name="Hurt Pitcher",
        stuff_plus=108.0,
        fb_velo_avg=92.0,
        velo_decline=2.5,
        pitches_per_game=101.0,
        injury_risk_score=65.0,
    )
    # Pitcher: breakout candidate
    _pitcher_cache["breakout pitcher"] = StatcastPitcher(
        name="Breakout Pitcher",
        stuff_plus=128.0,
        whiff_pct=32.0,
        chase_pct=31.0,
        breakout_candidate=True,
        injury_risk_score=15.0,
    )

    yield

    _batter_cache.clear()
    _pitcher_cache.clear()


# ---------------------------------------------------------------------------
# Cache lookup tests
# ---------------------------------------------------------------------------

def test_get_statcast_batter_returns_known_player():
    b = get_statcast_batter("Unlucky Hitter")
    assert b is not None
    assert b.xwoba_diff == pytest.approx(-0.035)


def test_get_statcast_batter_case_insensitive():
    b = get_statcast_batter("UNLUCKY HITTER")
    assert b is not None


def test_get_statcast_batter_unknown_returns_none():
    assert get_statcast_batter("Nobody McFakename") is None


def test_get_statcast_pitcher_returns_known_player():
    p = get_statcast_pitcher("Breakout Pitcher")
    assert p is not None
    assert p.breakout_candidate is True


def test_get_statcast_pitcher_unknown_returns_none():
    assert get_statcast_pitcher("Nobody McFakename") is None


# ---------------------------------------------------------------------------
# Signal generation tests
# ---------------------------------------------------------------------------

def test_buy_low_signal_for_unlucky_batter():
    signals, delta = build_statcast_signals("Unlucky Hitter", is_pitcher=False, owned_pct=40)
    assert "BUY_LOW" in signals
    assert delta == pytest.approx(-0.035)


def test_sell_high_signal_for_lucky_batter():
    signals, delta = build_statcast_signals("Lucky Hitter", is_pitcher=False, owned_pct=80)
    assert "SELL_HIGH" in signals
    assert delta == pytest.approx(0.040)


def test_breakout_signal_for_low_owned_young_player():
    signals, _ = build_statcast_signals("Young Breakout", is_pitcher=False, owned_pct=25)
    assert "BREAKOUT" in signals


def test_breakout_suppressed_for_high_owned_player():
    # High ownership means the market already knows — don't flag as hidden gem
    signals, _ = build_statcast_signals("Young Breakout", is_pitcher=False, owned_pct=75)
    assert "BREAKOUT" not in signals


def test_high_injury_risk_signal_for_pitcher():
    signals, _ = build_statcast_signals("Hurt Pitcher", is_pitcher=True)
    assert "HIGH_INJURY_RISK" in signals


def test_breakout_signal_for_breakout_pitcher():
    signals, _ = build_statcast_signals("Breakout Pitcher", is_pitcher=True)
    assert "BREAKOUT" in signals


def test_unknown_player_returns_empty_signals():
    signals, delta = build_statcast_signals("Nobody McFakename", is_pitcher=False, owned_pct=10)
    assert signals == []
    assert delta == 0.0


# ---------------------------------------------------------------------------
# DB tier: _load_from_db falls back gracefully when DB is unavailable
# ---------------------------------------------------------------------------

def test_load_from_db_returns_empty_dicts_on_db_error(monkeypatch):
    """_load_from_db must return ({}, {}) when SessionLocal raises, not propagate."""
    import backend.fantasy_baseball.statcast_loader as _mod

    def _boom():
        raise RuntimeError("no db in test")

    monkeypatch.setattr("backend.models.SessionLocal", _boom)
    batters, pitchers = _mod._load_from_db()
    assert batters == {}
    assert pitchers == {}


def test_load_from_db_builds_statcastbatter_from_rows(monkeypatch):
    """_load_from_db maps DB row fields onto StatcastBatter correctly."""
    import types
    import backend.fantasy_baseball.statcast_loader as _mod

    # Simulate a DB row with the columns our query returns
    class _Row:
        player_name = "Cody Bellinger"
        xwoba = 0.355
        barrel_percent = 11.2
        hard_hit_percent = 43.5
        avg_exit_velocity = 91.8
        sprint_speed = 28.5
        whiff_percent = 22.0
        avg_woba = 0.310

    class _FakeResult:
        def fetchall(self): return [_Row()]

    class _FakeConn:
        def execute(self, *a, **kw): return _FakeResult()
        def close(self): pass

    class _FakeDB(_FakeConn):
        pass

    monkeypatch.setattr("backend.models.SessionLocal", lambda: _FakeDB())

    # Patch pitcher query to return nothing so test focuses on batters
    _orig_execute = _FakeDB.execute
    call_count = [0]
    def _selective_execute(self, q, *a, **kw):
        call_count[0] += 1
        if call_count[0] == 2:  # second call = pitchers
            class _Empty:
                def fetchall(self): return []
            return _Empty()
        return _orig_execute(self, q, *a, **kw)
    monkeypatch.setattr(_FakeDB, "execute", _selective_execute)

    batters, pitchers = _mod._load_from_db()

    assert "cody bellinger" in batters
    b = batters["cody bellinger"]
    assert b.xwoba == 0.355
    assert b.barrel_pct == 11.2
    assert b.hard_hit_pct == 43.5
    xwoba_diff_expected = 0.355 - 0.310
    assert abs(b.xwoba_diff - xwoba_diff_expected) < 1e-6
    assert b.regression_down is True   # xwoba_diff > 0.030 → overperforming
    assert pitchers == {}


def test_signal_generation_never_raises_on_error():
    """Even with malformed inputs, build_statcast_signals must not raise."""
    signals, delta = build_statcast_signals("", is_pitcher=False)
    assert isinstance(signals, list)
    assert isinstance(delta, float)


# ---------------------------------------------------------------------------
# Boost calculation tests
# ---------------------------------------------------------------------------

def test_buy_low_boost_is_positive():
    boost = statcast_need_score_boost(["BUY_LOW"])
    assert boost > 0


def test_breakout_boost_is_positive():
    boost = statcast_need_score_boost(["BREAKOUT"])
    assert boost > 0


def test_sell_high_boost_is_negative():
    boost = statcast_need_score_boost(["SELL_HIGH"])
    assert boost < 0


def test_high_injury_risk_reduces_score():
    boost = statcast_need_score_boost(["HIGH_INJURY_RISK"])
    assert boost < 0


def test_combined_buy_low_and_breakout_boost():
    boost = statcast_need_score_boost(["BUY_LOW", "BREAKOUT"])
    assert boost == pytest.approx(0.9)  # 0.4 + 0.5


def test_empty_signals_no_boost():
    assert statcast_need_score_boost([]) == pytest.approx(0.0)


def test_neutral_player_no_boost():
    signals, _ = build_statcast_signals("Nobody McFakename", is_pitcher=False)
    assert statcast_need_score_boost(signals) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Schema integration test
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fuzzy name normalization tests (accent + suffix stripping via pybaseball_loader)
# ---------------------------------------------------------------------------

def test_get_statcast_batter_strips_accent():
    _batter_cache["jose ramirez"] = StatcastBatter(name="Jose Ramirez", xwoba_diff=0.01)
    assert get_statcast_batter("Jos\u00e9 Ram\u00edrez") is not None


def test_get_statcast_batter_strips_suffix():
    _batter_cache["willi castro"] = StatcastBatter(name="Willi Castro", xwoba_diff=-0.02)
    assert get_statcast_batter("Willi Castro Jr.") is not None


def test_get_statcast_pitcher_strips_accent():
    _pitcher_cache["jose quintana"] = StatcastPitcher(name="Jose Quintana")
    assert get_statcast_pitcher("Jos\u00e9 Quintana") is not None


def test_roster_move_recommendation_accepts_statcast_fields():
    from backend.schemas import RosterMoveRecommendation, WaiverPlayerOut
    from datetime import date

    fa = WaiverPlayerOut(
        player_id="mlb.p.1",
        name="Test Player",
        team="NYY",
        position="2B",
        need_score=3.5,
        category_contributions={},
        owned_pct=18.0,
        starts_this_week=0,
    )
    rec = RosterMoveRecommendation(
        action="ADD_DROP",
        add_player=fa,
        drop_player_name="Bench Guy",
        drop_player_position="2B",
        rationale="Test rationale [BUY_LOW (xwOBA delta=-0.035)].",
        category_targets=[],
        need_score=4.2,
        confidence=0.75,
        statcast_signals=["BUY_LOW"],
        regression_delta=-0.035,
    )
    assert rec.statcast_signals == ["BUY_LOW"]
    assert rec.regression_delta == pytest.approx(-0.035)
