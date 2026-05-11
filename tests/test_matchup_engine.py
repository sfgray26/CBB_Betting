"""
Tests for backend/services/matchup_engine.py — PR 5.2/5.3/5.4

All tests are pure-function: no DB calls.  MatchupContext / sub-dataclasses
constructed inline; compute_matchup_z() and helpers imported directly.
"""

import math
import pytest
from datetime import date

from backend.services.matchup_engine import (
    _MLB_BASELINES_DEFAULT,
    BullpenStats,
    HitterSplits,
    MatchupContext,
    PitcherStats,
    WeatherData,
    compute_baselines,
    compute_bullpen_score,
    compute_handedness_score,
    compute_matchup_confidence,
    compute_matchup_z,
    compute_park_score,
    compute_pitcher_score,
    compute_weather_bonus,
)

_BASELINES = dict(_MLB_BASELINES_DEFAULT)
_GAME_DATE = date(2026, 4, 15)


def _neutral_context() -> MatchupContext:
    return MatchupContext(
        bdl_player_id=1,
        game_date=_GAME_DATE,
        opponent_team="NYY",
        home_team="NYY",
        pitcher=None,
        splits=None,
        bullpen=None,
        weather=None,
        park_factor_runs=1.0,
        park_factor_hr=1.0,
    )


class TestHandednessScore:
    def test_neutral_hand_returns_zero(self):
        splits = HitterSplits(woba_vs_hand=0.320, woba_overall=0.320,
                              k_pct_vs_hand=0.25, iso_vs_hand=0.15, pa_vs_hand=80)
        assert compute_handedness_score(splits, _BASELINES) == pytest.approx(0.0)

    def test_advantage_positive(self):
        splits = HitterSplits(woba_vs_hand=0.390, woba_overall=0.330,
                              k_pct_vs_hand=0.18, iso_vs_hand=0.20, pa_vs_hand=100)
        assert compute_handedness_score(splits, _BASELINES) > 0.0

    def test_disadvantage_negative(self):
        splits = HitterSplits(woba_vs_hand=0.270, woba_overall=0.330,
                              k_pct_vs_hand=0.32, iso_vs_hand=0.10, pa_vs_hand=90)
        assert compute_handedness_score(splits, _BASELINES) < 0.0

    def test_none_splits_returns_zero(self):
        assert compute_handedness_score(None, _BASELINES) == 0.0

    def test_missing_woba_returns_zero(self):
        splits = HitterSplits(woba_vs_hand=None, woba_overall=0.320,
                              k_pct_vs_hand=None, iso_vs_hand=None)
        assert compute_handedness_score(splits, _BASELINES) == 0.0


class TestPitcherScore:
    def test_average_pitcher_near_zero(self):
        p = PitcherStats(name="Avg", hand="R",
                         era=_BASELINES["mean_era"], whip=_BASELINES["mean_whip"], k_per_nine=8.5)
        assert compute_pitcher_score(p, _BASELINES) == pytest.approx(0.0, abs=0.05)

    def test_elite_pitcher_negative(self):
        p = PitcherStats(name="Ace", hand="L", era=2.10, whip=0.90, k_per_nine=12.0)
        assert compute_pitcher_score(p, _BASELINES) < 0.0

    def test_high_era_pitcher_positive(self):
        p = PitcherStats(name="Journeyman", hand="R", era=5.80, whip=1.60, k_per_nine=5.5)
        assert compute_pitcher_score(p, _BASELINES) > 0.0

    def test_none_pitcher_returns_zero(self):
        assert compute_pitcher_score(None, _BASELINES) == 0.0


class TestParkScore:
    def test_neutral_park_zero(self):
        assert compute_park_score(1.0) == pytest.approx(0.0)

    def test_pitchers_park_negative(self):
        assert compute_park_score(0.88) < 0.0

    def test_hitters_park_positive(self):
        assert compute_park_score(1.25) > 0.0

    def test_coors_value(self):
        assert compute_park_score(1.25) == pytest.approx(5.0, abs=0.01)


class TestWeatherBonus:
    def test_none_weather_zero(self):
        assert compute_weather_bonus(None) == 0.0

    def test_cold_no_bonus(self):
        w = WeatherData(temp_f=50.0, wind_mph=5.0, wind_direction="cross", precip_chance=10.0)
        assert compute_weather_bonus(w) == pytest.approx(0.0)

    def test_hot_day_bonus(self):
        w = WeatherData(temp_f=90.0, wind_mph=5.0, wind_direction="cross", precip_chance=0.0)
        assert compute_weather_bonus(w) == pytest.approx(1.5)

    def test_strong_wind_out_bonus(self):
        w = WeatherData(temp_f=72.0, wind_mph=20.0, wind_direction="out", precip_chance=0.0)
        assert compute_weather_bonus(w) == pytest.approx(3.0)

    def test_strong_wind_in_penalty(self):
        w = WeatherData(temp_f=72.0, wind_mph=20.0, wind_direction="in", precip_chance=0.0)
        assert compute_weather_bonus(w) == pytest.approx(-1.5)

    def test_high_precip_penalty(self):
        w = WeatherData(temp_f=72.0, wind_mph=5.0, wind_direction="cross", precip_chance=60.0)
        assert compute_weather_bonus(w) == pytest.approx(-2.0)

    def test_combined_hot_wind_out(self):
        w = WeatherData(temp_f=90.0, wind_mph=20.0, wind_direction="out", precip_chance=0.0)
        assert compute_weather_bonus(w) == pytest.approx(4.5)


class TestBullpenScore:
    def test_average_bullpen_near_zero(self):
        bp = BullpenStats(era=_BASELINES["mean_bullpen_era"], whip=1.28, pitcher_count=7)
        assert compute_bullpen_score(bp, _BASELINES) == pytest.approx(0.0, abs=0.01)

    def test_elite_bullpen_negative(self):
        bp = BullpenStats(era=2.80, whip=1.05, pitcher_count=7)
        assert compute_bullpen_score(bp, _BASELINES) < 0.0

    def test_terrible_bullpen_positive(self):
        bp = BullpenStats(era=5.50, whip=1.55, pitcher_count=5)
        assert compute_bullpen_score(bp, _BASELINES) > 0.0

    def test_none_bullpen_returns_zero(self):
        assert compute_bullpen_score(None, _BASELINES) == 0.0


class TestMatchupConfidence:
    def test_no_data_low_confidence(self):
        conf = compute_matchup_confidence(None, None, _BASELINES)
        assert conf < 0.30

    def test_partial_confidence_range(self):
        splits = HitterSplits(woba_vs_hand=0.300, woba_overall=0.310,
                              k_pct_vs_hand=0.25, iso_vs_hand=0.14, pa_vs_hand=10)
        p = PitcherStats(name="T", hand="R", era=4.0, whip=1.25, k_per_nine=8.0)
        conf = compute_matchup_confidence(splits, p, _BASELINES)
        assert 0.0 <= conf <= 1.0

    def test_full_data_high_confidence(self):
        splits = HitterSplits(woba_vs_hand=0.350, woba_overall=0.320,
                              k_pct_vs_hand=0.20, iso_vs_hand=0.18, pa_vs_hand=90)
        p = PitcherStats(name="Ace", hand="L", era=3.50, whip=1.15, k_per_nine=9.5)
        conf = compute_matchup_confidence(splits, p, _BASELINES)
        assert conf >= 0.60


class TestComputeMatchupZ:
    def test_all_neutral_returns_50(self):
        result = compute_matchup_z(_neutral_context(), _BASELINES)
        assert result.matchup_score == pytest.approx(50.0, abs=2.0)

    def test_coors_boosts_score(self):
        ctx = _neutral_context()
        ctx.park_factor_runs = 1.25
        assert compute_matchup_z(ctx, _BASELINES).matchup_score > 50.0

    def test_score_clamped_0_to_100(self):
        ctx = _neutral_context()
        ctx.park_factor_runs = 2.5
        ctx.bullpen = BullpenStats(era=9.0, whip=2.0, pitcher_count=5)
        r = compute_matchup_z(ctx, _BASELINES)
        assert 0.0 <= r.matchup_score <= 100.0

    def test_full_positive_matchup(self):
        ctx = MatchupContext(
            bdl_player_id=42, game_date=_GAME_DATE,
            opponent_team="HOU", home_team="COL",
            pitcher=PitcherStats(name="W", hand="L", era=5.5, whip=1.60, k_per_nine=5.0),
            splits=HitterSplits(woba_vs_hand=0.390, woba_overall=0.330,
                                k_pct_vs_hand=0.18, iso_vs_hand=0.22, pa_vs_hand=90),
            bullpen=BullpenStats(era=5.20, whip=1.50, pitcher_count=7),
            weather=WeatherData(temp_f=88.0, wind_mph=18.0,
                                wind_direction="out", precip_chance=0.0),
            park_factor_runs=1.25, park_factor_hr=1.20,
        )
        r = compute_matchup_z(ctx, _BASELINES)
        assert r.matchup_score > 55.0

    def test_full_negative_matchup(self):
        ctx = MatchupContext(
            bdl_player_id=43, game_date=_GAME_DATE,
            opponent_team="LAD", home_team="SD",
            pitcher=PitcherStats(name="Ace", hand="R", era=2.1, whip=0.88, k_per_nine=12.5),
            splits=HitterSplits(woba_vs_hand=0.270, woba_overall=0.330,
                                k_pct_vs_hand=0.33, iso_vs_hand=0.10, pa_vs_hand=80),
            bullpen=BullpenStats(era=2.80, whip=1.02, pitcher_count=8),
            weather=WeatherData(temp_f=62.0, wind_mph=22.0,
                                wind_direction="in", precip_chance=0.0),
            park_factor_runs=0.88, park_factor_hr=0.85,
        )
        r = compute_matchup_z(ctx, _BASELINES)
        assert r.matchup_score < 50.0

    def test_weights_sum_to_one(self):
        ctx = MatchupContext(
            bdl_player_id=1, game_date=_GAME_DATE,
            opponent_team="NYM", home_team="NYY",
            pitcher=PitcherStats(name="P", hand="R", era=4.0, whip=1.25, k_per_nine=8.0),
            splits=None,
            bullpen=BullpenStats(era=4.2, whip=1.28, pitcher_count=6),
            weather=None,
            park_factor_runs=1.04, park_factor_hr=1.04,
        )
        r = compute_matchup_z(ctx, _BASELINES)
        assert sum(r.component_weights.values()) == pytest.approx(1.0, abs=0.001)

    def test_confidence_gate_fires_with_low_data(self):
        ctx = _neutral_context()
        ctx.park_factor_runs = 1.20
        ctx.pitcher = PitcherStats(name="Ace", hand="R", era=2.0, whip=0.85, k_per_nine=12.0)
        r = compute_matchup_z(ctx, _BASELINES)
        assert r.matchup_confidence < 0.4


class TestComputeBaselines:
    def test_none_db_returns_defaults(self):
        assert compute_baselines(db=None) == _MLB_BASELINES_DEFAULT

    def test_returns_dict_with_required_keys(self):
        bl = compute_baselines(db=None)
        for key in ("mean_era", "std_era", "mean_whip", "std_woba_gap"):
            assert key in bl


