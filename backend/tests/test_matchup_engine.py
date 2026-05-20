"""
Comprehensive tests for the matchup engine service.

Tests:
- Pitcher stat fetching (_fetch_pitcher_stats)
- Hitter splits fetching (_fetch_hitter_splits)
- Bullpen stats fetching (_fetch_bullpen_stats)
- Park factor fetching (_fetch_park_factor)
- Weather fetching (_fetch_weather)
- Matchup context collection (collect_matchup_context)
- Scoring functions (compute_handedness_score, compute_pitcher_score, etc.)
- Matchup Z-score computation (compute_matchup_z)
- Baseline computation (compute_baselines)
"""

import pytest
from datetime import date
from unittest.mock import Mock, patch, MagicMock
import math

from backend.services.matchup_engine import (
    # Data classes
    PitcherStats,
    HitterSplits,
    BullpenStats,
    WeatherData,
    MatchupContext,
    MatchupResult,
    # Fetch functions
    _fetch_pitcher_stats,
    _fetch_hitter_splits,
    _fetch_bullpen_stats,
    _fetch_park_factor,
    _fetch_weather,
    collect_matchup_context,
    # Scoring functions
    compute_handedness_score,
    compute_pitcher_score,
    compute_park_score,
    compute_weather_bonus,
    compute_bullpen_score,
    compute_matchup_confidence,
    compute_matchup_z,
    compute_baselines,
    # Constants
    _MLB_BASELINES_DEFAULT,
    _NEUTRAL_RESULT,
)


# =============================================================================
# Pitcher Stats Tests
# =============================================================================

class TestFetchPitcherStats:
    """Tests for _fetch_pitcher_stats function."""

    def test_fetch_pitcher_stats_success(self, mock_db_session, mock_db_row):
        """Successfully fetch pitcher stats with all fields."""
        mock_row = mock_db_row(
            pitcher_name="Clayton Kershaw",
            mlbam_id=477132,
            era=2.45,
            whip=0.98,
            k_9=9.8,
            handedness="L",
        )
        mock_db_session.execute.return_value.fetchone.return_value = mock_row

        result = _fetch_pitcher_stats("LAD", date(2026, 5, 15), mock_db_session)

        assert result is not None
        assert result.name == "Clayton Kershaw"
        assert result.hand == "L"
        assert result.era == 2.45
        assert result.whip == 0.98
        assert result.k_per_nine == 9.8
        assert result.mlbam_id == 477132

    def test_fetch_pitcher_stats_right_handed(self, mock_db_session, mock_db_row):
        """Fetch right-handed pitcher stats."""
        mock_row = mock_db_row(
            pitcher_name="Max Scherzer",
            mlbam_id=453286,
            era=3.12,
            whip=1.05,
            k_9=10.2,
            handedness="R",
        )
        mock_db_session.execute.return_value.fetchone.return_value = mock_row

        result = _fetch_pitcher_stats("NYM", date(2026, 5, 15), mock_db_session)

        assert result.hand == "R"

    def test_fetch_pitcher_stats_no_handedness(self, mock_db_session, mock_db_row):
        """Handle pitcher with no handedness data."""
        mock_row = mock_db_row(
            pitcher_name="Unknown Pitcher",
            mlbam_id=12345,
            era=4.50,
            whip=1.30,
            k_9=8.0,
            handedness=None,
        )
        mock_db_session.execute.return_value.fetchone.return_value = mock_row

        result = _fetch_pitcher_stats("XXX", date(2026, 5, 15), mock_db_session)

        assert result.hand is None

    def test_fetch_pitcher_stats_no_result(self, mock_db_session):
        """Return None when no pitcher found."""
        mock_db_session.execute.return_value.fetchone.return_value = None

        result = _fetch_pitcher_stats("LAD", date(2026, 5, 15), mock_db_session)

        assert result is None

    def test_fetch_pitcher_stats_db_error(self, mock_db_session):
        """Gracefully handle database errors."""
        mock_db_session.execute.side_effect = Exception("DB Error")

        result = _fetch_pitcher_stats("LAD", date(2026, 5, 15), mock_db_session)

        assert result is None

    def test_fetch_pitcher_stats_sql_includes_handedness(self, mock_db_session):
        """Verify SQL query includes handedness column."""
        mock_db_session.execute.return_value.fetchone.return_value = None

        _fetch_pitcher_stats("NYY", date(2026, 5, 15), mock_db_session)

        call_args = mock_db_session.execute.call_args
        sql_text = str(call_args[0][0])
        assert "pp.handedness" in sql_text
        assert "SELECT" in sql_text
        assert "FROM probable_pitchers" in sql_text

    def test_fetch_pitcher_stats_none_era(self, mock_db_session, mock_db_row):
        """Handle pitcher with NULL ERA."""
        mock_row = mock_db_row(
            pitcher_name="Rookie Pitcher",
            mlbam_id=99999,
            era=None,
            whip=1.20,
            k_9=None,
            handedness="R",
        )
        mock_db_session.execute.return_value.fetchone.return_value = mock_row

        result = _fetch_pitcher_stats("TEX", date(2026, 5, 15), mock_db_session)

        assert result.era is None
        assert result.whip == 1.20
        assert result.k_per_nine is None


# =============================================================================
# Hitter Splits Tests
# =============================================================================

class TestFetchHitterSplits:
    """Tests for _fetch_hitter_splits function."""

    def test_fetch_hitter_splits_success(self, mock_db_session, mock_db_row):
        """Successfully fetch hitter splits vs LHP."""
        # Split row
        split_row = mock_db_row(
            pa_vs_hand=50,
            h_vs=15,
            bb_vs=5,
            hr_vs=3,
            ab_vs=45,
            k_vs=10,
            d_vs=3,
            t_vs=0,
        )
        # Overall row
        overall_row = mock_db_row(
            total_h=120,
            total_bb=30,
            total_hr=15,
            total_ab=400,
        )
        
        mock_db_session.execute.side_effect = [
            MagicMock(fetchone=Mock(return_value=split_row)),
            MagicMock(fetchone=Mock(return_value=overall_row)),
        ]

        result = _fetch_hitter_splits(1, "L", mock_db_session)

        assert result is not None
        assert result.pa_vs_hand == 50
        assert result.woba_vs_hand is not None
        assert result.woba_overall is not None
        assert result.k_pct_vs_hand is not None
        assert result.iso_vs_hand is not None

    def test_fetch_hitter_splits_no_pitcher_hand(self, mock_db_session):
        """Return None when pitcher hand is None."""
        result = _fetch_hitter_splits(1, None, mock_db_session)

        assert result is None
        mock_db_session.execute.assert_not_called()

    def test_fetch_hitter_splits_no_pa(self, mock_db_session, mock_db_row):
        """Return splits with zero PA when no plate appearances."""
        split_row = mock_db_row(
            pa_vs_hand=0,
            h_vs=0,
            bb_vs=0,
            hr_vs=0,
            ab_vs=0,
            k_vs=0,
            d_vs=0,
            t_vs=0,
        )
        
        mock_db_session.execute.return_value.fetchone.return_value = split_row

        result = _fetch_hitter_splits(1, "R", mock_db_session)

        assert result is not None
        assert result.pa_vs_hand == 0
        assert result.woba_vs_hand is None

    def test_fetch_hitter_splits_db_error(self, mock_db_session):
        """Gracefully handle database errors."""
        mock_db_session.execute.side_effect = Exception("DB Error")

        result = _fetch_hitter_splits(1, "L", mock_db_session)

        assert result is None


# =============================================================================
# Bullpen Stats Tests
# =============================================================================

class TestFetchBullpenStats:
    """Tests for _fetch_bullpen_stats function."""

    def test_fetch_bullpen_stats_success(self, mock_db_session, mock_db_row):
        """Successfully fetch bullpen stats."""
        mock_row = mock_db_row(
            bullpen_era=3.85,
            bullpen_whip=1.25,
            pitcher_count=8,
        )
        mock_db_session.execute.return_value.fetchone.return_value = mock_row

        result = _fetch_bullpen_stats("NYY", 543037, mock_db_session)

        assert result is not None
        assert result.era == 3.85
        assert result.whip == 1.25
        assert result.pitcher_count == 8

    def test_fetch_bullpen_stats_no_starter_exclude(self, mock_db_session, mock_db_row):
        """Fetch bullpen stats without excluding starter."""
        mock_row = mock_db_row(
            bullpen_era=4.20,
            bullpen_whip=1.30,
            pitcher_count=10,
        )
        mock_db_session.execute.return_value.fetchone.return_value = mock_row

        result = _fetch_bullpen_stats("LAD", None, mock_db_session)

        assert result is not None
        assert result.pitcher_count == 10

    def test_fetch_bullpen_stats_insufficient_pitchers(self, mock_db_session, mock_db_row):
        """Return None when fewer than 2 pitchers qualify."""
        mock_row = mock_db_row(
            bullpen_era=3.50,
            bullpen_whip=1.15,
            pitcher_count=1,
        )
        mock_db_session.execute.return_value.fetchone.return_value = mock_row

        result = _fetch_bullpen_stats("BOS", None, mock_db_session)

        assert result is None

    def test_fetch_bullpen_stats_db_error(self, mock_db_session):
        """Gracefully handle database errors."""
        mock_db_session.execute.side_effect = Exception("DB Error")

        result = _fetch_bullpen_stats("NYY", None, mock_db_session)

        assert result is None


# =============================================================================
# Park Factor Tests
# =============================================================================

class TestFetchParkFactor:
    """Tests for _fetch_park_factor function."""

    def test_fetch_park_factor_success(self):
        """Fetch park factor for known team."""
        with patch('backend.services.matchup_engine.get_park_factor') as mock_get_pf:
            mock_get_pf.return_value = 1.15
            
            result = _fetch_park_factor("COL")
            
            assert result == 1.15

    def test_fetch_park_factor_fallback(self):
        """Return 1.0 on any error."""
        with patch('backend.services.matchup_engine.get_park_factor') as mock_get_pf:
            mock_get_pf.side_effect = Exception("API Error")
            
            result = _fetch_park_factor("COL")
            
            assert result == 1.0

    def test_fetch_park_factor_unknown_team(self):
        """Return 1.0 for unknown team."""
        with patch('backend.services.matchup_engine.get_park_factor') as mock_get_pf:
            mock_get_pf.side_effect = KeyError("Unknown team")
            
            result = _fetch_park_factor("XYZ")
            
            assert result == 1.0


# =============================================================================
# Weather Tests
# =============================================================================

class TestFetchWeather:
    """Tests for _fetch_weather function."""

    def test_fetch_weather_success(self, mock_db_session, mock_db_row):
        """Successfully fetch weather data."""
        mock_row = mock_db_row(
            temperature_high=25.0,  # Celsius
            wind_speed=15.0,  # km/h
            wind_direction="out",
            precipitation_probability=10.0,
        )
        mock_db_session.execute.return_value.fetchone.return_value = mock_row

        result = _fetch_weather("LAD", date(2026, 5, 15), mock_db_session)

        assert result is not None
        assert result.temp_f == pytest.approx(77.0, abs=0.5)  # 25C -> 77F
        assert result.wind_mph == pytest.approx(9.32, abs=0.1)  # 15km/h -> 9.32mph
        assert result.wind_direction == "out"
        assert result.precip_chance == 10.0

    def test_fetch_weather_unknown_team(self, mock_db_session):
        """Return None for unknown team."""
        result = _fetch_weather("XXX", date(2026, 5, 15), mock_db_session)

        assert result is None
        mock_db_session.execute.assert_not_called()

    def test_fetch_weather_no_forecast(self, mock_db_session):
        """Return None when no forecast found."""
        mock_db_session.execute.return_value.fetchone.return_value = None

        result = _fetch_weather("LAD", date(2026, 5, 15), mock_db_session)

        assert result is None

    def test_fetch_weather_db_error(self, mock_db_session):
        """Gracefully handle database errors."""
        mock_db_session.execute.side_effect = Exception("DB Error")

        result = _fetch_weather("LAD", date(2026, 5, 15), mock_db_session)

        assert result is None


# =============================================================================
# Matchup Context Tests
# =============================================================================

class TestCollectMatchupContext:
    """Tests for collect_matchup_context function."""

    def test_collect_matchup_context_success(self, mock_db_session, mock_db_row):
        """Successfully collect full matchup context."""
        # Setup pitcher row
        pitcher_row = mock_db_row(
            pitcher_name="Gerrit Cole",
            mlbam_id=543037,
            era=3.12,
            whip=1.05,
            k_9=10.2,
            handedness="R",
        )
        # Setup splits rows
        splits_row = mock_db_row(
            pa_vs_hand=50, h_vs=15, bb_vs=5, hr_vs=3,
            ab_vs=45, k_vs=10, d_vs=3, t_vs=0,
        )
        overall_row = mock_db_row(
            total_h=120, total_bb=30, total_hr=15, total_ab=400,
        )
        # Setup bullpen row
        bullpen_row = mock_db_row(
            bullpen_era=3.85, bullpen_whip=1.25, pitcher_count=8,
        )
        # Setup weather row
        weather_row = mock_db_row(
            temperature_high=25.0, wind_speed=15.0,
            wind_direction="out", precipitation_probability=10.0,
        )

        mock_db_session.execute.side_effect = [
            MagicMock(fetchone=Mock(return_value=pitcher_row)),
            MagicMock(fetchone=Mock(return_value=splits_row)),
            MagicMock(fetchone=Mock(return_value=overall_row)),
            MagicMock(fetchone=Mock(return_value=bullpen_row)),
            MagicMock(fetchone=Mock(return_value=weather_row)),
        ]

        with patch('backend.services.matchup_engine._fetch_park_factor', return_value=1.05):
            result = collect_matchup_context(
                bdl_player_id=1,
                game_date=date(2026, 5, 15),
                opponent_team="NYY",
                home_team="LAD",
                db=mock_db_session,
            )

        assert result is not None
        assert result.bdl_player_id == 1
        assert result.opponent_team == "NYY"
        assert result.home_team == "LAD"
        assert result.pitcher is not None
        assert result.splits is not None
        assert result.bullpen is not None
        assert result.weather is not None

    def test_collect_matchup_context_partial_data(self, mock_db_session):
        """Collect context with partial data (some fetches fail)."""
        mock_db_session.execute.return_value.fetchone.return_value = None

        with patch('backend.services.matchup_engine._fetch_park_factor', return_value=1.0):
            result = collect_matchup_context(
                bdl_player_id=1,
                game_date=date(2026, 5, 15),
                opponent_team="NYY",
                home_team="LAD",
                db=mock_db_session,
            )

        assert result is not None
        assert result.pitcher is None
        assert result.splits is None
        assert result.bullpen is None
        assert result.weather is None


# =============================================================================
# Scoring Function Tests
# =============================================================================

class TestComputeHandednessScore:
    """Tests for compute_handedness_score function."""

    def test_handedness_score_positive_advantage(self):
        """Positive score when hitter performs better vs pitcher's hand."""
        splits = HitterSplits(
            woba_vs_hand=0.380,
            woba_overall=0.340,
            k_pct_vs_hand=0.18,
            iso_vs_hand=0.200,
            pa_vs_hand=50,
        )
        baselines = {"std_woba_gap": 0.045}

        result = compute_handedness_score(splits, baselines)

        # (0.380 - 0.340) / 0.045 = 0.889
        assert result > 0
        assert result == pytest.approx(0.889, abs=0.01)

    def test_handedness_score_negative_disadvantage(self):
        """Negative score when hitter performs worse vs pitcher's hand."""
        splits = HitterSplits(
            woba_vs_hand=0.310,
            woba_overall=0.340,
            k_pct_vs_hand=0.22,
            iso_vs_hand=0.150,
            pa_vs_hand=50,
        )
        baselines = {"std_woba_gap": 0.045}

        result = compute_handedness_score(splits, baselines)

        assert result < 0
        assert result == pytest.approx(-0.667, abs=0.01)

    def test_handedness_score_no_splits(self):
        """Return 0 when no splits data."""
        baselines = {"std_woba_gap": 0.045}

        result = compute_handedness_score(None, baselines)

        assert result == 0.0

    def test_handedness_score_missing_woba(self):
        """Return 0 when wOBA values are missing."""
        splits = HitterSplits(
            woba_vs_hand=None,
            woba_overall=0.340,
            k_pct_vs_hand=0.18,
            iso_vs_hand=0.200,
            pa_vs_hand=50,
        )
        baselines = {"std_woba_gap": 0.045}

        result = compute_handedness_score(splits, baselines)

        assert result == 0.0


class TestComputePitcherScore:
    """Tests for compute_pitcher_score function."""

    def test_pitcher_score_elite_pitcher(self):
        """Negative score for elite pitcher (bad for hitter)."""
        pitcher = PitcherStats(
            name="Elite Pitcher",
            hand="R",
            era=2.20,
            whip=0.95,
            k_per_nine=11.0,
            mlbam_id=1,
        )
        baselines = {
            "mean_era": 4.20,
            "std_era": 0.85,
            "mean_whip": 1.28,
            "std_whip": 0.18,
        }

        result = compute_pitcher_score(pitcher, baselines)

        # Elite pitcher should give negative score (bad for hitter)
        assert result < 0

    def test_pitcher_score_weak_pitcher(self):
        """Positive score for weak pitcher (good for hitter)."""
        pitcher = PitcherStats(
            name="Weak Pitcher",
            hand="R",
            era=5.50,
            whip=1.50,
            k_per_nine=7.0,
            mlbam_id=1,
        )
        baselines = {
            "mean_era": 4.20,
            "std_era": 0.85,
            "mean_whip": 1.28,
            "std_whip": 0.18,
        }

        result = compute_pitcher_score(pitcher, baselines)

        # Weak pitcher should give positive score (good for hitter)
        assert result > 0

    def test_pitcher_score_no_pitcher(self):
        """Return 0 when no pitcher data."""
        baselines = {"mean_era": 4.20, "std_era": 0.85}

        result = compute_pitcher_score(None, baselines)

        assert result == 0.0

    def test_pitcher_score_only_era(self):
        """Compute score with only ERA available."""
        pitcher = PitcherStats(
            name="Partial Data",
            hand="R",
            era=3.50,
            whip=None,
            k_per_nine=None,
            mlbam_id=1,
        )
        baselines = {
            "mean_era": 4.20,
            "std_era": 0.85,
            "mean_whip": 1.28,
            "std_whip": 0.18,
        }

        result = compute_pitcher_score(pitcher, baselines)

        assert result != 0.0


class TestComputeParkScore:
    """Tests for compute_park_score function."""

    def test_park_score_coors_field(self):
        """High positive score for Coors Field."""
        result = compute_park_score(1.35)

        # (1.35 - 1.0) * 20 = 7.0
        assert result == pytest.approx(7.0, abs=0.01)

    def test_park_score_petco_park(self):
        """Negative score for pitcher-friendly park."""
        result = compute_park_score(0.88)

        # (0.88 - 1.0) * 20 = -2.4
        assert result == pytest.approx(-2.4, abs=0.01)

    def test_park_score_neutral(self):
        """Zero score for neutral park."""
        result = compute_park_score(1.0)

        assert result == 0.0


class TestComputeWeatherBonus:
    """Tests for compute_weather_bonus function."""

    def test_weather_wind_out(self):
        """Bonus for wind blowing out."""
        weather = WeatherData(
            temp_f=70.0,
            wind_mph=20.0,
            wind_direction="out",
            precip_chance=0.0,
        )

        result = compute_weather_bonus(weather)

        assert result == 3.0

    def test_weather_wind_in(self):
        """Penalty for wind blowing in."""
        weather = WeatherData(
            temp_f=70.0,
            wind_mph=20.0,
            wind_direction="in",
            precip_chance=0.0,
        )

        result = compute_weather_bonus(weather)

        assert result == -1.5

    def test_weather_hot_day(self):
        """Bonus for hot temperature."""
        weather = WeatherData(
            temp_f=90.0,
            wind_mph=5.0,
            wind_direction="cross",
            precip_chance=0.0,
        )

        result = compute_weather_bonus(weather)

        assert result == 1.5

    def test_weather_high_precip(self):
        """Penalty for high precipitation chance."""
        weather = WeatherData(
            temp_f=70.0,
            wind_mph=5.0,
            wind_direction="cross",
            precip_chance=60.0,
        )

        result = compute_weather_bonus(weather)

        assert result == -2.0

    def test_weather_combined(self):
        """Combined weather effects."""
        weather = WeatherData(
            temp_f=90.0,
            wind_mph=20.0,
            wind_direction="out",
            precip_chance=60.0,
        )

        result = compute_weather_bonus(weather)

        # Wind out (+3.0) + Hot (+1.5) + Precip (-2.0) = +2.5
        assert result == 2.5

    def test_weather_no_data(self):
        """Return 0 when no weather data."""
        result = compute_weather_bonus(None)

        assert result == 0.0


class TestComputeBullpenScore:
    """Tests for compute_bullpen_score function."""

    def test_bullpen_score_weak(self):
        """Positive score for weak bullpen (good for hitter)."""
        bullpen = BullpenStats(era=5.20, whip=1.45, pitcher_count=8)
        baselines = {"mean_bullpen_era": 4.35, "std_bullpen_era": 0.75}

        result = compute_bullpen_score(bullpen, baselines)

        # (5.20 - 4.35) / 0.75 = 1.13
        assert result > 0
        assert result == pytest.approx(1.133, abs=0.01)

    def test_bullpen_score_strong(self):
        """Negative score for strong bullpen (bad for hitter)."""
        bullpen = BullpenStats(era=3.20, whip=1.10, pitcher_count=8)
        baselines = {"mean_bullpen_era": 4.35, "std_bullpen_era": 0.75}

        result = compute_bullpen_score(bullpen, baselines)

        # (3.20 - 4.35) / 0.75 = -1.53
        assert result < 0
        assert result == pytest.approx(-1.533, abs=0.01)

    def test_bullpen_score_no_data(self):
        """Return 0 when no bullpen data."""
        baselines = {"mean_bullpen_era": 4.35, "std_bullpen_era": 0.75}

        result = compute_bullpen_score(None, baselines)

        assert result == 0.0

    def test_bullpen_score_no_era(self):
        """Return 0 when ERA is None."""
        bullpen = BullpenStats(era=None, whip=1.20, pitcher_count=8)
        baselines = {"mean_bullpen_era": 4.35, "std_bullpen_era": 0.75}

        result = compute_bullpen_score(bullpen, baselines)

        assert result == 0.0


# =============================================================================
# Confidence Tests
# =============================================================================

class TestComputeMatchupConfidence:
    """Tests for compute_matchup_confidence function."""

    def test_confidence_high_pa(self):
        """High confidence with many plate appearances."""
        splits = HitterSplits(
            woba_vs_hand=0.360,
            woba_overall=0.340,
            k_pct_vs_hand=0.18,
            iso_vs_hand=0.200,
            pa_vs_hand=100,
        )
        pitcher = PitcherStats(
            name="Pitcher", hand="R", era=3.50,
            whip=1.15, k_per_nine=9.0, mlbam_id=1,
        )
        baselines = {"min_split_pa": 30}

        result = compute_matchup_confidence(splits, pitcher, baselines)

        assert result > 0.7
        assert result <= 1.0

    def test_confidence_low_pa(self):
        """Lower confidence with few plate appearances."""
        splits = HitterSplits(
            woba_vs_hand=0.360,
            woba_overall=0.340,
            k_pct_vs_hand=0.18,
            iso_vs_hand=0.200,
            pa_vs_hand=10,
        )
        pitcher = PitcherStats(
            name="Pitcher", hand="R", era=3.50,
            whip=1.15, k_per_nine=9.0, mlbam_id=1,
        )
        baselines = {"min_split_pa": 30}

        result = compute_matchup_confidence(splits, pitcher, baselines)

        assert result < 0.5

    def test_confidence_no_pitcher(self):
        """Reduced confidence when pitcher data missing."""
        splits = HitterSplits(
            woba_vs_hand=0.360,
            woba_overall=0.340,
            k_pct_vs_hand=0.18,
            iso_vs_hand=0.200,
            pa_vs_hand=100,
        )
        baselines = {"min_split_pa": 30}

        result = compute_matchup_confidence(splits, None, baselines)

        # Should have penalty for missing pitcher
        assert result < 0.8

    def test_confidence_no_splits(self):
        """Reduced confidence when splits data missing."""
        pitcher = PitcherStats(
            name="Pitcher", hand="R", era=3.50,
            whip=1.15, k_per_nine=9.0, mlbam_id=1,
        )
        baselines = {"min_split_pa": 30}

        result = compute_matchup_confidence(None, pitcher, baselines)

        # Base confidence of 0.10 minus penalty
        assert result < 0.2


# =============================================================================
# Matchup Z-Score Tests
# =============================================================================

class TestComputeMatchupZ:
    """Tests for compute_matchup_z function."""

    def test_compute_matchup_z_full_data(self, sample_matchup_context):
        """Compute matchup Z with full context."""
        baselines = {
            "mean_era": 4.20,
            "std_era": 0.85,
            "mean_whip": 1.28,
            "std_whip": 0.18,
            "mean_bullpen_era": 4.35,
            "std_bullpen_era": 0.75,
            "std_woba_gap": 0.045,
            "min_split_pa": 30,
        }

        result = compute_matchup_z(sample_matchup_context, baselines)

        assert isinstance(result, MatchupResult)
        assert 0 <= result.matchup_score <= 100
        assert -5 <= result.matchup_z <= 5
        assert 0 <= result.matchup_confidence <= 1.0
        assert len(result.component_weights) > 0

    def test_compute_matchup_z_neutral(self):
        """Return neutral result with minimal data."""
        context = MatchupContext(
            bdl_player_id=1,
            game_date=date(2026, 5, 15),
            opponent_team="NYY",
            home_team="LAD",
            pitcher=None,
            splits=None,
            bullpen=None,
            weather=None,
            park_factor_runs=1.0,
            park_factor_hr=1.0,
        )

        result = compute_matchup_z(context)

        assert result.matchup_score == 50.0
        assert result.matchup_z == 0.0
        assert result.matchup_confidence == 0.0

    def test_compute_matchup_z_no_baselines(self, sample_matchup_context):
        """Use default baselines when none provided."""
        result = compute_matchup_z(sample_matchup_context, None)

        assert isinstance(result, MatchupResult)
        assert 0 <= result.matchup_score <= 100

    def test_compute_matchup_z_low_confidence_gate(self, sample_matchup_context):
        """Apply confidence gate to dampen signal."""
        # Create context with low PA (low confidence)
        sample_matchup_context.splits.pa_vs_hand = 5
        baselines = {
            "mean_era": 4.20, "std_era": 0.85,
            "mean_whip": 1.28, "std_whip": 0.18,
            "mean_bullpen_era": 4.35, "std_bullpen_era": 0.75,
            "std_woba_gap": 0.045, "min_split_pa": 30,
        }

        result = compute_matchup_z(sample_matchup_context, baselines)

        # Low confidence should dampen z-score
        assert result.matchup_confidence < 0.4

    def test_compute_matchup_z_score_clamping(self):
        """Ensure matchup score is clamped to [0, 100]."""
        # Create context that would produce extreme Z
        context = MatchupContext(
            bdl_player_id=1,
            game_date=date(2026, 5, 15),
            opponent_team="NYY",
            home_team="LAD",
            pitcher=PitcherStats(
                name="Bad Pitcher", hand="R", era=8.0,
                whip=2.0, k_per_nine=5.0, mlbam_id=1,
            ),
            splits=HitterSplits(
                woba_vs_hand=0.450, woba_overall=0.300,
                k_pct_vs_hand=0.10, iso_vs_hand=0.300, pa_vs_hand=200,
            ),
            bullpen=BullpenStats(era=6.0, whip=1.60, pitcher_count=8),
            weather=WeatherData(
                temp_f=95.0, wind_mph=25.0,
                wind_direction="out", precip_chance=0.0,
            ),
            park_factor_runs=1.35,
            park_factor_hr=1.40,
        )
        baselines = {
            "mean_era": 4.20, "std_era": 0.85,
            "mean_whip": 1.28, "std_whip": 0.18,
            "mean_bullpen_era": 4.35, "std_bullpen_era": 0.75,
            "std_woba_gap": 0.045, "min_split_pa": 30,
        }

        result = compute_matchup_z(context, baselines)

        assert result.matchup_score <= 100.0
        assert result.matchup_score >= 0.0


# =============================================================================
# Baseline Tests
# =============================================================================

class TestComputeBaselines:
    """Tests for compute_baselines function."""

    def test_compute_baselines_no_db(self):
        """Return defaults when no DB provided."""
        result = compute_baselines(None)

        assert result == _MLB_BASELINES_DEFAULT

    def test_compute_baselines_db_success(self, mock_db_session, mock_db_row):
        """Fetch baselines from database."""
        row = mock_db_row(
            mean_era=4.10,
            std_era=0.80,
            mean_whip=1.25,
            std_whip=0.16,
        )
        bullpen_row = mock_db_row(
            mean_bullpen_era=4.20,
            std_bullpen_era=0.70,
        )
        
        mock_db_session.execute.side_effect = [
            MagicMock(fetchone=Mock(return_value=row)),
            MagicMock(fetchone=Mock(return_value=bullpen_row)),
        ]

        result = compute_baselines(mock_db_session)

        assert result["mean_era"] == 4.10
        assert result["mean_whip"] == 1.25
        assert result["mean_bullpen_era"] == 4.20

    def test_compute_baselines_db_error(self, mock_db_session):
        """Return defaults on database error."""
        mock_db_session.execute.side_effect = Exception("DB Error")

        result = compute_baselines(mock_db_session)

        assert result == _MLB_BASELINES_DEFAULT

    def test_compute_baselines_partial_data(self, mock_db_session, mock_db_row):
        """Handle partial data from database."""
        row = mock_db_row(
            mean_era=None,
            std_era=None,
            mean_whip=None,
            std_whip=None,
        )
        bullpen_row = mock_db_row(
            mean_bullpen_era=None,
            std_bullpen_era=None,
        )
        
        mock_db_session.execute.side_effect = [
            MagicMock(fetchone=Mock(return_value=row)),
            MagicMock(fetchone=Mock(return_value=bullpen_row)),
        ]

        result = compute_baselines(mock_db_session)

        # Should fall back to defaults
        assert result["mean_era"] == _MLB_BASELINES_DEFAULT["mean_era"]


# =============================================================================
# Integration Tests
# =============================================================================

class TestMatchupEngineIntegration:
    """Integration tests for the full matchup engine pipeline."""

    def test_full_pipeline_favorable_matchup(self, mock_db_session, mock_db_row):
        """Test full pipeline for favorable hitter matchup."""
        # Setup: Hitter with good splits vs bad pitcher in hitter-friendly park
        pitcher_row = mock_db_row(
            pitcher_name="Weak Pitcher",
            mlbam_id=99999,
            era=6.00,
            whip=1.60,
            k_9=6.0,
            handedness="L",
        )
        splits_row = mock_db_row(
            pa_vs_hand=75, h_vs=25, bb_vs=8, hr_vs=5,
            ab_vs=67, k_vs=12, d_vs=5, t_vs=1,
        )
        overall_row = mock_db_row(
            total_h=120, total_bb=30, total_hr=15, total_ab=400,
        )
        bullpen_row = mock_db_row(
            bullpen_era=5.20, bullpen_whip=1.50, pitcher_count=8,
        )
        weather_row = mock_db_row(
            temperature_high=30.0, wind_speed=20.0,
            wind_direction="out", precipitation_probability=0.0,
        )

        mock_db_session.execute.side_effect = [
            MagicMock(fetchone=Mock(return_value=pitcher_row)),
            MagicMock(fetchone=Mock(return_value=splits_row)),
            MagicMock(fetchone=Mock(return_value=overall_row)),
            MagicMock(fetchone=Mock(return_value=bullpen_row)),
            MagicMock(fetchone=Mock(return_value=weather_row)),
        ]

        with patch('backend.services.matchup_engine._fetch_park_factor', return_value=1.20):
            context = collect_matchup_context(
                bdl_player_id=1,
                game_date=date(2026, 5, 15),
                opponent_team="COL",
                home_team="COL",
                db=mock_db_session,
            )

            result = compute_matchup_z(context)

        # Favorable matchup should have score > 50
        assert result.matchup_score > 50
        assert result.matchup_confidence > 0.5

    def test_full_pipeline_unfavorable_matchup(self, mock_db_session, mock_db_row):
        """Test full pipeline for unfavorable hitter matchup."""
        # Setup: Hitter with bad splits vs elite pitcher in pitcher-friendly park
        pitcher_row = mock_db_row(
            pitcher_name="Elite Pitcher",
            mlbam_id=88888,
            era=1.80,
            whip=0.85,
            k_9=12.0,
            handedness="R",
        )
        splits_row = mock_db_row(
            pa_vs_hand=40, h_vs=8, bb_vs=3, hr_vs=0,
            ab_vs=37, k_vs=15, d_vs=1, t_vs=0,
        )
        overall_row = mock_db_row(
            total_h=100, total_bb=25, total_hr=12, total_ab=350,
        )
        bullpen_row = mock_db_row(
            bullpen_era=3.00, bullpen_whip=1.10, pitcher_count=8,
        )
        weather_row = mock_db_row(
            temperature_high=15.0, wind_speed=5.0,
            wind_direction="in", precipitation_probability=0.0,
        )

        mock_db_session.execute.side_effect = [
            MagicMock(fetchone=Mock(return_value=pitcher_row)),
            MagicMock(fetchone=Mock(return_value=splits_row)),
            MagicMock(fetchone=Mock(return_value=overall_row)),
            MagicMock(fetchone=Mock(return_value=bullpen_row)),
            MagicMock(fetchone=Mock(return_value=weather_row)),
        ]

        with patch('backend.services.matchup_engine._fetch_park_factor', return_value=0.85):
            context = collect_matchup_context(
                bdl_player_id=1,
                game_date=date(2026, 5, 15),
                opponent_team="LAD",
                home_team="LAD",
                db=mock_db_session,
            )

            result = compute_matchup_z(context)

        # Unfavorable matchup should have score < 50
        assert result.matchup_score < 50
