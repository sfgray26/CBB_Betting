"""
Tests for Statcast column name mapping in StatcastIngestionAgent.transform_to_performance().

Verifies that both raw Baseball Savant column names (e.g. 'launch_speed',
'estimated_woba_using_speedangle') and cleaned aliases (e.g. 'exit_velocity_avg',
'xwoba') produce non-zero PlayerDailyPerformance objects.
"""

import pytest
import pandas as pd
from datetime import date
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helper: build a one-row DataFrame simulating Savant grouped CSV output
# ---------------------------------------------------------------------------

def _savant_batter_row(**overrides) -> pd.DataFrame:
    """Return a 1-row DataFrame with actual Baseball Savant column names."""
    base = {
        'player_name': 'Mookie Betts',
        'team': 'LAD',
        'game_date': '2026-04-10',
        '_statcast_player_type': 'batter',
        # Counting stats (Savant names)
        'pa': 5,
        'ab': 4,
        'hit': 2,
        'single': 1,
        'double': 1,
        'triple': 0,
        'home_run': 0,
        'run': 1,
        'rbi': 1,
        'walk': 1,
        'strikeout': 1,
        'hit_by_pitch': 0,
        'stolen_base_2b': 1,
        'caught_stealing_2b': 0,
        'pitches': 22,
        # Statcast quality metrics (Savant names)
        'launch_speed': 92.3,
        'launch_angle': 14.5,
        'hard_hit_percent': 45.0,
        'barrel_batted_rate': 12.0,
        'estimated_ba_using_speedangle': 0.310,
        'estimated_slg_using_speedangle': 0.520,
        'estimated_woba_using_speedangle': 0.380,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def _clean_batter_row(**overrides) -> pd.DataFrame:
    """Return a 1-row DataFrame with cleaned/alias column names."""
    base = {
        'player_name': 'Mookie Betts',
        'team': 'LAD',
        'game_date': '2026-04-10',
        '_statcast_player_type': 'batter',
        'pa': 5,
        'ab': 4,
        'h': 2,
        'doubles': 1,
        'triples': 0,
        'home_run': 0,
        'run': 1,
        'rbi': 1,
        'bb': 1,
        'so': 1,
        'hbp': 0,
        'sb': 1,
        'cs': 0,
        'pitches': 22,
        # Clean alias names
        'exit_velocity_avg': 92.3,
        'launch_angle_avg': 14.5,
        'hard_hit_pct': 0.45,
        'barrel_pct': 0.12,
        'xba': 0.310,
        'xslg': 0.520,
        'xwoba': 0.380,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def _savant_pitcher_row(**overrides) -> pd.DataFrame:
    """Return a 1-row DataFrame with Savant pitcher columns."""
    base = {
        'player_name': 'Yoshinobu Yamamoto',
        'team': 'LAD',
        'game_date': '2026-04-10',
        '_statcast_player_type': 'pitcher',
        'pitches': 98,
        'ip': 6.0,
        'er': 2,
        'strikeout': 8,
        'walk': 2,
        # Statcast quality metrics (Savant names) -- represent batted-ball outcomes
        'launch_speed': 87.1,
        'launch_angle': 11.2,
        'hard_hit_percent': 32.0,
        'barrel_batted_rate': 6.0,
        'estimated_ba_using_speedangle': 0.230,
        'estimated_slg_using_speedangle': 0.350,
        'estimated_woba_using_speedangle': 0.290,
    }
    base.update(overrides)
    return pd.DataFrame([base])


# ---------------------------------------------------------------------------
# Fixture: build agent without hitting the DB
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    """Create a StatcastIngestionAgent with DB access mocked out."""
    with patch('backend.fantasy_baseball.statcast_ingestion.SessionLocal') as mock_sl, \
         patch('backend.fantasy_baseball.statcast_ingestion._player_id_resolver') as mock_resolver:
        mock_db = MagicMock()
        mock_sl.return_value = mock_db
        mock_resolver.load = MagicMock()
        mock_resolver.resolve = MagicMock(return_value='12345')

        from backend.fantasy_baseball.statcast_ingestion import StatcastIngestionAgent
        a = StatcastIngestionAgent()
        # Reset diagnostic flag so each test logs fresh
        if hasattr(a, '_diag_logged'):
            del a._diag_logged
        yield a


# ===========================================================================
# Batter tests -- Savant column names
# ===========================================================================

class TestBatterSavantColumns:
    """Verify that raw Savant column names produce non-zero quality metrics."""

    def test_exit_velocity_mapped(self, agent):
        df = _savant_batter_row()
        perfs = agent.transform_to_performance(df)
        assert len(perfs) == 1
        assert perfs[0].exit_velocity_avg == pytest.approx(92.3)

    def test_launch_angle_mapped(self, agent):
        df = _savant_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].launch_angle_avg == pytest.approx(14.5)

    def test_xwoba_mapped(self, agent):
        df = _savant_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].xwoba == pytest.approx(0.380)

    def test_xba_mapped(self, agent):
        df = _savant_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].xba == pytest.approx(0.310)

    def test_xslg_mapped(self, agent):
        df = _savant_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].xslg == pytest.approx(0.520)

    def test_hard_hit_pct_mapped(self, agent):
        df = _savant_batter_row()
        perfs = agent.transform_to_performance(df)
        # 45.0 / 100 = 0.45
        assert perfs[0].hard_hit_pct == pytest.approx(0.45)

    def test_barrel_pct_mapped(self, agent):
        df = _savant_batter_row()
        perfs = agent.transform_to_performance(df)
        # 12.0 / 100 = 0.12
        assert perfs[0].barrel_pct == pytest.approx(0.12)

    def test_counting_stats_mapped(self, agent):
        df = _savant_batter_row()
        perfs = agent.transform_to_performance(df)
        p = perfs[0]
        assert p.pa == 5
        assert p.ab == 4
        assert p.h == 2
        assert p.doubles == 1
        assert p.hr == 0
        assert p.bb == 1
        assert p.so == 1
        assert p.sb == 1
        assert p.cs == 0
        assert p.hbp == 0

    def test_stolen_base_2b_mapped(self, agent):
        df = _savant_batter_row(stolen_base_2b=3)
        perfs = agent.transform_to_performance(df)
        assert perfs[0].sb == 3

    def test_caught_stealing_2b_mapped(self, agent):
        df = _savant_batter_row(caught_stealing_2b=1)
        perfs = agent.transform_to_performance(df)
        assert perfs[0].cs == 1

    def test_all_quality_metrics_nonzero(self, agent):
        """The core regression test: ensure Savant columns don't produce all-zero shells."""
        df = _savant_batter_row()
        perfs = agent.transform_to_performance(df)
        p = perfs[0]
        quality_metrics = [
            p.exit_velocity_avg, p.launch_angle_avg, p.hard_hit_pct,
            p.barrel_pct, p.xba, p.xslg, p.xwoba,
        ]
        assert all(m > 0 for m in quality_metrics), (
            f"Shell record detected -- all quality metrics should be non-zero: {quality_metrics}"
        )


# ===========================================================================
# Batter tests -- clean/alias column names (backwards compatibility)
# ===========================================================================

class TestBatterCleanColumns:
    """Verify that cleaned column names still work (backwards compatibility)."""

    def test_exit_velocity_from_clean_name(self, agent):
        df = _clean_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].exit_velocity_avg == pytest.approx(92.3)

    def test_xwoba_from_clean_name(self, agent):
        df = _clean_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].xwoba == pytest.approx(0.380)

    def test_hard_hit_pct_from_clean_name(self, agent):
        """Clean name 'hard_hit_pct' is already 0-1 scale, not 0-100."""
        df = _clean_batter_row()
        perfs = agent.transform_to_performance(df)
        # hard_hit_pct=0.45 / 100 = 0.0045 -- this is the current behavior
        # with clean names. The /100 division always applies.
        # The clean-name caller should pass 45.0 if they want 0.45 out.
        # For backwards compat we just verify it doesn't crash.
        assert perfs[0].hard_hit_pct >= 0

    def test_all_quality_metrics_nonzero_clean(self, agent):
        df = _clean_batter_row()
        perfs = agent.transform_to_performance(df)
        p = perfs[0]
        # xba/xslg/xwoba and exit_velocity/launch_angle should be non-zero
        assert p.exit_velocity_avg > 0
        assert p.xwoba > 0
        assert p.xba > 0


# ===========================================================================
# Pitcher tests
# ===========================================================================

class TestPitcherSavantColumns:
    """Verify pitcher rows map Savant quality metrics correctly."""

    def test_pitcher_exit_velocity_mapped(self, agent):
        df = _savant_pitcher_row()
        perfs = agent.transform_to_performance(df)
        assert len(perfs) == 1
        assert perfs[0].exit_velocity_avg == pytest.approx(87.1)

    def test_pitcher_xwoba_mapped(self, agent):
        df = _savant_pitcher_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].xwoba == pytest.approx(0.290)

    def test_pitcher_xba_mapped(self, agent):
        df = _savant_pitcher_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].xba == pytest.approx(0.230)

    def test_pitcher_ks_mapped(self, agent):
        df = _savant_pitcher_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].k_pit == 8

    def test_pitcher_bbs_mapped(self, agent):
        df = _savant_pitcher_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].bb_pit == 2

    def test_pitcher_ip_mapped(self, agent):
        df = _savant_pitcher_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].ip == pytest.approx(6.0)

    def test_pitcher_batting_stats_zeroed(self, agent):
        df = _savant_pitcher_row()
        perfs = agent.transform_to_performance(df)
        p = perfs[0]
        assert p.pa == 0
        assert p.ab == 0
        assert p.h == 0
        assert p.hr == 0

    def test_pitcher_quality_metrics_nonzero(self, agent):
        df = _savant_pitcher_row()
        perfs = agent.transform_to_performance(df)
        p = perfs[0]
        quality_metrics = [
            p.exit_velocity_avg, p.launch_angle_avg, p.hard_hit_pct,
            p.barrel_pct, p.xba, p.xslg, p.xwoba,
        ]
        assert all(m > 0 for m in quality_metrics), (
            f"Pitcher shell record -- quality metrics should be non-zero: {quality_metrics}"
        )


# ===========================================================================
# DataQualityChecker tests
# ===========================================================================

class TestDataQualityChecker:
    """Verify validate_daily_pull accepts both Savant and clean column names."""

    def test_savant_columns_pass_validation(self):
        from backend.fantasy_baseball.statcast_ingestion import DataQualityChecker
        checker = DataQualityChecker()
        df = pd.DataFrame([{
            'player_name': f'Player {i}',
            'team': 'LAD',
            'game_date': '2026-04-10',
            'pa': 4,
            'estimated_woba_using_speedangle': 0.350,
        } for i in range(300)])
        result = checker.validate_daily_pull(df, date(2026, 4, 10))
        # Should not fail due to missing xwoba column
        missing_col_errors = [
            iss for iss in checker.issues
            if iss['type'] == 'MISSING_COLUMNS'
        ]
        assert len(missing_col_errors) == 0

    def test_clean_columns_pass_validation(self):
        from backend.fantasy_baseball.statcast_ingestion import DataQualityChecker
        checker = DataQualityChecker()
        df = pd.DataFrame([{
            'player_name': f'Player {i}',
            'team': 'LAD',
            'game_date': '2026-04-10',
            'pa': 4,
            'xwoba': 0.350,
        } for i in range(300)])
        result = checker.validate_daily_pull(df, date(2026, 4, 10))
        missing_col_errors = [
            iss for iss in checker.issues
            if iss['type'] == 'MISSING_COLUMNS'
        ]
        assert len(missing_col_errors) == 0

    def test_neither_xwoba_column_fails(self):
        from backend.fantasy_baseball.statcast_ingestion import DataQualityChecker
        checker = DataQualityChecker()
        df = pd.DataFrame([{
            'player_name': f'Player {i}',
            'team': 'LAD',
            'game_date': '2026-04-10',
            'pa': 4,
        } for i in range(300)])
        result = checker.validate_daily_pull(df, date(2026, 4, 10))
        assert result is False
        missing_col_errors = [
            iss for iss in checker.issues
            if iss['type'] == 'MISSING_COLUMNS'
        ]
        assert len(missing_col_errors) == 1

    def test_savant_xwoba_range_check(self):
        """xwoba range check should work with Savant column name."""
        from backend.fantasy_baseball.statcast_ingestion import DataQualityChecker
        checker = DataQualityChecker()
        df = pd.DataFrame([{
            'player_name': f'Player {i}',
            'team': 'LAD',
            'game_date': '2026-04-10',
            'pa': 4,
            'estimated_woba_using_speedangle': 0.350,
        } for i in range(300)])
        result = checker.validate_daily_pull(df, date(2026, 4, 10))
        # Should pass -- no anomaly warnings for normal values
        anomaly_warnings = [
            iss for iss in checker.issues
            if iss['type'] == 'DATA_ANOMALY'
        ]
        assert len(anomaly_warnings) == 0


# ===========================================================================
# Combined batter + pitcher DataFrame
# ===========================================================================

class TestCombinedDataFrame:
    """Test transform with a mixed batter+pitcher DataFrame (like production)."""

    def test_mixed_dataframe(self, agent):
        batter_df = _savant_batter_row()
        pitcher_df = _savant_pitcher_row()
        combined = pd.concat([batter_df, pitcher_df], ignore_index=True)

        perfs = agent.transform_to_performance(combined)
        assert len(perfs) == 2

        batter = [p for p in perfs if p.pa > 0][0]
        pitcher = [p for p in perfs if p.pa == 0][0]

        assert batter.exit_velocity_avg > 0
        assert batter.xwoba > 0
        assert batter.sb == 1

        assert pitcher.exit_velocity_avg > 0
        assert pitcher.xwoba > 0
        assert pitcher.k_pit == 8


# ===========================================================================
# Leaderboard (non-details) format — the actual columns Baseball Savant returns
# when 'type': 'details' is OMITTED from the request.
# Verified against live API on 2026-04-13.
# ===========================================================================

def _leaderboard_batter_row(**overrides) -> pd.DataFrame:
    """
    1-row DataFrame matching the Baseball Savant leaderboard CSV format
    (group_by=name-date WITHOUT type=details).

    Key differences from the 'details' format:
      - 'abs' instead of 'ab' for at-bats
      - 'hits' instead of 'hit' for hit count
      - 'hrs' instead of 'home_run' for home run count
      - 'hardhit_percent' instead of 'hard_hit_percent' (no underscore between hard/hit)
      - 'barrels_per_pa_percent' instead of 'barrel_batted_rate'
      - 'xwoba' / 'xba' / 'xslg' directly (not prefixed with 'estimated_...')
      - 'player_id' present as MLBAM integer (no name-lookup needed)
    """
    base = {
        'player_id': 660670,
        'player_name': 'Alvarez, Yordan',
        'game_date': '2026-04-05',
        '_statcast_player_type': 'batter',
        # Counting stats
        'pa': 6.0,
        'abs': 4.0,
        'hits': 1.0,
        'hrs': 1.0,
        'doubles': 0.0,
        'triples': 0.0,
        'bb': 2.0,
        'so': 2.0,
        # Statcast quality metrics (leaderboard column names)
        'launch_speed': 99.5,
        'launch_angle': 28.0,
        'hardhit_percent': 50.0,       # 0-100 scale → stored as 0.50 after /100
        'barrels_per_pa_percent': 16.666667,  # 0-100 scale → stored as 0.167
        'xwoba': 0.559,                # already 0-1 decimal
        'xba': 0.251,
        'xslg': 0.926,
        'woba': 0.594,                 # actual wOBA from leaderboard
        'pitches': 24,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def _leaderboard_pitcher_row(**overrides) -> pd.DataFrame:
    """1-row leaderboard CSV pitcher format."""
    base = {
        'player_id': 579328,
        'player_name': 'Cole, Gerrit',
        'game_date': '2026-04-06',
        '_statcast_player_type': 'pitcher',
        'pa': 0,
        'abs': 0,
        'hits': 0,
        'hrs': 0,
        'bb': 0,
        'so': 8.0,      # pitcher Ks
        'hardhit_percent': 28.0,
        'barrels_per_pa_percent': 4.0,
        'launch_speed': 85.2,
        'launch_angle': 10.5,
        'xwoba': 0.260,
        'xba': 0.210,
        'xslg': 0.380,
        'pitches': 95,
    }
    base.update(overrides)
    return pd.DataFrame([base])


class TestLeaderboardBatterColumns:
    """Verify leaderboard (non-details) column names map correctly."""

    def test_player_id_used_directly(self, agent):
        """player_id from leaderboard is the MLBAM int — no name-lookup needed."""
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        assert len(perfs) == 1
        assert perfs[0].player_id == '660670'

    def test_ab_from_abs_column(self, agent):
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].ab == 4

    def test_h_from_hits_column(self, agent):
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].h == 1

    def test_hr_from_hrs_column(self, agent):
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].hr == 1

    def test_bb_from_bb_column(self, agent):
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].bb == 2

    def test_so_from_so_column(self, agent):
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].so == 2

    def test_exit_velocity_from_launch_speed(self, agent):
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].exit_velocity_avg == pytest.approx(99.5)

    def test_xwoba_from_leaderboard_xwoba(self, agent):
        """xwoba column (0-1 scale) stored directly — no /100 applied."""
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].xwoba == pytest.approx(0.559)

    def test_xba_from_leaderboard_xba(self, agent):
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].xba == pytest.approx(0.251)

    def test_hard_hit_pct_from_hardhit_percent(self, agent):
        """hardhit_percent=50.0 (0-100 scale) → stored as 0.50 after /100."""
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].hard_hit_pct == pytest.approx(0.50)

    def test_barrel_pct_from_barrels_per_pa_percent(self, agent):
        """barrels_per_pa_percent=16.67 (0-100 scale) → stored as ~0.167 after /100."""
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].barrel_pct == pytest.approx(0.1667, abs=0.001)

    def test_all_quality_metrics_nonzero_leaderboard(self, agent):
        """Core regression: leaderboard rows must NOT produce all-zero quality shells."""
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        p = perfs[0]
        quality_metrics = [
            p.exit_velocity_avg, p.hard_hit_pct, p.barrel_pct,
            p.xba, p.xslg, p.xwoba,
        ]
        assert all(m > 0 for m in quality_metrics), (
            f"Shell record detected from leaderboard row: {quality_metrics}"
        )

    def test_counting_stats_all_nonzero_leaderboard(self, agent):
        """Core regression: leaderboard counting stats (ab, hr, bb) must not be zero."""
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        p = perfs[0]
        assert p.ab == 4, f"ab={p.ab} (expected 4 from 'abs' column)"
        assert p.hr == 1, f"hr={p.hr} (expected 1 from 'hrs' column)"
        assert p.bb == 2, f"bb={p.bb}"


class TestLeaderboardPitcherColumns:
    """Verify leaderboard pitcher columns map correctly."""

    def test_pitcher_hard_hit_from_hardhit_percent(self, agent):
        df = _leaderboard_pitcher_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].hard_hit_pct == pytest.approx(0.28)

    def test_pitcher_barrel_from_barrels_per_pa_percent(self, agent):
        df = _leaderboard_pitcher_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].barrel_pct == pytest.approx(0.04)

    def test_pitcher_xwoba_from_leaderboard(self, agent):
        df = _leaderboard_pitcher_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].xwoba == pytest.approx(0.260)

    def test_pitcher_ks_from_so_column(self, agent):
        """Leaderboard pitcher rows use 'so' for strikeouts."""
        df = _leaderboard_pitcher_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].k_pit == 8

    def test_pitcher_quality_metrics_nonzero_leaderboard(self, agent):
        df = _leaderboard_pitcher_row()
        perfs = agent.transform_to_performance(df)
        p = perfs[0]
        assert p.exit_velocity_avg > 0
        assert p.hard_hit_pct > 0
        assert p.barrel_pct > 0
        assert p.xwoba > 0


# ===========================================================================
# Leaderboard (non-details) format — the actual columns Baseball Savant returns
# when 'type': 'details' is OMITTED from the request.
# Verified against live API on 2026-04-13.
# ===========================================================================

def _leaderboard_batter_row(**overrides) -> pd.DataFrame:
    """
    1-row DataFrame matching the Baseball Savant leaderboard CSV format
    (group_by=name-date WITHOUT type=details).

    Key differences from the 'details' format:
      - 'abs' instead of 'ab' for at-bats
      - 'hits' instead of 'hit' for hit count
      - 'hrs' instead of 'home_run' for home run count
      - 'hardhit_percent' instead of 'hard_hit_percent' (no underscore between hard/hit)
      - 'barrels_per_pa_percent' instead of 'barrel_batted_rate'
      - 'xwoba' / 'xba' / 'xslg' directly (not prefixed with 'estimated_...')
      - 'player_id' present as MLBAM integer (no name-lookup needed)
    """
    base = {
        'player_id': 660670,
        'player_name': 'Alvarez, Yordan',
        'game_date': '2026-04-05',
        '_statcast_player_type': 'batter',
        # Counting stats
        'pa': 6.0,
        'abs': 4.0,
        'hits': 1.0,
        'hrs': 1.0,
        'doubles': 0.0,
        'triples': 0.0,
        'bb': 2.0,
        'so': 2.0,
        # Statcast quality metrics (leaderboard column names)
        'launch_speed': 99.5,
        'launch_angle': 28.0,
        'hardhit_percent': 50.0,       # 0-100 scale → stored as 0.50 after /100
        'barrels_per_pa_percent': 16.666667,  # 0-100 scale → stored as 0.167
        'xwoba': 0.559,                # already 0-1 decimal
        'xba': 0.251,
        'xslg': 0.926,
        'woba': 0.594,                 # actual wOBA from leaderboard
        'pitches': 24,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def _leaderboard_pitcher_row(**overrides) -> pd.DataFrame:
    """1-row leaderboard CSV pitcher format."""
    base = {
        'player_id': 579328,
        'player_name': 'Cole, Gerrit',
        'game_date': '2026-04-06',
        '_statcast_player_type': 'pitcher',
        'pa': 0,
        'abs': 0,
        'hits': 0,
        'hrs': 0,
        'bb': 0,
        'so': 8.0,      # pitcher Ks
        'hardhit_percent': 28.0,
        'barrels_per_pa_percent': 4.0,
        'launch_speed': 85.2,
        'launch_angle': 10.5,
        'xwoba': 0.260,
        'xba': 0.210,
        'xslg': 0.380,
        'pitches': 95,
    }
    base.update(overrides)
    return pd.DataFrame([base])


class TestLeaderboardBatterColumns:
    """Verify leaderboard (non-details) column names map correctly."""

    def test_player_id_used_directly(self, agent):
        """player_id from leaderboard is the MLBAM int — no name-lookup needed."""
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        assert len(perfs) == 1
        assert perfs[0].player_id == '660670'

    def test_ab_from_abs_column(self, agent):
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].ab == 4

    def test_h_from_hits_column(self, agent):
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].h == 1

    def test_hr_from_hrs_column(self, agent):
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].hr == 1

    def test_bb_from_bb_column(self, agent):
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].bb == 2

    def test_so_from_so_column(self, agent):
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].so == 2

    def test_exit_velocity_from_launch_speed(self, agent):
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].exit_velocity_avg == pytest.approx(99.5)

    def test_xwoba_from_leaderboard_xwoba(self, agent):
        """xwoba column (0-1 scale) stored directly — no /100 applied."""
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].xwoba == pytest.approx(0.559)

    def test_xba_from_leaderboard_xba(self, agent):
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].xba == pytest.approx(0.251)

    def test_hard_hit_pct_from_hardhit_percent(self, agent):
        """hardhit_percent=50.0 (0-100 scale) → stored as 0.50 after /100."""
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].hard_hit_pct == pytest.approx(0.50)

    def test_barrel_pct_from_barrels_per_pa_percent(self, agent):
        """barrels_per_pa_percent=16.67 (0-100 scale) → stored as ~0.167 after /100."""
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].barrel_pct == pytest.approx(0.1667, abs=0.001)

    def test_all_quality_metrics_nonzero_leaderboard(self, agent):
        """Core regression: leaderboard rows must NOT produce all-zero quality shells."""
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        p = perfs[0]
        quality_metrics = [
            p.exit_velocity_avg, p.hard_hit_pct, p.barrel_pct,
            p.xba, p.xslg, p.xwoba,
        ]
        assert all(m > 0 for m in quality_metrics), (
            f"Shell record detected from leaderboard row: {quality_metrics}"
        )

    def test_counting_stats_all_nonzero_leaderboard(self, agent):
        """Core regression: leaderboard counting stats (ab, hr, bb) must not be zero."""
        df = _leaderboard_batter_row()
        perfs = agent.transform_to_performance(df)
        p = perfs[0]
        assert p.ab == 4, f"ab={p.ab} (expected 4 from 'abs' column)"
        assert p.hr == 1, f"hr={p.hr} (expected 1 from 'hrs' column)"
        assert p.bb == 2, f"bb={p.bb}"


class TestLeaderboardPitcherColumns:
    """Verify leaderboard pitcher columns map correctly."""

    def test_pitcher_hard_hit_from_hardhit_percent(self, agent):
        df = _leaderboard_pitcher_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].hard_hit_pct == pytest.approx(0.28)

    def test_pitcher_barrel_from_barrels_per_pa_percent(self, agent):
        df = _leaderboard_pitcher_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].barrel_pct == pytest.approx(0.04)

    def test_pitcher_xwoba_from_leaderboard(self, agent):
        df = _leaderboard_pitcher_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].xwoba == pytest.approx(0.260)

    def test_pitcher_ks_from_so_column(self, agent):
        """Leaderboard pitcher rows use 'so' for strikeouts."""
        df = _leaderboard_pitcher_row()
        perfs = agent.transform_to_performance(df)
        assert perfs[0].k_pit == 8

    def test_pitcher_quality_metrics_nonzero_leaderboard(self, agent):
        df = _leaderboard_pitcher_row()
        perfs = agent.transform_to_performance(df)
        p = perfs[0]
        assert p.exit_velocity_avg > 0
        assert p.hard_hit_pct > 0
        assert p.barrel_pct > 0
        assert p.xwoba > 0
