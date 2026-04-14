"""Tests for StatcastIngestionAgent.transform_to_performance()."""
import pytest
import pandas as pd
from datetime import date
from unittest.mock import MagicMock, patch
from backend.fantasy_baseball.statcast_ingestion import StatcastIngestionAgent


@pytest.fixture(autouse=True)
def mock_resolver():
    """Mock the player_id_resolver to avoid DB connection."""
    mock_resolver = MagicMock()
    mock_resolver.resolve.return_value = "mlbam_12345"  # Mock MLB ID

    with patch('backend.fantasy_baseball.statcast_ingestion._player_id_resolver', mock_resolver):
        yield


def test_transform_to_performance_handles_missing_player_id_column():
    """CSVs without player_id column should use player_name as identifier."""
    agent = StatcastIngestionAgent()

    # Simulate Baseball Savant CSV schema: player_name present, player_id absent
    df = pd.DataFrame([
        {
            'player_name': 'Judge, Aaron',
            'team': 'NYY',
            'game_date': '2026-04-09',
            'pa': 5,
            'ab': 4,
            'h': 2,
            'double': 0,
            'doubles': 1,  # Alternate name
            'triple': 0,
            'triples': 0,
            'hr': 1,
            'r': 2,
            'rbi': 3,
            'bb': 1,
            'strikeout': 1,
            'so': 1,  # Alternate name
            'hbp': 0,
            'sb': 0,
            'cs': 0,
            'exit_velocity_avg': 95.5,
            'launch_angle_avg': 15.2,
            'hard_hit_percent': 55.0,
            'barrel_batted_rate': 12.0,
            'xba': 0.320,
            'xslg': 0.650,
            'xwoba': 0.400,
            'pitches': 25,
            '_statcast_player_type': 'batter',
        }
    ])

    performances = agent.transform_to_performance(df)

    # Should NOT return empty list (the bug we're fixing)
    assert len(performances) == 1, f"Expected 1 performance, got {len(performances)}"

    perf = performances[0]
    # player_id should be resolved from player_name (either mlbam_id or player_name fallback)
    assert perf.player_id is not None
    assert len(perf.player_id) > 0
    assert perf.player_name == 'Judge, Aaron'
    assert perf.pa == 5
    assert perf.hr == 1
    assert perf.xwoba == pytest.approx(0.400)


def test_transform_to_performance_skips_rows_with_missing_player_name():
    """Rows without player_name or player_id should be skipped."""
    agent = StatcastIngestionAgent()

    df = pd.DataFrame([
        {'player_name': None, 'team': 'NYY', 'game_date': '2026-04-09'},
        {'player_name': '', 'team': 'NYY', 'game_date': '2026-04-09'},
        {'player_name': 'nan', 'team': 'NYY', 'game_date': '2026-04-09'},
        {'player_name': 'Valid Player', 'team': 'NYY', 'game_date': '2026-04-09', 'pa': 1},
    ])

    performances = agent.transform_to_performance(df)

    # Only the valid row should be included
    assert len(performances) == 1
    assert performances[0].player_name == 'Valid Player'


def test_transform_to_performance_with_pitcher_rows():
    """Pitcher rows (_statcast_player_type='pitcher') should use zeroed batting stats."""
    agent = StatcastIngestionAgent()

    df = pd.DataFrame([
        {
            'player_name': 'Cole, Gerrit',
            'team': 'NYY',
            'game_date': '2026-04-09',
            'exit_velocity_avg': 88.0,
            'launch_angle_avg': 12.0,
            'hard_hit_percent': 35.0,
            'barrel_batted_rate': 5.0,
            'xba': 0.250,
            'xslg': 0.400,
            'xwoba': 0.300,
            'ip': 6.0,
            'er': 2,
            'strikeout': 8,
            'walk': 2,
            'bb': 2,
            'pitches': 95,
            '_statcast_player_type': 'pitcher',
        }
    ])

    performances = agent.transform_to_performance(df)

    assert len(performances) == 1
    perf = performances[0]
    # Pitcher rows should have zeroed batting stats
    assert perf.pa == 0
    assert perf.ab == 0
    assert perf.hr == 0
    # But pitching stats populated
    assert perf.ip == pytest.approx(6.0)
    assert perf.er == 2
    assert perf.k_pit == 8
    assert perf.bb_pit == 2
    assert perf.pitches == 95


def test_batter_performance_has_is_pitcher_false():
    """Batter rows should have is_pitcher=False."""
    agent = StatcastIngestionAgent()

    df = pd.DataFrame([{
        'player_name': 'Judge, Aaron',
        'team': 'NYY',
        'game_date': '2026-04-09',
        'pa': 5, 'ab': 4, 'h': 2,
        'doubles': 1, 'triples': 0, 'hr': 1, 'r': 2, 'rbi': 3,
        'bb': 1, 'so': 1, 'hbp': 0, 'sb': 0, 'cs': 0,
        'exit_velocity_avg': 95.5, 'launch_angle_avg': 15.2,
        'hard_hit_percent': 55.0, 'barrel_batted_rate': 12.0,
        'xba': 0.320, 'xslg': 0.650, 'xwoba': 0.400,
        'pitches': 25,
        '_statcast_player_type': 'batter',
    }])

    performances = agent.transform_to_performance(df)
    assert len(performances) == 1
    assert performances[0].is_pitcher is False


def test_pitcher_performance_has_is_pitcher_true():
    """Pitcher rows should have is_pitcher=True."""
    agent = StatcastIngestionAgent()

    df = pd.DataFrame([{
        'player_name': 'Cole, Gerrit',
        'team': 'NYY',
        'game_date': '2026-04-09',
        'exit_velocity_avg': 88.0, 'launch_angle_avg': 12.0,
        'hard_hit_percent': 35.0, 'barrel_batted_rate': 5.0,
        'xba': 0.250, 'xslg': 0.400, 'xwoba': 0.300,
        'ip': 6.0, 'er': 2, 'strikeout': 8, 'walk': 2, 'bb': 2,
        'pitches': 95,
        '_statcast_player_type': 'pitcher',
    }])

    performances = agent.transform_to_performance(df)
    assert len(performances) == 1
    assert performances[0].is_pitcher is True


# ---------------------------------------------------------------------------
# Tests for _aggregate_to_daily
# ---------------------------------------------------------------------------

class TestAggregateToDaily:
    """Tests for StatcastIngestionAgent._aggregate_to_daily()."""

    @pytest.fixture
    def agent(self):
        with patch('backend.fantasy_baseball.statcast_ingestion.SessionLocal') as mock_sl, \
             patch('backend.fantasy_baseball.statcast_ingestion._player_id_resolver') as mock_res:
            mock_sl.return_value = MagicMock()
            mock_res.load = MagicMock()
            mock_res.resolve = MagicMock(return_value='12345')
            a = StatcastIngestionAgent()
            if hasattr(a, '_diag_logged'):
                del a._diag_logged
            yield a

    def test_per_pitch_rows_aggregate_counting_stats(self, agent):
        """3 per-pitch rows for same batter/date, verify counting stats are SUMmed."""
        row = {
            'player_name': 'Judge, Aaron', 'team': 'NYY',
            'game_date': '2026-04-09', '_statcast_player_type': 'batter',
            'pa': 1, 'ab': 1, 'h': 1, 'hr': 0, 'bb': 0, 'so': 0,
            'sb': 1, 'pitches': 5,
        }
        df = pd.DataFrame([
            {**row, 'hr': 1, 'h': 1, 'sb': 0, 'pitches': 6},
            {**row, 'hr': 0, 'h': 0, 'sb': 1, 'pitches': 4},
            {**row, 'hr': 0, 'h': 1, 'sb': 0, 'pitches': 5},
        ])

        result = agent._aggregate_to_daily(df)
        assert len(result) == 1
        r = result.iloc[0]
        assert r['pa'] == 3
        assert r['ab'] == 3
        assert r['h'] == 2
        assert r['hr'] == 1
        assert r['bb'] == 0
        assert r['so'] == 0
        assert r['sb'] == 1
        assert r['pitches'] == 15

    def test_caught_stealing_indicators_sum_correctly(self, agent):
        """10 per-pitch rows, 3 with caught_stealing_2b=1, verify cs/sb sums."""
        base = {
            'player_name': 'Acuna, Ronald', 'team': 'ATL',
            'game_date': '2026-04-09', '_statcast_player_type': 'batter',
            'pa': 0, 'ab': 0, 'h': 0, 'cs': 0, 'caught_stealing_2b': 0,
            'sb': 0,
        }
        rows = []
        for i in range(10):
            r = dict(base)
            if i < 3:
                r['caught_stealing_2b'] = 1
                r['cs'] = 1
            if i < 2:
                r['sb'] = 1
            rows.append(r)

        df = pd.DataFrame(rows)
        result = agent._aggregate_to_daily(df)
        assert len(result) == 1
        assert result.iloc[0]['cs'] == 3
        assert result.iloc[0]['caught_stealing_2b'] == 3
        assert result.iloc[0]['sb'] == 2

    def test_quality_metrics_averaged_not_summed(self, agent):
        """2 rows with different exit_velocity/xwoba, verify they are averaged."""
        base = {
            'player_name': 'Soto, Juan', 'team': 'NYM',
            'game_date': '2026-04-09', '_statcast_player_type': 'batter',
            'pa': 1, 'ab': 1,
        }
        df = pd.DataFrame([
            {**base, 'exit_velocity_avg': 90.0, 'xwoba': 0.300},
            {**base, 'exit_velocity_avg': 100.0, 'xwoba': 0.500},
        ])

        result = agent._aggregate_to_daily(df)
        assert len(result) == 1
        assert result.iloc[0]['exit_velocity_avg'] == pytest.approx(95.0)
        assert result.iloc[0]['xwoba'] == pytest.approx(0.400)
        # Counting stats should be summed, not averaged
        assert result.iloc[0]['pa'] == 2

    def test_leaderboard_single_row_passthrough(self, agent):
        """Single row per player passes through unchanged."""
        df = pd.DataFrame([{
            'player_name': 'Ohtani, Shohei', 'team': 'LAD',
            'game_date': '2026-04-09', '_statcast_player_type': 'batter',
            'pa': 5, 'ab': 4, 'h': 2, 'hr': 1,
            'exit_velocity_avg': 95.0, 'xwoba': 0.450,
        }])

        result = agent._aggregate_to_daily(df)
        assert len(result) == 1
        assert result.iloc[0]['pa'] == 5
        assert result.iloc[0]['hr'] == 1
        assert result.iloc[0]['exit_velocity_avg'] == pytest.approx(95.0)

    def test_multiple_players_stay_separate(self, agent):
        """2 different players on same date produce 2 output rows."""
        base = {
            'game_date': '2026-04-09', '_statcast_player_type': 'batter',
            'pa': 1, 'ab': 1, 'h': 1,
        }
        df = pd.DataFrame([
            {**base, 'player_name': 'Judge, Aaron', 'team': 'NYY'},
            {**base, 'player_name': 'Judge, Aaron', 'team': 'NYY'},
            {**base, 'player_name': 'Soto, Juan', 'team': 'NYM'},
            {**base, 'player_name': 'Soto, Juan', 'team': 'NYM'},
        ])

        result = agent._aggregate_to_daily(df)
        assert len(result) == 2
        names = set(result['player_name'].tolist())
        assert names == {'Judge, Aaron', 'Soto, Juan'}
        # Each should have pa=2 (summed from 2 rows)
        for _, row in result.iterrows():
            assert row['pa'] == 2

    def test_batter_and_pitcher_rows_stay_separate(self, agent):
        """Same player as batter AND pitcher produces 2 output rows."""
        base = {
            'player_name': 'Ohtani, Shohei', 'team': 'LAD',
            'game_date': '2026-04-09', 'pa': 1, 'ab': 1, 'h': 1,
        }
        df = pd.DataFrame([
            {**base, '_statcast_player_type': 'batter'},
            {**base, '_statcast_player_type': 'batter'},
            {**base, '_statcast_player_type': 'pitcher'},
            {**base, '_statcast_player_type': 'pitcher'},
        ])

        result = agent._aggregate_to_daily(df)
        assert len(result) == 2
        types = set(result['_statcast_player_type'].tolist())
        assert types == {'batter', 'pitcher'}
