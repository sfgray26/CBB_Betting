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
