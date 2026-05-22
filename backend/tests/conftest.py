"""
Shared pytest fixtures for backend tests.

This module provides:
- Mock fixtures for database sessions
- Mock fixtures for external API clients (Yahoo, BDL, Statcast)
- Sample data factories for testing
- Configuration patches
"""

import pytest
from datetime import date, datetime
from decimal import Decimal
from unittest.mock import Mock, MagicMock, patch
from zoneinfo import ZoneInfo
import sys
import os

# Add the backend to the path
sys.path.insert(0, '/mnt/c/Users/sfgra/repos/Fixed/cbb-edge')
sys.path.insert(0, '/mnt/c/Users/sfgra/repos/Fixed/cbb-edge/backend')


# =============================================================================
# Date and Time Fixtures
# =============================================================================

@pytest.fixture
def sample_date() -> date:
    """Return a sample date for testing."""
    return date(2026, 5, 15)


@pytest.fixture
def sample_datetime() -> datetime:
    """Return a sample datetime in ET for testing."""
    return datetime(2026, 5, 15, 12, 0, 0, tzinfo=ZoneInfo("America/New_York"))


@pytest.fixture
def opening_day_date() -> date:
    """Return the MLB Opening Day 2026 date."""
    return date(2026, 3, 27)


# =============================================================================
# Database Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_db_session():
    """Return a mock SQLAlchemy session."""
    mock_session = Mock()
    mock_session.execute = Mock()
    mock_session.query = Mock()
    mock_session.add = Mock()
    mock_session.commit = Mock()
    mock_session.rollback = Mock()
    mock_session.close = Mock()
    return mock_session


@pytest.fixture
def mock_db_row():
    """Factory for creating mock database row results."""
    def _create_row(**kwargs):
        row = Mock()
        # Support both tuple-like and dict-like access
        row.__iter__ = Mock(return_value=iter(kwargs.values()))
        row._asdict = Mock(return_value=kwargs)
        row._mapping = kwargs
        for key, value in kwargs.items():
            setattr(row, key, value)
        return row
    return _create_row


# =============================================================================
# Model Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_player_rolling_stats():
    """Factory for creating mock PlayerRollingStats objects."""
    def _create_stats(
        bdl_player_id=1,
        as_of_date=None,
        window_days=14,
        **kwargs
    ):
        if as_of_date is None:
            as_of_date = date(2026, 5, 15)
        
        mock_stats = Mock()
        mock_stats.bdl_player_id = bdl_player_id
        mock_stats.as_of_date = as_of_date
        mock_stats.window_days = window_days
        
        # Default batting stats
        defaults = {
            'w_runs': 5.0,
            'w_hits': 8.5,
            'w_home_runs': 1.5,
            'w_rbi': 5.2,
            'w_strikeouts_bat': 7.0,
            'w_tb': 13.5,
            'w_net_stolen_bases': 0.5,
            'w_avg': 0.275,
            'w_ops': 0.850,
            'w_ab': 30.0,
            'w_walks': 3.5,
            # Pitching stats
            'w_strikeouts_pit': 12.0,
            'w_era': 3.50,
            'w_whip': 1.15,
            'w_k_per_9': 9.5,
            'w_qs': 1.0,
            'w_ip': 6.0,
            'w_earned_runs': 2.5,
            'w_hits_allowed': 5.0,
            'w_walks_allowed': 2.0,
        }
        defaults.update(kwargs)
        
        for key, value in defaults.items():
            setattr(mock_stats, key, value)
        
        return mock_stats
    return _create_stats


@pytest.fixture
def mock_player_id_mapping():
    """Factory for creating mock PlayerIDMapping objects."""
    def _create_mapping(
        bdl_id=1,
        yahoo_key="mlb.p.12345",
        yahoo_id="12345",
        mlbam_id=477132,
        name="Test Player",
    ):
        mock_mapping = Mock()
        mock_mapping.bdl_id = bdl_id
        mock_mapping.yahoo_key = yahoo_key
        mock_mapping.yahoo_id = yahoo_id
        mock_mapping.mlbam_id = mlbam_id
        mock_mapping.name = name
        return mock_mapping
    return _create_mapping


# =============================================================================
# External API Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_yahoo_client():
    """Return a mock Yahoo Fantasy client."""
    mock_client = Mock()
    mock_client.get_roster = Mock(return_value=[{
        "player_key": "mlb.p.12345",
        "name": "Test Player",
        "full_name": "Test Player",
        "team": "LAD",
        "positions": ["1B", "OF"],
        "status": "Active",
        "injury_status": None,
        "percent_owned": 85.5,
        "stats": {"7": 25.0, "8": 45.0, "12": 8.0},  # stat_id: value
        "selected_position": "1B",
    }])
    mock_client.get_player_stats = Mock(return_value={
        "7": 25.0,  # Runs
        "8": 45.0,  # Hits
        "12": 8.0,  # HR
    })
    return mock_client


@pytest.fixture
def mock_statcast_data():
    """Return sample Statcast pitcher metrics data."""
    return {
        "mlbam_id": 477132,
        "name": "Clayton Kershaw",
        "team": "LAD",
        "season": 2026,
        "era": 2.45,
        "whip": 0.98,
        "k_9": 9.8,
        "ip": 45.0,
        "handedness": "L",
    }


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_yahoo_player():
    """Return a sample Yahoo player dict."""
    return {
        "player_key": "mlb.p.12345",
        "name": "Test Player",
        "full_name": "Test Player",
        "team": "LAD",
        "positions": ["1B", "OF"],
        "status": "Active",
        "injury_status": None,
        "percent_owned": 85.5,
        "ownership_pct": 85.5,
        "stats": {
            "7": 25.0,   # Runs (yahoo stat_id)
            "8": 45.0,   # Hits
            "12": 8.0,   # HR
            "13": 32.0,  # RBI
        },
        "selected_position": "1B",
    }


@pytest.fixture
def sample_yahoo_pitcher():
    """Return a sample Yahoo pitcher dict."""
    return {
        "player_key": "mlb.p.67890",
        "name": "Ace Pitcher",
        "full_name": "Ace Pitcher",
        "team": "NYY",
        "positions": ["SP"],
        "status": "Active",
        "injury_status": None,
        "percent_owned": 92.0,
        "ownership_pct": 92.0,
        "stats": {
            "39": 85.0,  # Strikeouts
            "46": 2.95,  # ERA
            "47": 1.08,  # WHIP
        },
        "selected_position": "SP",
    }


@pytest.fixture
def sample_rolling_stats_dict():
    """Return sample rolling stats dict for row_projector."""
    return {
        "w_runs": 5.0,
        "w_hits": 8.5,
        "w_home_runs": 1.5,
        "w_rbi": 5.2,
        "w_strikeouts_bat": 7.0,
        "w_tb": 13.5,
        "w_net_stolen_bases": 0.5,
        "w_avg": 0.275,
        "w_ops": 0.850,
        "w_ab": 30.0,
        "w_walks": 3.5,
        "w_strikeouts_pit": 0.0,
        "w_era": 0.0,
        "w_whip": 0.0,
        "w_k_per_9": 0.0,
        "w_qs": 0.0,
        "w_ip": 0.0,
    }


@pytest.fixture
def sample_pitcher_rolling_stats():
    """Return sample pitcher rolling stats dict."""
    return {
        "w_strikeouts_pit": 12.0,
        "w_era": 3.50,
        "w_whip": 1.15,
        "w_k_per_9": 9.5,
        "w_qs": 1.0,
        "w_ip": 6.0,
        "w_earned_runs": 2.5,
        "w_hits_allowed": 5.0,
        "w_walks_allowed": 2.0,
        "w_runs": 0.0,
        "w_hits": 0.0,
        "w_home_runs": 0.0,
        "w_rbi": 0.0,
        "w_strikeouts_bat": 0.0,
        "w_tb": 0.0,
        "w_net_stolen_bases": 0.0,
        "w_avg": 0.0,
        "w_ops": 0.0,
        "w_ab": 0.0,
        "w_walks": 0.0,
    }


@pytest.fixture
def sample_matchup_context():
    """Return a sample MatchupContext for testing."""
    from backend.services.matchup_engine import (
        MatchupContext,
        PitcherStats,
        HitterSplits,
        BullpenStats,
        WeatherData,
    )
    
    return MatchupContext(
        bdl_player_id=1,
        game_date=date(2026, 5, 15),
        opponent_team="NYY",
        home_team="LAD",
        pitcher=PitcherStats(
            name="Gerrit Cole",
            hand="R",
            era=3.12,
            whip=1.05,
            k_per_nine=10.2,
            mlbam_id=543037,
        ),
        splits=HitterSplits(
            woba_vs_hand=0.360,
            woba_overall=0.340,
            k_pct_vs_hand=0.18,
            iso_vs_hand=0.200,
            pa_vs_hand=45,
        ),
        bullpen=BullpenStats(
            era=3.85,
            whip=1.25,
            pitcher_count=8,
        ),
        weather=WeatherData(
            temp_f=75.0,
            wind_mph=8.0,
            wind_direction="out",
            precip_chance=10.0,
        ),
        park_factor_runs=1.05,
        park_factor_hr=1.08,
    )


# =============================================================================
# Configuration Patches
# =============================================================================

@pytest.fixture(autouse=True)
def reset_config_cache():
    """Reset the config service cache before each test."""
    from backend.services import config_service
    config_service.invalidate_cache()
    yield
    config_service.invalidate_cache()


@pytest.fixture
def mock_config_values():
    """Patch config service to return test values."""
    test_values = {
        "matchup.weight.handedness": 0.35,
        "matchup.weight.pitcher": 0.25,
        "matchup.weight.park": 0.15,
        "matchup.weight.weather": 0.10,
        "matchup.weight.bullpen": 0.15,
        "matchup.boost.cap": 0.2,
        "matchup.boost.z_scale": 0.1,
        "matchup.confidence_gate": 0.4,
    }
    
    with patch('backend.services.config_service._threshold_cache', test_values):
        with patch('backend.services.config_service._cache_expiry', float('inf')):
            yield test_values
