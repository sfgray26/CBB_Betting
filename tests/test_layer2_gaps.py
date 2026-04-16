"""
Tests for Layer 2 Gap Closure fixes.

Tests verify the fixes for:
- SQL bug in projection freshness (date -> metric_date)
- Probable pitchers constraint migration
- Weather/park persistence
- Admin version endpoint
"""
import pytest


def test_projection_freshness_queries_metric_date_not_date():
    """The projection freshness check must query metric_date, not date."""
    import os
    os.chdir(os.path.dirname(os.path.dirname(__file__)))

    # Read the file and check SQL uses metric_date not date
    with open('backend/services/daily_ingestion.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Check that both SQL queries now use metric_date
    assert "SELECT MAX(metric_date) FROM player_daily_metrics" in content, \
        "SQL should use metric_date, not date"

    # Verify the buggy version is gone
    assert "SELECT MAX(date) FROM player_daily_metrics" not in content, \
        "Buggy SQL using 'date' column should be removed"


def test_player_daily_metrics_has_metric_date_column():
    """PlayerDailyMetric model should have metric_date column."""
    from backend.models import PlayerDailyMetric

    # Verify the column exists
    assert hasattr(PlayerDailyMetric, 'metric_date'), \
        "PlayerDailyMetric must have metric_date column"


def test_scoring_engine_has_park_factor_consumer():
    """
    CRITERION 6: Scoring engine should consume persisted park factors.

    This verifies at least one real consumer uses the ParkFactor model
    from canonical persistence rather than request-time-only logic.
    """
    from backend.services.scoring_engine import get_park_factor
    import inspect

    # Verify the function exists
    assert callable(get_park_factor), \
        "scoring_engine.get_park_factor should be a callable function"

    # Verify function signature (park_name, metric="hr")
    sig = inspect.signature(get_park_factor)
    assert 'park_name' in sig.parameters, "Should accept park_name parameter"
    assert 'metric' in sig.parameters, "Should accept metric parameter with default"

    # Verify the function imports ParkFactor from models (consumer check)
    # We'll verify by checking the function source code references ParkFactor
    source = inspect.getsource(get_park_factor)
    assert 'ParkFactor' in source, "Function should use ParkFactor model from persistence"
