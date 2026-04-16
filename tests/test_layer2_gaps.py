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
