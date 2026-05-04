"""Contract tests for /admin/pipeline-health response shape.

Production probe 2026-04-22 found overall=null after a tables dict->list refactor.
The pipeline_health_summary() function returned overall_healthy (new) but no longer
populated the legacy overall key. Fix: populate overall as an alias of overall_healthy.
"""

from unittest.mock import MagicMock, patch
from datetime import date


def _make_healthy_check(name: str):
    from backend.services.pipeline_validator import TableHealth
    return TableHealth(
        table_name=name,
        row_count=5000,
        expected_min_rows=1000,
        latest_date=date(2026, 4, 22),
        max_staleness_days=2,
        is_healthy=True,
        issues=[],
    )


def _make_unhealthy_check(name: str):
    from backend.services.pipeline_validator import TableHealth
    return TableHealth(
        table_name=name,
        row_count=0,
        expected_min_rows=1000,
        latest_date=None,
        max_staleness_days=2,
        is_healthy=False,
        issues=["No rows"],
    )


def test_pipeline_health_summary_overall_is_bool_when_all_healthy():
    from backend.services.pipeline_validator import pipeline_health_summary
    checks = [_make_healthy_check("player_rolling_stats"), _make_healthy_check("player_scores")]
    result = pipeline_health_summary(checks)
    assert "overall" in result, "pipeline_health_summary must include 'overall' key"
    assert isinstance(result["overall"], bool), f"overall must be bool, got {type(result['overall'])}"
    assert result["overall"] is True


def test_pipeline_health_summary_overall_matches_overall_healthy():
    from backend.services.pipeline_validator import pipeline_health_summary
    checks = [_make_healthy_check("a"), _make_unhealthy_check("b")]
    result = pipeline_health_summary(checks)
    assert result["overall"] == result["overall_healthy"], (
        "overall and overall_healthy must be identical"
    )


def test_pipeline_health_summary_overall_false_when_unhealthy():
    from backend.services.pipeline_validator import pipeline_health_summary
    checks = [_make_unhealthy_check("player_rolling_stats")]
    result = pipeline_health_summary(checks)
    assert result["overall"] is False


def test_pipeline_health_summary_tables_is_list():
    from backend.services.pipeline_validator import pipeline_health_summary
    checks = [_make_healthy_check("x")]
    result = pipeline_health_summary(checks)
    assert isinstance(result["tables"], list), "tables must be a list"


def test_pipeline_health_summary_table_entries_have_name_key():
    from backend.services.pipeline_validator import pipeline_health_summary
    checks = [_make_healthy_check("player_rolling_stats")]
    result = pipeline_health_summary(checks)
    assert result["tables"][0]["name"] == "player_rolling_stats"
