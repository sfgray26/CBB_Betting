"""
Tests for Streaming Recommendations API endpoint.

Tests for GET /api/fantasy/streaming/recommendations.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import date, datetime


@pytest.fixture
def fantasy_client():
    with patch("backend.schedulers.fantasy_scheduler.start_fantasy_scheduler"):
        with patch("backend.schedulers.fantasy_scheduler.stop_fantasy_scheduler"):
            from backend.fantasy_app import app
            from backend.models import get_db as get_db_dependency
            from fastapi.testclient import TestClient

            mock_db = MagicMock()

            def override_get_db():
                try:
                    yield mock_db
                finally:
                    pass

            app.dependency_overrides[get_db_dependency] = override_get_db
            with TestClient(app) as client:
                yield client
            app.dependency_overrides = {}


class TestStreamingRecommendationsEndpoint:
    """Tests for GET /api/fantasy/streaming/recommendations endpoint."""

    def test_streaming_recommendations_returns_two_start_pitchers(self, fantasy_client):
        """Endpoint should return pitchers with 2+ starts and quality ratings."""
        from zoneinfo import ZoneInfo
        from backend.models import get_db as get_db_dependency

        mock_rows = [
            MagicMock(
                bdl_player_id=12345,
                pitcher_name="Gerrit Cole",
                team="NYY",
                handedness="R",
                game_date=date(2026, 6, 24),
                opponent="BOS",
                is_home=True,
                quality_score=1.2,
                is_confirmed=True,
                game_time_et="7:05 PM",
            ),
            MagicMock(
                bdl_player_id=12345,
                pitcher_name="Gerrit Cole",
                team="NYY",
                handedness="R",
                game_date=date(2026, 6, 29),
                opponent="BAL",
                is_home=False,
                quality_score=0.8,
                is_confirmed=True,
                game_time_et="1:05 PM",
            ),
        ]

        # Mock SQLAlchemy query chain
        mock_base_query = MagicMock()
        mock_filter = MagicMock()
        mock_order = MagicMock()

        mock_base_query.filter.return_value = mock_filter
        mock_filter.order_by.return_value = mock_order
        mock_order.all.return_value = mock_rows

        mock_freshness = MagicMock()
        mock_freshness.scalar.return_value = datetime.now(ZoneInfo("America/New_York"))

        mock_db = MagicMock()
        mock_db.query.side_effect = [mock_base_query, mock_freshness]

        fantasy_client.app.dependency_overrides[get_db_dependency] = lambda: mock_db

        response = fantasy_client.get("/api/fantasy/streaming/recommendations?target_date=2026-06-24&days_ahead=7")

        assert response.status_code == 200
        data = response.json()

        assert "two_start_pitchers" in data
        assert len(data["two_start_pitchers"]) == 1

        pitcher = data["two_start_pitchers"][0]
        assert pitcher["bdl_player_id"] == 12345
        assert pitcher["name"] == "Gerrit Cole"
        assert pitcher["team"] == "NYY"
        assert pitcher["handedness"] == "R"
        assert len(pitcher["starts"]) == 2

        assert pitcher["starts"][0]["opponent"] == "BOS"
        assert pitcher["starts"][0]["is_home"] is True
        assert pitcher["starts"][0]["quality_score"] == 1.2

        assert pitcher["starts"][1]["opponent"] == "BAL"
        assert pitcher["starts"][1]["is_home"] is False
        assert pitcher["starts"][1]["quality_score"] == 0.8

        # Check overall quality calculation (average of 1.2 and 0.8)
        assert pitcher["overall_quality"] == 1.0

        # Check recommendation tier (1.0 -> EXCELLENT)
        assert pitcher["recommendation"] == "EXCELLENT"

        # Check transparency fields
        assert "transparency" in pitcher
        assert pitcher["transparency"]["quality_score"] == 1.0
        assert pitcher["transparency"]["confidence"] == "HIGH"  # Both confirmed
        assert "starts_count: 2" in pitcher["transparency"]["factors"]
        assert "avg_quality: 1.00" in pitcher["transparency"]["factors"]

        # Check freshness
        assert "freshness" in data
        assert data["freshness"]["staleness_ms"] < 100  # Fresh data (<100ms)
        assert data["data_sources"] == ["ProbablePitcherSnapshot", "StatcastPerformances (quality_score)"]

    def test_streaming_recommendations_handles_edge_cases_gracefully(self, fantasy_client):
        """Endpoint should handle no 2-start pitchers or only 1-start pitchers gracefully."""
        from backend.models import get_db as get_db_dependency

        # Edge case 1: No pitchers at all
        mock_base_query = MagicMock()
        mock_filter = MagicMock()
        mock_order = MagicMock()

        mock_base_query.filter.return_value = mock_filter
        mock_filter.order_by.return_value = mock_order
        mock_order.all.return_value = []

        # Mock freshness query (second db.query call)
        mock_freshness_query = MagicMock()
        mock_freshness_filter = MagicMock()
        mock_freshness_query.filter.return_value = mock_freshness_filter
        mock_freshness_filter.scalar.return_value = None

        mock_db = MagicMock()
        mock_db.query.side_effect = [mock_base_query, mock_freshness_query]

        fantasy_client.app.dependency_overrides[get_db_dependency] = lambda: mock_db

        response = fantasy_client.get("/api/fantasy/streaming/recommendations?target_date=2026-06-24&days_ahead=7")

        assert response.status_code == 200
        data = response.json()

        assert data["two_start_pitchers"] == []
        assert data["freshness"]["last_refresh_at"] is None
        assert data["freshness"]["staleness_ms"] is None

        # Edge case 2: Only 1-start pitchers (should return empty 2-start list)
        mock_row_single = MagicMock(
            bdl_player_id=67890,
            pitcher_name="Single Starter",
            team="LAD",
            handedness="L",
            game_date=date(2026, 6, 24),
            opponent="SF",
            is_home=True,
            quality_score=0.5,
            is_confirmed=True,
            game_time_et="10:10 PM",
        )

        mock_query_single = MagicMock()
        mock_filter_single = MagicMock()
        mock_order_single = MagicMock()

        mock_query_single.filter.return_value = mock_filter_single
        mock_filter_single.order_by.return_value = mock_order_single
        mock_order_single.all.return_value = [mock_row_single]

        mock_db.query.side_effect = [mock_query_single, mock_freshness_query]

        response = fantasy_client.get("/api/fantasy/streaming/recommendations?target_date=2026-06-24&days_ahead=7")

        assert response.status_code == 200
        data = response.json()

        # Should have no 2-start pitchers
        assert data["two_start_pitchers"] == []

    def test_streaming_recommendations_validates_date_format(self, fantasy_client):
        """Endpoint should reject invalid date formats."""
        response = fantasy_client.get("/api/fantasy/streaming/recommendations?target_date=invalid-date")

        assert response.status_code == 400
        assert "Invalid date format" in response.json()["detail"]
