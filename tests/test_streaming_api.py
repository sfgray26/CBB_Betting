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
                source="official",
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
                source="official",
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

        # Check recommendation tier (1.0 + HIGH confidence -> EXCELLENT)
        assert pitcher["recommendation"] == "EXCELLENT"

        # Check risk_note (both confirmed -> safe stream)
        assert "risk_note" in pitcher
        assert pitcher["risk_note"] == "Both starts confirmed — safe stream"

        # Check transparency fields
        assert "transparency" in pitcher
        assert pitcher["transparency"]["quality_score"] == 1.0
        assert pitcher["transparency"]["confidence"] == "HIGH"  # Both confirmed
        assert "starts_count: 2" in pitcher["transparency"]["factors"]
        assert "avg_quality: 1.00" in pitcher["transparency"]["factors"]

        # Check freshness
        assert "freshness" in data
        assert data["freshness"]["staleness_ms"] < 100  # Fresh data (<100ms)
        assert data["data_sources"] == ["MLB probable & projected starters", "Park-adjusted ERA quality score"]

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

    def test_streaming_recommendations_confidence_weighted_matrix(self, fantasy_client):
        """Endpoint should apply confidence-weighted recommendation matrix."""
        from backend.models import get_db as get_db_dependency

        def run_test_with_rows(mock_rows, expected_quality, expected_recommendation, expected_confidence, expected_risk_note):
            """Helper to run a test case with specific mock rows."""
            mock_base_query = MagicMock()
            mock_filter = MagicMock()
            mock_order = MagicMock()
            mock_base_query.filter.return_value = mock_filter
            mock_filter.order_by.return_value = mock_order
            mock_order.all.return_value = mock_rows

            mock_freshness = MagicMock()
            mock_freshness.scalar.return_value = datetime.now()

            mock_db = MagicMock()
            mock_db.query.side_effect = [mock_base_query, mock_freshness]

            fantasy_client.app.dependency_overrides[get_db_dependency] = lambda: mock_db

            response = fantasy_client.get("/api/fantasy/streaming/recommendations?target_date=2026-06-24&days_ahead=7")
            assert response.status_code == 200
            data = response.json()
            pitcher = data["two_start_pitchers"][0]

            assert pitcher["overall_quality"] == expected_quality
            assert pitcher["recommendation"] == expected_recommendation
            assert pitcher["transparency"]["confidence"] == expected_confidence
            assert pitcher["risk_note"] == expected_risk_note
            return pitcher

        # Test Case 1: HIGH quality (1.0) + MEDIUM confidence -> GOOD (not EXCELLENT)
        mock_rows_medium_conf = [
            MagicMock(
                bdl_player_id=11111,
                pitcher_name="Medium Conf Ace",
                team="NYY",
                handedness="R",
                game_date=date(2026, 6, 24),
                opponent="BOS",
                is_home=True,
                quality_score=1.2,
                is_confirmed=True,  # 1 confirmed
                source="official",
                game_time_et="7:05 PM",
            ),
            MagicMock(
                bdl_player_id=11111,
                pitcher_name="Medium Conf Ace",
                team="NYY",
                handedness="R",
                game_date=date(2026, 6, 29),
                opponent="BAL",
                is_home=False,
                quality_score=0.8,
                is_confirmed=False,  # 1 projected
                source="projected",
                game_time_et="1:05 PM",
            ),
        ]
        run_test_with_rows(
            mock_rows_medium_conf,
            expected_quality=1.0,
            expected_recommendation="GOOD",
            expected_confidence="MEDIUM",
            expected_risk_note="One start projected — monitor for scratches"
        )

        # Test Case 2: both starts rotation-projected -> PROJECTED tier (not the
        # old LOW->AVOID that hid every projected 2-start pitcher — §4.5).
        mock_rows_low_conf = [
            MagicMock(
                bdl_player_id=22222,
                pitcher_name="Low Conf Quality",
                team="LAD",
                handedness="L",
                game_date=date(2026, 6, 24),
                opponent="SF",
                is_home=True,
                quality_score=1.5,  # High quality
                is_confirmed=False,  # Both projected
                source="projected",
                game_time_et="10:10 PM",
            ),
            MagicMock(
                bdl_player_id=22222,
                pitcher_name="Low Conf Quality",
                team="LAD",
                handedness="L",
                game_date=date(2026, 6, 29),
                opponent="SD",
                is_home=False,
                quality_score=1.0,
                is_confirmed=False,
                source="projected",
                game_time_et="4:15 PM",
            ),
        ]
        run_test_with_rows(
            mock_rows_low_conf,
            expected_quality=1.25,
            expected_recommendation="PROJECTED",
            expected_confidence="PROJECTED",
            expected_risk_note="Both starts projected — high variance, have backup ready"
        )

        # Test Case 3: Negative quality but HIGH confidence -> AVOID (quality < -0.3)
        mock_rows_negative = [
            MagicMock(
                bdl_player_id=33333,
                pitcher_name="Struggling Ace",
                team="CHC",
                handedness="R",
                game_date=date(2026, 6, 24),
                opponent="MIL",
                is_home=True,
                quality_score=-1.0,
                is_confirmed=True,
                source="official",
                game_time_et="2:20 PM",
            ),
            MagicMock(
                bdl_player_id=33333,
                pitcher_name="Struggling Ace",
                team="CHC",
                handedness="R",
                game_date=date(2026, 6, 29),
                opponent="PIT",
                is_home=False,
                quality_score=-0.8,
                is_confirmed=True,
                source="official",
                game_time_et="7:05 PM",
            ),
        ]
        run_test_with_rows(
            mock_rows_negative,
            expected_quality=-0.9,
            expected_recommendation="AVOID",
            expected_confidence="HIGH",
            expected_risk_note="Both starts confirmed — safe stream"
        )

    def test_streaming_recommendations_validates_date_format(self, fantasy_client):
        """Endpoint should reject invalid date formats."""
        response = fantasy_client.get("/api/fantasy/streaming/recommendations?target_date=invalid-date")

        assert response.status_code == 400
        assert "Invalid date format" in response.json()["detail"]
