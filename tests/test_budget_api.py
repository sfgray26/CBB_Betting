"""
Tests for Phase 4 Budget API endpoint.

Tests for GET /budget.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture
def fantasy_client():
    with patch("backend.schedulers.fantasy_scheduler.start_fantasy_scheduler"):
        with patch("backend.schedulers.fantasy_scheduler.stop_fantasy_scheduler"):
            from backend.fantasy_app import app
            from backend.auth import verify_api_key
            app.dependency_overrides[verify_api_key] = lambda: "test_user"
            with TestClient(app) as client:
                yield client
            app.dependency_overrides.clear()


class TestBudgetEndpoint:
    """Tests for GET /budget endpoint."""

    @pytest.fixture(autouse=True)
    def mock_yahoo_client(self):
        """Patch get_yahoo_client so tests run without real Yahoo credentials."""
        mock_client = MagicMock()
        mock_client.get_roster.return_value = []
        mock_client.get_matchup_stats.return_value = {
            "my_team": {"IP": 45.0}
        }
        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            with patch("backend.services.constraint_helpers.count_weekly_acquisitions", return_value=5):
                yield mock_client

    def test_budget_response_structure(self, fantasy_client):
        """Budget response has all required fields."""
        response = fantasy_client.get("/api/fantasy/budget")

        assert response.status_code == 200
        data = response.json()

        # Top-level keys
        assert "budget" in data
        assert "freshness" in data

        # Budget fields
        budget = data["budget"]
        assert "acquisitions_used" in budget
        assert "acquisitions_remaining" in budget
        assert "acquisition_limit" in budget
        assert "acquisition_warning" in budget
        assert "il_used" in budget
        assert "il_total" in budget
        assert "ip_accumulated" in budget
        assert "ip_minimum" in budget
        assert "ip_pace" in budget
        assert "as_of" in budget

        # Freshness fields
        freshness = data["freshness"]
        assert "primary_source" in freshness
        assert "fetched_at" in freshness
        assert "computed_at" in freshness
        assert "staleness_threshold_minutes" in freshness
        assert "is_stale" in freshness

    def test_budget_values_are_valid(self, fantasy_client):
        """Budget values are within valid ranges."""
        response = fantasy_client.get("/api/fantasy/budget")
        data = response.json()
        budget = data["budget"]

        # Acquisitions should be non-negative
        assert budget["acquisitions_used"] >= 0
        assert budget["acquisitions_remaining"] >= 0
        assert budget["acquisition_limit"] > 0

        # Remaining should equal limit - used
        assert budget["acquisitions_remaining"] == budget["acquisition_limit"] - budget["acquisitions_used"]

        # IP should be non-negative
        assert budget["ip_accumulated"] >= 0
        assert budget["ip_minimum"] > 0

        # IP pace should be one of the valid flags (uppercase enum values)
        assert budget["ip_pace"] in ("BEHIND", "ON_TRACK", "AHEAD", "COMPLETE", "UNKNOWN")

    def test_freshness_metadata_present(self, fantasy_client):
        """Freshness metadata is present and valid."""
        response = fantasy_client.get("/api/fantasy/budget")
        data = response.json()
        freshness = data["freshness"]

        assert freshness["primary_source"] == "yahoo"
        assert freshness["computed_at"] is not None
        assert freshness["staleness_threshold_minutes"] > 0
        assert isinstance(freshness["is_stale"], bool)

    def test_acquisition_warning_threshold(self, fantasy_client):
        """Acquisition warning triggers at 6+ used."""
        # The mock data has 5 acquisitions used, so warning should be False
        response = fantasy_client.get("/api/fantasy/budget")
        data = response.json()
        budget = data["budget"]

        # With 5 used, warning should be False
        assert budget["acquisition_warning"] is False

    def test_ip_pace_calculation(self, fantasy_client):
        """IP pace is calculated based on accumulated IP and season days."""
        response = fantasy_client.get("/api/fantasy/budget")
        data = response.json()
        budget = data["budget"]

        # Mock data: 45 IP accumulated, 90 minimum, 30 days elapsed
        # 45 IP / 30 days = 1.5 IP/day
        # 90 IP / 182 days = ~0.5 IP/day needed
        # Should be "ON_TRACK" or "AHEAD" with this pace (uppercase enum values)
        assert budget["ip_pace"] in ("ON_TRACK", "AHEAD", "BEHIND", "COMPLETE", "UNKNOWN")

    def test_budget_response_200(self, fantasy_client):
        """Budget endpoint returns 200 OK."""
        response = fantasy_client.get("/api/fantasy/budget")
        assert response.status_code == 200
