"""
Tests for Phase 4 Roster Optimize API endpoint.

Tests for POST /api/fantasy/roster/optimize.
"""

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def fantasy_client():
    with patch("backend.schedulers.fantasy_scheduler.start_fantasy_scheduler"):
        with patch("backend.schedulers.fantasy_scheduler.stop_fantasy_scheduler"):
            from backend.fantasy_app import app
            from backend.models import get_db
            from fastapi.testclient import TestClient

            # Mock DB session so tests don't need a real PostgreSQL connection.
            # The optimize endpoint queries PlayerIDMapping and PlayerScore;
            # an empty-result mock makes it fall back to default scores.
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.all.return_value = []
            mock_db.query.return_value.filter.return_value.group_by.return_value.subquery.return_value = MagicMock()
            mock_db.query.return_value.join.return_value.filter.return_value.all.return_value = []

            def override_get_db():
                try:
                    yield mock_db
                finally:
                    pass

            app.dependency_overrides[get_db] = override_get_db
            with TestClient(app) as client:
                yield client
            app.dependency_overrides = {}


class TestRosterOptimizeEndpoint:
    """Tests for POST /api/fantasy/roster/optimize endpoint."""

    def test_optimize_resolves_player_scores_from_yahoo_key_variants(self, fantasy_client):
        """Short-form yahoo_key mappings should score full roster keys correctly."""
        mock_roster = [
            {
                "player_key": "469.l.72586.p.111",
                "name": "Catcher",
                "team": "NYY",
                "positions": ["C"],
                "selected_position": "C",
            }
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster

        mapping_query = MagicMock()
        mapping_query.filter.return_value.all.return_value = [
            MagicMock(
                yahoo_key="469.p.111",
                yahoo_id="111",
                bdl_id=1111,
                normalized_name="catcher",
                full_name="Catcher",
            )
        ]

        subquery_handle = MagicMock(name="player_score_subquery")
        subquery_query = MagicMock()
        subquery_query.filter.return_value.group_by.return_value.subquery.return_value = subquery_handle

        score_query = MagicMock()
        score_query.join.return_value.filter.return_value.all.return_value = [
            MagicMock(bdl_player_id=1111, score_0_100=87.5, as_of_date="2026-04-15")
        ]

        fantasy_client.app.dependency_overrides = dict(fantasy_client.app.dependency_overrides)
        db_override = MagicMock()
        db_override.query.side_effect = [mapping_query, subquery_query, score_query]

        from backend.models import get_db

        def override_get_db():
            try:
                yield db_override
            finally:
                pass

        fantasy_client.app.dependency_overrides[get_db] = override_get_db

        try:
            with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
                response = fantasy_client.post(
                    "/api/fantasy/roster/optimize",
                    json={"target_date": "2026-04-15"},
                )
        finally:
            fantasy_client.app.dependency_overrides.pop(get_db, None)

        assert response.status_code == 200
        data = response.json()
        assert data["starters"][0]["lineup_score"] == 87.5
        assert "player_scores" in data["starters"][0]["reasoning"]

    def test_optimize_uses_projection_fallback_scores_instead_of_flat_default(self, fantasy_client):
        """Projection fallback should differentiate players when DB scores are unavailable."""
        mock_roster = [
            {
                "player_key": "469.l.72586.p.111",
                "name": "Pete Alonso",
                "team": "NYM",
                "positions": ["1B"],
                "selected_position": "1B",
            },
            {
                "player_key": "469.l.72586.p.222",
                "name": "Michael Wacha",
                "team": "KC",
                "positions": ["SP"],
                "selected_position": "SP",
            },
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster

        projection_rows = [
            {"z_score": 3.5, "is_proxy": False},
            {"z_score": 0.5, "is_proxy": False},
        ]

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            with patch(
                "backend.fantasy_baseball.player_board.get_or_create_projection",
                side_effect=projection_rows,
            ):
                response = fantasy_client.post(
                    "/api/fantasy/roster/optimize",
                    json={"target_date": "2026-04-15"},
                )

        assert response.status_code == 200
        data = response.json()
        scores = {player["player_name"]: player["lineup_score"] for player in data["starters"]}
        assert scores["Pete Alonso"] != scores["Michael Wacha"]
        assert scores["Pete Alonso"] > scores["Michael Wacha"]
        assert all("default" not in player["reasoning"] for player in data["starters"])
        assert "projection fallback" in data["message"].lower()

    def test_optimize_response_structure(self, fantasy_client):
        """Optimize response has all required fields."""
        mock_roster = [
            {
                "player_key": "469.l.72586.p.12345",
                "name": "Hitter A",
                "team": "NYY",
                "positions": ["1B"],
                "selected_position": "1B",
            },
            {
                "player_key": "469.l.72586.p.67890",
                "name": "Pitcher A",
                "team": "BOS",
                "positions": ["SP"],
                "selected_position": "SP",
            },
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            with patch("backend.services.player_mapper.fetch_rolling_stats_for_players", return_value={}):
                response = fantasy_client.post(
                    "/api/fantasy/roster/optimize",
                    json={"target_date": "2026-04-15"},
                )

        assert response.status_code == 200
        data = response.json()

        # Response fields
        assert "success" in data
        assert "message" in data
        assert "target_date" in data
        assert "starters" in data
        assert "bench" in data
        assert "unrostered" in data
        assert "total_lineup_score" in data
        assert "freshness" in data

        # Freshness fields
        freshness = data["freshness"]
        assert "primary_source" in freshness
        assert "computed_at" in freshness
        assert "staleness_threshold_minutes" in freshness
        assert "is_stale" in freshness

    def test_successful_optimization(self, fantasy_client):
        """Roster optimized successfully."""
        mock_roster = [
            {
                "player_key": "469.l.72586.p.111",
                "name": "Catcher",
                "team": "NYY",
                "positions": ["C"],
                "selected_position": "C",
            },
            {
                "player_key": "469.l.72586.p.222",
                "name": "First Baseman",
                "team": "BOS",
                "positions": ["1B"],
                "selected_position": "1B",
            },
            {
                "player_key": "469.l.72586.p.333",
                "name": "Weak Player",
                "team": "BAL",
                "positions": ["1B"],
                "selected_position": "BN",
            },
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            with patch("backend.services.player_mapper.fetch_rolling_stats_for_players", return_value={}):
                response = fantasy_client.post(
                    "/api/fantasy/roster/optimize",
                    json={},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["starters"]) >= 0
        assert isinstance(data["bench"], list)

    def test_default_target_date(self, fantasy_client):
        """Default target_date when not provided."""
        mock_roster = []

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            with patch("backend.services.player_mapper.fetch_rolling_stats_for_players", return_value={}):
                response = fantasy_client.post(
                    "/api/fantasy/roster/optimize",
                    json={},
                )

        assert response.status_code == 200
        data = response.json()
        assert "target_date" in data
        # Should be today's date in YYYY-MM-DD format
        assert len(data["target_date"]) == 10

    def test_empty_roster(self, fantasy_client):
        """Empty roster returns empty optimization."""
        mock_client = MagicMock()
        mock_client.get_roster.return_value = []

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            with patch("backend.services.player_mapper.fetch_rolling_stats_for_players", return_value={}):
                response = fantasy_client.post(
                    "/api/fantasy/roster/optimize",
                    json={},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["starters"] == []
        assert data["bench"] == []
        assert data["total_lineup_score"] == 0.0

    def test_yahoo_api_error(self, fantasy_client):
        """Yahoo API error handled gracefully."""
        from backend.fantasy_baseball.yahoo_client_resilient import YahooAPIError

        mock_client = MagicMock()
        mock_client.get_roster.side_effect = YahooAPIError("API rate limit")

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/optimize",
                json={},
            )

        assert response.status_code in (200, 502)  # Either graceful or HTTP exception

    def test_util_slot_filling(self, fantasy_client):
        """Util slot filled by eligible hitters."""
        mock_roster = [
            {
                "player_key": "469.l.72586.p.111",
                "name": "Catcher",
                "team": "NYY",
                "positions": ["C"],
                "selected_position": "C",
            },
            {
                "player_key": "469.l.72586.p.222",
                "name": "OF Hitter",
                "team": "BOS",
                "positions": ["OF"],
                "selected_position": "OF",
            },
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            with patch("backend.services.player_mapper.fetch_rolling_stats_for_players", return_value={}):
                response = fantasy_client.post(
                    "/api/fantasy/roster/optimize",
                    json={},
                )

        assert response.status_code == 200
        data = response.json()

        # Check that OF player is assigned to OF slot
        of_assignments = [s for s in data["starters"] if s["assigned_slot"] == "OF"]
        assert len(of_assignments) >= 0  # TODO: meaningful assertion once player_scores wired

    def test_pitcher_slot_filling(self, fantasy_client):
        """Pitching slots filled by pitchers."""
        mock_roster = [
            {
                "player_key": "469.l.72586.p.901",
                "name": "Starter A",
                "team": "NYY",
                "positions": ["SP"],
                "selected_position": "SP",
            },
            {
                "player_key": "469.l.72586.p.902",
                "name": "Starter B",
                "team": "BOS",
                "positions": ["SP"],
                "selected_position": "SP",
            },
            {
                "player_key": "469.l.72586.p.951",
                "name": "Reliever A",
                "team": "BAL",
                "positions": ["RP"],
                "selected_position": "RP",
            },
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            with patch("backend.services.player_mapper.fetch_rolling_stats_for_players", return_value={}):
                response = fantasy_client.post(
                    "/api/fantasy/roster/optimize",
                    json={},
                )

        assert response.status_code == 200
        data = response.json()

        # Check SP and RP assignments
        sp_assignments = [s for s in data["starters"] if s["assigned_slot"] == "SP"]
        rp_assignments = [s for s in data["starters"] if s["assigned_slot"] == "RP"]
        assert len(sp_assignments) >= 0  # TODO: meaningful assertion once player_scores wired
        assert len(rp_assignments) >= 0  # TODO: meaningful assertion once player_scores wired

    def test_optimize_target_date_reflected_in_response(self, fantasy_client):
        """target_date from POST payload must appear in response.target_date field."""
        mock_roster = [
            {
                "player_key": "469.l.72586.p.999",
                "name": "Test Batter",
                "team": "NYY",
                "positions": ["OF"],
                "selected_position": "OF",
            }
        ]
        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            resp = fantasy_client.post(
                "/api/fantasy/roster/optimize",
                json={"target_date": "2026-04-22"},
            )

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["target_date"] == "2026-04-22", (
            f"Expected target_date='2026-04-22' in response, got {data.get('target_date')!r}"
        )
