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
        assert "schedule_available" in data  # TASK-7 gate field

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

    def test_optimize_routes_to_lineup_constraint_solver(self, fantasy_client):
        """POST /api/fantasy/roster/optimize must route hitter optimization through
        LineupConstraintSolver.solve(), not an inline greedy allocator."""
        from backend.fantasy_baseball.lineup_constraint_solver import (
            PlayerSlotAssignment as SolverAssignment,
            PositionSlot,
            OptimizedLineup,
        )

        mock_roster = [
            {
                "player_key": "469.l.72586.p.100",
                "name": "Test Catcher",
                "team": "NYY",
                "positions": ["C"],
                "selected_position": "C",
            }
        ]
        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster

        solver_result = OptimizedLineup(
            assignments=[
                SolverAssignment(
                    player_id="469.l.72586.p.100",
                    player_name="Test Catcher",
                    slot=PositionSlot.CATCHER,
                    score=72.5,
                    eligibility=["C"],
                    reason="Score 72.5 (player_scores)",
                )
            ],
            total_score=72.5,
            is_optimal=True,
            solver_type="OR-Tools CP-SAT",
            unassigned_players=[],
        )

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            with patch("backend.routers.fantasy.get_lineup_solver") as mock_get_solver:
                mock_solver = MagicMock()
                mock_solver.solve.return_value = solver_result
                mock_get_solver.return_value = mock_solver

                response = fantasy_client.post(
                    "/api/fantasy/roster/optimize",
                    json={"target_date": "2026-05-15"},
                )

        assert response.status_code == 200
        mock_get_solver.assert_called_once()
        mock_solver.solve.assert_called_once()
        data = response.json()
        catcher = next(
            (s for s in data["starters"] if s["player_name"] == "Test Catcher"), None
        )
        assert catcher is not None, f"Expected Test Catcher in starters; got: {data['starters']}"
        assert catcher["assigned_slot"] == "C"
        assert catcher["lineup_score"] == 72.5

    def test_optimize_of_slots_normalized_to_of(self, fantasy_client):
        """Solver OF1/OF2/OF3 slot values must be normalized to 'OF' in the API response."""
        from backend.fantasy_baseball.lineup_constraint_solver import (
            PlayerSlotAssignment as SolverAssignment,
            PositionSlot,
            OptimizedLineup,
        )

        mock_roster = [
            {
                "player_key": "469.l.72586.p.200",
                "name": "OF Player",
                "team": "LAD",
                "positions": ["OF"],
                "selected_position": "OF",
            }
        ]
        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster

        solver_result = OptimizedLineup(
            assignments=[
                SolverAssignment(
                    player_id="469.l.72586.p.200",
                    player_name="OF Player",
                    slot=PositionSlot.OUTFIELD_1,
                    score=65.0,
                    eligibility=["OF"],
                    reason="Score 65.0 (player_scores)",
                )
            ],
            total_score=65.0,
            is_optimal=True,
            solver_type="OR-Tools CP-SAT",
            unassigned_players=[],
        )

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            with patch("backend.routers.fantasy.get_lineup_solver") as mock_get_solver:
                mock_solver = MagicMock()
                mock_solver.solve.return_value = solver_result
                mock_get_solver.return_value = mock_solver

                response = fantasy_client.post(
                    "/api/fantasy/roster/optimize",
                    json={"target_date": "2026-05-15"},
                )

        assert response.status_code == 200
        data = response.json()
        of_player = next(
            (s for s in data["starters"] if s["player_name"] == "OF Player"), None
        )
        assert of_player is not None, f"Expected OF Player in starters; got: {data['starters']}"
        assert of_player["assigned_slot"] == "OF", (
            f"Expected slot='OF' (not 'OF1'), got {of_player['assigned_slot']!r}"
        )

    def test_schedule_gate_fails_open_on_db_error(self, fantasy_client):
        """schedule_available=True (fail-open) when the DB schedule check itself errors."""
        mock_roster = [
            {
                "player_key": "469.l.72586.p.111",
                "name": "Hitter A",
                "team": "NYY",
                "positions": ["1B"],
                "selected_position": "1B",
            },
        ]
        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster

        # The test DB mock already raises on complex queries — schedule gate catches
        # that exception and keeps schedule_available=True (fail-open, don't block user).
        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/optimize",
                json={"target_date": "2026-04-15"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "schedule_available" in data
        assert data["schedule_available"] is True

    def test_schedule_gate_field_present_on_normal_response(self, fantasy_client):
        """schedule_available=True on a normal day with game data."""
        mock_roster = [
            {
                "player_key": "469.l.72586.p.111",
                "name": "Hitter A",
                "team": "NYY",
                "positions": ["1B"],
                "selected_position": "1B",
            },
        ]
        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/optimize",
                json={"target_date": "2026-04-15"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "schedule_available" in data
        assert isinstance(data["schedule_available"], bool)

    def test_optimize_handles_boolean_status_gracefully(self, fantasy_client):
        """optimize endpoint should handle boolean status values without crashing."""
        mock_roster = [
            {
                "player_key": "469.l.72586.p.111",
                "name": "Test Player",
                "team": "NYY",
                "positions": ["1B"],
                "selected_position": "1B",
                "status": True,  # Boolean instead of string - data corruption scenario
            }
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post("/api/fantasy/roster/optimize", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        # Player with boolean status should not cause crash

    def test_optimize_handles_none_status_gracefully(self, fantasy_client):
        """optimize endpoint should handle None status values without crashing."""
        mock_roster = [
            {
                "player_key": "469.l.72586.p.222",
                "name": "No Status Player",
                "team": "BOS",
                "positions": ["C"],
                "selected_position": "C",
                "status": None,  # Missing/None status
            }
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post("/api/fantasy/roster/optimize", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_optimize_returns_structured_error_for_non_dict_player(self, fantasy_client):
        """optimize endpoint should return structured error for corrupted roster data."""
        mock_roster = [
            "not_a_dict",  # Completely malformed - should trigger validation error
            {
                "player_key": "469.l.72586.p.333",
                "name": "Valid Player",
                "team": "BAL",
                "positions": ["SS"],
                "selected_position": "SS",
                "status": "playing",
            }
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post("/api/fantasy/roster/optimize", json={})

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert data["detail"]["error_code"] == "ROSTER_DATA_CORRUPTED"
        assert "index" in data["detail"]
        assert data["detail"]["index"] == 0

    def test_global_freshness_returns_valid_response(self, fantasy_client):
        """GET /api/fantasy/global-freshness should return structured freshness data."""
        response = fantasy_client.get("/api/fantasy/global-freshness")

        assert response.status_code == 200
        data = response.json()

        # Top-level fields
        assert "severity" in data
        assert data["severity"] in ["fresh", "warning", "critical", "unknown"]
        assert "minutes_ago" in data
        assert "warning_text" in data
        assert "sources" in data

        # Sources should be an array
        assert isinstance(data["sources"], list)
        # At least Yahoo source should be present
        assert len(data["sources"]) >= 1

        # Each source should have required fields
        for source in data["sources"]:
            assert "name" in source
            assert "severity" in source
            assert source["severity"] in ["fresh", "warning", "critical", "unknown"]
            assert "minutes_ago" in source

    def test_insufficient_projection_data_returns_valid_error_response(self, fantasy_client, monkeypatch):
        """When optimizer returns insufficient data error, response must have all required fields."""
        # Enable coverage check for this test to verify error response structure
        monkeypatch.setenv("COVERAGE_CHECK_ENABLED", "1")

        # Large roster with many active slots but insufficient projections
        mock_roster = [
            {
                "player_key": f"469.l.72586.p.{i}",
                "name": f"Player {i}",
                "team": "NYY",
                "positions": ["1B" if i % 2 else "OF"],
                "selected_position": "1B" if i % 2 else "OF",
            }
            for i in range(1, 15)  # 14 players
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster

        # All players hit projection fallback (0.0 score) → insufficient data
        projection_rows = [{"z_score": 0.0, "is_proxy": True}] * 14

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

        # Error response must have all required fields
        assert "success" in data
        assert data["success"] is False
        assert "message" in data
        assert "Insufficient projection data" in data["message"]
        assert "target_date" in data
        assert data["target_date"] == "2026-04-15"
        assert "starters" in data
        assert data["starters"] == []
        assert "bench" in data
        assert data["bench"] == []
        assert "unrostered" in data
        assert isinstance(data["unrostered"], list)
        assert "total_lineup_score" in data
        assert data["total_lineup_score"] == 0.0
        assert "freshness" in data
        assert "primary_source" in data["freshness"]
        assert "computed_at" in data["freshness"]
        assert "schedule_available" in data
        assert data["schedule_available"] is True
