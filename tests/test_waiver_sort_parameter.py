"""Tests for the /api/fantasy/waiver `sort` query parameter.

These lock in the V2 fix (triage §V2 / §9): the "Overall Value" toggle
(``sort=overall_value`` / ``sort=projected_points``) must reorder candidates by
the season-long composite ``z_score`` instead of being a no-op that always fell
through to ``need_score``.

Contract notes (verified against backend/routers/fantasy.py + schemas.py):
- The endpoint returns a ``WaiverWireResponse`` whose player list is
  ``top_available`` (NOT ``available_players``).
- ``WaiverPlayerOut.z_score`` is recomputed from the projection board
  (``get_or_create_projection``), so it is controlled here by patching that
  function — the raw Yahoo free-agent dict's fields do NOT survive.
- ``WaiverPlayerOut.percent_owned`` passes through from the free-agent dict, so
  the ``percent_owned`` sort is controlled directly via the mock input.
- ``need_score`` is recomputed from live category deficits and is not
  deterministic under a MagicMock Yahoo client, so the default-sort test asserts
  only the ordering invariant, not specific values.
"""
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def fantasy_client():
    """TestClient fixture for the FastAPI fantasy router."""
    with patch("backend.schedulers.fantasy_scheduler.start_fantasy_scheduler"):
        with patch("backend.schedulers.fantasy_scheduler.stop_fantasy_scheduler"):
            from backend.fantasy_app import app
            from backend.auth import verify_api_key
            from fastapi.testclient import TestClient
            app.dependency_overrides[verify_api_key] = lambda: "test_user"
            try:
                with TestClient(app) as client:
                    yield client
            finally:
                app.dependency_overrides.pop(verify_api_key, None)


def _mock_yahoo_client(free_agents):
    client = MagicMock()
    client.get_free_agents.return_value = free_agents
    client.get_my_team_key.return_value = "469.l.72586.t.7"
    return client


def _fake_projection_by_name(z_by_name):
    """Return a get_or_create_projection replacement that yields a controlled
    z_score per player name and otherwise-empty projection scaffolding."""
    def _fake(p):
        name = (p.get("name") or "").strip()
        return {
            "z_score": z_by_name.get(name, 0.0),
            "cat_scores": {},
            "proj": {},
            "is_proxy": True,
        }
    return _fake


class TestWaiverSortParameter:
    """The `sort` query parameter must route to distinct sort keys."""

    def test_overall_value_orders_by_zscore(self, fantasy_client):
        """sort=overall_value must order top_available by z_score descending.

        This is the exact bug from triage §V2: the toggle used to be a no-op.
        z_score is the real "overall value" metric; projected_points is vestigial.
        """
        free_agents = [
            {"player_key": "1.p.1", "name": "Player A", "positions": ["OF"]},
            {"player_key": "1.p.2", "name": "Player B", "positions": ["OF"]},
            {"player_key": "1.p.3", "name": "Player C", "positions": ["OF"]},
        ]
        # Intentionally NOT in list order, so a no-op sort would be detectable.
        z_by_name = {"Player A": 1.0, "Player B": 3.0, "Player C": 2.0}

        with patch(
            "backend.routers.fantasy.get_yahoo_client",
            return_value=_mock_yahoo_client(free_agents),
        ), patch(
            "backend.fantasy_baseball.player_board.get_or_create_projection",
            side_effect=_fake_projection_by_name(z_by_name),
        ):
            response = fantasy_client.get("/api/fantasy/waiver?sort=overall_value")

        assert response.status_code == 200
        players = response.json()["top_available"]
        assert len(players) == 3
        z_scores = [p["z_score"] for p in players]
        assert z_scores == sorted(z_scores, reverse=True), z_scores
        # Expected order: B(3.0) > C(2.0) > A(1.0)
        assert [p["name"] for p in players] == ["Player B", "Player C", "Player A"]

    def test_projected_points_alias_orders_by_zscore(self, fantasy_client):
        """sort=projected_points is an alias of overall_value → z_score order."""
        free_agents = [
            {"player_key": "1.p.1", "name": "Player A", "positions": ["OF"]},
            {"player_key": "1.p.2", "name": "Player B", "positions": ["OF"]},
            {"player_key": "1.p.3", "name": "Player C", "positions": ["OF"]},
        ]
        z_by_name = {"Player A": 2.5, "Player B": -1.0, "Player C": 0.5}

        with patch(
            "backend.routers.fantasy.get_yahoo_client",
            return_value=_mock_yahoo_client(free_agents),
        ), patch(
            "backend.fantasy_baseball.player_board.get_or_create_projection",
            side_effect=_fake_projection_by_name(z_by_name),
        ):
            response = fantasy_client.get("/api/fantasy/waiver?sort=projected_points")

        assert response.status_code == 200
        players = response.json()["top_available"]
        assert [p["name"] for p in players] == ["Player A", "Player C", "Player B"]

    def test_percent_owned_orders_by_ownership(self, fantasy_client):
        """sort=percent_owned must order by percent_owned descending.

        percent_owned passes through from the free-agent dict, so it is set
        directly here.
        """
        free_agents = [
            {"player_key": "1.p.1", "name": "Player A", "positions": ["OF"], "percent_owned": 10.0},
            {"player_key": "1.p.2", "name": "Player B", "positions": ["OF"], "percent_owned": 90.0},
            {"player_key": "1.p.3", "name": "Player C", "positions": ["OF"], "percent_owned": 50.0},
        ]

        with patch(
            "backend.routers.fantasy.get_yahoo_client",
            return_value=_mock_yahoo_client(free_agents),
        ), patch(
            "backend.fantasy_baseball.player_board.get_or_create_projection",
            side_effect=_fake_projection_by_name({}),
        ):
            response = fantasy_client.get("/api/fantasy/waiver?sort=percent_owned")

        assert response.status_code == 200
        players = response.json()["top_available"]
        owned = [p["percent_owned"] for p in players]
        assert owned == sorted(owned, reverse=True), owned
        assert [p["name"] for p in players] == ["Player B", "Player C", "Player A"]

    def test_default_sort_returns_ordered_response(self, fantasy_client):
        """Default sort (need_score) returns a valid response ordered
        non-increasing by need_score.

        need_score is recomputed from live category deficits, so we assert the
        ordering invariant guaranteed by the endpoint rather than exact values.
        """
        free_agents = [
            {"player_key": "1.p.1", "name": "Player A", "positions": ["OF"]},
            {"player_key": "1.p.2", "name": "Player B", "positions": ["OF"]},
            {"player_key": "1.p.3", "name": "Player C", "positions": ["OF"]},
        ]

        with patch(
            "backend.routers.fantasy.get_yahoo_client",
            return_value=_mock_yahoo_client(free_agents),
        ), patch(
            "backend.fantasy_baseball.player_board.get_or_create_projection",
            side_effect=_fake_projection_by_name({}),
        ):
            response = fantasy_client.get("/api/fantasy/waiver")

        assert response.status_code == 200
        players = response.json()["top_available"]
        assert len(players) == 3
        need_scores = [p["need_score"] for p in players]
        assert need_scores == sorted(need_scores, reverse=True), need_scores
