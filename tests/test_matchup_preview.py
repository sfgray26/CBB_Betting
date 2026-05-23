"""
Tests for GET /api/fantasy/matchup-preview endpoint.

Wave 4 — Weekly Matchup Preview Backend
"""

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def fantasy_client():
    with patch("backend.schedulers.fantasy_scheduler.start_fantasy_scheduler"):
        with patch("backend.schedulers.fantasy_scheduler.stop_fantasy_scheduler"):
            from backend.fantasy_app import app
            from fastapi.testclient import TestClient
            with TestClient(app) as client:
                yield client


_MOCK_ROSTER = [
    {
        "player_key": "469.l.72586.p.11111",
        "name": "Slugger One",
        "team": "NYY",
        "positions": ["1B", "Util"],
        "eligible_positions": ["1B", "Util"],
        "selected_position": "1B",
    },
    {
        "player_key": "469.l.72586.p.22222",
        "name": "Ace Pitcher",
        "team": "LAD",
        "positions": ["SP"],
        "eligible_positions": ["SP"],
        "selected_position": "SP",
    },
]

_MOCK_SIM_RESULT = {
    "win_prob": 0.62,
    "category_win_probs": {
        "r": 0.70,
        "hr_b": 0.65,
        "rbi": 0.55,   # bubble
        "avg": 0.35,   # loss → weak_category
        "era": 0.30,   # loss → weak_category
        "k_p": 0.72,
    },
    "expected_cats_won": 9.4,
    "n_sims": 2000,
    "elapsed_ms": 42.0,
    "categories_simulated": ["r", "hr_b", "rbi", "avg", "era", "k_p"],
    "data_quality": "ok",
    "my_projection_coverage": 0.85,
    "opp_projection_coverage": 0.0,
}


class TestMatchupPreviewEndpoint:

    def _mock_client(self):
        mock = MagicMock()
        mock.get_league.return_value = {"current_week": 8}
        mock.get_matchup_stats.return_value = {
            "opponent_name": "Team Rocket",
            "my_stats": {},
            "opp_stats": {},
        }
        mock.get_roster.return_value = _MOCK_ROSTER
        mock.get_my_team_key.return_value = "469.l.72586.t.7"
        mock.get_scoreboard.return_value = []
        return mock

    def test_response_structure(self, fantasy_client):
        """Response contains all required fields matching frontend MatchupPreviewResponse."""
        mock_client = self._mock_client()
        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client), \
             patch("backend.fantasy_baseball.mcmc_simulator.simulate_weekly_matchup",
                   return_value=_MOCK_SIM_RESULT):
            response = fantasy_client.get("/api/fantasy/matchup-preview")

        assert response.status_code == 200
        data = response.json()

        # Field names must match frontend types.ts MatchupPreviewResponse
        assert "opponent_name" in data
        assert "week_number" in data
        assert "overall_win_prob" in data
        assert "category_projections" in data
        assert "weak_categories" in data
        assert "schedule_advantage" in data

        assert isinstance(data["category_projections"], list)
        assert isinstance(data["weak_categories"], list)
        assert "my_games" in data["schedule_advantage"]
        assert "opponent_games" in data["schedule_advantage"]

    def test_category_projection_fields(self, fantasy_client):
        """CategoryProjection has win_prob, my_proj, opp_proj matching frontend CategoryProjection."""
        mock_client = self._mock_client()
        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client), \
             patch("backend.fantasy_baseball.mcmc_simulator.simulate_weekly_matchup",
                   return_value=_MOCK_SIM_RESULT):
            response = fantasy_client.get("/api/fantasy/matchup-preview")

        data = response.json()
        proj = data["category_projections"][0]
        assert "category" in proj
        assert "win_prob" in proj    # not my_win_prob — must match frontend type
        assert "my_proj" in proj
        assert "opp_proj" in proj

    def test_weak_categories_for_loss_categories(self, fantasy_client):
        """weak_categories generated for categories with win_prob < 0.4."""
        mock_client = self._mock_client()
        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client), \
             patch("backend.fantasy_baseball.mcmc_simulator.simulate_weekly_matchup",
                   return_value=_MOCK_SIM_RESULT):
            response = fantasy_client.get("/api/fantasy/matchup-preview")

        data = response.json()
        weak_cats = {w["category"] for w in data["weak_categories"]}

        # avg (0.35) and era (0.30) are < 0.4 → must appear
        assert "avg" in weak_cats
        assert "era" in weak_cats
        # wins and bubbles must NOT appear
        assert "r" not in weak_cats
        assert "rbi" not in weak_cats

    def test_next_week_fallback_when_unavailable(self, fantasy_client):
        """Falls back to current-week opponent when next week isn't published."""
        mock_client = self._mock_client()

        def _side_effect(week=None):
            if week is not None and week > 8:
                raise Exception("Yahoo: matchup not published yet")
            return {"opponent_name": "Current Foe", "my_stats": {}, "opp_stats": {}}

        mock_client.get_matchup_stats.side_effect = _side_effect

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client), \
             patch("backend.fantasy_baseball.mcmc_simulator.simulate_weekly_matchup",
                   return_value=_MOCK_SIM_RESULT):
            response = fantasy_client.get("/api/fantasy/matchup-preview")

        assert response.status_code == 200
        data = response.json()
        assert data["opponent_name"] == "Current Foe"
        assert data["week_number"] == 8  # fell back to current week
