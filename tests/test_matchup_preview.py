"""
Tests for GET /api/fantasy/matchup-preview endpoint.

Wave 4 — Weekly Matchup Preview Backend
"""

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def fantasy_client():
    # Patch at source module matching backend/fantasy_app.py line 27 import
    with patch("backend.schedulers.fantasy_scheduler.start_fantasy_scheduler"):
        with patch("backend.schedulers.fantasy_scheduler.stop_fantasy_scheduler"):
            from backend.fantasy_app import app
            from fastapi.testclient import TestClient
            with TestClient(app) as client:
                yield client


_MOCK_YAHOO_ROSTER = [
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

# Mock rosters with cat_scores matching simulate_weekly_matchup expectations.
# Each player dict needs: name, positions, cat_scores, starts_this_week.
_MOCK_SIM_ROSTERS = (
    [  # my_roster
        {
            "name": "Slugger One",
            "positions": ["1B", "Util"],
            "cat_scores": {"r": 1.5, "hr_b": 0.8, "rbi": 1.2, "avg": 0.3},
            "starts_this_week": 0,
        },
        {
            "name": "Ace Pitcher",
            "positions": ["SP"],
            "cat_scores": {"w": 0.5, "era": -0.3, "whip": -0.2, "k_p": 1.8},
            "starts_this_week": 1,
        },
    ],
    [],  # opponent_roster (empty = league-average baseline)
)

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
        mock.get_roster.return_value = _MOCK_YAHOO_ROSTER
        mock.get_my_team_key.return_value = "469.l.72586.t.7"
        # Minimal valid scoreboard so category projection tests reach the simulator.
        # Structure matches Yahoo API: matchup dict with teams keyed by "0"/"1".
        mock.get_scoreboard.return_value = [
            {
                "teams": {
                    "count": 2,
                    "0": {"team": [
                        [{"team_key": "469.l.72586.t.7"}, {"name": "My Team"}],
                        {"team_stats": {"stats": []}},
                    ]},
                    "1": {"team": [
                        [{"team_key": "469.l.72586.t.3"}, {"name": "Team Rocket"}],
                        {"team_stats": {"stats": []}},
                    ]},
                }
            }
        ]
        return mock

    def test_response_structure(self, fantasy_client):
        """Response contains all required fields matching frontend MatchupPreviewResponse."""
        mock_client = self._mock_client()
        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client), \
             patch("backend.routers.fantasy._fetch_rosters_for_simulate",
                   return_value=_MOCK_SIM_ROSTERS), \
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
        # schedule_advantage is now nullable — hidden (None) when a real
        # two-sided games-scheduled comparison can't be computed (triage §P3).
        # When present it must carry both team counts.
        if data["schedule_advantage"] is not None:
            assert "my_games" in data["schedule_advantage"]
            assert "opponent_games" in data["schedule_advantage"]

    def test_category_projection_fields(self, fantasy_client):
        """CategoryProjection has category and win_prob fields."""
        mock_client = self._mock_client()
        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client), \
             patch("backend.routers.fantasy._fetch_rosters_for_simulate",
                   return_value=_MOCK_SIM_ROSTERS), \
             patch("backend.fantasy_baseball.mcmc_simulator.simulate_weekly_matchup",
                   return_value=_MOCK_SIM_RESULT):
            response = fantasy_client.get("/api/fantasy/matchup-preview")

        data = response.json()
        proj = data["category_projections"][0]
        assert "category" in proj
        assert "win_prob" in proj    # not my_win_prob — must match frontend type

    def test_weak_categories_for_loss_categories(self, fantasy_client):
        """weak_categories generated for categories with win_prob < 0.4."""
        mock_client = self._mock_client()
        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client), \
             patch("backend.routers.fantasy._fetch_rosters_for_simulate",
                   return_value=_MOCK_SIM_ROSTERS), \
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
        """Falls back to current-week opponent when next week isn't published.

        The matchup-preview endpoint resolves opponent via get_scoreboard() +
        _iter_scoreboard_matchup_teams() (not get_matchup_stats).  Week 9 returns
        an empty scoreboard (not yet published); week 8 returns a matchup with the
        known team key so the opponent name can be extracted.
        """
        mock_client = self._mock_client()

        # Build a minimal Yahoo scoreboard payload that _iter_scoreboard_matchup_teams
        # can parse: a list of matchup dicts, each with a "teams" dict keyed "0"/"1".
        # team_key "469.l.72586.t.7" matches mock_client.get_my_team_key().
        _week8_scoreboard = [
            {
                "teams": {
                    "count": 2,
                    "0": {"team": [
                        [{"team_key": "469.l.72586.t.7"}, {"name": "My Team"}],
                        {"team_stats": {"stats": []}},
                    ]},
                    "1": {"team": [
                        [{"team_key": "469.l.72586.t.3"}, {"name": "Current Foe"}],
                        {"team_stats": {"stats": []}},
                    ]},
                }
            }
        ]

        def _sb_side_effect(week=None):
            if week is not None and week > 8:
                return []  # week 9 not published yet
            return _week8_scoreboard

        mock_client.get_scoreboard.side_effect = _sb_side_effect

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client), \
             patch("backend.routers.fantasy._fetch_rosters_for_simulate",
                   return_value=_MOCK_SIM_ROSTERS), \
             patch("backend.fantasy_baseball.mcmc_simulator.simulate_weekly_matchup",
                   return_value=_MOCK_SIM_RESULT):
            response = fantasy_client.get("/api/fantasy/matchup-preview")

        assert response.status_code == 200
        data = response.json()
        assert data["opponent_name"] == "Current Foe"
        assert data["week_number"] == 8  # fell back to current week

    def test_opponent_resolved_when_get_my_team_key_fails(self, fantasy_client):
        """Opponent is still found when get_my_team_key() throws — env default used.

        Regression for the production 'Unknown opponent' bug where get_my_team_key()
        threw silently and left the matching key as an empty string, causing every
        scoreboard entry to be skipped.
        """
        mock_client = self._mock_client()
        mock_client.get_my_team_key.side_effect = Exception("Yahoo API timeout")

        _current_scoreboard = [
            {
                "teams": {
                    "count": 2,
                    "0": {"team": [
                        [{"team_key": "469.l.72586.t.7"}, {"name": "My Team"}],
                        {"team_stats": {"stats": []}},
                    ]},
                    "1": {"team": [
                        [{"team_key": "469.l.72586.t.5"}, {"name": "Week Foe"}],
                        {"team_stats": {"stats": []}},
                    ]},
                }
            }
        ]
        mock_client.get_scoreboard.return_value = _current_scoreboard

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client), \
             patch("backend.routers.fantasy._fetch_rosters_for_simulate",
                   return_value=_MOCK_SIM_ROSTERS), \
             patch("backend.fantasy_baseball.mcmc_simulator.simulate_weekly_matchup",
                   return_value=_MOCK_SIM_RESULT), \
             patch.dict("os.environ", {"YAHOO_TEAM_KEY": "469.l.72586.t.7"}):
            response = fantasy_client.get("/api/fantasy/matchup-preview")

        assert response.status_code == 200
        data = response.json()
        # Must resolve from scoreboard using env fallback — not "Unknown"
        assert data["opponent_name"] == "Week Foe"
