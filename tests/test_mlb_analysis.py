"""
tests/test_mlb_analysis.py — EMAC-080

10 unit tests for MLBAnalysisService and SportConfig.mlb().
No real API calls or DB connections required.
"""

from __future__ import annotations

import asyncio
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from backend.core.sport_config import SportConfig, SPORT_ID_MLB
from backend.services.mlb_analysis import (
    LEAGUE_AVG_ERA,
    MLBAnalysisService,
    MLBGameProjection,
)


# ---------------------------------------------------------------------------
# Helper: minimal game dict in statsapi format
# ---------------------------------------------------------------------------

def _make_game(
    game_id: int = 1001,
    home_name: str = "New York Yankees",
    away_name: str = "Boston Red Sox",
    home_pitcher: str = "gerrit cole",
    away_pitcher: str = "chris sale",
    venue: str = "Yankee Stadium",
) -> dict:
    return {
        "game_id": game_id,
        "home_name": home_name,
        "away_name": away_name,
        "home_probable_pitcher": home_pitcher,
        "away_probable_pitcher": away_pitcher,
        "venue_name": venue,
    }


# ---------------------------------------------------------------------------
# SportConfig.mlb() tests
# ---------------------------------------------------------------------------

class TestSportConfigMlb:
    def test_sport_config_mlb_returns_correct_sport_id(self):
        cfg = SportConfig.mlb()
        assert cfg.sport_id == "mlb"
        assert cfg.sport_id == SPORT_ID_MLB
        assert cfg.sport_name == "MLB"
        assert cfg.odds_api_sport_key == "baseball_mlb"

    def test_sport_config_mlb_home_advantage_is_small(self):
        cfg = SportConfig.mlb()
        # MLB home field is ~0.25 runs — much smaller than CBB (3.09 pts)
        assert cfg.home_advantage_pts == 0.25
        # Verify it is less than CBB home advantage
        cbb = SportConfig.ncaa_basketball()
        assert cfg.home_advantage_pts < cbb.home_advantage_pts

    def test_sport_config_mlb_sd_multiplier_in_range(self):
        cfg = SportConfig.mlb()
        # 0.86 derived from 2.5/sqrt(8.5); verify it is near that value
        assert 0.80 <= cfg.base_sd_multiplier <= 0.92
        # League avg total and per-team run averages
        assert cfg.d1_avg_pace == 8.5
        assert cfg.d1_avg_adj_o == 4.25
        assert cfg.d1_avg_adj_de == 4.25
        # Basketball four-factor fields are zeroed out
        assert cfg.d1_avg_efg == 0.0
        assert cfg.d1_avg_to_pct == 0.0
        assert cfg.pace_rho == 0.20


# ---------------------------------------------------------------------------
# MLBAnalysisService._project_game() tests
# ---------------------------------------------------------------------------

class TestProjectGame:
    def setup_method(self):
        self.svc = MLBAnalysisService()
        self.target_date = date(2025, 7, 1)

    def test_project_game_uses_park_factor(self):
        """Coors Field (1.22) must produce higher totals than a neutral park."""
        game_coors = _make_game(venue="Coors Field")
        game_neutral = _make_game(venue="SomeRandomPark")  # defaults to 1.0
        proj_coors = self.svc._project_game(game_coors, {}, {}, self.target_date)
        proj_neutral = self.svc._project_game(game_neutral, {}, {}, self.target_date)
        assert proj_coors.projected_total > proj_neutral.projected_total

    def test_project_game_applies_home_advantage(self):
        """Home team projected runs must exceed away team runs when all else equal."""
        game = _make_game(venue="Unknown")
        proj = self.svc._project_game(game, {}, {}, self.target_date)
        # Home advantage of +0.25 means home runs > away runs in equal matchup
        assert proj.projected_home_runs > proj.projected_away_runs
        assert proj.projected_runline_margin > 0

    def test_project_game_falls_back_to_league_avg_when_no_pitcher_data(self):
        """With empty pitcher_stats, both pitch factors should equal 1.0 (league avg)."""
        game = _make_game()
        proj = self.svc._project_game(game, {}, {}, self.target_date)
        # league_avg * 1.0 (off) * 1.0 (pitch) * park_factor + hfa
        # Yankee Stadium park_factor = 1.05
        expected_away = round(4.25 * 1.0 * 1.0 * 1.05, 2)
        expected_home = round(4.25 * 1.0 * 1.0 * 1.05 + 0.25, 2)
        assert abs(proj.projected_away_runs - expected_away) < 0.01
        assert abs(proj.projected_home_runs - expected_home) < 0.01

    def test_project_game_coors_field_inflates_total(self):
        """Coors Field (1.22) should push the projected total above 10 runs
        for an average matchup (league avg offense, league avg pitching)."""
        game = _make_game(
            home_name="Colorado Rockies",
            away_name="Los Angeles Dodgers",
            venue="Coors Field",
        )
        proj = self.svc._project_game(game, {}, {}, self.target_date)
        # league_avg 4.25 * 1.22 * 2 sides + home_advantage ~= 10.37
        assert proj.projected_total > 10.0

    def test_project_game_better_pitcher_lowers_runs(self):
        """A pitcher with xERA = 2.00 should allow fewer runs than league avg."""
        game = _make_game(
            home_pitcher="ace pitcher",
            away_pitcher="ace pitcher",
            venue="Unknown",
        )
        # Excellent home pitcher (xERA 2.00 vs league 4.25)
        pitcher_stats = {"ace pitcher": 2.00}
        proj = self.svc._project_game(game, pitcher_stats, {}, self.target_date)
        # Away pitcher (ace, low xERA) -> away_pitch_factor = 2.0/4.25 < 1.0
        # -> home team scores fewer runs than league avg
        assert proj.projected_home_runs < 4.25

    def test_project_game_uses_team_wrc_plus(self):
        """Team with above-average offense (wRC+=130) should project more runs than league avg."""
        game = _make_game(home_name="New York Yankees", away_name="Boston Red Sox", venue="Unknown")
        team_stats = {
            "New York Yankees": {"wrc_plus": 130},
            "Boston Red Sox": {"wrc_plus": 100},
        }
        proj = self.svc._project_game(game, {}, team_stats, self.target_date)
        # Home team has 130 wRC+ — should score more than league avg + home advantage
        assert proj.projected_home_runs > 4.25 + 0.25

    def test_load_team_stats_returns_dict(self):
        """_load_team_stats should return a dict (may be empty if pybaseball not available)."""
        svc = MLBAnalysisService()
        result = svc._load_team_stats()
        assert isinstance(result, dict)
        # Each value must be a dict with 'wrc_plus' key if non-empty
        for team, stats in result.items():
            assert "wrc_plus" in stats
            assert isinstance(stats["wrc_plus"], (int, float))


# ---------------------------------------------------------------------------
# MLBAnalysisService.run_analysis() tests
# ---------------------------------------------------------------------------

class TestRunAnalysis:
    def test_run_analysis_returns_empty_for_no_schedule(self):
        """When _fetch_schedule returns [], run_analysis returns []."""
        svc = MLBAnalysisService()
        with patch.object(svc, "_fetch_schedule", return_value=[]):
            result = asyncio.run(svc.run_analysis(target_date=date(2025, 7, 1)))
        assert result == []

    def test_run_analysis_handles_statsapi_failure_gracefully(self):
        """When _fetch_schedule raises, run_analysis returns [] without crashing."""
        svc = MLBAnalysisService()
        with patch.object(svc, "_fetch_schedule", side_effect=RuntimeError("network down")):
            result = asyncio.run(svc.run_analysis(target_date=date(2025, 7, 1)))
        assert result == []

    def test_run_analysis_returns_projection_list_for_valid_games(self):
        """When schedule has games, run_analysis returns one projection per game."""
        svc = MLBAnalysisService()
        fake_games = [_make_game(game_id=i) for i in range(3)]
        with patch.object(svc, "_fetch_schedule", return_value=fake_games):
            with patch.object(svc, "_fetch_mlb_odds", return_value={}):
                with patch.object(svc, "_load_pitcher_stats", return_value={}):
                    with patch.object(svc, "_load_team_stats", return_value={}):
                        result = asyncio.run(svc.run_analysis(target_date=date(2025, 7, 1)))
        assert len(result) == 3
        for proj in result:
            assert isinstance(proj, MLBGameProjection)
            assert proj.model_version == "v1.0-mlb"


# ---------------------------------------------------------------------------
# Edge calculation test
# ---------------------------------------------------------------------------

class TestCalculateEdge:
    def test_calculate_edge_returns_zero_when_no_market(self):
        """_calculate_edge returns 0.0 when market dict is empty."""
        svc = MLBAnalysisService()
        proj = MLBGameProjection(
            game_id="test",
            home_team="Yankees",
            away_team="Red Sox",
            game_date=date(2025, 7, 1),
            projected_home_runs=4.75,
            projected_away_runs=4.25,
            projected_total=9.0,
            projected_runline_margin=0.5,
            home_win_prob=0.55,
        )
        assert svc._calculate_edge(proj, {}) == 0.0
        assert svc._calculate_edge(proj, None) == 0.0

    def test_calculate_edge_with_negative_ml_odds(self):
        """_calculate_edge correctly converts negative American odds to implied prob."""
        svc = MLBAnalysisService()
        proj = MLBGameProjection(
            game_id="test",
            home_team="Yankees",
            away_team="Red Sox",
            game_date=date(2025, 7, 1),
            projected_home_runs=4.75,
            projected_away_runs=4.25,
            projected_total=9.0,
            projected_runline_margin=0.5,
            home_win_prob=0.55,
        )
        # -130 American odds -> implied prob = 130 / (130 + 100) = 0.5652
        market = {"ml_home_odds": -130}
        # Edge = 0.55 - 0.5652 = -0.0152, rounded to 4 decimals
        result = svc._calculate_edge(proj, market)
        assert result == round(0.55 - (130 / (130 + 100)), 4)
        assert result == -0.0152

    def test_calculate_edge_with_positive_ml_odds(self):
        """_calculate_edge correctly converts positive American odds to implied prob."""
        svc = MLBAnalysisService()
        proj = MLBGameProjection(
            game_id="test",
            home_team="Yankees",
            away_team="Red Sox",
            game_date=date(2025, 7, 1),
            projected_home_runs=4.75,
            projected_away_runs=4.25,
            projected_total=9.0,
            projected_runline_margin=0.5,
            home_win_prob=0.55,
        )
        # +110 American odds -> implied prob = 100 / (110 + 100) = 0.4762
        market = {"ml_home_odds": 110}
        # Edge = 0.55 - 0.4762 = 0.0738, rounded to 4 decimals
        result = svc._calculate_edge(proj, market)
        assert result == round(0.55 - (100 / (110 + 100)), 4)
        assert result == 0.0738


# ---------------------------------------------------------------------------
# Odds fetch tests (DB tier)
# ---------------------------------------------------------------------------


class TestFetchMlbOdds:
    def test_fetch_mlb_odds_returns_empty_when_snapshot_empty(self):
        """_fetch_mlb_odds returns {} when mlb_odds_snapshot table is empty."""
        svc = MLBAnalysisService()
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.order_by.return_value.all.return_value = []

        from unittest.mock import patch
        with patch("backend.models.SessionLocal", return_value=mock_session):
            result = svc._fetch_mlb_odds()

        assert result == {}
        mock_session.close.assert_called_once()

    def test_fetch_mlb_odds_returns_keyed_dict_with_fresh_data(self):
        """_fetch_mlb_odds returns dict keyed by AwayAbbr@HomeAbbr with fresh snapshot data."""
        svc = MLBAnalysisService()

        mock_session = MagicMock()

        mock_home_team_1 = MagicMock()
        mock_home_team_1.abbreviation = "NYY"

        mock_home_team_2 = MagicMock()
        mock_home_team_2.abbreviation = "HOU"

        mock_game_1 = MagicMock()
        mock_game_1.home_team_obj = mock_home_team_1

        mock_game_2 = MagicMock()
        mock_game_2.home_team_obj = mock_home_team_2

        mock_away_team_1 = MagicMock()
        mock_away_team_1.abbreviation = "BOS"

        mock_away_team_2 = MagicMock()
        mock_away_team_2.abbreviation = "LAA"

        mock_odds_1 = MagicMock()
        mock_odds_1.game_id = 1001
        mock_odds_1.ml_home_odds = -130
        mock_odds_1.ml_away_odds = 110
        mock_odds_1.spread_home = "1.5"
        mock_odds_1.spread_home_odds = -110
        mock_odds_1.total = "8.5"
        mock_odds_1.vendor = "draftkings"

        mock_odds_2 = MagicMock()
        mock_odds_2.game_id = 1002
        mock_odds_2.ml_home_odds = -145
        mock_odds_2.ml_away_odds = 125
        mock_odds_2.spread_home = "-1.5"
        mock_odds_2.spread_home_odds = -105
        mock_odds_2.total = "9.0"
        mock_odds_2.vendor = "pinnacle"

        mock_query_result = [
            (mock_odds_1, mock_game_1, mock_away_team_1, 1),
            (mock_odds_2, mock_game_2, mock_away_team_2, 0),
        ]

        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.order_by.return_value.all.return_value = mock_query_result

        from unittest.mock import patch
        with patch("backend.models.SessionLocal", return_value=mock_session):
            result = svc._fetch_mlb_odds()

        assert len(result) == 2
        assert "BOS@NYY" in result
        assert "LAA@HOU" in result
        assert result["BOS@NYY"]["ml_home_odds"] == -130
        assert result["BOS@NYY"]["vendor"] == "draftkings"
        assert result["LAA@HOU"]["ml_home_odds"] == -145
        assert result["LAA@HOU"]["vendor"] == "pinnacle"  # preferred vendor wins
        mock_session.close.assert_called_once()
