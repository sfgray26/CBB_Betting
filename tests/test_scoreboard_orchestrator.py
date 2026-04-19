"""
Tests for Phase 4 Scoreboard Orchestrator.

Tests for assemble_matchup_scoreboard() and the /api/fantasy/scoreboard endpoint.
"""

import pytest

from backend.services.scoreboard_orchestrator import (
    assemble_matchup_scoreboard,
    build_scoreboard_rows,
    compute_budget_state,
    _map_status_tag,
    _format_delta_to_flip,
    _build_ip_context,
    _project_row_from_player_scores,
)
from backend.services.row_projector import ROWProjectionResult
from backend.stat_contract import SCORING_CATEGORY_CODES


class TestStatusTagMapping:
    """Tests for _map_status_tag()."""

    def test_locked_win(self):
        assert _map_status_tag(0.95).value == "locked_win"

    def test_locked_loss(self):
        assert _map_status_tag(0.05).value == "locked_loss"

    def test_leaning_win(self):
        assert _map_status_tag(0.70).value == "leaning_win"

    def test_leaning_loss(self):
        assert _map_status_tag(0.30).value == "leaning_loss"

    def test_bubble(self):
        assert _map_status_tag(0.50).value == "bubble"
        assert _map_status_tag(0.60).value == "bubble"
        assert _map_status_tag(0.40).value == "bubble"


class TestDeltaToFlipFormatting:
    """Tests for _format_delta_to_flip()."""

    def test_tied(self):
        assert _format_delta_to_flip("R", 0, 0, True) == "Tied"

    def test_leading_batting(self):
        assert _format_delta_to_flip("HR_B", 5, 0, True) == "Leading by 5"

    def test_needing_batting(self):
        assert _format_delta_to_flip("HR_B", -5, 3, True) == "Need +3 HR"

    def test_leading_pitching(self):
        assert _format_delta_to_flip("W", 2, 0, False) == "Leading by 2"

    def test_needing_pitching(self):
        assert _format_delta_to_flip("W", -2, 1, False) == "Need +1 W"


class TestIPContext:
    """Tests for _build_ip_context()."""

    def test_ip_minimum_met(self):
        result = _build_ip_context(95.0, 90.0)
        assert result == "IP minimum met (95.0/90)"

    def test_ip_remaining(self):
        result = _build_ip_context(45.0, 90.0)
        assert result == "45.0/90 IP, 45.0 remaining"


class TestBudgetState:
    """Tests for compute_budget_state()."""

    def test_full_budget(self):
        budget = compute_budget_state(
            acquisitions_used=2,
            acquisition_limit=8,
            il_used=0,
            il_total=3,
            ip_accumulated=50.0,
            ip_minimum=90.0,
            days_remaining=7,
        )

        assert budget.acquisitions_used == 2
        assert budget.acquisitions_remaining == 6
        assert budget.acquisition_limit == 8
        assert budget.acquisition_warning is False
        assert budget.il_used == 0
        assert budget.il_total == 3
        assert budget.ip_accumulated == 50.0
        assert budget.ip_minimum == 90.0

    def test_acquisition_warning(self):
        budget = compute_budget_state(
            acquisitions_used=6,
            acquisition_limit=8,
        )

        assert budget.acquisition_warning is True

    def test_critical_pace(self):
        budget = compute_budget_state(
            acquisitions_used=0,
            ip_accumulated=30.0,
            ip_minimum=90.0,
            days_remaining=2,
        )

        # With only 30 IP and 2 days left, pace should be BEHIND
        assert "behind" in budget.ip_pace.value


class TestProjectRowFromPlayerScores:
    """Tests for _project_row_from_player_scores()."""

    def test_empty_player_scores(self):
        result = _project_row_from_player_scores([])

        assert isinstance(result, ROWProjectionResult)
        # All categories should be 0.0 for empty input
        for cat in SCORING_CATEGORY_CODES:
            assert getattr(result, cat) == 0.0

    def test_basic_player_projection(self):
        player_scores = [
            {
                "yahoo_player_key": "123.p.456",
                "rolling_14d": {
                    "w_runs": 50.0,
                    "w_hits": 120.0,
                    "w_home_runs": 15.0,
                    "w_rbi": 50.0,
                    "w_strikeouts_bat": 70.0,
                    "w_tb": 180.0,
                    "w_net_stolen_bases": 4.0,
                    "w_ab": 450.0,
                },
                "runs": 100,
                "hits": 250,
                "home_runs": 30,
                "rbi": 100,
                "strikeouts_bat": 140,
                "total_bases": 400,
                "net_stolen_bases": 10,
                "at_bats": 900,
            }
        ]

        result = _project_row_from_player_scores(
            player_scores,
            games_remaining={"123.p.456": 3},
        )

        # Should have non-zero projections
        assert result.R > 0
        assert result.H > 0
        assert result.HR_B > 0
        assert result.RBI > 0


class TestBuildScoreboardRows:
    """Tests for build_scoreboard_rows()."""

    def test_all_18_categories(self):
        my_row = ROWProjectionResult(
            R=50.0, H=120.0, HR_B=18.0, RBI=55.0, K_B=80.0, TB=180.0, NSB=5.0,
            AVG=0.270, OPS=0.780,
            W=4.0, L=2.0, HR_P=12.0, K_P=90.0, QS=5.0, NSV=-2.0,
            ERA=3.50, WHIP=1.20, K_9=9.5,
        )
        opp_row = ROWProjectionResult(
            R=45.0, H=115.0, HR_B=15.0, RBI=50.0, K_B=75.0, TB=170.0, NSB=3.0,
            AVG=0.265, OPS=0.770,
            W=3.0, L=3.0, HR_P=15.0, K_P=85.0, QS=4.0, NSV=-1.0,
            ERA=3.80, WHIP=1.25, K_9=9.0,
        )

        my_current = {cat: getattr(my_row, cat) * 0.5 for cat in SCORING_CATEGORY_CODES}
        opp_current = {cat: getattr(opp_row, cat) * 0.5 for cat in SCORING_CATEGORY_CODES}

        rows = build_scoreboard_rows(
            my_current=my_current,
            opp_current=opp_current,
            my_row=my_row,
            opp_row=opp_row,
            category_math={},
            monte_carlo_result=None,
            ip_accumulated=45.0,
            ip_minimum=90.0,
            games_remaining=3,
        )

        assert len(rows) == 18
        assert all(r.category_label for r in rows)
        assert all(r.my_current is not None for r in rows)
        assert all(r.opp_current is not None for r in rows)

    def test_lower_better_flag(self):
        rows = build_scoreboard_rows(
            my_current={cat: 0.0 for cat in SCORING_CATEGORY_CODES},
            opp_current={cat: 0.0 for cat in SCORING_CATEGORY_CODES},
            my_row=ROWProjectionResult(**{cat: 0.0 for cat in SCORING_CATEGORY_CODES}),
            opp_row=ROWProjectionResult(**{cat: 0.0 for cat in SCORING_CATEGORY_CODES}),
            category_math={},
            monte_carlo_result=None,
        )

        era_row = next(r for r in rows if r.category == "ERA")
        assert era_row.is_lower_better is True

        runs_row = next(r for r in rows if r.category == "R")
        assert runs_row.is_lower_better is False


class TestAssembleMatchupScoreboard:
    """Tests for assemble_matchup_scoreboard()."""

    def test_complete_response(self):
        # Provide ratio stat components via player_scores
        player_scores = [
            {
                "yahoo_player_key": "test_player",
                "rolling_14d": {
                    "w_earned_runs": 20.0,
                    "w_ip": 15.0,
                    "w_whip_numer": 30.0,
                    "w_hits": 50.0,
                    "w_ab": 200.0,
                    "w_strikeouts_pit": 80.0,
                },
                "runs": 100,
                "hits": 50,
                "home_runs": 15,
                "rbi": 48,
                "strikeouts_bat": 78,
                "total_bases": 165,
                "net_stolen_bases": 3,
                "at_bats": 200,
                "earned_runs": 20,
                "ip": 15,
                "hits_allowed": 30,
                "walks_allowed": 10,
                "strikeouts_pit": 80,
                "quality_starts": 3,
            }
        ]

        my_current = {
            "R": 45.0, "H": 110.0, "HR_B": 15.0, "RBI": 48.0,
            "K_B": 78.0, "TB": 165.0, "AVG": 0.268, "OPS": 0.765,
            "NSB": 3.0, "W": 3.0, "L": 4.0, "HR_P": 16.0,
            "K_P": 80.0, "ERA": 4.00, "WHIP": 1.30, "K_9": 8.5,
            "QS": 3.0, "NSV": -3.0,
        }
        opp_current = {
            "R": 50.0, "H": 115.0, "HR_B": 18.0, "RBI": 52.0,
            "K_B": 72.0, "TB": 175.0, "AVG": 0.275, "OPS": 0.780,
            "NSB": 5.0, "W": 4.0, "L": 3.0, "HR_P": 12.0,
            "K_P": 88.0, "ERA": 3.60, "WHIP": 1.20, "K_9": 9.2,
            "QS": 5.0, "NSV": -1.0,
        }

        result = assemble_matchup_scoreboard(
            week=5,
            opponent_name="Thundercats",
            my_current_stats=my_current,
            opp_current_stats=opp_current,
            my_player_scores=player_scores,
            ip_accumulated=45.0,
            ip_minimum=90.0,
            games_remaining=3,
            days_remaining=4,
            acquisitions_used=5,
            il_used=1,
            n_monte_carlo_sims=100,
        )

        assert result.week == 5
        assert result.opponent_name == "Thundercats"
        assert len(result.rows) == 18
        assert result.budget is not None
        assert result.freshness is not None
        assert 0 <= result.categories_won <= 18
        assert 0 <= result.categories_lost <= 18

    def test_category_counts_match(self):
        player_scores = [
            {
                "yahoo_player_key": "test_player",
                "rolling_14d": {
                    "w_earned_runs": 20.0,
                    "w_ip": 15.0,
                    "w_whip_numer": 30.0,
                    "w_hits": 50.0,
                    "w_ab": 200.0,
                    "w_strikeouts_pit": 80.0,
                },
                "hits": 50,
                "at_bats": 200,
                "total_bases": 165,
                "walks": 20,
                "earned_runs": 20,
                "ip": 15,
                "hits_allowed": 30,
                "walks_allowed": 10,
                "strikeouts_pit": 80,
            }
        ]

        my_current = {cat: 50.0 for cat in SCORING_CATEGORY_CODES}
        opp_current = {cat: 40.0 for cat in SCORING_CATEGORY_CODES}

        result = assemble_matchup_scoreboard(
            week=1,
            opponent_name="Opponent",
            my_current_stats=my_current,
            opp_current_stats=opp_current,
            my_player_scores=player_scores,
        )

        # All categories should be won (higher is better for most)
        # Note: lower_is_better categories will have opposite margin
        total = result.categories_won + result.categories_lost + result.categories_tied
        assert total == 18


class TestBudgetStateFunction:
    """Unit tests for compute_budget_state() function."""

    def test_acquisitions_remaining_calculation(self):
        """acquisitions_remaining = limit - used."""
        budget = compute_budget_state(
            acquisitions_used=5,
            acquisition_limit=8,
        )
        assert budget.acquisitions_remaining == 3

    def test_acquisition_warning_at_6(self):
        """acquisition_warning is True when 6 or more acquisitions used."""
        budget = compute_budget_state(
            acquisitions_used=6,
            acquisition_limit=8,
        )
        assert budget.acquisition_warning is True

    def test_no_acquisition_warning_below_6(self):
        """acquisition_warning is False when fewer than 6 acquisitions used."""
        budget = compute_budget_state(
            acquisitions_used=5,
            acquisition_limit=8,
        )
        assert budget.acquisition_warning is False

    def test_ip_pace_flag_is_valid(self):
        """ip_pace returns a valid IPPaceFlag enum."""
        budget = compute_budget_state(
            acquisitions_used=0,
            ip_accumulated=45.0,
            ip_minimum=90.0,
            season_days_elapsed=30,
        )
        # Should be one of the valid flags
        assert budget.ip_pace.value in ("behind", "on_track", "ahead", "complete", "unknown")

    def test_budget_as_of_is_recent(self):
        """as_of timestamp is set to current time."""
        from datetime import datetime
        budget = compute_budget_state(acquisitions_used=0)
        now = datetime.now(tz=budget.as_of.tzinfo)
        # Should be within 1 second
        assert (now - budget.as_of).total_seconds() < 1.0

