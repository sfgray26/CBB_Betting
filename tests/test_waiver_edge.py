"""Tests for WaiverEdgeDetector."""
import pytest
from unittest.mock import MagicMock, patch
from backend.services.waiver_edge_detector import WaiverEdgeDetector


def _make_player(name, positions, cat_scores, is_undroppable=False):
    return {
        "name": name,
        "positions": positions,
        "cat_scores": cat_scores,
        "is_undroppable": is_undroppable,
    }


def _detector_with_fas(fas):
    det = WaiverEdgeDetector(mcmc_simulator=None)
    with patch.object(det, "_fetch_fas", return_value=fas):
        return det, fas


class TestWaiverEdgeDetector:

    def test_weakest_player_selected_as_drop(self):
        det = WaiverEdgeDetector()
        roster = [
            _make_player("Strong", ["OF"], {"hr": 2.0, "rbi": 1.5}),
            _make_player("Weak", ["1B"], {"hr": -0.5, "rbi": -0.8}),
        ]
        result = det._weakest_droppable(roster)
        assert result["name"] == "Weak"

    def test_score_fa_zero_deficit(self):
        det = WaiverEdgeDetector()
        # deficit = 0 on all cats -> score must be 0
        fa = _make_player("FA", ["OF"], {"hr": 2.0, "rbi": 1.0})
        score = det._score_fa_against_deficits(fa, {"hr": 0.0, "rbi": 0.0})
        assert score == 0.0

    def test_score_fa_positive_deficit(self):
        det = WaiverEdgeDetector()
        fa = _make_player("FA", ["OF"], {"hr": 2.0})
        # opponent leads by 3.0 in hr -> deficit 3.0
        score = det._score_fa_against_deficits(fa, {"hr": 3.0})
        assert score == pytest.approx(6.0)

    def test_dead_2b_detection(self):
        det = WaiverEdgeDetector()
        roster = [
            _make_player("Good2B", ["2B"], {"hr": 1.0, "rbi": 1.0}),
            _make_player("Dead2B", ["2B"], {"hr": -0.6, "rbi": -0.8}),  # sum = -1.4 < -1.0
        ]
        assert det._has_dead_2b(roster) is True

    def test_dead_2b_not_triggered_above_threshold(self):
        det = WaiverEdgeDetector()
        roster = [
            _make_player("OK2B", ["2B"], {"hr": 0.0, "rbi": 0.0}),  # sum = 0, not dead
        ]
        assert det._has_dead_2b(roster) is False

    def test_westburg_2b_boost(self):
        dead_roster = [_make_player("Dead2B", ["2B"], {"hr": -0.8, "rbi": -0.5})]
        opp_roster = [_make_player("Opp", ["OF"], {"hr": 1.0})]
        fa_2b = _make_player("Westburg", ["2B"], {"hr": 1.5}, )

        det = WaiverEdgeDetector(mcmc_simulator=None)
        with patch.object(det, "_fetch_fas", return_value=[fa_2b]):
            moves = det.get_top_moves(dead_roster, opp_roster, n_candidates=5)
        assert len(moves) == 1
        # need_score should be boosted by 1.25x relative to un-boosted version
        base_score = det._score_fa_against_deficits(
            fa_2b, det._compute_deficits(dead_roster, opp_roster)
        )
        assert moves[0]["need_score"] == pytest.approx(base_score * 1.25)

    def test_empty_free_agents_returns_empty(self):
        det = WaiverEdgeDetector(mcmc_simulator=None)
        with patch.object(det, "_fetch_fas", return_value=[]):
            result = det.get_top_moves([], [], n_candidates=5)
        assert result == []

    def test_mcmc_disabled_when_no_simulator(self):
        fa = _make_player("SomeFA", ["OF"], {"hr": 1.0, "rbi": 0.5})
        my_roster = [_make_player("Weak", ["OF"], {"hr": -0.2})]
        opp_roster = [_make_player("Opp", ["OF"], {"hr": 0.5})]

        det = WaiverEdgeDetector(mcmc_simulator=None)
        with patch.object(det, "_fetch_fas", return_value=[fa]):
            moves = det.get_top_moves(my_roster, opp_roster, n_candidates=5)
        assert len(moves) == 1
        assert moves[0]["mcmc_enabled"] is False
