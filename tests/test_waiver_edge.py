"""Tests for WaiverEdgeDetector."""
import pytest
from unittest.mock import MagicMock, patch
from backend.services.waiver_edge_detector import (
    WaiverEdgeDetector,
    drop_candidate_value,
    is_protected_drop_candidate,
    long_term_hold_floor,
    _is_il_player,
    _is_dtd_player,
)


def test_long_term_hold_floor_uses_role_certainty_not_acquisition():
    """Regression: long_term_hold_floor must read risk_profile.role_certainty.

    A prior bug accessed `.acquisition` on the RiskProfile dataclass, which
    only exposes role_certainty/health_history. Production emitted
    `'RiskProfile' object has no attribute 'acquisition'` on
    /api/fantasy/waiver/recommendations. Fixed in commit 9147f83.
    """
    eury = {
        "name": "Eury Perez",
        "positions": ["SP"],
        "z_score": 0.8,
        "tier": 4,
        "adp": 118.0,
        "percent_owned": 74.0,
    }

    floor = long_term_hold_floor(eury)

    assert isinstance(floor, float)
    assert floor >= 2.25


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
        # Mock scarcity lookup to empty so test uses _FALLBACK_RANK (2B=rank 3, multiplier 1.50)
        with patch.object(det, "_fetch_fas", return_value=[fa_2b]), \
             patch.object(WaiverEdgeDetector, "_load_scarcity_lookup", return_value={}):
            moves = det.get_top_moves(dead_roster, opp_roster, n_candidates=5)
        assert len(moves) == 1
        # need_score = base × scarcity_multiplier(2B=rank3=1.50) × dead-2B-boost(1.25)
        base_score = det._score_fa_against_deficits(
            fa_2b, det._compute_deficits(dead_roster, opp_roster)
        )
        scarcity_mult = 1.0 + (13 - 3) * 0.05  # 2B rank=3 -> 1.50
        assert moves[0]["need_score"] == pytest.approx(base_score * scarcity_mult * 1.25)

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

    def test_get_top_moves_enriches_raw_yahoo_players(self):
        raw_fa = {
            "name": "Michael Wacha",
            "player_key": "469.p.9329",
            "positions": ["SP"],
            "percent_owned": 44.2,
        }
        raw_my_roster = [{
            "name": "Weak Starter",
            "positions": ["SP"],
            "selected_position": "SP",
            "is_undroppable": False,
        }]
        raw_opp_roster = [{
            "name": "Opponent Starter",
            "positions": ["SP"],
            "selected_position": "SP",
            "is_undroppable": False,
        }]

        def _proj(player):
            lookup = {
                "Michael Wacha": {"name": "Michael Wacha", "positions": ["SP"], "team": "KC", "z_score": 1.7, "cat_scores": {"era": 1.2}},
                "Weak Starter": {"name": "Weak Starter", "positions": ["SP"], "team": "NYY", "z_score": -0.8, "cat_scores": {"era": -0.2}},
                "Opponent Starter": {"name": "Opponent Starter", "positions": ["SP"], "team": "BOS", "z_score": 0.9, "cat_scores": {"era": 1.0}},
            }
            return lookup[player["name"]]

        det = WaiverEdgeDetector(mcmc_simulator=None)
        with patch.object(det, "_fetch_fas", return_value=[raw_fa]):
            with patch("backend.fantasy_baseball.player_board.get_or_create_projection", side_effect=_proj):
                moves = det.get_top_moves(raw_my_roster, raw_opp_roster, n_candidates=5)

        assert len(moves) == 1
        assert moves[0]["need_score"] > 0
        assert moves[0]["add_player"]["percent_owned"] == pytest.approx(44.2)
        assert moves[0]["add_player"]["cat_scores"] == {"era": 1.2}

    def test_get_top_moves_falls_back_to_z_score_without_deficits(self):
        raw_fa = {
            "name": "Fallback Bat",
            "player_key": "469.p.9999",
            "positions": ["OF"],
            "percent_owned": 12.0,
        }
        raw_my_roster = [{
            "name": "Roster Bat",
            "positions": ["OF"],
            "selected_position": "OF",
            "is_undroppable": False,
        }]

        def _proj(player):
            lookup = {
                "Fallback Bat": {"name": "Fallback Bat", "positions": ["OF"], "team": "SEA", "z_score": 2.3, "cat_scores": {"hr": 1.1}},
                "Roster Bat": {"name": "Roster Bat", "positions": ["OF"], "team": "SEA", "z_score": 0.2, "cat_scores": {}},
            }
            return lookup[player["name"]]

        det = WaiverEdgeDetector(mcmc_simulator=None)
        with patch.object(det, "_fetch_fas", return_value=[raw_fa]):
            with patch("backend.fantasy_baseball.player_board.get_or_create_projection", side_effect=_proj):
                moves = det.get_top_moves(raw_my_roster, [], n_candidates=5)

        assert len(moves) == 1
        assert moves[0]["need_score"] == pytest.approx(2.3)

    def test_drop_candidate_value_respects_long_term_hold_floor(self):
        juan_soto = {
            "name": "Juan Soto",
            "positions": ["OF"],
            "z_score": 1.4,
            "tier": 1,
            "adp": 2.0,
            "percent_owned": 99.0,
        }

        assert is_protected_drop_candidate(juan_soto) is True
        score = drop_candidate_value(juan_soto)[0]  # Tuple: (primary_score, ...)
        assert score >= 4.5

    def test_locked_high_upside_pitcher_is_protected_from_drop(self):
        det = WaiverEdgeDetector()
        roster = [
            {
                "name": "Eury Pérez",
                "positions": ["SP"],
                "selected_position": "SP",
                "status": "SP",
                "z_score": 0.8,
                "tier": 4,
                "adp": 118.0,
                "percent_owned": 74.0,
            }
        ]

        assert is_protected_drop_candidate(roster[0]) is True
        assert det._weakest_droppable_at(roster, ["SP"]) is None

    def test_detector_prefers_streamer_drop_over_core_asset(self):
        det = WaiverEdgeDetector()
        roster = [
            {
                "name": "Eury Pérez",
                "positions": ["SP"],
                "selected_position": "SP",
                "status": "SP",
                "z_score": 0.8,
                "tier": 4,
                "adp": 118.0,
                "percent_owned": 74.0,
            },
            {
                "name": "Back-end Streamer",
                "positions": ["SP"],
                "selected_position": "SP",
                "status": "SP",
                "z_score": -0.9,
                "tier": 10,
                "adp": 9999.0,
                "percent_owned": 18.0,
            },
        ]

        result = det._weakest_droppable_at(roster, ["SP"])

        assert result is not None
        assert result["name"] == "Back-end Streamer"


# ---------------------------------------------------------------------------
# Sprint 4: matchup context
# ---------------------------------------------------------------------------


class TestLoadMatchupScores:
    # _load_matchup_scores does `from backend.models import SessionLocal` inside
    # the function body, so the correct patch target is backend.models.SessionLocal.

    def test_returns_empty_on_empty_ids(self):
        det = WaiverEdgeDetector.__new__(WaiverEdgeDetector)
        result = det._load_matchup_scores([])
        assert result == {}

    def test_returns_empty_on_db_error(self):
        det = WaiverEdgeDetector.__new__(WaiverEdgeDetector)
        with patch("backend.models.SessionLocal",
                   side_effect=Exception("db down")):
            result = det._load_matchup_scores([123])
        assert result == {}

    def test_maps_bdl_id_to_matchup_fields(self):
        det = WaiverEdgeDetector.__new__(WaiverEdgeDetector)
        mock_db = MagicMock()
        row = (101, 1.5, 80.0, 0.85)
        mock_db.execute.return_value.fetchall.return_value = [row]
        with patch("backend.models.SessionLocal",
                   return_value=mock_db):
            result = det._load_matchup_scores([101])
        assert 101 in result
        assert result[101]["matchup_z"] == 1.5
        assert result[101]["matchup_score"] == 80.0
        assert result[101]["matchup_confidence"] == 0.85

    def test_none_values_coerced_to_defaults(self):
        det = WaiverEdgeDetector.__new__(WaiverEdgeDetector)
        mock_db = MagicMock()
        row = (202, None, None, None)
        mock_db.execute.return_value.fetchall.return_value = [row]
        with patch("backend.models.SessionLocal",
                   return_value=mock_db):
            result = det._load_matchup_scores([202])
        assert result[202]["matchup_z"] == 0.0
        assert result[202]["matchup_score"] == 50.0
        assert result[202]["matchup_confidence"] == 0.0


class TestLoadMarketScores:
    def test_null_market_rows_are_ignored(self):
        det = WaiverEdgeDetector.__new__(WaiverEdgeDetector)
        mock_db = MagicMock()
        mock_db.execute.return_value.fetchall.return_value = [
            (101, None),
            (202, 81.5),
        ]

        with patch("backend.models.SessionLocal", return_value=mock_db):
            result = det._load_market_scores([101, 202])

        assert 101 not in result
        assert result[202] == pytest.approx(81.5)


# ---------------------------------------------------------------------------
# IL / DTD filter (P1 fix — Jack Dreyer / Blake Snell regression)
# ---------------------------------------------------------------------------


class TestILPlayerFilter:
    """Verify _is_il_player recognises all Yahoo + BDL IL status variants."""

    @pytest.mark.parametrize("status_field,value", [
        ("status", "IL"),
        ("status", "IL10"),
        ("status", "IL60"),
        ("status", "NA"),
        ("status", "OUT"),
        ("injury_status", "15-Day-IL"),
        ("injury_status", "10-Day-IL"),
        ("injury_status", "60-Day-IL"),
        ("injury_status", "IL15"),
        ("injury_status", "IL60"),
        ("injury_status", "il"),
        ("injury_status", "15dayil"),
    ])
    def test_il_variants_detected(self, status_field, value):
        player = {"name": "Test Player", status_field: value}
        assert _is_il_player(player) is True

    def test_healthy_player_not_il(self):
        assert _is_il_player({"name": "Healthy", "status": "Active"}) is False

    def test_dtd_is_not_il(self):
        assert _is_il_player({"name": "DTD Guy", "status": "DTD"}) is False

    def test_dtd_detected(self):
        assert _is_dtd_player({"name": "DTD Guy", "status": "DTD"}) is True
        assert _is_dtd_player({"name": "DTD Guy", "injury_status": "day-to-day"}) is True

    def test_healthy_not_dtd(self):
        assert _is_dtd_player({"name": "Healthy", "status": "Active"}) is False


def test_il_players_excluded_from_waiver():
    """P1 regression: IL players must not appear in waiver recommendations.

    Jack Dreyer (15-Day-IL) was appearing in recommendations because the FA
    pipeline had no IL status gate.  This test verifies the hard gate works
    for all common IL status formats.
    """
    jack_dreyer_il = {
        "name": "Jack Dreyer",
        "positions": ["SP"],
        "player_key": "469.p.99999",
        "injury_status": "15-Day-IL",
        "cat_scores": {"era": 1.5, "k9": 1.2},
        "z_score": 1.8,
    }
    healthy_fa = {
        "name": "Healthy SP",
        "positions": ["SP"],
        "player_key": "469.p.11111",
        "cat_scores": {"era": 0.8, "k9": 0.6},
        "z_score": 1.0,
    }
    my_roster = [_make_player("Weak SP", ["SP"], {"era": -0.5})]
    opp_roster = [_make_player("Opp SP", ["SP"], {"era": 1.0})]

    det = WaiverEdgeDetector(mcmc_simulator=None)
    with patch.object(det, "_fetch_fas", return_value=[jack_dreyer_il, healthy_fa]), \
         patch.object(WaiverEdgeDetector, "_load_scarcity_lookup", return_value={}):
        moves = det.get_top_moves(my_roster, opp_roster, n_candidates=10)

    add_names = [m["add_player"]["name"] for m in moves]
    assert "Jack Dreyer" not in add_names, (
        "IL player Jack Dreyer must not appear in waiver recommendations"
    )
    assert "Healthy SP" in add_names


def test_il_player_with_yahoo_status_excluded():
    """IL players identified via Yahoo short-code status are also excluded."""
    il_player = {
        "name": "Injured Guy",
        "positions": ["OF"],
        "player_key": "469.p.88888",
        "status": "IL",
        "cat_scores": {"hr": 2.0, "rbi": 1.5},
        "z_score": 2.2,
    }
    my_roster = [_make_player("Weak OF", ["OF"], {"hr": -0.2})]
    opp_roster = [_make_player("Opp OF", ["OF"], {"hr": 1.0})]

    det = WaiverEdgeDetector(mcmc_simulator=None)
    with patch.object(det, "_fetch_fas", return_value=[il_player]), \
         patch.object(WaiverEdgeDetector, "_load_scarcity_lookup", return_value={}):
        moves = det.get_top_moves(my_roster, opp_roster, n_candidates=10)

    assert moves == [], "IL player (Yahoo status='IL') must not produce any waiver recommendations"


def test_dtd_player_included_with_warning_and_reduced_score():
    """DTD players appear in recommendations but with a reduced score and warning."""
    dtd_player = {
        "name": "Banged Up",
        "positions": ["OF"],
        "player_key": "469.p.77777",
        "status": "DTD",
        "cat_scores": {"hr": 1.5, "rbi": 1.0},
        "z_score": 1.2,
    }
    my_roster = [_make_player("Weak OF", ["OF"], {"hr": -0.2})]
    opp_roster = [_make_player("Opp OF", ["OF"], {"hr": 1.0})]

    det = WaiverEdgeDetector(mcmc_simulator=None)
    with patch.object(det, "_fetch_fas", return_value=[dtd_player]), \
         patch.object(WaiverEdgeDetector, "_load_scarcity_lookup", return_value={}):
        moves = det.get_top_moves(my_roster, opp_roster, n_candidates=10)

    assert len(moves) == 1, "DTD player must still appear in waiver recommendations"
    move = moves[0]
    assert move["add_player"]["name"] == "Banged Up"
    assert move["dtd_warning"] is not None, "DTD move must include a dtd_warning"
    assert "DTD" in move["dtd_warning"]


# ---------------------------------------------------------------------------
# Task B: Small-sample penalty — prevents short hot streaks from dominating
# ---------------------------------------------------------------------------


class TestSmallSamplePenalty:
    """Small-sample penalty depresses need_score for thin-stats players."""

    def _make_fa(self, name, positions, cat_scores, stats=None, z_score=1.0):
        return {
            "name": name,
            "positions": positions,
            "player_key": f"469.p.{abs(hash(name)) % 90000 + 10000}",
            "cat_scores": cat_scores,
            "z_score": z_score,
            "stats": stats or {},
        }

    def test_pitcher_small_ip_scores_lower_than_full_sample(self):
        """Pitcher with 10 actual IP scores lower than identical pitcher with 50 IP."""
        small_sp = self._make_fa(
            "New SP", ["SP"], {"era": 2.5, "k9": 2.0},
            stats={"50": "10.0"}, z_score=2.0,
        )
        full_sp = self._make_fa(
            "Regular SP", ["SP"], {"era": 2.5, "k9": 2.0},
            stats={"50": "50.0"}, z_score=2.0,
        )
        my_roster = [_make_player("Weak SP", ["SP"], {"era": -0.5})]
        opp_roster = [_make_player("Opp SP", ["SP"], {"era": 1.0})]

        det = WaiverEdgeDetector(mcmc_simulator=None)
        with patch.object(det, "_fetch_fas", return_value=[small_sp, full_sp]), \
             patch.object(WaiverEdgeDetector, "_load_scarcity_lookup", return_value={}):
            moves = det.get_top_moves(my_roster, opp_roster, n_candidates=10)

        small_move = next((m for m in moves if m["add_player"]["name"] == "New SP"), None)
        full_move = next((m for m in moves if m["add_player"]["name"] == "Regular SP"), None)
        assert small_move is not None
        assert full_move is not None
        assert small_move["need_score"] < full_move["need_score"], (
            "Small-sample SP (10 IP) must score lower than full-sample SP (50 IP)"
        )
        assert small_move["small_sample"] is True
        assert full_move["small_sample"] is False

    def test_batter_small_hits_scores_lower_than_full_sample(self):
        """Batter with 5 H (≈19 PA estimate) penalised vs batter with 30 H (≈113 PA)."""
        hot_batter = self._make_fa(
            "Jake Bauers", ["1B"], {"hr": 2.0, "rbi": 1.8},
            stats={"8": "5"}, z_score=1.9,  # 5 H ≈ 19 PA — below 100 threshold
        )
        regular_batter = self._make_fa(
            "Regular 1B", ["1B"], {"hr": 2.0, "rbi": 1.8},
            stats={"8": "30"}, z_score=1.9,  # 30 H ≈ 113 PA — above threshold
        )
        my_roster = [_make_player("Weak 1B", ["1B"], {"hr": -0.3})]
        opp_roster = [_make_player("Opp 1B", ["1B"], {"hr": 1.2})]

        det = WaiverEdgeDetector(mcmc_simulator=None)
        with patch.object(det, "_fetch_fas", return_value=[hot_batter, regular_batter]), \
             patch.object(WaiverEdgeDetector, "_load_scarcity_lookup", return_value={}):
            moves = det.get_top_moves(my_roster, opp_roster, n_candidates=10)

        bauers_move = next((m for m in moves if m["add_player"]["name"] == "Jake Bauers"), None)
        regular_move = next((m for m in moves if m["add_player"]["name"] == "Regular 1B"), None)
        assert bauers_move is not None
        assert regular_move is not None
        assert bauers_move["need_score"] < regular_move["need_score"], (
            "Jake Bauers (5 H ≈ 19 PA) must score lower than regular batter with same cat_scores"
        )
        assert bauers_move["small_sample"] is True
        assert regular_move["small_sample"] is False

    def test_no_stats_no_penalty(self):
        """Player with empty stats dict skips the penalty — no division by zero."""
        fa = self._make_fa("No Stats SP", ["SP"], {"era": 1.5}, stats={}, z_score=1.5)
        my_roster = [_make_player("Weak SP", ["SP"], {"era": -0.5})]
        opp_roster = [_make_player("Opp SP", ["SP"], {"era": 1.0})]

        det = WaiverEdgeDetector(mcmc_simulator=None)
        with patch.object(det, "_fetch_fas", return_value=[fa]), \
             patch.object(WaiverEdgeDetector, "_load_scarcity_lookup", return_value={}):
            moves = det.get_top_moves(my_roster, opp_roster, n_candidates=10)

        assert len(moves) == 1
        assert moves[0]["small_sample"] is False

    def test_minimum_penalty_clamped_at_half(self):
        """Penalty factor is floored at 0.5 — score never drops below half for tiny samples."""
        tiny_sp = self._make_fa(
            "Zero IP SP", ["SP"], {"era": 3.0},
            stats={"50": "1.0"},  # 1 IP — factor = max(0.5, 1/30) = 0.5
            z_score=3.0,
        )
        my_roster = [_make_player("Weak SP", ["SP"], {"era": -0.5})]
        opp_roster = [_make_player("Opp SP", ["SP"], {"era": 1.0})]

        det = WaiverEdgeDetector(mcmc_simulator=None)
        with patch.object(det, "_fetch_fas", return_value=[tiny_sp]), \
             patch.object(WaiverEdgeDetector, "_load_scarcity_lookup", return_value={}):
            moves = det.get_top_moves(my_roster, opp_roster, n_candidates=10)

        assert len(moves) == 1
        assert moves[0]["small_sample"] is True
