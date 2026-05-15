"""
Edge case tests for scarcity solver integration.

Covers three areas from the P1 bug audit (reports/p1_bug_audit_2025-05-15.md):
  1. _can_fill_slot() — module-level eligibility helper in backend/routers/fantasy.py
  2. Scarcity bonus logic — mirrors the _scarcity_score() inner function in optimize_roster
  3. LineupConstraintSolver (greedy path) — catcher scarcity + multi-position handling

All tests are pure / no-DB / no-network.
"""
import pytest

from backend.routers.fantasy import _can_fill_slot, _HITTER_POSITIONS
from backend.fantasy_baseball.lineup_constraint_solver import (
    LineupConstraintSolver,
    PositionSlot,
)
from backend.fantasy_baseball.elite_lineup_scorer import EliteScore


# ─── helpers ──────────────────────────────────────────────────────────────


def _make_score(total: float) -> EliteScore:
    return EliteScore(
        total_score=total,
        environment_score=total,
        matchup_multiplier=1.0,
        platoon_multiplier=1.0,
        form_adjusted_woba=0.310,
        regression_boost=0.0,
        lineup_spot_bonus=0.0,
        confidence=0.8,
        data_quality="tier_1",
        reasoning="test",
    )


def _make_player(pid: str, name: str) -> dict:
    return {"player_id": pid, "name": name}


# Mirrors the _scarcity_score bonus formula from fantasy.py optimize_roster.
# Used to keep scarcity bonus tests independent of the endpoint import path.
_SCARCE_POSITIONS = ["C", "SS", "2B", "3B", "1B"]


def _scarcity_bonus(eligible_positions: list) -> int:
    positions = [p.upper() for p in (eligible_positions or [])]
    bonus = 0
    for i, pos in enumerate(_SCARCE_POSITIONS):
        if pos in positions:
            bonus = max(bonus, 10 - i)
    hitting = set(positions) & _HITTER_POSITIONS
    if len(hitting) >= 3:
        bonus += 3
    return bonus


# ─── _can_fill_slot ────────────────────────────────────────────────────────


class TestCanFillSlot:
    """Eligibility mapping: position eligibility → slot assignment."""

    # --- catcher ---

    def test_catcher_fills_c_slot(self):
        assert _can_fill_slot(["C"], "C", "Catcher") is True

    def test_catcher_fills_util_slot(self):
        # C is a hitter — should qualify for Util
        assert _can_fill_slot(["C"], "Util", "Catcher") is True

    def test_catcher_cannot_fill_ss_slot(self):
        assert _can_fill_slot(["C"], "SS", "Catcher") is False

    def test_catcher_cannot_fill_of_slot(self):
        assert _can_fill_slot(["C"], "OF", "Catcher") is False

    def test_catcher_cannot_fill_1b_slot(self):
        assert _can_fill_slot(["C"], "1B", "Catcher") is False

    # --- outfield variants ---

    def test_lf_fills_of_slot(self):
        assert _can_fill_slot(["LF"], "OF", "LF player") is True

    def test_cf_fills_of_slot(self):
        assert _can_fill_slot(["CF"], "OF", "CF player") is True

    def test_rf_fills_of_slot(self):
        assert _can_fill_slot(["RF"], "OF", "RF player") is True

    def test_of_fills_of_slot(self):
        assert _can_fill_slot(["OF"], "OF", "OF player") is True

    def test_lf_cannot_fill_c_slot(self):
        assert _can_fill_slot(["LF"], "C", "LF player") is False

    # --- DH / Util ---

    def test_dh_fills_util_slot(self):
        assert _can_fill_slot(["DH"], "Util", "DH player") is True

    def test_dh_cannot_fill_c_slot(self):
        assert _can_fill_slot(["DH"], "C", "DH player") is False

    # --- pitchers ---

    def test_sp_fills_p_slot(self):
        assert _can_fill_slot(["SP"], "P", "SP pitcher") is True

    def test_rp_fills_p_slot(self):
        assert _can_fill_slot(["RP"], "P", "RP pitcher") is True

    def test_sp_cannot_fill_rp_slot(self):
        # "RP" slot requires exact match; SP doesn't qualify
        assert _can_fill_slot(["SP"], "RP", "SP pitcher") is False

    def test_pitcher_cannot_fill_util_slot(self):
        assert _can_fill_slot(["SP"], "Util", "SP pitcher") is False

    # --- edge cases ---

    def test_empty_positions_returns_false(self):
        assert _can_fill_slot([], "C", "Player") is False

    def test_none_positions_returns_false(self):
        assert _can_fill_slot(None, "C", "Player") is False

    def test_lowercase_catcher_fills_c_slot(self):
        # Function must normalise to uppercase
        assert _can_fill_slot(["c"], "C", "Catcher lowercase") is True

    # --- multi-position ---

    def test_c_ss_player_fills_c_slot(self):
        assert _can_fill_slot(["C", "SS"], "C", "C/SS player") is True

    def test_c_ss_player_fills_ss_slot(self):
        assert _can_fill_slot(["C", "SS"], "SS", "C/SS player") is True

    def test_c_ss_player_fills_util_slot(self):
        assert _can_fill_slot(["C", "SS"], "Util", "C/SS player") is True

    def test_c_ss_player_cannot_fill_of_slot(self):
        assert _can_fill_slot(["C", "SS"], "OF", "C/SS player") is False


# ─── Scarcity bonus logic ──────────────────────────────────────────────────


class TestScarcityBonus:
    """
    Verifies the scarcity bonus formula used in optimize_roster._scarcity_score.

    SCARCE_POSITIONS = ["C", "SS", "2B", "3B", "1B"]
    bonus = max(10 - index) for matching positions
    +3 when >= 3 hitting positions are eligible
    """

    def test_catcher_bonus(self):
        # C is index 0 → 10 - 0 = 10
        assert _scarcity_bonus(["C"]) == 10

    def test_ss_bonus(self):
        # SS is index 1 → 10 - 1 = 9
        assert _scarcity_bonus(["SS"]) == 9

    def test_2b_bonus(self):
        assert _scarcity_bonus(["2B"]) == 8

    def test_3b_bonus(self):
        assert _scarcity_bonus(["3B"]) == 7

    def test_1b_bonus(self):
        assert _scarcity_bonus(["1B"]) == 6

    def test_of_no_scarcity_bonus(self):
        assert _scarcity_bonus(["OF"]) == 0

    def test_dh_no_scarcity_bonus(self):
        assert _scarcity_bonus(["DH"]) == 0

    def test_pitcher_no_bonus(self):
        assert _scarcity_bonus(["SP"]) == 0

    def test_empty_positions_no_bonus(self):
        assert _scarcity_bonus([]) == 0

    def test_c_bonus_dominates_ss_for_dual_eligible(self):
        # C(10) > SS(9) — max of both
        assert _scarcity_bonus(["C", "SS"]) == 10

    def test_three_hitting_positions_adds_3(self):
        # C bonus (10) + multi-eligible bonus (3) = 13
        assert _scarcity_bonus(["C", "1B", "OF"]) == 13

    def test_two_hitting_positions_no_multi_bonus(self):
        # Only 2 hitter slots → no +3; just C bonus
        assert _scarcity_bonus(["C", "1B"]) == 10

    def test_three_of_variants_multi_bonus_only(self):
        # LF+CF+RF = 3 hitting positions, none scarce → only +3
        assert _scarcity_bonus(["LF", "CF", "RF"]) == 3

    def test_catcher_bonus_sorts_over_equal_score_1b(self):
        # C effective=50+10=60; 1B effective=50+6=56 → C ranks higher
        c_eff = 50 + _scarcity_bonus(["C"])
        b1_eff = 50 + _scarcity_bonus(["1B"])
        assert c_eff > b1_eff

    def test_large_score_gap_overrides_catcher_bonus(self):
        # 1B with score=70 effective=76; C with score=50 effective=60 → 1B ranks higher
        c_eff = 50 + _scarcity_bonus(["C"])
        b1_eff = 70 + _scarcity_bonus(["1B"])
        assert b1_eff > c_eff


# ─── LineupConstraintSolver — greedy path ─────────────────────────────────
#
# Force greedy by setting solver.use_ortools = False so tests work without
# OR-Tools installed.


def _greedy_solver() -> LineupConstraintSolver:
    s = LineupConstraintSolver()
    s.use_ortools = False
    return s


def _build_inputs(players_cfg: list):
    """
    players_cfg: list of dicts with keys pid, name, score, positions.
    Returns (roster, scores, eligibility) ready for LineupConstraintSolver.solve().
    """
    roster = [_make_player(p["pid"], p["name"]) for p in players_cfg]
    scores = {p["pid"]: _make_score(p["score"]) for p in players_cfg}
    elig = {p["pid"]: p["positions"] for p in players_cfg}
    return roster, scores, elig


# Minimal 9-player roster that covers all infield + OF + Util slots.
_BASE_ROSTER = [
    {"pid": "c1",  "name": "Catcher",  "score": 60.0, "positions": ["C"]},
    {"pid": "ss1", "name": "Shortstop", "score": 70.0, "positions": ["SS"]},
    {"pid": "2b1", "name": "SecondBase","score": 65.0, "positions": ["2B"]},
    {"pid": "3b1", "name": "ThirdBase", "score": 63.0, "positions": ["3B"]},
    {"pid": "1b1", "name": "FirstBase", "score": 72.0, "positions": ["1B"]},
    {"pid": "of1", "name": "Outfield1", "score": 68.0, "positions": ["OF"]},
    {"pid": "of2", "name": "Outfield2", "score": 67.0, "positions": ["OF"]},
    {"pid": "of3", "name": "Outfield3", "score": 66.0, "positions": ["OF"]},
    {"pid": "dh1", "name": "DH",        "score": 55.0, "positions": ["DH"]},
]


class TestGreedySolverCatcherScarcity:
    """Catcher must fill C slot; C is most scarce (SLOT_CONFIG rank 1)."""

    def test_catcher_assigned_to_c_slot(self):
        solver = _greedy_solver()
        roster, scores, elig = _build_inputs(_BASE_ROSTER)
        result = solver.solve(roster, scores, elig)

        c_asgn = next((a for a in result.assignments if a.slot == PositionSlot.CATCHER), None)
        assert c_asgn is not None
        assert c_asgn.player_id == "c1", "Catcher should fill C slot"

    def test_c_slot_empty_when_no_catcher_on_roster(self):
        no_catcher = [p for p in _BASE_ROSTER if "C" not in p["positions"]]
        solver = _greedy_solver()
        roster, scores, elig = _build_inputs(no_catcher)
        result = solver.solve(roster, scores, elig)

        c_asgn = next((a for a in result.assignments if a.slot == PositionSlot.CATCHER), None)
        assert c_asgn is not None
        assert c_asgn.player_id == "", "C slot must show EMPTY when no catcher on roster"
        assert c_asgn.player_name == "EMPTY"

    def test_catcher_goes_to_c_not_util_when_both_available(self):
        # C slot has scarcity rank 1; Util rank 9 — solver fills C first.
        # Even if catcher has a lower score than Util candidates, it still fills C.
        roster_cfg = list(_BASE_ROSTER)  # catcher score=60, all others higher
        solver = _greedy_solver()
        roster, scores, elig = _build_inputs(roster_cfg)
        result = solver.solve(roster, scores, elig)

        c_asgn = next((a for a in result.assignments if a.slot == PositionSlot.CATCHER), None)
        util_asgn = next((a for a in result.assignments if a.slot == PositionSlot.UTILITY), None)
        assert c_asgn is not None and c_asgn.player_id == "c1"
        assert util_asgn is not None and util_asgn.player_id != "c1", \
            "Catcher must not be placed in Util instead of C"

    def test_catcher_score_reported_correctly_in_c_slot(self):
        solver = _greedy_solver()
        roster, scores, elig = _build_inputs(_BASE_ROSTER)
        result = solver.solve(roster, scores, elig)

        c_asgn = next(a for a in result.assignments if a.slot == PositionSlot.CATCHER)
        assert c_asgn.score == pytest.approx(60.0)


class TestGreedySolverMultiPosition:
    """Multi-position players fill the scarcest eligible slot first."""

    def test_ss_of_player_fills_ss_not_of(self):
        # No other SS on roster; SS/OF player should fill SS (rank 2) before OF (rank 6-8)
        roster_cfg = [
            {"pid": "c1",    "name": "Catcher",   "score": 60.0, "positions": ["C"]},
            {"pid": "multi", "name": "SS/OF",      "score": 65.0, "positions": ["SS", "OF"]},
            {"pid": "2b1",   "name": "SecondBase", "score": 70.0, "positions": ["2B"]},
            {"pid": "3b1",   "name": "ThirdBase",  "score": 68.0, "positions": ["3B"]},
            {"pid": "1b1",   "name": "FirstBase",  "score": 72.0, "positions": ["1B"]},
            {"pid": "of1",   "name": "Outfield1",  "score": 75.0, "positions": ["OF"]},
            {"pid": "of2",   "name": "Outfield2",  "score": 73.0, "positions": ["OF"]},
            {"pid": "of3",   "name": "Outfield3",  "score": 71.0, "positions": ["OF"]},
            {"pid": "dh1",   "name": "DH",         "score": 55.0, "positions": ["DH"]},
        ]
        solver = _greedy_solver()
        roster, scores, elig = _build_inputs(roster_cfg)
        result = solver.solve(roster, scores, elig)

        ss_asgn = next((a for a in result.assignments if a.slot == PositionSlot.SHORTSTOP), None)
        assert ss_asgn is not None
        assert ss_asgn.player_id == "multi", "SS/OF player should fill SS slot (more scarce)"

    def test_c_ss_player_fills_c_not_ss(self):
        # C/SS dual-eligible with no other catcher → fills C (rank 1 < rank 2)
        roster_cfg = [
            {"pid": "multi", "name": "C/SS",      "score": 65.0, "positions": ["C", "SS"]},
            {"pid": "2b1",   "name": "SecondBase","score": 70.0, "positions": ["2B"]},
            {"pid": "3b1",   "name": "ThirdBase", "score": 68.0, "positions": ["3B"]},
            {"pid": "1b1",   "name": "FirstBase", "score": 72.0, "positions": ["1B"]},
            {"pid": "of1",   "name": "Outfield1", "score": 75.0, "positions": ["OF"]},
            {"pid": "of2",   "name": "Outfield2", "score": 73.0, "positions": ["OF"]},
            {"pid": "of3",   "name": "Outfield3", "score": 71.0, "positions": ["OF"]},
            {"pid": "dh1",   "name": "DH",        "score": 55.0, "positions": ["DH"]},
            {"pid": "dh2",   "name": "DH2",       "score": 52.0, "positions": ["DH"]},
        ]
        solver = _greedy_solver()
        roster, scores, elig = _build_inputs(roster_cfg)
        result = solver.solve(roster, scores, elig)

        c_asgn = next((a for a in result.assignments if a.slot == PositionSlot.CATCHER), None)
        assert c_asgn is not None
        assert c_asgn.player_id == "multi", "C/SS player should fill C (rank 1) not SS (rank 2)"

    def test_no_double_assignment_for_multi_position_player(self):
        roster_cfg = [
            {"pid": "multi", "name": "C/SS/2B", "score": 65.0, "positions": ["C", "SS", "2B"]},
            {"pid": "1b1",   "name": "1B",      "score": 72.0, "positions": ["1B"]},
            {"pid": "3b1",   "name": "3B",      "score": 68.0, "positions": ["3B"]},
            {"pid": "of1",   "name": "OF1",     "score": 75.0, "positions": ["OF"]},
            {"pid": "of2",   "name": "OF2",     "score": 73.0, "positions": ["OF"]},
            {"pid": "of3",   "name": "OF3",     "score": 71.0, "positions": ["OF"]},
            {"pid": "dh1",   "name": "DH",      "score": 55.0, "positions": ["DH"]},
            {"pid": "dh2",   "name": "DH2",     "score": 52.0, "positions": ["DH"]},
            {"pid": "dh3",   "name": "DH3",     "score": 50.0, "positions": ["DH"]},
        ]
        solver = _greedy_solver()
        roster, scores, elig = _build_inputs(roster_cfg)
        result = solver.solve(roster, scores, elig)

        multi_slots = [a for a in result.assignments if a.player_id == "multi"]
        assert len(multi_slots) == 1, "Multi-position player must appear in exactly one slot"

    def test_multi_eligible_does_not_prevent_dedicated_player_from_scarcer_slot(self):
        # C/OF player and a dedicated SS with higher score.
        # Expected: C/OF fills C, SS fills SS independently.
        roster_cfg = [
            {"pid": "c_of", "name": "C/OF",     "score": 60.0, "positions": ["C", "OF"]},
            {"pid": "ss1",  "name": "Shortstop", "score": 80.0, "positions": ["SS"]},
            {"pid": "2b1",  "name": "2B",        "score": 70.0, "positions": ["2B"]},
            {"pid": "3b1",  "name": "3B",        "score": 68.0, "positions": ["3B"]},
            {"pid": "1b1",  "name": "1B",        "score": 72.0, "positions": ["1B"]},
            {"pid": "of1",  "name": "OF1",       "score": 75.0, "positions": ["OF"]},
            {"pid": "of2",  "name": "OF2",       "score": 73.0, "positions": ["OF"]},
            {"pid": "of3",  "name": "OF3",       "score": 71.0, "positions": ["OF"]},
            {"pid": "dh1",  "name": "DH",        "score": 55.0, "positions": ["DH"]},
        ]
        solver = _greedy_solver()
        roster, scores, elig = _build_inputs(roster_cfg)
        result = solver.solve(roster, scores, elig)

        c_asgn = next((a for a in result.assignments if a.slot == PositionSlot.CATCHER), None)
        ss_asgn = next((a for a in result.assignments if a.slot == PositionSlot.SHORTSTOP), None)
        assert c_asgn is not None and c_asgn.player_id == "c_of", \
            "C/OF player fills C slot (only catcher on roster)"
        assert ss_asgn is not None and ss_asgn.player_id == "ss1", \
            "Dedicated SS fills SS slot independently"


# ─── LineupConstraintSolver.analyze_scarcity ──────────────────────────────


class TestAnalyzeScarcity:
    """Roster depth and scarcity detection."""

    def test_single_catcher_is_scarce(self):
        roster = [
            {"player_id": "c1", "name": "Catcher"},
            {"player_id": "1b1", "name": "First Base"},
        ]
        elig = {"c1": ["C"], "1b1": ["1B"]}
        solver = LineupConstraintSolver()
        result = solver.analyze_scarcity(roster, elig)

        assert result["position_depth"]["C"]["is_scarce"] is True
        assert result["position_depth"]["C"]["count"] == 1

    def test_zero_catchers_is_scarce(self):
        roster = [{"player_id": "1b1", "name": "First Base"}]
        elig = {"1b1": ["1B"]}
        solver = LineupConstraintSolver()
        result = solver.analyze_scarcity(roster, elig)

        assert result["position_depth"]["C"]["is_scarce"] is True
        assert result["position_depth"]["C"]["count"] == 0

    def test_two_catchers_not_scarce(self):
        roster = [
            {"player_id": "c1", "name": "Catcher1"},
            {"player_id": "c2", "name": "Catcher2"},
        ]
        elig = {"c1": ["C"], "c2": ["C"]}
        solver = LineupConstraintSolver()
        result = solver.analyze_scarcity(roster, elig)

        assert result["position_depth"]["C"]["is_scarce"] is False
        assert result["position_depth"]["C"]["count"] == 2

    def test_multi_eligible_catcher_counted_for_c_and_ss(self):
        # A C/SS player should increase depth for both positions
        roster = [{"player_id": "multi", "name": "C/SS player"}]
        elig = {"multi": ["C", "SS"]}
        solver = LineupConstraintSolver()
        result = solver.analyze_scarcity(roster, elig)

        assert result["position_depth"]["C"]["count"] == 1
        assert result["position_depth"]["SS"]["count"] == 1

    def test_scarcity_warnings_include_empty_positions(self):
        # Roster has no C or SS — both should appear in warnings
        roster = [{"player_id": "1b1", "name": "1B player"}]
        elig = {"1b1": ["1B"]}
        solver = LineupConstraintSolver()
        result = solver.analyze_scarcity(roster, elig)

        warnings = result["scarcity_warnings"]
        assert any("C" in w for w in warnings), "Scarcity warnings must include missing catcher"
        assert any("SS" in w for w in warnings), "Scarcity warnings must include missing SS"
