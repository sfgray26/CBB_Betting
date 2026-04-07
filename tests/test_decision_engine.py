"""
Tests for backend/services/decision_engine.py (P17 Decision Engine).

Pure-function tests only -- no DB, no mocks, no I/O.
All test data is constructed inline.
"""

import math
import pytest
from datetime import date

from backend.services.decision_engine import (
    PlayerDecisionInput,
    DecisionResult,
    LineupDecision,
    WaiverDecision,
    ROSTER_SLOTS,
    optimize_lineup,
    optimize_waivers,
    _lineup_score,
    _momentum_bonus,
    _proj_bonus,
    _composite_value,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_hitter(
    pid: int,
    name: str = "Player",
    positions: list = None,
    score: float = 50.0,
    momentum: str = "STABLE",
    proj_hr=15.0,
    proj_rbi=60.0,
    proj_sb=5.0,
    delta_z: float = 0.0,
) -> PlayerDecisionInput:
    return PlayerDecisionInput(
        bdl_player_id=pid,
        name=name,
        player_type="hitter",
        eligible_positions=positions or ["1B", "OF"],
        score_0_100=score,
        composite_z=score / 50.0 - 1.0,
        momentum_signal=momentum,
        delta_z=delta_z,
        proj_hr_p50=proj_hr,
        proj_rbi_p50=proj_rbi,
        proj_sb_p50=proj_sb,
        proj_avg_p50=0.270,
        proj_k_p50=None,
        proj_era_p50=None,
        proj_whip_p50=None,
        downside_p25=None,
        upside_p75=None,
    )


def _make_pitcher(
    pid: int,
    name: str = "Pitcher",
    positions: list = None,
    score: float = 50.0,
    momentum: str = "STABLE",
    proj_k: float = 150.0,
    proj_era: float = 3.50,
    proj_whip: float = 1.15,
    delta_z: float = 0.0,
) -> PlayerDecisionInput:
    return PlayerDecisionInput(
        bdl_player_id=pid,
        name=name,
        player_type="pitcher",
        eligible_positions=positions or ["SP"],
        score_0_100=score,
        composite_z=score / 50.0 - 1.0,
        momentum_signal=momentum,
        delta_z=delta_z,
        proj_hr_p50=None,
        proj_rbi_p50=None,
        proj_sb_p50=None,
        proj_avg_p50=None,
        proj_k_p50=proj_k,
        proj_era_p50=proj_era,
        proj_whip_p50=proj_whip,
        downside_p25=None,
        upside_p75=None,
    )


def _make_full_roster() -> list:
    """
    14-player roster with enough positional coverage to fill all active slots.
    """
    players = [
        _make_hitter(1,  "Catcher",    positions=["C"],        score=60.0),
        _make_hitter(2,  "FirstBase",  positions=["1B"],       score=70.0),
        _make_hitter(3,  "SecondBase", positions=["2B"],       score=55.0),
        _make_hitter(4,  "ThirdBase",  positions=["3B"],       score=65.0),
        _make_hitter(5,  "ShortStop",  positions=["SS"],       score=62.0),
        _make_hitter(6,  "Outfield1",  positions=["OF"],       score=80.0),
        _make_hitter(7,  "Outfield2",  positions=["OF"],       score=75.0),
        _make_hitter(8,  "Outfield3",  positions=["OF"],       score=68.0),
        _make_hitter(9,  "UtilPlayer", positions=["1B", "OF"], score=58.0),
        _make_pitcher(10, "Starter1",  positions=["SP"],       score=72.0),
        _make_pitcher(11, "Starter2",  positions=["SP"],       score=69.0),
        _make_pitcher(12, "Reliever1", positions=["RP"],       score=60.0),
        _make_pitcher(13, "Reliever2", positions=["RP"],       score=57.0),
        _make_pitcher(14, "AnyPitch",  positions=["SP", "RP"], score=55.0),
    ]
    return players


# ---------------------------------------------------------------------------
# 1. Lineup optimizer fills all required slots
# ---------------------------------------------------------------------------

class TestLineupOptimizerFillsAllSlots:
    def test_fills_all_non_bench_slots(self):
        players = _make_full_roster()
        lineup, results = optimize_lineup(players, date(2026, 4, 5))

        # Verify every required slot (except BN) is represented
        filled_base_slots = {k.split("_")[0] for k in lineup.selected}
        required = {slot for slot in ROSTER_SLOTS if slot != "BN"}
        for slot in required:
            assert slot in filled_base_slots, f"Slot {slot} was not filled"

    def test_bench_capped_at_five(self):
        players = _make_full_roster()
        lineup, _ = optimize_lineup(players, date(2026, 4, 5))
        assert len(lineup.bench) <= ROSTER_SLOTS["BN"]

    def test_no_player_placed_twice(self):
        players = _make_full_roster()
        lineup, _ = optimize_lineup(players, date(2026, 4, 5))
        all_placed = list(lineup.selected.values()) + lineup.bench
        assert len(all_placed) == len(set(all_placed)), "Duplicate placement detected"

    def test_returns_decision_result_rows(self):
        players = _make_full_roster()
        _, results = optimize_lineup(players, date(2026, 4, 5))
        assert len(results) > 0
        for r in results:
            assert isinstance(r, DecisionResult)
            assert r.decision_type == "lineup"
            assert r.as_of_date == date(2026, 4, 5)
            assert r.target_slot is not None
            assert 0.0 <= r.confidence <= 1.0


# ---------------------------------------------------------------------------
# 2. Momentum bonus applied correctly
# ---------------------------------------------------------------------------

class TestMomentumBonusApplied:
    def test_surging_scores_higher_than_stable(self):
        surging = _make_hitter(101, momentum="SURGING", score=50.0)
        stable  = _make_hitter(102, momentum="STABLE",  score=50.0)
        assert _lineup_score(surging) > _lineup_score(stable)

    def test_collapsing_scores_lower_than_stable(self):
        collapsing = _make_hitter(103, momentum="COLLAPSING", score=50.0)
        stable     = _make_hitter(104, momentum="STABLE",     score=50.0)
        assert _lineup_score(collapsing) < _lineup_score(stable)

    def test_momentum_bonus_values(self):
        assert _momentum_bonus("SURGING")    ==  10.0
        assert _momentum_bonus("HOT")        ==   5.0
        assert _momentum_bonus("STABLE")     ==   0.0
        assert _momentum_bonus("COLD")       ==  -5.0
        assert _momentum_bonus("COLLAPSING") == -10.0

    def test_unknown_momentum_defaults_to_zero(self):
        assert _momentum_bonus("UNKNOWN_SIGNAL") == 0.0
        assert _momentum_bonus("") == 0.0

    def test_surging_preferred_in_lineup(self):
        surging = _make_hitter(105, positions=["OF"], momentum="SURGING", score=60.0)
        cold    = _make_hitter(106, positions=["OF"], momentum="COLD",    score=60.0)
        lineup, _ = optimize_lineup([surging, cold], date(2026, 4, 5))
        # OF slot should be filled with the surging player first
        of_key = next((k for k in lineup.selected if k.startswith("OF")), None)
        if of_key:
            assert lineup.selected[of_key] == surging.bdl_player_id


# ---------------------------------------------------------------------------
# 3. Waiver decision picks best add
# ---------------------------------------------------------------------------

class TestWaiverDecisionPicksBestAdd:
    def _make_weak_roster(self) -> list:
        return [
            _make_hitter(201, "WeakHitter1", positions=["OF"], score=20.0, proj_hr=2.0, proj_rbi=10.0),
            _make_hitter(202, "WeakHitter2", positions=["OF"], score=15.0, proj_hr=1.0, proj_rbi=8.0),
        ]

    def test_strong_waiver_candidate_generates_positive_gain(self):
        roster = self._make_weak_roster()
        strong_waiver = _make_hitter(
            301, "StrongAdd", positions=["OF"],
            score=85.0, proj_hr=28.0, proj_rbi=90.0, momentum="SURGING"
        )
        _, results = optimize_waivers(roster, [strong_waiver], date(2026, 4, 5))
        assert len(results) == 1
        assert results[0].value_gain > 0.0

    def test_waiver_result_structure(self):
        roster = self._make_weak_roster()
        candidate = _make_hitter(302, "CandidateA", positions=["1B"], score=70.0)
        _, results = optimize_waivers(roster, [candidate], date(2026, 4, 5))
        assert len(results) == 1
        r = results[0]
        assert r.decision_type == "waiver"
        assert r.bdl_player_id == candidate.bdl_player_id
        assert r.drop_player_id is not None
        assert 0.0 <= r.confidence <= 1.0

    def test_best_candidate_sorted_first(self):
        roster = self._make_weak_roster()
        weak_add   = _make_hitter(303, "WeakAdd",   positions=["OF"], score=30.0,  proj_hr=5.0, proj_rbi=20.0)
        strong_add = _make_hitter(304, "StrongAdd2", positions=["OF"], score=90.0, proj_hr=30.0, proj_rbi=100.0)
        waiver_decision, results = optimize_waivers(roster, [weak_add, strong_add], date(2026, 4, 5))
        assert results[0].bdl_player_id == strong_add.bdl_player_id

    def test_multiple_candidates_all_returned(self):
        roster = self._make_weak_roster()
        pool = [_make_hitter(400 + i, positions=["OF"], score=50.0 + i) for i in range(5)]
        _, results = optimize_waivers(roster, pool, date(2026, 4, 5))
        assert len(results) == 5


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_roster_returns_empty_lineup(self):
        lineup, results = optimize_lineup([], date(2026, 4, 5))
        assert isinstance(lineup, LineupDecision)
        assert lineup.selected == {}
        assert lineup.bench == []
        assert results == []

    def test_empty_waiver_pool_returns_empty_decisions(self):
        roster = [_make_hitter(501, positions=["OF"], score=50.0)]
        waiver, results = optimize_waivers(roster, [], date(2026, 4, 5))
        assert isinstance(waiver, WaiverDecision)
        assert results == []

    def test_single_player_roster_no_crash(self):
        player = _make_hitter(601, positions=["C"], score=75.0)
        lineup, results = optimize_lineup([player], date(2026, 4, 5))
        assert "C" in lineup.selected
        assert lineup.selected["C"] == player.bdl_player_id

    def test_proj_bonus_zero_for_missing_projections(self):
        # Build a PlayerDecisionInput directly with None proj fields
        player = PlayerDecisionInput(
            bdl_player_id=701,
            name="NullProj",
            player_type="hitter",
            eligible_positions=["OF"],
            score_0_100=50.0,
            composite_z=0.0,
            momentum_signal="STABLE",
            delta_z=0.0,
            proj_hr_p50=None,
            proj_rbi_p50=None,
            proj_sb_p50=None,
            proj_avg_p50=None,
            proj_k_p50=None,
            proj_era_p50=None,
            proj_whip_p50=None,
            downside_p25=None,
            upside_p75=None,
        )
        bonus = _proj_bonus(player)
        assert bonus == 0.0

    def test_lineup_score_bounded(self):
        """lineup_score should not produce NaN or extreme values for edge inputs."""
        player = _make_hitter(801, score=100.0, momentum="SURGING", proj_hr=30.0, proj_rbi=100.0)
        score = _lineup_score(player)
        assert not math.isnan(score)
        assert not math.isinf(score)
        assert score >= 0.0

    def test_pitcher_composite_value_uses_k(self):
        elite = _make_pitcher(901, proj_k=200.0, proj_era=2.50, proj_whip=1.00)
        poor  = _make_pitcher(902, proj_k=50.0,  proj_era=6.00, proj_whip=1.80)
        assert _composite_value(elite) > _composite_value(poor)
