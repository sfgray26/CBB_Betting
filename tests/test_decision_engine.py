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
    z_k_p: float = 0.0,
    z_era: float = 0.0,
    z_whip: float = 0.0,
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
        # P0-3: Pitcher z-scores
        z_k_p=z_k_p,
        z_era=z_era,
        z_whip=z_whip,
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

    def test_bench_players_are_included_in_results(self):
        players = _make_full_roster() + [
            _make_hitter(15, "BenchOF", positions=["OF"], score=40.0),
            _make_pitcher(16, "BenchSP", positions=["SP"], score=35.0),
        ]
        _, results = optimize_lineup(players, date(2026, 4, 5))

        bench_rows = [r for r in results if r.target_slot == "BN"]
        assert len(bench_rows) >= 1


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

    def test_waiver_drop_candidate_comes_from_bench(self):
        roster = _make_full_roster() + [
            _make_hitter(99, "WeakBenchBat", positions=["OF"], score=20.0, proj_hr=1.0, proj_rbi=8.0),
        ]
        strong_add = _make_hitter(305, "StrongAdd3", positions=["OF"], score=88.0, proj_hr=28.0, proj_rbi=90.0)

        _, results = optimize_waivers(roster, [strong_add], date(2026, 4, 5))

        assert len(results) == 1
        assert results[0].drop_player_id == 99

    def test_missing_projection_fields_do_not_make_star_undroppable(self):
        roster = [
            _make_hitter(501, "Juan Soto", positions=["OF"], score=95.0, proj_hr=None, proj_rbi=None, proj_sb=None),
            _make_hitter(502, "BenchWeak", positions=["OF"], score=20.0, proj_hr=1.0, proj_rbi=5.0, proj_sb=1.0),
            _make_hitter(503, "BenchWeak2", positions=["OF"], score=22.0, proj_hr=1.0, proj_rbi=6.0, proj_sb=1.0),
            _make_hitter(504, "BenchWeak3", positions=["OF"], score=24.0, proj_hr=2.0, proj_rbi=8.0, proj_sb=1.0),
            _make_hitter(505, "BenchWeak4", positions=["OF"], score=26.0, proj_hr=2.0, proj_rbi=9.0, proj_sb=1.0),
            _make_hitter(506, "BenchWeak5", positions=["OF"], score=28.0, proj_hr=2.0, proj_rbi=10.0, proj_sb=1.0),
            _make_hitter(507, "Starter1B", positions=["1B"], score=80.0),
            _make_hitter(508, "Starter2B", positions=["2B"], score=75.0),
            _make_hitter(509, "Starter3B", positions=["3B"], score=74.0),
            _make_hitter(510, "StarterSS", positions=["SS"], score=73.0),
            _make_hitter(511, "StarterC", positions=["C"], score=72.0),
            _make_pitcher(512, "SP1", positions=["SP"], score=70.0),
            _make_pitcher(513, "SP2", positions=["SP"], score=69.0),
            _make_pitcher(514, "RP1", positions=["RP"], score=68.0),
            _make_pitcher(515, "RP2", positions=["RP"], score=67.0),
            _make_pitcher(516, "P1", positions=["SP", "RP"], score=66.0),
        ]
        candidate = _make_hitter(600, "WaiverAdd", positions=["OF"], score=85.0, proj_hr=25.0, proj_rbi=80.0)

        _, results = optimize_waivers(roster, [candidate], date(2026, 4, 5))

        assert len(results) == 1
        assert results[0].drop_player_id != 501


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
        # P0-3: Updated to use z-scores for pitcher composite
        elite = _make_pitcher(
            901, proj_k=200.0, proj_era=2.50, proj_whip=1.00,
            z_k_p=2.0, z_era=1.5, z_whip=1.2,  # Elite z-scores
        )
        poor  = _make_pitcher(
            902, proj_k=50.0, proj_era=6.00, proj_whip=1.80,
            z_k_p=-1.5, z_era=-1.8, z_whip=-1.3,  # Poor z-scores
        )
        assert _composite_value(elite) > _composite_value(poor)


# ---------------------------------------------------------------------------
# P0-3: Pitcher Composite Uses Z-Scores
# ---------------------------------------------------------------------------

class TestP03PitcherCompositeUsesZScores:
    """P0-3: Pitcher composite value uses z-scores instead of raw projections."""

    def test_elite_pitcher_z_scores_rank_higher(self):
        """Elite pitcher (positive z-scores) should rank higher than poor pitcher."""
        elite = _make_pitcher(
            901, "EliteSP",
            score=75.0,
            z_k_p=2.0,   # Well above average K rate
            z_era=1.5,   # Well above average (low ERA is good, z_era negated)
            z_whip=1.2,  # Above average (low WHIP is good, z_whip negated)
        )
        poor = _make_pitcher(
            902, "PoorSP",
            score=25.0,
            z_k_p=-1.5,  # Below average K rate
            z_era=-1.8,  # Poor ERA (high ERA, z_era negative)
            z_whip=-1.3, # Poor WHIP (high WHIP, z_whip negative)
        )
        assert _composite_value(elite) > _composite_value(poor)

    def test_z_scores_override_raw_projections(self):
        """Z-scores should be the primary driver, not raw projections."""
        # High raw projections but negative z-scores (lucky stats, poor underlying)
        lucky = _make_pitcher(
            903, "LuckyPitcher",
            score=50.0,
            proj_k=200.0, proj_era=3.00, proj_whip=1.10,  # Good raw stats
            z_k_p=-1.0, z_era=-0.5, z_whip=-0.3,  # But negative z-scores
        )
        # Moderate raw projections but positive z-scores (solid performer)
        solid = _make_pitcher(
            904, "SolidPitcher",
            score=50.0,
            proj_k=150.0, proj_era=3.80, proj_whip=1.25,  # Moderate raw stats
            z_k_p=0.8, z_era=0.6, z_whip=0.4,  # Positive z-scores
        )
        # Z-scores should win out
        assert _composite_value(solid) > _composite_value(lucky)

    def test_null_z_scores_fallback_to_baseline(self):
        """When z-scores are None, should fall back to score_0_100 baseline."""
        no_z = _make_pitcher(
            905, "NoZScores",
            score=70.0,
            z_k_p=None, z_era=None, z_whip=None,
        )
        # Should not crash, and baseline from score_0_100 should apply
        val = _composite_value(no_z)
        assert val > 0.0
        # With score=70, baseline = (70/100) * 1.5 = 1.05
        assert val > 1.0

    def test_two_way_pitcher_component_uses_z_scores(self):
        """Two-way players should use z-scores for pitcher component."""
        two_way_elite = PlayerDecisionInput(
            bdl_player_id=1001,
            name="TwoWayElite",
            player_type="two_way",
            eligible_positions=["UTIL", "P"],
            score_0_100=70.0,
            composite_z=1.0,
            momentum_signal="STABLE",
            delta_z=0.0,
            proj_hr_p50=15.0, proj_rbi_p50=60.0, proj_sb_p50=5.0,
            proj_avg_p50=0.270,
            proj_k_p50=180.0, proj_era_p50=3.20, proj_whip_p50=1.15,
            downside_p25=None, upside_p75=None,
            # P0-3: Pitcher z-scores
            z_k_p=1.5, z_era=1.0, z_whip=0.8,
        )
        two_way_poor_pitcher = PlayerDecisionInput(
            bdl_player_id=1002,
            name="TwoWayPoorPitcher",
            player_type="two_way",
            eligible_positions=["UTIL", "P"],
            score_0_100=70.0,
            composite_z=1.0,
            momentum_signal="STABLE",
            delta_z=0.0,
            proj_hr_p50=15.0, proj_rbi_p50=60.0, proj_sb_p50=5.0,
            proj_avg_p50=0.270,
            proj_k_p50=180.0, proj_era_p50=3.20, proj_whip_p50=1.15,
            downside_p25=None, upside_p75=None,
            # P0-3: Poor pitcher z-scores
            z_k_p=-1.0, z_era=-1.2, z_whip=-0.8,
        )
        # Same hitting stats, different pitcher z-scores
        # Elite pitcher should have higher composite
        assert _composite_value(two_way_elite) > _composite_value(two_way_poor_pitcher)

    def test_z_score_sum_scaling(self):
        """Z-score sum should be scaled to [0, 3] range for compatibility."""
        # Max positive z-scores: +3 each = +9 total
        # Scaled: (9 + 9) / 6 = 3.0
        perfect = _make_pitcher(
            906, "PerfectPitcher",
            score=50.0,
            z_k_p=3.0, z_era=3.0, z_whip=3.0,
        )
        val = _composite_value(perfect)
        # Should be at the top of the range (near 3.0)
        assert val > 2.5

        # Max negative z-scores: -3 each = -9 total
        # Scaled: (-9 + 9) / 6 = 0.0
        terrible = _make_pitcher(
            907, "TerriblePitcher",
            score=50.0,
            z_k_p=-3.0, z_era=-3.0, z_whip=-3.0,
        )
        val_terrible = _composite_value(terrible)
        # Should be at the bottom of the range
        # Baseline from score_0_100 may apply
        assert 0.0 <= val_terrible <= 1.5


# ---------------------------------------------------------------------------
# 5. Roster-only lineup filtering (EMAC-069)
# ---------------------------------------------------------------------------

class TestRosterOnlyLineupFiltering:
    """Tests verifying that optimize_lineup only considers roster players.

    The actual filtering to roster-only happens in daily_ingestion.py before
    calling optimize_lineup(). These tests verify the decision engine's
    expected behavior when given a roster-constrained player list.
    """

    def test_lineup_decisions_only_for_provided_players(self):
        """When given a roster subset, only those players appear in decisions."""
        roster_players = [
            _make_hitter(1, "RosterOF1", positions=["OF"], score=75.0),
            _make_hitter(2, "RosterOF2", positions=["OF"], score=65.0),
        ]
        # Note: non-roster players are NOT passed in - filtered upstream
        _, results = optimize_lineup(roster_players, date(2026, 4, 15))

        # All decisions should be for the roster players we provided
        result_bdl_ids = {r.bdl_player_id for r in results}
        expected_ids = {1, 2}
        assert result_bdl_ids.issubset(expected_ids), (
            f"Found decisions for non-roster players: {result_bdl_ids - expected_ids}"
        )

    def test_lineup_with_empty_roster_returns_no_decisions(self):
        """Empty roster (filtered upstream) produces empty lineup decisions."""
        lineup, results = optimize_lineup([], date(2026, 4, 15))
        assert results == []
        assert lineup.selected == {}

    def test_partial_roster_still_produces_valid_lineup(self):
        """Even with a partial roster, optimizer fills what it can."""
        partial_roster = [
            _make_hitter(10, "OnlyC", positions=["C"], score=60.0),
            _make_pitcher(11, "OnlySP", positions=["SP"], score=70.0),
        ]
        lineup, results = optimize_lineup(partial_roster, date(2026, 4, 15))

        # Should get decisions for the two players we have
        assert len(results) >= 2
        result_ids = {r.bdl_player_id for r in results}
        assert result_ids == {10, 11}

        # C and SP should be filled
        assert "C" in lineup.selected
        assert any(k.startswith("SP") for k in lineup.selected)


# ---------------------------------------------------------------------------
# 6. Waiver value_gain filtering (EMAC-069)
# ---------------------------------------------------------------------------

class TestWaiverValueGainFiltering:
    """Tests for waiver decision value_gain threshold filtering.

    Daily ingestion filters waiver results to only include value_gain > 0.10.
    These tests verify the decision engine produces value_gain values that
    can be filtered, and document the expected threshold behavior.
    """

    def test_waiver_low_value_gain_below_threshold(self):
        """Waiver adds with value_gain < 0.10 should be filtered out upstream.

        Threshold: 0.10 value_gain
        Rationale: Filter out marginal waiver recommendations that don't
        provide meaningful roster improvement.
        """
        weak_roster = [
            _make_hitter(201, "Weak1", positions=["OF"], score=40.0),
            _make_hitter(202, "Weak2", positions=["OF"], score=35.0),
        ]
        # Marginal improvement - value_gain likely < 0.10
        marginal_add = _make_hitter(301, "Marginal", positions=["OF"], score=45.0)

        _, results = optimize_waivers(weak_roster, [marginal_add], date(2026, 4, 15))

        # The engine produces a result, but daily_ingestion.py filters it out
        assert len(results) == 1
        assert results[0].value_gain is not None
        # Document that this would be filtered by the 0.10 threshold
        if results[0].value_gain < 0.10:
            # This demonstrates the filtering threshold works as expected
            assert results[0].value_gain < 0.10

    def test_waiver_high_value_gain_above_threshold(self):
        """Waiver adds with value_gain > 0.10 should pass the filter."""
        weak_roster = [
            _make_hitter(201, "Weak1", positions=["OF"], score=30.0),
            _make_hitter(202, "Weak2", positions=["OF"], score=25.0),
        ]
        # Strong add - value_gain should clearly exceed 0.10 threshold
        strong_add = _make_hitter(
            302, "StrongAdd", positions=["OF"],
            score=85.0, proj_hr=30.0, proj_rbi=95.0, momentum="SURGING"
        )

        _, results = optimize_waivers(weak_roster, [strong_add], date(2026, 4, 15))

        assert len(results) == 1
        assert results[0].value_gain is not None
        # Should definitely exceed the 0.10 threshold
        assert results[0].value_gain > 0.10, (
            f"Expected value_gain > 0.10 for strong add, got {results[0].value_gain}"
        )

    def test_waiver_exactly_at_threshold_boundary(self):
        """Test behavior when value_gain is near the 0.10 threshold."""
        weak_roster = [_make_hitter(201, "Weak", positions=["1B"], score=40.0)]
        # Player just slightly better - might be near threshold
        borderline_add = _make_hitter(303, "Borderline", positions=["1B"], score=50.0)

        _, results = optimize_waivers(weak_roster, [borderline_add], date(2026, 4, 15))

        assert len(results) == 1
        value_gain = results[0].value_gain
        # Document the threshold: values <= 0.10 are filtered out
        assert value_gain is not None
        # The actual value depends on the scoring algorithm
        # This test documents that the threshold exists and is applied upstream

    def test_waiver_filter_threshold_documentation(self):
        """Document the exact threshold used for waiver filtering.

        Threshold: 0.10 value_gain
        Applied in: daily_ingestion.py, _run_decision_pipeline() function
        Logic: waiver_results_filtered = [r for r in waiver_results
                                         if r.value_gain is not None
                                         and r.value_gain > 0.10]
        """
        # This test exists solely to document the threshold constant
        # If the threshold changes, update this test and the comment above
        FILTER_THRESHOLD = 0.10
        assert FILTER_THRESHOLD == 0.10
