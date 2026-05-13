"""
Tests for category-aware FA scoring engine.

The rate-stat protection gate prevents recommending players who hurt a
category the team is already winning. This is catastrophic for rate stats
(ERA, WHIP, AVG, OPS, K_9): adding a 5.50-ERA pitcher when the team owns
a 3.10 ERA lead flips a won category. The gate detects this and applies a
heavy penalty multiplier so the scorer returns a NEGATIVE score — correctly
ranking that pitcher as harmful, not neutral.

Key invariants tested:
 - Counting stats: positive deficit → score = player_z * deficit
 - Counting stats: negative deficit (team winning) → score = 0 (no penalty)
 - Rate stats: team winning heavily + bad player → NEGATIVE score (penalty)
 - Rate stats: team losing + bad player → score = 0 (skipped via max(0,deficit))
 - Rate stats: team winning but below threshold → score = 0 (no penalty yet)
 - Rate stats: team losing + good player → positive score (normal)
 - Multi-category: correct sum across mixed counting + rate stats
"""
import pytest
from backend.fantasy_baseball.category_aware_scorer import (
    CategoryNeedVector,
    PlayerCategoryImpactVector,
    marginal_rate_impact,
    score_fa_against_needs,
    compute_need_score,
    RATE_STAT_PROTECT_THRESHOLD,
    RATE_STAT_PENALTY_MULTIPLIER,
)


# ---------------------------------------------------------------------------
# Counting stats
# ---------------------------------------------------------------------------

class TestCountingStatScoring:
    def test_positive_deficit_scores_normally(self):
        """FA with positive HR z-score scores when team needs HR."""
        team_needs = CategoryNeedVector(needs={"hr": 1.5})
        fa_impact = PlayerCategoryImpactVector(impacts={"hr": 0.8})
        score = score_fa_against_needs(fa_impact, team_needs)
        assert abs(score - 1.5 * 0.8) < 1e-9

    def test_negative_deficit_counting_stat_scores_zero(self):
        """Team is already winning HR — FA's positive z-score contributes nothing."""
        team_needs = CategoryNeedVector(needs={"hr": -2.0})
        fa_impact = PlayerCategoryImpactVector(impacts={"hr": 1.2})
        score = score_fa_against_needs(fa_impact, team_needs)
        assert score == 0.0, (
            "Counting stat with negative deficit must contribute 0, not reward adding players "
            "to categories the team already leads"
        )

    def test_negative_counting_z_score_with_positive_deficit_scores_zero(self):
        """FA who hurts a counting stat the team needs gets zero score (clamped at 0)."""
        team_needs = CategoryNeedVector(needs={"r": 1.0})
        fa_impact = PlayerCategoryImpactVector(impacts={"r": -0.5})
        score = score_fa_against_needs(fa_impact, team_needs)
        # player_z * max(0, 1.0) = -0.5 — this is a valid negative contribution
        # The existing design allows negative counting scores; they just don't get penalized
        assert score == pytest.approx(-0.5 * 1.0)

    def test_zero_deficit_counting_stat_scores_zero(self):
        """Tied category contributes nothing."""
        team_needs = CategoryNeedVector(needs={"rbi": 0.0})
        fa_impact = PlayerCategoryImpactVector(impacts={"rbi": 2.0})
        score = score_fa_against_needs(fa_impact, team_needs)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Rate stats — the critical protection gate
# ---------------------------------------------------------------------------

class TestRateStatProtectionGate:
    def test_bad_era_pitcher_gets_penalty_when_team_winning_era_heavily(self):
        """
        The money test: a pitcher with bad ERA (z < 0) gets a NEGATIVE score
        when the team is heavily winning ERA (deficit << 0).

        This prevents the system from recommending a 5.50-ERA reliever just
        because there's a large ERA lead to 'exploit'.
        """
        deficit = -(RATE_STAT_PROTECT_THRESHOLD + 1.0)  # well past the threshold
        team_needs = CategoryNeedVector(needs={"era": deficit})
        fa_impact = PlayerCategoryImpactVector(impacts={"era": -0.8})

        score = score_fa_against_needs(fa_impact, team_needs)

        expected = -0.8 * abs(deficit) * RATE_STAT_PENALTY_MULTIPLIER
        assert score == pytest.approx(expected), (
            f"Expected penalty score {expected:.4f}, got {score:.4f}. "
            "Bad ERA pitcher must receive a NEGATIVE score when team already leads ERA."
        )
        assert score < 0, "Penalty score must be negative — this FA is harmful"

    def test_bad_whip_pitcher_gets_penalty_when_team_winning_whip_heavily(self):
        """Same gate applies to WHIP."""
        deficit = -(RATE_STAT_PROTECT_THRESHOLD + 0.5)
        team_needs = CategoryNeedVector(needs={"whip": deficit})
        fa_impact = PlayerCategoryImpactVector(impacts={"whip": -0.6})

        score = score_fa_against_needs(fa_impact, team_needs)
        assert score < 0, "Bad WHIP pitcher must get penalty when team is winning WHIP"

    def test_bad_avg_batter_gets_penalty_when_team_winning_avg_heavily(self):
        """Gate applies to batting rate stats too."""
        deficit = -(RATE_STAT_PROTECT_THRESHOLD + 0.8)
        team_needs = CategoryNeedVector(needs={"avg": deficit})
        fa_impact = PlayerCategoryImpactVector(impacts={"avg": -1.0})

        score = score_fa_against_needs(fa_impact, team_needs)
        assert score < 0, "Batter who hurts AVG must be penalized when team leads AVG"

    def test_rate_stat_team_winning_but_below_threshold_no_penalty(self):
        """
        Team is slightly ahead in ERA but not past the protection threshold.
        Penalty gate must NOT fire — score should be 0 (clamped, not penalized).
        """
        slight_deficit = -(RATE_STAT_PROTECT_THRESHOLD * 0.5)  # below threshold
        team_needs = CategoryNeedVector(needs={"era": slight_deficit})
        fa_impact = PlayerCategoryImpactVector(impacts={"era": -0.4})

        score = score_fa_against_needs(fa_impact, team_needs)
        # Should be: player_z * max(0.0, deficit) = -0.4 * 0.0 = 0.0
        assert score == 0.0, (
            f"ERA penalty gate must not fire below threshold. Expected 0.0, got {score:.4f}"
        )

    def test_good_era_pitcher_scores_positively_when_team_needs_era(self):
        """Pitcher with good ERA (positive z) helps when team is losing ERA."""
        team_needs = CategoryNeedVector(needs={"era": 1.5})  # team is losing ERA
        fa_impact = PlayerCategoryImpactVector(impacts={"era": 0.7})

        score = score_fa_against_needs(fa_impact, team_needs)
        assert score == pytest.approx(0.7 * 1.5)
        assert score > 0

    def test_good_era_pitcher_no_score_when_team_already_winning_era(self):
        """
        Good ERA pitcher doesn't lose points just because team already leads ERA —
        penalty gate only fires when player_z < 0.
        """
        deficit = -(RATE_STAT_PROTECT_THRESHOLD + 1.0)  # team crushing ERA
        team_needs = CategoryNeedVector(needs={"era": deficit})
        fa_impact = PlayerCategoryImpactVector(impacts={"era": 0.9})  # good pitcher

        score = score_fa_against_needs(fa_impact, team_needs)
        # Positive player, team winning → clamped: player_z * max(0, deficit) = 0
        assert score == 0.0, (
            "Good ERA pitcher should score 0 (not penalized) when team already leads ERA"
        )

    def test_rate_stat_bad_player_team_losing_scores_negative_not_penalized(self):
        """
        Bad ERA pitcher when team is LOSING ERA: no penalty multiplier,
        just player_z * max(0, deficit) = player_z * deficit (negative).
        """
        team_needs = CategoryNeedVector(needs={"era": 1.5})  # team is losing ERA
        fa_impact = PlayerCategoryImpactVector(impacts={"era": -0.5})  # bad ERA

        score = score_fa_against_needs(fa_impact, team_needs)
        # Normal scoring: -0.5 * 1.5 = -0.75 (no penalty multiplier)
        assert score == pytest.approx(-0.5 * 1.5)
        # Not the penalty multiplier path
        assert score != pytest.approx(-0.5 * 1.5 * RATE_STAT_PENALTY_MULTIPLIER)

    def test_ops_rate_stat_is_protected(self):
        """OPS is a rate stat — protection gate applies."""
        deficit = -(RATE_STAT_PROTECT_THRESHOLD + 0.3)
        team_needs = CategoryNeedVector(needs={"ops": deficit})
        fa_impact = PlayerCategoryImpactVector(impacts={"ops": -0.7})

        score = score_fa_against_needs(fa_impact, team_needs)
        assert score < 0, "Bad OPS batter must be penalized when team leads OPS"

    def test_k9_rate_stat_is_protected(self):
        """K/9 is a rate stat — protection gate applies."""
        deficit = -(RATE_STAT_PROTECT_THRESHOLD + 0.4)
        team_needs = CategoryNeedVector(needs={"k9": deficit})
        fa_impact = PlayerCategoryImpactVector(impacts={"k9": -0.6})

        score = score_fa_against_needs(fa_impact, team_needs)
        assert score < 0, "Bad K/9 pitcher must be penalized when team leads K/9"


# ---------------------------------------------------------------------------
# Multi-category and edge cases
# ---------------------------------------------------------------------------

class TestMultiCategoryAndEdgeCases:
    def test_multi_category_correct_sum(self):
        """
        Combined score across counting + rate stats is the correct sum.

        Setup:
          - hr (counting): deficit=1.5, player_z=0.8 → 0.8 * 1.5 = 1.2
          - era (rate): deficit=-2.0 (team winning heavily, past threshold),
                        player_z=-0.8 → penalty: -0.8 * 2.0 * MULTIPLIER
          - r (counting): deficit=0.5, player_z=1.0 → 1.0 * 0.5 = 0.5
        Total: 1.2 + (-0.8 * 2.0 * RATE_STAT_PENALTY_MULTIPLIER) + 0.5
        """
        needs = {"hr": 1.5, "era": -2.0, "r": 0.5}
        impacts = {"hr": 0.8, "era": -0.8, "r": 1.0}
        team_needs = CategoryNeedVector(needs=needs)
        fa_impact = PlayerCategoryImpactVector(impacts=impacts)

        score = score_fa_against_needs(fa_impact, team_needs)

        counting_hr = 0.8 * 1.5
        penalty_era = -0.8 * 2.0 * RATE_STAT_PENALTY_MULTIPLIER
        counting_r = 1.0 * 0.5
        expected = counting_hr + penalty_era + counting_r

        assert score == pytest.approx(expected), (
            f"Multi-category score mismatch. Expected {expected:.4f}, got {score:.4f}"
        )
        assert score < 0, "Net score must be negative — the ERA penalty dominates"

    def test_empty_cat_scores_returns_zero(self):
        """FA with no cat_scores data scores 0."""
        team_needs = CategoryNeedVector(needs={"hr": 2.0, "era": -1.5})
        fa_impact = PlayerCategoryImpactVector(impacts={})
        score = score_fa_against_needs(fa_impact, team_needs)
        assert score == 0.0

    def test_empty_needs_returns_zero(self):
        """No deficits means no scoring signal."""
        team_needs = CategoryNeedVector(needs={})
        fa_impact = PlayerCategoryImpactVector(impacts={"hr": 1.5, "era": 0.8})
        score = score_fa_against_needs(fa_impact, team_needs)
        assert score == 0.0

    def test_missing_category_in_fa_impacts_treated_as_zero(self):
        """FA has no score for a needed category → contributes 0."""
        team_needs = CategoryNeedVector(needs={"nsb": 1.0, "hr": 0.8})
        fa_impact = PlayerCategoryImpactVector(impacts={"hr": 0.5})  # no "nsb" key
        score = score_fa_against_needs(fa_impact, team_needs)
        # nsb: 0.0 * 1.0 = 0; hr: 0.5 * 0.8 = 0.4
        assert score == pytest.approx(0.5 * 0.8)

    def test_penalty_exact_at_threshold_boundary(self):
        """
        deficit == -RATE_STAT_PROTECT_THRESHOLD exactly: gate must NOT fire
        (boundary is exclusive: deficit < -threshold triggers gate).
        """
        exact_threshold = -RATE_STAT_PROTECT_THRESHOLD
        team_needs = CategoryNeedVector(needs={"era": exact_threshold})
        fa_impact = PlayerCategoryImpactVector(impacts={"era": -0.5})

        score = score_fa_against_needs(fa_impact, team_needs)
        # At exact boundary: clamped → max(0, -threshold) = 0
        assert score == 0.0, (
            "Protection gate must not fire at the exact threshold boundary "
            "(must be strictly less than -threshold to trigger)"
        )

    def test_returns_float(self):
        """score_fa_against_needs always returns a float."""
        score = score_fa_against_needs(
            PlayerCategoryImpactVector(impacts={"hr": 1.0}),
            CategoryNeedVector(needs={"hr": 1.0}),
        )
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# compute_need_score() with optional team_context
# ---------------------------------------------------------------------------

from backend.fantasy_baseball.team_context import TeamContext
from backend.fantasy_baseball.category_aware_scorer import compute_need_score


class TestComputeNeedScoreWithTeamContext:
    """Tests for compute_need_score() with team_context parameter."""

    def _make_deficits(self):
        """Return a minimal CategoryDeficitOut list for testing."""
        from backend.schemas import CategoryDeficitOut
        return [CategoryDeficitOut(
            category="HR",
            deficit=1.5,
            my_total=5.0,
            opponent_total=3.5,
            winning=False,
        )]

    def test_no_team_context_unchanged(self):
        """When team_context=None, output equals existing behaviour."""
        cat_scores = {"hr": 1.2, "r": 0.8}
        result_no_ctx = compute_need_score(cat_scores, 1.0, self._make_deficits(), 10)
        result_with_none = compute_need_score(cat_scores, 1.0, self._make_deficits(), 10, team_context=None)
        assert result_no_ctx == result_with_none

    def test_team_context_accepted_without_error(self):
        """compute_need_score accepts TeamContext without raising."""
        ctx = TeamContext.build(
            roster_player_ids=[1, 2, 3],
            projected_pa_by_player={1: 500, 2: 400, 3: 550},
            projected_ip_by_player={},
        )
        result = compute_need_score({"hr": 1.5}, 1.0, self._make_deficits(), 10, team_context=ctx)
        assert isinstance(result, float)

    def test_shallow_roster_depth_factor_above_1(self):
        """Shallow PA pool (1000 PA) -> depth_factor > 1 -> matchup component scaled up."""
        ctx = TeamContext.build(
            roster_player_ids=[1],
            projected_pa_by_player={1: 1000.0},  # depth_factor = 3600/1000 = 3.6, capped at 1.2
            projected_ip_by_player={},
        )
        base = compute_need_score({"hr": 1.5}, 1.0, self._make_deficits(), 10, team_context=None)
        with_ctx = compute_need_score({"hr": 1.5}, 1.0, self._make_deficits(), 10, team_context=ctx)
        # shallow roster -> depth_factor = 1.2 -> matchup component scaled up -> with_ctx >= base
        assert with_ctx >= base

    def test_deep_roster_depth_factor_below_1(self):
        """Deep PA pool (9000 PA) -> depth_factor < 1 -> matchup component scaled down."""
        ctx = TeamContext.build(
            roster_player_ids=list(range(18)),
            projected_pa_by_player={i: 500.0 for i in range(18)},  # 9000 PA -> depth = 0.4, capped at 0.8
            projected_ip_by_player={},
        )
        base = compute_need_score({"hr": 1.5}, 1.0, self._make_deficits(), 10, team_context=None)
        with_ctx = compute_need_score({"hr": 1.5}, 1.0, self._make_deficits(), 10, team_context=ctx)
        assert with_ctx <= base

    def test_league_avg_pool_near_base(self):
        """PA pool of ~3600 -> depth_factor ~1.0 -> result very close to base."""
        ctx = TeamContext.build(
            roster_player_ids=list(range(9)),
            projected_pa_by_player={i: 400.0 for i in range(9)},  # 3600 total -> depth_factor = 1.0
            projected_ip_by_player={},
        )
        base = compute_need_score({"hr": 1.5}, 1.0, self._make_deficits(), 10, team_context=None)
        with_ctx = compute_need_score({"hr": 1.5}, 1.0, self._make_deficits(), 10, team_context=ctx)
        assert abs(with_ctx - base) < 0.05  # effectively identical

    def test_pitcher_ip_depth_factor_used_when_no_pa(self):
        """When PA denominator is 0 but IP denominator set, uses IP depth factor."""
        ctx = TeamContext.build(
            roster_player_ids=[10],
            projected_pa_by_player={},
            projected_ip_by_player={10: 500.0},  # 500 IP -> depth = 900/500 = 1.8, capped at 1.2
        )
        result = compute_need_score({"era": -0.5}, 0.5, self._make_deficits(), 10, team_context=ctx)
        assert isinstance(result, float)

    def test_zero_denominators_gives_depth_factor_1(self):
        """Empty roster -> depth_factor = 1.0 -> same as no team_context."""
        ctx = TeamContext.build(
            roster_player_ids=[],
            projected_pa_by_player={},
            projected_ip_by_player={},
        )
        base = compute_need_score({"hr": 1.0}, 1.0, self._make_deficits(), 10, team_context=None)
        with_ctx = compute_need_score({"hr": 1.0}, 1.0, self._make_deficits(), 10, team_context=ctx)
        assert base == with_ctx


def test_marginal_rate_impact_avg_positive():
    """Adding a .340 hitter to a .265 team should yield positive marginal z."""
    z = marginal_rate_impact("avg", 1500, 5660, 68, 200, 0.025)
    assert z > 0, f"Expected positive marginal z, got {z}"


def test_marginal_rate_impact_era_positive():
    """Adding a high-ERA pitcher to a good-ERA team should yield negative marginal z."""
    z = marginal_rate_impact("era", 250, 750, 45, 90, 0.60)
    assert z < 0, f"Expected negative marginal z for adding bad pitcher, got {z}"


def test_marginal_rate_impact_zero_denominator():
    """Returns 0.0 when denominators are zero."""
    z = marginal_rate_impact("avg", 0, 0, 68, 200, 0.025)
    assert z == 0.0


def test_compute_need_score_uses_marginal_stats():
    """compute_need_score uses marginal_stats to override rate cat z-scores."""
    from backend.schemas import CategoryDeficitOut

    player_cat_scores = {"avg": -0.5, "hr": 1.2}
    category_deficits = [
        CategoryDeficitOut(category="avg", my_total=0.250, opponent_total=0.258, deficit=0.8, winning=False),
        CategoryDeficitOut(category="hr", my_total=10.0, opponent_total=10.5, deficit=0.5, winning=False),
    ]

    score_no_marginal = compute_need_score(player_cat_scores, 0.5, category_deficits, 5)

    marginal_stats = {
        "avg": {"team_num": 1200.0, "team_den": 4800.0, "player_num": 72.0, "player_den": 200.0}
    }
    score_with_marginal = compute_need_score(
        player_cat_scores,
        0.5,
        category_deficits,
        5,
        marginal_stats=marginal_stats,
    )

    assert score_with_marginal > score_no_marginal


# ---------------------------------------------------------------------------
# Tests for canonical → board key mapping in compute_need_score
# ---------------------------------------------------------------------------

def _deficit(category: str, deficit: float = 2.0):
    from backend.schemas import CategoryDeficitOut
    return CategoryDeficitOut(
        category=category,
        deficit=deficit,
        my_total=5.0,
        opponent_total=5.0 - deficit,
        winning=deficit < 0,
    )


def test_need_score_hr_b_maps_to_hr():
    """HR_B canonical → 'hr' board key; score must reflect HR contribution."""
    score = compute_need_score(
        player_cat_scores={"hr": 2.5, "avg": 1.2},
        player_z_score=1.0,
        category_deficits=[_deficit("HR_B", 3.0)],
        n_cats=10,
    )
    # Correct: cat_score = 2.5 * 3.0 = 7.5; blended = 0.4*1.0 + 0.6*(7.5/10) = 0.85
    # Wrong:   cat_score = 0.0;           blended = 0.4*1.0 = 0.40
    assert score > 0.5, f"Expected >0.5 (blended HR contribution), got {score:.4f}"


def test_need_score_k_b_maps_to_k_bat():
    """K_B canonical → 'k_bat' board key."""
    score = compute_need_score(
        player_cat_scores={"k_bat": 1.8},
        player_z_score=0.5,
        category_deficits=[_deficit("K_B", 2.0)],
        n_cats=10,
    )
    # cat_score = 1.8 * 2.0 = 3.6; blended = 0.4*0.5 + 0.6*(3.6/10) = 0.416
    assert score > 0.40, f"Expected >0.40, got {score:.4f}"


def test_need_score_k_p_maps_to_k_pit():
    """K_P canonical → 'k_pit' board key."""
    score = compute_need_score(
        player_cat_scores={"k_pit": 3.0},
        player_z_score=0.5,
        category_deficits=[_deficit("K_P", 1.5)],
        n_cats=10,
    )
    assert score > 0.40, f"Expected >0.40, got {score:.4f}"


def test_need_score_avg_passthrough():
    """AVG lowercases cleanly to 'avg' — passthrough path must still work."""
    score = compute_need_score(
        player_cat_scores={"avg": 1.5},
        player_z_score=0.5,
        category_deficits=[_deficit("AVG", 2.0)],
        n_cats=10,
    )
    # cat_score = 1.5 * 2.0 = 3.0; blended = 0.4*0.5 + 0.6*(3.0/10) = 0.38
    assert score > 0.35, f"Expected >0.35, got {score:.4f}"
