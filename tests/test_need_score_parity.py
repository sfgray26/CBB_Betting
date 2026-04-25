"""
Parity test for need_score computation across waiver and recommendations endpoints.

Phase 8 Step 5 (P1C): Both endpoints must return identical need_score values
for the same player given the same league state.

This test directly calls compute_need_score() and verifies it produces
consistent results, then validates the sort contract (float return type).
"""
import pytest
from backend.schemas import CategoryDeficitOut
from backend.fantasy_baseball.category_aware_scorer import (
    compute_need_score,
    CategoryNeedVector,
    PlayerCategoryImpactVector,
    score_fa_against_needs,
)


class TestComputeNeedScore:
    """Tests for the unified compute_need_score helper."""

    def test_returns_float_for_valid_input(self):
        """Need score must be a plain float for sort contract compatibility."""
        cat_scores = {"r": 1.2, "h": 0.8, "hr": 1.5, "rbi": 0.9, "k_bat": -0.3,
                      "tb": 1.0, "avg": 0.5, "ops": 0.7, "nsb": 0.2}
        z_score = 0.85

        # Create category deficits (team losing in R, HR, RBI)
        deficits = [
            CategoryDeficitOut(category="R", my_total=40.0, opponent_total=50.0, deficit=10.0, winning=False),
            CategoryDeficitOut(category="HR", my_total=20.0, opponent_total=25.0, deficit=5.0, winning=False),
            CategoryDeficitOut(category="RBI", my_total=37.0, opponent_total=45.0, deficit=8.0, winning=False),
        ]

        result = compute_need_score(cat_scores, z_score, deficits, n_cats=9)

        assert isinstance(result, float), "compute_need_score must return plain float"
        assert not isinstance(result, tuple), "Must not return tuple (sort contract)"
        # Should blend player_z with matchup-specific score
        assert result > 0, "Player with positive z-scores in needed categories should have positive need_score"

    def test_fallback_to_z_score_when_no_deficits(self):
        """Empty category_deficits list should return plain z_score."""
        cat_scores = {"r": 1.0, "h": 0.5, "hr": 1.2}
        z_score = 0.75

        result = compute_need_score(cat_scores, z_score, category_deficits=[], n_cats=9)

        assert result == 0.75, "Empty deficits should fallback to z_score"

    def test_fallback_to_z_score_when_deficits_is_none(self):
        """None category_deficits should return plain z_score."""
        cat_scores = {"r": 1.0}
        z_score = 0.5

        result = compute_need_score(cat_scores, z_score, category_deficits=None, n_cats=9)

        assert result == 0.5, "None deficits should fallback to z_score"

    def test_fallback_on_exception_in_scorer(self):
        """Empty valid cat_scores should fallback to z_score."""
        # cat_scores with only non-numeric values should be treated as empty
        cat_scores = {"invalid_cat": "not_a_number"}
        z_score = 0.3

        deficits = [
            CategoryDeficitOut(category="R", my_total=45.0, opponent_total=50.0, deficit=5.0, winning=False),
        ]

        # With no valid numeric cat_scores, should fallback to z_score
        result = compute_need_score(cat_scores, z_score, deficits, n_cats=9)

        assert result == z_score, "No valid numeric cat_scores should fallback to z_score"

    def test_rate_stat_protection_applied(self):
        """Player with negative rate-stats when team is winning should be penalized."""
        # Team winning ERA category (negative deficit)
        cat_scores = {"r": 0.5, "h": 0.3, "era": -1.2, "whip": -0.8, "avg": 0.2, "ops": 0.1,
                      "hr": 0.4, "rbi": 0.3, "k_bat": 0.6, "tb": 0.5, "k9": 0.7, "nsb": 0.0}
        z_score = 0.4

        # Team is winning ERA by 0.8 z-score units (deficit < -0.5 threshold)
        # This should trigger rate-stat protection penalty
        deficits = [
            CategoryDeficitOut(category="ERA", my_total=3.70, opponent_total=4.50, deficit=-0.8, winning=True),
        ]

        result = compute_need_score(cat_scores, z_score, deficits, n_cats=9)

        # Result should be lower than plain z_score due to penalty
        assert result < z_score, "Rate-stat penalty should reduce need_score"

    def test_positive_z_scores_in_needed_categories_increase_score(self):
        """Player helping categories team is losing should get boost."""
        cat_scores = {"r": 2.0, "h": 1.5, "hr": 1.8, "rbi": 1.6, "avg": 0.8, "ops": 1.2,
                      "k_bat": -0.5, "tb": 1.4, "nsb": 0.3}
        z_score = 1.0

        # Team needs help in R, HR, RBI
        deficits = [
            CategoryDeficitOut(category="R", my_total=45.0, opponent_total=60.0, deficit=15.0, winning=False),
            CategoryDeficitOut(category="HR", my_total=22.0, opponent_total=30.0, deficit=8.0, winning=False),
            CategoryDeficitOut(category="RBI", my_total=43.0, opponent_total=55.0, deficit=12.0, winning=False),
        ]

        result = compute_need_score(cat_scores, z_score, deficits, n_cats=9)

        # Should be higher than plain z_score due to matchup-specific boost
        assert result > z_score, "Positive matchup impact should increase need_score"

    def test_zero_cat_scores_returns_z_score(self):
        """Empty cat_scores dict should return plain z_score."""
        z_score = 0.6

        deficits = [
            CategoryDeficitOut(category="R", my_total=40.0, opponent_total=50.0, deficit=10.0, winning=False),
        ]

        result = compute_need_score({}, z_score, deficits, n_cats=9)

        assert result == 0.6, "Empty cat_scores should return z_score"


class TestNeedScoreContract:
    """Validate contract requirements for need_score computation."""

    def test_returns_four_decimal_place_precision(self):
        """Need score should be usable in sorting without coercion issues."""
        cat_scores = {"r": 1.234, "h": 0.567, "hr": 1.89}
        z_score = 0.777

        deficits = [
            CategoryDeficitOut(category="R", my_total=44.5, opponent_total=50.0, deficit=5.5, winning=False),
        ]

        result = compute_need_score(cat_scores, z_score, deficits, n_cats=9)

        # Should be a float that can be rounded to 4 decimal places
        rounded = round(result, 4)
        assert isinstance(rounded, float)
        # Verify no NaN or Infinity
        assert not (result != result)  # NaN check
        assert abs(result) < float('inf')  # Infinity check


class TestDirectScorerParity:
    """Verify compute_need_score wraps score_fa_against_needs correctly."""

    def test_compute_need_score_matches_direct_scorer(self):
        """compute_need_score should produce same result as manual scorer + blend."""
        cat_scores = {"r": 1.5, "h": 0.8, "hr": 1.2, "rbi": 1.0, "avg": 0.6, "ops": 0.9,
                      "k_bat": -0.2, "tb": 0.7, "nsb": 0.4}
        z_score = 0.85
        n_cats = 9

        deficits = [
            CategoryDeficitOut(category="R", my_total=45.0, opponent_total=55.0, deficit=10.0, winning=False),
            CategoryDeficitOut(category="HR", my_total=22.0, opponent_total=28.0, deficit=6.0, winning=False),
        ]

        # Direct computation via score_fa_against_needs + blend
        # Lowercase category names to match board keys
        needs_dict = {cd.category.lower(): cd.deficit for cd in deficits}
        team_needs = CategoryNeedVector(needs=needs_dict)
        impacts_dict = {k: float(v) for k, v in cat_scores.items()}
        fa_impact = PlayerCategoryImpactVector(impacts=impacts_dict)

        cat_score = score_fa_against_needs(fa_impact, team_needs)
        direct_result = 0.4 * z_score + 0.6 * (cat_score / n_cats)

        # Via compute_need_score
        helper_result = compute_need_score(cat_scores, z_score, deficits, n_cats)

        # Should match to 4 decimal places (allowing for minor float differences)
        assert round(helper_result, 4) == round(direct_result, 4), (
            "compute_need_score should match direct scorer computation"
        )
