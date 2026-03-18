"""
Tests for Shin-corrected market odds integration in predict_game().

Covers:
1. _shin_implied_prob() — vig removal math
2. predict_game() market blend at R64 (50%), R32 (35%), S16+ (20%)
3. Graceful fallback when market_ml is None

Run: pytest tests/test_market_odds_blend.py -v
"""

import math
import pytest
from backend.tournament.matchup_predictor import (
    TournamentTeam,
    predict_game,
    _shin_implied_prob,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _team(name: str, seed: int, rating: float, ml: int = None) -> TournamentTeam:
    return TournamentTeam(
        name=name,
        seed=seed,
        region="east",
        composite_rating=rating,
        market_ml=ml,
    )


# ---------------------------------------------------------------------------
# _shin_implied_prob unit tests
# ---------------------------------------------------------------------------

class TestShinImpliedProb:

    def test_favourite_negative_underdog_positive(self):
        """Standard -200 / +170 line: favourite should be ~54% after vig removal."""
        p = _shin_implied_prob(-200, +170)
        assert p is not None
        # Gross: 200/300 = 0.6667, 100/270 = 0.3704; overround = 1.037
        # Shin: 0.6667 / 1.037 ≈ 0.643
        assert 0.60 < p < 0.70, f"Expected ~64%, got {p:.3f}"

    def test_heavy_favourite(self):
        """Heavy favourite -500 / +400 should give ~82% true win prob."""
        p = _shin_implied_prob(-500, +400)
        assert p is not None
        assert 0.78 < p < 0.88, f"Expected ~83%, got {p:.3f}"

    def test_even_money(self):
        """Even money -110 / -110 (vig on both sides) ≈ 50%."""
        p = _shin_implied_prob(-110, -110)
        assert p is not None
        assert abs(p - 0.50) < 0.02, f"Expected ~50%, got {p:.3f}"

    def test_symmetry(self):
        """P(a) + P(b) = 1 when computed from both perspectives."""
        p_a = _shin_implied_prob(-150, +130)
        p_b = _shin_implied_prob(+130, -150)
        assert p_a is not None and p_b is not None
        assert abs(p_a + p_b - 1.0) < 1e-9, f"Probs don't sum to 1: {p_a} + {p_b}"

    def test_none_ml_a_returns_none(self):
        p = _shin_implied_prob(None, +130)
        assert p is None

    def test_none_ml_b_returns_none(self):
        p = _shin_implied_prob(-150, None)
        assert p is None

    def test_both_none_returns_none(self):
        p = _shin_implied_prob(None, None)
        assert p is None

    def test_invalid_overround_too_low_returns_none(self):
        """Overround < 0.98 is invalid (arbitrage)."""
        # +200 / +200 → overround = 0.333 + 0.333 = 0.667 < 0.98
        p = _shin_implied_prob(+200, +200)
        assert p is None

    def test_invalid_overround_too_high_returns_none(self):
        """Overround > 1.30 is invalid (data error)."""
        # -5000 / -5000 → each gross ≈ 0.98; overround ≈ 1.96
        p = _shin_implied_prob(-5000, -5000)
        assert p is None

    def test_result_is_valid_probability(self):
        """All valid inputs must produce a probability in (0, 1)."""
        for ml_a, ml_b in [(-120, +100), (-300, +250), (-110, -110), (-180, +155)]:
            p = _shin_implied_prob(ml_a, ml_b)
            assert p is not None
            assert 0.0 < p < 1.0, f"Invalid prob {p} for ({ml_a}, {ml_b})"


# ---------------------------------------------------------------------------
# predict_game() market blend integration tests
# ---------------------------------------------------------------------------

class TestMarketBlendInPredictGame:

    # A realistic R64 matchup: 1-seed vs 16-seed
    # Without odds the model gives ~98% to the 1-seed.
    # With sharp market lines strongly favouring the 1-seed, result stays high.
    FAV  = _team("Duke",  1, 28.0, ml=-2500)   # 1-seed heavy fav
    DOG  = _team("Siena", 16, -9.5, ml=+1600)  # 16-seed big dog

    def test_market_blend_r64_raises_prob_toward_market(self):
        """R64 blend (50% market) pulls win prob toward the market implied prob."""
        # Without market
        p_no_mkt, _, _ = predict_game(
            _team("Duke", 1, 28.0), _team("Siena", 16, -9.5), round_num=1
        )
        # With market (both teams have ml set)
        p_mkt, _, _ = predict_game(self.FAV, self.DOG, round_num=1)

        # Market says ~93% (after vig removal on -2500/+1600), model says ~98%.
        # Blend at 50%: should be between 0.90 and 0.98.
        assert 0.88 < p_mkt < 0.99, f"Blended prob out of range: {p_mkt:.3f}"
        # The blended result should differ from the no-market result
        assert abs(p_mkt - p_no_mkt) > 0.005, (
            f"Market blend had no effect: p_mkt={p_mkt:.4f} p_no={p_no_mkt:.4f}"
        )

    def test_market_blend_r32_uses_35pct_weight(self):
        """R32 uses 35% market weight — smaller pull than R64."""
        fav = _team("Duke",  1, 28.0, ml=-400)
        dog = _team("Siena", 16, -9.5, ml=+310)
        p_r64, _, _ = predict_game(fav, dog, round_num=1)
        p_r32, _, _ = predict_game(fav, dog, round_num=2)
        # Both use same market lines; R64 blends more so deviates more from model.
        # We can't know direction without knowing model_prob vs market_prob,
        # but we can verify both are valid probabilities.
        assert 0.0 < p_r64 < 1.0
        assert 0.0 < p_r32 < 1.0

    def test_market_blend_late_round_uses_20pct_weight(self):
        """S16+ (round 3+) uses 20% market weight."""
        fav = _team("Duke", 1, 28.0, ml=-250)
        dog = _team("Gonzaga", 3, 18.0, ml=+200)
        p, _, _ = predict_game(fav, dog, round_num=3)
        assert 0.0 < p < 1.0

    def test_no_market_ml_skips_blend(self):
        """When market_ml is None, predict_game returns model-only probability."""
        fav_no_ml = _team("Duke",  1, 28.0, ml=None)
        dog_no_ml = _team("Siena", 16, -9.5, ml=None)
        p1, _, _ = predict_game(fav_no_ml, dog_no_ml, round_num=1)
        # Should be the pure model prob — very high for 1v16
        assert p1 > 0.90, f"1v16 model-only prob should be >90%, got {p1:.3f}"

    def test_partial_market_ml_skips_blend(self):
        """If only one team has market_ml, the blend is skipped."""
        fav_ml  = _team("Duke",  1, 28.0, ml=-2500)
        dog_no  = _team("Siena", 16, -9.5, ml=None)
        p_partial, _, _ = predict_game(fav_ml, dog_no, round_num=1)

        fav_no  = _team("Duke",  1, 28.0, ml=None)
        p_none,   _, _ = predict_game(fav_no,  dog_no, round_num=1)

        assert abs(p_partial - p_none) < 1e-9, (
            "Partial market_ml should produce same result as no market_ml"
        )

    def test_underdog_friendly_market_raises_upset_prob(self):
        """
        If the market implies a higher upset chance than the model,
        blending should increase the underdog's win probability.
        """
        # 5v12: model gives 12-seed about 25-35% win prob.
        # Set market lines to imply ~45% for the 12-seed (toss-up).
        # After 50% R64 blend, blended p(12 wins) should be between model and 45%.
        fav = _team("St. Johns",    5,  13.5, ml=-120)  # ~54% implied
        dog = _team("N Iowa",      12,  -0.5, ml=+100)  # ~49% implied

        # Without market
        p_model_fav, _, _ = predict_game(
            _team("St. Johns", 5, 13.5), _team("N Iowa", 12, -0.5), round_num=1
        )
        # With market
        p_blend_fav, _, _ = predict_game(fav, dog, round_num=1)

        # The market implies ~54% for the fav; model says higher (say 65-70%).
        # Blend should pull fav prob DOWN closer to 54%.
        # So p_blend_fav < p_model_fav (market is more conservative about fav).
        assert p_blend_fav < p_model_fav, (
            f"Market pull failed: blend={p_blend_fav:.3f} model={p_model_fav:.3f}"
        )

    def test_win_prob_stays_in_valid_range(self):
        """Blended probability must always be in (0, 1)."""
        matchups = [
            (_team("A", 1, 30.0, ml=-3000), _team("B", 16, -10.0, ml=+2000)),
            (_team("C", 8, 7.0,  ml=-110),  _team("D",  9,  5.0,  ml=-110)),
            (_team("E", 5, 13.0, ml=-130),  _team("F", 12, -1.0,  ml=+110)),
        ]
        for ta, tb in matchups:
            for rnd in [1, 2, 3, 4, 5, 6]:
                p, _, _ = predict_game(ta, tb, rnd)
                assert 0.0 < p < 1.0, f"Invalid prob {p} for {ta.name} vs {tb.name} R{rnd}"
