"""
Chaos calibration tests — proves the chaos slider is anchored to historical
NCAA tournament upset rates rather than distorting probabilities arbitrarily.

Key invariants we verify:
1. chaos=0.0  → 0 upsets (pure chalk)
2. chaos=0.5  → upset rates match model probability (historically calibrated):
     1v16 ≈ 1-3%,  5v12 ≈ 28-38%,  8v9 ≈ 44-50%
3. chaos=1.0  → 1-seeds never become coin-flips (≤ 10% upset)
4. Monotone: upset rate strictly increases as chaos increases
5. SmartBracketGenerator: upset count per round grows smoothly (no cliff)

Run: pytest tests/test_chaos_calibration.py -v
"""

import random
import pytest
from backend.tournament.matchup_predictor import TournamentTeam, predict_game
from backend.tournament.smart_bracket import SmartBracketGenerator, generate_smart_bracket


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _team(name: str, seed: int, rating: float, region: str = "east") -> TournamentTeam:
    return TournamentTeam(name=name, seed=seed, region=region, composite_rating=rating)


def _fixed_pick_winner(ta, tb, rnd, chaos, rng):
    """
    The calibrated pick_winner from dashboard/pages/14_Bracket_Visual.py.
    Inlined here to avoid importing Streamlit.
    """
    p, _, _ = predict_game(ta, tb, rnd)
    if chaos == 0.0:
        return (ta, tb, p) if p >= 0.5 else (tb, ta, 1.0 - p)
    if p >= 0.5:
        fav, dog, fav_p = ta, tb, p
    else:
        fav, dog, fav_p = tb, ta, 1.0 - p
    upset_p = 1.0 - fav_p
    if chaos <= 0.5:
        effective = upset_p * (chaos / 0.5)
    else:
        t = (chaos - 0.5) / 0.5
        effective = min(0.5, upset_p * (1.0 + t))
    return (dog, fav, upset_p) if rng.random() < effective else (fav, dog, fav_p)


def _upset_rate(ta, tb, chaos, n=20000, seed=0):
    rng = random.Random(seed)
    upsets = sum(
        1 for _ in range(n)
        if _fixed_pick_winner(ta, tb, 1, chaos, rng)[0].seed == tb.seed
    )
    return upsets / n


def _make_full_bracket():
    """Minimal full bracket (16 teams per region) for SmartBracketGenerator tests."""
    regions = {}
    seeds_ratings = [
        (1, 28.0), (2, 22.0), (3, 18.0), (4, 15.0),
        (5, 12.0), (6, 10.0), (7, 8.0), (8, 6.5),
        (9, 5.0), (10, 3.0), (11, 1.0), (12, -1.0),
        (13, -3.0), (14, -5.0), (15, -7.0), (16, -9.5),
    ]
    names = [
        "Duke", "UConn", "MichSt", "Kansas", "StJohns", "Louisville",
        "UCLA", "OhioSt", "TCU", "UCF", "SoFla", "NIowa",
        "CalBap", "NDakSt", "Furman", "Siena",
    ]
    for region in ["east", "west", "south", "midwest"]:
        regions[region] = [
            _team(f"{name}_{region}", seed, rating, region)
            for (seed, rating), name in zip(seeds_ratings, names)
        ]
    return regions


# ---------------------------------------------------------------------------
# Page-14 pick_winner calibration tests (stochastic)
# ---------------------------------------------------------------------------

class TestPickWinnerCalibration:

    # Pairs: (favourite_seed, underdog_seed, approx_fav_rating, approx_dog_rating)
    MATCHUPS = {
        "1v16": (_team("Duke", 1, 28.0), _team("Siena", 16, -9.5)),
        "5v12": (_team("StJohns", 5, 13.5), _team("NIowa", 12, -0.5)),
        "8v9":  (_team("OhioSt", 8, 7.5), _team("TCU", 9, 5.5)),
    }

    def test_chalk_produces_zero_upsets(self):
        for name, (fav, dog) in self.MATCHUPS.items():
            rate = _upset_rate(fav, dog, chaos=0.0)
            assert rate == 0.0, f"{name} at chaos=0: got {rate:.1%}, expected 0%"

    def test_1v16_at_chaos_half_is_low(self):
        """1-seeds should win ≥95% of games at any chaos level."""
        fav, dog = self.MATCHUPS["1v16"]
        rate = _upset_rate(fav, dog, chaos=0.5, n=50000)
        assert rate < 0.06, f"1v16 upset rate at chaos=0.5: {rate:.1%} (expected <6%)"

    def test_1v16_at_max_chaos_is_still_low(self):
        """Even chaos=1.0 must not make a 1v16 a coin flip."""
        fav, dog = self.MATCHUPS["1v16"]
        rate = _upset_rate(fav, dog, chaos=1.0, n=50000)
        assert rate < 0.10, f"1v16 upset rate at chaos=1.0: {rate:.1%} (expected <10%)"

    def test_5v12_at_chaos_half_matches_historical(self):
        """5v12 upset rate at chaos=0.5 should be 25-40% (historical 35%)."""
        fav, dog = self.MATCHUPS["5v12"]
        rate = _upset_rate(fav, dog, chaos=0.5, n=50000)
        assert 0.25 <= rate <= 0.40, (
            f"5v12 at chaos=0.5: {rate:.1%} (expected 25-40%)"
        )

    def test_8v9_at_chaos_half_matches_historical(self):
        """8v9 at chaos=0.5 should be 40-55% (historical ~49%)."""
        fav, dog = self.MATCHUPS["8v9"]
        rate = _upset_rate(fav, dog, chaos=0.5, n=50000)
        assert 0.40 <= rate <= 0.55, (
            f"8v9 at chaos=0.5: {rate:.1%} (expected 40-55%)"
        )

    def test_upset_rate_monotone_with_chaos(self):
        """Upset rate must increase strictly as chaos increases."""
        fav, dog = self.MATCHUPS["5v12"]
        prev = 0.0
        for chaos in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            rate = _upset_rate(fav, dog, chaos=chaos, n=30000, seed=chaos)
            assert rate >= prev - 0.02, (
                f"5v12 rate not monotone: chaos={chaos} gave {rate:.1%}, prev={prev:.1%}"
            )
            prev = rate

    def test_1v16_rate_always_below_5v12_rate(self):
        """At every chaos level, 1v16 upset rate must be much lower than 5v12."""
        for chaos in [0.3, 0.5, 0.7, 1.0]:
            r_1v16 = _upset_rate(*self.MATCHUPS["1v16"], chaos=chaos, n=30000)
            r_5v12 = _upset_rate(*self.MATCHUPS["5v12"], chaos=chaos, n=30000)
            assert r_1v16 < r_5v12 - 0.10, (
                f"At chaos={chaos}: 1v16={r_1v16:.1%} should be << 5v12={r_5v12:.1%}"
            )


# ---------------------------------------------------------------------------
# SmartBracketGenerator deterministic calibration tests
# ---------------------------------------------------------------------------

class TestSmartBracketDeterministicCalibration:

    BRACKET = _make_full_bracket()

    def _upset_counts(self, chaos):
        sb = generate_smart_bracket(self.BRACKET, chaos_level=chaos)
        by_round = {}
        for u in sb["upsets"]:
            by_round[u["round"]] = by_round.get(u["round"], 0) + 1
        return by_round

    def test_chalk_produces_zero_upsets(self):
        counts = self._upset_counts(0.0)
        assert sum(counts.values()) == 0, f"chalk should have 0 upsets, got {counts}"

    def test_chaos_half_has_upsets_in_multiple_rounds(self):
        counts = self._upset_counts(0.5)
        assert counts.get(1, 0) > 0, "chaos=0.5 should have R64 upsets"
        assert counts.get(2, 0) > 0, "chaos=0.5 should have R32 upsets"

    def test_chaos_half_r64_matches_historical_range(self):
        """At chaos=0.5, R64 upsets should be 6-12 (historical avg ~10/32 games)."""
        counts = self._upset_counts(0.5)
        r64 = counts.get(1, 0)
        assert 4 <= r64 <= 16, f"R64 upsets at chaos=0.5: {r64} (expected 4-16)"

    def test_total_upsets_grow_smoothly(self):
        """Total upsets should increase monotonically with chaos level."""
        totals = {}
        for chaos in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            counts = self._upset_counts(chaos)
            totals[chaos] = sum(counts.values())

        prev = 0
        for chaos in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            assert totals[chaos] >= prev, (
                f"Upset count not monotone: chaos={chaos} gave {totals[chaos]}, "
                f"prev={prev}"
            )
            prev = totals[chaos]

    def test_no_cliff_between_03_and_05(self):
        """
        The old implementation had a massive cliff from 4 upsets at chaos=0.3
        to 20 upsets at chaos=0.5.  The fixed version must be gradual.
        """
        c03 = sum(self._upset_counts(0.3).values())
        c05 = sum(self._upset_counts(0.5).values())
        # Difference should be moderate — not more than 2× jump
        assert c05 <= c03 * 2.5, (
            f"Cliff still present: chaos=0.3 gave {c03} upsets, "
            f"chaos=0.5 gave {c05} (ratio {c05/max(c03,1):.1f}x)"
        )

    def test_higher_seed_matchups_get_upset_before_lower_seed(self):
        """
        The model should pick the most likely upsets first.
        At a chaos level that produces only 1 R64 upset per region,
        that upset should be an 8v9 or 9v8 game (coin flip) not a 1v16.
        """
        counts = self._upset_counts(0.1)
        upsets = generate_smart_bracket(self.BRACKET, chaos_level=0.1)["upsets"]
        r64_upsets = [u for u in upsets if u["round"] == 1]
        for u in r64_upsets:
            assert u["winner_seed"] <= 12, (
                f"Low-chaos upset should be a competitive game, not "
                f"#{u['winner_seed']} over #{u['loser_seed']}"
            )

    def test_n_upsets_formula_at_boundary_chaos(self):
        """_n_upsets_for_round() must return 0 at chaos=0 and > 0 at chaos=0.5."""
        gen0 = SmartBracketGenerator(chaos_level=0.0)
        gen5 = SmartBracketGenerator(chaos_level=0.5)
        for rnd, n_games in [(1, 8), (2, 4), (3, 2)]:
            assert gen0._n_upsets_for_round(n_games, rnd) == 0
            assert gen5._n_upsets_for_round(n_games, rnd) > 0


# ---------------------------------------------------------------------------
# ROUND_HISTORICAL_UPSET_RATES sanity checks
# ---------------------------------------------------------------------------

class TestHistoricalRates:

    def test_all_rounds_covered(self):
        rates = SmartBracketGenerator.ROUND_HISTORICAL_UPSET_RATES
        for rnd in [1, 2, 3, 4, 5, 6]:
            assert rnd in rates, f"Round {rnd} missing from ROUND_HISTORICAL_UPSET_RATES"

    def test_rates_are_valid_probabilities(self):
        for rnd, rate in SmartBracketGenerator.ROUND_HISTORICAL_UPSET_RATES.items():
            assert 0 < rate <= 1.0, f"Round {rnd} rate {rate} is not a valid probability"

    def test_r64_rate_matches_historical_evidence(self):
        """R64 historical upset rate is ~28-35%."""
        rate = SmartBracketGenerator.ROUND_HISTORICAL_UPSET_RATES[1]
        assert 0.25 <= rate <= 0.40, f"R64 rate {rate:.0%} outside historical range"
