"""
Tests for parlay_engine.py

Run with: pytest tests/test_parlay_engine.py -v
"""

import pytest
from backend.services.parlay_engine import (
    build_optimal_parlays,
    format_parlay_ticket,
    _american_to_decimal,
    _calculate_parlay_metrics,
    MIN_EDGE_THRESHOLD,
    MIN_PARLAY_UNITS,
    PARLAY_KELLY_DIVISOR,
)


class TestAmericanToDecimal:
    """Test odds conversion."""

    def test_positive_odds(self):
        """Test conversion of positive American odds."""
        assert _american_to_decimal(100) == pytest.approx(2.0)
        assert _american_to_decimal(150) == pytest.approx(2.5)
        assert _american_to_decimal(200) == pytest.approx(3.0)

    def test_negative_odds(self):
        """Test conversion of negative American odds."""
        assert _american_to_decimal(-110) == pytest.approx(1.909, abs=0.01)
        assert _american_to_decimal(-150) == pytest.approx(1.667, abs=0.01)
        assert _american_to_decimal(-200) == pytest.approx(1.5)


class TestCalculateParlayMetrics:
    """Test parlay probability and edge calculations."""

    def test_two_leg_parlay(self):
        """Test metrics for a simple 2-leg parlay."""
        bets = [
            {"pick": "Duke -4.5"},
            {"pick": "UNC +3.5"},
        ]
        win_probs = [0.60, 0.55]  # 60% and 55% win probability
        decimal_odds = [1.909, 1.909]  # -110 on both legs

        metrics = _calculate_parlay_metrics(bets, win_probs, decimal_odds)

        # Joint probability should be 0.60 * 0.55 = 0.33
        assert metrics["joint_prob"] == pytest.approx(0.33)

        # Parlay odds should be 1.909 * 1.909 = 3.644
        assert metrics["parlay_odds"] == pytest.approx(3.644, abs=0.01)

        # Edge should be positive (both legs have positive edge)
        assert metrics["edge"] > 0

        # Kelly should be conservative (divided by 4)
        assert metrics["kelly_fractional"] < metrics["kelly_full"] / 3.0

    def test_three_leg_parlay(self):
        """Test metrics for a 3-leg parlay."""
        bets = [
            {"pick": "Duke -4.5"},
            {"pick": "UNC +3.5"},
            {"pick": "Kansas -6.5"},
        ]
        win_probs = [0.60, 0.55, 0.65]
        decimal_odds = [1.909, 1.909, 1.909]

        metrics = _calculate_parlay_metrics(bets, win_probs, decimal_odds)

        # Joint probability should be 0.60 * 0.55 * 0.65 = 0.2145
        assert metrics["joint_prob"] == pytest.approx(0.2145, abs=0.001)

        # Parlay odds should be (1.909)^3 = 6.954
        assert metrics["parlay_odds"] == pytest.approx(6.954, abs=0.01)

    def test_kelly_cap_at_5_percent(self):
        """Test that Kelly fraction is capped at 5%."""
        bets = [{"pick": "Lock 1"}, {"pick": "Lock 2"}]
        win_probs = [0.90, 0.90]  # Unrealistic locks
        decimal_odds = [1.5, 1.5]  # Short odds

        metrics = _calculate_parlay_metrics(bets, win_probs, decimal_odds)

        # Kelly should be capped at 5% (0.05)
        assert metrics["kelly_fractional"] <= 0.05


def _make_bet(game_id, pick, edge_conservative, lower_ci_prob, bet_odds=-110):
    """
    Build a slate_bet fixture with a correctly populated full_analysis.calculations.

    market_prob is derived as lower_ci_prob - edge_conservative so that:
        true_leg_prob = market_prob + edge_conservative == lower_ci_prob

    This mirrors the betting model, where:
        edge_conservative = our_cover_lower - market_prob
        market_prob + edge_conservative = our_cover_lower  (side-aware)
    """
    market_prob = round(lower_ci_prob - edge_conservative, 6)
    return {
        "game_id": game_id,
        "pick": pick,
        "edge_conservative": edge_conservative,
        "lower_ci_prob": lower_ci_prob,
        "full_analysis": {
            "calculations": {
                "bet_odds": bet_odds,
                "market_prob": market_prob,
                "edge_conservative": edge_conservative,
            }
        },
    }


class TestBuildOptimalParlays:
    """Test parlay builder main logic."""

    def test_insufficient_bets(self):
        """Test that no parlays are built with < 2 qualified bets."""
        slate = [_make_bet(1, "Duke -4.5", 0.025, 0.60)]

        parlays = build_optimal_parlays(slate)
        assert len(parlays) == 0

    def test_two_leg_parlays(self):
        """Test building 2-leg parlays from 3 qualified bets.

        With overlap prevention, the greedy pass accepts the top parlay and
        then skips all remaining parlays that share a game_id with it.
        From 3 games with max_legs=2, the best 2-leg ticket claims 2 game_ids
        leaving only 1 game, which cannot form another 2-leg parlay alone.
        Result: 1 non-overlapping ticket.
        """
        slate = [
            _make_bet(1, "Duke -4.5",    0.025, 0.60),
            _make_bet(2, "UNC +3.5",     0.030, 0.58),
            _make_bet(3, "Kansas -6.5",  0.028, 0.62),
        ]

        parlays = build_optimal_parlays(slate, max_legs=2)

        # Overlap prevention leaves 1 non-overlapping ticket from 3 games
        assert len(parlays) == 1

        # The returned parlay must have 2 legs
        assert parlays[0]["num_legs"] == 2

    def test_three_leg_parlays(self):
        """Test building up to 3-leg parlays with overlap prevention.

        With 5 distinct games and max_legs=3, the greedy pass yields:
          - 1 best-EV 3-leg ticket (claims 3 game_ids)
          - 1 best-EV 2-leg ticket from the 2 remaining games
        Both leg-count types appear in the non-overlapping result set.
        """
        slate = [
            _make_bet(1, "Duke -4.5",      0.025, 0.60),
            _make_bet(2, "UNC +3.5",       0.030, 0.58),
            _make_bet(3, "Kansas -6.5",    0.028, 0.62),
            _make_bet(4, "Villanova +4.5", 0.032, 0.59),
            _make_bet(5, "Kentucky -3.5",  0.027, 0.61),
        ]

        parlays = build_optimal_parlays(slate, max_legs=3)

        # Greedy: best 3-leg (3 games) + best 2-leg from remaining 2 games
        assert len(parlays) == 2

        # Check that we have both 2-leg and 3-leg parlays
        num_legs_set = {p["num_legs"] for p in parlays}
        assert 2 in num_legs_set
        assert 3 in num_legs_set

    def test_same_game_rejection(self):
        """Test that parlays with bets from same game are rejected.

        Same-game filter removes the Duke+UNC-total combo.  The two surviving
        candidates (Duke+Kansas, UNC-total+Kansas) both contain Kansas (game 2),
        so overlap-prevention skips the second after the first is accepted.
        Result: 1 non-overlapping ticket with no same-game legs.
        """
        slate = [
            _make_bet(1, "Duke -4.5",       0.025, 0.60),
            _make_bet(1, "Duke/UNC U145.5", 0.030, 0.58),  # Same game_id
            _make_bet(2, "Kansas -6.5",     0.028, 0.62),
        ]

        parlays = build_optimal_parlays(slate, max_legs=2)

        # 1 non-overlapping ticket survives both filters
        assert len(parlays) == 1

        # Verify no parlay contains bets from same game
        for parlay in parlays:
            game_ids = [leg["game_id"] for leg in parlay["legs"]]
            assert len(game_ids) == len(set(game_ids))

    def test_edge_threshold_filtering(self):
        """Test that bets below MIN_EDGE_THRESHOLD are excluded."""
        slate = [
            _make_bet(1, "Weak Edge",      0.005, 0.52),  # Below 1% threshold
            _make_bet(2, "Strong Edge",    0.025, 0.60),
            _make_bet(3, "Another Strong", 0.030, 0.58),
        ]

        parlays = build_optimal_parlays(slate, max_legs=2)

        # Should only use the 2 strong edges, generating 1 parlay
        assert len(parlays) == 1
        assert parlays[0]["num_legs"] == 2

        # Verify weak edge bet not included
        for parlay in parlays:
            picks = [leg["pick"] for leg in parlay["legs"]]
            assert "Weak Edge" not in picks

    def test_max_parlays_limit(self):
        """Test that max_parlays parameter is respected as an upper bound.

        With 5 distinct game IDs and overlap prevention the greedy pass yields
        at most 2 non-overlapping tickets (one 3-leg + one 2-leg from the
        remaining 2 games).  max_parlays=5 is an upper bound; the actual count
        is limited by how many non-overlapping tickets can be formed.
        """
        slate = [_make_bet(i, f"Bet {i}", 0.025, 0.58) for i in range(1, 6)]

        parlays = build_optimal_parlays(slate, max_legs=3, max_parlays=5)

        # Never exceeds max_parlays
        assert len(parlays) <= 5
        # With 5 distinct games the overlap filter yields exactly 2 tickets
        assert len(parlays) == 2

    def test_leg_overlap_prevention(self):
        """Test that returned tickets share no game_ids across tickets."""
        # 4 distinct games — any two games form a valid 2-leg parlay.
        # C(4,2) = 6 candidates, but once the top ticket claims games A+B,
        # no subsequent ticket may include A or B, leaving only game C+D.
        # So max non-overlapping 2-leg tickets from 4 games = 2.
        slate = [
            _make_bet(1, "Duke -4.5",      0.030, 0.60),
            _make_bet(2, "UNC +3.5",       0.028, 0.58),
            _make_bet(3, "Kansas -6.5",    0.025, 0.62),
            _make_bet(4, "Villanova +4.5", 0.032, 0.59),
        ]

        parlays = build_optimal_parlays(slate, max_legs=2, max_parlays=10)

        # Overlap check: no game_id may appear in more than one ticket
        seen: set = set()
        for parlay in parlays:
            game_ids = {leg["game_id"] for leg in parlay["legs"]}
            assert not (game_ids & seen), (
                f"Overlap detected: {game_ids & seen} already in accepted tickets"
            )
            seen.update(game_ids)

        # With 4 games and 2-leg parlays the greedy pass yields exactly 2 tickets
        assert len(parlays) == 2

    def test_parlay_american_odds_conversion(self):
        """Test that parlay odds are correctly converted to American format."""
        slate = [
            _make_bet(1, "Duke -4.5", 0.025, 0.60),
            _make_bet(2, "UNC +3.5",  0.030, 0.58),
        ]

        parlays = build_optimal_parlays(slate, max_legs=2)

        assert len(parlays) == 1
        parlay = parlays[0]

        # 2-leg parlay at -110/-110 should have American odds around +264
        # (1.909 * 1.909 = 3.644 decimal = +264 American)
        assert parlay["parlay_american_odds"] == pytest.approx(264, abs=5)


class TestFormatParlayTicket:
    """Test parlay ticket formatting."""

    def test_format_two_leg_parlay(self):
        """Test formatting a 2-leg parlay ticket."""
        parlay = {
            "num_legs": 2,
            "parlay_american_odds": 264,
            "leg_summary": "Duke -4.5 + UNC +3.5",
            "joint_prob": 0.33,
            "expected_value": 0.1234,
            "edge": 0.0456,
            "recommended_units": 0.78,
        }

        output = format_parlay_ticket(parlay)

        assert "2-Leg Parlay" in output
        assert "+264" in output
        assert "Duke -4.5 + UNC +3.5" in output
        assert "33.00%" in output
        assert "0.1234" in output
        assert "0.78 units" in output


class TestPortfolioCapAwareness:
    """Test global portfolio cap enforcement on parlay sizing."""

    def test_zero_capacity_returns_no_parlays(self):
        """When remaining_capacity_dollars=0, no parlays are returned at all.

        The per-ticket clamp drops the combo immediately when capacity is
        exhausted, so the output list is empty rather than containing a ticket
        with recommended_units=0 (which would be a misleading ghost parlay).
        """
        slate = [
            _make_bet(1, "Duke -4.5", 0.025, 0.60),
            _make_bet(2, "UNC +3.5",  0.030, 0.58),
        ]

        parlays = build_optimal_parlays(
            slate, max_legs=2,
            remaining_capacity_dollars=0.0,
            bankroll=1000.0,
        )

        assert len(parlays) == 0

    def test_tight_capacity_scales_units_down(self):
        """When capacity is less than raw parlay dollars, units are scaled."""
        slate = [
            _make_bet(1, "Duke -4.5", 0.025, 0.60),
            _make_bet(2, "UNC +3.5",  0.030, 0.58),
        ]

        # Without cap, get the raw recommended_units
        uncapped = build_optimal_parlays(slate, max_legs=2)
        raw_units = uncapped[0]["recommended_units"]

        # bankroll=$1000 → 1 unit = $10. Cap at $5 (0.5 units).
        parlays = build_optimal_parlays(
            slate, max_legs=2,
            remaining_capacity_dollars=5.0,
            bankroll=1000.0,
        )

        assert len(parlays) == 1
        scaled_units = parlays[0]["recommended_units"]

        # Scaled must be ≤ 0.5 units ($5 / ($1000/100))
        assert scaled_units <= 0.5 + 1e-6
        # And strictly less than the uncapped value
        assert scaled_units < raw_units

    def test_ample_capacity_leaves_units_unchanged(self):
        """When capacity far exceeds parlay dollars, units are not changed."""
        slate = [
            _make_bet(1, "Duke -4.5", 0.025, 0.60),
            _make_bet(2, "UNC +3.5",  0.030, 0.58),
        ]

        uncapped = build_optimal_parlays(slate, max_legs=2)
        raw_units = uncapped[0]["recommended_units"]

        # $10,000 remaining capacity — way more than any parlay needs
        parlays = build_optimal_parlays(
            slate, max_legs=2,
            remaining_capacity_dollars=10_000.0,
            bankroll=1000.0,
        )

        assert parlays[0]["recommended_units"] == pytest.approx(raw_units, abs=0.001)


class TestParlayKellyConservatism:
    """Test that parlay Kelly is properly conservative."""

    def test_divisor_applied(self):
        """Verify PARLAY_KELLY_DIVISOR is applied correctly."""
        bets = [{"pick": "A"}, {"pick": "B"}]
        win_probs = [0.60, 0.58]
        decimal_odds = [1.909, 1.909]

        metrics = _calculate_parlay_metrics(bets, win_probs, decimal_odds)

        # Kelly fractional should be full / PARLAY_KELLY_DIVISOR
        expected = metrics["kelly_full"] / PARLAY_KELLY_DIVISOR
        assert metrics["kelly_fractional"] == pytest.approx(expected, abs=0.001)

    def test_parlay_sizing_vs_single(self):
        """Verify parlay sizing is more conservative than single bet."""
        # Single bet Kelly
        single_prob = 0.60
        single_odds = 1.909
        single_payout = single_odds - 1.0
        single_kelly = (single_prob * single_payout - (1 - single_prob)) / single_payout

        # 2-leg parlay Kelly
        bets = [{"pick": "A"}, {"pick": "B"}]
        win_probs = [0.60, 0.60]
        decimal_odds = [1.909, 1.909]
        parlay_metrics = _calculate_parlay_metrics(bets, win_probs, decimal_odds)

        # Parlay fractional Kelly should be significantly smaller
        # (due to both lower joint probability AND 4x divisor)
        assert parlay_metrics["kelly_fractional"] < single_kelly / 3.0
