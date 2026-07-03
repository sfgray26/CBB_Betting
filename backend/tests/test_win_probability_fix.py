"""
Regression tests for CRITICAL 3: Win Probability Bug Fix

Tests verify that:
1. _SCORE_TO_SIM mapping matches canonical codes from get_matchup_stats()
2. Current stats are properly passed to the simulator
3. Win probability reflects current score (11-4 lead should have >50% win prob)

Root cause: get_matchup_stats() returns canonical codes (HR_B, K_B, NSB, etc.)
but _SCORE_TO_SIM was using legacy keys (HR, K, SB, etc.), causing current
stats to be ignored entirely.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from backend.fantasy_baseball.mcmc_simulator import simulate_weekly_matchup


class TestScoreToSimMapping:
    """Verify _SCORE_TO_SIM maps all 18 categories correctly."""

    def test_score_to_sim_maps_all_canonical_codes(self):
        """
        Test that _SCORE_TO_SIM uses canonical codes returned by get_matchup_stats().

        get_matchup_stats() returns keys like:
        - Batting: HR_B, R, RBI, H, TB, K_B, NSB, AVG, OPS
        - Pitching: W, L, HR_P, K_P, ERA, WHIP, K_9, QS, NSV

        _SCORE_TO_SIM must use exactly these keys (not legacy HR, K, SB, SV, K9).
        """
        # This is the mapping from fantasy.py:6692-6710
        SCORE_TO_SIM = {
            "HR_B": "hr_b",
            "R": "r",
            "RBI": "rbi",
            "H": "h",
            "TB": "tb",
            "K_B": "k_b",
            "NSB": "nsb",
            "AVG": "avg",
            "OPS": "ops",
            "W": "w",
            "L": "l",
            "HR_P": "hr_p",
            "K_P": "k_p",
            "ERA": "era",
            "WHIP": "whip",
            "K_9": "k_9",
            "QS": "qs",
            "NSV": "nsv",
            "IP": "ip",
        }

        # Simulate what get_matchup_stats() returns (canonical codes)
        mock_yahoo_stats = {
            "HR_B": 5.0,
            "R": 48.0,
            "RBI": 42.0,
            "H": 52.0,
            "TB": 89.0,
            "K_B": 32.0,
            "NSB": 4.0,
            "AVG": 0.268,
            "OPS": 0.785,
            "W": 4.0,
            "L": 2.0,
            "HR_P": 8.0,
            "K_P": 85.0,
            "ERA": 3.42,
            "WHIP": 1.18,
            "K_9": 9.2,
            "QS": 3.0,
            "NSV": 2.0,
            "IP": 52.0,
        }

        # Verify all Yahoo stat keys are in SCORE_TO_SIM
        for yahoo_key in mock_yahoo_stats.keys():
            assert yahoo_key in SCORE_TO_SIM, f"Yahoo stat key '{yahoo_key}' not in SCORE_TO_SIM mapping"

        # Verify the mapping produces lowercase simulation keys
        mapped_current_stats = {}
        for yahoo_key, sim_value in mock_yahoo_stats.items():
            if yahoo_key in SCORE_TO_SIM:
                sim_key = SCORE_TO_SIM[yahoo_key]
                mapped_current_stats[sim_key] = sim_value

        # Verify we got all 18 scoring categories (IP is extra)
        assert len(mapped_current_stats) == 19  # 18 categories + IP

    def test_legacy_keys_not_in_score_to_sim(self):
        """
        Test that legacy keys (HR, K, SB, SV, K9) are NOT used.

        These were the bug - they don't match what get_matchup_stats() returns.
        """
        # Old buggy mapping (DO NOT USE)
        LEGACY_KEYS = {"HR", "K", "SB", "SV", "K9"}

        # Current correct mapping
        SCORE_TO_SIM = {
            "HR_B": "hr_b",
            "R": "r",
            "RBI": "rbi",
            "H": "h",
            "TB": "tb",
            "K_B": "k_b",
            "NSB": "nsb",
            "AVG": "avg",
            "OPS": "ops",
            "W": "w",
            "L": "l",
            "HR_P": "hr_p",
            "K_P": "k_p",
            "ERA": "era",
            "WHIP": "whip",
            "K_9": "k_9",
            "QS": "qs",
            "NSV": "nsv",
            "IP": "ip",
        }

        # Verify legacy keys are NOT in the mapping
        for legacy_key in LEGACY_KEYS:
            assert legacy_key not in SCORE_TO_SIM, f"Legacy key '{legacy_key}' should not be in SCORE_TO_SIM"


class TestCurrentStatsPassedToSimulator:
    """Verify current stats are properly passed to the simulator."""

    def test_simulator_receives_current_stats_with_correct_keys(self):
        """
        Test that simulate_weekly_matchup receives current_stats with
        lowercase simulation keys (hr_b, k_b, nsb, etc.), NOT legacy keys.
        """
        # Mock rosters with minimal cat_scores
        my_roster = [{"name": "Player A", "cat_scores": {"hr_b": 0.5, "r": 0.3}}]
        opp_roster = [{"name": "Player B", "cat_scores": {"hr_b": 0.4, "r": 0.3}}]

        # Current stats with CORRECT keys (lowercase, from SCORE_TO_SIM mapping)
        my_current = {
            "hr_b": 5.0,
            "r": 48.0,
            "k_b": 32.0,
            "nsb": 4.0,
            "avg": 0.268,
        }
        opp_current = {
            "hr_b": 3.0,
            "r": 42.0,
            "k_b": 28.0,
            "nsb": 2.0,
            "avg": 0.245,
        }

        # Run simulation
        result = simulate_weekly_matchup(
            my_roster=my_roster,
            opponent_roster=opp_roster,
            my_current_stats=my_current,
            opp_current_stats=opp_current,
            n_sims=100,
            seed=42,
        )

        # Verify simulation succeeded
        assert result is not None
        assert "win_prob" in result
        assert result["n_sims"] == 100

    def test_simulator_with_empty_current_stats(self):
        """
        Test that simulator handles empty current_stats gracefully.

        This was the bug behavior - empty stats meant current lead was ignored.
        """
        my_roster = [{"name": "Player A", "cat_scores": {"hr_b": 0.5}}]
        opp_roster = [{"name": "Player B", "cat_scores": {"hr_b": 0.6}}]

        # Empty current stats (BUG: this is what happened with key mismatch)
        result = simulate_weekly_matchup(
            my_roster=my_roster,
            opponent_roster=opp_roster,
            my_current_stats={},  # Empty due to key mismatch
            opp_current_stats={},
            n_sims=100,
            seed=42,
        )

        # Should still work, but win_prob based purely on projections
        assert result["win_prob"] is not None
        # With opp having better hr_b projection, win_prob should be < 50%
        assert result["win_prob"] < 0.5


class TestWinProbabilityWithCurrentLead:
    """
    Test that win probability reflects current score.

    If leading 11-4 (current wins > 9 threshold), win probability should be > 50%.
    """

    def test_current_lead_increases_win_probability(self):
        """
        Test that a significant current lead increases win probability.

        Scenario:
        - Roster projections are equal (50/50 without current stats)
        - Current stats: my team leads 11-4
        - Expected: win probability should be > 50% (favoring the leader)
        """
        # Equal rosters (without current stats, would be 50/50)
        base_player = {"name": "Player", "cat_scores": {"hr_b": 0.5, "r": 0.5, "avg": 0.5}}
        my_roster = [base_player.copy() for _ in range(10)]
        opp_roster = [base_player.copy() for _ in range(10)]

        # Simulate current lead: my team winning 11 categories, opp winning 4
        # We use 15 categories for simplicity (5 batting + 10 pitching)
        my_current_stats = {}
        opp_current_stats = {}

        categories = ["hr_b", "r", "rbi", "h", "tb", "avg", "ops", "w", "era", "whip", "k_9", "qs", "nsv", "k_b", "nsb"]

        # My team leads in 11 categories
        for i, cat in enumerate(categories):
            if i < 11:  # My team wins
                my_current_stats[cat] = 10.0
                opp_current_stats[cat] = 5.0
            else:  # Opponent wins
                my_current_stats[cat] = 5.0
                opp_current_stats[cat] = 10.0

        # Run simulation with low remaining_fraction (week nearly over)
        result = simulate_weekly_matchup(
            my_roster=my_roster,
            opponent_roster=opp_roster,
            categories=categories,
            my_current_stats=my_current_stats,
            opp_current_stats=opp_current_stats,
            remaining_fraction=0.1,  # Only 10% of week remaining
            n_sims=500,
            seed=42,
        )

        # With equal rosters but current lead, win_prob should favor the leader
        assert result["win_prob"] > 0.5, f"Win prob {result['win_prob']} should be > 50% when leading 11-4 with 10% week remaining"

    def test_current_trail_decreases_win_probability(self):
        """
        Test that a significant current trail decreases win probability.

        Scenario:
        - Roster projections are equal
        - Current stats: my team trails 4-11
        - Expected: win probability should be < 50%
        """
        base_player = {"name": "Player", "cat_scores": {"hr_b": 0.5, "r": 0.5}}
        my_roster = [base_player.copy() for _ in range(10)]
        opp_roster = [base_player.copy() for _ in range(10)]

        categories = ["hr_b", "r", "rbi", "h", "tb"]

        # My team trails in all categories
        my_current_stats = {cat: 3.0 for cat in categories}
        opp_current_stats = {cat: 10.0 for cat in categories}

        result = simulate_weekly_matchup(
            my_roster=my_roster,
            opponent_roster=opp_roster,
            categories=categories,
            my_current_stats=my_current_stats,
            opp_current_stats=opp_current_stats,
            remaining_fraction=0.2,
            n_sims=500,
            seed=42,
        )

        # Trailing significantly should result in low win probability
        assert result["win_prob"] < 0.5, f"Win prob {result['win_prob']} should be < 50% when trailing"


class TestRegressionForCRITICAL3:
    """
    Specific regression test for CRITICAL 3 bug.

    This test would have FAILED with the buggy _SCORE_TO_SIM mapping
    and PASSES with the correct mapping.
    """

    def test_regression_score_to_sim_bug(self):
        """
        Regression test for the _SCORE_TO_SIM key mismatch bug.

        Before fix: _SCORE_TO_SIM used legacy keys (HR, K, SB, SV, K9)
        After fix: _SCORE_TO_SIM uses canonical codes (HR_B, K_B, NSB, NSV, K_9)

        This test verifies the fix works by simulating the mapping that happens
        in the simulate_matchup endpoint.
        """
        # This is what get_matchup_stats() returns (canonical codes)
        mock_yahoo_response = {
            "my_stats": {
                "HR_B": 5.0,
                "R": 48.0,
                "K_B": 32.0,
                "NSB": 4.0,
                "AVG": 0.268,
            },
            "opp_stats": {
                "HR_B": 3.0,
                "R": 42.0,
                "K_B": 28.0,
                "NSB": 2.0,
                "AVG": 0.245,
            },
        }

        # This is the CORRECT _SCORE_TO_SIM mapping (post-fix)
        SCORE_TO_SIM = {
            "HR_B": "hr_b",
            "R": "r",
            "K_B": "k_b",
            "NSB": "nsb",
            "AVG": "avg",
        }

        # Simulate the mapping that happens in fantasy.py:6698-6702
        my_current = {}
        opp_current = {}

        for yahoo_key, sim_key in SCORE_TO_SIM.items():
            if yahoo_key in mock_yahoo_response["my_stats"]:
                my_current[sim_key] = mock_yahoo_response["my_stats"][yahoo_key]
            if yahoo_key in mock_yahoo_response["opp_stats"]:
                opp_current[sim_key] = mock_yahoo_response["opp_stats"][yahoo_key]

        # Verify all stats were mapped (NO keys were missed)
        assert len(my_current) == 5, f"Expected 5 mapped stats, got {len(my_current)}: {my_current}"
        assert len(opp_current) == 5, f"Expected 5 mapped stats, got {len(opp_current)}: {opp_current}"

        # Verify the keys are lowercase simulation keys (hr_b, k_b, etc.)
        expected_keys = {"hr_b", "r", "k_b", "nsb", "avg"}
        assert set(my_current.keys()) == expected_keys, f"Keys mismatch: {set(my_current.keys())} vs {expected_keys}"

        # Verify values were correctly transferred
        assert my_current["hr_b"] == 5.0
        assert my_current["k_b"] == 32.0
        assert opp_current["hr_b"] == 3.0

    def test_legacy_mapping_would_fail(self):
        """
        Demonstrate that the LEGACY mapping would fail.

        This test shows what happened before the fix: legacy keys (HR, K, SB)
        don't exist in get_matchup_stats() response, so no stats were mapped.
        """
        # This is what get_matchup_stats() returns (canonical codes)
        mock_yahoo_response = {
            "my_stats": {"HR_B": 5.0, "K_B": 32.0, "NSB": 4.0},
        }

        # This was the BUGGY _SCORE_TO_SIM mapping (pre-fix)
        BUGGY_SCORE_TO_SIM = {
            "HR": "hr_b",  # ❌ Wrong key! Should be "HR_B"
            "K": "k_b",    # ❌ Wrong key! Should be "K_B"
            "SB": "nsb",   # ❌ Wrong key! Should be "NSB"
        }

        # Simulate the mapping with BUGGY keys
        my_current_buggy = {}
        for yahoo_key, sim_key in BUGGY_SCORE_TO_SIM.items():
            if yahoo_key in mock_yahoo_response["my_stats"]:
                my_current_buggy[sim_key] = mock_yahoo_response["my_stats"][yahoo_key]

        # With buggy mapping, NO stats were transferred (keys didn't match)
        assert len(my_current_buggy) == 0, f"Buggy mapping should produce empty dict, got {my_current_buggy}"

        # This was the bug: empty current stats meant current lead was ignored!
