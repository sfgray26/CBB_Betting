"""
EMAC-064: Tests for _resolve_home_away and the calculate_bet_outcome integration path.

These tests verify the mascot-aware team resolution fix in bet_tracker.py.
No database is required — BetLog and Game objects are mocked via MagicMock.

Run:
    pytest tests/test_bet_settlement_fix.py -v
"""

import pytest
from unittest.mock import MagicMock

from backend.services.bet_tracker import _resolve_home_away, calculate_bet_outcome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_game(home, away, home_score=None, away_score=None):
    game = MagicMock()
    game.home_team = home
    game.away_team = away
    game.home_score = home_score
    game.away_score = away_score
    return game


def _make_bet(pick, odds=-110, dollars=10.0, bankroll=1000.0):
    bet = MagicMock()
    bet.pick = pick
    bet.odds_taken = odds
    bet.bet_size_dollars = dollars
    bet.bankroll_at_bet = bankroll
    return bet


# ---------------------------------------------------------------------------
# TestResolveHomeAway — pure resolution logic, no scores needed
# ---------------------------------------------------------------------------

class TestResolveHomeAway:

    def test_exact_match_home(self):
        """Step 1: exact case-sensitive match selects home."""
        assert _resolve_home_away("Duke", "Duke", "UNC") == "home"

    def test_exact_match_away(self):
        """Step 1: exact case-sensitive match selects away."""
        assert _resolve_home_away("UNC", "Duke", "UNC") == "away"

    def test_case_insensitive_home(self):
        """Step 1: lowercase pick still resolves to home."""
        assert _resolve_home_away("duke", "Duke", "UNC") == "home"

    def test_case_insensitive_away(self):
        """Step 1: uppercase pick still resolves to away."""
        assert _resolve_home_away("UNC", "Duke", "unc") == "away"

    def test_mascot_mismatch_home(self):
        """Step 2: mascot stripped from pick, then exact match to home team."""
        # "Samford Bulldogs" -> strip "Bulldogs" -> "Samford" == home_team
        assert _resolve_home_away("Samford Bulldogs", "Samford", "Furman") == "home"

    def test_mascot_mismatch_away(self):
        """Step 2: mascot stripped from pick, then exact match to away team."""
        # "Furman Paladins" -> strip "Paladins" -> "Furman" == away_team
        assert _resolve_home_away("Furman Paladins", "Samford", "Furman") == "away"

    def test_spread_notation_stripped_before_call(self):
        """
        Calling code strips spread before calling _resolve_home_away.
        Verify that calling with the already-stripped team name resolves correctly.
        pick "UNC +3.5" -> parse_pick gives team="UNC" -> resolve against Duke/UNC.
        """
        # Simulate what calculate_bet_outcome does: parse_pick extracts "UNC"
        assert _resolve_home_away("UNC", "Duke", "UNC") == "away"

    def test_unresolvable_returns_none(self):
        """Step 3 fallback: pick team matches neither side -> None."""
        assert _resolve_home_away("Kentucky", "Duke", "UNC") is None

    def test_empty_string_pick_returns_none(self):
        """Empty pick string must not crash and must return None."""
        result = _resolve_home_away("", "Duke", "UNC")
        assert result is None

    def test_pick_with_punctuation_st_johns(self):
        """
        Step 3 fuzzy resolver: punctuation variant 'St. John's' resolves to
        'St. John's NY' (home) via fuzzy matching.
        The apostrophe-period combination should not block resolution.
        """
        # Use the canonical KenPom form as the home team so Step 2 or Step 3
        # has a concrete exact target to land on.
        result = _resolve_home_away("St. John's Red Storm", "St. John's NY", "Georgetown")
        # "St. John's Red Storm" is in ODDS_TO_KENPOM -> "St. John's NY"
        # normalize_team_name will map it; then resolved == home_team
        assert result == "home"

    def test_abbreviation_gonzaga_via_mascot_strip(self):
        """
        'Gonzaga Bulldogs' is in the manual overrides mapping -> 'Gonzaga'.
        After mascot strip or fuzzy, should resolve against 'Gonzaga' home team.
        """
        result = _resolve_home_away("Gonzaga Bulldogs", "Gonzaga", "Saint Mary's CA")
        assert result == "home"

    def test_home_case_mixed(self):
        """Step 1: mixed-case variations of the same word are equal."""
        assert _resolve_home_away("KANSAS", "Kansas", "Duke") == "home"

    def test_away_mascot_multi_word(self):
        """
        Multi-word mascot: 'Blue Devils' should be stripped leaving 'Duke',
        resolving to away team 'Duke'.
        """
        assert _resolve_home_away("Duke Blue Devils", "UNC", "Duke") == "away"


# ---------------------------------------------------------------------------
# TestCalculateBetOutcomeIntegration — verify fix is wired into the full path
# ---------------------------------------------------------------------------

class TestCalculateBetOutcomeIntegration:

    def test_mascot_in_pick_home_team_wins_covers(self):
        """
        Integration: pick 'Samford Bulldogs -1.5' with mascot in name.
        Home team 'Samford' wins 72-68, margin=4, cover_margin = 4 + (-1.5) = 2.5 > 0.
        Expected: outcome == win (1).
        """
        game = _make_game("Samford", "Furman", home_score=72, away_score=68)
        bet = _make_bet("Samford Bulldogs -1.5", odds=-110, dollars=10.0)
        result = calculate_bet_outcome(bet, game)
        assert result is not None, "Expected OutcomeResult, got None"
        assert result.outcome == 1, f"Expected win (1), got {result.outcome}"
        assert result.profit_loss_dollars == pytest.approx(9.09, abs=0.01)

    def test_mascot_in_pick_away_team_covers(self):
        """
        Integration: pick 'Furman Paladins +4' (away underdog).
        Samford wins 72-68, margin from Furman's perspective = -4, cover_margin = -4 + 4 = 0 -> push.
        """
        game = _make_game("Samford", "Furman", home_score=72, away_score=68)
        bet = _make_bet("Furman Paladins +4", odds=-110, dollars=10.0)
        result = calculate_bet_outcome(bet, game)
        assert result is not None
        assert result.outcome == -1  # push: -4 + 4 = 0

    def test_mascot_in_pick_away_beats_spread(self):
        """
        Integration: pick 'Furman Paladins +5' (away underdog with buffer).
        Samford wins 72-68, Furman margin = -4, cover_margin = -4 + 5 = +1 > 0 -> win.
        """
        game = _make_game("Samford", "Furman", home_score=72, away_score=68)
        bet = _make_bet("Furman Paladins +5", odds=-110, dollars=10.0)
        result = calculate_bet_outcome(bet, game)
        assert result is not None
        assert result.outcome == 1

    def test_unresolvable_pick_returns_none(self):
        """
        Integration: pick team does not match either game participant.
        calculate_bet_outcome must return None rather than raising or guessing.
        """
        game = _make_game("Duke", "UNC", home_score=80, away_score=72)
        bet = _make_bet("Kentucky -3.0", odds=-110, dollars=10.0)
        result = calculate_bet_outcome(bet, game)
        assert result is None

    def test_missing_scores_still_returns_none(self):
        """
        Sanity check: if scores are missing, result is None regardless of pick.
        """
        game = _make_game("Samford", "Furman", home_score=None, away_score=None)
        bet = _make_bet("Samford Bulldogs -1.5", odds=-110, dollars=10.0)
        assert calculate_bet_outcome(bet, game) is None

    def test_exact_team_name_unchanged_path(self):
        """
        Regression guard: non-mascot picks still work via the exact-match path,
        ensuring the mascot fix did not break the primary resolution branch.
        """
        game = _make_game("Duke", "UNC", home_score=82, away_score=74)
        bet = _make_bet("Duke -4.5", odds=-110, dollars=10.0)
        result = calculate_bet_outcome(bet, game)
        assert result is not None
        assert result.outcome == 1  # margin=8, cover_margin=8-4.5=3.5 > 0

    def test_case_insensitive_pick_integration(self):
        """
        Integration: lowercase team name in pick resolves correctly.
        """
        game = _make_game("Kansas", "Duke", home_score=77, away_score=73)
        bet = _make_bet("kansas -2.5", odds=-110, dollars=10.0)
        result = calculate_bet_outcome(bet, game)
        assert result is not None
        # margin=4, cover_margin = 4 - 2.5 = 1.5 > 0 -> win
        assert result.outcome == 1
