"""Tests for bet_tracker: parse_pick and calculate_bet_outcome."""

import pytest
from unittest.mock import MagicMock
from backend.services.bet_tracker import parse_pick, calculate_bet_outcome, OutcomeResult


# ---------------------------------------------------------------------------
# parse_pick
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pick, expected_team, expected_spread", [
    ("Duke -4.5",       "Duke",   -4.5),
    ("Kansas +3",       "Kansas",  3.0),
    ("North Carolina -1.5", "North Carolina", -1.5),
    ("Michigan +10.5",  "Michigan", 10.5),
    ("Duke",            "Duke",   None),   # moneyline — no spread
    ("UConn -110",      "UConn",  -110.0), # handles negative-only picks
])
def test_parse_pick(pick, expected_team, expected_spread):
    team, spread = parse_pick(pick)
    assert team == expected_team
    assert spread == expected_spread


# ---------------------------------------------------------------------------
# calculate_bet_outcome helpers
# ---------------------------------------------------------------------------

def _make_game(home="Duke", away="Kansas", home_score=80, away_score=72):
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
# Spread outcomes
# ---------------------------------------------------------------------------

def test_home_favourite_covers():
    # Duke -4.5, wins by 8 → covers
    game = _make_game(home="Duke", away="Kansas", home_score=80, away_score=72)  # margin = 8
    bet  = _make_bet("Duke -4.5", odds=-110, dollars=10.0)
    result = calculate_bet_outcome(bet, game)
    assert result is not None
    assert result.outcome == 1
    assert result.profit_loss_dollars == pytest.approx(9.09, abs=0.01)


def test_home_favourite_fails_to_cover():
    # Duke -4.5, wins by 3 → does not cover
    game = _make_game(home="Duke", away="Kansas", home_score=75, away_score=72)  # margin = 3
    bet  = _make_bet("Duke -4.5", odds=-110, dollars=10.0)
    result = calculate_bet_outcome(bet, game)
    assert result is not None
    assert result.outcome == 0
    assert result.profit_loss_dollars == pytest.approx(-10.0)


def test_away_underdog_covers():
    # Kansas +3, loses by 2 → covers (margin from Kansas' perspective = -2, -2 + 3 = +1 > 0)
    game = _make_game(home="Duke", away="Kansas", home_score=78, away_score=76)
    bet  = _make_bet("Kansas +3", odds=-110, dollars=10.0)
    result = calculate_bet_outcome(bet, game)
    assert result is not None
    assert result.outcome == 1


def test_away_underdog_fails_to_cover():
    # Kansas +3, loses by 5 → does not cover
    game = _make_game(home="Duke", away="Kansas", home_score=80, away_score=75)
    bet  = _make_bet("Kansas +3", odds=-110, dollars=10.0)
    result = calculate_bet_outcome(bet, game)
    assert result is not None
    assert result.outcome == 0


def test_push():
    # Duke -4, wins by exactly 4 → push
    game = _make_game(home="Duke", away="Kansas", home_score=80, away_score=76)
    bet  = _make_bet("Duke -4.0", odds=-110, dollars=10.0)
    result = calculate_bet_outcome(bet, game)
    assert result is not None
    assert result.outcome == -1
    assert result.profit_loss_dollars == 0.0


# ---------------------------------------------------------------------------
# Moneyline outcomes
# ---------------------------------------------------------------------------

def test_moneyline_win():
    game = _make_game(home="Duke", away="Kansas", home_score=80, away_score=70)
    bet  = _make_bet("Duke", odds=-150, dollars=15.0)
    result = calculate_bet_outcome(bet, game)
    assert result is not None
    assert result.outcome == 1
    assert result.profit_loss_dollars == pytest.approx(10.0, abs=0.01)  # 15 * 100/150


def test_moneyline_loss():
    game = _make_game(home="Duke", away="Kansas", home_score=68, away_score=70)
    bet  = _make_bet("Duke", odds=-150, dollars=15.0)
    result = calculate_bet_outcome(bet, game)
    assert result is not None
    assert result.outcome == 0
    assert result.profit_loss_dollars == pytest.approx(-15.0)


# ---------------------------------------------------------------------------
# P&L calculation for positive American odds
# ---------------------------------------------------------------------------

def test_plus_odds_payout():
    # +120 favourite bet wins → profit = 10 * 120/100 = 12.0
    game = _make_game(home="Duke", away="Kansas", home_score=80, away_score=70)
    bet  = _make_bet("Duke -1.5", odds=120, dollars=10.0)
    result = calculate_bet_outcome(bet, game)
    assert result is not None
    assert result.outcome == 1
    assert result.profit_loss_dollars == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# Missing scores
# ---------------------------------------------------------------------------

def test_missing_scores_returns_none():
    game = _make_game()
    game.home_score = None
    bet = _make_bet("Duke -4.5")
    assert calculate_bet_outcome(bet, game) is None
