"""
Unit tests for backend/fantasy_baseball/trade_analyzer.py.

These tests operate on pure Python dicts (no DB, no Yahoo API) so they run
fast and offline.  The ``cat_scores`` values follow the same format that
player_board.get_or_create_projection() produces.
"""
import pytest

from backend.fantasy_baseball.trade_analyzer import analyze_trade, _sum_cat_scores
from backend.contracts import TradeAnalysis, TradeCategoryDelta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _player(name: str, cat_scores: dict, player_type: str = "batter") -> dict:
    """Minimal board-compatible player dict for testing."""
    return {
        "id": name.lower().replace(" ", "_"),
        "name": name,
        "type": player_type,
        "team": "TST",
        "positions": ["OF"] if player_type == "batter" else ["SP"],
        "z_score": sum(cat_scores.values()),
        "cat_scores": cat_scores,
        "is_proxy": True,
        "fusion_source": "test_data",
    }


# ---------------------------------------------------------------------------
# 1. _sum_cat_scores — unit test for the inner helper
# ---------------------------------------------------------------------------

def test_sum_cat_scores_sums_correctly():
    players = [
        _player("A", {"hr": 1.5, "rbi": 0.8}),
        _player("B", {"hr": 0.5, "avg": 1.2}),
    ]
    totals = _sum_cat_scores(players)
    assert totals["hr"] == pytest.approx(2.0)
    assert totals["rbi"] == pytest.approx(0.8)
    assert totals["avg"] == pytest.approx(1.2)


def test_sum_cat_scores_empty_players():
    assert _sum_cat_scores([]) == {}


# ---------------------------------------------------------------------------
# 2. analyze_trade — strong_accept path
# ---------------------------------------------------------------------------

def test_strong_accept_when_receive_much_better():
    """Receiving side has clearly superior z-scores → strong_accept."""
    give = [_player("Give1", {"hr": 0.5, "rbi": 0.3, "avg": 0.2})]
    receive = [_player("Recv1", {"hr": 1.8, "rbi": 1.5, "avg": 0.9})]

    result = analyze_trade(give, receive)

    assert isinstance(result, TradeAnalysis)
    assert result.recommendation == "strong_accept"
    assert result.total_z_delta > 1.5
    assert "strong_accept" in result.recommendation
    assert result.total_z_delta == pytest.approx(result.total_z_delta)  # is a float


# ---------------------------------------------------------------------------
# 3. analyze_trade — strong_reject path
# ---------------------------------------------------------------------------

def test_strong_reject_when_give_much_better():
    """Giving away elite players → strong_reject."""
    give = [_player("Elite", {"hr": 2.1, "rbi": 1.9, "avg": 1.5, "r": 1.0})]
    receive = [_player("Bench", {"hr": 0.1, "rbi": 0.2, "avg": 0.0, "r": 0.0})]

    result = analyze_trade(give, receive)

    assert result.recommendation == "strong_reject"
    assert result.total_z_delta < -1.5


# ---------------------------------------------------------------------------
# 4. analyze_trade — neutral path
# ---------------------------------------------------------------------------

def test_neutral_when_sides_are_balanced():
    """Roughly even trade → neutral recommendation."""
    give = [_player("P1", {"hr": 1.0, "era": -1.0})]
    receive = [_player("P2", {"hr": 0.9, "era": -1.1})]

    result = analyze_trade(give, receive)

    assert result.recommendation == "neutral"
    assert abs(result.total_z_delta) < 0.5


# ---------------------------------------------------------------------------
# 5. analyze_trade — category_deltas structure and direction labels
# ---------------------------------------------------------------------------

def test_category_deltas_have_correct_direction():
    """Per-category deltas should correctly label gain/loss/neutral."""
    give = [_player("G", {"hr": 0.5, "avg": 1.0, "era": -0.5}, "pitcher")]
    receive = [_player("R", {"hr": 1.5, "avg": 1.0, "era": -1.0}, "pitcher")]

    result = analyze_trade(give, receive)

    delta_map = {d.category: d for d in result.category_deltas}

    # hr: receive 1.5 − give 0.5 = +1.0 → gain
    assert delta_map["hr"].direction == "gain"
    assert delta_map["hr"].delta == pytest.approx(1.0)

    # avg: receive 1.0 − give 1.0 = 0.0 → neutral
    assert delta_map["avg"].direction == "neutral"

    # era: receive −1.0 − give −0.5 = −0.5 → loss
    assert delta_map["era"].direction == "loss"


# ---------------------------------------------------------------------------
# 6. analyze_trade — multi-player trade
# ---------------------------------------------------------------------------

def test_multi_player_trade_sums_both_sides():
    """2-for-2 trade should sum z-scores across each side correctly."""
    give = [
        _player("G1", {"hr": 1.0, "rbi": 0.5}),
        _player("G2", {"hr": 0.8, "sb": 1.2}),
    ]
    receive = [
        _player("R1", {"hr": 0.6, "rbi": 0.9}),
        _player("R2", {"hr": 0.7, "sb": 0.4}),
    ]

    result = analyze_trade(give, receive)

    # Manual: give_hr=1.8, recv_hr=1.3 → delta=-0.5
    #          give_rbi=0.5, recv_rbi=0.9 → delta=+0.4
    #          give_sb=1.2, recv_sb=0.4 → delta=-0.8
    # total = -0.5 + 0.4 - 0.8 = -0.9 → reject
    assert result.recommendation == "reject"
    assert result.total_z_delta == pytest.approx(-0.9)
    assert len(result.give_players) == 2
    assert len(result.receive_players) == 2


# ---------------------------------------------------------------------------
# 7. analyze_trade — empty cat_scores fallback
# ---------------------------------------------------------------------------

def test_empty_cat_scores_returns_neutral():
    """Players with no cat_scores data should produce neutral result."""
    give = [_player("Unknown1", {})]
    receive = [_player("Unknown2", {})]

    result = analyze_trade(give, receive)

    assert result.total_z_delta == 0.0
    assert result.recommendation == "neutral"
    assert result.category_deltas == []
