"""
Tests for OddsMonitor Level 5 Real-Time Pulse — Verdict Flips and O-10 BET Adverse Moves.

Run:
    pytest tests/test_odds_monitor.py -v
"""

import pytest
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timedelta
from backend.services.odds_monitor import OddsMonitor, LineMovement
from backend.betting_model import GameAnalysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tipoff_iso(minutes_from_now: float) -> str:
    """Return a UTC ISO-8601 timestamp N minutes from now with a 'Z' suffix.

    The 'Z' suffix ensures datetime.fromisoformat() (after .replace("Z", "+00:00"))
    returns an aware datetime, keeping minutes_to_tipoff arithmetic timezone-safe.
    """
    return (datetime.utcnow() + timedelta(minutes=minutes_from_now)).isoformat() + "Z"


def _make_bet_engine(reanalyze_verdict: str = "Bet 1.0u @ -110") -> MagicMock:
    """Return a mock ReanalysisEngine whose original verdict is a BET."""
    mock_analysis = MagicMock(spec=GameAnalysis)
    mock_analysis.verdict = reanalyze_verdict
    mock_analysis.edge_conservative = 0.05

    engine = MagicMock()
    engine.reanalyze.return_value = mock_analysis
    engine._ctx.original_verdict = "Bet 1.0u @ -110"
    engine._ctx.recommended_units = 1.0
    return engine


def _make_pass_engine(reanalyze_verdict: str = "Bet 1.0u @ -110") -> MagicMock:
    """Return a mock ReanalysisEngine whose original verdict was PASS."""
    mock_analysis = MagicMock(spec=GameAnalysis)
    mock_analysis.verdict = reanalyze_verdict
    mock_analysis.edge_conservative = 0.05

    engine = MagicMock()
    engine.reanalyze.return_value = mock_analysis
    engine._ctx.original_verdict = "PASS"
    engine._ctx.recommended_units = 1.0
    return engine


def _two_poll(monitor: OddsMonitor, game_v1: dict, game_v2: dict) -> dict:
    """Run two sequential polls and return the result of the second."""
    with patch.object(monitor._client, "get_todays_games", return_value=[game_v1]):
        monitor.poll()
    with patch.object(monitor._client, "get_todays_games", return_value=[game_v2]):
        return monitor.poll()


# ---------------------------------------------------------------------------
# Existing test — VERDICT_FLIP
# ---------------------------------------------------------------------------

def test_odds_monitor_verdict_flip():
    """
    A significant spread move on a PASS game that flips to BET should emit
    a VERDICT_FLIP event.
    """
    monitor = OddsMonitor(api_key="test")

    mock_analysis = MagicMock(spec=GameAnalysis)
    mock_analysis.verdict = "Bet 1.0u @ -110"
    mock_analysis.edge_conservative = 0.05

    mock_engine = MagicMock()
    mock_engine.reanalyze.return_value = mock_analysis
    mock_engine._ctx.original_verdict = "PASS"

    game_id = "test_game_1"
    game_key = "Away@Home"
    monitor._reanalysis_cache[game_key] = mock_engine

    callback = MagicMock()
    monitor.on_significant_move(callback)

    game_v1 = {
        "game_id": game_id,
        "home_team": "Home",
        "away_team": "Away",
        "best_spread": -3.5,
        "best_spread_odds": -110,
        "best_total": 145.0,
        "commence_time": datetime.utcnow().isoformat(),
    }

    game_v2 = {
        "game_id": game_id,
        "home_team": "Home",
        "away_team": "Away",
        "best_spread": -5.5,
        "best_spread_odds": -110,
        "best_total": 145.0,
        "commence_time": datetime.utcnow().isoformat(),
    }

    with patch.object(monitor._client, "get_todays_games", return_value=[game_v1]):
        monitor.poll()

    with patch.object(monitor._client, "get_todays_games", return_value=[game_v2]):
        result = monitor.poll()

    assert result["significant_movements"] == 1

    assert callback.called
    movement = callback.call_args[0][0]
    assert isinstance(movement, LineMovement)
    assert movement.event_type == "VERDICT_FLIP"
    assert movement.fresh_analysis == mock_analysis
    assert movement.new_value == -5.5

    mock_engine.reanalyze.assert_called_with(new_spread=-5.5)


# ---------------------------------------------------------------------------
# O-10 tests — BET_ADVERSE_MOVE
# ---------------------------------------------------------------------------

class TestBetAdverseMove:
    """O-10: BET_ADVERSE_MOVE detection in the golden window."""

    GAME_ID = "o10_game_1"
    GAME_KEY = "Visitor@HomeTeam"

    def _base_game(self, spread: float, minutes_from_now: float) -> dict:
        return {
            "game_id": self.GAME_ID,
            "home_team": "HomeTeam",
            "away_team": "Visitor",
            "best_spread": spread,
            "best_spread_odds": -110,
            "best_total": 150.0,
            "commence_time": _tipoff_iso(minutes_from_now),
        }

    # ------------------------------------------------------------------

    def test_adverse_move_not_fired_outside_golden_window(self):
        """
        A 3.0-point spread move at T-180 min (outside the 120-min golden
        window) on a BET game should NOT fire BET_ADVERSE_MOVE.
        """
        monitor = OddsMonitor(api_key="test")
        engine = _make_bet_engine()
        monitor._reanalysis_cache[self.GAME_KEY] = engine

        adverse_cb = MagicMock()
        monitor.on_adverse_move(adverse_cb)

        # T-180 min: tipoff is 3 hours away — outside golden window
        game_v1 = self._base_game(spread=-3.5, minutes_from_now=180)
        game_v2 = self._base_game(spread=-6.5, minutes_from_now=180)  # delta = -3.0

        result = _two_poll(monitor, game_v1, game_v2)

        # The move is >= SPREAD_MOVE_THRESHOLD (1.5) so it IS significant,
        # but T-180 is outside the golden window — adverse callback must NOT fire.
        assert result["significant_movements"] >= 1
        adverse_cb.assert_not_called()
        assert self.GAME_KEY not in monitor._bet_adverse_fired

    # ------------------------------------------------------------------

    def test_adverse_move_fired_in_golden_window_bet(self):
        """
        A 2.5-point spread move at T-90 min on a BET game should fire
        BET_ADVERSE_MOVE, set event_type, and call the adverse callback.
        """
        monitor = OddsMonitor(api_key="test")
        engine = _make_bet_engine()
        monitor._reanalysis_cache[self.GAME_KEY] = engine

        adverse_cb = MagicMock()
        monitor.on_adverse_move(adverse_cb)

        # T-90 min: inside golden window
        game_v1 = self._base_game(spread=-3.5, minutes_from_now=90)
        game_v2 = self._base_game(spread=-6.0, minutes_from_now=90)  # delta = -2.5

        with patch(
            "backend.services.coordinator.escalate_if_needed", return_value=True
        ) as mock_escalate:
            result = _two_poll(monitor, game_v1, game_v2)

        assert result["significant_movements"] >= 1

        # Adverse callback must have fired exactly once
        adverse_cb.assert_called_once()
        movement = adverse_cb.call_args[0][0]
        assert isinstance(movement, LineMovement)
        assert movement.event_type == "BET_ADVERSE_MOVE"
        assert movement.field == "spread"

        # Dedup set must be populated
        assert self.GAME_KEY in monitor._bet_adverse_fired

        # Escalation must have been triggered with VOLATILE
        mock_escalate.assert_called_once()
        call_kwargs = mock_escalate.call_args[1]
        assert call_kwargs["integrity_verdict"] == "VOLATILE"
        assert call_kwargs["game_key"] == self.GAME_KEY

    # ------------------------------------------------------------------

    def test_adverse_move_deduped(self):
        """
        When the same BET game moves adversely a second time, the
        BET_ADVERSE_MOVE alert must NOT fire again (one-fire-per-game).
        """
        monitor = OddsMonitor(api_key="test")
        engine = _make_bet_engine()
        monitor._reanalysis_cache[self.GAME_KEY] = engine

        adverse_cb = MagicMock()
        monitor.on_adverse_move(adverse_cb)

        # T-60 min: inside golden window
        game_v1 = self._base_game(spread=-3.5, minutes_from_now=60)
        game_v2 = self._base_game(spread=-6.0, minutes_from_now=60)  # delta = -2.5, fires
        game_v3 = self._base_game(spread=-8.5, minutes_from_now=60)  # delta = -2.5 again

        with patch("backend.services.coordinator.escalate_if_needed", return_value=True):
            _two_poll(monitor, game_v1, game_v2)
            with patch.object(monitor._client, "get_todays_games", return_value=[game_v3]):
                monitor.poll()

        # Adverse callback must have fired exactly once across both moves
        assert adverse_cb.call_count == 1

    # ------------------------------------------------------------------

    def test_adverse_move_not_fired_below_threshold(self):
        """
        A 1.5-point move at T-60 min on a BET game is below
        BET_ADVERSE_MOVE_THRESHOLD (2.0 pts) and must NOT fire
        BET_ADVERSE_MOVE (it may still be a regular significant move).
        """
        monitor = OddsMonitor(api_key="test")
        engine = _make_bet_engine()
        monitor._reanalysis_cache[self.GAME_KEY] = engine

        adverse_cb = MagicMock()
        monitor.on_adverse_move(adverse_cb)

        # T-60 min: inside golden window; delta = 1.5 (exactly SPREAD_MOVE_THRESHOLD)
        game_v1 = self._base_game(spread=-3.5, minutes_from_now=60)
        game_v2 = self._base_game(spread=-5.0, minutes_from_now=60)  # delta = -1.5

        result = _two_poll(monitor, game_v1, game_v2)

        # The move IS a regular significant move (1.5 >= SPREAD_MOVE_THRESHOLD),
        # but it is < BET_ADVERSE_MOVE_THRESHOLD (2.0), so no BET_ADVERSE_MOVE.
        assert result["significant_movements"] >= 1
        adverse_cb.assert_not_called()
        assert self.GAME_KEY not in monitor._bet_adverse_fired


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
