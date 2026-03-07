"""
Tests for OddsMonitor Level 5 Real-Time Pulse — Verdict Flips and O-10 BET Adverse Moves.

Run:
    pytest tests/test_odds_monitor.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from backend.services.odds_monitor import OddsMonitor, LineMovement
from backend.betting_model import GameAnalysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bet_engine(reanalyze_verdict: str = "Bet 1.0u @ -110") -> MagicMock:
    """Return a mock ReanalysisEngine whose original nightly verdict was a BET."""
    mock_analysis = MagicMock(spec=GameAnalysis)
    mock_analysis.verdict = reanalyze_verdict
    mock_analysis.edge_conservative = 0.05

    engine = MagicMock()
    engine.reanalyze.return_value = mock_analysis
    engine._ctx.original_verdict = "Bet 1.0u @ -110"
    engine._ctx.recommended_units = 1.0
    return engine


def _make_spread_movement(
    game_key: str,
    delta: float,
    minutes_to_tipoff: float,
) -> LineMovement:
    """Build a pre-constructed significant spread LineMovement."""
    away, home = game_key.split("@")
    return LineMovement(
        game_id=game_key,
        home_team=home,
        away_team=away,
        field="spread",
        old_value=-3.5,
        new_value=-3.5 + delta,
        delta=delta,
        timestamp=datetime.utcnow(),
        is_significant=True,
        minutes_to_tipoff=minutes_to_tipoff,
    )


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
#
# Strategy: bypass the timestamp arithmetic in _detect_movements by
# injecting pre-built LineMovement objects with exact minutes_to_tipoff
# values into the significant-movement loop via a patch on poll().
# We patch the portion of poll() that builds the `significant` list,
# replacing it with our controlled fixture movement.
# ---------------------------------------------------------------------------

class TestBetAdverseMove:
    """O-10: BET_ADVERSE_MOVE detection in the golden window."""

    GAME_KEY = "Visitor@HomeTeam"

    def _run_poll_with_movement(
        self, monitor: OddsMonitor, movement: LineMovement
    ) -> dict:
        """
        Run poll() with a single injected significant movement.

        Patches _detect_movements to return the provided movement so that
        poll()'s significant-movements loop processes it exactly as if it
        came from real API data, without relying on time arithmetic.
        """
        fake_game = {
            "game_id": movement.game_id,
            "home_team": movement.home_team,
            "away_team": movement.away_team,
            "best_spread": movement.new_value,
            "best_spread_odds": -110,
            "best_total": 150.0,
            "commence_time": None,  # irrelevant — _detect_movements is patched
        }

        # Seed history so the per-game detection branch runs
        from backend.services.odds_monitor import LineSnapshot
        monitor._history[movement.game_id] = [
            LineSnapshot(
                game_id=movement.game_id,
                home_team=movement.home_team,
                away_team=movement.away_team,
                spread=movement.old_value,
                spread_odds=-110,
                total=150.0,
                moneyline_home=None,
                moneyline_away=None,
            )
        ]

        with patch.object(monitor._client, "get_todays_games", return_value=[fake_game]):
            with patch.object(
                monitor, "_detect_movements", return_value=[movement]
            ):
                return monitor.poll()

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

        movement = _make_spread_movement(
            game_key=self.GAME_KEY,
            delta=-3.0,
            minutes_to_tipoff=180.0,  # outside golden window (> 120)
        )

        with patch("backend.services.coordinator.escalate_if_needed", return_value=True):
            self._run_poll_with_movement(monitor, movement)

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

        movement = _make_spread_movement(
            game_key=self.GAME_KEY,
            delta=-2.5,
            minutes_to_tipoff=90.0,  # inside golden window (<= 120)
        )

        with patch(
            "backend.services.coordinator.escalate_if_needed", return_value=True
        ) as mock_escalate:
            self._run_poll_with_movement(monitor, movement)

        # Adverse callback must have fired exactly once
        adverse_cb.assert_called_once()
        fired_movement = adverse_cb.call_args[0][0]
        assert isinstance(fired_movement, LineMovement)
        assert fired_movement.event_type == "BET_ADVERSE_MOVE"
        assert fired_movement.field == "spread"

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

        movement_first = _make_spread_movement(
            game_key=self.GAME_KEY,
            delta=-2.5,
            minutes_to_tipoff=90.0,
        )
        movement_second = _make_spread_movement(
            game_key=self.GAME_KEY,
            delta=-2.5,
            minutes_to_tipoff=85.0,
        )

        with patch("backend.services.coordinator.escalate_if_needed", return_value=True):
            self._run_poll_with_movement(monitor, movement_first)
            self._run_poll_with_movement(monitor, movement_second)

        # Adverse callback must have fired exactly once across both movements
        assert adverse_cb.call_count == 1

    # ------------------------------------------------------------------

    def test_adverse_move_not_fired_below_threshold(self):
        """
        A 1.5-point move at T-60 min on a BET game is below
        BET_ADVERSE_MOVE_THRESHOLD (2.0 pts) and must NOT fire
        BET_ADVERSE_MOVE.  The regular significant-move callback
        may still fire but the adverse callback must stay silent.
        """
        monitor = OddsMonitor(api_key="test")
        engine = _make_bet_engine()
        monitor._reanalysis_cache[self.GAME_KEY] = engine

        adverse_cb = MagicMock()
        monitor.on_adverse_move(adverse_cb)

        movement = _make_spread_movement(
            game_key=self.GAME_KEY,
            delta=-1.5,           # exactly SPREAD_MOVE_THRESHOLD but < BET_ADVERSE_MOVE_THRESHOLD
            minutes_to_tipoff=60.0,  # inside golden window
        )

        with patch("backend.services.coordinator.escalate_if_needed", return_value=True):
            self._run_poll_with_movement(monitor, movement)

        adverse_cb.assert_not_called()
        assert self.GAME_KEY not in monitor._bet_adverse_fired


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
