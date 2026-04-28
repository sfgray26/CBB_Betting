"""Tests for OpenClawAutonomousLoop."""
import pytest
from unittest.mock import MagicMock, patch, call


def _make_loop(moves=None, brief_raises=False, detector_raises=False):
    """Build an OpenClawAutonomousLoop with mocked dependencies."""
    from backend.services.openclaw_autonomous import OpenClawAutonomousLoop
    from backend.services.discord_router import DiscordRouter

    detector = MagicMock()
    if detector_raises:
        detector.get_top_moves.side_effect = RuntimeError("Yahoo down")
    else:
        detector.get_top_moves.return_value = moves or []

    router = MagicMock(spec=DiscordRouter)
    router.should_flush.return_value = False

    loop = OpenClawAutonomousLoop(detector, router, my_roster=[], opponent_roster=[])

    # generate_and_send_morning_brief is async; force a plain MagicMock so that
    # side_effect raises synchronously even when called without await.
    if brief_raises:
        brief_patch = patch(
            "backend.services.openclaw_briefs_improved.generate_and_send_morning_brief",
            new=MagicMock(side_effect=RuntimeError("brief failed")),
        )
    else:
        brief_patch = patch(
            "backend.services.openclaw_briefs_improved.generate_and_send_morning_brief",
            new=MagicMock(return_value=True),
        )

    return loop, detector, router, brief_patch


_PAUSED = pytest.mark.xfail(
    reason="OpenClaw paused 2026-04-21 until baseball module complete — stub returns immediately",
    strict=True,
)


class TestOpenClawAutonomousLoop:

    @_PAUSED
    def test_run_morning_workflow_returns_summary_keys(self):
        loop, _, _, brief_patch = _make_loop(moves=[])
        with brief_patch:
            summary = loop.run_morning_workflow()
        assert set(summary.keys()) == {"brief_sent", "moves_evaluated", "high_impact", "digest_sent"}

    @_PAUSED
    def test_high_impact_move_fires_immediate_alert(self):
        high_impact_move = {
            "add_player": {"name": "Westburg"},
            "drop_player_name": "Dead2B",
            "win_prob_gain": 0.08,
        }
        loop, _, router, brief_patch = _make_loop(moves=[high_impact_move])
        with brief_patch:
            summary = loop.run_morning_workflow()
        assert summary["high_impact"] == 1
        router.route.assert_called_once()
        pkg = router.route.call_args[0][0]
        assert pkg.priority == 2
        assert pkg.mention_admin is True
        assert "HIGH IMPACT" in pkg.message

    @_PAUSED
    def test_low_impact_move_goes_to_digest(self):
        low_move = {
            "add_player": {"name": "LowFA"},
            "drop_player_name": "Weak",
            "win_prob_gain": 0.02,
        }
        loop, _, router, brief_patch = _make_loop(moves=[low_move])
        with brief_patch, patch(
            "backend.services.discord_notifier.send_batch_digest", return_value=True
        ) as mock_digest:
            summary = loop.run_morning_workflow()
        assert summary["high_impact"] == 0
        mock_digest.assert_called_once()
        items_arg = mock_digest.call_args[0][1]
        assert any("LowFA" in item for item in items_arg)

    @_PAUSED
    def test_brief_failure_does_not_abort_workflow(self):
        moves = [{"add_player": {"name": "X"}, "drop_player_name": "Y", "win_prob_gain": 0.01}]
        loop, _, _, brief_patch = _make_loop(moves=moves, brief_raises=True)
        with brief_patch, patch("backend.services.discord_notifier.send_batch_digest", return_value=True):
            summary = loop.run_morning_workflow()
        assert summary["brief_sent"] is False
        assert summary["moves_evaluated"] == 1

    def test_detector_failure_returns_zero_moves(self):
        loop, _, _, brief_patch = _make_loop(detector_raises=True)
        with brief_patch:
            summary = loop.run_morning_workflow()
        assert summary["moves_evaluated"] == 0

    @_PAUSED
    def test_batch_digest_called_with_correct_items(self):
        moves = [
            {"add_player": {"name": "Alpha"}, "drop_player_name": "Z", "win_prob_gain": 0.01},
            {"add_player": {"name": "Beta"}, "drop_player_name": "Z", "win_prob_gain": 0.02},
        ]
        loop, _, _, brief_patch = _make_loop(moves=moves)
        with brief_patch, patch(
            "backend.services.discord_notifier.send_batch_digest", return_value=True
        ) as mock_digest:
            loop.run_morning_workflow()
        items = mock_digest.call_args[0][1]
        assert len(items) == 2
        names_in_items = " ".join(items)
        assert "Alpha" in names_in_items
        assert "Beta" in names_in_items

    def test_update_rosters_replaces_context(self):
        from backend.services.openclaw_autonomous import OpenClawAutonomousLoop
        loop = OpenClawAutonomousLoop(MagicMock(), MagicMock())
        new_my = [{"name": "NewPlayer"}]
        new_opp = [{"name": "OppPlayer"}]
        loop.update_rosters(new_my, new_opp)
        assert loop.my_roster is new_my
        assert loop.opponent_roster is new_opp

    @_PAUSED
    def test_discord_router_flush_called_when_threshold_met(self):
        loop, _, router, brief_patch = _make_loop(moves=[])
        router.should_flush.return_value = True
        with brief_patch:
            loop.run_morning_workflow()
        router.flush_batch.assert_called_once()
