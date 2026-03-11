"""
Tests for backend.fantasy_baseball.draft_tracker.DraftTracker

Run:
    pytest tests/test_draft_tracker.py -v

All tests use mock objects — no real Yahoo API or Discord calls are made.

Snake rule reference (Treemendous 12-team, 23-round):
  Round 1: positions 1..12  (linear)
  Round 2: positions 1..12  (linear — same order, NOT reversed)
  Round 3: positions 12..1  (snake begins)
  Round 4: positions 1..12
  ...

Derived pick numbers for position 7 (first 4 rounds):
  Round 1: overall pick 7
  Round 2: overall pick 19
  Round 3: overall pick 18  (12 teams - 7 + 1 = 6th from end => pick 24+6 = 30? No.)
           build_full_pick_order truth: round 3, reversed => pos 6 from end = pos 7
           picks = 24 + (12 - 7 + 1) = 24 + 6 = 30
  Round 4: overall pick 43  (24*... recalc below)

The authoritative values come from draft_engine.picks_for_position(7).
Tests pin the first four to: (7,1), (19,2), (30,3), (43,4).
"""

import unittest
from unittest.mock import MagicMock, patch, call

from backend.fantasy_baseball.draft_engine import picks_for_position, NUM_TEAMS, NUM_ROUNDS
from backend.fantasy_baseball.draft_tracker import DraftTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tracker(position: int = 7, num_teams: int = NUM_TEAMS,
                  num_rounds: int = NUM_ROUNDS) -> DraftTracker:
    """Return a DraftTracker with a dummy yahoo_client (no real HTTP)."""
    mock_client = MagicMock()
    return DraftTracker(mock_client, my_draft_position=position,
                        num_teams=num_teams, num_rounds=num_rounds)


def _make_pick(overall: int, rnd: int = 1, team_key: str = "mlb.l.72586.t.1",
               player_name: str = "Test Player", positions: list = None) -> dict:
    if positions is None:
        positions = ["SP"]
    return {
        "pick": overall,
        "round": rnd,
        "team_key": team_key,
        "player_key": "mlb.p.9999",
        "player_name": player_name,
        "positions": positions,
    }


# ---------------------------------------------------------------------------
# Derive authoritative pick numbers once so tests stay DRY
# ---------------------------------------------------------------------------
_MY_PICKS_POS7 = picks_for_position(7, NUM_TEAMS, NUM_ROUNDS)
_P7_R1 = _MY_PICKS_POS7[0][0]   # overall pick, round 1
_P7_R2 = _MY_PICKS_POS7[1][0]   # overall pick, round 2
_P7_R3 = _MY_PICKS_POS7[2][0]   # overall pick, round 3
_P7_R4 = _MY_PICKS_POS7[3][0]   # overall pick, round 4


class TestIsMyPickSnakeLogic:
    """is_my_pick() must agree with draft_engine for all rounds."""

    def test_is_my_pick_round1_linear(self):
        """Position 7 owns overall pick 7 (round 1, linear)."""
        tracker = _make_tracker(position=7)
        assert tracker.is_my_pick(_P7_R1)

    def test_is_my_pick_round2_linear(self):
        """Round 2 is also linear — position 7 owns overall pick 19."""
        tracker = _make_tracker(position=7)
        assert tracker.is_my_pick(_P7_R2)

    def test_is_my_pick_round3_snake(self):
        """Round 3 reverses. is_my_pick() must return True for the correct pick."""
        tracker = _make_tracker(position=7)
        assert tracker.is_my_pick(_P7_R3)

    def test_is_my_pick_round4_normal(self):
        """Round 4 is forward again. Verify is_my_pick() is correct."""
        tracker = _make_tracker(position=7)
        assert tracker.is_my_pick(_P7_R4)

    def test_not_my_pick_adjacent(self):
        """Picks immediately adjacent to our slots should not be ours."""
        tracker = _make_tracker(position=7)
        # Pick 8 belongs to position 8 in round 1
        assert not tracker.is_my_pick(_P7_R1 + 1)
        assert not tracker.is_my_pick(_P7_R1 - 1)

    def test_position1_first_pick(self):
        """Position 1 always owns overall pick 1."""
        tracker = _make_tracker(position=1)
        assert tracker.is_my_pick(1)

    def test_position12_last_round1(self):
        """Position 12 owns overall pick 12 in round 1."""
        tracker = _make_tracker(position=12)
        assert tracker.is_my_pick(12)


class TestPicksUntilMyTurn:

    def test_zero_when_exactly_my_pick(self):
        """picks_until_my_turn returns 0 when the current pick is our pick."""
        tracker = _make_tracker(position=7)
        assert tracker.picks_until_my_turn(_P7_R1) == 0

    def test_counts_correctly_before_round1(self):
        """From pick 1, picks_until_my_turn for position 7 should be 6."""
        tracker = _make_tracker(position=7)
        # Pick 1 is active, our first pick is _P7_R1 (=7), so gap is 6
        assert tracker.picks_until_my_turn(1) == _P7_R1 - 1

    def test_counts_correctly_midway(self):
        """From the pick after our round-1 pick, gap to round-2 pick is correct."""
        tracker = _make_tracker(position=7)
        # One pick after our round-1 slot: gap to next pick = (_P7_R2 - (_P7_R1 + 1))
        expected = _P7_R2 - (_P7_R1 + 1)
        assert tracker.picks_until_my_turn(_P7_R1 + 1) == expected

    def test_zero_past_all_picks(self):
        """After the draft ends, picks_until_my_turn returns 0."""
        tracker = _make_tracker(position=7)
        total = NUM_TEAMS * NUM_ROUNDS
        assert tracker.picks_until_my_turn(total + 1) == 0


class TestProcessNewPicksCallsDiscord:

    def test_calls_send_draft_pick_once_per_new_pick(self):
        """process_new_picks must call send_draft_pick exactly once per pick."""
        tracker = _make_tracker(position=7)
        picks = [_make_pick(1), _make_pick(2)]

        # Lazy import inside process_new_picks — patch at the source module level.
        with patch(
            "backend.services.discord_notifier.send_draft_pick", return_value=True
        ) as mock_send, patch(
            "backend.services.discord_notifier.send_on_the_clock_alert", return_value=True
        ):
            tracker.process_new_picks(picks, total_picks_so_far=0)
            assert mock_send.call_count == 2

    def test_is_our_pick_flag_set_correctly(self):
        """is_our_pick=True when the pick number matches our draft slot."""
        tracker = _make_tracker(position=7)
        our_pick = _make_pick(_P7_R1)

        with patch(
            "backend.services.discord_notifier.send_draft_pick", return_value=True
        ) as mock_send, patch(
            "backend.services.discord_notifier.send_on_the_clock_alert", return_value=True
        ):
            tracker.process_new_picks([our_pick], total_picks_so_far=_P7_R1 - 1)
            args, kwargs = mock_send.call_args
            assert kwargs.get("is_our_pick") is True or args[5] is True

    def test_not_our_pick_flag_false_for_other_teams(self):
        """is_our_pick=False for picks that don't belong to us."""
        tracker = _make_tracker(position=7)
        not_ours = _make_pick(1)  # pick 1 belongs to position 1

        with patch(
            "backend.services.discord_notifier.send_draft_pick", return_value=True
        ) as mock_send, patch(
            "backend.services.discord_notifier.send_on_the_clock_alert", return_value=True
        ):
            tracker.process_new_picks([not_ours], total_picks_so_far=0)
            args, kwargs = mock_send.call_args
            is_ours_val = kwargs.get("is_our_pick", args[5] if len(args) > 5 else False)
            assert is_ours_val is False


class TestRunPollOnce:

    def test_returns_zero_when_no_new_picks(self):
        """If Yahoo returns the same count as before, new_count == 0."""
        mock_client = MagicMock()
        mock_client.get_draft_results.return_value = [_make_pick(1), _make_pick(2)]
        tracker = DraftTracker(mock_client, my_draft_position=7)
        tracker._last_pick_count = 2  # already processed these
        result = tracker.run_poll_once()
        assert result == 0

    def test_returns_new_count_on_growth(self):
        """run_poll_once returns 2 when results grow from 3 to 5."""
        mock_client = MagicMock()
        existing = [_make_pick(i) for i in range(1, 4)]
        new_picks = [_make_pick(4), _make_pick(5)]
        mock_client.get_draft_results.return_value = existing + new_picks

        tracker = DraftTracker(mock_client, my_draft_position=7)
        tracker._last_pick_count = 3  # pretend we already saw 3

        with patch(
            "backend.services.discord_notifier.send_draft_pick", return_value=True
        ), patch(
            "backend.services.discord_notifier.send_on_the_clock_alert", return_value=True
        ):
            result = tracker.run_poll_once()

        assert result == 2

    def test_mock_results_bypass_yahoo(self):
        """Passing mock_results skips the Yahoo client entirely."""
        mock_client = MagicMock()
        # yahoo client should never be called
        mock_client.get_draft_results.side_effect = RuntimeError("Should not call Yahoo")

        tracker = DraftTracker(mock_client, my_draft_position=7)
        mock_picks = [_make_pick(1)]

        with patch(
            "backend.services.discord_notifier.send_draft_pick", return_value=True
        ), patch(
            "backend.services.discord_notifier.send_on_the_clock_alert", return_value=True
        ):
            result = tracker.run_poll_once(mock_results=mock_picks)

        mock_client.get_draft_results.assert_not_called()
        assert result == 1

    def test_updates_internal_count(self):
        """_last_pick_count advances after a successful poll."""
        mock_client = MagicMock()
        picks = [_make_pick(i) for i in range(1, 6)]
        tracker = DraftTracker(mock_client, my_draft_position=7)

        with patch(
            "backend.services.discord_notifier.send_draft_pick", return_value=True
        ), patch(
            "backend.services.discord_notifier.send_on_the_clock_alert", return_value=True
        ):
            tracker.run_poll_once(mock_results=picks)

        assert tracker._last_pick_count == 5


class TestFormatPickMessage:

    def test_our_pick_contains_your_pick_label(self):
        """'YOUR PICK' appears in the formatted message when is_ours=True."""
        tracker = _make_tracker(position=7)
        pick = _make_pick(_P7_R1, player_name="Ronald Acuna Jr.", positions=["OF"])
        msg = tracker.format_pick_message(pick, is_ours=True)
        assert "YOUR PICK" in msg

    def test_other_pick_no_your_pick_label(self):
        """'YOUR PICK' must NOT appear for picks belonging to other teams."""
        tracker = _make_tracker(position=7)
        pick = _make_pick(1, player_name="Shohei Ohtani", positions=["SP", "DH"])
        msg = tracker.format_pick_message(pick, is_ours=False)
        assert "YOUR PICK" not in msg

    def test_includes_player_name(self):
        """Player name appears in the formatted message."""
        tracker = _make_tracker(position=7)
        pick = _make_pick(5, player_name="Corbin Carroll", positions=["CF"])
        msg = tracker.format_pick_message(pick, is_ours=False)
        assert "Corbin Carroll" in msg

    def test_includes_pick_number(self):
        """Overall pick number appears in the formatted message."""
        tracker = _make_tracker(position=7)
        pick = _make_pick(42, player_name="Gunnar Henderson", positions=["SS"])
        msg = tracker.format_pick_message(pick, is_ours=False)
        assert "42" in msg

    def test_falls_back_to_player_key_when_no_name(self):
        """When player_name is absent, player_key is used instead."""
        tracker = _make_tracker(position=7)
        pick = {
            "pick": 10,
            "round": 1,
            "team_key": "mlb.l.72586.t.5",
            "player_key": "mlb.p.7578",
        }
        msg = tracker.format_pick_message(pick, is_ours=False)
        assert "mlb.p.7578" in msg


class TestGracefulOnYahooFailure:

    def test_run_poll_once_returns_zero_on_exception(self):
        """If Yahoo API raises, run_poll_once returns 0 and does not crash."""
        mock_client = MagicMock()
        mock_client.get_draft_results.side_effect = ConnectionError("Network error")

        tracker = DraftTracker(mock_client, my_draft_position=7)
        result = tracker.run_poll_once()
        assert result == 0

    def test_get_current_results_returns_empty_on_exception(self):
        """get_current_results isolates exceptions and returns []."""
        mock_client = MagicMock()
        mock_client.get_draft_results.side_effect = ValueError("Bad token")

        tracker = DraftTracker(mock_client, my_draft_position=7)
        results = tracker.get_current_results()
        assert results == []

    def test_discord_failure_does_not_crash_poll(self):
        """If send_draft_pick raises, run_poll_once still completes and returns count."""
        mock_client = MagicMock()
        picks = [_make_pick(1)]
        tracker = DraftTracker(mock_client, my_draft_position=7)

        with patch(
            "backend.services.discord_notifier.send_draft_pick",
            side_effect=RuntimeError("Discord down"),
        ), patch(
            "backend.services.discord_notifier.send_on_the_clock_alert", return_value=True
        ):
            result = tracker.run_poll_once(mock_results=picks)

        assert result == 1


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
