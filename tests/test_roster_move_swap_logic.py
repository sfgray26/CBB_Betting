"""
CRITICAL 2 REGRESSION TEST
Tests for automatic swap logic when moving into an occupied slot.

When a player is moved to a slot that's already occupied, the endpoint must
automatically swap the occupant to the source slot (or BN) rather than failing
with Yahoo's "That position has already been filled" error.
"""

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def fantasy_client():
    with patch("backend.schedulers.fantasy_scheduler.start_fantasy_scheduler"):
        with patch("backend.schedulers.fantasy_scheduler.stop_fantasy_scheduler"):
            from backend.fantasy_app import app
            from backend.auth import verify_api_key
            from fastapi.testclient import TestClient
            app.dependency_overrides[verify_api_key] = lambda: "test_user"
            try:
                with TestClient(app) as client:
                    yield client
            finally:
                app.dependency_overrides.pop(verify_api_key, None)


class TestRosterMoveSwapLogic:
    """Regression tests for automatic swap logic when target slot is occupied."""

    def test_move_into_occupied_util_slot_triggers_swap(self, fantasy_client):
        """
        CRITICAL 2 REGRESSION TEST

        Moving Player A (BN) → Util when Util is occupied by Player B should:
        1. Move Player A to Util
        2. Move Player B (Util occupant) to BN (source slot)
        3. Submit complete lineup as single set_lineup call
        4. Return success=True

        This prevents Yahoo's "That position has already been filled" error.
        """
        # Setup: Walker on BN, Montgomery on Util
        mock_roster = [
            {
                "player_key": "469.p.12024",  # Jordan Walker
                "name": "Jordan Walker",
                "team": "STL",
                "positions": ["OF"],
                "selected_position": "BN",
                "status": "playing",
            },
            {
                "player_key": "469.p.99999",  # Braden Montgomery (Util occupant)
                "name": "Braden Montgomery",
                "team": "BOS",
                "positions": ["OF", "Util"],
                "selected_position": "Util",
                "status": "playing",
            },
            {
                "player_key": "469.p.11111",
                "name": "Other Player",
                "team": "NYY",
                "positions": ["1B"],
                "selected_position": "1B",
                "status": "playing",
            },
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster
        mock_client.set_lineup.return_value = {
            "applied": ["469.p.12024", "469.p.99999"],  # Both moves applied
            "skipped": [],
            "warnings": [],
        }

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={
                    "player_key": "469.p.12024",
                    "target_position": "Util",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["player_key"] == "469.p.12024"
        assert data["from_position"] == "BN"
        assert data["to_position"] == "Util"
        assert "swapped" in data["message"].lower() or "moved" in data["message"].lower()

        # Verify set_lineup was called with complete lineup (swap)
        call_args = mock_client.set_lineup.call_args
        lineup = call_args[1]["lineup"]

        # Walker should be in Util
        walker_slot = next((p["position"] for p in lineup if p["player_key"] == "469.p.12024"), None)
        assert walker_slot == "Util"

        # Montgomery should have moved to BN (swap)
        montgomery_slot = next((p["position"] for p in lineup if p["player_key"] == "469.p.99999"), None)
        assert montgomery_slot == "BN"

        # Other player unchanged
        other_slot = next((p["position"] for p in lineup if p["player_key"] == "469.p.11111"), None)
        assert other_slot == "1B"

    def test_move_into_occupied_active_slot_swaps_to_source_slot(self, fantasy_client):
        """
        CRITICAL 2 REGRESSION TEST

        Moving Player A (1B) → SS when SS is occupied by Player B should:
        1. Move Player A to SS
        2. Move Player B (SS occupant) to 1B (source slot, not BN)
        """
        mock_roster = [
            {
                "player_key": "469.p.11111",  # Player A (currently 1B)
                "name": "Player A",
                "team": "NYY",
                "positions": ["1B", "SS"],
                "selected_position": "1B",
                "status": "playing",
            },
            {
                "player_key": "469.p.22222",  # Player B (occupying SS)
                "name": "Player B",
                "team": "BOS",
                # Must be eligible for the vacated 1B slot — an ineligible swap
                # partner is redirected to BN by swap-partner validation.
                "positions": ["SS", "1B"],
                "selected_position": "SS",
                "status": "playing",
            },
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster
        mock_client.set_lineup.return_value = {
            "applied": ["469.p.11111", "469.p.22222"],
            "skipped": [],
            "warnings": [],
        }

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={
                    "player_key": "469.p.11111",
                    "target_position": "SS",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify swap: Player A → SS, Player B → 1B
        call_args = mock_client.set_lineup.call_args
        lineup = call_args[1]["lineup"]

        player_a_slot = next((p["position"] for p in lineup if p["player_key"] == "469.p.11111"), None)
        assert player_a_slot == "SS"

        player_b_slot = next((p["position"] for p in lineup if p["player_key"] == "469.p.22222"), None)
        assert player_b_slot == "1B"  # Swapped to source slot, not BN

    def test_move_into_empty_slot_no_swap_needed(self, fantasy_client):
        """
        CRITICAL 2 REGRESSION TEST

        Moving into an empty slot should NOT trigger swap logic.
        Only the moving player's position should change.
        """
        mock_roster = [
            {
                "player_key": "469.p.11111",
                "name": "Player A",
                "team": "NYY",
                "positions": ["1B"],
                "selected_position": "BN",
                "status": "playing",
            },
            {
                "player_key": "469.p.22222",
                "name": "Player B",
                "team": "BOS",
                "positions": ["SS"],
                "selected_position": "SS",
                "status": "playing",
            },
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster
        mock_client.set_lineup.return_value = {
            "applied": ["469.p.11111"],
            "skipped": [],
            "warnings": [],
        }

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={
                    "player_key": "469.p.11111",
                    "target_position": "1B",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify only Player A moved, Player B unchanged
        call_args = mock_client.set_lineup.call_args
        lineup = call_args[1]["lineup"]

        player_a_slot = next((p["position"] for p in lineup if p["player_key"] == "469.p.11111"), None)
        assert player_a_slot == "1B"

        player_b_slot = next((p["position"] for p in lineup if p["player_key"] == "469.p.22222"), None)
        assert player_b_slot == "SS"  # Unchanged

    def test_move_from_il_to_occupied_slot_swaps_to_bn(self, fantasy_client):
        """
        CRITICAL 2 REGRESSION TEST

        Moving Player A (IL) → Util when Util is occupied should:
        1. Move Player A to Util
        2. Move Player B (Util occupant) to BN (not IL, since source is IL)

        IL is a special slot — when swapping from IL, the displaced player goes to BN.
        """
        mock_roster = [
            {
                "player_key": "469.p.11111",  # Player A (activating from IL)
                "name": "Player A",
                "team": "NYY",
                "positions": ["1B", "Util"],
                "selected_position": "IL",
                "status": "playing",
            },
            {
                "player_key": "469.p.22222",  # Player B (Util occupant)
                "name": "Player B",
                "team": "BOS",
                "positions": ["OF"],
                "selected_position": "Util",
                "status": "playing",
            },
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster
        mock_client.set_lineup.return_value = {
            "applied": ["469.p.11111", "469.p.22222"],
            "skipped": [],
            "warnings": [],
        }

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={
                    "player_key": "469.p.11111",
                    "target_position": "Util",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify swap: Player A → Util, Player B → BN (not IL)
        call_args = mock_client.set_lineup.call_args
        lineup = call_args[1]["lineup"]

        player_a_slot = next((p["position"] for p in lineup if p["player_key"] == "469.p.11111"), None)
        assert player_a_slot == "Util"

        player_b_slot = next((p["position"] for p in lineup if p["player_key"] == "469.p.22222"), None)
        assert player_b_slot == "BN"  # Swapped to BN, not IL

    def test_successful_move_clears_yahoo_client_cache(self, fantasy_client):
        """
        CRITICAL 2 REGRESSION TEST

        After a successful move, the Yahoo client cache must be cleared
        so subsequent roster fetches return fresh data from Yahoo, not
        stale pre-move lineup data from the 5-minute cache.

        Without cache clearing, the UI shows stale data even after refresh.
        """
        mock_roster = [
            {
                "player_key": "469.p.11111",
                "name": "Player A",
                "team": "NYY",
                "positions": ["1B"],
                "selected_position": "BN",
                "status": "playing",
            },
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster
        mock_client.set_lineup.return_value = {
            "applied": ["469.p.11111"],
            "skipped": [],
            "warnings": [],
        }
        mock_client.clear_cache = MagicMock()  # Track cache clearing

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={
                    "player_key": "469.p.11111",
                    "target_position": "1B",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify cache was cleared after successful move
        mock_client.clear_cache.assert_called_once()

    def test_failed_move_does_not_clear_cache(self, fantasy_client):
        """
        CRITICAL 2 REGRESSION TEST

        When a move fails (Yahoo rejects it), the cache should NOT be cleared.
        Only successful moves should invalidate the cache.
        """
        mock_roster = [
            {
                "player_key": "469.p.11111",
                "name": "Player A",
                "team": "NYY",
                "positions": ["1B"],
                "selected_position": "BN",
                "status": "playing",
            },
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster
        # Yahoo rejects the move (empty applied list)
        mock_client.set_lineup.return_value = {
            "applied": [],  # No players applied
            "skipped": ["469.p.11111"],
            "warnings": [],
        }
        mock_client.clear_cache = MagicMock()

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={
                    "player_key": "469.p.11111",
                    "target_position": "1B",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False

        # Verify cache was NOT cleared after failed move
        mock_client.clear_cache.assert_not_called()
