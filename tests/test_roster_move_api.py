"""
Tests for Phase 4 Roster Move API endpoint.

Tests for POST /api/fantasy/roster/move.
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


class TestRosterMoveEndpoint:
    """Tests for POST /api/fantasy/roster/move endpoint."""

    def test_move_response_structure(self, fantasy_client):
        """Move response has all required fields."""
        # Mock the Yahoo client to return a test roster
        mock_roster = [
            {
                "player_key": "469.l.72586.p.12345",
                "name": "Test Player",
                "team": "NYY",
                "positions": ["1B"],
                "selected_position": "BN",
            },
            {
                "player_key": "469.l.72586.p.67890",
                "name": "Another Player",
                "team": "BOS",
                "positions": ["LF", "CF", "RF"],  # Specific OF positions
                "selected_position": "LF",  # Specific OF slot
            },
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster
        mock_client.set_lineup.return_value = {
            "applied": ["469.l.72586.p.12345"],
            "skipped": [],
            "warnings": [],
        }

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={
                    "player_key": "469.l.72586.p.12345",
                    "target_position": "1B",
                },
            )

        assert response.status_code == 200
        data = response.json()

        # Response fields
        assert "success" in data
        assert "player_key" in data
        assert "from_position" in data
        assert "to_position" in data
        assert "message" in data
        assert "warnings" in data
        assert "freshness" in data

        # Freshness fields
        freshness = data["freshness"]
        assert "primary_source" in freshness
        assert "computed_at" in freshness
        assert "staleness_threshold_minutes" in freshness
        assert "is_stale" in freshness

    def test_successful_move(self, fantasy_client):
        """Player moved successfully."""
        mock_roster = [
            {
                "player_key": "469.l.72586.p.12345",
                "name": "Test Player",
                "team": "NYY",
                "positions": ["1B"],
                "selected_position": "BN",
            },
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster
        mock_client.set_lineup.return_value = {
            "applied": ["469.l.72586.p.12345"],
            "skipped": [],
            "warnings": [],
        }

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={
                    "player_key": "469.l.72586.p.12345",
                    "target_position": "1B",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["player_key"] == "469.l.72586.p.12345"
        assert data["from_position"] == "BN"
        assert data["to_position"] == "1B"
        assert "Moved Test Player" in data["message"]

    def test_invalid_position(self, fantasy_client):
        """Invalid position returns error."""
        response = fantasy_client.post(
            "/api/fantasy/roster/move",
            json={
                "player_key": "469.l.72586.p.12345",
                "target_position": "INVALID",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "Invalid position" in data["message"]

    def test_player_not_found(self, fantasy_client):
        """Player not on roster returns error."""
        mock_roster = [
            {
                "player_key": "469.l.72586.p.99999",
                "name": "Different Player",
                "team": "BOS",
                "positions": ["LF", "CF", "RF"],  # Specific OF positions
                "selected_position": "LF",  # Specific OF slot
            },
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={
                    "player_key": "469.l.72586.p.12345",
                    "target_position": "1B",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "not found on roster" in data["message"]

    def test_yahoo_api_error(self, fantasy_client):
        """Yahoo API error handled gracefully."""
        from backend.fantasy_baseball.yahoo_client_resilient import YahooAPIError

        mock_client = MagicMock()
        mock_client.get_roster.side_effect = YahooAPIError("API rate limit")

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={
                    "player_key": "469.l.72586.p.12345",
                    "target_position": "1B",
                },
            )

        assert response.status_code in (200, 502)  # Either graceful or HTTP exception

    def test_move_to_il(self, fantasy_client):
        """Player moved to IL slot."""
        mock_roster = [
            {
                "player_key": "469.l.72586.p.12345",
                "name": "Injured Player",
                "team": "NYY",
                "positions": ["1B"],
                "selected_position": "1B",
                "status": "DTD",
            },
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster
        mock_client.set_lineup.return_value = {
            "applied": ["469.l.72586.p.12345"],
            "skipped": [],
            "warnings": [],
        }

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={
                    "player_key": "469.l.72586.p.12345",
                    "target_position": "IL",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["from_position"] == "1B"
        assert data["to_position"] == "IL"

    def test_il_player_to_active_slot_returns_400(self, fantasy_client):
        """IL-designated player cannot be moved to an active slot — returns 400."""
        mock_roster = [
            {
                "player_key": "469.l.72586.p.12345",
                "name": "Hurt Shortstop",
                "team": "NYY",
                "positions": ["SS"],
                "selected_position": "IL",
                "status": "IL",
            },
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={
                    "player_key": "469.l.72586.p.12345",
                    "target_position": "SS",
                },
            )

        assert response.status_code == 400
        assert "IL designation" in response.json()["detail"]

    def test_il_player_to_il60_slot_is_allowed(self, fantasy_client):
        """IL-designated player can be moved between IL-type slots (IL → IL60)."""
        mock_roster = [
            {
                "player_key": "469.l.72586.p.12345",
                "name": "Hurt Shortstop",
                "team": "NYY",
                "positions": ["SS"],
                "selected_position": "IL",
                "status": "IL60",
            },
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster
        mock_client.set_lineup.return_value = {
            "applied": ["469.l.72586.p.12345"],
            "skipped": [],
            "warnings": [],
        }

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={
                    "player_key": "469.l.72586.p.12345",
                    "target_position": "IL60",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_dtd_player_can_move_to_active_slot(self, fantasy_client):
        """Day-to-Day (DTD) players are NOT blocked — only IL designations are."""
        mock_roster = [
            {
                "player_key": "469.l.72586.p.12345",
                "name": "Dinged Catcher",
                "team": "HOU",
                "positions": ["C"],
                "selected_position": "BN",
                "status": "DTD",
            },
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster
        mock_client.set_lineup.return_value = {
            "applied": ["469.l.72586.p.12345"],
            "skipped": [],
            "warnings": [],
        }

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={
                    "player_key": "469.l.72586.p.12345",
                    "target_position": "C",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_out_player_without_il_designation_can_remain_active(self, fantasy_client):
        """OUT/injury notes are not Yahoo IL designations for mutation validation."""
        mock_roster = [
            {
                "player_key": "469.l.72586.p.12345",
                "name": "Unavailable Pitcher",
                "team": "HOU",
                "positions": ["SP"],
                "selected_position": "SP",
                "status": "OUT",
                "injury_note": "Shoulder",
            },
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster
        mock_client.set_lineup.return_value = {
            "applied": ["469.l.72586.p.12345"],
            "skipped": [],
            "warnings": [],
        }

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={
                    "player_key": "469.l.72586.p.12345",
                    "target_position": "SP",
                },
            )

        assert response.status_code == 200

    def test_ineligible_position_returns_400(self, fantasy_client):
        """Moving a player to a slot they're not eligible for returns 400."""
        mock_roster = [
            {
                "player_key": "469.l.72586.p.12345",
                "name": "Kyle Harrison",
                "team": "SF",
                "positions": ["SP"],
                "selected_position": "BN",
            },
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = mock_roster

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={
                    "player_key": "469.l.72586.p.12345",
                    "target_position": "RP",
                },
            )

        assert response.status_code == 400
        assert "not eligible" in response.json()["detail"]

    def test_valid_positions_accepted(self, fantasy_client):
        """All valid positions are accepted."""
        valid_positions = [
            "C", "1B", "2B", "3B", "SS", "OF", "Util",
            "SP", "RP", "P", "BN", "IL", "IL60",
        ]

        for pos in valid_positions:
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={
                    "player_key": "469.l.72586.p.12345",
                    "target_position": pos,
                },
            )

            # Should not return "Invalid position" error
            # (may fail for other reasons like player not found)
            if response.status_code == 200:
                data = response.json()
                if not data.get("success") and "Invalid position" in data.get("message", ""):
                    assert False, f"Position {pos} was rejected as invalid"


class TestMoveThreeLayerValidation:
    """Regression tests for the IL guard failure (Crochet→SP) and swap-bump bug (Soroka)."""

    @staticmethod
    def _client_for(roster):
        mock_client = MagicMock()
        mock_client.get_roster.return_value = roster
        mock_client.set_lineup.return_value = {
            "applied": [p["player_key"] for p in roster],
            "skipped": [],
            "warnings": [],
        }
        return mock_client

    @staticmethod
    def _lineup_positions(mock_client):
        """Extract {player_key: position} from the set_lineup call payload."""
        lineup = mock_client.set_lineup.call_args.kwargs.get("lineup")
        if lineup is None:
            lineup = mock_client.set_lineup.call_args.args[-1]
        return {e["player_key"]: e["position"] for e in lineup}

    # ── Layer 1: source player ───────────────────────────────────────────────

    def test_move_il_player_to_active_blocked(self, fantasy_client):
        """IL-designated (via status) player cannot be moved to an active slot."""
        roster = [
            {
                "player_key": "469.l.72586.p.44444",
                "name": "Garrett Crochet",
                "team": "BOS",
                "positions": ["SP"],
                "selected_position": "IL60",
                "status": "60-Day-IL",
            },
        ]
        mock_client = self._client_for(roster)

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={"player_key": "469.l.72586.p.44444", "target_position": "SP"},
            )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "Cannot move" in detail
        assert "IL" in detail
        mock_client.set_lineup.assert_not_called()

    def test_move_il_player_with_null_status_to_active_blocked(self, fantasy_client):
        """Regression (Bug 1): Yahoo returns status=None for IL players — the IL slot
        itself must be treated as the IL designation."""
        roster = [
            {
                "player_key": "469.l.72586.p.44444",
                "name": "Garrett Crochet",
                "team": "BOS",
                "positions": ["SP"],
                "selected_position": "IL60",
                "status": None,
            },
        ]
        mock_client = self._client_for(roster)

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={"player_key": "469.l.72586.p.44444", "target_position": "SP"},
            )

        assert response.status_code == 400
        assert "Cannot move" in response.json()["detail"]
        mock_client.set_lineup.assert_not_called()

    def test_move_il_player_to_bn_allowed(self, fantasy_client):
        """IL-designated players may be parked on BN (that's where Yahoo puts them pre-IL)."""
        roster = [
            {
                "player_key": "469.l.72586.p.44444",
                "name": "Garrett Crochet",
                "team": "BOS",
                "positions": ["SP"],
                "selected_position": "IL60",
                "status": "IL60",
            },
        ]
        mock_client = self._client_for(roster)

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={"player_key": "469.l.72586.p.44444", "target_position": "BN"},
            )

        assert response.status_code == 200
        assert response.json()["success"] is True

    # ── Layer 2: target slot ─────────────────────────────────────────────────

    def test_move_active_player_to_full_il_blocked(self, fantasy_client):
        """Moving a player to IL when all IL slots are occupied returns 400."""
        roster = [
            {
                "player_key": "469.l.72586.p.11111",
                "name": "Healthy Hitter",
                "team": "NYY",
                "positions": ["1B"],
                "selected_position": "BN",
            },
        ] + [
            {
                "player_key": f"469.l.72586.p.7777{i}",
                "name": f"IL Occupant {i}",
                "team": "NYM",
                "positions": ["RP"],
                "selected_position": slot,
                "status": "IL60",
            }
            for i, slot in enumerate(["IL", "IL", "IL60"])
        ]
        mock_client = self._client_for(roster)

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={"player_key": "469.l.72586.p.11111", "target_position": "IL"},
            )

        assert response.status_code == 400
        assert "IL slot full" in response.json()["detail"]
        mock_client.set_lineup.assert_not_called()

    def test_move_healthy_player_to_il_with_space_allowed(self, fantasy_client):
        """Moving a player to IL when a slot is free succeeds."""
        roster = [
            {
                "player_key": "469.l.72586.p.11111",
                "name": "Healthy Hitter",
                "team": "NYY",
                "positions": ["1B"],
                "selected_position": "BN",
            },
            {
                "player_key": "469.l.72586.p.88888",
                "name": "Michael Soroka",
                "team": "WSH",
                "positions": ["SP", "RP"],
                "selected_position": "IL",
                "status": "IL15",
            },
        ]
        mock_client = self._client_for(roster)

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={"player_key": "469.l.72586.p.11111", "target_position": "IL"},
            )

        assert response.status_code == 200
        assert response.json()["success"] is True

    # ── Layer 3: swap partner ────────────────────────────────────────────────

    def test_move_to_il_never_bumps_il_occupant(self, fantasy_client):
        """Regression (Bug 2): reverting an IL player back to IL must NOT displace
        another IL player (Soroka) out of the IL slot — IL is multi-occupancy."""
        roster = [
            {
                "player_key": "469.l.72586.p.44444",
                "name": "Garrett Crochet",
                "team": "BOS",
                "positions": ["SP"],
                "selected_position": "SP",  # corrupted state from Bug 1
                "status": "IL60",
            },
            {
                "player_key": "469.l.72586.p.88888",
                "name": "Michael Soroka",
                "team": "WSH",
                "positions": ["SP", "RP"],
                "selected_position": "IL",
                "status": "IL15",
            },
        ]
        mock_client = self._client_for(roster)

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={"player_key": "469.l.72586.p.44444", "target_position": "IL"},
            )

        assert response.status_code == 200
        positions = self._lineup_positions(mock_client)
        assert positions["469.l.72586.p.44444"] == "IL"
        assert positions["469.l.72586.p.88888"] == "IL"  # Soroka untouched

    def test_move_to_bn_never_bumps_bench_occupant(self, fantasy_client):
        """BN is multi-occupancy — moving a player to BN must not promote a bench player."""
        roster = [
            {
                "player_key": "469.l.72586.p.11111",
                "name": "Active Hitter",
                "team": "NYY",
                "positions": ["1B"],
                "selected_position": "1B",
            },
            {
                "player_key": "469.l.72586.p.22222",
                "name": "Bench Hitter",
                "team": "BOS",
                "positions": ["1B"],
                "selected_position": "BN",
            },
        ]
        mock_client = self._client_for(roster)

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={"player_key": "469.l.72586.p.11111", "target_position": "BN"},
            )

        assert response.status_code == 200
        positions = self._lineup_positions(mock_client)
        assert positions["469.l.72586.p.11111"] == "BN"
        assert positions["469.l.72586.p.22222"] == "BN"  # not promoted to 1B

    def test_swap_into_active_slot_still_bumps_occupant(self, fantasy_client):
        """Single-occupancy active slots keep the swap behavior (occupant → BN)."""
        roster = [
            {
                "player_key": "469.l.72586.p.11111",
                "name": "Bench Hitter",
                "team": "NYY",
                "positions": ["1B"],
                "selected_position": "BN",
            },
            {
                "player_key": "469.l.72586.p.22222",
                "name": "Current First Baseman",
                "team": "BOS",
                "positions": ["1B"],
                "selected_position": "1B",
            },
        ]
        mock_client = self._client_for(roster)

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={"player_key": "469.l.72586.p.11111", "target_position": "1B"},
            )

        assert response.status_code == 200
        positions = self._lineup_positions(mock_client)
        assert positions["469.l.72586.p.11111"] == "1B"
        assert positions["469.l.72586.p.22222"] == "BN"

    def test_swap_partner_with_il_designation_goes_to_bn_not_active(self, fantasy_client):
        """An IL-designated occupant of an active slot (corrupted state) must be
        displaced to BN, never into another active slot."""
        roster = [
            {
                "player_key": "469.l.72586.p.11111",
                "name": "Healthy Pitcher",
                "team": "NYY",
                "positions": ["SP", "RP"],
                "selected_position": "SP",
            },
            {
                "player_key": "469.l.72586.p.44444",
                "name": "Garrett Crochet",
                "team": "BOS",
                "positions": ["SP", "RP"],
                "selected_position": "RP",  # corrupted state from Bug 1
                "status": "IL60",
            },
        ]
        mock_client = self._client_for(roster)

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={"player_key": "469.l.72586.p.11111", "target_position": "RP"},
            )

        assert response.status_code == 200
        positions = self._lineup_positions(mock_client)
        assert positions["469.l.72586.p.11111"] == "RP"
        assert positions["469.l.72586.p.44444"] == "BN"  # NOT swapped into SP

    def test_valid_move_allowed(self, fantasy_client):
        """Healthy bench player into an open Util slot succeeds."""
        roster = [
            {
                "player_key": "469.l.72586.p.11111",
                "name": "Healthy Hitter",
                "team": "NYY",
                "positions": ["1B"],
                "selected_position": "BN",
            },
        ]
        mock_client = self._client_for(roster)

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/move",
                json={"player_key": "469.l.72586.p.11111", "target_position": "Util"},
            )

        assert response.status_code == 200
        assert response.json()["success"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Bulk Apply Endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestBulkRosterMoveEndpoint:
    """Tests for POST /api/fantasy/roster/bulk-apply endpoint."""

    _MOCK_ROSTER = [
        {
            "player_key": "469.l.72586.p.11111",
            "name": "Player One",
            "team": "NYY",
            "positions": ["1B", "Util"],
            "selected_position": "BN",
        },
        {
            "player_key": "469.l.72586.p.22222",
            "name": "Player Two",
            "team": "BOS",
            "positions": ["SP"],
            "selected_position": "BN",
        },
        {
            "player_key": "469.l.72586.p.33333",
            "name": "Player Three",
            "team": "LAD",
            "positions": ["LF", "CF", "RF"],  # Specific OF positions, not generic "OF"
            "selected_position": "LF",  # Specific OF slot
        },
    ]

    def test_bulk_apply_success(self, fantasy_client):
        """All moves applied — returns applied_count and zero errors."""
        mock_client = MagicMock()
        mock_client.get_roster.return_value = self._MOCK_ROSTER
        mock_client.set_lineup.return_value = {
            "applied": ["469.l.72586.p.11111", "469.l.72586.p.22222"],
            "skipped": [],
            "warnings": [],
        }

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/bulk-apply",
                json={
                    "moves": [
                        {"player_key": "469.l.72586.p.11111", "target_position": "1B"},
                        {"player_key": "469.l.72586.p.22222", "target_position": "SP"},
                    ]
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["applied_count"] == 2
        assert data["failed_count"] == 0
        assert data["errors"] == []

    def test_bulk_apply_response_fields(self, fantasy_client):
        """Response has applied_count, failed_count, errors fields."""
        mock_client = MagicMock()
        mock_client.get_roster.return_value = self._MOCK_ROSTER
        mock_client.set_lineup.return_value = {
            "applied": ["469.l.72586.p.11111"],
            "skipped": [],
            "warnings": [],
        }

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/bulk-apply",
                json={"moves": [{"player_key": "469.l.72586.p.11111", "target_position": "1B"}]},
            )

        assert response.status_code == 200
        data = response.json()
        assert "applied_count" in data
        assert "failed_count" in data
        assert "errors" in data
        assert isinstance(data["errors"], list)

    def test_bulk_apply_allows_legal_duplicate_pitcher_slots(self, fantasy_client):
        """Existing Yahoo rosters can legally contain multiple SP/RP/P slots."""
        pitcher_roster = self._MOCK_ROSTER + [
            {
                "player_key": "469.l.72586.p.44444",
                "name": "Starter Two",
                "team": "ATL",
                "positions": ["SP"],
                "selected_position": "SP",
            },
            {
                "player_key": "469.l.72586.p.55555",
                "name": "Pitcher One",
                "team": "SEA",
                "positions": ["SP", "RP"],
                "selected_position": "P",
            },
            {
                "player_key": "469.l.72586.p.66666",
                "name": "Pitcher Two",
                "team": "HOU",
                "positions": ["SP", "RP"],
                "selected_position": "P",
            },
        ]
        mock_client = MagicMock()
        mock_client.get_roster.return_value = pitcher_roster
        mock_client.set_lineup.return_value = {
            "applied": ["469.l.72586.p.11111"],
            "skipped": [],
            "warnings": [],
        }

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/bulk-apply",
                json={"moves": [{"player_key": "469.l.72586.p.11111", "target_position": "1B"}]},
            )

        assert response.status_code == 200

    def test_bulk_apply_invalid_position_returns_400(self, fantasy_client):
        """Invalid position in any move triggers 400 before Yahoo call."""
        response = fantasy_client.post(
            "/api/fantasy/roster/bulk-apply",
            json={
                "moves": [
                    {"player_key": "469.l.72586.p.11111", "target_position": "INVALID"},
                    {"player_key": "469.l.72586.p.22222", "target_position": "SP"},
                ]
            },
        )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "errors" in detail
        assert any("Invalid position" in e for e in detail["errors"])

    def test_bulk_apply_player_not_on_roster_returns_400(self, fantasy_client):
        """Player not on roster triggers 400 (no partial execution)."""
        mock_client = MagicMock()
        mock_client.get_roster.return_value = self._MOCK_ROSTER

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/bulk-apply",
                json={
                    "moves": [
                        {"player_key": "469.l.72586.p.99999", "target_position": "BN"},
                    ]
                },
            )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "errors" in detail
        assert any("not found" in e for e in detail["errors"])

    def test_bulk_apply_empty_moves_returns_400(self, fantasy_client):
        """Empty moves list is rejected before touching Yahoo."""
        response = fantasy_client.post(
            "/api/fantasy/roster/bulk-apply",
            json={"moves": []},
        )

        assert response.status_code == 400

    def test_bulk_apply_single_set_lineup_call(self, fantasy_client):
        """All moves executed in ONE set_lineup call (atomic)."""
        mock_client = MagicMock()
        mock_client.get_roster.return_value = self._MOCK_ROSTER
        mock_client.set_lineup.return_value = {
            "applied": ["469.l.72586.p.11111", "469.l.72586.p.22222"],
            "skipped": [],
            "warnings": [],
        }

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            fantasy_client.post(
                "/api/fantasy/roster/bulk-apply",
                json={
                    "moves": [
                        {"player_key": "469.l.72586.p.11111", "target_position": "1B"},
                        {"player_key": "469.l.72586.p.22222", "target_position": "SP"},
                    ]
                },
            )

        mock_client.set_lineup.assert_called_once()

    def test_bulk_apply_partial_failure_reports_errors(self, fantasy_client):
        """Yahoo confirms only some moves — failed_count and errors reflect the gap."""
        mock_client = MagicMock()
        mock_client.get_roster.return_value = self._MOCK_ROSTER
        # Yahoo only confirms p.11111 but not p.22222
        mock_client.set_lineup.return_value = {
            "applied": ["469.l.72586.p.11111"],
            "skipped": ["469.l.72586.p.22222"],
            "warnings": [],
        }

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/bulk-apply",
                json={
                    "moves": [
                        {"player_key": "469.l.72586.p.11111", "target_position": "1B"},
                        {"player_key": "469.l.72586.p.22222", "target_position": "SP"},
                    ]
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["applied_count"] == 1
        assert data["failed_count"] == 1
        assert len(data["errors"]) == 1
        assert "469.l.72586.p.22222" in data["errors"][0]

    def test_bulk_apply_il_player_to_active_slot_returns_400(self, fantasy_client):
        """IL-designated player in a bulk move to an active slot triggers 400."""
        il_roster = self._MOCK_ROSTER + [
            {
                "player_key": "469.l.72586.p.44444",
                "name": "IL Pitcher",
                "team": "ATL",
                "positions": ["SP"],
                "selected_position": "IL",
                "status": "IL",
            }
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = il_roster

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/bulk-apply",
                json={
                    "moves": [
                        {"player_key": "469.l.72586.p.11111", "target_position": "1B"},
                        {"player_key": "469.l.72586.p.44444", "target_position": "SP"},
                    ]
                },
            )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "errors" in detail
        assert any("IL designation" in e for e in detail["errors"])

    def test_bulk_apply_il_null_status_in_il_slot_to_active_blocked(self, fantasy_client):
        """Regression: Yahoo returns status=None for IL players — slot must be the fallback signal."""
        il_roster = self._MOCK_ROSTER + [
            {
                "player_key": "469.l.72586.p.44444",
                "name": "Garrett Crochet",
                "team": "BOS",
                "positions": ["SP"],
                "selected_position": "IL60",
                "status": None,
            }
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = il_roster

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/bulk-apply",
                json={
                    "moves": [
                        {"player_key": "469.l.72586.p.44444", "target_position": "SP"},
                    ]
                },
            )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert any("Cannot move" in e and "IL" in e for e in detail["errors"])
        mock_client.set_lineup.assert_not_called()

    def test_bulk_apply_healthy_to_full_il_blocked(self, fantasy_client):
        """Bulk move of a player into IL when all IL slots are occupied triggers 400."""
        full_il_roster = self._MOCK_ROSTER + [
            {
                "player_key": f"469.l.72586.p.7777{i}",
                "name": f"IL Occupant {i}",
                "team": "NYM",
                "positions": ["RP"],
                "selected_position": slot,
                "status": "IL60",
            }
            for i, slot in enumerate(["IL", "IL", "IL60"])
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = full_il_roster

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/bulk-apply",
                json={
                    "moves": [
                        {"player_key": "469.l.72586.p.11111", "target_position": "IL"},
                    ]
                },
            )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert any("IL slot full" in e for e in detail["errors"])
        mock_client.set_lineup.assert_not_called()

    def test_bulk_apply_il_rotation_within_batch_allowed(self, fantasy_client):
        """A batch that vacates an IL slot and fills it in the same call is legal."""
        roster = self._MOCK_ROSTER + [
            {
                "player_key": "469.l.72586.p.44444",
                "name": "Garrett Crochet",
                "team": "BOS",
                "positions": ["SP"],
                "selected_position": "IL60",
                "status": "IL60",
            },
            {
                "player_key": "469.l.72586.p.55555",
                "name": "IL Occupant A",
                "team": "NYM",
                "positions": ["RP"],
                "selected_position": "IL",
                "status": "IL15",
            },
            {
                "player_key": "469.l.72586.p.66666",
                "name": "IL Occupant B",
                "team": "NYM",
                "positions": ["RP"],
                "selected_position": "IL",
                "status": "IL15",
            },
            {
                "player_key": "469.l.72586.p.88888",
                "name": "Michael Soroka",
                "team": "WSH",
                "positions": ["SP", "RP"],
                "selected_position": "BN",
                "status": "IL15",
            },
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = roster
        mock_client.set_lineup.return_value = {
            "applied": ["469.l.72586.p.44444", "469.l.72586.p.88888"],
            "skipped": [],
            "warnings": [],
        }

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/bulk-apply",
                json={
                    "moves": [
                        # IL full (3/3): Crochet vacates IL60→BN, Soroka fills IL
                        {"player_key": "469.l.72586.p.44444", "target_position": "BN"},
                        {"player_key": "469.l.72586.p.88888", "target_position": "IL"},
                    ]
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["applied_count"] == 2

    def test_bulk_apply_allows_out_and_injury_notes_in_active_slots(self, fantasy_client):
        """Bulk validation does not confuse OUT/body-part notes with IL designations."""
        roster = self._MOCK_ROSTER + [
            {
                "player_key": "469.l.72586.p.44444",
                "name": "Unavailable Pitcher",
                "team": "ATL",
                "positions": ["SP"],
                "selected_position": "SP",
                "status": "OUT",
                "injury_note": "Shoulder",
            }
        ]

        mock_client = MagicMock()
        mock_client.get_roster.return_value = roster
        mock_client.set_lineup.return_value = {
            "applied": ["469.l.72586.p.11111"],
            "skipped": [],
            "warnings": [],
        }

        with patch("backend.routers.fantasy.get_yahoo_client", return_value=mock_client):
            response = fantasy_client.post(
                "/api/fantasy/roster/bulk-apply",
                json={"moves": [{"player_key": "469.l.72586.p.11111", "target_position": "1B"}]},
            )

        assert response.status_code == 200
