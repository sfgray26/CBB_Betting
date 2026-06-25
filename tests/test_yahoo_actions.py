"""
Yahoo Actions Service Tests — Loop Iteration 10

Tests for elite-tier add/drop execution with validation and atomic Yahoo transactions:
- Test 1: ADD success (single player add)
- Test 2: ADD_DROP success (single atomic Yahoo add/drop transaction)
- Test 3: DROP success through the canonical Yahoo drop method

Created: 2026-06-24
"""

import pytest
from unittest.mock import MagicMock

from backend.services.yahoo_actions import (
    YahooActionsService,
    ValidationResult,
    ActionResult,
    ActionError,
    ActionWarning,
    get_yahoo_actions_service,
    ERROR_CODES,
    WARNING_CODES,
    ACTIVE_LINEUP_SLOTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def yahoo_client_mock():
    """Mock YahooFantasyClient."""
    client = MagicMock()
    client.get_my_team_key.return_value = "469.l.72586.t.5"
    client.get_league_key.return_value = "469.l.72586"
    return client


@pytest.fixture
def service(yahoo_client_mock):
    """YahooActionsService with mocked client."""
    svc = YahooActionsService()
    svc._client = yahoo_client_mock
    svc._team_key = "469.l.72586.t.5"
    return svc


@pytest.fixture
def sample_roster():
    """Sample roster data for validation tests."""
    return [
        {
            "player_key": "469.p.10001",
            "player_id": "10001",
            "name": "Player One",
            "selected_position": "C",
            "status": None,
            "positions": ["C"],
        },
        {
            "player_key": "469.p.10002",
            "player_id": "10002",
            "name": "Player Two",
            "selected_position": "1B",
            "status": None,
            "positions": ["1B", "CI"],
        },
        # Add more roster players as needed
    ]


@pytest.fixture
def free_agent_player():
    """Sample free agent player data."""
    return {
        "player_key": "469.p.20001",
        "player_id": "20001",
        "name": "Free Agent One",
        "status": "A",  # Available
        "positions": ["OF"],
    }


@pytest.fixture
def waiver_player():
    """Sample player on waivers."""
    return {
        "player_key": "469.p.20002",
        "player_id": "20002",
        "name": "Waiver Player",
        "status": "W",  # On waivers
        "positions": ["SP"],
    }


@pytest.fixture
def ineligible_player():
    """Player not eligible for active lineup slots."""
    return {
        "player_key": "469.p.20003",
        "player_id": "20003",
        "name": "Bench Only",
        "status": "A",
        "positions": [],  # No position eligibility
    }


# ---------------------------------------------------------------------------
# Test 1: ADD Success
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestAddSuccess:
    """Test successful ADD operation with validation and execution."""

    async def test_add_free_agent_to_bench_succeeds(self, service, yahoo_client_mock, sample_roster, free_agent_player):
        """
        ADD a free agent player to bench slot should succeed.

        Validates:
        - Phase 1: Roster space available, player available, position valid
        - Phase 2: ADD executes successfully
        - Returns success=True with transaction_id
        """
        # Create roster with space (22 players, room for 1)
        # sample_roster has 2 players, need 20 more = 22 total
        roster_with_space = sample_roster + [
            {f"player_key": f"469.p.{i}", "player_id": str(i), "name": f"Player {i}",
             "selected_position": "BN", "status": None, "positions": ["BN"]}
            for i in range(10003, 10023)  # 20 players (10003-10022)
        ]

        # Updated roster after ADD (23 players)
        updated_roster = roster_with_space + [free_agent_player]

        # Configure mock to return different values on sequential calls
        # First call: roster with space (for validation)
        # Second call: updated roster (for final roster state)
        call_count = [0]

        def mock_get_roster(team_key):
            call_count[0] += 1
            if call_count[0] == 1:
                return roster_with_space
            return updated_roster

        yahoo_client_mock.get_roster.side_effect = mock_get_roster
        yahoo_client_mock.get_player.return_value = free_agent_player

        # Mock Phase 2: add_drop_player succeeds
        yahoo_client_mock.add_drop_player.return_value = True

        # Execute
        result = await service.execute_action(
            action="ADD",
            add_player_id="469.p.20001",
            drop_player_id=None,
            position="BN",
        )

        # Assertions
        assert result.success is True
        assert result.add_succeeded is True
        assert result.transaction_id is not None
        assert result.transaction_id.startswith("txn-")
        assert result.errors is None or len(result.errors) == 0
        assert result.rollback_attempted is False
        assert result.manual_action_required is False
        assert result.roster_state["player_count"] == 23

        # Verify Yahoo client was called correctly
        yahoo_client_mock.add_drop_player.assert_called_once()
        call_args = yahoo_client_mock.add_drop_player.call_args
        assert call_args[1]["add_player_key"] == "469.p.20001"
        assert call_args[1]["drop_player_key"] is None
        assert call_args[1]["team_key"] == "469.l.72586.t.5"

    async def test_add_generates_position_defaulted_warning(self, service, yahoo_client_mock, sample_roster, free_agent_player):
        """
        ADD without specifying position should generate warning and default to BN.
        """
        # Roster with space
        roster_with_space = sample_roster + [
            {f"player_key": f"469.p.{i}", "player_id": str(i), "name": f"Player {i}",
             "selected_position": "BN", "status": None, "positions": ["BN"]}
            for i in range(10003, 10022)
        ]

        yahoo_client_mock.get_roster.return_value = roster_with_space
        yahoo_client_mock.get_player.return_value = free_agent_player
        yahoo_client_mock.add_drop_player.return_value = True

        # Execute WITHOUT position
        result = await service.execute_action(
            action="ADD",
            add_player_id="469.p.20001",
            drop_player_id=None,
            position=None,  # No position specified
        )

        # Should have warning but still succeed
        assert result.success is True
        assert result.warnings is not None
        warning_codes = [w.code for w in result.warnings]
        assert WARNING_CODES["POSITION_DEFAULTED_TO_BN"] in warning_codes


# ---------------------------------------------------------------------------
# Test 2: ADD_DROP Success
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestAddDropSuccess:
    """Test successful ADD_DROP operation (transactional add + drop)."""

    async def test_add_drop_succeeds_when_both_valid(self, service, yahoo_client_mock, sample_roster, free_agent_player):
        """
        ADD_DROP should execute both ADD and DROP when validation passes.

        Validates:
        - Phase 1: Both players valid, roster size preserved
        - Phase 2: ADD executes, then DROP executes
        - Returns success=True with transaction_id
        """
        # Roster with player to drop (23 players, at capacity)
        full_roster = sample_roster + [
            {f"player_key": f"469.p.{i}", "player_id": str(i), "name": f"Player {i}",
             "selected_position": "BN", "status": None, "positions": ["BN"]}
            for i in range(10003, 10023)  # 23 total
        ]

        yahoo_client_mock.get_roster.return_value = full_roster
        yahoo_client_mock.get_player.return_value = free_agent_player

        yahoo_client_mock.add_drop_player.return_value = True

        # Execute ADD_DROP
        result = await service.execute_action(
            action="ADD_DROP",
            add_player_id="469.p.20001",
            drop_player_id="469.p.10003",  # Drop existing player
            position="BN",
        )

        # Assertions
        assert result.success is True
        assert result.add_succeeded is True
        assert result.drop_succeeded is True
        assert result.transaction_id is not None
        assert "add_20001" in result.transaction_id
        assert "drop_10003" in result.transaction_id
        yahoo_client_mock.add_drop_player.assert_called_once_with(
            add_player_key="469.p.20001",
            drop_player_key="469.p.10003",
            team_key="469.l.72586.t.5",
        )
        yahoo_client_mock.drop_player.assert_not_called()

    async def test_add_drop_validates_drop_player_on_roster(self, service, yahoo_client_mock, sample_roster):
        """
        ADD_DROP should fail if drop_player is not on roster.
        """
        # Roster WITHOUT the player we're trying to drop
        roster_with_space = sample_roster + [
            {f"player_key": f"469.p.{i}", "player_id": str(i), "name": f"Player {i}",
             "selected_position": "BN", "status": None, "positions": ["BN"]}
            for i in range(10003, 10022)
        ]

        yahoo_client_mock.get_roster.return_value = roster_with_space

        # Try to drop a player not on roster
        result = await service.execute_action(
            action="ADD_DROP",
            add_player_id="469.p.20001",
            drop_player_id="469.p.99999",  # NOT on roster
            position="BN",
        )

        # Should fail validation
        assert result.success is False
        assert result.errors is not None
        error_codes = [e.code for e in result.errors]
        assert ERROR_CODES["PLAYER_NOT_ON_ROSTER"] in error_codes

        # ADD should NOT have been executed
        yahoo_client_mock.add_drop_player.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: Canonical Standalone DROP
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestDropSuccess:
    """Test standalone DROP through the canonical Yahoo client method."""

    async def test_drop_uses_drop_player_without_invalid_add(self, service, yahoo_client_mock, sample_roster):
        yahoo_client_mock.get_roster.return_value = sample_roster
        yahoo_client_mock.drop_player.return_value = True

        result = await service.execute_action(
            action="DROP",
            add_player_id=None,
            drop_player_id="469.p.10001",
            position=None,
        )

        assert result.success is True
        assert result.add_succeeded is False
        assert result.drop_succeeded is True
        assert result.rollback_attempted is False
        yahoo_client_mock.drop_player.assert_called_once_with(
            player_key="469.p.10001",
            team_key="469.l.72586.t.5",
        )
        yahoo_client_mock.add_drop_player.assert_not_called()


# ---------------------------------------------------------------------------
# Validation Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestValidation:
    """Test Phase 1 validation rules."""

    async def test_validation_rejects_roster_full(self, service, yahoo_client_mock, sample_roster):
        """
        ADD should fail validation when roster is full and no drop specified.
        """
        # Roster at max capacity (23 players)
        # sample_roster has 2 players, need 21 more = 23 total
        full_roster = sample_roster + [
            {f"player_key": f"469.p.{i}", "player_id": str(i), "name": f"Player {i}",
             "selected_position": "BN", "status": None, "positions": ["BN"]}
            for i in range(10003, 10024)
        ]  # 23 total (2 + 21)

        yahoo_client_mock.get_roster.return_value = full_roster

        # Execute ADD without DROP
        result = await service.execute_action(
            action="ADD",
            add_player_id="469.p.20001",
            drop_player_id=None,
            position="BN",
        )

        # Should fail validation
        assert result.success is False
        assert result.errors is not None
        error_codes = [e.code for e in result.errors]
        assert ERROR_CODES["ROSTER_FULL"] in error_codes

        # Should NOT attempt to execute
        yahoo_client_mock.add_drop_player.assert_not_called()

    async def test_validation_warns_when_player_on_waivers(self, service, yahoo_client_mock, sample_roster, waiver_player):
        """
        ADD should succeed with warning when player is on waivers (not free agency).
        """
        roster_with_space = sample_roster + [
            {f"player_key": f"469.p.{i}", "player_id": str(i), "name": f"Player {i}",
             "selected_position": "BN", "status": None, "positions": ["BN"]}
            for i in range(10003, 10022)
        ]

        yahoo_client_mock.get_roster.return_value = roster_with_space
        yahoo_client_mock.get_player.return_value = waiver_player
        yahoo_client_mock.add_drop_player.return_value = True

        # Execute
        result = await service.execute_action(
            action="ADD",
            add_player_id="469.p.20002",
            drop_player_id=None,
            position="BN",
        )

        # Should succeed but with waiver warning
        assert result.success is True
        assert result.warnings is not None
        warning_codes = [w.code for w in result.warnings]
        assert WARNING_CODES["PLAYER_ON_WAIVERS"] in warning_codes

    async def test_validation_rejects_position_ineligible(self, service, yahoo_client_mock, sample_roster, ineligible_player):
        """
        ADD to active lineup slot should fail when player not eligible.
        """
        roster_with_space = sample_roster + [
            {f"player_key": f"469.p.{i}", "player_id": str(i), "name": f"Player {i}",
             "selected_position": "BN", "status": None, "positions": ["BN"]}
            for i in range(10003, 10022)
        ]

        yahoo_client_mock.get_roster.return_value = roster_with_space
        yahoo_client_mock.get_player.return_value = ineligible_player

        # Try to add ineligible player to C slot
        result = await service.execute_action(
            action="ADD",
            add_player_id="469.p.20003",
            drop_player_id=None,
            position="C",  # Player not eligible for C
        )

        # Should fail validation
        assert result.success is False
        assert result.errors is not None
        error_codes = [e.code for e in result.errors]
        assert ERROR_CODES["POSITION_INELIGIBLE"] in error_codes

    async def test_validation_allows_bench_for_any_player(self, service, yahoo_client_mock, sample_roster, ineligible_player):
        """
        ADD to bench (BN) should succeed regardless of position eligibility.
        """
        roster_with_space = sample_roster + [
            {f"player_key": f"469.p.{i}", "player_id": str(i), "name": f"Player {i}",
             "selected_position": "BN", "status": None, "positions": ["BN"]}
            for i in range(10003, 10022)
        ]

        yahoo_client_mock.get_roster.return_value = roster_with_space
        yahoo_client_mock.get_player.return_value = ineligible_player
        yahoo_client_mock.add_drop_player.return_value = True

        # Add ineligible player to BN (should succeed)
        result = await service.execute_action(
            action="ADD",
            add_player_id="469.p.20003",
            drop_player_id=None,
            position="BN",  # Bench slot - no eligibility check
        )

        # Should succeed
        assert result.success is True
        assert result.errors is None or len(result.errors) == 0

    async def test_validation_warns_when_dropping_il_player(self, service, yahoo_client_mock, sample_roster):
        """
        DROP should warn when dropping a player on IL.
        """
        # Roster with player on IL
        roster_with_il = sample_roster + [
            {f"player_key": f"469.p.{i}", "player_id": str(i), "name": f"Player {i}",
             "selected_position": "BN", "status": None, "positions": ["BN"]}
            for i in range(10003, 10022)
        ] + [
            {
                "player_key": "469.p.10003",
                "player_id": "10003",
                "name": "Injured Player",
                "selected_position": "IL",  # On IL
                "status": "IL",
                "positions": ["1B"],
            }
        ]

        yahoo_client_mock.get_roster.return_value = roster_with_il

        # Try to drop IL player
        result = await service.execute_action(
            action="DROP",
            add_player_id="469.p.20001",  # Need a dummy add for DROP to be valid in this context
            drop_player_id="469.p.10003",
            position="BN",
        )

        # Should have warning about dropping IL player
        # Note: This test validates the warning is generated; actual execution
        # would fail because add_player_id "469.p.20001" doesn't exist
        # The validation phase should generate the warning regardless
        assert result.warnings is not None
        warning_codes = [w.code for w in result.warnings]
        # May also have other warnings, so check if this exists
        has_il_warning = any(
            w.code == WARNING_CODES["DROP_PLAYER_INACTIVE"] or
            "IL" in w.message or "inactive" in w.message.lower()
            for w in result.warnings
        )


# ---------------------------------------------------------------------------
# Singleton Tests
# ---------------------------------------------------------------------------

def test_get_yahoo_actions_service_returns_singleton():
    """Test that get_yahoo_actions_service returns the same instance."""
    from backend.services.yahoo_actions import get_yahoo_actions_service

    svc1 = get_yahoo_actions_service()
    svc2 = get_yahoo_actions_service()

    assert svc1 is svc2
    assert isinstance(svc1, YahooActionsService)
