"""
Tests for Player Search Auto-Heal (BDL #2)

Validates that unmapped Yahoo players are automatically healed via BDL search,
tracking heal attempts and healed_at timestamps.

Test Coverage:
1. Auto-heal success - BDL search finds match, mapping created
2. Auto-heal no match - BDL search returns empty, graceful fallback
3. Downstream enrichment - after auto-heal, ownership % is populated
4. Batch heal - multiple unmapped players processed
5. Source tracking - source="bdl_search" and healed_at set
6. No regression - existing mapped players untouched
"""

import pytest
from datetime import datetime
from dataclasses import dataclass
from unittest.mock import Mock, MagicMock, patch

from backend.services.player_autoheal import (
    PlayerAutoHealService, _normalize, _name_confidence, _is_fresh,
)


@dataclass
class MockBDLPlayer:
    """Mock BDL player search result."""
    id: int
    full_name: str


class TestAutoHealSuccess:
    """Test successful auto-heal scenarios."""

    def test_heal_player_creates_new_mapping(self):
        """When BDL search finds match, new mapping is created with correct data."""
        mock_db = Mock()
        mock_bdl = Mock()

        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_bdl.search_mlb_players.return_value = [
            MockBDLPlayer(id=12345, full_name="John Doe")
        ]

        service = PlayerAutoHealService(db_session=mock_db, bdl_client=mock_bdl)
        result = service.heal_player(
            yahoo_id="7590",
            yahoo_key="469.p.7590",
            name="John Doe",
        )

        assert result is True
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called()

        new_row = mock_db.add.call_args[0][0]
        assert new_row.yahoo_id == "7590"
        assert new_row.yahoo_key == "469.p.7590"
        assert new_row.bdl_id == 12345
        assert new_row.full_name == "John Doe"
        assert new_row.source == "bdl_search"
        assert new_row.heal_attempts == 1
        assert new_row.healed_at is not None

    def test_heal_player_updates_existing_stale_mapping(self):
        """When mapping exists but is stale (old failed heal), it's updated."""
        mock_db = Mock()
        mock_bdl = Mock()

        existing = Mock()
        existing.source = "bdl_search"
        existing.bdl_id = None  # Previous heal failed
        existing.updated_at = datetime(2020, 1, 1)
        existing.heal_attempts = 0
        existing.healed_at = None

        mock_db.query.return_value.filter.return_value.first.return_value = existing
        mock_bdl.search_mlb_players.return_value = [
            MockBDLPlayer(id=12345, full_name="John Doe")
        ]

        service = PlayerAutoHealService(db_session=mock_db, bdl_client=mock_bdl)
        result = service.heal_player(
            yahoo_id="7590",
            yahoo_key="469.p.7590",
            name="John Doe",
        )

        assert result is True
        assert existing.bdl_id == 12345
        assert existing.source == "bdl_search"
        assert existing.healed_at is not None
        assert existing.heal_attempts == 1
        mock_db.commit.assert_called()


class TestAutoHealNoMatch:
    """Test graceful fallback when BDL search finds no match."""

    def test_heal_player_no_bdl_match(self):
        """When BDL search returns empty, returns False gracefully."""
        mock_db = Mock()
        mock_bdl = Mock()

        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_bdl.search_mlb_players.return_value = []

        service = PlayerAutoHealService(db_session=mock_db, bdl_client=mock_bdl)
        result = service.heal_player(
            yahoo_id="9999",
            yahoo_key="469.p.9999",
            name="Unknown Player",
        )

        assert result is False
        mock_db.add.assert_not_called()

    def test_heal_player_low_confidence_match(self):
        """When best match is below confidence threshold, returns False."""
        mock_db = Mock()
        mock_bdl = Mock()

        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_bdl.search_mlb_players.return_value = [
            MockBDLPlayer(id=12345, full_name="Completely Different Name")
        ]

        service = PlayerAutoHealService(db_session=mock_db, bdl_client=mock_bdl)
        result = service.heal_player(
            yahoo_id="7590",
            yahoo_key="469.p.7590",
            name="John Doe",
        )

        assert result is False


class TestSourceTracking:
    """Test that source and timestamp tracking works correctly."""

    def test_source_set_to_bdl_search_on_heal(self):
        """Auto-healed mappings have source='bdl_search'."""
        mock_db = Mock()
        mock_bdl = Mock()

        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_bdl.search_mlb_players.return_value = [
            MockBDLPlayer(id=12345, full_name="John Doe")
        ]

        service = PlayerAutoHealService(db_session=mock_db, bdl_client=mock_bdl)
        service.heal_player(yahoo_id="7590", yahoo_key="469.p.7590", name="John Doe")

        new_row = mock_db.add.call_args[0][0]
        assert new_row.source == "bdl_search"

    def test_healed_at_set_on_success(self):
        """healed_at timestamp is set when heal succeeds."""
        mock_db = Mock()
        mock_bdl = Mock()

        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_bdl.search_mlb_players.return_value = [
            MockBDLPlayer(id=12345, full_name="John Doe")
        ]

        service = PlayerAutoHealService(db_session=mock_db, bdl_client=mock_bdl)
        service.heal_player(yahoo_id="7590", yahoo_key="469.p.7590", name="John Doe")

        new_row = mock_db.add.call_args[0][0]
        assert new_row.healed_at is not None
        assert isinstance(new_row.healed_at, datetime)

    def test_heal_attempts_incremented_on_retry(self):
        """heal_attempts counter is incremented each time the row is visited."""
        mock_db = Mock()
        mock_bdl = Mock()

        existing = Mock()
        existing.source = "bdl_search"
        existing.bdl_id = 12345
        existing.updated_at = datetime(2020, 1, 1)  # Stale
        existing.heal_attempts = 2
        existing.healed_at = datetime(2020, 1, 1)

        mock_db.query.return_value.filter.return_value.first.return_value = existing
        mock_bdl.search_mlb_players.return_value = [
            MockBDLPlayer(id=12345, full_name="John Doe")
        ]

        service = PlayerAutoHealService(db_session=mock_db, bdl_client=mock_bdl)
        service.heal_player(yahoo_id="7590", yahoo_key="469.p.7590", name="John Doe")

        assert existing.heal_attempts == 3


class TestNoRegression:
    """Test that existing mapped players are not affected."""

    def test_manual_mapping_not_overwritten(self):
        """Manual mappings (source='manual') are never overwritten."""
        mock_db = Mock()
        mock_bdl = Mock()

        existing = Mock()
        existing.source = "manual"
        existing.heal_attempts = 0

        mock_db.query.return_value.filter.return_value.first.return_value = existing

        service = PlayerAutoHealService(db_session=mock_db, bdl_client=mock_bdl)
        result = service.heal_player(yahoo_id="7590", yahoo_key="469.p.7590", name="John Doe")

        assert result is False
        mock_bdl.search_mlb_players.assert_not_called()
        assert existing.heal_attempts == 1  # Still incremented for observability

    def test_fresh_bdl_search_not_re_processed(self):
        """Fresh auto-healed mappings (< 7 days) are not re-queried."""
        from zoneinfo import ZoneInfo

        mock_db = Mock()
        mock_bdl = Mock()

        existing = Mock()
        existing.source = "bdl_search"
        existing.bdl_id = 12345
        existing.updated_at = datetime.now(ZoneInfo("America/New_York"))  # Fresh
        existing.heal_attempts = 1

        mock_db.query.return_value.filter.return_value.first.return_value = existing

        service = PlayerAutoHealService(db_session=mock_db, bdl_client=mock_bdl)
        result = service.heal_player(yahoo_id="7590", yahoo_key="469.p.7590", name="John Doe")

        assert result is False
        mock_bdl.search_mlb_players.assert_not_called()


class TestBatchHeal:
    """Test batch healing of multiple players."""

    def test_batch_heal_processes_multiple_players(self):
        """batch_heal processes multiple unmapped players."""
        mock_db = Mock()
        mock_bdl = Mock()

        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_bdl.search_mlb_players.side_effect = [
            [MockBDLPlayer(id=1, full_name="Player One")],
            [MockBDLPlayer(id=2, full_name="Player Two")],
        ]

        service = PlayerAutoHealService(db_session=mock_db, bdl_client=mock_bdl)

        unmatched = [
            {"name": "Player One", "yahoo_id": "1", "yahoo_key": "469.p.1"},
            {"name": "Player Two", "yahoo_id": "2", "yahoo_key": "469.p.2"},
        ]

        summary = service.batch_heal(unmatched)

        assert summary["healed"] == 2
        assert summary["failed"] == 0
        assert summary["skipped"] == 0

    def test_batch_heal_skips_collision_reasons(self):
        """batch_heal skips players with collision/conflict reasons."""
        mock_db = Mock()
        mock_bdl = Mock()

        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_bdl.search_mlb_players.return_value = [
            MockBDLPlayer(id=2, full_name="Player Two")
        ]

        service = PlayerAutoHealService(db_session=mock_db, bdl_client=mock_bdl)

        unmatched = [
            {"name": "Player One", "yahoo_id": "1", "yahoo_key": "469.p.1", "reason": "bdl_name_collision"},
            {"name": "Player Two", "yahoo_id": "2", "yahoo_key": "469.p.2"},
        ]

        summary = service.batch_heal(unmatched)

        assert summary["skipped"] == 1
        assert summary["healed"] == 1
        mock_bdl.search_mlb_players.assert_called_once()


class TestHelperFunctions:
    """Test pure helper functions."""

    def test_normalize_removes_accents(self):
        assert _normalize("José García") == "jose garcia"
        assert _normalize("  Mixed CASE  ") == "mixed case"

    def test_name_confidence_exact_match(self):
        assert _name_confidence("john doe", "john doe") == 1.0

    def test_name_confidence_substring(self):
        assert _name_confidence("john", "johnny doe") == 0.9
        assert _name_confidence("johnny doe", "john") == 0.9

    def test_name_confidence_no_match(self):
        assert _name_confidence("john doe", "jane smith") == 0.0

    def test_is_fresh_within_ttl(self):
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/New_York"))
        assert _is_fresh(now) is True

    def test_is_fresh_beyond_ttl(self):
        from zoneinfo import ZoneInfo
        old = datetime(2020, 1, 1, tzinfo=ZoneInfo("America/New_York"))
        assert _is_fresh(old) is False


class TestYahooClientIntegration:
    """Test integration wiring in yahoo_client_resilient."""

    @patch("backend.fantasy_baseball.yahoo_client_resilient.threading.Thread")
    def test_trigger_auto_heal_starts_background_thread(self, mock_thread):
        """_trigger_auto_heal_for_unmapped starts a daemon background thread."""
        from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient

        client = MagicMock(spec=YahooFantasyClient)
        players = [
            {"player_key": "469.p.7590", "player_id": "7590", "name": "Test Player"}
        ]

        YahooFantasyClient._trigger_auto_heal_for_unmapped(client, players)

        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()
