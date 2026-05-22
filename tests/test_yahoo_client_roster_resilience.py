"""
Unit tests for YahooFantasyClient.get_roster() resilience and edge case handling.

These tests verify robust handling of:
- Missing 'count' field entirely
- count=0 but players array has items
- count mismatch with actual players length
- Empty players array
- None/null values in various fields
- Invalid response formats
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock

# Import the client and exceptions
from backend.fantasy_baseball.yahoo_client_resilient import (
    YahooFantasyClient,
    YahooAPIError,
)


class TestGetRosterResilience:
    """Test suite for get_roster() resilience patterns."""

    @pytest.fixture
    def client(self):
        """Create a mock client with necessary attributes."""
        with patch.dict('os.environ', {
            'YAHOO_CLIENT_ID': 'test_client_id_12345',
            'YAHOO_CLIENT_SECRET': 'test_client_secret_12345',
            'YAHOO_LEAGUE_ID': '123',
            'YAHOO_REFRESH_TOKEN': 'test_refresh_token_12345'
        }):
            client = YahooFantasyClient()
            client.token = {'access_token': 'test_token'}
            client.league_id = '123'
            return client

    @pytest.fixture
    def mock_player_data(self):
        """Return sample player data structure."""
        return {
            "player_key": "123.p.456",
            "player_id": "456",
            "name": {"full": "Test Player"},
            "display_position": "OF",
        }

    def _create_roster_response(self, players_dict, count=None):
        """Helper to create Yahoo API roster response structure."""
        players_data = {"players": players_dict}
        if count is not None:
            players_data["players"]["count"] = count
        
        return {
            "fantasy_content": {
                "team": [
                    {},
                    {
                        "roster": {
                            "0": players_data
                        }
                    }
                ]
            }
        }

    # Test 1: Missing count field entirely - should infer from keys
    def test_missing_count_field_infers_from_keys(self, client, mock_player_data, caplog):
        """When 'count' is missing, infer from numeric keys."""
        caplog.set_level(logging.WARNING)
        
        # Setup: players without count field
        players_dict = {
            "0": {"player": mock_player_data},
            "1": {"player": {**mock_player_data, "player_key": "123.p.457", "player_id": "457"}},
            "2": {"player": {**mock_player_data, "player_key": "123.p.458", "player_id": "458"}},
        }
        # No count key at all
        response = self._create_roster_response(players_dict, count=None)
        
        with patch.object(client, '_get', return_value=response):
            with patch.object(client, '_enrich_ownership_batch'):
                with patch.object(client, '_trigger_auto_heal_for_unmapped'):
                    roster = client.get_roster("123.l.1.t.1")
        
        assert len(roster) == 3
        assert "missing 'count' field" in caplog.text

    # Test 2: count=0 but players array has items
    def test_count_zero_with_players_infers_from_keys(self, client, mock_player_data, caplog):
        """When count=0 but players exist, infer from keys."""
        caplog.set_level(logging.WARNING)
        
        players_dict = {
            "0": {"player": mock_player_data},
            "1": {"player": {**mock_player_data, "player_key": "123.p.457", "player_id": "457"}},
            "count": 0,  # Explicitly 0
        }
        response = self._create_roster_response(players_dict, count=0)
        
        with patch.object(client, '_get', return_value=response):
            with patch.object(client, '_enrich_ownership_batch'):
                with patch.object(client, '_trigger_auto_heal_for_unmapped'):
                    roster = client.get_roster("123.l.1.t.1")
        
        assert len(roster) == 2
        assert "count=0" in caplog.text
        assert "inferred" in caplog.text.lower()

    # Test 3: count mismatch with actual players length
    def test_count_mismatch_logs_warning(self, client, mock_player_data, caplog):
        """When count doesn't match actual players, log warning."""
        caplog.set_level(logging.WARNING)
        
        players_dict = {
            "0": {"player": mock_player_data},
            "1": {"player": {**mock_player_data, "player_key": "123.p.457", "player_id": "457"}},
            "count": 5,  # Says 5 but only 2 players
        }
        response = self._create_roster_response(players_dict, count=5)
        
        with patch.object(client, '_get', return_value=response):
            with patch.object(client, '_enrich_ownership_batch'):
                with patch.object(client, '_trigger_auto_heal_for_unmapped'):
                    roster = client.get_roster("123.l.1.t.1")
        
        assert len(roster) == 2
        assert "count mismatch" in caplog.text
        assert "missing indices" in caplog.text

    # Test 4: Empty players array
    def test_empty_players_array_raises_error(self, client, caplog):
        """When players_raw is empty, raise YahooAPIError."""
        caplog.set_level(logging.ERROR)
        
        response = {
            "fantasy_content": {
                "team": [
                    {},
                    {"roster": {"0": {"players": {}}}}
                ]
            }
        }
        
        with patch.object(client, '_get', return_value=response):
            with pytest.raises(YahooAPIError) as exc_info:
                client.get_roster("123.l.1.t.1")
        
        assert "Empty roster response" in str(exc_info.value)
        assert "players_raw is empty" in caplog.text

    # Test 5: None/null values for count
    def test_null_count_value_logs_warning(self, client, mock_player_data, caplog):
        """When count is explicitly null/None, log warning and infer."""
        caplog.set_level(logging.WARNING)
        
        players_dict = {
            "0": {"player": mock_player_data},
            "count": None,  # Explicitly null
        }
        response = self._create_roster_response(players_dict, count=None)
        # Manually set to None since _create_roster_response skips None
        response["fantasy_content"]["team"][1]["roster"]["0"]["players"]["count"] = None
        
        with patch.object(client, '_get', return_value=response):
            with patch.object(client, '_enrich_ownership_batch'):
                with patch.object(client, '_trigger_auto_heal_for_unmapped'):
                    roster = client.get_roster("123.l.1.t.1")
        
        assert len(roster) == 1
        assert "count' field is null" in caplog.text

    # Test 6: Invalid count type (string instead of int)
    def test_invalid_count_type_logs_warning(self, client, mock_player_data, caplog):
        """When count is invalid type, log warning and infer from keys."""
        caplog.set_level(logging.WARNING)
        
        players_dict = {
            "0": {"player": mock_player_data},
            "1": {"player": {**mock_player_data, "player_key": "123.p.457", "player_id": "457"}},
            "count": "invalid",  # String instead of int
        }
        response = self._create_roster_response(players_dict, count="invalid")
        
        with patch.object(client, '_get', return_value=response):
            with patch.object(client, '_enrich_ownership_batch'):
                with patch.object(client, '_trigger_auto_heal_for_unmapped'):
                    roster = client.get_roster("123.l.1.t.1")
        
        assert len(roster) == 2
        assert "count' field is invalid" in caplog.text

    # Test 7: Non-dict players_raw raises error - but _safe_get converts to empty dict
    def test_non_dict_players_raw_raises_error(self, client, caplog):
        """When players_raw is not a dict, _safe_get converts to empty dict, raises YahooAPIError."""
        caplog.set_level(logging.ERROR)
        
        response = {
            "fantasy_content": {
                "team": [
                    {},
                    {"roster": {"0": {"players": "invalid"}}}  # String instead of dict
                ]
            }
        }
        
        with patch.object(client, '_get', return_value=response):
            with pytest.raises(YahooAPIError) as exc_info:
                client.get_roster("123.l.1.t.1")
        
        # _safe_get converts non-dict to empty dict, which triggers empty check
        assert "Empty roster response" in str(exc_info.value)

    # Test 8: Extra indices beyond count
    def test_extra_indices_logs_warning(self, client, mock_player_data, caplog):
        """When there are more player indices than count, log warning."""
        caplog.set_level(logging.WARNING)
        
        players_dict = {
            "0": {"player": mock_player_data},
            "1": {"player": {**mock_player_data, "player_key": "123.p.457", "player_id": "457"}},
            "2": {"player": {**mock_player_data, "player_key": "123.p.458", "player_id": "458"}},
            "count": 2,  # Says 2 but has 3 players
        }
        response = self._create_roster_response(players_dict, count=2)
        
        with patch.object(client, '_get', return_value=response):
            with patch.object(client, '_enrich_ownership_batch'):
                with patch.object(client, '_trigger_auto_heal_for_unmapped'):
                    roster = client.get_roster("123.l.1.t.1")
        
        assert len(roster) == 2  # Only processes up to count
        assert "extra indices" in caplog.text

    # Test 9: Normal operation with correct count
    def test_normal_operation_succeeds(self, client, mock_player_data, caplog):
        """Normal case with correct count works as expected."""
        caplog.set_level(logging.INFO)
        
        players_dict = {
            "0": {"player": mock_player_data},
            "1": {"player": {**mock_player_data, "player_key": "123.p.457", "player_id": "457"}},
            "2": {"player": {**mock_player_data, "player_key": "123.p.458", "player_id": "458"}},
            "count": 3,
        }
        response = self._create_roster_response(players_dict, count=3)
        
        with patch.object(client, '_get', return_value=response):
            with patch.object(client, '_enrich_ownership_batch'):
                with patch.object(client, '_trigger_auto_heal_for_unmapped'):
                    roster = client.get_roster("123.l.1.t.1")
        
        assert len(roster) == 3
        assert "Processed roster" in caplog.text
        # Should not have warnings for normal case
        assert "missing" not in caplog.text.lower() or "mismatch" not in caplog.text.lower()

    # Test 10: Empty entry at index (gap in sequence)
    def test_empty_entry_at_index_skips_gracefully(self, client, mock_player_data, caplog):
        """When an index has no data, skip gracefully with debug log."""
        caplog.set_level(logging.DEBUG)
        
        players_dict = {
            "0": {"player": mock_player_data},
            "1": {},  # Empty entry
            "2": {"player": {**mock_player_data, "player_key": "123.p.458", "player_id": "458"}},
            "count": 3,
        }
        response = self._create_roster_response(players_dict, count=3)
        
        with patch.object(client, '_get', return_value=response):
            with patch.object(client, '_enrich_ownership_batch'):
                with patch.object(client, '_trigger_auto_heal_for_unmapped'):
                    roster = client.get_roster("123.l.1.t.1")
        
        assert len(roster) == 2  # Only 2 valid players
        assert "No player data at index 1" in caplog.text

    # Test 11: Count > 0 but no players processed logs error
    def test_count_positive_but_no_players_logs_error(self, client, caplog):
        """When count > 0 but no valid players, log error."""
        caplog.set_level(logging.ERROR)
        
        players_dict = {
            "0": {},  # Empty
            "1": {},  # Empty
            "count": 2,
        }
        response = self._create_roster_response(players_dict, count=2)
        
        with patch.object(client, '_get', return_value=response):
            with patch.object(client, '_enrich_ownership_batch'):
                with patch.object(client, '_trigger_auto_heal_for_unmapped'):
                    roster = client.get_roster("123.l.1.t.1")
        
        assert len(roster) == 0
        assert "Roster parsing failed" in caplog.text

    # Test 12: Legitimate empty roster (count=0, no players)
    def test_legitimate_empty_roster_returns_empty_list(self, client, caplog):
        """When roster is truly empty (new team), return empty list with warning."""
        caplog.set_level(logging.WARNING)
        
        players_dict = {
            "count": 0,
        }
        response = self._create_roster_response(players_dict, count=0)
        
        with patch.object(client, '_get', return_value=response):
            with patch.object(client, '_enrich_ownership_batch'):
                with patch.object(client, '_trigger_auto_heal_for_unmapped'):
                    roster = client.get_roster("123.l.1.t.1")
        
        assert roster == []
        assert "Empty roster" in caplog.text

    # Test 13: Negative count value - should infer from keys
    def test_negative_count_infers_from_keys(self, client, mock_player_data, caplog):
        """Negative count should trigger inference from keys."""
        caplog.set_level(logging.WARNING)
        
        players_dict = {
            "0": {"player": mock_player_data},
            "count": -1,  # Negative
        }
        response = self._create_roster_response(players_dict, count=-1)
        
        with patch.object(client, '_get', return_value=response):
            with patch.object(client, '_enrich_ownership_batch'):
                with patch.object(client, '_trigger_auto_heal_for_unmapped'):
                    roster = client.get_roster("123.l.1.t.1")
        
        # With negative count fix, should infer from keys
        assert len(roster) == 1
        assert "count' field is negative" in caplog.text

    # Test 14: Player without player_key uses fallback deduplication
    def test_player_without_key_uses_fallback(self, client, mock_player_data, caplog):
        """When player lacks player_key, use player_id or name for dedup."""
        players_dict = {
            "0": {"player": {**mock_player_data, "player_key": None}},  # No player_key
            "count": 1,
        }
        response = self._create_roster_response(players_dict, count=1)
        
        with patch.object(client, '_get', return_value=response):
            with patch.object(client, '_enrich_ownership_batch'):
                with patch.object(client, '_trigger_auto_heal_for_unmapped'):
                    roster = client.get_roster("123.l.1.t.1")
        
        assert len(roster) == 1
        assert roster[0].get("player_id") == "456"

    # Test 15: Large roster performance
    def test_large_roster_performance(self, client, mock_player_data):
        """Test that large rosters are handled efficiently."""
        players_dict = {str(i): {"player": {**mock_player_data, "player_key": f"123.p.{i}", "player_id": str(i)}} for i in range(100)}
        players_dict["count"] = 100
        response = self._create_roster_response(players_dict, count=100)
        
        with patch.object(client, '_get', return_value=response):
            with patch.object(client, '_enrich_ownership_batch'):
                with patch.object(client, '_trigger_auto_heal_for_unmapped'):
                    roster = client.get_roster("123.l.1.t.1")
        
        assert len(roster) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])