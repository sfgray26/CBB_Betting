"""
Cache Service Unit Tests

Tests for Redis caching implementation based on K-31 Railway Redis research.
Validates serialization, TTL strategies, and cache operations.
"""
import os
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, MagicMock, patch

from backend.services.cache_service import (
    CacheService,
    SerializationManager,
    TTLConfig,
    FantasyBaseballTTL,
    get_dynamic_ttl,
)


class TestSerializationManager:
    """Test serialization manager (K-31 Section 4.2)."""

    def test_json_encode_decode(self):
        """Test JSON serialization format."""
        data = {"player_id": 123, "name": "Test Player", "stats": {"hr": 25, "avg": 0.285}}

        encoded = SerializationManager.encode(data, format="json")
        decoded = SerializationManager.decode(encoded)

        assert decoded == data
        assert isinstance(encoded, bytes)

    def test_msgpack_encode_decode(self):
        """Test msgpack serialization format (if available)."""
        pytest.importorskip("msgpack")

        data = {"player_id": 123, "name": "Test Player", "stats": {"hr": 25, "avg": 0.285}}

        encoded = SerializationManager.encode(data, format="msgpack")
        decoded = SerializationManager.decode(encoded)

        assert decoded == data
        assert isinstance(encoded, bytes)

    def test_auto_format_chooses_msgpack(self):
        """Test auto format prefers msgpack if available."""
        pytest.importorskip("msgpack")

        data = {"test": "data"}

        # Should use msgpack when available
        encoded = SerializationManager.encode(data, format="auto")
        decoded = SerializationManager.decode(encoded)

        assert decoded == data

    def test_auto_format_fallback_to_json(self):
        """Test auto format falls back to JSON if msgpack unavailable."""
        # Mock msgpack as unavailable
        with patch('backend.services.cache_service.MSGPACK_AVAILABLE', False):
            data = {"test": "data"}

            encoded = SerializationManager.encode(data, format="auto")
            decoded = SerializationManager.decode(encoded)

            assert decoded == data

    def test_compression_for_large_objects(self):
        """Test compression is applied to objects >1KB (K-31)."""
        # Create data larger than 1KB
        large_data = {"data": "x" * 2000}  # ~2KB

        encoded = SerializationManager.encode(large_data)

        # Should have compression flag (\\x01)
        assert encoded[0] == 1  # Compressed

        # Decode should work
        decoded = SerializationManager.decode(encoded)
        assert decoded == large_data

    def test_no_compression_for_small_objects(self):
        """Test compression is skipped for objects <1KB."""
        small_data = {"data": "x" * 100}  # ~100 bytes

        encoded = SerializationManager.encode(small_data)

        # Should have uncompressed flag (\\x00)
        assert encoded[0] == 0  # Not compressed

        decoded = SerializationManager.decode(encoded)
        assert decoded == small_data

    def test_decode_none(self):
        """Test decoding None returns None."""
        assert SerializationManager.decode(None) is None

    def test_decode_empty_bytes(self):
        """Test decoding empty bytes returns None."""
        assert SerializationManager.decode(b'') is None


class TestTTLConfig:
    """Test TTL configuration with jitter (K-31 Section 5.1)."""

    def test_ttl_without_jitter(self):
        """Test TTL with zero jitter returns exact base TTL."""
        config = TTLConfig(base_ttl=300, jitter_pct=0.0)
        assert config.calculate() == 300

    def test_ttl_with_jitter(self):
        """Test TTL with jitter varies within expected range."""
        config = TTLConfig(base_ttl=300, jitter_pct=0.1)  # ±10% = ±30 seconds

        # Run multiple times to test distribution
        ttls = [config.calculate() for _ in range(100)]

        # All should be within [270, 330]
        assert all(270 <= ttl <= 330 for ttl in ttls)

        # Average should be close to base
        avg_ttl = sum(ttls) / len(ttls)
        assert 295 <= avg_ttl <= 305  # Allow small sampling error


class TestFantasyBaseballTTL:
    """Test fantasy baseball TTL constants (K-31 Section 5.1)."""

    def test_player_stats_ttl(self):
        """Test player stats TTL is 15 minutes ± jitter."""
        ttl = FantasyBaseballTTL.PLAYER_STATS.calculate()
        assert 810 <= ttl <= 990  # 15 min ± 10%

    def test_scarcity_index_ttl(self):
        """Test scarcity index TTL is 1 minute ± jitter."""
        ttl = FantasyBaseballTTL.SCARCITY_INDEX.calculate()
        assert 48 <= ttl <= 72  # 1 min ± 20%

    def test_live_odds_ttl(self):
        """Test live odds TTL is 5 minutes ± jitter."""
        ttl = FantasyBaseballTTL.LIVE_ODDS.calculate()
        assert 270 <= ttl <= 330  # 5 min ± 10%

    def test_two_start_sp_ttl(self):
        """Test two-start SP TTL is 24 hours ± jitter."""
        ttl = FantasyBaseballTTL.TWO_START_SP.calculate()
        assert 82080 <= ttl <= 90552  # 24 hr ± 5%

    def test_user_session_ttl(self):
        """Test user session TTL is exactly 1 hour (no jitter)."""
        ttl = FantasyBaseballTTL.USER_SESSION.calculate()
        assert ttl == 3600  # Exactly 1 hour


class TestDynamicTTL:
    """Test dynamic TTL based on game time (K-31 Section 5.2)."""

    def test_far_out_game_gt_24h(self):
        """Test TTL for games >24 hours away."""
        game_time = datetime.now(timezone.utc) + timedelta(hours=30)
        ttl = get_dynamic_ttl(game_time)
        assert ttl == 21600  # 6 hours

    def test_approaching_game_6_24h(self):
        """Test TTL for games 6-24 hours away."""
        game_time = datetime.now(timezone.utc) + timedelta(hours=12)
        ttl = get_dynamic_ttl(game_time)
        assert ttl == 1800  # 30 minutes

    def test_close_game_2_6h(self):
        """Test TTL for games 2-6 hours away."""
        game_time = datetime.now(timezone.utc) + timedelta(hours=4)
        ttl = get_dynamic_ttl(game_time)
        assert ttl == 300  # 5 minutes

    def test_live_game_lt_2h(self):
        """Test TTL for games <2 hours away."""
        game_time = datetime.now(timezone.utc) + timedelta(hours=1)
        ttl = get_dynamic_ttl(game_time)
        assert ttl == 60  # 1 minute

    def test_in_progress_game(self):
        """Test TTL for games that have started."""
        game_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        ttl = get_dynamic_ttl(game_time)
        assert ttl == 30  # 30 seconds


class TestCacheService:
    """Test cache service operations (K-31 Sections 3-6)."""

    @staticmethod
    def _create_enabled_cache():
        """Helper to create a CacheService with mocked Redis client (enabled)."""
        mock_redis = MagicMock()

        with patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379"}):
            cache = CacheService()
            cache.client = mock_redis
            cache.enabled = True  # Force enabled for testing
            return cache, mock_redis

    def test_cache_service_initialization_no_redis(self):
        """Test CacheService handles missing REDIS_URL gracefully."""
        with patch.dict(os.environ, clear=True):
            cache = CacheService()

            # Should be disabled but not crash
            assert cache.enabled is False
            assert cache.client is None

    def test_cache_service_initialization_with_redis(self):
        """Test CacheService initializes with Redis."""
        # Mock Redis client and ensure REDIS_URL is set
        with patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379"}):
            with patch('backend.services.cache_service.get_redis') as mock_get_redis:
                mock_redis = MagicMock()
                mock_get_redis.return_value = mock_redis

                cache = CacheService()

                assert cache.enabled is True
                assert cache.client == mock_redis

    def test_get_odds_cache_miss(self):
        """Test get_odds returns None on cache miss."""
        cache, mock_redis = self._create_enabled_cache()
        mock_redis.get.return_value = None

        result = cache.get_odds("game_123", hours_to_game=4)

        assert result is None
        mock_redis.get.assert_called_once_with("odds:mlb:game_123")

    def test_set_and_get_odds(self):
        """Test set_odds and get_odds round trip."""
        cache, mock_redis = self._create_enabled_cache()

        odds_data = {"total": 9.5, "moneyline": {"home": -150, "away": 130}}
        cache.set_odds("game_123", odds_data, hours_to_game=4)

        # Verify setex was called
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == "odds:mlb:game_123"  # Key
        # TTL should be around 300 sec for 4h out (2-6h window = 5 min TTL)

    def test_get_weather_cache_miss(self):
        """Test get_weather returns None on cache miss."""
        cache, mock_redis = self._create_enabled_cache()
        mock_redis.get.return_value = None

        result = cache.get_weather("COL", "2026-04-10")

        assert result is None
        mock_redis.get.assert_called_once_with("weather:COL:2026-04-10")

    def test_set_and_get_weather(self):
        """Test set_weather and get_weather round trip."""
        cache, mock_redis = self._create_enabled_cache()

        weather_data = {"temp_f": 75.0, "wind_speed_mph": 10.0}
        cache.set_weather("COL", "2026-04-10", weather_data)

        # Verify setex was called
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args

        # Check TTL is in expected range (30 min ± 3 min)
        ttl = call_args[0][1]
        assert 1620 <= ttl <= 1980  # 30 min ± 10%

    def test_session_operations(self):
        """Test session set/get/delete operations."""
        cache, mock_redis = self._create_enabled_cache()

        session_data = {"lineup": ["player1", "player2"], "locked": True}

        # Set session
        cache.set_session("user_123", "lineup", session_data)
        mock_redis.setex.assert_called_once()

        # Mock cache hit
        mock_redis.get.return_value = SerializationManager.encode(session_data)
        result = cache.get_session("user_123", "lineup")

        assert result == session_data

        # Delete session
        cache.delete_session("user_123", "lineup")
        mock_redis.delete.assert_called_once_with("session:user_123:lineup")

    def test_clear_all_odds_cache(self):
        """Test clearing all odds cache."""
        mock_redis = MagicMock()
        # Mock scan_iter to return some keys
        mock_redis.scan_iter.return_value = [
            b"odds:mlb:game_1",
            b"odds:mlb:game_2",
            b"odds:mlb:game_3"
        ]

        cache, _ = self._create_enabled_cache()
        cache.client = mock_redis

        cache.clear_all_odds_cache()

        # Verify delete was called for each key
        assert mock_redis.delete.call_count == 3

    def test_get_cache_stats_enabled(self):
        """Test get_cache_stats returns metrics when enabled."""
        mock_redis = MagicMock()
        mock_redis.dbsize.return_value = 100

        # Mock info() to return different dicts based on parameter
        def mock_info(section=None):
            if section == 'stats':
                return {
                    'keyspace_hits': 500,
                    'keyspace_misses': 200,
                    'connected_clients': 5,
                    'evicted_keys': 10,
                }
            elif section == 'memory':
                return {
                    'used_memory': 1048576,  # 1MB
                }
            return {}

        mock_redis.info.side_effect = mock_info

        cache, _ = self._create_enabled_cache()
        cache.client = mock_redis

        stats = cache.get_cache_stats()

        assert stats["enabled"] is True
        assert stats["total_keys"] == 100
        assert stats["hits"] == 500
        assert stats["misses"] == 200
        assert abs(stats["hit_rate"] - 0.714) < 0.001  # 500 / 700 ≈ 0.714
        assert stats["used_memory_mb"] == 1.0

    def test_get_cache_stats_disabled(self):
        """Test get_cache_stats returns disabled status when Redis unavailable."""
        cache = CacheService()
        cache.enabled = False
        cache.client = None

        stats = cache.get_cache_stats()

        assert stats["enabled"] is False

    def test_operations_when_disabled(self):
        """Test all operations are safe when cache disabled."""
        cache = CacheService()
        cache.enabled = False
        cache.client = None

        # All should return None or be no-ops
        assert cache.get_odds("game_123", 4) is None
        assert cache.get_weather("COL", "2026-04-10") is None
        assert cache.get_session("user_123", "lineup") is None

        # Set operations should be no-ops (no exceptions)
        cache.set_odds("game_123", {}, 4)
        cache.set_weather("COL", "2026-04-10", {})
        cache.set_session("user_123", "lineup", {})
        cache.delete_session("user_123", "lineup")
        cache.clear_all_odds_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
