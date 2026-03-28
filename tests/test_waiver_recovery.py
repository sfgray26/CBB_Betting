"""
Tests for waiver recovery resilience patterns.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile
import shutil

from backend.fantasy_baseball.circuit_breaker import CircuitBreaker, CircuitOpenError
from backend.fantasy_baseball.cache_manager import StaleCacheManager, CacheResult, NoDataAvailableError
from backend.fantasy_baseball.position_normalizer import PositionNormalizer, Player, RosterSlot, YahooRoster


class TestCircuitBreaker:
    """Test circuit breaker state transitions."""
    
    def test_initial_state_is_closed(self):
        cb = CircuitBreaker("test")
        assert cb.state.value == "closed"
        assert cb.failure_count == 0
    
    def test_successful_call_stays_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        
        result = cb.call(lambda: "success")
        
        assert result == "success"
        assert cb.state.value == "closed"
        assert cb.failure_count == 0
    
    def test_failures_increment_count(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        
        def fail():
            raise ValueError("error")
        
        with pytest.raises(ValueError):
            cb.call(fail)
        
        assert cb.failure_count == 1
        assert cb.state.value == "closed"  # Still closed, not at threshold
    
    def test_opens_after_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=2)
        
        def fail():
            raise ValueError("error")
        
        cb.call(fail)  # Failure 1
        with pytest.raises(ValueError):
            pass  # Exception was raised and caught
        
        with pytest.raises(ValueError):
            cb.call(fail)  # Failure 2 - should open
        
        assert cb.state.value == "open"
    
    def test_rejects_calls_when_open(self):
        cb = CircuitBreaker("test", failure_threshold=1)
        
        def fail():
            raise ValueError("error")
        
        try:
            cb.call(fail)
        except ValueError:
            pass
        
        with pytest.raises(CircuitOpenError):
            cb.call(lambda: "should not run")
    
    def test_manual_open_and_close(self):
        cb = CircuitBreaker("test")
        
        cb.force_open()
        assert cb.state.value == "open"
        
        cb.force_close()
        assert cb.state.value == "closed"
        assert cb.failure_count == 0


class TestStaleCacheManager:
    """Test cache manager operations."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_write_and_read(self, temp_cache_dir):
        cache = StaleCacheManager(cache_dir=temp_cache_dir, max_age=timedelta(hours=1))
        
        test_data = {"players": [{"name": "Mike Trout", "id": "1"}]}
        cache.write("test_key", test_data)
        
        entry = cache.read("test_key")
        assert entry is not None
        assert entry.data == test_data
    
    def test_missing_key_returns_none(self, temp_cache_dir):
        cache = StaleCacheManager(cache_dir=temp_cache_dir, max_age=timedelta(hours=1))
        
        entry = cache.read("nonexistent")
        assert entry is None
    
    def test_fresh_entry_detection(self, temp_cache_dir):
        cache = StaleCacheManager(cache_dir=temp_cache_dir, max_age=timedelta(hours=1))
        
        cache.write("fresh", {"data": "test"})
        entry = cache.read("fresh")
        
        assert cache.is_fresh(entry)
    
    def test_stale_entry_detection(self, temp_cache_dir):
        cache = StaleCacheManager(cache_dir=temp_cache_dir, max_age=timedelta(seconds=1))
        
        cache.write("stale", {"data": "test"})
        
        # Simulate time passing
        import time
        time.sleep(1.1)
        
        entry = cache.read("stale")
        assert not cache.is_fresh(entry)
    
    def test_invalidate_removes_entry(self, temp_cache_dir):
        cache = StaleCacheManager(cache_dir=temp_cache_dir, max_age=timedelta(hours=1))
        
        cache.write("remove_me", {"data": "test"})
        cache.invalidate("remove_me")
        
        assert cache.read("remove_me") is None
    
    def test_clear_all_removes_all(self, temp_cache_dir):
        cache = StaleCacheManager(cache_dir=temp_cache_dir, max_age=timedelta(hours=1))
        
        cache.write("key1", {"data": "1"})
        cache.write("key2", {"data": "2"})
        cache.clear_all()
        
        assert cache.read("key1") is None
        assert cache.read("key2") is None
    
    @pytest.mark.asyncio
    async def test_get_with_fallback_success(self, temp_cache_dir):
        cache = StaleCacheManager(cache_dir=temp_cache_dir, max_age=timedelta(hours=1))
        
        async def fetch():
            return {"fresh": "data"}
        
        result = await cache.get_with_fallback("test", fetch)
        
        assert result.fresh is True
        assert result.source == "api"
        assert result.data == {"fresh": "data"}
    
    @pytest.mark.asyncio
    async def test_get_with_fallback_uses_cache(self, temp_cache_dir):
        cache = StaleCacheManager(cache_dir=temp_cache_dir, max_age=timedelta(hours=1))
        
        # Pre-populate cache
        cache.write("cached_key", {"cached": "data"})
        
        async def failing_fetch():
            raise ValueError("API down")
        
        result = await cache.get_with_fallback("cached_key", failing_fetch)
        
        assert result.fresh is False
        assert result.source == "cache"
        assert result.data == {"cached": "data"}
    
    @pytest.mark.asyncio
    async def test_get_with_fallback_raises_when_no_cache(self, temp_cache_dir):
        cache = StaleCacheManager(cache_dir=temp_cache_dir, max_age=timedelta(hours=1))
        
        async def failing_fetch():
            raise ValueError("API down")
        
        with pytest.raises(NoDataAvailableError):
            await cache.get_with_fallback("no_cache", failing_fetch)


class TestPositionNormalizer:
    """Test position normalization logic."""
    
    def test_normalize_single_position(self):
        assert PositionNormalizer.normalize_position("2B") == "2B"
        assert PositionNormalizer.normalize_position("sp") == "SP"
    
    def test_normalize_outfield_positions(self):
        assert PositionNormalizer.normalize_position("LF") == "LF"
        assert PositionNormalizer.normalize_position("of") == "OF"
    
    def test_normalize_dh_to_util(self):
        assert PositionNormalizer.normalize_position("DH") == "Util"
    
    def test_is_eligible_for_slot_direct_match(self):
        player = Player(id="1", name="Test", positions=["2B"])
        player.yahoo_positions = ["2B"]
        
        assert PositionNormalizer.is_eligible_for_slot(player, "2B")
        assert not PositionNormalizer.is_eligible_for_slot(player, "3B")
    
    def test_is_eligible_outfield_flexibility(self):
        player = Player(id="1", name="Test", positions=["CF"])
        player.yahoo_positions = ["CF"]
        
        # LF/CF/RF can fill OF slot
        assert PositionNormalizer.is_eligible_for_slot(player, "OF")
    
    def test_is_eligible_utility_slot(self):
        player = Player(id="1", name="Test", positions=["1B"])
        player.yahoo_positions = ["1B"]
        
        # Hitters can fill Util
        assert PositionNormalizer.is_eligible_for_slot(player, "Util")
    
    def test_normalize_lineup_basic(self):
        optimized = {
            "starters": [
                {"id": "p1", "name": "Player 1", "positions": ["2B"]},
                {"id": "p2", "name": "Player 2", "positions": ["3B"]},
            ]
        }
        
        roster = YahooRoster(
            slots=[
                RosterSlot(id="s1", position="2B"),
                RosterSlot(id="s2", position="3B"),
            ],
            players=[]
        )
        
        assignments = PositionNormalizer.normalize_lineup(optimized, roster)
        
        assert assignments == {"s1": "p1", "s2": "p2"}
    
    def test_normalize_lineup_with_util_fallback(self):
        optimized = {
            "starters": [
                {"id": "p1", "name": "DH Only", "positions": ["DH"]},
            ]
        }
        
        roster = YahooRoster(
            slots=[
                RosterSlot(id="s1", position="Util"),
            ],
            players=[]
        )
        
        assignments = PositionNormalizer.normalize_lineup(optimized, roster)
        
        assert assignments == {"s1": "p1"}
    
    def test_validate_lineup_detects_mismatch(self):
        assignments = {"slot1": "player1"}
        
        roster = YahooRoster(
            slots=[RosterSlot(id="slot1", position="2B")],
            players=[
                Player(
                    id="player1", 
                    name="Wrong Pos", 
                    positions=["SP"],
                    yahoo_positions=["SP"]
                )
            ]
        )
        
        result = PositionNormalizer.validate_lineup_before_submit(assignments, roster)
        
        assert not result.valid
        assert len(result.errors) > 0


class TestIntegration:
    """Integration tests for full waiver recovery flow."""
    
    @pytest.mark.asyncio
    async def test_circuit_opens_then_uses_cache(self):
        """Full flow: API fails -> circuit opens -> serves cache."""
        
        # Setup
        cb = CircuitBreaker("test_api", failure_threshold=2)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = StaleCacheManager(cache_dir=temp_dir, max_age=timedelta(hours=1))
            
            # Pre-populate cache
            cache.write("waiver_data", [{"name": "Cached Player"}])
            
            # Simulate API failing twice (opens circuit)
            call_count = 0
            def failing_api():
                nonlocal call_count
                call_count += 1
                raise ValueError(f"API Error {call_count}")
            
            # First failure
            with pytest.raises(ValueError):
                cb.call(failing_api)
            
            # Second failure - circuit should open
            with pytest.raises(ValueError):
                cb.call(failing_api)
            
            assert cb.state.value == "open"
            
            # Third call should raise CircuitOpenError
            with pytest.raises(CircuitOpenError):
                cb.call(failing_api)
            
            # Now simulate serving from cache when circuit is open
            cached = cache.read("waiver_data")
            assert cached is not None
            assert cached.data == [{"name": "Cached Player"}]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
