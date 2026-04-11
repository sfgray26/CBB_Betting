"""
test_statcast_retry.py — Tests for Statcast retry logic with exponential backoff.

Tests the retry_logic module and ensures proper handling of 502 errors from
pybaseball/Statcast API calls.
"""

import asyncio
import time
from unittest.mock import Mock, patch
import pytest

from backend.services.retry_logic import async_retry, sync_retry


class TestSyncRetry:
    """Test synchronous retry decorator."""

    def test_sync_retry_success_on_first_attempt(self):
        """Test that successful calls work without retries."""
        @sync_retry(max_retries=3, base_delay=0.1)
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"

    def test_sync_retry_success_after_failure(self):
        """Test that retry works after initial failure."""
        call_count = 0

        @sync_retry(max_retries=3, base_delay=0.1)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert call_count == 2

    def test_sync_retry_exhausted_retries(self):
        """Test that function raises after max retries exhausted."""
        @sync_retry(max_retries=2, base_delay=0.1)
        def always_failing_function():
            raise Exception("Permanent failure")

        with pytest.raises(Exception) as exc_info:
            always_failing_function()

        assert str(exc_info.value) == "Permanent failure"

    def test_sync_retry_502_error_detection(self):
        """Test that 502 errors are properly detected and logged."""
        call_count = 0

        @sync_retry(max_retries=3, base_delay=0.1)
        def function_with_502():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("HTTP 502 Bad Gateway")
            return "recovered"

        result = function_with_502()
        assert result == "recovered"
        assert call_count == 2

    def test_sync_retry_exponential_backoff(self):
        """Test that exponential backoff delays are correct."""
        call_times = []

        @sync_retry(max_retries=4, base_delay=0.1, max_delay=1.0)
        def timing_function():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise Exception("Fail")
            return "done"

        start = time.time()
        result = timing_function()
        elapsed = time.time() - start

        assert result == "done"
        assert len(call_times) == 3

        # Check exponential backoff: 0.1s delay, then 0.2s delay
        # First retry delay should be ~0.1s
        first_retry_delay = call_times[1] - call_times[0]
        # Second retry delay should be ~0.2s
        second_retry_delay = call_times[2] - call_times[1]

        assert first_retry_delay >= 0.08  # Allow some tolerance
        assert second_retry_delay >= 0.15  # Should be roughly 2x first delay

    def test_sync_retry_max_delay_cap(self):
        """Test that max_delay cap is respected."""
        call_count = 0

        @sync_retry(max_retries=5, base_delay=1.0, max_delay=0.2)
        def capped_delay_function():
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise Exception("Fail")
            return "done"

        start = time.time()
        result = capped_delay_function()
        elapsed = time.time() - start

        assert result == "done"
        assert call_count == 4

        # With 3 retries, even with base_delay=1.0, max_delay=0.2 should cap each retry
        # Maximum total time should be approximately 3 * 0.2 = 0.6s
        assert elapsed < 1.0  # Should complete quickly due to cap

    def test_sync_retry_specific_exception_only(self):
        """Test retry only on specific exception types."""
        call_count = 0

        @sync_retry(max_retries=3, base_delay=0.1, retry_on=(ValueError,))
        def selective_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Retry this")
            elif call_count == 2:
                raise TypeError("Don't retry this")
            return "success"

        # Should retry on ValueError but not on TypeError
        with pytest.raises(TypeError) as exc_info:
            selective_function()

        assert str(exc_info.value) == "Don't retry this"
        assert call_count == 2  # Called twice (initial + 1 retry)


class TestAsyncRetry:
    """Test async retry decorator."""

    @pytest.mark.asyncio
    async def test_async_retry_success_on_first_attempt(self):
        """Test that successful async calls work without retries."""
        @async_retry(max_retries=3, base_delay=0.1)
        async def successful_async_function():
            return "async_success"

        result = await successful_async_function()
        assert result == "async_success"

    @pytest.mark.asyncio
    async def test_async_retry_success_after_failure(self):
        """Test that async retry works after initial failure."""
        call_count = 0

        @async_retry(max_retries=3, base_delay=0.1)
        async def flaky_async_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Async temporary failure")
            return "async_success"

        result = await flaky_async_function()
        assert result == "async_success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_retry_exhausted_retries(self):
        """Test that async function raises after max retries exhausted."""
        @async_retry(max_retries=2, base_delay=0.1)
        async def always_failing_async_function():
            raise Exception("Async permanent failure")

        with pytest.raises(Exception) as exc_info:
            await always_failing_async_function()

        assert str(exc_info.value) == "Async permanent failure"

    @pytest.mark.asyncio
    async def test_async_retry_exponential_backoff(self):
        """Test that async exponential backoff delays are correct."""
        call_times = []

        @async_retry(max_retries=4, base_delay=0.1, max_delay=1.0)
        async def timing_async_function():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise Exception("Async fail")
            return "async_done"

        start = time.time()
        result = await timing_async_function()
        elapsed = time.time() - start

        assert result == "async_done"
        assert len(call_times) == 3

        # Check exponential backoff timing
        first_retry_delay = call_times[1] - call_times[0]
        second_retry_delay = call_times[2] - call_times[1]

        assert first_retry_delay >= 0.08
        assert second_retry_delay >= 0.15  # Should be roughly 2x first delay


class TestPybaseballIntegration:
    """Test integration with pybaseball loader."""

    def test_retry_functions_exist(self):
        """Test that retry-protected functions are properly defined."""
        import backend.fantasy_baseball.pybaseball_loader as loader

        # Verify the retry-protected functions exist
        assert hasattr(loader, '_fetch_batting_stats_with_retry')
        assert hasattr(loader, '_fetch_pitching_stats_with_retry')
        assert hasattr(loader, '_fetch_sprint_speed_with_retry')

        # Verify they are callable
        assert callable(loader._fetch_batting_stats_with_retry)
        assert callable(loader._fetch_pitching_stats_with_retry)
        assert callable(loader._fetch_sprint_speed_with_retry)

    def test_retry_functions_have_correct_signature(self):
        """Test that retry functions have expected signatures."""
        import backend.fantasy_baseball.pybaseball_loader as loader
        import inspect

        # Check batting stats signature
        batting_sig = inspect.signature(loader._fetch_batting_stats_with_retry)
        assert 'year' in batting_sig.parameters
        assert 'qual' in batting_sig.parameters

        # Check pitching stats signature
        pitching_sig = inspect.signature(loader._fetch_pitching_stats_with_retry)
        assert 'year' in pitching_sig.parameters
        assert 'qual' in pitching_sig.parameters

        # Check sprint speed signature
        sprint_sig = inspect.signature(loader._fetch_sprint_speed_with_retry)
        assert 'year' in sprint_sig.parameters


class TestErrorScenarios:
    """Test various error scenarios."""

    def test_network_timeout_retry(self):
        """Test retry behavior on network timeouts."""
        call_count = 0

        @sync_retry(max_retries=3, base_delay=0.1)
        def timeout_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                import requests
                raise requests.exceptions.Timeout("Request timed out")
            return "success"

        result = timeout_function()
        assert result == "success"
        assert call_count == 2

    def test_connection_error_retry(self):
        """Test retry behavior on connection errors."""
        call_count = 0

        @sync_retry(max_retries=3, base_delay=0.1)
        def connection_error_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                import requests
                raise requests.exceptions.ConnectionError("Connection refused")
            return "success"

        result = connection_error_function()
        assert result == "success"
        assert call_count == 2

    def test_non_retryable_error_immediate_fail(self):
        """Test that non-retryable errors fail immediately."""
        call_count = 0

        @sync_retry(max_retries=3, base_delay=0.1, retry_on=(ValueError,))
        def mixed_error_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Retry this")
            elif call_count == 2:
                raise RuntimeError("Don't retry this")
            return "success"

        # Should retry on ValueError but fail on RuntimeError
        with pytest.raises(RuntimeError):
            mixed_error_function()

        assert call_count == 2  # Initial + 1 retry
