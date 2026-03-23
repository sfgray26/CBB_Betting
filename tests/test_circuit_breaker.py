"""
Tests for backend/core/circuit_breaker.py

Run with: pytest tests/test_circuit_breaker.py -v
"""

import time
import pytest
from unittest.mock import patch
from backend.core.circuit_breaker import CircuitBreaker, CBState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cb(failure_threshold=3, recovery_timeout=30.0, window_seconds=60.0):
    return CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        window_seconds=window_seconds,
    )


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_starts_closed(self):
        cb = _make_cb()
        assert cb.state == "closed"

    def test_allows_requests_when_closed(self):
        cb = _make_cb()
        assert cb.is_closed() if hasattr(cb, "is_closed") else cb.should_allow_request()

    def test_no_failures_initially(self):
        cb = _make_cb()
        assert len(cb._failure_times) == 0


# ---------------------------------------------------------------------------
# CLOSED → OPEN transition
# ---------------------------------------------------------------------------


class TestClosedToOpen:
    def test_trips_after_threshold_failures(self):
        cb = _make_cb(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"

    def test_does_not_trip_before_threshold(self):
        cb = _make_cb(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "closed"

    def test_blocks_requests_when_open(self):
        cb = _make_cb(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.should_allow_request() is False

    def test_success_in_closed_state_is_noop(self):
        cb = _make_cb(failure_threshold=3)
        cb.record_success()  # should not raise
        assert cb.state == "closed"


# ---------------------------------------------------------------------------
# OPEN → HALF_OPEN transition (time-based)
# ---------------------------------------------------------------------------


class TestOpenToHalfOpen:
    def test_transitions_to_half_open_after_timeout(self):
        cb = _make_cb(failure_threshold=3, recovery_timeout=5.0)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"

        # Advance monotonic clock past recovery_timeout
        fake_now = time.monotonic() + 10.0
        with patch("backend.core.circuit_breaker.time.monotonic", return_value=fake_now):
            allowed = cb.should_allow_request()

        assert allowed is True
        assert cb.state == "half_open"

    def test_stays_open_before_timeout(self):
        cb = _make_cb(failure_threshold=3, recovery_timeout=300.0)
        for _ in range(3):
            cb.record_failure()
        assert cb.should_allow_request() is False
        assert cb.state == "open"


# ---------------------------------------------------------------------------
# HALF_OPEN → CLOSED (probe success)
# ---------------------------------------------------------------------------


class TestHalfOpenToClosed:
    def _trip_and_recover(self, cb):
        """Trip the breaker then advance time to trigger HALF_OPEN."""
        for _ in range(cb.failure_threshold):
            cb.record_failure()
        fake_now = time.monotonic() + cb.recovery_timeout + 1.0
        with patch("backend.core.circuit_breaker.time.monotonic", return_value=fake_now):
            cb.should_allow_request()  # triggers OPEN → HALF_OPEN

    def test_probe_success_closes_breaker(self):
        cb = _make_cb(failure_threshold=3, recovery_timeout=5.0)
        self._trip_and_recover(cb)
        assert cb.state == "half_open"

        cb.record_success()
        assert cb.state == "closed"

    def test_failure_times_cleared_on_close(self):
        cb = _make_cb(failure_threshold=3, recovery_timeout=5.0)
        self._trip_and_recover(cb)
        cb.record_success()
        assert len(cb._failure_times) == 0


# ---------------------------------------------------------------------------
# HALF_OPEN → OPEN (probe failure)
# ---------------------------------------------------------------------------


class TestHalfOpenToOpen:
    def _trip_and_recover(self, cb):
        for _ in range(cb.failure_threshold):
            cb.record_failure()
        fake_now = time.monotonic() + cb.recovery_timeout + 1.0
        with patch("backend.core.circuit_breaker.time.monotonic", return_value=fake_now):
            cb.should_allow_request()

    def test_probe_failure_reopens_breaker(self):
        cb = _make_cb(failure_threshold=3, recovery_timeout=5.0)
        self._trip_and_recover(cb)
        assert cb.state == "half_open"

        cb.record_failure()
        assert cb.state == "open"

    def test_probe_failure_resets_recovery_clock(self):
        cb = _make_cb(failure_threshold=3, recovery_timeout=5.0)
        self._trip_and_recover(cb)
        cb.record_failure()
        # Immediately after re-opening, should still be blocked
        assert cb.should_allow_request() is False


# ---------------------------------------------------------------------------
# Rolling window expiry
# ---------------------------------------------------------------------------


class TestRollingWindow:
    def test_old_failures_expire_and_do_not_trip(self):
        cb = _make_cb(failure_threshold=3, window_seconds=60.0)

        # Record 2 failures far in the past (>60s ago)
        old_time = time.monotonic() - 120.0
        cb._failure_times.extend([old_time, old_time])

        # One new failure — with expired ones gone, total = 1 < threshold
        cb.record_failure()
        assert cb.state == "closed"

    def test_failures_within_window_accumulate(self):
        cb = _make_cb(failure_threshold=3, window_seconds=60.0)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"


# ---------------------------------------------------------------------------
# State property
# ---------------------------------------------------------------------------


class TestStateProperty:
    def test_state_returns_string(self):
        cb = _make_cb()
        assert isinstance(cb.state, str)
        assert cb.state == "closed"

    def test_state_values_match_enum(self):
        cb = _make_cb(failure_threshold=1)
        assert cb.state == CBState.CLOSED.value
        cb.record_failure()
        assert cb.state == CBState.OPEN.value
