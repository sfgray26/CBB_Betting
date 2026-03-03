"""
Circuit breaker for external HTTP dependencies.

States
------
CLOSED     Normal operation — requests pass through.
OPEN       Too many recent failures — requests blocked immediately.
HALF_OPEN  Recovery probe — a single request is allowed through to test the service.

Transitions
-----------
CLOSED  → OPEN       after ``failure_threshold`` failures inside ``window_seconds``
OPEN    → HALF_OPEN  after ``recovery_timeout`` seconds
HALF_OPEN → CLOSED   on success
HALF_OPEN → OPEN     on failure (reset recovery clock)
"""

import logging
import time
from enum import Enum
from typing import List

logger = logging.getLogger(__name__)


class CBState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Rolling-window circuit breaker.

    Args:
        failure_threshold: Number of failures within ``window_seconds``
            required to trip the breaker.
        recovery_timeout:  Seconds to wait in OPEN state before allowing
            a probe request (transition to HALF_OPEN).
        window_seconds:    Width of the rolling failure window.  Failures
            older than this are discarded.

    Example::

        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=300)

        if cb.should_allow_request():
            try:
                resp = requests.get(url)
                cb.record_success()
            except Exception:
                cb.record_failure()
                raise
        else:
            raise RuntimeError("Circuit open — service unavailable")
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 300.0,
        window_seconds: float = 600.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.window_seconds = window_seconds

        self._state: CBState = CBState.CLOSED
        self._failure_times: List[float] = []   # epoch seconds
        self._opened_at: float = 0.0            # epoch seconds when last tripped

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_allow_request(self) -> bool:
        """Return True if a request should be attempted."""
        now = time.monotonic()
        self._prune_failures(now)

        if self._state is CBState.CLOSED:
            return True

        if self._state is CBState.OPEN:
            if now - self._opened_at >= self.recovery_timeout:
                logger.info(
                    "CircuitBreaker: OPEN → HALF_OPEN after %.0fs",
                    now - self._opened_at,
                )
                self._state = CBState.HALF_OPEN
                return True
            return False

        # HALF_OPEN — only one probe at a time (caller must call
        # record_success / record_failure to advance state)
        return True

    def record_success(self) -> None:
        """Record a successful call.  Closes the breaker if it was half-open."""
        if self._state is CBState.HALF_OPEN:
            logger.info("CircuitBreaker: HALF_OPEN → CLOSED (probe succeeded)")
            self._state = CBState.CLOSED
            self._failure_times.clear()

    def record_failure(self) -> None:
        """Record a failed call.  May trip the breaker to OPEN."""
        now = time.monotonic()
        self._prune_failures(now)
        self._failure_times.append(now)

        if self._state is CBState.HALF_OPEN:
            # Failed probe → stay open, reset recovery clock
            logger.warning(
                "CircuitBreaker: probe failed — HALF_OPEN → OPEN (recovery reset)"
            )
            self._state = CBState.OPEN
            self._opened_at = now
            return

        if len(self._failure_times) >= self.failure_threshold:
            if self._state is not CBState.OPEN:
                logger.warning(
                    "CircuitBreaker: CLOSED → OPEN (%d failures in %.0fs window)",
                    len(self._failure_times),
                    self.window_seconds,
                )
                self._state = CBState.OPEN
                self._opened_at = now

    @property
    def state(self) -> str:
        return self._state.value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune_failures(self, now: float) -> None:
        """Discard failure timestamps older than the rolling window."""
        cutoff = now - self.window_seconds
        self._failure_times = [t for t in self._failure_times if t > cutoff]
