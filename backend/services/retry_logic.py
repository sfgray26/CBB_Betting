"""
retry_logic.py — Async retry decorators with exponential backoff.

Provides async_retry decorator for handling transient failures (502 errors, rate limits)
in Statcast and other external API calls.
"""

import asyncio
import logging
from functools import wraps
from typing import Callable, TypeVar

T = TypeVar('T')

logger = logging.getLogger(__name__)


def async_retry(
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    retry_on: tuple = (Exception,)
) -> Callable:
    """
    Async decorator to retry function on failure with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 2.0)
        max_delay: Maximum delay between retries (default: 30.0)
        retry_on: Tuple of exception types to retry on (default: all Exceptions)

    Returns:
        Decorated function with retry logic

    Example:
        @async_retry(max_retries=3, base_delay=2.0)
        async def fetch_data():
            response = await api_call()
            return response
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e

                    # Don't retry on the last attempt
                    if attempt == max_retries - 1:
                        break

                    # Calculate exponential backoff delay
                    delay = min(base_delay * (2 ** attempt), max_delay)

                    # Check for specific error types
                    error_msg = str(e)
                    is_502 = "502" in error_msg or hasattr(e, 'status') and getattr(e, 'status') == 502

                    if is_502:
                        logger.warning(
                            "Statcast 502 error in %s (attempt %d/%d): retrying in %.1fs",
                            func.__name__,
                            attempt + 1,
                            max_retries,
                            delay
                        )
                    else:
                        logger.warning(
                            "Retryable error in %s (attempt %d/%d): %s - retrying in %.1fs",
                            func.__name__,
                            attempt + 1,
                            max_retries,
                            type(e).__name__,
                            delay
                        )

                    await asyncio.sleep(delay)

            # All retries exhausted
            logger.error(
                "All %d retry attempts exhausted for %s",
                max_retries,
                func.__name__
            )
            raise last_exception

        return wrapper
    return decorator


def sync_retry(
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    retry_on: tuple = (Exception,)
) -> Callable:
    """
    Synchronous retry decorator with exponential backoff.

    For use with blocking I/O operations that cannot be made async.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 2.0)
        max_delay: Maximum delay between retries (default: 30.0)
        retry_on: Tuple of exception types to retry on (default: all Exceptions)

    Returns:
        Decorated function with retry logic
    """
    import time

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e

                    # Don't retry on the last attempt
                    if attempt == max_retries - 1:
                        break

                    # Calculate exponential backoff delay
                    delay = min(base_delay * (2 ** attempt), max_delay)

                    # Check for specific error types
                    error_msg = str(e)
                    is_502 = "502" in error_msg

                    if is_502:
                        logger.warning(
                            "Statcast 502 error in %s (attempt %d/%d): retrying in %.1fs",
                            func.__name__,
                            attempt + 1,
                            max_retries,
                            delay
                        )
                    else:
                        logger.warning(
                            "Retryable error in %s (attempt %d/%d): %s - retrying in %.1fs",
                            func.__name__,
                            attempt + 1,
                            max_retries,
                            type(e).__name__,
                            delay
                        )

                    time.sleep(delay)

            # All retries exhausted
            logger.error(
                "All %d retry attempts exhausted for %s",
                max_retries,
                func.__name__
            )
            raise last_exception

        return wrapper
    return decorator
