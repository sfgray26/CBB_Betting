"""
Season configuration — single source of truth for the current MLB season year.

Read from CURRENT_MLB_SEASON environment variable; defaults to 2026.
Use get_current_season() everywhere instead of hardcoding year integers.
"""
import os


def get_current_season() -> int:
    """Return the current MLB season year. Configurable via CURRENT_MLB_SEASON env var."""
    return int(os.getenv("CURRENT_MLB_SEASON", "2026"))
