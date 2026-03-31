"""Tests for fantasy baseball performance and data quality fixes."""
import threading
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Task 1: Yahoo client singleton
# ---------------------------------------------------------------------------

def test_get_yahoo_client_returns_same_instance():
    """get_yahoo_client() must return the identical object on repeated calls."""
    import backend.fantasy_baseball.yahoo_client_resilient as mod
    mod._client = None

    with patch.dict("os.environ", {
        "YAHOO_CLIENT_ID": "test_id",
        "YAHOO_CLIENT_SECRET": "test_secret",
        "YAHOO_LEAGUE_ID": "12345",
        "YAHOO_REFRESH_TOKEN": "test_refresh",
        "YAHOO_ACCESS_TOKEN": "test_access",
    }):
        from backend.fantasy_baseball.yahoo_client_resilient import get_yahoo_client
        c1 = get_yahoo_client()
        c2 = get_yahoo_client()
        assert c1 is c2


def test_get_resilient_yahoo_client_returns_same_instance():
    """get_resilient_yahoo_client() must return the identical object on repeated calls."""
    import backend.fantasy_baseball.yahoo_client_resilient as mod
    mod._resilient_client = None

    with patch.dict("os.environ", {
        "YAHOO_CLIENT_ID": "test_id",
        "YAHOO_CLIENT_SECRET": "test_secret",
        "YAHOO_LEAGUE_ID": "12345",
        "YAHOO_REFRESH_TOKEN": "test_refresh",
        "YAHOO_ACCESS_TOKEN": "test_access",
    }):
        from backend.fantasy_baseball.yahoo_client_resilient import get_resilient_yahoo_client
        c1 = get_resilient_yahoo_client()
        c2 = get_resilient_yahoo_client()
        assert c1 is c2


def test_singleton_thread_safety():
    """Concurrent calls must produce the same instance (no double-init)."""
    import backend.fantasy_baseball.yahoo_client_resilient as mod
    mod._client = None

    results = []

    with patch.dict("os.environ", {
        "YAHOO_CLIENT_ID": "test_id",
        "YAHOO_CLIENT_SECRET": "test_secret",
        "YAHOO_LEAGUE_ID": "12345",
        "YAHOO_REFRESH_TOKEN": "test_refresh",
        "YAHOO_ACCESS_TOKEN": "test_access",
    }):
        from backend.fantasy_baseball.yahoo_client_resilient import get_yahoo_client

        def grab():
            results.append(get_yahoo_client())

        threads = [threading.Thread(target=grab) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert len(set(id(r) for r in results)) == 1
