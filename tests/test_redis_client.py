import pytest
from unittest.mock import MagicMock, patch


def test_namespaced_cache_get_prefixes_key():
    from backend.redis_client import NamespacedCache
    mock_redis = MagicMock()
    mock_redis.get.return_value = "value"
    cache = NamespacedCache("fantasy", client=mock_redis)
    result = cache.get("ros:today")
    mock_redis.get.assert_called_once_with("fantasy:ros:today")
    assert result == "value"


def test_namespaced_cache_set_prefixes_key():
    from backend.redis_client import NamespacedCache
    mock_redis = MagicMock()
    cache = NamespacedCache("edge", client=mock_redis)
    cache.set("token", "abc123", ex=3600)
    mock_redis.set.assert_called_once_with("edge:token", "abc123", ex=3600)


def test_namespaced_cache_delete_prefixes_key():
    from backend.redis_client import NamespacedCache
    mock_redis = MagicMock()
    cache = NamespacedCache("fantasy", client=mock_redis)
    cache.delete("old-key")
    mock_redis.delete.assert_called_once_with("fantasy:old-key")


def test_edge_and_fantasy_caches_use_different_prefixes():
    from backend.redis_client import NamespacedCache
    mock_redis = MagicMock()
    edge = NamespacedCache("edge", client=mock_redis)
    fantasy = NamespacedCache("fantasy", client=mock_redis)
    edge.get("token")
    fantasy.get("token")
    calls = [str(c) for c in mock_redis.get.call_args_list]
    assert any("edge:token" in c for c in calls)
    assert any("fantasy:token" in c for c in calls)


def test_get_redis_raises_if_no_url(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    # Re-import to reset module-level state
    import importlib
    import backend.redis_client as rc
    importlib.reload(rc)
    with pytest.raises(RuntimeError, match="REDIS_URL"):
        rc.get_redis()
