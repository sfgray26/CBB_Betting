"""Regression tests for Yahoo auth hardening."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from scripts import migration_yahoo_oauth_tokens as token_migration
from backend.fantasy_baseball import yahoo_client_resilient as yahoo_mod
from backend.fantasy_baseball.yahoo_client_resilient import YahooAPIError, YahooFantasyClient


def _response(status_code: int, text: str = "forbidden") -> Mock:
    resp = Mock()
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = {"ok": True}
    return resp


@pytest.fixture
def yahoo_env(monkeypatch):
    monkeypatch.setenv("YAHOO_CLIENT_ID", "client-id")
    monkeypatch.setenv("YAHOO_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("YAHOO_LEAGUE_ID", "72586")
    monkeypatch.setenv("YAHOO_ACCESS_TOKEN", "env-access")
    monkeypatch.setenv("YAHOO_REFRESH_TOKEN", "env-refresh")
    monkeypatch.setattr("backend.services.yahoo_token_store.load_yahoo_tokens", lambda: None)


def test_client_prefers_db_tokens_over_stale_env(monkeypatch, yahoo_env):
    expires_at = datetime.now(ZoneInfo("America/New_York")) + timedelta(minutes=30)
    monkeypatch.setattr(
        "backend.services.yahoo_token_store.load_yahoo_tokens",
        lambda: {
            "access_token": "db-access",
            "refresh_token": "db-refresh",
            "expires_at": expires_at,
            "token_type": "bearer",
        },
    )

    client = YahooFantasyClient()

    assert client._access_token == "db-access"
    assert client._refresh_token == "db-refresh"
    assert client._token_expiry == pytest.approx(expires_at.timestamp())


def test_store_tokens_persists_to_db_without_logging_values(monkeypatch, yahoo_env, caplog):
    persisted = {}

    def fake_persist(**kwargs):
        persisted.update(kwargs)
        return True

    monkeypatch.setattr("backend.services.yahoo_token_store.persist_yahoo_tokens", fake_persist)
    monkeypatch.setattr(yahoo_mod, "set_key", Mock())
    client = YahooFantasyClient()

    client._store_tokens(
        {
            "access_token": "new-access-secret",
            "refresh_token": "new-refresh-secret",
            "expires_in": 3600,
            "token_type": "bearer",
        }
    )

    assert persisted["access_token"] == "new-access-secret"
    assert persisted["refresh_token"] == "new-refresh-secret"
    assert "new-access-secret" not in caplog.text
    assert "new-refresh-secret" not in caplog.text


def test_403_recovery_refresh_is_bounded(monkeypatch, yahoo_env):
    client = YahooFantasyClient()
    client._token_expiry = time.time() + 3600
    client._session.get = Mock(side_effect=[_response(403), _response(403), _response(403)])
    client._refresh_access_token = Mock()

    with pytest.raises(YahooAPIError):
        client._get("league/469.l.72586")

    with pytest.raises(YahooAPIError):
        client._get("league/469.l.72586")

    assert client._refresh_access_token.call_count == 1
    assert client._auth_failure_count == 2


def test_repeated_403_opens_auth_circuit_and_alerts_once(monkeypatch, yahoo_env):
    monkeypatch.setattr(yahoo_mod, "YAHOO_AUTH_FAILURE_THRESHOLD", 2)
    monkeypatch.setattr(yahoo_mod, "YAHOO_AUTH_CIRCUIT_OPEN_SECONDS", 900)
    client = YahooFantasyClient()
    client._token_expiry = time.time() + 3600
    client._session.get = Mock(side_effect=[_response(403), _response(403), _response(403)])
    client._refresh_access_token = Mock()
    client._emit_yahoo_auth_outage_alert = Mock()

    with pytest.raises(YahooAPIError):
        client._get("league/469.l.72586")
    with pytest.raises(YahooAPIError):
        client._get("league/469.l.72586")

    assert client._auth_circuit_open_until is not None
    client._emit_yahoo_auth_outage_alert.assert_called_once_with(status_code=403)

    calls_before = client._session.get.call_count
    with pytest.raises(YahooAPIError) as exc:
        client._get("league/469.l.72586")
    assert exc.value.status_code == 503
    assert client._session.get.call_count == calls_before


def test_refresh_failure_before_request_opens_auth_circuit(monkeypatch, yahoo_env):
    monkeypatch.setattr(yahoo_mod, "YAHOO_AUTH_FAILURE_THRESHOLD", 2)
    client = YahooFantasyClient()
    client._token_expiry = 0
    client._refresh_access_token = Mock(side_effect=yahoo_mod.YahooAuthError("refresh failed"))
    client._emit_yahoo_auth_outage_alert = Mock()

    with pytest.raises(yahoo_mod.YahooAuthError):
        client._get("league/469.l.72586")
    with pytest.raises(yahoo_mod.YahooAuthError):
        client._get("league/469.l.72586")

    assert client._auth_circuit_open_until is not None
    client._emit_yahoo_auth_outage_alert.assert_called_once_with(status_code=401)

    with pytest.raises(YahooAPIError) as exc:
        client._get("league/469.l.72586")
    assert exc.value.status_code == 503
    assert client._refresh_access_token.call_count == 2


def test_yahoo_token_migration_seeds_from_env_without_printing_tokens(monkeypatch, capsys):
    executed = []

    class FakeConnection:
        def execute(self, stmt, params=None):
            executed.append((str(stmt), params))

    class FakeBegin:
        def __enter__(self):
            return FakeConnection()

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeEngine:
        def begin(self):
            return FakeBegin()

    monkeypatch.setenv("DATABASE_URL", "postgresql://example/db")
    monkeypatch.setenv("YAHOO_ACCESS_TOKEN", "seed-access-secret")
    monkeypatch.setenv("YAHOO_REFRESH_TOKEN", "seed-refresh-secret")
    monkeypatch.setattr(token_migration, "create_engine", lambda url: FakeEngine())

    token_migration.main()

    assert len(executed) == 2
    assert "CREATE TABLE IF NOT EXISTS yahoo_oauth_tokens" in executed[0][0]
    assert "ON CONFLICT (provider) DO UPDATE" in executed[1][0]
    params = executed[1][1]
    assert params["provider"] == "yahoo_fantasy"
    assert params["access_token"] == "seed-access-secret"
    assert params["refresh_token"] == "seed-refresh-secret"
    assert params["expires_at"] > params["now"]

    out = capsys.readouterr().out
    assert "upserted" in out
    assert "seed-access-secret" not in out
    assert "seed-refresh-secret" not in out


def test_yahoo_token_migration_skips_seed_without_both_tokens(monkeypatch, capsys):
    executed = []

    class FakeConnection:
        def execute(self, stmt, params=None):
            executed.append((str(stmt), params))

    class FakeBegin:
        def __enter__(self):
            return FakeConnection()

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeEngine:
        def begin(self):
            return FakeBegin()

    monkeypatch.setenv("DATABASE_URL", "postgresql://example/db")
    monkeypatch.delenv("YAHOO_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("YAHOO_REFRESH_TOKEN", "seed-refresh-secret")
    monkeypatch.setattr(token_migration, "create_engine", lambda url: FakeEngine())

    token_migration.main()

    assert len(executed) == 1
    out = capsys.readouterr().out
    assert "seed skipped" in out
    assert "seed-refresh-secret" not in out


@pytest.mark.asyncio
async def test_yahoo_health_initializes_canonical_client(monkeypatch):
    from backend.routers import fantasy

    fake_client = SimpleNamespace(
        _cb=SimpleNamespace(state="closed"),
        _auth_circuit_open_until=None,
        get_league=Mock(return_value={"league": "ok"}),
    )
    monkeypatch.setattr(fantasy, "get_yahoo_client", lambda: fake_client)

    result = await fantasy.yahoo_health()

    assert result["status"] == "healthy"
    assert result["error"] is None
    fake_client.get_league.assert_called_once()
