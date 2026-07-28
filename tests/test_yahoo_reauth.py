"""Tests for the production Yahoo re-auth admin endpoints.

/admin/yahoo/auth-url  -> consent URL
/admin/yahoo/reauth    -> exchange code, persist to prod DB, reset circuit, verify

The reauth endpoint must distinguish a valid fresh grant ("ok") from one where the
token exchange works but the Fantasy API still 403s (the Yahoo app grant lacks
Fantasy authorization) — so the operator knows to fix the Yahoo developer app.
"""
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.auth import verify_admin_api_key


def _client():
    app.dependency_overrides[verify_admin_api_key] = lambda: "admin"
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def test_auth_url_returns_consent_url():
    mock = MagicMock()
    mock.client_id = "cid"
    mock.get_authorization_url.return_value = "https://api.login.yahoo.com/oauth2/request_auth?client_id=cid"
    with patch("backend.main.get_yahoo_client", return_value=mock):
        resp = _client().get("/admin/yahoo/auth-url")
    assert resp.status_code == 200
    assert resp.json()["auth_url"].startswith("https://api.login.yahoo.com/oauth2/request_auth")


def test_reauth_ok_when_grant_valid():
    mock = MagicMock()
    mock.get_my_team_key.return_value = "469.l.72586.t.7"
    mock.get_league.return_value = {"name": "Lindor Truffles"}
    with patch("backend.main.get_yahoo_client", return_value=mock):
        resp = _client().post("/admin/yahoo/reauth?code=ABC123")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["verified"] is True
    assert data["my_team_key"] == "469.l.72586.t.7"
    # The circuit must be reset and the code exchanged.
    mock.reset_auth_circuit.assert_called_once()
    mock.exchange_code_for_tokens.assert_called_once_with("ABC123")


def test_reauth_reports_unauthorized_grant():
    """Exchange succeeds but the Fantasy call 403s -> grant lacks Fantasy auth."""
    mock = MagicMock()
    mock.get_my_team_key.side_effect = Exception("403: not authorized")
    with patch("backend.main.get_yahoo_client", return_value=mock):
        resp = _client().post("/admin/yahoo/reauth?code=ABC123")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "tokens_stored_but_unauthorized"
    assert data["verified"] is False
    assert "Fantasy Sports" in data["message"]


def test_reauth_exchange_failure_is_400():
    mock = MagicMock()
    mock.exchange_code_for_tokens.side_effect = Exception("Token exchange failed: 400")
    with patch("backend.main.get_yahoo_client", return_value=mock):
        resp = _client().post("/admin/yahoo/reauth?code=BAD")
    assert resp.status_code == 400
