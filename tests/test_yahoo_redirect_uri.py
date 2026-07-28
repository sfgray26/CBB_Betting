"""Regression tests for YAHOO_REDIRECT_URI support (2026-07 auth saga).

The consent-code exchange must send a redirect_uri that EXACTLY matches a
Redirect URI registered on the Yahoo developer app. The previously hardcoded
"oob" was NOT registered on the production app; grants minted through it
returned tokens Yahoo rejected on every Fantasy API call ("This application
is not authorized to perform this action"). The client now reads
YAHOO_REDIRECT_URI, defaulting to "oob" for backward compatibility.

No token values appear in these tests.
"""
import os
from unittest.mock import MagicMock, patch

from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient

_BASE_ENV = {
    "YAHOO_CLIENT_ID": "test-cid",
    "YAHOO_CLIENT_SECRET": "test-secret",
    "YAHOO_REFRESH_TOKEN": "test-refresh-token",
}
_REGISTERED = "https://localhost:8000/callback"


def _make_client(extra_env=None, unset=()):
    env = dict(_BASE_ENV)
    if extra_env:
        env.update(extra_env)
    with patch.dict(os.environ, env, clear=False):
        for key in unset:
            os.environ.pop(key, None)
        return YahooFantasyClient()


def test_redirect_uri_defaults_to_oob():
    """Without YAHOO_REDIRECT_URI, legacy 'oob' behavior is preserved."""
    client = _make_client(unset=("YAHOO_REDIRECT_URI",))
    assert client.redirect_uri == "oob"
    assert "redirect_uri=oob" in client.get_authorization_url()


def test_redirect_uri_env_override_in_auth_url():
    """YAHOO_REDIRECT_URI must appear (urlencoded) in the consent URL."""
    client = _make_client({"YAHOO_REDIRECT_URI": _REGISTERED})
    assert client.redirect_uri == _REGISTERED
    assert "redirect_uri=https%3A%2F%2Flocalhost%3A8000%2Fcallback" in (
        client.get_authorization_url()
    )


def test_exchange_posts_matching_redirect_uri():
    """exchange_code_for_tokens must POST the SAME redirect_uri as the auth URL —
    a mismatch is what produced scope-less grants that 403 every Fantasy call."""
    client = _make_client({"YAHOO_REDIRECT_URI": _REGISTERED})
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "access_token": "redacted",
        "refresh_token": "redacted",
        "expires_in": 3600,
    }
    with patch(
        "backend.fantasy_baseball.yahoo_client_resilient.requests.post",
        return_value=mock_resp,
    ) as mock_post, patch.object(YahooFantasyClient, "_store_tokens"):
        client.exchange_code_for_tokens("dummy-code")
    _, kwargs = mock_post.call_args
    assert kwargs["data"]["redirect_uri"] == _REGISTERED
    assert kwargs["data"]["grant_type"] == "authorization_code"
