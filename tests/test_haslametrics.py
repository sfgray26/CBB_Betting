"""
Tests for backend/services/haslametrics.py

Run:
    pytest tests/test_haslametrics.py -v

These tests exercise fetch_haslametrics_ratings() and
get_haslametrics_ratings() in isolation via mocked HTTP calls.
No database or network access is required.
"""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Sample HTML containing a minimal Haslametrics-style ratings table.
# Columns: Rk, Team, Conf, AdjOE, AdjDE, Net
# ---------------------------------------------------------------------------
_SAMPLE_HTML = """
<html><body><table>
<tr><th>Rk</th><th>Team</th><th>Conf</th><th>AdjOE</th><th>AdjDE</th><th>Net</th></tr>
<tr><td>1</td><td>Auburn</td><td>SEC</td><td>123.5</td><td>91.2</td><td>32.3</td></tr>
<tr><td>2</td><td>Duke</td><td>ACC</td><td>121.0</td><td>93.5</td><td>27.5</td></tr>
<tr><td>3</td><td>Florida</td><td>SEC</td><td>119.2</td><td>93.8</td><td>25.4</td></tr>
</table></body></html>
"""

_NO_NET_HTML = """
<html><body><table>
<tr><th>Rk</th><th>Team</th><th>Conf</th><th>AdjOE</th><th>AdjDE</th></tr>
<tr><td>1</td><td>Auburn</td><td>SEC</td><td>123.5</td><td>91.2</td></tr>
</table></body></html>
"""


def _make_response(html: str, status_code: int = 200) -> MagicMock:
    """Build a mock requests.Response-like object."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.text = html
    if status_code >= 400:
        from requests.exceptions import HTTPError
        mock_resp.raise_for_status.side_effect = HTTPError(
            f"{status_code} Error", response=mock_resp
        )
    else:
        mock_resp.raise_for_status.return_value = None
    return mock_resp


# ---------------------------------------------------------------------------
# Reset the module-level circuit breaker before each test so state
# from one test cannot bleed into the next.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def reset_circuit_breaker():
    """Reset haslametrics circuit breaker state between tests."""
    import backend.services.haslametrics as mod
    from backend.core.circuit_breaker import CircuitBreaker
    mod._haslametrics_cb = CircuitBreaker(failure_threshold=3, recovery_timeout=300)
    yield


# ===========================================================================
# TestFetchHaslametricsRatings
# ===========================================================================

class TestFetchHaslametricsRatings:
    """Tests for fetch_haslametrics_ratings()."""

    def test_returns_dict_on_success(self):
        """Mock a valid HTML response and verify a non-empty dict is returned."""
        mock_resp = _make_response(_SAMPLE_HTML)
        with patch("backend.services.haslametrics.requests.get", return_value=mock_resp):
            from backend.services.haslametrics import fetch_haslametrics_ratings
            result = fetch_haslametrics_ratings()

        assert isinstance(result, dict)
        assert len(result) == 3

    def test_returns_empty_on_http_error(self):
        """A 500 HTTP error must cause the function to return {}."""
        mock_resp = _make_response("", status_code=500)
        with patch("backend.services.haslametrics.requests.get", return_value=mock_resp):
            from backend.services.haslametrics import fetch_haslametrics_ratings
            result = fetch_haslametrics_ratings()

        assert result == {}

    def test_returns_empty_on_network_error(self):
        """A ConnectionError must cause the function to return {}."""
        with patch(
            "backend.services.haslametrics.requests.get",
            side_effect=ConnectionError("unreachable"),
        ):
            from backend.services.haslametrics import fetch_haslametrics_ratings
            result = fetch_haslametrics_ratings()

        assert result == {}

    def test_circuit_breaker_blocks_after_failures(self):
        """After failure_threshold failures the circuit opens and blocks requests."""
        with patch(
            "backend.services.haslametrics.requests.get",
            side_effect=ConnectionError("unreachable"),
        ):
            from backend.services.haslametrics import fetch_haslametrics_ratings
            # Trigger 3 failures to open the breaker.
            for _ in range(3):
                fetch_haslametrics_ratings()

        # Fourth call — circuit should be OPEN; requests.get must NOT be called.
        call_tracker = MagicMock(return_value=_make_response(_SAMPLE_HTML))
        with patch("backend.services.haslametrics.requests.get", call_tracker):
            from backend.services.haslametrics import fetch_haslametrics_ratings
            result = fetch_haslametrics_ratings()

        assert result == {}
        call_tracker.assert_not_called()

    def test_team_names_normalized(self):
        """normalize_team_name must be called once per parsed team row."""
        mock_resp = _make_response(_SAMPLE_HTML)
        with patch("backend.services.haslametrics.requests.get", return_value=mock_resp):
            with patch(
                "backend.services.haslametrics.normalize_team_name",
                wraps=lambda name, choices: name,  # identity — return raw name
            ) as mock_normalize:
                from backend.services.haslametrics import fetch_haslametrics_ratings
                fetch_haslametrics_ratings()

        # One call per team row (3 teams in sample HTML).
        assert mock_normalize.call_count == 3

    def test_values_are_floats(self):
        """Every value in the returned dict must be a Python float."""
        mock_resp = _make_response(_SAMPLE_HTML)
        with patch("backend.services.haslametrics.requests.get", return_value=mock_resp):
            from backend.services.haslametrics import fetch_haslametrics_ratings
            result = fetch_haslametrics_ratings()

        assert all(isinstance(v, float) for v in result.values()), (
            "All rating values must be float; got: "
            + str({k: type(v) for k, v in result.items()})
        )

    def test_correct_net_values_extracted(self):
        """The Net column values must match the sample HTML exactly."""
        mock_resp = _make_response(_SAMPLE_HTML)
        with patch("backend.services.haslametrics.requests.get", return_value=mock_resp):
            with patch(
                "backend.services.haslametrics.normalize_team_name",
                side_effect=lambda name, choices: name,
            ):
                from backend.services.haslametrics import fetch_haslametrics_ratings
                result = fetch_haslametrics_ratings()

        assert result.get("Auburn") == pytest.approx(32.3)
        assert result.get("Duke") == pytest.approx(27.5)
        assert result.get("Florida") == pytest.approx(25.4)

    def test_returns_empty_when_no_net_column(self):
        """HTML table without a Net/Margin/AdjEM column must return {}."""
        mock_resp = _make_response(_NO_NET_HTML)
        with patch("backend.services.haslametrics.requests.get", return_value=mock_resp):
            from backend.services.haslametrics import fetch_haslametrics_ratings
            result = fetch_haslametrics_ratings()

        assert result == {}

    def test_returns_empty_on_empty_html(self):
        """HTML with no tables at all must return {}."""
        mock_resp = _make_response("<html><body><p>No data</p></body></html>")
        with patch("backend.services.haslametrics.requests.get", return_value=mock_resp):
            from backend.services.haslametrics import fetch_haslametrics_ratings
            result = fetch_haslametrics_ratings()

        assert result == {}

    def test_year_argument_accepted(self):
        """fetch_haslametrics_ratings(year=2026) must not raise."""
        mock_resp = _make_response(_SAMPLE_HTML)
        with patch("backend.services.haslametrics.requests.get", return_value=mock_resp):
            from backend.services.haslametrics import fetch_haslametrics_ratings
            result = fetch_haslametrics_ratings(year=2026)

        assert isinstance(result, dict)


# ===========================================================================
# TestGetHaslametricsRatings
# ===========================================================================

class TestGetHaslametricsRatings:
    """Tests for the public wrapper get_haslametrics_ratings()."""

    def test_returns_same_as_fetch(self):
        """get_haslametrics_ratings() must delegate to fetch and return same dict."""
        expected = {"Auburn": 32.3, "Duke": 27.5, "Florida": 25.4}

        with patch(
            "backend.services.haslametrics.fetch_haslametrics_ratings",
            return_value=expected,
        ) as mock_fetch:
            from backend.services.haslametrics import get_haslametrics_ratings
            result = get_haslametrics_ratings()

        mock_fetch.assert_called_once_with()
        assert result == expected

    def test_returns_dict_type(self):
        """Return type must always be dict even on failure path."""
        mock_resp = _make_response("", status_code=500)
        with patch("backend.services.haslametrics.requests.get", return_value=mock_resp):
            from backend.services.haslametrics import get_haslametrics_ratings
            result = get_haslametrics_ratings()

        assert isinstance(result, dict)
