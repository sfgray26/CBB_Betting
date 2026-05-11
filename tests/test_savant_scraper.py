"""
Tests for backend/ingestion/savant_scraper.py.

All HTTP is mocked — no real network calls.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from backend.ingestion.savant_scraper import fetch_sprint_speed, _empty_df, _parse_sprint_speed_csv as _parse_csv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_CSV = (
    "player_id,player_name,sprint_speed,team,year\n"
    "660271,Acuna Jr.,30.2,ATL,2026\n"
    "592450,Trout M.,28.5,LAA,2026\n"
    "665742,Judge A.,27.1,NYY,2026\n"
)

_MINIMAL_CSV = (
    "player_id,sprint_speed\n"
    "660271,30.2\n"
    "592450,28.5\n"
)

_MALFORMED_CSV = "not,valid,csv,data\nhello,world,foo,bar\n"

_EMPTY_CSV = "player_id,player_name,sprint_speed\n"


def _mock_response(text: str, status_code: int = 200):
    resp = MagicMock()
    resp.text = text
    resp.status_code = status_code
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(
            response=resp, request=MagicMock()
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# fetch_sprint_speed — happy path
# ---------------------------------------------------------------------------

def test_returns_dataframe_with_correct_columns():
    with patch("backend.ingestion.savant_scraper.requests.get", return_value=_mock_response(_VALID_CSV)):
        df = fetch_sprint_speed(year=2026)

    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"mlbam_id", "player_name", "sprint_speed"}


def test_returns_correct_row_count():
    with patch("backend.ingestion.savant_scraper.requests.get", return_value=_mock_response(_VALID_CSV)):
        df = fetch_sprint_speed(year=2026)
    assert len(df) == 3


def test_mlbam_ids_are_integers():
    with patch("backend.ingestion.savant_scraper.requests.get", return_value=_mock_response(_VALID_CSV)):
        df = fetch_sprint_speed(year=2026)
    assert df["mlbam_id"].dtype == int or df["mlbam_id"].dtype.kind == "i"


def test_sprint_speed_values_are_floats():
    with patch("backend.ingestion.savant_scraper.requests.get", return_value=_mock_response(_VALID_CSV)):
        df = fetch_sprint_speed(year=2026)
    assert df["sprint_speed"].dtype.kind == "f"


def test_specific_values_parsed_correctly():
    with patch("backend.ingestion.savant_scraper.requests.get", return_value=_mock_response(_VALID_CSV)):
        df = fetch_sprint_speed(year=2026)
    acuna_row = df[df["mlbam_id"] == 660271].iloc[0]
    assert acuna_row["sprint_speed"] == pytest.approx(30.2)


def test_works_with_minimal_csv_no_name_column():
    with patch("backend.ingestion.savant_scraper.requests.get", return_value=_mock_response(_MINIMAL_CSV)):
        df = fetch_sprint_speed(year=2026)
    assert len(df) == 2
    assert "sprint_speed" in df.columns


def test_correct_url_called_for_year():
    with patch("backend.ingestion.savant_scraper.requests.get", return_value=_mock_response(_VALID_CSV)) as mock_get:
        fetch_sprint_speed(year=2025)
    called_url = mock_get.call_args[0][0]
    assert "year=2025" in called_url


# ---------------------------------------------------------------------------
# fetch_sprint_speed — failure paths
# ---------------------------------------------------------------------------

def test_returns_empty_dataframe_on_http_error():
    with patch("backend.ingestion.savant_scraper.requests.get", return_value=_mock_response("", status_code=503)):
        df = fetch_sprint_speed(year=2026)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert set(df.columns) == {"mlbam_id", "player_name", "sprint_speed"}


def test_returns_empty_dataframe_on_connection_error():
    with patch("backend.ingestion.savant_scraper.requests.get", side_effect=requests.ConnectionError("timeout")):
        df = fetch_sprint_speed(year=2026)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_returns_empty_dataframe_on_timeout():
    with patch("backend.ingestion.savant_scraper.requests.get", side_effect=requests.Timeout("timed out")):
        df = fetch_sprint_speed(year=2026)

    assert len(df) == 0


def test_returns_empty_dataframe_on_malformed_csv():
    """CSV that parses but has no usable columns."""
    with patch("backend.ingestion.savant_scraper.requests.get", return_value=_mock_response(_MALFORMED_CSV)):
        df = fetch_sprint_speed(year=2026)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_returns_empty_dataframe_on_empty_csv_body():
    with patch("backend.ingestion.savant_scraper.requests.get", return_value=_mock_response(_EMPTY_CSV)):
        df = fetch_sprint_speed(year=2026)

    assert len(df) == 0


def test_does_not_raise_on_any_failure():
    """Verifies failure contract: never raises, always returns DataFrame."""
    with patch("backend.ingestion.savant_scraper.requests.get", side_effect=Exception("unexpected")):
        result = fetch_sprint_speed(year=2026)

    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# _parse_csv — unit tests
# ---------------------------------------------------------------------------

def test_parse_csv_drops_rows_with_null_speed():
    csv = "player_id,sprint_speed\n123,30.0\n456,\n789,28.5\n"
    df = _parse_csv(csv)
    assert len(df) == 2
    assert 456 not in df["mlbam_id"].values


def test_parse_csv_drops_rows_with_null_id():
    csv = "player_id,sprint_speed\n123,30.0\n,28.5\n789,27.0\n"
    df = _parse_csv(csv)
    assert len(df) == 2


def test_empty_df_has_correct_schema():
    df = _empty_df(columns=["mlbam_id", "player_name", "sprint_speed"])
    assert list(df.columns) == ["mlbam_id", "player_name", "sprint_speed"]
    assert len(df) == 0
