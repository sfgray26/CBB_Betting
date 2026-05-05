"""
Tests for backend/ingestion/fangraphs_scraper.py

Covers fetch_pitcher_quality() including happy path, empty returns, exception
handling, partial ID mapping, null metric values, and _empty_df column contract.

pybaseball functions are imported inside the function body with
    from pybaseball import fg_pitching_data, playerid_reverse_lookup
so they are patched at the pybaseball module level.
"""
import pytest
import pandas as pd
from unittest.mock import patch

from backend.ingestion.fangraphs_scraper import fetch_pitcher_quality, _empty_df, _COLS


# ---------------------------------------------------------------------------
# Shared mock data
# ---------------------------------------------------------------------------

_FG_DF = pd.DataFrame({
    "IDfg": [1234, 5678],
    "Name": ["Max Scherzer", "Clayton Kershaw"],
    "Stuff+": [112.0, 108.0],
    "Location+": [108.0, 105.0],
    "Pitching+": [110.0, 106.0],
})

_ID_DF = pd.DataFrame({
    "key_fangraphs": [1234, 5678],
    "key_mlbam": [456789, 123456],
})


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestFetchPitcherQualityHappyPath:
    """Verify correct output shape, types, and values when both mocks succeed."""

    def test_returns_correct_columns(self):
        with patch("pybaseball.fg_pitching_data", return_value=_FG_DF), \
             patch("pybaseball.playerid_reverse_lookup", return_value=_ID_DF):
            df = fetch_pitcher_quality(season=2026)

        assert list(df.columns) == _COLS

    def test_row_count_matches_mapped_pitchers(self):
        with patch("pybaseball.fg_pitching_data", return_value=_FG_DF), \
             patch("pybaseball.playerid_reverse_lookup", return_value=_ID_DF):
            df = fetch_pitcher_quality(season=2026)

        assert len(df) == 2

    def test_mlbam_id_is_string(self):
        with patch("pybaseball.fg_pitching_data", return_value=_FG_DF), \
             patch("pybaseball.playerid_reverse_lookup", return_value=_ID_DF):
            df = fetch_pitcher_quality(season=2026)

        assert df.iloc[0]["mlbam_id"] == "456789"
        assert isinstance(df.iloc[0]["mlbam_id"], str)

    def test_stuff_plus_value(self):
        with patch("pybaseball.fg_pitching_data", return_value=_FG_DF), \
             patch("pybaseball.playerid_reverse_lookup", return_value=_ID_DF):
            df = fetch_pitcher_quality(season=2026)

        assert df.iloc[0]["stuff_plus"] == 112.0

    def test_location_plus_value(self):
        with patch("pybaseball.fg_pitching_data", return_value=_FG_DF), \
             patch("pybaseball.playerid_reverse_lookup", return_value=_ID_DF):
            df = fetch_pitcher_quality(season=2026)

        assert df.iloc[0]["location_plus"] == 108.0

    def test_pitching_plus_value(self):
        with patch("pybaseball.fg_pitching_data", return_value=_FG_DF), \
             patch("pybaseball.playerid_reverse_lookup", return_value=_ID_DF):
            df = fetch_pitcher_quality(season=2026)

        assert df.iloc[0]["pitching_plus"] == 110.0

    def test_second_row_mlbam_id(self):
        with patch("pybaseball.fg_pitching_data", return_value=_FG_DF), \
             patch("pybaseball.playerid_reverse_lookup", return_value=_ID_DF):
            df = fetch_pitcher_quality(season=2026)

        assert df.iloc[1]["mlbam_id"] == "123456"


# ---------------------------------------------------------------------------
# Empty / missing data paths
# ---------------------------------------------------------------------------

class TestFetchPitcherQualityEmptyReturns:

    def test_fg_pitching_data_returns_empty_df(self):
        empty = pd.DataFrame(columns=_FG_DF.columns)
        with patch("pybaseball.fg_pitching_data", return_value=empty), \
             patch("pybaseball.playerid_reverse_lookup", return_value=_ID_DF):
            df = fetch_pitcher_quality(season=2026)

        assert df.empty
        assert list(df.columns) == _COLS

    def test_fg_pitching_data_returns_none(self):
        with patch("pybaseball.fg_pitching_data", return_value=None), \
             patch("pybaseball.playerid_reverse_lookup", return_value=_ID_DF):
            df = fetch_pitcher_quality(season=2026)

        assert df.empty
        assert list(df.columns) == _COLS

    def test_playerid_reverse_lookup_returns_empty_df(self):
        empty_id = pd.DataFrame(columns=["key_fangraphs", "key_mlbam"])
        with patch("pybaseball.fg_pitching_data", return_value=_FG_DF), \
             patch("pybaseball.playerid_reverse_lookup", return_value=empty_id):
            df = fetch_pitcher_quality(season=2026)

        assert df.empty
        assert list(df.columns) == _COLS

    def test_playerid_reverse_lookup_returns_none(self):
        with patch("pybaseball.fg_pitching_data", return_value=_FG_DF), \
             patch("pybaseball.playerid_reverse_lookup", return_value=None):
            df = fetch_pitcher_quality(season=2026)

        assert df.empty
        assert list(df.columns) == _COLS


# ---------------------------------------------------------------------------
# Exception handling
# ---------------------------------------------------------------------------

class TestFetchPitcherQualityExceptions:

    def test_exception_in_fg_pitching_data_returns_empty(self):
        with patch("pybaseball.fg_pitching_data",
                   side_effect=RuntimeError("network timeout")), \
             patch("pybaseball.playerid_reverse_lookup", return_value=_ID_DF):
            df = fetch_pitcher_quality(season=2026)

        assert df.empty
        assert list(df.columns) == _COLS

    def test_exception_in_playerid_reverse_lookup_returns_empty(self):
        with patch("pybaseball.fg_pitching_data", return_value=_FG_DF), \
             patch("pybaseball.playerid_reverse_lookup",
                   side_effect=ValueError("lookup table unavailable")):
            df = fetch_pitcher_quality(season=2026)

        assert df.empty
        assert list(df.columns) == _COLS


# ---------------------------------------------------------------------------
# Partial ID mapping
# ---------------------------------------------------------------------------

class TestPartialIdMapping:

    def test_only_matched_fg_ids_returned(self):
        """When only one of two fg_ids maps to an mlbam_id, only one row is returned."""
        partial_id_df = pd.DataFrame({
            "key_fangraphs": [1234],
            "key_mlbam": [456789],
        })
        with patch("pybaseball.fg_pitching_data", return_value=_FG_DF), \
             patch("pybaseball.playerid_reverse_lookup", return_value=partial_id_df):
            df = fetch_pitcher_quality(season=2026)

        assert len(df) == 1
        assert df.iloc[0]["mlbam_id"] == "456789"
        assert df.iloc[0]["stuff_plus"] == 112.0

    def test_zero_matched_ids_returns_empty(self):
        """When no fg_ids map (e.g. Chadwick lookup returned wrong IDs), return empty."""
        wrong_id_df = pd.DataFrame({
            "key_fangraphs": [9999],
            "key_mlbam": [111111],
        })
        with patch("pybaseball.fg_pitching_data", return_value=_FG_DF), \
             patch("pybaseball.playerid_reverse_lookup", return_value=wrong_id_df):
            df = fetch_pitcher_quality(season=2026)

        assert df.empty
        assert list(df.columns) == _COLS


# ---------------------------------------------------------------------------
# Null metric values
# ---------------------------------------------------------------------------

class TestNullMetricValues:

    def test_null_stuff_plus_preserved(self):
        fg_with_nulls = pd.DataFrame({
            "IDfg": [1234],
            "Name": ["Max Scherzer"],
            "Stuff+": [None],
            "Location+": [108.0],
            "Pitching+": [110.0],
        })
        with patch("pybaseball.fg_pitching_data", return_value=fg_with_nulls), \
             patch("pybaseball.playerid_reverse_lookup", return_value=_ID_DF):
            df = fetch_pitcher_quality(season=2026)

        assert len(df) == 1
        assert df.iloc[0]["stuff_plus"] is None

    def test_null_location_plus_preserved(self):
        fg_with_nulls = pd.DataFrame({
            "IDfg": [1234],
            "Name": ["Max Scherzer"],
            "Stuff+": [112.0],
            "Location+": [None],
            "Pitching+": [110.0],
        })
        with patch("pybaseball.fg_pitching_data", return_value=fg_with_nulls), \
             patch("pybaseball.playerid_reverse_lookup", return_value=_ID_DF):
            df = fetch_pitcher_quality(season=2026)

        assert len(df) == 1
        assert df.iloc[0]["location_plus"] is None

    def test_all_metrics_null_row_still_included(self):
        """A pitcher with all-null metrics is still included -- the row itself is valid."""
        fg_all_null = pd.DataFrame({
            "IDfg": [1234],
            "Name": ["Max Scherzer"],
            "Stuff+": [None],
            "Location+": [None],
            "Pitching+": [None],
        })
        with patch("pybaseball.fg_pitching_data", return_value=fg_all_null), \
             patch("pybaseball.playerid_reverse_lookup", return_value=_ID_DF):
            df = fetch_pitcher_quality(season=2026)

        assert len(df) == 1
        assert df.iloc[0]["mlbam_id"] == "456789"


# ---------------------------------------------------------------------------
# _empty_df contract
# ---------------------------------------------------------------------------

class TestEmptyDf:

    def test_empty_df_has_correct_columns(self):
        df = _empty_df()
        assert list(df.columns) == _COLS

    def test_empty_df_has_zero_rows(self):
        df = _empty_df()
        assert len(df) == 0

    def test_empty_df_columns_match_expected(self):
        df = _empty_df()
        assert list(df.columns) == [
            "mlbam_id", "player_name", "stuff_plus", "location_plus", "pitching_plus"
        ]
