"""
PR 2.x Tests for Savant scraper advanced metrics (stuff_plus, location_plus).

Tests fetch_pitcher_advanced() function and pitcher advanced backfill.
"""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from backend.ingestion.savant_scraper import (
    fetch_pitcher_advanced,
    _parse_pitching_csv,
    _empty_df,
)


class TestFetchPitcherAdvanced:
    """Test fetch_pitcher_advanced function."""

    def test_returns_dataframe_with_correct_columns(self):
        """Should return DataFrame with mlbam_id, player_name, stuff_plus, location_plus."""
        with patch("backend.ingestion.savant_scraper.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = (
                "player_id,player_name,stuff_plus,location_plus\n"
                "12345,Max Scherzer,112,108\n"
                "6789,Clayton Kershaw,108,105\n"
            )
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            df = fetch_pitcher_advanced(year=2026)

            assert list(df.columns) == ["mlbam_id", "player_name", "stuff_plus", "location_plus"]
            assert len(df) == 2
            assert df.iloc[0]["mlbam_id"] == 12345
            assert df.iloc[0]["stuff_plus"] == 112.0
            assert df.iloc[0]["location_plus"] == 108.0

    def test_returns_empty_dataframe_on_http_error(self):
        """Should return empty DataFrame with correct columns on HTTP error."""
        with patch("backend.ingestion.savant_scraper.requests.get") as mock_get:
            mock_get.side_effect = Exception("Network error")

            df = fetch_pitcher_advanced(year=2026)

            assert list(df.columns) == ["mlbam_id", "player_name", "stuff_plus", "location_plus"]
            assert len(df) == 0

    def test_handles_missing_columns_gracefully(self):
        """Should handle missing Stuff+/Location+ columns by setting them to NaN."""
        with patch("backend.ingestion.savant_scraper.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = (
                "player_id,player_name,team\n"
                "12345,Max Scherzer,WAS\n"
                "6789,Clayton Kershaw,LAD\n"
            )
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            df = fetch_pitcher_advanced(year=2026)

            assert len(df) == 2
            assert pd.isna(df.iloc[0]["stuff_plus"])
            assert pd.isna(df.iloc[0]["location_plus"])

    def test_handles_malformed_csv_with_skip(self):
        """Should skip rows with extra fields; valid rows still parsed."""
        with patch("backend.ingestion.savant_scraper.requests.get") as mock_get:
            mock_response = MagicMock()
            # Row 2 has 6 fields (one too many) → skipped; rows 1 and 3 are valid 4-field rows
            mock_response.text = (
                "player_id,player_name,stuff_plus,location_plus\n"
                "12345,Max Scherzer,112,108\n"
                "6789,Kershaw,extra,field,oops\n"
                "99999,Other Player,100,100\n"
            )
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            df = fetch_pitcher_advanced(year=2026)

            assert len(df) == 2
            assert df.iloc[0]["mlbam_id"] == 12345
            assert df.iloc[1]["mlbam_id"] == 99999


class TestParsePitchingCsv:
    """Test _parse_pitching_csv helper function."""

    def test_parse_valid_csv(self):
        """Should parse valid CSV with all columns."""
        csv_text = (
            "player_id,player_name,stuff_plus,location_plus\n"
            "12345,Max Scherzer,112,108\n"
            "6789,Clayton Kershaw,108,105\n"
        )

        df = _parse_pitching_csv(csv_text)

        assert len(df) == 2
        assert df.iloc[0]["mlbam_id"] == 12345
        assert df.iloc[0]["stuff_plus"] == 112.0
        assert df.iloc[0]["location_plus"] == 108.0

    def test_empty_csv_returns_empty_dataframe(self):
        """Should return empty DataFrame for header-only CSV."""
        csv_text = "player_id,player_name,stuff_plus,location_plus\n"

        df = _parse_pitching_csv(csv_text)

        assert len(df) == 0
        assert list(df.columns) == ["mlbam_id", "player_name", "stuff_plus", "location_plus"]

    def test_flexible_column_names(self):
        """Should find Stuff+ and Location+ with various capitalizations."""
        # Baseball Savant uses "Stuff+" and "Location+" in some exports
        csv_text = (
            "player_id,player_name,Stuff+,Location+\n"
            "12345,Max Scherzer,112,108\n"
            "6789,Clayton Kershaw,108,105\n"
        )

        df = _parse_pitching_csv(csv_text)

        assert len(df) == 2
        assert df.iloc[0]["stuff_plus"] == 112.0
        assert df.iloc[0]["location_plus"] == 108.0

    def test_null_values_preserved(self):
        """Should preserve NULL values for missing metrics."""
        csv_text = (
            "player_id,player_name,stuff_plus,location_plus\n"
            "12345,Max Scherzer,,\n"
            "6789,Clayton Kershaw,108,\n"
        )

        df = _parse_pitching_csv(csv_text)

        assert len(df) == 2
        assert pd.isna(df.iloc[0]["stuff_plus"])
        assert pd.isna(df.iloc[0]["location_plus"])
        assert df.iloc[1]["stuff_plus"] == 108.0
        assert pd.isna(df.iloc[1]["location_plus"])


class TestEmptyDf:
    """Test _empty_df helper function."""

    def test_returns_empty_dataframe_with_specified_columns(self):
        """Should return empty DataFrame with specified columns."""
        df = _empty_df(columns=["col1", "col2", "col3"])

        assert len(df) == 0
        assert list(df.columns) == ["col1", "col2", "col3"]


class TestIntegration:
    """Integration tests for pitcher advanced metrics pipeline."""

    def test_fetch_and_parse_workflow(self):
        """End-to-end: fetch and parse work together; extra columns are tolerated."""
        with patch("backend.ingestion.savant_scraper.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = (
                "player_id,player_name,team,age,stuff_plus,location_plus\n"
                "12345,Max Scherzer,WAS,39,112,108\n"
                "6789,Clayton Kershaw,LAD,35,108,105\n"
            )
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            df = fetch_pitcher_advanced(year=2026)

            assert len(df) == 2
            assert df.iloc[0]["mlbam_id"] == 12345
            assert df.iloc[0]["stuff_plus"] == 112.0
            assert df.iloc[1]["location_plus"] == 105.0

    def test_idempotent_empty_handling(self):
        """Multiple calls with empty data should all return empty DataFrames."""
        with patch("backend.ingestion.savant_scraper.requests.get") as mock_get:
            mock_get.side_effect = Exception("No data")

            df1 = fetch_pitcher_advanced(year=2026)
            df2 = fetch_pitcher_advanced(year=2026)

            assert len(df1) == 0
            assert len(df2) == 0
            assert list(df1.columns) == list(df2.columns)
