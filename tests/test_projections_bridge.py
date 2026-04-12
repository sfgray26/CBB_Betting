"""Tests for the FanGraphs RoS -> Steamer CSV bridge function."""

import pandas as pd
import pytest
from pathlib import Path

from backend.fantasy_baseball.projections_loader import (
    export_ros_to_steamer_csvs,
    load_steamer_batting,
    load_steamer_pitching,
)


def _make_bat_df(rows=None):
    """Build a minimal FanGraphs-style batting DataFrame."""
    if rows is None:
        rows = [
            {"Name": "Shohei Ohtani", "Team": "LAD", "PA": 600, "HR": 40,
             "R": 100, "RBI": 95, "SB": 15, "SO": 130, "AVG": 0.285,
             "OBP": 0.370, "SLG": 0.580, "OPS": 0.950, "H": 160},
            {"Name": "Aaron Judge", "Team": "NYY", "PA": 580, "HR": 45,
             "R": 105, "RBI": 110, "SB": 5, "SO": 160, "AVG": 0.270,
             "OBP": 0.390, "SLG": 0.600, "OPS": 0.990, "H": 145},
        ]
    return pd.DataFrame(rows)


def _make_pit_df(rows=None):
    """Build a minimal FanGraphs-style pitching DataFrame."""
    if rows is None:
        rows = [
            {"Name": "Gerrit Cole", "Team": "NYY", "IP": 190, "W": 14,
             "SV": 0, "SO": 220, "ERA": 3.10, "WHIP": 1.05, "GS": 30,
             "BB": 45, "K/9": 10.4},
            {"Name": "Edwin Diaz", "Team": "NYM", "IP": 60, "W": 3,
             "SV": 35, "SO": 85, "ERA": 2.80, "WHIP": 0.95, "GS": 0,
             "BB": 18, "K/9": 12.8},
        ]
    return pd.DataFrame(rows)


class TestExportRosToSteamerCsvs:
    """Tests for export_ros_to_steamer_csvs()."""

    def test_writes_csvs_with_correct_data(self, tmp_path):
        bat_raw = {"steamer": _make_bat_df()}
        pit_raw = {"steamer": _make_pit_df()}

        result = export_ros_to_steamer_csvs(bat_raw, pit_raw, data_dir=tmp_path)

        assert result["batting_rows"] == 2
        assert result["pitching_rows"] == 2
        assert (tmp_path / "steamer_batting_2026.csv").exists()
        assert (tmp_path / "steamer_pitching_2026.csv").exists()

    def test_empty_data_writes_nothing(self, tmp_path):
        result = export_ros_to_steamer_csvs({}, {}, data_dir=tmp_path)

        assert result == {"batting_rows": 0, "pitching_rows": 0}
        assert not (tmp_path / "steamer_batting_2026.csv").exists()
        assert not (tmp_path / "steamer_pitching_2026.csv").exists()

    def test_empty_dataframe_writes_nothing(self, tmp_path):
        bat_raw = {"steamer": pd.DataFrame()}
        pit_raw = {"steamer": pd.DataFrame()}

        result = export_ros_to_steamer_csvs(bat_raw, pit_raw, data_dir=tmp_path)

        assert result["batting_rows"] == 0
        assert result["pitching_rows"] == 0

    def test_prefers_steamer_system(self, tmp_path):
        steamer_bat = _make_bat_df([
            {"Name": "Steamer Player", "Team": "LAD", "PA": 500, "HR": 30,
             "R": 80, "RBI": 85, "SB": 10, "SO": 100, "AVG": 0.270,
             "OBP": 0.340, "SLG": 0.500, "OPS": 0.840, "H": 130},
        ])
        atc_bat = _make_bat_df([
            {"Name": "ATC Player", "Team": "NYY", "PA": 500, "HR": 25,
             "R": 75, "RBI": 80, "SB": 8, "SO": 110, "AVG": 0.260,
             "OBP": 0.330, "SLG": 0.480, "OPS": 0.810, "H": 125},
        ])

        bat_raw = {"atc": atc_bat, "steamer": steamer_bat}
        result = export_ros_to_steamer_csvs(bat_raw, {}, data_dir=tmp_path)

        assert result["batting_rows"] == 1
        # Verify steamer was chosen, not atc
        df = pd.read_csv(tmp_path / "steamer_batting_2026.csv")
        assert df.iloc[0]["Name"] == "Steamer Player"

    def test_falls_back_to_first_available(self, tmp_path):
        atc_bat = _make_bat_df([
            {"Name": "ATC Player", "Team": "NYY", "PA": 500, "HR": 25,
             "R": 75, "RBI": 80, "SB": 8, "SO": 110, "AVG": 0.260,
             "OBP": 0.330, "SLG": 0.480, "OPS": 0.810, "H": 125},
        ])

        bat_raw = {"atc": atc_bat}
        result = export_ros_to_steamer_csvs(bat_raw, {}, data_dir=tmp_path)

        assert result["batting_rows"] == 1
        df = pd.read_csv(tmp_path / "steamer_batting_2026.csv")
        assert df.iloc[0]["Name"] == "ATC Player"

    def test_written_csvs_loadable_by_steamer_batting(self, tmp_path):
        bat_raw = {"steamer": _make_bat_df()}
        export_ros_to_steamer_csvs(bat_raw, {}, data_dir=tmp_path)

        players = load_steamer_batting(tmp_path / "steamer_batting_2026.csv")
        assert len(players) == 2
        names = {p["name"] for p in players}
        assert "Shohei Ohtani" in names
        assert "Aaron Judge" in names
        # Verify stats parsed correctly
        ohtani = [p for p in players if p["name"] == "Shohei Ohtani"][0]
        assert ohtani["proj"]["hr"] == 40
        assert ohtani["proj"]["avg"] == 0.285

    def test_written_csvs_loadable_by_steamer_pitching(self, tmp_path):
        pit_raw = {"steamer": _make_pit_df()}
        export_ros_to_steamer_csvs({}, pit_raw, data_dir=tmp_path)

        players = load_steamer_pitching(tmp_path / "steamer_pitching_2026.csv")
        assert len(players) == 2
        cole = [p for p in players if p["name"] == "Gerrit Cole"][0]
        assert cole["proj"]["w"] == 14
        assert cole["proj"]["era"] == 3.10
        assert "SP" in cole["positions"]

    def test_adds_pos_column_batting_when_missing(self, tmp_path):
        bat_df = _make_bat_df()
        assert "POS" not in bat_df.columns

        bat_raw = {"steamer": bat_df}
        export_ros_to_steamer_csvs(bat_raw, {}, data_dir=tmp_path)

        df = pd.read_csv(tmp_path / "steamer_batting_2026.csv")
        assert "POS" in df.columns
        assert (df["POS"] == "DH").all()

    def test_preserves_existing_pos_column(self, tmp_path):
        bat_df = _make_bat_df()
        bat_df["POS"] = ["OF", "OF"]

        bat_raw = {"steamer": bat_df}
        export_ros_to_steamer_csvs(bat_raw, {}, data_dir=tmp_path)

        df = pd.read_csv(tmp_path / "steamer_batting_2026.csv")
        assert (df["POS"] == "OF").all()

    def test_pitching_pos_derived_from_gs(self, tmp_path):
        pit_raw = {"steamer": _make_pit_df()}
        export_ros_to_steamer_csvs({}, pit_raw, data_dir=tmp_path)

        df = pd.read_csv(tmp_path / "steamer_pitching_2026.csv")
        cole_row = df[df["Name"] == "Gerrit Cole"].iloc[0]
        diaz_row = df[df["Name"] == "Edwin Diaz"].iloc[0]
        assert cole_row["POS"] == "SP"   # GS=30 >= 10
        assert diaz_row["POS"] == "RP"   # GS=0 < 10

    def test_pitching_pos_defaults_sp_when_no_gs(self, tmp_path):
        pit_df = pd.DataFrame([
            {"Name": "Mystery Pitcher", "Team": "LAD", "IP": 100, "W": 5,
             "SV": 0, "SO": 90, "ERA": 3.50, "WHIP": 1.10, "BB": 30,
             "K/9": 8.1},
        ])
        # No GS column at all
        assert "GS" not in pit_df.columns

        pit_raw = {"steamer": pit_df}
        export_ros_to_steamer_csvs({}, pit_raw, data_dir=tmp_path)

        df = pd.read_csv(tmp_path / "steamer_pitching_2026.csv")
        assert df.iloc[0]["POS"] == "SP"

    def test_creates_data_dir_if_missing(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "dir"
        bat_raw = {"steamer": _make_bat_df()}
        result = export_ros_to_steamer_csvs(bat_raw, {}, data_dir=nested)

        assert result["batting_rows"] == 2
        assert (nested / "steamer_batting_2026.csv").exists()
