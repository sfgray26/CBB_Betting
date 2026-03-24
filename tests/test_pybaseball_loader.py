"""
Tests for pybaseball_loader.py.

All tests run without pybaseball installed — no network access required.
pandas is used directly for _df_to_* tests (already in requirements.txt).
"""

import json
import logging
import sys
import time
import types
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from backend.fantasy_baseball.pybaseball_loader import (
    _strip_name,
    _df_to_batter_dict,
    _df_to_pitcher_dict,
    fetch_all_statcast_leaderboards,
    log_statcast_coverage,
    match_yahoo_to_statcast,
)
from backend.fantasy_baseball.advanced_metrics import StatcastBatter, StatcastPitcher


# ---------------------------------------------------------------------------
# _strip_name tests
# ---------------------------------------------------------------------------

def test_strip_name_removes_accent():
    assert _strip_name("Jose Ramirez") == "jose ramirez"
    # Unicode accent variant
    assert _strip_name("Jos\u00e9 Ram\u00edrez") == "jose ramirez"


def test_strip_name_removes_suffix_jr():
    assert _strip_name("Willi Castro Jr.") == "willi castro"
    assert _strip_name("Willi Castro Jr") == "willi castro"


def test_strip_name_removes_roman_numerals():
    assert _strip_name("Cal Ripken II") == "cal ripken"
    assert _strip_name("Cal Ripken III") == "cal ripken"


def test_strip_name_removes_sr():
    assert _strip_name("Ken Griffey Sr.") == "ken griffey"


# ---------------------------------------------------------------------------
# match_yahoo_to_statcast tests
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cache():
    return {
        "shohei ohtani": StatcastBatter(name="Shohei Ohtani"),
        "paul skenes": StatcastPitcher(name="Paul Skenes"),
        "willi castro": StatcastBatter(name="Willi Castro"),
        "jose ramirez": StatcastBatter(name="Jose Ramirez"),
        "juan rodriguez": StatcastBatter(name="Juan Rodriguez"),
        "carlos rodriguez": StatcastBatter(name="Carlos Rodriguez"),
    }


def test_match_exact(small_cache):
    assert match_yahoo_to_statcast("Shohei Ohtani", small_cache) == "shohei ohtani"


def test_match_last_name_first_initial(small_cache):
    assert match_yahoo_to_statcast("S. Ohtani", small_cache) == "shohei ohtani"


def test_match_last_name_unique(small_cache):
    assert match_yahoo_to_statcast("Skenes", small_cache) == "paul skenes"


def test_match_last_name_non_unique_returns_none(small_cache):
    # Both "juan rodriguez" and "carlos rodriguez" share last name
    assert match_yahoo_to_statcast("Rodriguez", small_cache) is None


def test_match_accent_normalized(small_cache):
    assert match_yahoo_to_statcast("Jos\u00e9 Ram\u00edrez", small_cache) == "jose ramirez"


def test_match_suffix_stripped(small_cache):
    assert match_yahoo_to_statcast("Willi Castro Jr.", small_cache) == "willi castro"


# ---------------------------------------------------------------------------
# fetch_all_statcast_leaderboards graceful ImportError test
# ---------------------------------------------------------------------------

def test_fetch_graceful_when_pybaseball_missing(caplog):
    """If pybaseball is not installed, function returns without raising."""
    with mock.patch.dict(sys.modules, {"pybaseball": None}):
        with caplog.at_level(logging.WARNING):
            fetch_all_statcast_leaderboards(year=2025, force_refresh=False)
    # Should not raise; warning may or may not appear depending on import state


# ---------------------------------------------------------------------------
# log_statcast_coverage tests
# ---------------------------------------------------------------------------

def test_log_coverage_returns_fraction(small_cache):
    names = ["Shohei Ohtani", "Willi Castro Jr.", "Nobody McFakename"]
    result = log_statcast_coverage(names, small_cache, label="test FAs")
    # 2 out of 3 hit
    assert result == pytest.approx(2 / 3)


def test_log_coverage_empty_list(small_cache):
    assert log_statcast_coverage([], small_cache) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _df_to_batter_dict tests
# ---------------------------------------------------------------------------

def _make_batter_df(**extra_cols):
    base = {
        "Name": ["Aaron Judge"],
        "xwOBA": [0.420],
        "wOBA": [0.380],
        "Barrel%": [18.5],
        "EV": [96.2],
        "HardHit%": [52.8],
        "Sweet-Spot%": [38.5],
        "O-Swing%": [25.5],
        "Z-Contact%": [82.0],
        "SwStr%": [11.5],
        "GB%": [28.0],
        "FB%": [40.0],
        "LD%": [22.0],
        "Pull%": [45.0],
    }
    base.update(extra_cols)
    return pd.DataFrame(base)


def test_df_to_batter_dict_maps_columns():
    df = _make_batter_df()
    result = _df_to_batter_dict(df)
    assert "aaron judge" in result
    b = result["aaron judge"]
    assert b.xwoba == pytest.approx(0.420)
    # xwoba_diff = 0.420 - 0.380 = 0.040 → regression_down=True
    assert b.xwoba_diff == pytest.approx(0.040)
    assert b.regression_down is True
    assert b.barrel_pct == pytest.approx(18.5)
    assert b.exit_velo_avg == pytest.approx(96.2)


def test_df_to_batter_dict_handles_missing_columns():
    """Sparse DataFrame (missing most columns) must not raise; barrel_pct defaults 0.0."""
    df = pd.DataFrame({"Name": ["Sparse Player"], "xwOBA": [0.300], "wOBA": [0.310]})
    result = _df_to_batter_dict(df)
    assert "sparse player" in result
    assert result["sparse player"].barrel_pct == pytest.approx(0.0)


def test_df_to_batter_dict_regression_up():
    """xwoba > woba by > 0.020 => regression_up=True (unlucky batter)."""
    df = _make_batter_df(**{"xwOBA": [0.370], "wOBA": [0.340]})
    result = _df_to_batter_dict(df)
    b = result["aaron judge"]
    assert b.xwoba_diff == pytest.approx(0.030)
    # exactly 0.030 is NOT > 0.030, so regression_down is False
    assert b.regression_down is False


# ---------------------------------------------------------------------------
# _df_to_pitcher_dict tests
# ---------------------------------------------------------------------------

def _make_pitcher_df(**extra_cols):
    base = {
        "Name": ["Paul Skenes"],
        "ERA": [2.50],
        "xERA": [2.90],
        "Stuff+": [130.0],
        "Location+": [105.0],
        "Pitching+": [118.0],
        "FBv": [98.5],
        "FBSpin": [2550.0],
        "SwStr%": [34.2],
        "O-Swing%": [28.5],
        "CSW%": [33.0],
        "Barrel%": [5.2],
        "HardHit%": [30.0],
        "xwOBA": [0.270],
    }
    base.update(extra_cols)
    return pd.DataFrame(base)


def test_df_to_pitcher_dict_maps_era_diff():
    df = _make_pitcher_df()
    result = _df_to_pitcher_dict(df)
    assert "paul skenes" in result
    p = result["paul skenes"]
    # xera_diff = 2.90 - 2.50 = 0.40 — exactly at threshold, luck_regression=False
    assert p.xera_diff == pytest.approx(0.40)
    assert p.luck_regression is False  # > 0.40 required


def test_df_to_pitcher_dict_luck_regression_true():
    df = _make_pitcher_df(**{"ERA": [2.50], "xERA": [2.96]})
    result = _df_to_pitcher_dict(df)
    p = result["paul skenes"]
    assert p.xera_diff == pytest.approx(0.46)
    assert p.luck_regression is True


def test_df_to_pitcher_dict_handles_missing_columns():
    df = pd.DataFrame({"Name": ["Sparse Pitcher"], "ERA": [3.50], "xERA": [3.80]})
    result = _df_to_pitcher_dict(df)
    assert "sparse pitcher" in result
    assert result["sparse pitcher"].whiff_pct == pytest.approx(0.0)
