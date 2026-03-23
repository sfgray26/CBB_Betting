"""
Tests for backend/utils/compress_payload.py and backend/utils/env_utils.py

Run with: pytest tests/test_utils.py -v
"""

import os
import copy
import pytest
from unittest.mock import patch
from backend.utils.compress_payload import compress_prediction_payload
from backend.utils.env_utils import get_float_env


# ---------------------------------------------------------------------------
# compress_prediction_payload
# ---------------------------------------------------------------------------


def _make_prediction(**overrides):
    """Build a minimal prediction dict with full_analysis block."""
    pred = {
        "game_id": "test_game_1",
        "full_analysis": {
            "inputs": {
                "home_style": {
                    "pace": 72.5,
                    "efg_pct": 0.52,
                    "def_efg_pct": 0.48,
                    "is_heuristic": False,
                    "kenpom_four_factors": {"adj_o": 110.5},
                    "extra_field": "should_be_dropped",
                    "another_extra": 99,
                },
                "away_style": {
                    "pace": 68.0,
                    "efg_pct": 0.49,
                    "def_efg_pct": 0.51,
                    "is_heuristic": True,
                    "kenpom_four_factors": {"adj_o": 105.0},
                    "extra_field": "should_be_dropped",
                },
            },
            "notes": "These notes should be removed",
        },
    }
    pred.update(overrides)
    return pred


class TestCompressPredictionPayload:
    def test_returns_list(self):
        result = compress_prediction_payload([_make_prediction()])
        assert isinstance(result, list)

    def test_returns_correct_count(self):
        preds = [_make_prediction(), _make_prediction(game_id="game_2")]
        result = compress_prediction_payload(preds)
        assert len(result) == 2

    def test_notes_are_removed(self):
        result = compress_prediction_payload([_make_prediction()])
        fa = result[0]["full_analysis"]
        assert "notes" not in fa

    def test_essential_fields_retained_home(self):
        result = compress_prediction_payload([_make_prediction()])
        home = result[0]["full_analysis"]["inputs"]["home_style"]
        assert home["pace"] == 72.5
        assert home["efg_pct"] == 0.52
        assert home["def_efg_pct"] == 0.48
        assert home["is_heuristic"] is False
        assert home["kenpom_four_factors"] == {"adj_o": 110.5}

    def test_extra_fields_dropped_home(self):
        result = compress_prediction_payload([_make_prediction()])
        home = result[0]["full_analysis"]["inputs"]["home_style"]
        assert "extra_field" not in home
        assert "another_extra" not in home

    def test_essential_fields_retained_away(self):
        result = compress_prediction_payload([_make_prediction()])
        away = result[0]["full_analysis"]["inputs"]["away_style"]
        assert away["pace"] == 68.0
        assert away["is_heuristic"] is True

    def test_extra_fields_dropped_away(self):
        result = compress_prediction_payload([_make_prediction()])
        away = result[0]["full_analysis"]["inputs"]["away_style"]
        assert "extra_field" not in away

    def test_does_not_mutate_original(self):
        pred = _make_prediction()
        original = copy.deepcopy(pred)
        compress_prediction_payload([pred])
        assert pred == original

    def test_empty_list_returns_empty(self):
        assert compress_prediction_payload([]) == []

    def test_prediction_without_full_analysis(self):
        pred = {"game_id": "bare_game"}
        result = compress_prediction_payload([pred])
        assert result[0]["game_id"] == "bare_game"

    def test_prediction_without_style_blocks(self):
        pred = {
            "game_id": "no_styles",
            "full_analysis": {"inputs": {}, "notes": "drop me"},
        }
        result = compress_prediction_payload([pred])
        assert "notes" not in result[0]["full_analysis"]

    def test_none_style_fields_preserved_as_none(self):
        pred = _make_prediction()
        pred["full_analysis"]["inputs"]["home_style"]["pace"] = None
        result = compress_prediction_payload([pred])
        assert result[0]["full_analysis"]["inputs"]["home_style"]["pace"] is None

    def test_top_level_fields_preserved(self):
        pred = _make_prediction()
        pred["pick"] = "Duke -5.5"
        pred["edge"] = 0.042
        result = compress_prediction_payload([pred])
        assert result[0]["pick"] == "Duke -5.5"
        assert result[0]["edge"] == 0.042


# ---------------------------------------------------------------------------
# get_float_env
# ---------------------------------------------------------------------------


class TestGetFloatEnv:
    def test_returns_float_from_env(self):
        with patch.dict(os.environ, {"TEST_KEY": "1.25"}):
            result = get_float_env("TEST_KEY", "1.0")
        assert result == pytest.approx(1.25)

    def test_returns_default_when_key_missing(self):
        result = get_float_env("NONEXISTENT_KEY_XYZ", "2.5")
        assert result == pytest.approx(2.5)

    def test_strips_leading_spaces(self):
        with patch.dict(os.environ, {"TEST_KEY": "  3.14"}):
            result = get_float_env("TEST_KEY", "1.0")
        assert result == pytest.approx(3.14)

    def test_strips_trailing_spaces(self):
        with patch.dict(os.environ, {"TEST_KEY": "3.14  "}):
            result = get_float_env("TEST_KEY", "1.0")
        assert result == pytest.approx(3.14)

    def test_strips_leading_equals_sign(self):
        with patch.dict(os.environ, {"TEST_KEY": "=1.15"}):
            result = get_float_env("TEST_KEY", "1.0")
        assert result == pytest.approx(1.15)

    def test_strips_equals_with_spaces(self):
        with patch.dict(os.environ, {"TEST_KEY": " =1.15 "}):
            result = get_float_env("TEST_KEY", "1.0")
        assert result == pytest.approx(1.15)

    def test_returns_zero_when_both_invalid(self):
        with patch.dict(os.environ, {"TEST_KEY": "not_a_float"}):
            result = get_float_env("TEST_KEY", "also_not_a_float")
        assert result == 0.0

    def test_falls_back_to_default_on_invalid_value(self):
        with patch.dict(os.environ, {"TEST_KEY": "bad_value"}):
            result = get_float_env("TEST_KEY", "0.75")
        assert result == pytest.approx(0.75)

    def test_returns_float_type(self):
        with patch.dict(os.environ, {"TEST_KEY": "5"}):
            result = get_float_env("TEST_KEY", "1.0")
        assert isinstance(result, float)

    def test_integer_string_parsed_as_float(self):
        with patch.dict(os.environ, {"TEST_KEY": "5"}):
            result = get_float_env("TEST_KEY", "1.0")
        assert result == pytest.approx(5.0)

    def test_zero_value_in_env(self):
        with patch.dict(os.environ, {"TEST_KEY": "0.0"}):
            result = get_float_env("TEST_KEY", "1.0")
        assert result == pytest.approx(0.0)
