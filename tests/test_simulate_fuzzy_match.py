"""Tests for fuzzy name matching in _fetch_rosters_for_simulate."""
import difflib
from unittest.mock import MagicMock


def _make_proj(name: str, cat_scores: dict):
    p = MagicMock()
    p.player_name = name
    p.cat_scores = cat_scores
    return p


def _fuzzy_lookup(name_map: dict, player_name: str):
    """Replicates the fuzzy lookup logic from _player_dict."""
    from backend.routers.fantasy import _normalize_identity_name

    key = _normalize_identity_name(player_name)
    if key in name_map:
        return name_map[key]

    matches = difflib.get_close_matches(key, name_map.keys(), n=1, cutoff=0.85)
    if matches:
        return name_map[matches[0]]
    return None


def test_fuzzy_match_finds_projection():
    """'Aaron Judge' in roster matches 'Aaron Judge' projection exactly."""
    from backend.routers.fantasy import _normalize_identity_name
    name_map = {_normalize_identity_name("Aaron Judge"): _make_proj("Aaron Judge", {"hr": 0.9})}

    result = _fuzzy_lookup(name_map, "Aaron Judge")
    assert result is not None
    assert result.cat_scores["hr"] == 0.9


def test_fuzzy_match_handles_suffix():
    """'Ronald Acuna Jr' matches 'Ronald Acuña Jr.' after normalization."""
    from backend.routers.fantasy import _normalize_identity_name
    name_map = {_normalize_identity_name("Ronald Acuña Jr."): _make_proj("Ronald Acuña Jr.", {"hr": 0.8})}

    result = _fuzzy_lookup(name_map, "Ronald Acuna Jr")
    assert result is not None


def test_fuzzy_match_no_false_positive():
    """'Aaron Nola' should not match 'Aaron Judge' (too different at cutoff=0.85)."""
    from backend.routers.fantasy import _normalize_identity_name
    name_map = {_normalize_identity_name("Aaron Judge"): _make_proj("Aaron Judge", {"hr": 0.9})}

    result = _fuzzy_lookup(name_map, "Aaron Nola")
    assert result is None
