"""Tests for player projection coverage — name-based fallback path."""
import pytest
from unittest.mock import MagicMock, patch


def _make_projection_row(name: str, cat_scores: dict):
    row = MagicMock()
    row.player_name = name
    row.player_type = "pitcher"
    row.cat_scores = cat_scores
    row.player_id = "12345"
    row.z_score = 1.5
    row.era = 2.80
    row.whip = 0.98
    row.k_per_nine = 9.5
    row.w = 12
    row.qs = 18
    return row


def test_name_fallback_finds_projection_when_identity_missing():
    """When Yahoo player has no identity chain, name fallback must find player_projections row."""
    from backend.fantasy_baseball.player_board import get_or_create_projection

    yahoo_player = {
        "name": "Cristopher Sanchez",
        "player_key": "mlb.p.99999",
        "positions": ["SP"],
        "team": "PHI",
        "percent_owned": 65.0,
    }

    mock_row = _make_projection_row("Cristopher Sanchez", {"era": -1.2, "k_p": 0.9, "w": 0.7})

    with patch("backend.fantasy_baseball.player_board._lookup_projection_by_name") as mock_lookup:
        mock_lookup.return_value = mock_row
        result = get_or_create_projection(yahoo_player)

    assert result.get("cat_scores") == {"era": -1.2, "k_p": 0.9, "w": 0.7}
    assert result.get("fusion_source") != "draft_board_fallback"


def test_draft_board_fallback_only_when_no_projection():
    """Draft board fallback should only fire when player_projections has no match either."""
    from backend.fantasy_baseball.player_board import get_or_create_projection

    yahoo_player = {
        "name": "Completely Unknown Minor Leaguer",
        "player_key": "mlb.p.00001",
        "positions": ["SP"],
        "team": "AAA",
        "percent_owned": 0.0,
    }

    with patch("backend.fantasy_baseball.player_board._lookup_projection_by_name", return_value=None):
        result = get_or_create_projection(yahoo_player)

    assert result is not None
    assert "cat_scores" in result


def test_cristopher_spelling_handled():
    """'Cristopher' (no h) must fuzzy-match or exact-match 'Cristopher Sanchez' in player_projections."""
    import difflib
    from backend.fantasy_baseball.id_resolution_service import _normalize_name

    db_names = [_normalize_name("Cristopher Sánchez")]
    yahoo_name = _normalize_name("Cristopher Sanchez")

    assert yahoo_name in db_names or difflib.get_close_matches(yahoo_name, db_names, n=1, cutoff=0.85)
