"""Tests for fuzzy name matching (step 3b) in get_or_create_projection().

Fix A: difflib.SequenceMatcher step added after exact and accent-strip checks.
"""
import pytest


def _make_player(name: str, positions: list[str]) -> dict:
    return {"name": name, "positions": positions, "player_key": None}


def _make_board_entry(name: str, z: float) -> dict:
    return {
        "id": name.lower().replace(" ", "_"),
        "name": name,
        "team": "TEST",
        "positions": ["SP"],
        "type": "pitcher",
        "tier": 1,
        "rank": 1,
        "adp": 27.0,
        "z_score": z,
        "cat_scores": {},
        "proj": {},
        "is_keeper": False,
        "keeper_round": None,
        "is_proxy": False,
    }


# ---------------------------------------------------------------------------
# Patch the board so tests don't depend on the real CSV file
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def patch_board(monkeypatch):
    """Inject a controlled board for all fuzzy tests."""
    import backend.fantasy_baseball.player_board as pb

    board = [
        _make_board_entry("Cristopher Sanchez", 3.76),
        _make_board_entry("Jose Ramirez", 4.10),
        _make_board_entry("Carlos Correa", 2.50),
        _make_board_entry("Willi Castro", 0.80),
    ]

    monkeypatch.setattr(pb, "_BOARD", board)
    monkeypatch.setattr(pb, "_projection_cache", {})
    # get_board() must return the patched _BOARD
    monkeypatch.setattr(pb, "get_board", lambda apply_park_factors=True: board)
    yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFuzzyNameMatching:
    def test_christopher_vs_cristopher(self):
        """Yahoo 'Christopher Sanchez' must match board 'Cristopher Sanchez', z=+3.76."""
        from backend.fantasy_baseball.player_board import get_or_create_projection, _projection_cache
        _projection_cache.clear()

        yahoo_player = _make_player("Christopher Sanchez", ["SP"])
        result = get_or_create_projection(yahoo_player)

        assert result["name"] == "Cristopher Sanchez"
        assert result["z_score"] == pytest.approx(3.76)
        assert result.get("is_proxy") is not True

    def test_accent_still_works(self):
        """José Ramírez → Jose Ramirez via accent-strip (step 3, not step 3b)."""
        from backend.fantasy_baseball.player_board import get_or_create_projection, _projection_cache
        _projection_cache.clear()

        yahoo_player = _make_player("Jose Ramirez", ["3B"])
        result = get_or_create_projection(yahoo_player)

        assert result["z_score"] == pytest.approx(4.10)
        assert result.get("is_proxy") is not True

    def test_no_false_positive(self):
        """'Carlos Correa' exact-matches the board entry and does NOT pick up 'Carlos Santana'."""
        import backend.fantasy_baseball.player_board as pb
        from backend.fantasy_baseball.player_board import get_or_create_projection
        pb._projection_cache.clear()

        # Add Santana to the board for this test
        extended = pb.get_board() + [_make_board_entry("Carlos Santana", 1.00)]
        pb._BOARD = extended
        pb.get_board = lambda apply_park_factors=True: extended

        yahoo_player = _make_player("Carlos Correa", ["SS"])
        result = get_or_create_projection(yahoo_player)

        # Correa is an exact match — should never drift to Santana
        assert result["name"] == "Carlos Correa"

    def test_proxy_for_unknown(self):
        """Completely unknown player gets a proxy entry with is_proxy=True."""
        from backend.fantasy_baseball.player_board import get_or_create_projection, _projection_cache
        _projection_cache.clear()

        yahoo_player = _make_player("Totally Unknown Player Xyz", ["OF"])
        result = get_or_create_projection(yahoo_player)

        assert result.get("is_proxy") is True

    def test_cache_hit(self):
        """Second call for same player_key returns cached entry without re-scanning board."""
        from backend.fantasy_baseball.player_board import get_or_create_projection, _projection_cache
        _projection_cache.clear()

        yahoo_player = {"name": "Christopher Sanchez", "positions": ["SP"], "player_key": "469.p.9999"}
        result1 = get_or_create_projection(yahoo_player)
        assert "469.p.9999" in _projection_cache

        # Second call should hit cache (result must be identical object)
        result2 = get_or_create_projection(yahoo_player)
        assert result1 is result2
