"""Tests for fantasy baseball performance and data quality fixes."""
import threading
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def reset_singletons():
    """Reset Yahoo client singletons before and after each test."""
    import backend.fantasy_baseball.yahoo_client_resilient as mod
    mod._client = None
    mod._resilient_client = None
    yield
    mod._client = None
    mod._resilient_client = None


FAKE_ENV = {
    "YAHOO_CLIENT_ID": "test_id",
    "YAHOO_CLIENT_SECRET": "test_secret",
    "YAHOO_LEAGUE_ID": "12345",
    "YAHOO_REFRESH_TOKEN": "test_refresh",
    "YAHOO_ACCESS_TOKEN": "test_access",
}


# ---------------------------------------------------------------------------
# Task 1: Yahoo client singleton
# ---------------------------------------------------------------------------

def test_get_yahoo_client_returns_same_instance(reset_singletons):
    """get_yahoo_client() must return the identical object on repeated calls."""
    with patch.dict("os.environ", FAKE_ENV):
        from backend.fantasy_baseball.yahoo_client_resilient import get_yahoo_client
        c1 = get_yahoo_client()
        c2 = get_yahoo_client()
        assert c1 is c2


def test_get_resilient_yahoo_client_returns_same_instance(reset_singletons):
    """get_resilient_yahoo_client() must return the identical object on repeated calls."""
    with patch.dict("os.environ", FAKE_ENV):
        from backend.fantasy_baseball.yahoo_client_resilient import get_resilient_yahoo_client
        c1 = get_resilient_yahoo_client()
        c2 = get_resilient_yahoo_client()
        assert c1 is c2


def test_singleton_thread_safety(reset_singletons):
    """Concurrent calls must produce exactly one construction, not multiple."""
    import backend.fantasy_baseball.yahoo_client_resilient as mod
    construction_count = {"n": 0}
    original_init = mod.YahooFantasyClient.__init__

    def slow_init(self, *args, **kwargs):
        import time
        time.sleep(0.01)  # Widen the race window
        construction_count["n"] += 1
        original_init(self, *args, **kwargs)

    results = []

    with patch.dict("os.environ", FAKE_ENV):
        with patch.object(mod.YahooFantasyClient, "__init__", slow_init):
            from backend.fantasy_baseball.yahoo_client_resilient import get_yahoo_client

            def grab():
                results.append(get_yahoo_client())

            threads = [threading.Thread(target=grab) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

    assert len(set(id(r) for r in results)) == 1
    assert construction_count["n"] == 1


def test_get_yahoo_client_raises_on_missing_credentials(reset_singletons):
    """get_yahoo_client() raises YahooAuthError immediately when credentials are absent.
    YahooFantasyClient.__init__ validates client_id/client_secret at construction time,
    so the error surfaces before any API call. _client must remain None after failure
    so the next call with correct credentials can succeed.
    """
    import backend.fantasy_baseball.yahoo_client_resilient as mod
    from backend.fantasy_baseball.yahoo_client_resilient import get_yahoo_client, YahooAuthError

    with patch.dict("os.environ", {
        "YAHOO_CLIENT_ID": "",
        "YAHOO_CLIENT_SECRET": "",
        "YAHOO_REFRESH_TOKEN": "",
        "YAHOO_ACCESS_TOKEN": "",
    }):
        with pytest.raises(YahooAuthError):
            get_yahoo_client()

    # _client must remain None after the failed construction attempt
    # so the next call can retry with correct credentials
    assert mod._client is None


# ---------------------------------------------------------------------------
# Task 2: ProjectionsLoader lru_cache
# ---------------------------------------------------------------------------

def test_load_full_board_cached():
    """load_full_board() must not re-execute CSV parsing on repeated calls."""
    from backend.fantasy_baseball.projections_loader import load_full_board
    load_full_board.cache_clear()

    with patch("backend.fantasy_baseball.projections_loader.load_steamer_batting") as mock_bat:
        with patch("backend.fantasy_baseball.projections_loader.load_steamer_pitching", return_value=[]):
            with patch("pathlib.Path.exists", return_value=True):
                mock_bat.return_value = []
                load_full_board()
                load_full_board()
                # The CSV loader must be called exactly once — second call hits cache
                assert mock_bat.call_count == 1


def test_load_full_board_cache_clear_triggers_reload():
    """cache_clear() must cause the next call to re-read CSVs."""
    from backend.fantasy_baseball.projections_loader import load_full_board
    load_full_board.cache_clear()

    with patch("backend.fantasy_baseball.projections_loader.load_steamer_batting") as mock_bat:
        with patch("backend.fantasy_baseball.projections_loader.load_steamer_pitching", return_value=[]):
            with patch("pathlib.Path.exists", return_value=True):
                mock_bat.return_value = []
                load_full_board()
                load_full_board.cache_clear()
                load_full_board()
                assert mock_bat.call_count == 2


# ---------------------------------------------------------------------------
# Task 3: ADP name normalization
# ---------------------------------------------------------------------------

def test_make_player_id_strips_suffix():
    from backend.fantasy_baseball.projections_loader import _make_player_id
    assert _make_player_id("Ronald Acuña Jr.") == _make_player_id("Ronald Acuna")
    assert _make_player_id("Ken Griffey Jr.") == _make_player_id("Ken Griffey")
    assert _make_player_id("Cal Ripken Jr.") == _make_player_id("Cal Ripken")


def test_make_player_id_handles_last_name_first():
    from backend.fantasy_baseball.projections_loader import _make_player_id
    assert _make_player_id("Ohtani, Shohei") == _make_player_id("Shohei Ohtani")
    assert _make_player_id("Betts, Mookie") == _make_player_id("Mookie Betts")


def test_make_player_id_handles_extra_accents():
    from backend.fantasy_baseball.projections_loader import _make_player_id
    assert _make_player_id("José Ramîrez") == _make_player_id("Jose Ramirez")


def test_apply_adp_last_name_initial_fallback():
    """_apply_adp must match when exact ID fails but last+initial matches.

    Some ADP sources (e.g. FantasyPros) export abbreviated first names like
    "A. Judge" while the player board has "Aaron Judge". After normalization
    these produce different IDs ("a_judge" vs "aaron_judge"), but the initial
    fallback bridges them.
    """
    from backend.fantasy_baseball.projections_loader import _apply_adp, _make_player_id

    # ADP CSV has abbreviated first name — normalizes to "a_judge"
    adp_map = {"a_judge": 4.0}

    # Player board has full name — normalizes to "aaron_judge"
    players = [{
        "id": _make_player_id("Aaron Judge"),
        "name": "Aaron Judge",
        "adp": 999.0,
    }]

    _apply_adp(players, adp_map)
    assert players[0]["adp"] == 4.0
