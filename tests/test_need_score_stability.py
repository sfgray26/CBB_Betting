"""Tests for need_score stability fixes."""


def test_reset_board_cache_clears_projection_cache():
    """reset_board_cache must clear _projection_cache so post-ingestion projections are served."""
    import backend.fantasy_baseball.player_board as pb

    # Seed the projection cache with a fake stale entry
    pb._projection_cache["test_key_12345"] = {"name": "Fake Player", "z_score": 99.9}
    assert "test_key_12345" in pb._projection_cache

    # Calling reset_board_cache must clear it
    pb.reset_board_cache()

    assert "test_key_12345" not in pb._projection_cache, (
        "reset_board_cache() must call _projection_cache.clear() so "
        "post-ingestion DB updates are not masked by in-process cached projections"
    )
