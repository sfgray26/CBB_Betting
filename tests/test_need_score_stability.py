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


def test_waiver_player_out_has_stability_fields():
    """WaiverPlayerOut must accept need_score_ci, need_score_volatile, projection_source."""
    from backend.schemas import WaiverPlayerOut
    p = WaiverPlayerOut(
        player_id="123.p.456",
        name="José Caballero",
        team="TB",
        position="SS",
        need_score=12.59,
        need_score_ci=1.89,
        need_score_volatile=True,
        projection_source="steamer+statcast",
    )
    assert p.need_score_ci == 1.89
    assert p.need_score_volatile is True
    assert p.projection_source == "steamer+statcast"


def test_waiver_player_out_stability_fields_default_safe():
    """Stability fields default to None/False so existing callers are unaffected."""
    from backend.schemas import WaiverPlayerOut
    p = WaiverPlayerOut(player_id="x", name="X", team="T", position="OF")
    assert p.need_score_ci is None
    assert p.need_score_volatile is False
    assert p.projection_source is None


def test_waiver_wire_response_has_metadata():
    """WaiverWireResponse must expose scored_at and scoring_model_version."""
    from backend.schemas import WaiverWireResponse
    from datetime import date, datetime
    r = WaiverWireResponse(
        week_end=date.today(),
        matchup_opponent="Opp",
        category_deficits=[],
        top_available=[],
        two_start_pitchers=[],
        scored_at=datetime(2026, 6, 11, 10, 0, 0),
        scoring_model_version="2.1",
    )
    assert r.scoring_model_version == "2.1"
    assert r.scored_at is not None


def test_n_cats_formula_consistent_in_source():
    """Both waiver endpoints must use len(category_deficits) for n_cats, not _need_vector.needs."""
    import pathlib
    src = (pathlib.Path(__file__).parent.parent / "backend" / "routers" / "fantasy.py").read_text()

    # The buggy line: n_cats = max(1, len(_need_vector.needs))
    assert "max(1, len(_need_vector.needs))" not in src, (
        "Recommendations endpoint must use len(category_deficits) for n_cats, "
        "not len(_need_vector.needs) — the translated dict can be smaller due to "
        "_CANONICAL_TO_BOARD key collisions, inflating need_score vs main waiver endpoint"
    )
