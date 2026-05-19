"""
Tests for dashboard_service.py correctness.

Covers the P1 bug where _get_waiver_targets passed raw Yahoo stat IDs
(e.g. "12", "13") as CategoryDeficitOut.category, preventing
compute_need_score from matching against the category board and forcing
a fallback to player_z_score (0.0) for every player.
"""
import pytest


def test_compute_need_score_requires_canonical_not_yahoo_stat_ids():
    """Verify that compute_need_score produces higher scores with canonical codes
    vs Yahoo stat IDs — the core of the need_score 0.00 bug."""
    from backend.fantasy_baseball.category_aware_scorer import compute_need_score
    from backend.schemas import CategoryDeficitOut

    # Yahoo stat IDs as category field (the buggy behavior).
    # "12" -> HR_B, "13" -> RBI in the league's yahoo_id_index.
    bad_deficits = [
        CategoryDeficitOut(category="12",  my_total=10.0, opponent_total=15.0, deficit=5.0, winning=False),
        CategoryDeficitOut(category="13",  my_total=20.0, opponent_total=23.0, deficit=3.0, winning=False),
    ]
    # Canonical codes (the correct behavior after the fix).
    good_deficits = [
        CategoryDeficitOut(category="HR_B", my_total=10.0, opponent_total=15.0, deficit=5.0, winning=False),
        CategoryDeficitOut(category="RBI",  my_total=20.0, opponent_total=23.0, deficit=3.0, winning=False),
    ]

    # cat_scores use lowercase board keys (hr, rbi).
    cat_scores = {"hr": 1.5, "rbi": 2.0}
    z_score = 1.0

    bad_score  = compute_need_score(cat_scores, z_score, bad_deficits,  10)
    good_score = compute_need_score(cat_scores, z_score, good_deficits, 10)

    # With Yahoo stat IDs: no category match -> falls back to z_score (1.0)
    # With canonical codes: category alignment -> need_score > z_score fallback
    assert good_score > bad_score, (
        f"canonical codes ({good_score:.3f}) must exceed Yahoo ID fallback ({bad_score:.3f})"
    )
