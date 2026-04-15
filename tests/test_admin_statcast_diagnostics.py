"""
Smoke tests for admin_statcast_diagnostics.

These tests verify the module imports, all endpoints register under the
expected paths, and the SQL strings reference only columns that exist in
the models (catches typos like at_bats vs ab or hits_total vs hits).
"""

import inspect

import pytest

from backend import admin_statcast_diagnostics as diag
from backend.models import (
    StatcastPerformance,
    MLBPlayerStats,
    PlayerIDMapping,
)


# ---------------------------------------------------------------------------
# Endpoint registration
# ---------------------------------------------------------------------------

EXPECTED_PATHS = {
    "/diagnose-statcast/summary",
    "/diagnose-statcast/by-date",
    "/diagnose-statcast/leaderboard",
    "/diagnose-statcast/player",
    "/diagnose-statcast/raw-sample",
    "/diagnose-statcast/sanity-check",
}


def test_all_endpoints_registered():
    registered = {r.path for r in diag.router.routes}
    missing = EXPECTED_PATHS - registered
    assert not missing, f"Missing expected endpoints: {missing}"


def test_all_endpoints_are_get():
    for r in diag.router.routes:
        if r.path in EXPECTED_PATHS:
            assert "GET" in r.methods, f"{r.path} must be GET (read-only diagnostic)"


# ---------------------------------------------------------------------------
# Column-existence sanity check (catches typos before prod)
# ---------------------------------------------------------------------------

def _model_columns(model):
    return {c.name for c in model.__table__.columns}


def test_statcast_sql_uses_real_columns_in_by_date_endpoint():
    """
    Verify the by-date endpoint's SUM/MAX targets are real StatcastPerformance
    columns. This is a targeted check on the one query that hardcodes every
    column we care about (ab, pa, h, sb, cs, ip).
    """
    source = inspect.getsource(diag.diagnose_statcast_by_date)
    sp_cols = _model_columns(StatcastPerformance)
    for col in ("ab", "pa", "h", "sb", "cs", "ip", "game_date", "player_id"):
        assert col in sp_cols, f"{col} missing from StatcastPerformance model"
        assert col in source, f"by-date endpoint should reference {col}"


def test_mlb_player_stats_sanity_query_uses_real_columns():
    """
    The sanity-check endpoint joins mlb_player_stats.ab/hits with
    player_id_mapping.bdl_id/full_name. Verify these column names are real.
    """
    source = inspect.getsource(diag.diagnose_statcast_sanity_check)
    mps_cols = _model_columns(MLBPlayerStats)
    pim_cols = _model_columns(PlayerIDMapping)

    # Hard-code the columns the sanity check depends on
    assert "ab" in mps_cols, "mlb_player_stats must have 'ab' column"
    assert "hits" in mps_cols, "mlb_player_stats must have 'hits' column"
    assert "bdl_player_id" in mps_cols
    assert "bdl_id" in pim_cols
    assert "full_name" in pim_cols

    # And that the source actually references them (no silent renames)
    assert "mps.ab" in source
    assert "mps.hits" in source
    assert "pim.bdl_id" in source
    assert "pim.full_name" in source


# ---------------------------------------------------------------------------
# Leaderboard metric whitelist
# ---------------------------------------------------------------------------

def test_leaderboard_metric_whitelist_is_subset_of_real_columns():
    """
    Every metric in the leaderboard whitelist must exist on StatcastPerformance.
    Otherwise a user-supplied metric would hit a SQL error instead of a 400.
    """
    sp_cols = _model_columns(StatcastPerformance)
    bad = diag._ALLOWED_METRICS - sp_cols
    assert not bad, f"Whitelist contains metrics not on model: {bad}"


def test_helpers_handle_none():
    assert diag._f(None) is None
    assert diag._f("3.14") == 3.14
    assert diag._f(7) == 7.0
    assert diag._f("not a number") is None
    assert diag._d(None) is None
