"""
Smoke tests for admin_scoring_diagnostics (P27 NSB rollout verification).

Mirrors the structure of test_admin_statcast_diagnostics: verify the module
imports, endpoints register under expected paths, SQL references real model
columns, and helpers/whitelists behave as documented. No DB required.
"""

import inspect

import pytest

from backend import admin_scoring_diagnostics as diag
from backend.models import (
    PlayerRollingStats,
    PlayerScore,
    PlayerIDMapping,
)


# ---------------------------------------------------------------------------
# Endpoint registration
# ---------------------------------------------------------------------------

EXPECTED_PATHS = {
    "/diagnose-scoring/nsb-rollout",
    "/diagnose-scoring/nsb-leaders",
    "/diagnose-scoring/nsb-player",
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


def test_rollout_sql_uses_real_rolling_stats_columns():
    """
    The rollout endpoint FILTERs on player_rolling_stats NSB columns. If any of
    these column names drift, the endpoint should fail in CI, not prod.
    """
    source = inspect.getsource(diag.diagnose_nsb_rollout)
    prs_cols = _model_columns(PlayerRollingStats)
    for col in ("w_ab", "w_stolen_bases", "w_caught_stealing", "w_net_stolen_bases",
                "as_of_date", "window_days"):
        assert col in prs_cols, f"{col} missing from PlayerRollingStats model"
        assert col in source, f"rollout endpoint should reference {col}"


def test_rollout_sql_uses_real_player_scores_columns():
    source = inspect.getsource(diag.diagnose_nsb_rollout)
    ps_cols = _model_columns(PlayerScore)
    for col in ("player_type", "z_sb", "z_nsb", "as_of_date", "window_days"):
        assert col in ps_cols, f"{col} missing from PlayerScore model"
        assert col in source, f"rollout endpoint should reference {col}"


def test_leaders_sql_uses_real_columns():
    """
    The leaders endpoint joins player_scores + player_id_mapping + player_rolling_stats.
    All columns it references must exist on the models.
    """
    source = inspect.getsource(diag.diagnose_nsb_leaders)
    ps_cols = _model_columns(PlayerScore)
    pim_cols = _model_columns(PlayerIDMapping)
    prs_cols = _model_columns(PlayerRollingStats)

    for col in ("bdl_player_id", "games_in_window", "z_sb", "z_nsb",
                "composite_z", "score_0_100"):
        assert col in ps_cols, f"{col} missing from PlayerScore"
    assert "bdl_id" in pim_cols
    assert "full_name" in pim_cols
    for col in ("w_stolen_bases", "w_caught_stealing", "w_net_stolen_bases"):
        assert col in prs_cols, f"{col} missing from PlayerRollingStats"

    # And the SQL actually names the aliased columns
    assert "ps.z_nsb" in source
    assert "pim.bdl_id" in source
    assert "pim.full_name" in source
    assert "prs.w_net_stolen_bases" in source


def test_player_detail_sql_uses_real_columns():
    source = inspect.getsource(diag.diagnose_nsb_player)
    ps_cols = _model_columns(PlayerScore)
    prs_cols = _model_columns(PlayerRollingStats)
    for col in ("window_days", "games_in_window", "player_type", "z_sb", "z_nsb",
                "composite_z", "score_0_100"):
        assert col in ps_cols, f"{col} missing from PlayerScore"
    for col in ("w_stolen_bases", "w_caught_stealing", "w_net_stolen_bases",
                "w_ab", "w_hits"):
        assert col in prs_cols, f"{col} missing from PlayerRollingStats"
    assert "ps.z_nsb" in source
    assert "prs.w_net_stolen_bases" in source


# ---------------------------------------------------------------------------
# Whitelist enforcement
# ---------------------------------------------------------------------------

def test_allowed_windows_constant():
    """The only rolling windows we compute downstream are 7 / 14 / 30."""
    assert diag._ALLOWED_WINDOWS == frozenset({7, 14, 30})


def test_rollout_rejects_bad_window_days():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        diag.diagnose_nsb_rollout(window_days=99, as_of_date=None)
    assert exc.value.status_code == 400
    assert "window_days" in exc.value.detail


def test_leaders_rejects_bad_window_days():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        diag.diagnose_nsb_leaders(
            direction="top", limit=20, window_days=42, as_of_date=None,
        )
    assert exc.value.status_code == 400
    assert "window_days" in exc.value.detail


def test_leaders_rejects_bad_direction():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        diag.diagnose_nsb_leaders(
            direction="sideways", limit=20, window_days=14, as_of_date=None,
        )
    assert exc.value.status_code == 400
    assert "direction" in exc.value.detail


# ---------------------------------------------------------------------------
# Helper behavior
# ---------------------------------------------------------------------------

def test_helpers_handle_none():
    assert diag._f(None) is None
    assert diag._f("3.14") == 3.14
    assert diag._f(7) == 7.0
    assert diag._f("not a number") is None
    assert diag._i(None) is None
    assert diag._i("5") == 5
    assert diag._i("nope") is None
    assert diag._d(None) is None


def test_nsb_integrity_check_ok():
    assert diag._nsb_integrity_check(10.0, 3.0, 7.0) == "ok"


def test_nsb_integrity_check_mismatch():
    result = diag._nsb_integrity_check(10.0, 3.0, 99.0)
    assert result.startswith("mismatch")


def test_nsb_integrity_check_all_none():
    assert diag._nsb_integrity_check(None, None, None).startswith("n/a")


def test_nsb_integrity_check_partial_source():
    # sb or cs missing but nsb populated -> partial_source
    assert diag._nsb_integrity_check(None, 2.0, 5.0) == "partial_source"
    assert diag._nsb_integrity_check(5.0, None, 5.0) == "partial_source"


def test_nsb_integrity_check_nsb_null():
    assert diag._nsb_integrity_check(10.0, 3.0, None) == "nsb_null"


def test_interpret_rollout_messages():
    """Each verdict branch must produce a non-empty human-readable string."""
    for verdict in ("no_data", "healthy", "partial", "empty"):
        msg = diag._interpret_rollout(verdict, 50.0)
        assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# Router mount check (main.py must include this router)
# ---------------------------------------------------------------------------

def test_router_is_mounted_in_main_app():
    """
    Catches the easy mistake of adding an APIRouter but forgetting to mount
    it on the FastAPI app.
    """
    from backend.main import app
    mounted_paths = {r.path for r in app.routes}
    # When mounted with prefix="/admin", diagnostic paths appear as /admin/diagnose-scoring/...
    for p in EXPECTED_PATHS:
        full = f"/admin{p}"
        assert full in mounted_paths, (
            f"Expected scoring-diagnostics endpoint {full} to be mounted on app"
        )
