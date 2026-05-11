"""
Tests for _update_projection_cat_scores MLBAM fallback (Item 2, Session J).

Verifies the three-tier MLBAM resolution added in Session J:
  Primary:   FanGraphs normalized player_id in PlayerIDMapping.normalized_name
  Secondary: raw player name (lowercased) in PlayerIDMapping.normalized_name
  Tertiary:  existing PlayerProjection row matched by player_name (ilike)
  Fallthrough: skip when all three fail

Also verifies:
  - empty team stored as "Unknown" (not "")
  - pitcher INSERT gets positions=["SP"]; batter gets ["Util"]
  - positions excluded from on_conflict_do_update set_ (Yahoo-synced positions preserved)
"""
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from backend.services.daily_ingestion import DailyIngestionOrchestrator as IngestionOrchestrator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _bat_df(player_id="mike trout", name="Mike Trout", team="LAA"):
    return pd.DataFrame([{
        "player_id": player_id, "name": name, "team": team,
        "HR": 30.0, "R": 90.0, "RBI": 80.0, "SB": 5.0, "SO": 100.0,
        "AVG": 0.300, "OPS": 0.900, "SLG": 0.500, "PA": 550.0,
    }])


def _pit_df(player_id="gerrit cole", name="Gerrit Cole", team="NYY"):
    return pd.DataFrame([{
        "player_id": player_id, "name": name, "team": team,
        "W": 14.0, "SV": 0.0, "SO": 200.0, "ERA": 3.20, "WHIP": 1.05,
        "GS": 30.0, "K/9": 10.5, "IP": 180.0,
    }])


def _fill_cat_scores(batters, pitchers):
    for p in batters:
        p["cat_scores"] = {"hr": 1.5, "r": 1.2, "rbi": 0.8, "sb": 0.3, "avg": 1.1, "ops": 0.9}
    for p in pitchers:
        p["cat_scores"] = {"era": -0.5, "whip": -0.3, "k_per_nine": 1.2}


def _make_orch():
    orch = IngestionOrchestrator.__new__(IngestionOrchestrator)
    orch._record_job_run = MagicMock()
    return orch


class _Capturing:
    """Intercepts pg_insert calls and records INSERT values + conflict sets."""

    def __init__(self):
        self.inserts: list = []
        self.conflict_sets: list = []

    def pg_insert(self, table):
        cap = self
        stmt = MagicMock()
        insert_vals: dict = {}

        def capture_values(**kwargs):
            nonlocal insert_vals
            insert_vals = dict(kwargs)
            return stmt

        def capture_ocu(index_elements=None, set_=None):
            cap.inserts.append(dict(insert_vals))
            cap.conflict_sets.append(dict(set_ or {}))
            return stmt

        stmt.values = capture_values
        stmt.on_conflict_do_update = capture_ocu
        return stmt


async def _passthrough_lock(lock_id, job_name, coro):
    return await coro()


def _make_db(id_rows, proj_row):
    """Mock DB session: first query() → PlayerIDMapping; subsequent → PlayerProjection."""
    db = MagicMock()

    id_query = MagicMock()
    id_query.filter.return_value = id_query
    id_query.all.return_value = id_rows

    proj_query = MagicMock()
    proj_query.filter.return_value = proj_query
    proj_query.first.return_value = proj_row

    db.query.side_effect = [id_query] + [proj_query] * 10
    db.begin_nested.return_value.__enter__ = MagicMock(return_value=None)
    db.begin_nested.return_value.__exit__ = MagicMock(return_value=False)
    return db


def _id_row(normalized_name, mlbam_id):
    r = MagicMock()
    r.normalized_name = normalized_name
    r.mlbam_id = mlbam_id
    return r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_primary_mlbam_lookup_upserts_player():
    """Primary path: fg_id matches normalized_name in PlayerIDMapping → upserted."""
    cap = _Capturing()
    orch = _make_orch()
    db = _make_db(id_rows=[_id_row("mike trout", 545361)], proj_row=None)

    with patch("backend.services.daily_ingestion._ROS_CACHE", {"bat": True, "pit": None}), \
         patch("backend.fantasy_baseball.fangraphs_loader.compute_ensemble_blend",
               return_value=_bat_df()), \
         patch("backend.fantasy_baseball.player_board._compute_zscores",
               side_effect=_fill_cat_scores), \
         patch("backend.services.daily_ingestion.SessionLocal", return_value=db), \
         patch("backend.services.daily_ingestion._with_advisory_lock",
               side_effect=_passthrough_lock), \
         patch("backend.services.daily_ingestion.pg_insert", side_effect=cap.pg_insert):

        result = await orch._update_projection_cat_scores()

    assert result["upserted"] == 1
    assert result["skipped"] == 0
    assert cap.inserts[0]["player_id"] == "545361"


@pytest.mark.asyncio
async def test_secondary_name_lookup_resolves_mismatched_fg_id():
    """Secondary path: fg_id doesn't match but name.lower().strip() does → upserted."""
    cap = _Capturing()
    orch = _make_orch()
    # MLBAM dict keyed by "mike trout"; fg_id in DataFrame is "trout, m"
    db = _make_db(id_rows=[_id_row("mike trout", 545361)], proj_row=None)

    with patch("backend.services.daily_ingestion._ROS_CACHE", {"bat": True, "pit": None}), \
         patch("backend.fantasy_baseball.fangraphs_loader.compute_ensemble_blend",
               return_value=_bat_df(player_id="trout, m", name="Mike Trout")), \
         patch("backend.fantasy_baseball.player_board._compute_zscores",
               side_effect=_fill_cat_scores), \
         patch("backend.services.daily_ingestion.SessionLocal", return_value=db), \
         patch("backend.services.daily_ingestion._with_advisory_lock",
               side_effect=_passthrough_lock), \
         patch("backend.services.daily_ingestion.pg_insert", side_effect=cap.pg_insert):

        result = await orch._update_projection_cat_scores()

    assert result["upserted"] == 1
    assert cap.inserts[0]["player_id"] == "545361"


@pytest.mark.asyncio
async def test_tertiary_projection_fallback_uses_existing_row():
    """Tertiary path: MLBAM dict empty; PlayerIdentity row found by normalized name → uses mlbam_id."""
    cap = _Capturing()
    orch = _make_orch()
    existing = MagicMock()
    existing.mlbam_id = "mlbam_fallback_999"
    db = _make_db(id_rows=[], proj_row=existing)

    with patch("backend.services.daily_ingestion._ROS_CACHE", {"bat": True, "pit": None}), \
         patch("backend.fantasy_baseball.fangraphs_loader.compute_ensemble_blend",
               return_value=_bat_df()), \
         patch("backend.fantasy_baseball.player_board._compute_zscores",
               side_effect=_fill_cat_scores), \
         patch("backend.services.daily_ingestion.SessionLocal", return_value=db), \
         patch("backend.services.daily_ingestion._with_advisory_lock",
               side_effect=_passthrough_lock), \
         patch("backend.services.daily_ingestion.pg_insert", side_effect=cap.pg_insert):

        result = await orch._update_projection_cat_scores()

    assert result["upserted"] == 1
    assert cap.inserts[0]["player_id"] == "mlbam_fallback_999"


@pytest.mark.asyncio
async def test_all_fallbacks_fail_player_is_skipped():
    """All three lookup paths fail → player skipped (upserted=0, skipped=1)."""
    cap = _Capturing()
    orch = _make_orch()
    db = _make_db(id_rows=[], proj_row=None)

    with patch("backend.services.daily_ingestion._ROS_CACHE", {"bat": True, "pit": None}), \
         patch("backend.fantasy_baseball.fangraphs_loader.compute_ensemble_blend",
               return_value=_bat_df()), \
         patch("backend.fantasy_baseball.player_board._compute_zscores",
               side_effect=_fill_cat_scores), \
         patch("backend.services.daily_ingestion.SessionLocal", return_value=db), \
         patch("backend.services.daily_ingestion._with_advisory_lock",
               side_effect=_passthrough_lock), \
         patch("backend.services.daily_ingestion.pg_insert", side_effect=cap.pg_insert):

        result = await orch._update_projection_cat_scores()

    assert result["upserted"] == 0
    assert result["skipped"] == 1
    assert len(cap.inserts) == 0


@pytest.mark.asyncio
async def test_empty_team_stored_as_unknown():
    """Player with empty team string → upserted row has team='Unknown', not ''."""
    cap = _Capturing()
    orch = _make_orch()
    db = _make_db(id_rows=[_id_row("mike trout", 545361)], proj_row=None)

    with patch("backend.services.daily_ingestion._ROS_CACHE", {"bat": True, "pit": None}), \
         patch("backend.fantasy_baseball.fangraphs_loader.compute_ensemble_blend",
               return_value=_bat_df(team="")), \
         patch("backend.fantasy_baseball.player_board._compute_zscores",
               side_effect=_fill_cat_scores), \
         patch("backend.services.daily_ingestion.SessionLocal", return_value=db), \
         patch("backend.services.daily_ingestion._with_advisory_lock",
               side_effect=_passthrough_lock), \
         patch("backend.services.daily_ingestion.pg_insert", side_effect=cap.pg_insert):

        result = await orch._update_projection_cat_scores()

    assert result["upserted"] == 1
    assert cap.inserts[0]["team"] == "Unknown"


@pytest.mark.asyncio
async def test_pitcher_gets_sp_default_positions_not_in_conflict_set():
    """New pitcher → positions=['SP'] on INSERT; positions absent from conflict UPDATE set."""
    cap = _Capturing()
    orch = _make_orch()
    db = _make_db(id_rows=[_id_row("gerrit cole", 543037)], proj_row=None)

    with patch("backend.services.daily_ingestion._ROS_CACHE", {"bat": None, "pit": True}), \
         patch("backend.fantasy_baseball.fangraphs_loader.compute_ensemble_blend",
               return_value=_pit_df()), \
         patch("backend.fantasy_baseball.player_board._compute_zscores",
               side_effect=_fill_cat_scores), \
         patch("backend.services.daily_ingestion.SessionLocal", return_value=db), \
         patch("backend.services.daily_ingestion._with_advisory_lock",
               side_effect=_passthrough_lock), \
         patch("backend.services.daily_ingestion.pg_insert", side_effect=cap.pg_insert):

        result = await orch._update_projection_cat_scores()

    assert result["upserted"] == 1
    assert cap.inserts[0]["positions"] == ["SP"]
    assert "positions" not in cap.conflict_sets[0]


@pytest.mark.asyncio
async def test_batter_gets_util_default_positions_not_in_conflict_set():
    """New batter → positions=['Util'] on INSERT; positions absent from conflict UPDATE set."""
    cap = _Capturing()
    orch = _make_orch()
    db = _make_db(id_rows=[_id_row("mike trout", 545361)], proj_row=None)

    with patch("backend.services.daily_ingestion._ROS_CACHE", {"bat": True, "pit": None}), \
         patch("backend.fantasy_baseball.fangraphs_loader.compute_ensemble_blend",
               return_value=_bat_df()), \
         patch("backend.fantasy_baseball.player_board._compute_zscores",
               side_effect=_fill_cat_scores), \
         patch("backend.services.daily_ingestion.SessionLocal", return_value=db), \
         patch("backend.services.daily_ingestion._with_advisory_lock",
               side_effect=_passthrough_lock), \
         patch("backend.services.daily_ingestion.pg_insert", side_effect=cap.pg_insert):

        result = await orch._update_projection_cat_scores()

    assert result["upserted"] == 1
    assert cap.inserts[0]["positions"] == ["Util"]
    assert "positions" not in cap.conflict_sets[0]
