"""
Tests for the statsapi supplement pipeline (PR #91 data quality audit findings).

Covers three areas identified as completely untested:
  1. _patch_counting_stats_batter  — including the strikeouts key bug fix
  2. _patch_counting_stats_pitcher
  3. _supplement_statsapi_counting_stats — async job, exercised via mocks

All tests use unittest.mock; no real DB or real API calls are made.
"""

import asyncio
import sys
import types
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal module stubs so daily_ingestion imports without a live DB /
# APScheduler.  This mirrors the pattern used in test_ingestion_orchestrator.py.
# ---------------------------------------------------------------------------

for _mod in (
    "apscheduler",
    "apscheduler.schedulers",
    "apscheduler.schedulers.asyncio",
    "apscheduler.triggers",
    "apscheduler.triggers.cron",
    "apscheduler.triggers.interval",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

if not hasattr(sys.modules["apscheduler.schedulers.asyncio"], "AsyncIOScheduler"):
    class _FakeScheduler:
        def __init__(self):
            self._jobs = {}
            self.running = False

        def add_job(self, func, trigger, id=None, name=None, replace_existing=False):
            self._jobs[id] = MagicMock(id=id, name=name, next_run_time=None)

        def get_job(self, job_id):
            return self._jobs.get(job_id)

        def get_jobs(self):
            return list(self._jobs.values())

        def start(self):
            self.running = True

    sys.modules["apscheduler.schedulers.asyncio"].AsyncIOScheduler = _FakeScheduler

if not hasattr(sys.modules["apscheduler.triggers.cron"], "CronTrigger"):
    sys.modules["apscheduler.triggers.cron"].CronTrigger = MagicMock
if not hasattr(sys.modules["apscheduler.triggers.interval"], "IntervalTrigger"):
    sys.modules["apscheduler.triggers.interval"].IntervalTrigger = MagicMock

# ---------------------------------------------------------------------------
# Import targets (after stubs are in place)
# ---------------------------------------------------------------------------

from backend.services.daily_ingestion import DailyIngestionOrchestrator  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _batter_row(**kwargs):
    """SimpleNamespace with every batter DB column, defaulting to None."""
    defaults = dict(
        ab=None, runs=None, hits=None, doubles=None, triples=None,
        home_runs=None, rbi=None, walks=None, strikeouts_bat=None,
        stolen_bases=None, caught_stealing=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _pitcher_row(**kwargs):
    """SimpleNamespace with every pitcher DB column, defaulting to None."""
    defaults = dict(
        hits_allowed=None, earned_runs=None, walks_allowed=None,
        strikeouts_pit=None, runs_allowed=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_orchestrator():
    """Create a DailyIngestionOrchestrator without triggering __init__."""
    orch = DailyIngestionOrchestrator.__new__(DailyIngestionOrchestrator)
    orch._scheduler = MagicMock()
    orch._record_job_run = MagicMock()
    return orch


# ---------------------------------------------------------------------------
# 1. _patch_counting_stats_batter
# ---------------------------------------------------------------------------

class TestPatchCountingStatsBatter:

    def test_patches_ab_from_boxscore(self):
        row = _batter_row()
        result = DailyIngestionOrchestrator._patch_counting_stats_batter(
            row,
            {"ab": 4, "r": 1, "h": 2, "doubles": 0, "triples": 0,
             "hr": 1, "rbi": 2, "bb": 0, "strikeouts": 1, "sb": 0, "cs": 0},
        )
        assert result is True
        assert row.ab == 4

    def test_patches_all_null_fields(self):
        row = _batter_row()
        DailyIngestionOrchestrator._patch_counting_stats_batter(
            row,
            {"ab": 3, "r": 0, "h": 1, "doubles": 0, "triples": 0,
             "hr": 0, "rbi": 0, "bb": 1, "strikeouts": 2, "sb": 0, "cs": 0},
        )
        assert row.ab == 3
        assert row.hits == 1
        assert row.walks == 1
        assert row.strikeouts_bat == 2

    def test_strikeouts_bat_uses_strikeouts_key_not_k(self):
        """
        PR #91 bug fix: batter strikeouts must come from the 'strikeouts' key,
        NOT from 'k' (which is the pitcher convention in statsapi).
        """
        # Providing only the old wrong key 'k' must leave strikeouts_bat as None
        row_old = _batter_row()
        DailyIngestionOrchestrator._patch_counting_stats_batter(row_old, {"k": 3})
        assert row_old.strikeouts_bat is None, (
            "strikeouts_bat must NOT be populated from 'k' key (pitcher convention)"
        )

        # Providing the correct key 'strikeouts' must patch the field
        row_new = _batter_row()
        DailyIngestionOrchestrator._patch_counting_stats_batter(row_new, {"strikeouts": 3})
        assert row_new.strikeouts_bat == 3, (
            "strikeouts_bat must be populated from 'strikeouts' key"
        )

    def test_does_not_overwrite_existing_values(self):
        row = _batter_row(ab=5, hits=2)
        DailyIngestionOrchestrator._patch_counting_stats_batter(
            row, {"ab": 99, "h": 99, "r": 1}
        )
        assert row.ab == 5      # unchanged — was not None
        assert row.hits == 2    # unchanged — was not None
        assert row.runs == 1    # patched   — was None

    def test_returns_false_when_boxscore_empty(self):
        row = _batter_row()
        assert DailyIngestionOrchestrator._patch_counting_stats_batter(row, {}) is False

    def test_returns_false_when_all_columns_already_populated(self):
        row = _batter_row(
            ab=4, runs=1, hits=2, doubles=0, triples=0, home_runs=1,
            rbi=2, walks=0, strikeouts_bat=1, stolen_bases=0, caught_stealing=0,
        )
        box = {
            "ab": 99, "r": 99, "h": 99, "doubles": 99, "triples": 99,
            "hr": 99, "rbi": 99, "bb": 99, "strikeouts": 99, "sb": 99, "cs": 99,
        }
        assert DailyIngestionOrchestrator._patch_counting_stats_batter(row, box) is False

    def test_invalid_value_skipped_gracefully(self):
        row = _batter_row()
        DailyIngestionOrchestrator._patch_counting_stats_batter(
            row, {"ab": "not_a_number"}
        )
        assert row.ab is None   # bad value skipped, no crash


# ---------------------------------------------------------------------------
# 2. _patch_counting_stats_pitcher
# ---------------------------------------------------------------------------

class TestPatchCountingStatsPitcher:

    def test_patches_all_pitcher_fields(self):
        row = _pitcher_row()
        result = DailyIngestionOrchestrator._patch_counting_stats_pitcher(
            row, {"h": 4, "er": 2, "bb": 1, "k": 7, "r": 2}
        )
        assert result is True
        assert row.hits_allowed == 4
        assert row.earned_runs == 2
        assert row.walks_allowed == 1
        assert row.strikeouts_pit == 7
        assert row.runs_allowed == 2

    def test_does_not_overwrite_existing_values(self):
        row = _pitcher_row(hits_allowed=3)
        DailyIngestionOrchestrator._patch_counting_stats_pitcher(
            row, {"h": 99, "er": 1, "bb": 0, "k": 5, "r": 1}
        )
        assert row.hits_allowed == 3    # unchanged
        assert row.earned_runs == 1     # patched from None

    def test_returns_false_when_boxscore_empty(self):
        row = _pitcher_row()
        assert DailyIngestionOrchestrator._patch_counting_stats_pitcher(row, {}) is False

    def test_uses_k_key_for_strikeouts_pit(self):
        """Pitcher strikeouts correctly read from 'k' statsapi key."""
        row = _pitcher_row()
        DailyIngestionOrchestrator._patch_counting_stats_pitcher(row, {"k": 9})
        assert row.strikeouts_pit == 9

    def test_invalid_value_skipped_gracefully(self):
        row = _pitcher_row()
        DailyIngestionOrchestrator._patch_counting_stats_pitcher(row, {"k": "x"})
        assert row.strikeouts_pit is None


# ---------------------------------------------------------------------------
# 3. _supplement_statsapi_counting_stats (async integration tests)
# ---------------------------------------------------------------------------

class TestSupplementStatsApiCountingStats:
    """
    The supplement job is exercised end-to-end against mocked DB + statsapi
    to cover the code path flagged as completely untested in PR #91.
    """

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_skipped_when_statsapi_not_installed(self):
        """When the statsapi package is absent the job returns status='skipped'."""
        orch = _make_orchestrator()

        # Remove statsapi from sys.modules so the import inside the job fails
        statsapi_backup = sys.modules.pop("statsapi", None)
        try:
            async def fake_lock(lock_id, name, fn):
                return await fn()

            with patch("backend.services.daily_ingestion._with_advisory_lock",
                       side_effect=fake_lock):
                result = self._run(orch._supplement_statsapi_counting_stats())
        finally:
            if statsapi_backup is not None:
                sys.modules["statsapi"] = statsapi_backup

        assert result["status"] == "skipped"
        assert result["records"] == 0

    def test_returns_success_zero_records_when_no_rows_need_fill(self):
        """No ab=NULL rows → 0 records patched, status success."""
        orch = _make_orchestrator()

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.all.return_value = []
        mock_db.__enter__ = MagicMock(return_value=mock_db)
        mock_db.__exit__ = MagicMock(return_value=False)

        statsapi_stub = types.ModuleType("statsapi")

        async def fake_lock(lock_id, name, fn):
            return await fn()

        with patch("backend.services.daily_ingestion._with_advisory_lock",
                   side_effect=fake_lock), \
             patch("backend.services.daily_ingestion.SessionLocal",
                   return_value=mock_db), \
             patch.dict(sys.modules, {"statsapi": statsapi_stub}):
            result = self._run(orch._supplement_statsapi_counting_stats())

        assert result["status"] == "success"
        assert result["records"] == 0

    def test_skips_non_final_games(self):
        """Games with status other than Final/Game Over are not processed."""
        orch = _make_orchestrator()
        target_date = date(2026, 4, 10)

        fake_row = _batter_row()
        fake_row.game_date = target_date
        fake_row.raw_payload = {"player": {"full_name": "Mike Trout"}}

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.all.return_value = [fake_row]
        mock_db.__enter__ = MagicMock(return_value=mock_db)
        mock_db.__exit__ = MagicMock(return_value=False)

        statsapi_stub = types.ModuleType("statsapi")
        statsapi_stub.schedule = MagicMock(
            return_value=[{"game_id": 888, "status": "In Progress"}]
        )
        statsapi_stub.boxscore_data = MagicMock()

        async def fake_lock(lock_id, name, fn):
            return await fn()

        async def fake_to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        with patch("backend.services.daily_ingestion._with_advisory_lock",
                   side_effect=fake_lock), \
             patch("backend.services.daily_ingestion.SessionLocal",
                   return_value=mock_db), \
             patch("backend.services.daily_ingestion.today_et",
                   return_value=target_date), \
             patch("asyncio.to_thread", side_effect=fake_to_thread), \
             patch.dict(sys.modules, {"statsapi": statsapi_stub}):
            result = self._run(orch._supplement_statsapi_counting_stats())

        assert result["status"] == "success"
        assert result["records"] == 0
        assert fake_row.ab is None          # game was not final → no patch
        statsapi_stub.boxscore_data.assert_not_called()

    def test_patches_batter_row_from_final_game(self):
        """
        When a db row with ab=NULL exists and the statsapi boxscore returns a
        matching batter, the row must be patched and records count must be 1.
        """
        orch = _make_orchestrator()
        target_date = date(2026, 4, 10)

        fake_row = _batter_row()
        fake_row.game_date = target_date
        fake_row.raw_payload = {"player": {"full_name": "Jose Ramirez"}}

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.all.return_value = [fake_row]
        mock_db.__enter__ = MagicMock(return_value=mock_db)
        mock_db.__exit__ = MagicMock(return_value=False)

        statsapi_stub = types.ModuleType("statsapi")
        statsapi_stub.schedule = MagicMock(
            return_value=[{"game_id": 999, "status": "Final"}]
        )
        statsapi_stub.boxscore_data = MagicMock(return_value={
            "playerInfo": {"ID101": {"fullName": "Jose Ramirez"}},
            "awayBatters": [
                {
                    "personId": 101,
                    "ab": 4, "r": 1, "h": 2, "doubles": 0, "triples": 0,
                    "hr": 0, "rbi": 1, "bb": 0, "strikeouts": 1, "sb": 0, "cs": 0,
                }
            ],
            "homeBatters": [],
            "awayPitchers": [],
            "homePitchers": [],
        })

        async def fake_lock(lock_id, name, fn):
            return await fn()

        async def fake_to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        with patch("backend.services.daily_ingestion._with_advisory_lock",
                   side_effect=fake_lock), \
             patch("backend.services.daily_ingestion.SessionLocal",
                   return_value=mock_db), \
             patch("backend.services.daily_ingestion.today_et",
                   return_value=target_date), \
             patch("asyncio.to_thread", side_effect=fake_to_thread), \
             patch.dict(sys.modules, {"statsapi": statsapi_stub}):
            result = self._run(orch._supplement_statsapi_counting_stats())

        assert result["status"] == "success"
        assert result["records"] == 1
        assert fake_row.ab == 4
        assert fake_row.hits == 2
        assert fake_row.strikeouts_bat == 1   # populated via 'strikeouts' key (bug fix)
        mock_db.commit.assert_called_once()

    def test_duplicate_upsert_path_second_batter_row_skipped(self):
        """
        PR #91 structural weakness: the duplicate upsert path on
        (bdl_player_id, game_id) was never exercised.

        Scenario: same player name appears twice in name_lookup (e.g., two
        unresolved rows with the same full_name on the same date).  The first
        row has ab=None (needs fill); the second has every field already set.
        The patcher must skip the second row and record only 1 patch.
        """
        orch = _make_orchestrator()
        target_date = date(2026, 4, 10)

        row1 = _batter_row()
        row1.game_date = target_date
        row1.raw_payload = {"player": {"full_name": "Aaron Judge"}}

        # row2 has all batter fields populated — nothing should be overwritten
        row2 = _batter_row(
            ab=5, runs=0, hits=1, doubles=0, triples=0, home_runs=0,
            rbi=0, walks=0, strikeouts_bat=2, stolen_bases=0, caught_stealing=0,
        )
        row2.game_date = target_date
        row2.raw_payload = {"player": {"full_name": "Aaron Judge"}}

        mock_db = MagicMock()
        # Both rows are returned even though row2 already has ab set, so we can
        # verify the patcher's behaviour when it encounters a fully-populated row.
        mock_db.query.return_value.filter.return_value.all.return_value = [row1, row2]

        statsapi_stub = types.ModuleType("statsapi")
        statsapi_stub.schedule = MagicMock(
            return_value=[{"game_id": 777, "status": "Final"}]
        )
        statsapi_stub.boxscore_data = MagicMock(return_value={
            "playerInfo": {"ID202": {"fullName": "Aaron Judge"}},
            "awayBatters": [
                {
                    "personId": 202,
                    "ab": 4, "r": 0, "h": 1, "doubles": 0, "triples": 0,
                    "hr": 0, "rbi": 0, "bb": 0, "strikeouts": 2, "sb": 0, "cs": 0,
                }
            ],
            "homeBatters": [],
            "awayPitchers": [],
            "homePitchers": [],
        })

        async def fake_lock(lock_id, name, fn):
            return await fn()

        async def fake_to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        with patch("backend.services.daily_ingestion._with_advisory_lock",
                   side_effect=fake_lock), \
             patch("backend.services.daily_ingestion.SessionLocal",
                   return_value=mock_db), \
             patch("backend.services.daily_ingestion.today_et",
                   return_value=target_date), \
             patch("asyncio.to_thread", side_effect=fake_to_thread), \
             patch.dict(sys.modules, {"statsapi": statsapi_stub}):
            result = self._run(orch._supplement_statsapi_counting_stats())

        assert result["status"] == "success"
        # row1 was fully patched (all fields were None); row2 had everything set
        # so _patch_counting_stats_batter returned False for it.
        assert result["records"] == 1
        assert row1.ab == 4
        assert row2.ab == 5    # not overwritten
