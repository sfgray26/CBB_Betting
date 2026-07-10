# -*- coding: utf-8 -*-
"""Tests for backend/services/projection_coverage.py."""
from datetime import date
from unittest.mock import MagicMock, patch

from backend.services.projection_coverage import compute_roster_projection_coverage


def _roster(*entries):
    return [{"player_key": k, "name": n} for k, n in entries]


def _db_with_latest_scores(latest_by_bdl: dict):
    """Mock Session whose player_scores group-by query returns latest_by_bdl."""
    db = MagicMock()
    rows = []
    for bdl_id, max_date in latest_by_bdl.items():
        row = MagicMock()
        row.bdl_player_id = bdl_id
        row.max_date = max_date
        rows.append(row)
    db.query.return_value.filter.return_value.group_by.return_value.all.return_value = rows
    return db


TARGET = "2026-07-10"
FRESH = date(2026, 7, 10)
OLD = date(2026, 6, 20)


class TestComputeRosterProjectionCoverage:
    def test_all_covered_is_green(self):
        roster = _roster(("469.p.1", "Juan Soto"), ("469.p.2", "Cristopher Sánchez"))
        resolve = {"469.p.1": {"bdl_id": 1106}, "469.p.2": {"bdl_id": 40}}
        db = _db_with_latest_scores({1106: FRESH, 40: FRESH})

        with patch("backend.routers.fantasy._resolve_roster_player_bdl_ids", return_value=resolve):
            report = compute_roster_projection_coverage(db, roster, TARGET)

        assert report["status"] == "green"
        assert report["coverage_pct"] == 100.0
        assert report["missing"] == []

    def test_missing_mapping_flags_red(self):
        roster = _roster(("469.p.1", "Juan Soto"), ("469.p.9", "Mystery Man"))
        resolve = {"469.p.1": {"bdl_id": 1106}}  # p.9 unresolvable
        db = _db_with_latest_scores({1106: FRESH})

        with patch("backend.routers.fantasy._resolve_roster_player_bdl_ids", return_value=resolve):
            report = compute_roster_projection_coverage(db, roster, TARGET)

        assert report["status"] == "red"  # 50% < 90
        assert report["missing"][0]["coverage"] == "missing_mapping"
        assert report["missing"][0]["name"] == "Mystery Man"

    def test_missing_scores_tries_workaround_then_flags(self):
        roster = _roster(("469.p.2", "Cristopher Sánchez"))
        resolve = {"469.p.2": {"bdl_id": 40, "mlbam_id": None}}
        db = _db_with_latest_scores({})  # no scores at all

        with patch("backend.routers.fantasy._resolve_roster_player_bdl_ids", return_value=resolve), \
             patch("backend.services.player_id_resolver.find_alternative_player_score",
                   return_value=(None, "default")) as workaround:
            report = compute_roster_projection_coverage(db, roster, TARGET)

        workaround.assert_called_once()
        assert report["status"] == "red"
        assert report["missing"][0]["coverage"] == "missing_scores"

    def test_workaround_hit_counts_as_covered(self):
        roster = _roster(("469.p.2", "Cristopher Sánchez"))
        resolve = {"469.p.2": {"bdl_id": 676979, "mlbam_id": 676979}}
        db = _db_with_latest_scores({})

        with patch("backend.routers.fantasy._resolve_roster_player_bdl_ids", return_value=resolve), \
             patch("backend.services.player_id_resolver.find_alternative_player_score",
                   return_value=(88.5, "player_scores")):
            report = compute_roster_projection_coverage(db, roster, TARGET)

        assert report["status"] == "green"
        assert report["players"][0]["coverage"] == "covered_workaround"

    def test_stale_scores_flagged_but_yellow_at_90_pct(self):
        entries = [(f"469.p.{i}", f"Player {i}") for i in range(10)]
        roster = _roster(*entries)
        resolve = {f"469.p.{i}": {"bdl_id": i + 100} for i in range(10)}
        latest = {i + 100: FRESH for i in range(9)}
        latest[109] = OLD  # one stale player
        db = _db_with_latest_scores(latest)

        with patch("backend.routers.fantasy._resolve_roster_player_bdl_ids", return_value=resolve):
            report = compute_roster_projection_coverage(db, roster, TARGET)

        assert report["coverage_pct"] == 90.0
        assert report["status"] == "yellow"
        assert report["missing"][0]["coverage"] == "stale"

    def test_empty_roster_is_green(self):
        db = _db_with_latest_scores({})
        with patch("backend.routers.fantasy._resolve_roster_player_bdl_ids", return_value={}):
            report = compute_roster_projection_coverage(db, [], TARGET)
        assert report["status"] == "green"
        assert report["total"] == 0
