"""Contract test for GET /admin/mlb-odds/status.

Proves the MLB odds health endpoint reports the DailyIngestionOrchestrator
`mlb_odds` job state (BDL pipeline) — NOT the legacy CBB OddsMonitor
(/admin/odds-monitor/status), which is inactive when CBB_SEASON_ACTIVE is
unset/false and would falsely report "Last Poll: Never" / 0 games for the MLB
fantasy app.
"""
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.auth import verify_api_key
from backend.models import get_db


def _mock_db(games=14, snaps=84):
    db = MagicMock()
    row = MagicMock()
    row.games = games
    row.snaps = snaps
    db.execute.return_value.fetchone.return_value = row
    return db


def _orchestrator(last_status="success"):
    orch = MagicMock()
    orch.get_status.return_value = {
        "mlb_odds": {
            "name": "mlb_odds",
            "enabled": True,
            "last_status": last_status,
            "last_run": "2026-07-28T13:31:49-04:00",
            "next_run": "2026-07-28T13:36:48-04:00",
            "records_processed": 84,
        },
        # other jobs present but must be ignored
        "probable_pitchers_morning": {"last_status": "success"},
    }
    return orch


def _client(db):
    app.dependency_overrides[verify_api_key] = lambda: "test_user"
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def test_reports_mlb_odds_job_state_when_active():
    """Endpoint sources active/last_poll/games from the mlb_odds job + odds data."""
    db = _mock_db(games=14, snaps=84)
    with patch("backend.main._ingestion_orchestrator", _orchestrator("success")):
        resp = _client(db).get("/admin/mlb-odds/status", headers={"X-API-Key": "k"})

    assert resp.status_code == 200, resp.text
    data = resp.json()
    # Provenance: this is the MLB pipeline, not the legacy CBB monitor.
    assert data["source"] == "mlb_odds"
    # Job state drives active + last_poll (the CBB monitor would give null/Never).
    assert data["active"] is True
    assert data["last_status"] == "success"
    assert data["last_poll"] == "2026-07-28T13:31:49-04:00"
    assert data["next_run"] == "2026-07-28T13:36:48-04:00"
    # Real MLB odds data drives games_tracked (CBB monitor reports 0 off-season).
    assert data["games_tracked"] == 14
    assert data["snapshots_today"] == 84
    # BDL is not OddsAPI-quota-limited.
    assert data["quota_remaining"] is None
    assert data["quota_is_low"] is False


def test_inactive_when_last_job_failed():
    """A failed/never-run mlb_odds job reports active:false even if rows exist."""
    db = _mock_db(games=0, snaps=0)
    with patch("backend.main._ingestion_orchestrator", _orchestrator("error")):
        resp = _client(db).get("/admin/mlb-odds/status", headers={"X-API-Key": "k"})

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["source"] == "mlb_odds"
    assert data["active"] is False
    assert data["games_tracked"] == 0
