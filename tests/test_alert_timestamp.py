"""Regression: /api/performance/alerts must serialize the DB-stored (naive)
created_at with a correct UTC offset, not leave it naive / stamp it as ET.

Root cause (UAT 2026-07-28): DBAlert.created_at is a naive `DateTime`. The engine
sets no session timezone, so Postgres stores the UTC wall-clock and reads it back
naive. Serializing that bare — or stamping it as ET — made a just-created alert
render as ~4h in the FUTURE. db_ts_isoformat stamps UTC so the instant is correct.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.models import db_ts_isoformat, et_isoformat
from backend.main import app
from backend.auth import verify_api_key
from backend.models import get_db


def test_db_ts_isoformat_stamps_utc_for_naive():
    # A value stored as UTC wall-clock (17:31 UTC == 13:31 ET) must serialize with
    # a +00:00 offset — NOT an ET offset (which would read 4h in the future).
    out = db_ts_isoformat(datetime(2026, 7, 28, 17, 31, 0))
    assert out == "2026-07-28T17:31:00+00:00"
    # Legacy alias still resolves to the corrected behavior.
    assert et_isoformat(datetime(2026, 7, 28, 17, 31, 0)) == out
    # Aware values pass through unchanged.
    aware = datetime(2026, 7, 28, 13, 31, 0, tzinfo=timezone.utc)
    assert db_ts_isoformat(aware) == "2026-07-28T13:31:00+00:00"
    assert db_ts_isoformat(None) is None


def _alert(created_at):
    a = MagicMock()
    a.id = 1
    a.alert_type = "YAHOO_AUTH_OUTAGE"
    a.severity = "CRITICAL"
    a.message = "Yahoo auth circuit OPEN"
    a.threshold = 3.0
    a.current_value = 5.0
    a.acknowledged = False
    a.created_at = created_at
    return a


def test_alerts_endpoint_serializes_created_at_as_utc():
    """A naive created_at is returned with an explicit +00:00 offset (unambiguous),
    so the browser never renders it in the future."""
    naive = datetime(2026, 7, 28, 17, 31, 0)  # UTC wall-clock, no tzinfo
    db = MagicMock()
    chain = db.query.return_value.order_by.return_value
    chain.filter.return_value.limit.return_value.all.return_value = [_alert(naive)]
    chain.limit.return_value.all.return_value = [_alert(naive)]

    app.dependency_overrides[verify_api_key] = lambda: "test_user"
    app.dependency_overrides[get_db] = lambda: db
    try:
        with patch("backend.main.check_performance_alerts", return_value=[]):
            resp = TestClient(app).get("/api/performance/alerts")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200, resp.text
    alerts = resp.json()["alerts"]
    assert len(alerts) == 1
    created = alerts[0]["created_at"]
    assert created.endswith("+00:00"), created
    assert created == "2026-07-28T17:31:00+00:00"
