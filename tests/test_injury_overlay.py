from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from backend.services.injury_overlay import apply_injury_penalty, build_injury_overlay


_ET = ZoneInfo("America/New_York")


def test_build_injury_overlay_includes_eta_and_freshness_for_fresh_rows():
    now_et = datetime(2026, 5, 19, 12, 0, tzinfo=_ET)
    overlay = build_injury_overlay(
        status="15-Day-IL",
        note="Hamstring strain",
        return_date=datetime(2026, 5, 24, 0, 0, tzinfo=_ET),
        ingested_at=now_et - timedelta(minutes=42),
        now_et=now_et,
        freshness_minutes=180,
    )

    assert overlay.status == "15-Day-IL"
    assert overlay.note == "Hamstring strain"
    assert overlay.is_stale is False
    assert "ETA" in (overlay.return_timeline or "")
    assert "updated" in (overlay.return_timeline or "").lower()


def test_build_injury_overlay_marks_rows_stale_when_old():
    now_et = datetime(2026, 5, 19, 12, 0, tzinfo=_ET)
    overlay = build_injury_overlay(
        status="DTD",
        note="Day-to-day",
        return_date=None,
        ingested_at=now_et - timedelta(hours=7),
        now_et=now_et,
        freshness_minutes=180,
    )

    assert overlay.is_stale is True
    assert "stale" in (overlay.return_timeline or "").lower()


def test_apply_injury_penalty_discounts_il_more_than_dtd():
    now_et = datetime(2026, 5, 19, 12, 0, tzinfo=_ET)
    il_overlay = build_injury_overlay(
        status="15-Day-IL",
        note="Hamstring strain",
        return_date=None,
        ingested_at=now_et - timedelta(minutes=15),
        now_et=now_et,
    )
    dtd_overlay = build_injury_overlay(
        status="DTD",
        note="Sore wrist",
        return_date=None,
        ingested_at=now_et - timedelta(minutes=15),
        now_et=now_et,
    )

    il_score, il_note = apply_injury_penalty(2.0, il_overlay)
    dtd_score, dtd_note = apply_injury_penalty(2.0, dtd_overlay)

    assert il_score < dtd_score < 2.0
    assert "15-Day-IL" in (il_note or "")
    assert "DTD" in (dtd_note or "")
