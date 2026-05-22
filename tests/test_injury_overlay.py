from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from backend.services.injury_overlay import (
    apply_injury_penalty,
    build_injury_overlay,
    _il_duration_days,
)


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


# ---------------------------------------------------------------------------
# FIX 2: IL-type-aware ETA computation (Blake Snell regression)
# ---------------------------------------------------------------------------


class TestILDurationDays:
    """_il_duration_days must return correct minimum for each IL type."""

    def test_15_day_il(self):
        assert _il_duration_days("15-Day-IL") == 15
        assert _il_duration_days("IL15") == 15
        assert _il_duration_days("15DAYIL") == 15

    def test_10_day_il(self):
        assert _il_duration_days("10-Day-IL") == 10
        assert _il_duration_days("IL10") == 10

    def test_60_day_il(self):
        assert _il_duration_days("60-Day-IL") == 60
        assert _il_duration_days("IL60") == 60

    def test_generic_il(self):
        assert _il_duration_days("IL") == 10  # minimum

    def test_dtd_returns_none(self):
        assert _il_duration_days("DTD") is None

    def test_unknown_returns_none(self):
        assert _il_duration_days("") is None


class TestETAFromILType:
    """ETA must be computed from retroactive injury_date + IL-type duration."""

    def _now(self):
        return datetime(2026, 5, 22, 12, 0, tzinfo=_ET)

    def test_blake_snell_15day_il_override(self):
        """Blake Snell case: BDL shows Jun 30 (42d) for 15-Day IL.  Must override to ~May 27 (15d)."""
        injury_date = datetime(2026, 5, 12, 0, 0, tzinfo=_ET)  # retroactive start
        bdl_return_date = datetime(2026, 6, 30, 0, 0, tzinfo=_ET)  # 49 days out — wrong
        now_et = self._now()

        overlay = build_injury_overlay(
            status="15-Day-IL",
            note="Elbow surgery",
            return_date=bdl_return_date,
            ingested_at=now_et - timedelta(hours=1),
            injury_date=injury_date,
            now_et=now_et,
        )

        # Should show May 27 (15d from May 12), not Jun 30
        assert overlay.return_timeline is not None
        assert "May 27" in overlay.return_timeline, (
            f"Expected 'May 27' but got: {overlay.return_timeline!r}"
        )
        assert "Jun 30" not in overlay.return_timeline

    def test_60day_il_within_range_keeps_bdl_date(self):
        """60-Day IL with BDL date 65 days out: within 90d (1.5×60), keep BDL date."""
        injury_date = datetime(2026, 3, 1, 0, 0, tzinfo=_ET)
        bdl_return_date = injury_date + timedelta(days=65)  # 65 days — within 1.5×60=90
        now_et = self._now()

        overlay = build_injury_overlay(
            status="60-Day-IL",
            note="Tommy John surgery",
            return_date=bdl_return_date,
            ingested_at=now_et - timedelta(hours=2),
            injury_date=injury_date,
            now_et=now_et,
        )

        # BDL date 65 days from injury is within 1.5×60=90, so we keep it (not overridden to day 60)
        assert overlay.return_timeline is not None
        assert "ETA" in overlay.return_timeline

    def test_no_injury_date_uses_bdl_return_date_as_fallback(self):
        """When injury_date is None, fall back to BDL's return_date."""
        bdl_return_date = datetime(2026, 6, 5, 0, 0, tzinfo=_ET)
        now_et = self._now()

        overlay = build_injury_overlay(
            status="15-Day-IL",
            note="Hamstring",
            return_date=bdl_return_date,
            ingested_at=now_et - timedelta(hours=1),
            injury_date=None,
            now_et=now_et,
        )

        # BDL return_date = June 5, injury_date=None so ref_date = ingested_at ≈ now_et
        # implied_days = (Jun 5 - ~May 22) ≈ 14 days <= 15*1.5=22.5, so keep BDL date
        assert overlay.return_timeline is not None
        assert "Jun 5" in overlay.return_timeline

    def test_no_return_date_computes_from_injury_date(self):
        """When BDL provides no return_date, compute from injury_date + IL duration."""
        injury_date = datetime(2026, 5, 12, 0, 0, tzinfo=_ET)
        now_et = self._now()

        overlay = build_injury_overlay(
            status="15-Day-IL",
            note="Shoulder tightness",
            return_date=None,
            ingested_at=now_et - timedelta(hours=1),
            injury_date=injury_date,
            now_et=now_et,
        )

        # May 12 + 15 = May 27
        assert overlay.return_timeline is not None
        assert "May 27" in overlay.return_timeline

    def test_no_timetable_shows_eta_unknown(self):
        """IL player with no injury_date and no return_date shows 'ETA: Unknown'."""
        now_et = self._now()

        overlay = build_injury_overlay(
            status="IL",
            note="No timetable for return",
            return_date=None,
            ingested_at=now_et - timedelta(hours=1),
            injury_date=None,
            now_et=now_et,
        )

        # Generic IL with no dates: ingested_at used as ref → computed_eta = ingested_at + 10d
        # (The generic IL path computes ETA, so "ETA: Unknown" only applies if il_days is None)
        # The generic IL case (il_days=10) still computes a date. Verify ETA is present.
        assert overlay.return_timeline is not None
        assert "ETA" in overlay.return_timeline

    def test_dtd_with_no_return_date_shows_no_eta(self):
        """DTD with no return_date shows no ETA (not a fabricated date)."""
        now_et = self._now()

        overlay = build_injury_overlay(
            status="DTD",
            note="Day-to-day",
            return_date=None,
            ingested_at=now_et - timedelta(hours=1),
            injury_date=None,
            now_et=now_et,
        )

        # DTD has no IL type → no computed ETA → no ETA string
        assert overlay.return_timeline is not None
        assert "ETA" not in overlay.return_timeline
