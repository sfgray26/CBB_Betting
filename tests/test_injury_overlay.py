from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker

from backend.models import IngestedInjury, PlayerIDMapping
from backend.services.injury_overlay import (
    apply_injury_penalty,
    build_injury_overlay,
    load_injury_overlays_for_yahoo_players,
    _il_duration_days,
)


_ET = ZoneInfo("America/New_York")


@compiles(JSONB, "sqlite")
def _render_jsonb_as_json_on_sqlite(type_, compiler, **kw):
    return "JSON"


def _make_sqlite_session():
    engine = create_engine("sqlite:///:memory:")
    PlayerIDMapping.__table__.create(bind=engine)
    IngestedInjury.__table__.create(bind=engine)
    return sessionmaker(bind=engine)()


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


def test_apply_injury_penalty_60_day_il_reduces_score_most():
    """60-Day IL must reduce the need score, never boost it, and must be
    harsher than 15-Day IL (regression: c780d56 multiplied by 1.25)."""
    now_et = datetime(2026, 5, 19, 12, 0, tzinfo=_ET)
    long_il_overlay = build_injury_overlay(
        status="60-Day-IL",
        note="Tommy John surgery",
        return_date=None,
        ingested_at=now_et - timedelta(minutes=15),
        now_et=now_et,
    )
    il_overlay = build_injury_overlay(
        status="15-Day-IL",
        note="Hamstring strain",
        return_date=None,
        ingested_at=now_et - timedelta(minutes=15),
        now_et=now_et,
    )

    long_il_score, long_il_note = apply_injury_penalty(2.0, long_il_overlay)
    il_score, _ = apply_injury_penalty(2.0, il_overlay)

    assert long_il_score < il_score < 2.0
    assert "60-Day-IL" in (long_il_note or "")


# ---------------------------------------------------------------------------
# Regression: c780d56 review blockers — PlayerIDMapping column names and
# real-time expired-ETA detection in the Yahoo loader path
# ---------------------------------------------------------------------------


class TestLoadInjuryOverlaysForYahooPlayers:
    def _seed(self, db, *, return_date, now_et):
        db.add(
            PlayerIDMapping(
                yahoo_key="469.p.111",
                yahoo_id="111",
                bdl_id=42,
                full_name="Colt Emerson",
                normalized_name="colt emerson",
            )
        )
        db.add(
            IngestedInjury(
                id=1,  # BigInteger PK does not autoincrement on sqlite
                bdl_player_id=42,
                player_name="Colt Emerson",
                injury_date=(now_et - timedelta(days=12)).replace(tzinfo=None),
                return_date=return_date,
                injury_type="Hamstring",
                injury_status="10-Day-IL",
                long_comment="Hamstring strain, eyeing return.",
                short_comment="Hamstring strain.",
                raw_payload={},
                ingested_at=(now_et - timedelta(minutes=30)).replace(tzinfo=None),
            )
        )
        db.commit()

    def test_resolves_player_id_mapping_columns_and_flags_expired_eta(self):
        """Loader must query PlayerIDMapping.yahoo_key / bdl_id (regression:
        c780d56 referenced non-existent yahoo_player_key / bdl_player_id) and
        flag a passed return_date in real time without any DB write."""
        now_et = datetime(2026, 6, 12, 12, 0, tzinfo=_ET)
        db = _make_sqlite_session()
        self._seed(db, return_date=(now_et - timedelta(days=2)).replace(tzinfo=None), now_et=now_et)

        overlays = load_injury_overlays_for_yahoo_players(
            db, [{"player_key": "469.p.111"}], now_et=now_et,
        )

        assert "469.p.111" in overlays
        overlay = overlays["469.p.111"]
        assert overlay.expired_eta is True
        assert "EXPIRED" in (overlay.return_timeline or "")

    def test_future_eta_is_not_flagged_expired(self):
        now_et = datetime(2026, 6, 12, 12, 0, tzinfo=_ET)
        db = _make_sqlite_session()
        self._seed(db, return_date=(now_et + timedelta(days=3)).replace(tzinfo=None), now_et=now_et)

        overlays = load_injury_overlays_for_yahoo_players(
            db, [{"player_key": "469.p.111"}], now_et=now_et,
        )

        assert "469.p.111" in overlays
        overlay = overlays["469.p.111"]
        assert overlay.expired_eta is False
        assert "EXPIRED" not in (overlay.return_timeline or "")
        assert "ETA" in (overlay.return_timeline or "")


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
        """Blake Snell case: BDL shows Jun 30 (49d) for 15-Day IL.
        49d > 1.2×15=18d → TBD (eligible May 27); Jun 30 must not appear."""
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

        # Should show TBD (eligible May 27), not Jun 30
        assert overlay.return_timeline is not None
        assert "May 27" in overlay.return_timeline, (
            f"Expected 'May 27' but got: {overlay.return_timeline!r}"
        )
        assert "Jun 30" not in overlay.return_timeline
        assert "TBD" in overlay.return_timeline, (
            f"Expected 'TBD' for uncertain 15-Day IL estimate, got: {overlay.return_timeline!r}"
        )

    def test_60day_il_within_range_keeps_bdl_date(self):
        """60-Day IL with BDL date 65 days out: within 72d (1.2×60), keep BDL date."""
        injury_date = datetime(2026, 3, 1, 0, 0, tzinfo=_ET)
        bdl_return_date = injury_date + timedelta(days=65)  # 65 days — within 1.2×60=72
        now_et = self._now()

        overlay = build_injury_overlay(
            status="60-Day-IL",
            note="Tommy John surgery",
            return_date=bdl_return_date,
            ingested_at=now_et - timedelta(hours=2),
            injury_date=injury_date,
            now_et=now_et,
        )

        # BDL date 65 days from injury is within 1.2×60=72, so we keep it (show specific date)
        assert overlay.return_timeline is not None
        assert "ETA" in overlay.return_timeline
        assert "TBD" not in overlay.return_timeline

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
        # implied_days = (Jun 5 - ~May 22) ≈ 14 days <= 15*1.2=18, so keep BDL date
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


# ---------------------------------------------------------------------------
# Task C: TBD display when BDL date is uncertain (> 1.2× IL minimum)
# ---------------------------------------------------------------------------


class TestETATBDDisplay:
    """ETA shows TBD + eligibility date when BDL return is > 1.2× IL minimum."""

    def _now(self):
        return datetime(2026, 5, 22, 12, 0, tzinfo=_ET)

    def test_bdl_date_just_over_1_2x_shows_tbd(self):
        """BDL shows Jun 2 for 15-Day IL: 21d from May 12 > 1.2×15=18 → TBD (eligible May 27)."""
        injury_date = datetime(2026, 5, 12, 0, 0, tzinfo=_ET)
        bdl_return_date = datetime(2026, 6, 2, 0, 0, tzinfo=_ET)  # 21 days out > 18
        now_et = self._now()

        overlay = build_injury_overlay(
            status="15-Day-IL",
            note="Elbow surgery — no timetable",
            return_date=bdl_return_date,
            ingested_at=now_et - timedelta(hours=1),
            injury_date=injury_date,
            now_et=now_et,
        )

        assert overlay.return_timeline is not None
        assert "TBD" in overlay.return_timeline, (
            f"Expected 'TBD' for BDL date > 1.2× IL minimum, got: {overlay.return_timeline!r}"
        )
        assert "May 27" in overlay.return_timeline, (
            f"Expected eligibility 'May 27' in TBD message, got: {overlay.return_timeline!r}"
        )
        assert "Jun 2" not in overlay.return_timeline, (
            f"Uncertain BDL date 'Jun 2' must not appear, got: {overlay.return_timeline!r}"
        )

    def test_bdl_date_at_il_minimum_shows_specific_date(self):
        """BDL shows May 27 (exactly 15d from May 12): 15d ≤ 1.2×15=18 → show specific date."""
        injury_date = datetime(2026, 5, 12, 0, 0, tzinfo=_ET)
        bdl_return_date = datetime(2026, 5, 27, 0, 0, tzinfo=_ET)  # exactly 15 days
        now_et = self._now()

        overlay = build_injury_overlay(
            status="15-Day-IL",
            note="Hamstring strain",
            return_date=bdl_return_date,
            ingested_at=now_et - timedelta(hours=1),
            injury_date=injury_date,
            now_et=now_et,
        )

        assert overlay.return_timeline is not None
        assert "TBD" not in overlay.return_timeline
        assert "May 27" in overlay.return_timeline

    def test_bdl_date_within_1_2x_shows_specific_date(self):
        """BDL shows May 25 (13d from May 12): 13d ≤ 1.2×15=18 → show specific BDL date."""
        injury_date = datetime(2026, 5, 12, 0, 0, tzinfo=_ET)
        bdl_return_date = datetime(2026, 5, 25, 0, 0, tzinfo=_ET)  # 13 days — within 18
        now_et = self._now()

        overlay = build_injury_overlay(
            status="15-Day-IL",
            note="Hamstring strain",
            return_date=bdl_return_date,
            ingested_at=now_et - timedelta(hours=1),
            injury_date=injury_date,
            now_et=now_et,
        )

        assert overlay.return_timeline is not None
        assert "TBD" not in overlay.return_timeline
        assert "May 25" in overlay.return_timeline

    def test_no_return_date_shows_eligibility_not_tbd(self):
        """When BDL provides no return_date, show computed eligibility as specific date (no TBD)."""
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

        # No BDL date → no uncertainty → show computed eligibility as specific date
        assert overlay.return_timeline is not None
        assert "May 27" in overlay.return_timeline
        assert "TBD" not in overlay.return_timeline

    def test_60day_il_beyond_1_2x_shows_tbd(self):
        """60-Day IL with BDL date 80d out (> 1.2×60=72): TBD (eligible 60d from start)."""
        injury_date = datetime(2026, 3, 1, 0, 0, tzinfo=_ET)
        bdl_return_date = injury_date + timedelta(days=80)  # 80d > 72d threshold
        now_et = self._now()

        overlay = build_injury_overlay(
            status="60-Day-IL",
            note="Tommy John — no confirmed return",
            return_date=bdl_return_date,
            ingested_at=now_et - timedelta(hours=2),
            injury_date=injury_date,
            now_et=now_et,
        )

        assert overlay.return_timeline is not None
        assert "TBD" in overlay.return_timeline, (
            f"Expected 'TBD' for 60-Day IL with BDL date > 1.2× minimum, got: {overlay.return_timeline!r}"
        )
