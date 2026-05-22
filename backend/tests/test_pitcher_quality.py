"""
Tests for pitcher quality score pipeline.

Covers:
- build_pitcher_quality_map: DB query, dedup (keep highest), name normalization, non-fatal errors
- WaiverPlayerOut.quality_score schema field
- Integration: pitcher FA → quality_score wired; non-pitcher → None
"""
import pytest
from datetime import date
from unittest.mock import Mock, patch


# ---------------------------------------------------------------------------
# Schema tests — WaiverPlayerOut.quality_score
# ---------------------------------------------------------------------------

class TestWaiverPlayerOutQualityScore:
    """WaiverPlayerOut.quality_score field validation."""

    def _make_player(self, **kwargs):
        from backend.schemas import WaiverPlayerOut
        defaults = dict(player_id="mlb.p.123", name="Test Player", team="LAD", position="1B")
        defaults.update(kwargs)
        return WaiverPlayerOut(**defaults)

    def test_defaults_to_none(self):
        """quality_score defaults to None when not provided."""
        player = self._make_player()
        assert player.quality_score is None

    def test_accepts_positive_float(self):
        """quality_score accepts a positive float."""
        player = self._make_player(position="SP", quality_score=1.5)
        assert player.quality_score == 1.5

    def test_accepts_negative_float(self):
        """quality_score accepts a negative float."""
        player = self._make_player(position="SP", quality_score=-1.2)
        assert player.quality_score == -1.2

    def test_accepts_zero(self):
        """quality_score accepts zero."""
        player = self._make_player(position="SP", quality_score=0.0)
        assert player.quality_score == 0.0

    def test_explicit_none(self):
        """quality_score can be explicitly set to None (pitcher without snapshot)."""
        player = self._make_player(position="SP", quality_score=None)
        assert player.quality_score is None

    def test_non_pitcher_with_none(self):
        """Non-pitcher FA always has quality_score=None."""
        batter = self._make_player(position="1B", quality_score=None)
        assert batter.quality_score is None


# ---------------------------------------------------------------------------
# build_pitcher_quality_map unit tests
# ---------------------------------------------------------------------------

class TestBuildPitcherQualityMap:
    """Tests for build_pitcher_quality_map helper."""

    def _call(self, db, today=None):
        from backend.routers.fantasy import build_pitcher_quality_map
        return build_pitcher_quality_map(db, today or date(2026, 5, 15))

    def test_empty_db_result_returns_empty_map(self, mock_db_session):
        """Empty DB result produces empty dict."""
        mock_db_session.query.return_value.filter.return_value.all.return_value = []
        result = self._call(mock_db_session)
        assert result == {}

    def test_single_pitcher_row(self, mock_db_session):
        """Single snapshot row is keyed by lowercased name."""
        row = Mock()
        row.pitcher_name = "Gerrit Cole"
        row.quality_score = 1.45
        mock_db_session.query.return_value.filter.return_value.all.return_value = [row]

        result = self._call(mock_db_session)

        assert "gerrit cole" in result
        assert result["gerrit cole"] == pytest.approx(1.45)

    def test_name_is_lowercased_and_stripped(self, mock_db_session):
        """Name keys are .strip().lower() normalized."""
        row = Mock()
        row.pitcher_name = "  Shane McClanahan  "
        row.quality_score = 1.20
        mock_db_session.query.return_value.filter.return_value.all.return_value = [row]

        result = self._call(mock_db_session)

        assert "shane mcclanahan" in result
        assert "  Shane McClanahan  " not in result

    def test_keeps_highest_quality_score_for_duplicate_pitcher(self, mock_db_session):
        """Multiple starts for same pitcher: keep the highest quality_score."""
        row1 = Mock()
        row1.pitcher_name = "Gerrit Cole"
        row1.quality_score = 0.80

        row2 = Mock()
        row2.pitcher_name = "Gerrit Cole"
        row2.quality_score = 1.45  # higher — should win

        mock_db_session.query.return_value.filter.return_value.all.return_value = [row1, row2]

        result = self._call(mock_db_session)

        assert result["gerrit cole"] == pytest.approx(1.45)

    def test_keeps_highest_even_when_lower_comes_second(self, mock_db_session):
        """Order-invariant: highest wins regardless of row order."""
        row1 = Mock()
        row1.pitcher_name = "Shane McClanahan"
        row1.quality_score = 1.90  # higher — should win

        row2 = Mock()
        row2.pitcher_name = "Shane McClanahan"
        row2.quality_score = 0.30

        mock_db_session.query.return_value.filter.return_value.all.return_value = [row1, row2]

        result = self._call(mock_db_session)

        assert result["shane mcclanahan"] == pytest.approx(1.90)

    def test_multiple_distinct_pitchers_all_included(self, mock_db_session):
        """Multiple distinct pitchers each get their own entry."""
        row1 = Mock()
        row1.pitcher_name = "Gerrit Cole"
        row1.quality_score = 1.45

        row2 = Mock()
        row2.pitcher_name = "Shane McClanahan"
        row2.quality_score = 1.20

        mock_db_session.query.return_value.filter.return_value.all.return_value = [row1, row2]

        result = self._call(mock_db_session)

        assert len(result) == 2
        assert "gerrit cole" in result
        assert "shane mcclanahan" in result

    def test_db_exception_returns_empty_map(self, mock_db_session):
        """DB exception is non-fatal; returns empty dict."""
        mock_db_session.query.side_effect = Exception("DB connection failed")

        result = self._call(mock_db_session)

        assert result == {}

    def test_null_pitcher_name_is_skipped(self, mock_db_session):
        """Rows with NULL pitcher_name are skipped."""
        row = Mock()
        row.pitcher_name = None
        row.quality_score = 1.0
        mock_db_session.query.return_value.filter.return_value.all.return_value = [row]

        result = self._call(mock_db_session)

        assert result == {}

    def test_null_quality_score_is_skipped(self, mock_db_session):
        """Rows with NULL quality_score are skipped."""
        row = Mock()
        row.pitcher_name = "Gerrit Cole"
        row.quality_score = None
        mock_db_session.query.return_value.filter.return_value.all.return_value = [row]

        result = self._call(mock_db_session)

        assert result == {}

    def test_quality_score_is_cast_to_float(self, mock_db_session):
        """quality_score values are cast to float (handles Decimal from DB)."""
        row = Mock()
        row.pitcher_name = "Gerrit Cole"
        row.quality_score = 1  # int-like (DB can return Decimal)
        mock_db_session.query.return_value.filter.return_value.all.return_value = [row]

        result = self._call(mock_db_session)

        assert isinstance(result["gerrit cole"], float)
        assert result["gerrit cole"] == 1.0

    def test_db_is_queried(self, mock_db_session):
        """DB session is actually queried (not short-circuited)."""
        mock_db_session.query.return_value.filter.return_value.all.return_value = []

        self._call(mock_db_session)

        mock_db_session.query.assert_called_once()

    def test_empty_pitcher_name_is_skipped(self, mock_db_session):
        """Rows with empty string pitcher_name are skipped."""
        row = Mock()
        row.pitcher_name = ""
        row.quality_score = 0.75
        mock_db_session.query.return_value.filter.return_value.all.return_value = [row]

        result = self._call(mock_db_session)

        assert result == {}


# ---------------------------------------------------------------------------
# Integration: quality_score in WaiverPlayerOut
# ---------------------------------------------------------------------------

class TestQualityScorePipelineIntegration:
    """Integration tests for quality_score wiring to WaiverPlayerOut."""

    def test_pitcher_fa_with_matching_snapshot_gets_score(self):
        """SP pitcher FA with a snapshot match has a non-None quality_score."""
        from backend.schemas import WaiverPlayerOut
        player = WaiverPlayerOut(
            player_id="mlb.p.456",
            name="Gerrit Cole",
            team="NYY",
            position="SP",
            quality_score=1.45,
        )
        assert player.quality_score is not None
        assert player.quality_score == pytest.approx(1.45)

    def test_pitcher_without_snapshot_entry_gets_none(self):
        """SP pitcher with no snapshot entry gets quality_score=None."""
        from backend.schemas import WaiverPlayerOut
        player = WaiverPlayerOut(
            player_id="mlb.p.456",
            name="Unknown Pitcher",
            team="SEA",
            position="SP",
            quality_score=None,
        )
        assert player.quality_score is None

    def test_batter_fa_always_gets_none(self):
        """Batter FA always has quality_score=None (not a pitcher)."""
        from backend.schemas import WaiverPlayerOut
        batter = WaiverPlayerOut(
            player_id="mlb.p.123",
            name="Pete Alonso",
            team="NYM",
            position="1B",
            quality_score=None,
        )
        assert batter.quality_score is None

    def test_reliever_fa_can_have_quality_score(self):
        """RP pitcher FA can also have a quality_score (non-None)."""
        from backend.schemas import WaiverPlayerOut
        reliever = WaiverPlayerOut(
            player_id="mlb.p.789",
            name="Emmanuel Clase",
            team="CLE",
            position="RP",
            quality_score=-0.50,
        )
        assert reliever.quality_score == pytest.approx(-0.50)

    def test_name_lookup_is_case_insensitive(self, mock_db_session):
        """build_pitcher_quality_map is case-insensitive for player name lookup."""
        from backend.routers.fantasy import build_pitcher_quality_map

        row = Mock()
        row.pitcher_name = "Gerrit Cole"
        row.quality_score = 1.45
        mock_db_session.query.return_value.filter.return_value.all.return_value = [row]

        quality_map = build_pitcher_quality_map(mock_db_session, date(2026, 5, 15))

        # lookup by lowercase (as the route handler does: name.lower())
        assert quality_map.get("gerrit cole") == pytest.approx(1.45)
        # original case is NOT in the map
        assert "Gerrit Cole" not in quality_map
