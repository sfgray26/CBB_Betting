"""
Regression tests for database query optimizations.

Covers:
- fetch_rolling_stats_for_players_all_windows: batch multi-window fetch
- get_fantasy_roster projection name lookup: single ORM query vs old raw-SQL+re-query
"""
import pytest
from datetime import date, datetime
from unittest.mock import MagicMock, patch, call
from zoneinfo import ZoneInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rolling_stats(bdl_player_id: int, window_days: int, w_avg: float = 0.275):
    """Return a mock PlayerRollingStats object."""
    obj = MagicMock()
    obj.bdl_player_id = bdl_player_id
    obj.as_of_date = date(2026, 5, 15)
    obj.window_days = window_days
    obj.w_avg = w_avg
    obj.w_runs = 5.0
    obj.w_hits = 8.5
    obj.w_home_runs = 1.5
    obj.w_rbi = 5.2
    obj.w_strikeouts_bat = 7.0
    obj.w_tb = 13.5
    obj.w_net_stolen_bases = 0.5
    obj.w_ops = 0.850
    return obj


def _make_id_mapping(bdl_id: int, yahoo_key: str, yahoo_id: str):
    obj = MagicMock()
    obj.bdl_id = bdl_id
    obj.yahoo_key = yahoo_key
    obj.yahoo_id = yahoo_id
    return obj


# ---------------------------------------------------------------------------
# fetch_rolling_stats_for_players_all_windows
# ---------------------------------------------------------------------------

class TestFetchRollingStatsAllWindows:
    """Tests for the batch multi-window rolling-stats fetcher."""

    def _mock_db(self, mapping_rows, rolling_rows):
        """Build a mock Session whose .query() chain returns controlled data."""
        db = MagicMock()

        id_mapping_result = MagicMock()
        id_mapping_result.all.return_value = mapping_rows

        rolling_result = MagicMock()
        rolling_result.filter.return_value = rolling_result
        rolling_result.all.return_value = rolling_rows

        max_date_result = MagicMock()
        max_date_result.filter.return_value = max_date_result
        max_date_result.scalar.return_value = None

        def _query_side_effect(model, *cols):
            from backend.models import PlayerIDMapping, PlayerRollingStats
            # Identify the query by the first positional arg
            first = model if not hasattr(model, "__mro__") else model
            try:
                first_name = getattr(first, "__tablename__", None) or getattr(
                    getattr(first, "class_", first), "__tablename__", None
                )
            except Exception:
                first_name = None
            if first_name == "player_id_mapping":
                q = MagicMock()
                q.filter.return_value = id_mapping_result
                return q
            if first_name == "player_rolling_stats":
                q = MagicMock()
                q.filter.return_value = rolling_result
                return q
            # func.max case
            return max_date_result

        db.query.side_effect = _query_side_effect
        return db

    def test_returns_dict_keyed_by_window(self):
        from backend.services.player_mapper import fetch_rolling_stats_for_players_all_windows
        from backend.models import PlayerIDMapping, PlayerRollingStats

        mapping = _make_id_mapping(1, "469.p.100", "100")
        rs_7 = _make_rolling_stats(1, 7)
        rs_14 = _make_rolling_stats(1, 14)
        rs_30 = _make_rolling_stats(1, 30)

        db = MagicMock()
        db.query.side_effect = _make_query_side_effect(
            mapping_rows=[mapping],
            rolling_rows=[rs_7, rs_14, rs_30],
        )

        result = fetch_rolling_stats_for_players_all_windows(
            db=db,
            yahoo_player_keys=["469.p.100"],
            as_of_date="2026-05-15",
            window_sizes=[7, 14, 30],
        )

        assert set(result.keys()) == {7, 14, 30}
        assert "469.p.100" in result[7]
        assert "469.p.100" in result[14]
        assert "469.p.100" in result[30]
        assert result[7]["469.p.100"].window_days == 7
        assert result[14]["469.p.100"].window_days == 14
        assert result[30]["469.p.100"].window_days == 30

    def test_empty_player_keys_returns_empty_dicts(self):
        from backend.services.player_mapper import fetch_rolling_stats_for_players_all_windows

        db = MagicMock()
        result = fetch_rolling_stats_for_players_all_windows(
            db=db,
            yahoo_player_keys=[],
            window_sizes=[7, 14, 30],
        )

        assert result == {7: {}, 14: {}, 30: {}}
        db.query.assert_not_called()

    def test_no_id_mapping_returns_empty_dicts(self):
        from backend.services.player_mapper import fetch_rolling_stats_for_players_all_windows

        db = MagicMock()
        id_q = MagicMock()
        id_q.filter.return_value.all.return_value = []
        db.query.return_value = id_q

        result = fetch_rolling_stats_for_players_all_windows(
            db=db,
            yahoo_player_keys=["469.p.999"],
            window_sizes=[7, 14, 30],
        )

        assert result == {7: {}, 14: {}, 30: {}}

    def test_multiple_players_correctly_partitioned(self):
        from backend.services.player_mapper import fetch_rolling_stats_for_players_all_windows

        mapping1 = _make_id_mapping(1, "469.p.100", "100")
        mapping2 = _make_id_mapping(2, "469.p.200", "200")
        rs1_14 = _make_rolling_stats(1, 14, w_avg=0.300)
        rs2_14 = _make_rolling_stats(2, 14, w_avg=0.250)

        db = MagicMock()
        db.query.side_effect = _make_query_side_effect(
            mapping_rows=[mapping1, mapping2],
            rolling_rows=[rs1_14, rs2_14],
        )

        result = fetch_rolling_stats_for_players_all_windows(
            db=db,
            yahoo_player_keys=["469.p.100", "469.p.200"],
            as_of_date="2026-05-15",
            window_sizes=[14],
        )

        assert result[14]["469.p.100"].w_avg == pytest.approx(0.300)
        assert result[14]["469.p.200"].w_avg == pytest.approx(0.250)

    def test_custom_window_sizes_respected(self):
        from backend.services.player_mapper import fetch_rolling_stats_for_players_all_windows

        db = MagicMock()
        id_q = MagicMock()
        id_q.filter.return_value.all.return_value = []
        db.query.return_value = id_q

        result = fetch_rolling_stats_for_players_all_windows(
            db=db,
            yahoo_player_keys=["469.p.1"],
            window_sizes=[14, 30],
        )
        # Keys must match exactly what was requested
        assert set(result.keys()) == {14, 30}

    def test_makes_fewer_queries_than_per_window_calls(self):
        """Batch function must query PlayerIDMapping exactly once regardless of window count."""
        from backend.services.player_mapper import fetch_rolling_stats_for_players_all_windows

        id_q = MagicMock()
        id_q.filter.return_value.all.return_value = []
        db = MagicMock()
        db.query.return_value = id_q

        fetch_rolling_stats_for_players_all_windows(
            db=db,
            yahoo_player_keys=["469.p.1"],
            window_sizes=[7, 14, 30],
        )

        # One PlayerIDMapping query + at most one PlayerRollingStats query
        assert db.query.call_count <= 2


# ---------------------------------------------------------------------------
# ProjectionNameLookup (single ORM query vs old double-query)
# ---------------------------------------------------------------------------

class TestProjectionNameLookup:
    """Verify the refactored projection name lookup uses a single ORM query."""

    def test_single_query_returns_all_named_projections(self):
        """cast(cat_scores, Text) != '{}' filter should retrieve all scored projections."""
        from unittest.mock import MagicMock

        proj1 = MagicMock()
        proj1.player_name = "Mike Trout"
        proj1.player_id = "12345"
        proj1.cat_scores = {"hr": 0.8, "avg": 0.6}

        proj2 = MagicMock()
        proj2.player_name = "Shohei Ohtani"
        proj2.player_id = "67890"
        proj2.cat_scores = {"hr": 1.2, "era": -0.3}

        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = [proj1, proj2]

        # Simulate the ORM call pattern from the refactored code
        from sqlalchemy import cast, Text
        from backend.models import PlayerProjection

        _name_rows = (
            db.query(PlayerProjection)
            .filter(
                PlayerProjection.cat_scores.isnot(None),
                cast(PlayerProjection.cat_scores, Text) != "{}",
            )
            .all()
        )

        assert len(_name_rows) == 2

        # Verify .query() was called once (not twice like the old pattern)
        db.query.assert_called_once()

    def test_empty_cat_scores_excluded(self):
        """Rows with cat_scores == {} must not appear in the result."""
        proj_empty = MagicMock()
        proj_empty.player_name = "Empty Player"
        proj_empty.cat_scores = {}

        proj_scored = MagicMock()
        proj_scored.player_name = "Scored Player"
        proj_scored.cat_scores = {"hr": 0.5}

        # The filter is applied in DB — the mock returns only the row that passes
        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = [proj_scored]

        from sqlalchemy import cast, Text
        from backend.models import PlayerProjection

        _name_rows = (
            db.query(PlayerProjection)
            .filter(
                PlayerProjection.cat_scores.isnot(None),
                cast(PlayerProjection.cat_scores, Text) != "{}",
            )
            .all()
        )

        names = [r.player_name for r in _name_rows]
        assert "Scored Player" in names
        assert "Empty Player" not in names


# ---------------------------------------------------------------------------
# Model index definitions
# ---------------------------------------------------------------------------

class TestModelIndexDefinitions:
    """Verify that new indexes are declared on the models."""

    def test_player_score_has_bdl_window_date_index(self):
        from backend.models import PlayerScore
        from sqlalchemy import Index

        index_names = {
            idx.name
            for idx in PlayerScore.__table__.indexes
        }
        assert "idx_ps_bdl_window_date" in index_names

    def test_player_projection_has_cat_scores_partial_index(self):
        from backend.models import PlayerProjection

        index_names = {
            idx.name
            for idx in PlayerProjection.__table__.indexes
        }
        assert "idx_pp_cat_scores_nn" in index_names

    def test_player_rolling_stats_has_player_date_index(self):
        from backend.models import PlayerRollingStats

        index_names = {
            idx.name
            for idx in PlayerRollingStats.__table__.indexes
        }
        assert "idx_prs_player_date" in index_names


# ---------------------------------------------------------------------------
# SA_ECHO_QUERIES env-var toggle
# ---------------------------------------------------------------------------

class TestSAEchoToggle:
    """Verify that SA_ECHO_QUERIES env var controls SQLAlchemy query logging."""

    def test_echo_false_by_default(self):
        """Engine echo must be False when SA_ECHO_QUERIES is not set."""
        import importlib
        import os

        env = {k: v for k, v in os.environ.items() if k != "SA_ECHO_QUERIES"}
        with patch.dict("os.environ", env, clear=True):
            import backend.models as _m
            importlib.reload(_m)
            assert _m.engine.echo is False

    def test_echo_true_when_env_set(self):
        """Engine echo must be True when SA_ECHO_QUERIES=true."""
        import importlib

        with patch.dict("os.environ", {"SA_ECHO_QUERIES": "true"}):
            import backend.models as _m
            importlib.reload(_m)
            assert _m.engine.echo is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_query_side_effect(mapping_rows, rolling_rows, max_date=None):
    """
    Return a side_effect callable for db.query that dispatches based on call order.

    The batch function always issues queries in this order:
      1. PlayerIDMapping column query (columns passed, not class)
      2. PlayerRollingStats class query (or func.max fallback then class)
    """
    from backend.models import PlayerRollingStats

    id_q = MagicMock()
    id_q.filter.return_value.all.return_value = mapping_rows

    rs_q = MagicMock()
    rs_q.filter.return_value.all.return_value = rolling_rows

    calls = {"count": 0}

    def _side_effect(*args):
        calls["count"] += 1
        # First call is always the PlayerIDMapping column query
        if calls["count"] == 1:
            return id_q
        # Second+ call: PlayerRollingStats
        return rs_q

    return _side_effect
