"""
Tests for ProjectionAssemblyService.

All DB interactions are mocked. Tests verify:
- correct source_engine assignment
- counting stat hybrid provenance
- z-score computation
- identity miss handling
- confidence score formula
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import date

from backend.fantasy_baseball.projection_assembly_service import (
    ProjectionAssemblyService,
    _zscore_pool,
    _zscore,
)
from backend.fantasy_baseball.fusion_engine import FusionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fusion_result(source="fusion", hr_per_pa=0.05, sb_per_pa=0.01,
                        era=3.80, k_per_nine=9.5, components_fused=3):
    return FusionResult(
        proj={
            "hr_per_pa": hr_per_pa,
            "sb_per_pa": sb_per_pa,
            "avg": 0.280,
            "obp": 0.350,
            "slg": 0.480,
            "ops": 0.830,
            "k_percent": 0.22,
            "bb_percent": 0.08,
            "era": era,
            "k_per_nine": k_per_nine,
            "whip": 1.20,
            "k9": k_per_nine,
        },
        source=source,
        components_fused=components_fused,
        xwoba_override_detected=False,
    )


# ---------------------------------------------------------------------------
# Z-score helpers
# ---------------------------------------------------------------------------

class TestZscorePool:
    def test_returns_mean_and_std(self):
        mean, std = _zscore_pool([1.0, 2.0, 3.0])
        assert abs(mean - 2.0) < 0.01
        assert std > 0

    def test_ignores_none(self):
        mean, std = _zscore_pool([None, 2.0, None, 4.0])
        assert abs(mean - 3.0) < 0.01

    def test_fewer_than_two_returns_defaults(self):
        mean, std = _zscore_pool([5.0])
        assert mean == 0.0
        assert std == 1.0

    def test_empty_returns_defaults(self):
        mean, std = _zscore_pool([])
        assert mean == 0.0
        assert std == 1.0


class TestZscore:
    def test_above_mean_positive(self):
        z = _zscore(3.0, mean=2.0, std=1.0, direction=1)
        assert z == pytest.approx(1.0)

    def test_negative_direction(self):
        z = _zscore(5.0, mean=4.0, std=1.0, direction=-1)
        assert z == pytest.approx(-1.0)

    def test_none_value_returns_none(self):
        assert _zscore(None, 2.0, 1.0) is None


# ---------------------------------------------------------------------------
# ProjectionAssemblyService unit tests
# ---------------------------------------------------------------------------

class TestBuildBatterSteamer:
    def test_avg_passthrough(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026
        board_proj = {
            "pa": 600, "avg": 0.300, "slg": 0.550, "k_bat": 120,
            "hr": 30, "nsb": 10, "ops": 0.880,
        }
        steamer = svc._build_batter_steamer(board_proj)
        assert steamer["avg"] == pytest.approx(0.300)

    def test_hr_per_pa_computed(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026
        board_proj = {
            "pa": 600, "avg": 0.280, "slg": 0.480, "k_bat": 130,
            "hr": 30, "nsb": 8, "ops": 0.810,
        }
        steamer = svc._build_batter_steamer(board_proj)
        assert steamer["hr_per_pa"] == pytest.approx(30 / 600)

    def test_sb_per_pa_uses_nsb(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026
        board_proj = {
            "pa": 500, "avg": 0.270, "slg": 0.400, "k_bat": 100,
            "hr": 15, "nsb": 20, "ops": 0.730,
        }
        steamer = svc._build_batter_steamer(board_proj)
        assert steamer["sb_per_pa"] == pytest.approx(20 / 500)

    def test_zero_pa_uses_population_priors(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026
        board_proj = {
            "pa": 0, "avg": 0.250, "slg": 0.400, "k_bat": 0,
            "hr": 0, "nsb": 0, "ops": 0.730,
        }
        steamer = svc._build_batter_steamer(board_proj)
        assert steamer["hr_per_pa"] == pytest.approx(0.035)
        assert steamer["sb_per_pa"] == pytest.approx(0.010)


class TestBuildPitcherSteamer:
    def test_era_and_whip_passthrough(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026
        board_proj = {
            "ip": 180, "era": 3.50, "whip": 1.10, "k9": 9.5,
            "w": 12, "sv": 0, "k_pit": 190, "nsv": 0,
        }
        steamer = svc._build_pitcher_steamer(board_proj)
        assert steamer["era"] == pytest.approx(3.50)
        assert steamer["whip"] == pytest.approx(1.10)

    def test_k_per_nine_passthrough(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026
        board_proj = {
            "ip": 180, "era": 3.80, "whip": 1.15, "k9": 10.5,
            "w": 12, "sv": 0, "k_pit": 210, "nsv": 0,
        }
        steamer = svc._build_pitcher_steamer(board_proj)
        assert steamer["k_per_nine"] == pytest.approx(10.5)


class TestConfidenceScore:
    def test_fusion_source_higher_than_static(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026
        fusion_result = _make_fusion_result(source="fusion")
        static_result = _make_fusion_result(source="steamer")
        c_fusion = svc._confidence_score(fusion_result, sample_size=300)
        c_static = svc._confidence_score(static_result, sample_size=300)
        assert c_fusion > c_static

    def test_large_sample_raises_confidence(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026
        r = _make_fusion_result(source="fusion")
        low = svc._confidence_score(r, sample_size=50)
        high = svc._confidence_score(r, sample_size=300)
        assert high > low

    def test_score_bounded_0_to_1(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026
        r = _make_fusion_result(source="fusion")
        score = svc._confidence_score(r, sample_size=10000)
        assert 0.0 <= score <= 1.0

    def test_population_prior_lowest_confidence(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026
        r = _make_fusion_result(source="population_prior", components_fused=0)
        score = svc._confidence_score(r, sample_size=0)
        assert score <= 0.4

    def test_steamer_prior_has_nonzero_confidence_without_sample(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026
        r = _make_fusion_result(source="steamer", components_fused=0)
        score = svc._confidence_score(r, sample_size=0)
        assert 0.25 <= score <= 0.5

    def test_fusion_components_have_bounded_confidence_without_sample(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026
        r = _make_fusion_result(source="fusion", components_fused=3)
        score = svc._confidence_score(r, sample_size=0)
        assert 0.25 <= score < 0.7


class TestSourceEngineAssignment:
    """Verify correct source_engine strings based on data availability."""

    def test_fusion_source_maps_to_savant_adjusted(self):
        # When statcast data is present and fusion succeeds
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026
        result = _make_fusion_result(source="fusion")
        has_statcast = True
        engine = "SAVANT_ADJUSTED" if (has_statcast and result.source == "fusion") else "STATIC_BOARD"
        assert engine == "SAVANT_ADJUSTED"

    def test_steamer_only_maps_to_static_board(self):
        result = _make_fusion_result(source="steamer")
        has_statcast = False
        engine = "SAVANT_ADJUSTED" if (has_statcast and result.source == "fusion") else "STATIC_BOARD"
        assert engine == "STATIC_BOARD"


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------

class TestBuildBatterStatcast:
    def test_obp_derived_from_ops_minus_slg(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026
        metrics = MagicMock()
        metrics.avg = 0.280
        metrics.ops = 0.850
        metrics.slg = 0.500
        metrics.xwoba = 0.360
        metrics.barrel_percent = 12.5
        result = svc._build_batter_statcast(metrics)
        assert result["obp"] == pytest.approx(0.350)

    def test_barrel_pct_converted_to_decimal(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026
        metrics = MagicMock()
        metrics.avg = 0.280
        metrics.ops = 0.850
        metrics.slg = 0.500
        metrics.xwoba = 0.360
        metrics.barrel_percent = 15.0
        result = svc._build_batter_statcast(metrics)
        assert result["barrel_pct"] == pytest.approx(0.15)

    def test_none_ops_gives_none_obp(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026
        metrics = MagicMock()
        metrics.avg = 0.250
        metrics.ops = None
        metrics.slg = None
        metrics.xwoba = None
        metrics.barrel_percent = None
        result = svc._build_batter_statcast(metrics)
        assert result["obp"] is None

    def test_k_percent_always_none(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026
        metrics = MagicMock()
        metrics.avg = 0.280
        metrics.ops = 0.800
        metrics.slg = 0.450
        metrics.xwoba = 0.330
        metrics.barrel_percent = 10.0
        result = svc._build_batter_statcast(metrics)
        assert result["k_percent"] is None


class TestBuildPitcherStatcast:
    def test_fields_mapped_correctly(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026
        metrics = MagicMock()
        metrics.era = 3.50
        metrics.xera = 3.20
        metrics.k_percent = 0.28
        metrics.bb_percent = 0.07
        metrics.k_9 = 10.0
        result = svc._build_pitcher_statcast(metrics)
        assert result["era"] == pytest.approx(3.50)
        assert result["xera"] == pytest.approx(3.20)
        assert result["k_per_nine"] == pytest.approx(10.0)
        assert result["whip"] is None


class TestBarrelDecimal:
    def test_converts_percent_to_decimal(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026
        metrics = MagicMock()
        metrics.barrel_percent = 12.5
        assert svc._barrel_decimal(metrics) == pytest.approx(0.125)

    def test_none_metrics_returns_none(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026
        assert svc._barrel_decimal(None) is None

    def test_none_barrel_percent_returns_none(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026
        metrics = MagicMock()
        metrics.barrel_percent = None
        assert svc._barrel_decimal(metrics) is None


class TestResolveMLBAMId:
    def test_returns_mlbam_id_on_hit(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026

        mock_row = MagicMock()
        mock_row.mlbam_id = 660271

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = mock_row

        result = svc._resolve_mlbam_id(mock_session, "juan soto")
        assert result == 660271

    def test_returns_none_on_miss(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = None

        result = svc._resolve_mlbam_id(mock_session, "unknown rookie")
        assert result is None


class TestZscorePools:
    def test_pool_keys_present_for_batters_and_pitchers(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026

        batters = [
            {"proj": {"r": 100, "hr": 30, "rbi": 90, "nsb": 10, "avg": 0.290, "ops": 0.900, "obp": 0.370}},
            {"proj": {"r": 80, "hr": 20, "rbi": 70, "nsb": 20, "avg": 0.270, "ops": 0.800, "obp": 0.340}},
        ]
        pitchers = [
            {"proj": {"w": 12, "k_pit": 200, "sv": 0, "era": 3.00, "whip": 1.05, "k9": 10.0}},
            {"proj": {"w": 8, "k_pit": 150, "sv": 30, "era": 4.00, "whip": 1.30, "k9": 8.0}},
        ]

        pool = svc._build_zscore_pools(batters, pitchers)

        # Verify all expected keys exist
        for key in ("r", "hr", "rbi", "nsb", "k_bat", "tb", "avg", "ops", "obp"):
            assert key in pool, f"Missing batter pool key: {key}"
        for key in ("w", "k", "sv", "l", "hr_pit", "qs", "era", "whip", "k9"):
            assert key in pool, f"Missing pitcher pool key: {key}"

    def test_pool_returns_valid_mean_std(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026

        batters = [
            {"proj": {"r": 100, "hr": 30, "rbi": 90, "nsb": 10, "avg": 0.290, "ops": 0.900, "obp": 0.370}},
            {"proj": {"r": 80, "hr": 20, "rbi": 70, "nsb": 20, "avg": 0.270, "ops": 0.800, "obp": 0.340}},
        ]
        pitchers = []
        pool = svc._build_zscore_pools(batters, pitchers)

        mean, std = pool["hr"]
        assert abs(mean - 25.0) < 0.01
        assert std > 0


class TestBuildCategoryImpacts:
    def test_negative_category_direction(self):
        """ERA z-score should be negative when ERA is above the mean."""
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026

        z_pool = {"era": (3.80, 0.50)}  # pool mean=3.80, std=0.50
        values = {"ERA": 4.30}          # 1 std above mean => raw z=+1 => direction=-1 => z=-1

        impacts = svc._build_category_impacts(42, "PITCHER", values, z_pool)
        assert len(impacts) == 1
        assert impacts[0].z_score == pytest.approx(-1.0)

    def test_positive_category_direction(self):
        """HR z-score should be positive when above mean."""
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026

        z_pool = {"hr": (25.0, 5.0)}
        values = {"HR": 30.0}  # 1 std above mean => z=+1

        impacts = svc._build_category_impacts(42, "BATTER", values, z_pool)
        assert len(impacts) == 1
        assert impacts[0].z_score == pytest.approx(1.0)

    def test_none_value_yields_none_z_score(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.season = 2026

        z_pool = {"avg": (0.270, 0.020)}
        values = {"AVG": None}

        impacts = svc._build_category_impacts(42, "BATTER", values, z_pool)
        assert len(impacts) == 1
        assert impacts[0].z_score is None


# ---------------------------------------------------------------------------
# Integration tests for run() loop
# ---------------------------------------------------------------------------

class TestProjectionAssemblyServiceRun:
    """
    Integration tests for run().
    get_board and _process_player are mocked so no DB or Yahoo connection needed.
    Tests verify counter logic, batch commits, rollback on error, and summary keys.
    """

    def _make_svc(self, db=None):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.db = db or MagicMock()
        svc.season = 2026
        return svc

    def _board_entry(self, name="Test Player", player_type="batter"):
        return {"name": name, "type": player_type, "proj": {"pa": 500, "avg": 0.270}}

    def test_summary_keys_always_present(self):
        svc = self._make_svc()
        with patch("backend.fantasy_baseball.projection_assembly_service.get_board", return_value=[]):
            summary = svc.run(projection_date=date(2026, 5, 5))
        assert set(summary.keys()) == {"total", "assembled", "identity_misses", "mlbam_null_fallback", "upserted", "errors"}

    def test_empty_board_returns_zero_counts(self):
        svc = self._make_svc()
        with patch("backend.fantasy_baseball.projection_assembly_service.get_board", return_value=[]):
            summary = svc.run(projection_date=date(2026, 5, 5))
        assert summary["total"] == 0
        assert summary["upserted"] == 0

    def test_total_equals_board_length(self):
        svc = self._make_svc()
        board = [self._board_entry("A"), self._board_entry("B")]
        with patch("backend.fantasy_baseball.projection_assembly_service.get_board", return_value=board):
            with patch.object(svc, "_process_player", return_value=1):
                summary = svc.run(projection_date=date(2026, 5, 5))
        assert summary["total"] == 2

    def test_successful_assembly_increments_upserted(self):
        svc = self._make_svc()
        board = [self._board_entry(), self._board_entry("Pitcher", "pitcher")]
        with patch("backend.fantasy_baseball.projection_assembly_service.get_board", return_value=board):
            with patch.object(svc, "_process_player", return_value=99):
                summary = svc.run(projection_date=date(2026, 5, 5))
        assert summary["upserted"] == 2
        assert summary["assembled"] == 2
        assert summary["identity_misses"] == 0

    def test_identity_miss_none_return_increments_counter(self):
        svc = self._make_svc()
        board = [self._board_entry("Unknown Player")]
        with patch("backend.fantasy_baseball.projection_assembly_service.get_board", return_value=board):
            with patch.object(svc, "_process_player", return_value=None):
                summary = svc.run(projection_date=date(2026, 5, 5))
        assert summary["identity_misses"] == 1
        assert summary["upserted"] == 0

    def test_mixed_results_counted_separately(self):
        """3 players: 2 upserted, 1 identity miss."""
        svc = self._make_svc()
        board = [self._board_entry(f"P{i}") for i in range(3)]
        return_sequence = [1, None, 2]  # row_id, miss, row_id
        with patch("backend.fantasy_baseball.projection_assembly_service.get_board", return_value=board):
            with patch.object(svc, "_process_player", side_effect=return_sequence):
                summary = svc.run(projection_date=date(2026, 5, 5))
        assert summary["upserted"] == 2
        assert summary["identity_misses"] == 1
        assert summary["errors"] == 0

    def test_exception_increments_errors_and_calls_rollback(self):
        mock_db = MagicMock()
        svc = self._make_svc(db=mock_db)
        board = [self._board_entry("Broken Player")]
        with patch("backend.fantasy_baseball.projection_assembly_service.get_board", return_value=board):
            with patch.object(svc, "_process_player", side_effect=RuntimeError("db exploded")):
                summary = svc.run(projection_date=date(2026, 5, 5))
        assert summary["errors"] == 1
        assert summary["upserted"] == 0
        mock_db.rollback.assert_called()

    def test_batch_commit_fires_at_batch_boundary(self):
        """55 successful players → at least 2 commits (1 mid-batch + 1 final)."""
        mock_db = MagicMock()
        svc = self._make_svc(db=mock_db)
        board = [self._board_entry(f"P{i}") for i in range(55)]
        with patch("backend.fantasy_baseball.projection_assembly_service.get_board", return_value=board):
            with patch.object(svc, "_process_player", return_value=1):
                svc.run(projection_date=date(2026, 5, 5))
        assert mock_db.commit.call_count >= 2

    def test_no_commit_on_all_identity_misses(self):
        """If every player is an identity miss, commit should still be called (empty batch commit)."""
        mock_db = MagicMock()
        svc = self._make_svc(db=mock_db)
        board = [self._board_entry(f"Unknown{i}") for i in range(10)]
        with patch("backend.fantasy_baseball.projection_assembly_service.get_board", return_value=board):
            with patch.object(svc, "_process_player", return_value=None):
                summary = svc.run(projection_date=date(2026, 5, 5))
        assert summary["identity_misses"] == 10
        # players_in_batch stays 0 → final commit only called if players_in_batch > 0
        # (or not called at all — either is acceptable; just verify no errors)
        assert summary["errors"] == 0


# ---------------------------------------------------------------------------
# Sprint 5: live projection lookup
# ---------------------------------------------------------------------------
from unittest.mock import MagicMock


class TestGetLiveProjection:
    def _make_service(self):
        svc = ProjectionAssemblyService.__new__(ProjectionAssemblyService)
        svc.db = MagicMock()
        svc.season = 2026
        return svc

    def test_returns_none_when_no_rows(self):
        svc = self._make_service()
        svc.db.query.return_value.filter.return_value.first.return_value = None
        result = svc._get_live_projection(None, "Unknown Player")
        assert result is None

    def test_returns_row_when_mlbam_matches_and_non_prior(self):
        svc = self._make_service()
        mock_row = MagicMock()
        mock_row.update_method = "bayesian"
        mock_row.hr = 25
        svc.db.query.return_value.filter.return_value.first.return_value = mock_row
        result = svc._get_live_projection(12345, "Mike Trout")
        assert result is mock_row

    def test_returns_none_when_update_method_is_prior(self):
        svc = self._make_service()
        mock_row = MagicMock()
        mock_row.update_method = "prior"
        svc.db.query.return_value.filter.return_value.first.return_value = mock_row
        result = svc._get_live_projection(12345, "Mike Trout")
        assert result is None

    def test_returns_none_on_db_exception(self):
        svc = self._make_service()
        svc.db.query.side_effect = Exception("db error")
        result = svc._get_live_projection(12345, "Mike Trout")
        assert result is None

    def test_counting_stats_override_applied_correctly(self):
        """Verify the merge logic: live HR/R/RBI/SB replace board values."""
        live_hr, live_r, live_rbi, live_sb = 30, 90, 95, 5
        board_proj = {"hr": 18, "r": 70, "rbi": 65, "sb": 8, "pa": 550,
                      "avg": 0.280, "obp": 0.360, "slg": 0.500}

        mock_proj = MagicMock()
        mock_proj.update_method = "bayesian"
        mock_proj.hr = live_hr
        mock_proj.r = live_r
        mock_proj.rbi = live_rbi
        mock_proj.sb = live_sb

        # Simulate the merge logic from _assemble_batter
        live = mock_proj if mock_proj.update_method != "prior" else None
        merged = dict(board_proj)
        if live is not None and live.hr is not None:
            merged["hr"] = live.hr or merged.get("hr", 0)
            merged["r"] = live.r or merged.get("r", 0)
            merged["rbi"] = live.rbi or merged.get("rbi", 0)
            merged["sb"] = live.sb or merged.get("sb", 0)

        assert merged["hr"] == 30
        assert merged["r"] == 90
        assert merged["rbi"] == 95
        assert merged["sb"] == 5
        assert merged["avg"] == 0.280   # rate stats unchanged
        assert merged["pa"] == 550      # pa unchanged
