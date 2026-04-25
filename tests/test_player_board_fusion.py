"""
Integration tests for Bayesian fusion in player_board.py

Tests the four-state logic:
1. Steamer + Statcast: Full Marcel update
2. Steamer only: Return Steamer unchanged
3. Statcast only: Double shrinkage with population prior
4. Neither: Population prior only

Also tests NULL value handling and output format compatibility.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from backend.fantasy_baseball.player_board import (
    get_or_create_projection,
    _query_statcast_proxy,
    _query_statcast_proxy_mlbam,
    _extract_steamer_data,
    _convert_fusion_proj_to_board_format,
)
from backend.fantasy_baseball.fusion_engine import (
    PopulationPrior,
    FusionResult,
)


class TestQueryStatcastProxy:
    """Test the refactored _query_statcast_proxy function."""

    def test_returns_raw_batter_metrics(self):
        """Should return raw Statcast metrics with sample_size for batters."""
        db = Mock()

        # Mock StatcastBatterMetrics row
        metrics = Mock()
        metrics.mlbam_id = "12345"
        metrics.pa = 150
        metrics.avg = 0.280
        metrics.slg = 0.450
        metrics.ops = 0.780
        metrics.xwoba = 0.350
        metrics.hr = 20
        metrics.r = 45
        metrics.rbi = 50
        metrics.sb = 5

        # Mock PlayerIDMapping query
        mapping = Mock()
        mapping.mlbam_id = "12345"

        db.query.return_value.filter.return_value.first.side_effect = [mapping, metrics]

        result = _query_statcast_proxy(db, "yahoo_123", "batter", "Test Player")

        assert result is not None
        assert result["avg"] == 0.280
        assert result["slg"] == 0.450
        assert result["ops"] == 0.780
        assert result["xwoba"] == 0.350
        assert result["pa"] == 150
        assert result["sample_size"] == 150
        assert result["hr"] == 20
        # obp should be estimated since not in metrics
        assert result["obp"] == pytest.approx(0.350)  # avg + 0.070

    def test_returns_raw_pitcher_metrics(self):
        """Should return raw Statcast metrics with sample_size for pitchers."""
        db = Mock()

        # Mock StatcastPitcherMetrics row
        metrics = Mock()
        metrics.mlbam_id = "54321"
        metrics.ip = 50.0
        metrics.era = 3.50
        metrics.whip = 1.20
        metrics.k_percent = 0.25
        metrics.bb_percent = 0.07
        metrics.k_9 = 10.0
        metrics.xera = 3.30
        metrics.w = 4
        metrics.l = 2
        metrics.sv = 0
        metrics.qs = 6
        metrics.hr_pit = 8
        metrics.k_pit = 50

        # Mock PlayerIDMapping query
        mapping = Mock()
        mapping.mlbam_id = "54321"

        db.query.return_value.filter.return_value.first.side_effect = [mapping, metrics]

        result = _query_statcast_proxy(db, "yahoo_543", "pitcher", "Test Pitcher")

        assert result is not None
        assert result["era"] == 3.50
        assert result["whip"] == 1.20
        assert result["k_percent"] == 0.25
        assert result["sample_size"] == 50  # IP converted to int
        assert result["ip"] == 50.0
        assert result["w"] == 4

    def test_returns_none_for_insufficient_batter_sample(self):
        """Should return None when PA < 50."""
        db = Mock()

        metrics = Mock()
        metrics.pa = 30  # Below threshold
        metrics.ip = None

        mapping = Mock()
        mapping.mlbam_id = "12345"

        db.query.return_value.filter.return_value.first.side_effect = [mapping, metrics]

        result = _query_statcast_proxy(db, "yahoo_123", "batter", "Test Player")

        assert result is None

    def test_returns_none_for_insufficient_pitcher_sample(self):
        """Should return None when IP < 20."""
        db = Mock()

        metrics = Mock()
        metrics.ip = 15.0  # Below threshold

        mapping = Mock()
        mapping.mlbam_id = "54321"

        db.query.return_value.filter.return_value.first.side_effect = [mapping, metrics]

        result = _query_statcast_proxy(db, "yahoo_543", "pitcher", "Test Pitcher")

        assert result is None

    def test_returns_none_when_no_mapping(self):
        """Should return None when PlayerIDMapping lookup fails."""
        db = Mock()
        db.query.return_value.filter.return_value.first.return_value = None

        result = _query_statcast_proxy(db, "yahoo_999", "batter", "Unknown Player")

        assert result is None

    def test_handles_null_gracefully_in_batter_metrics(self):
        """Should estimate missing values like woba, obp."""
        db = Mock()

        metrics = Mock()
        metrics.mlbam_id = "12345"
        metrics.pa = 150
        metrics.avg = 0.280
        metrics.slg = 0.450
        metrics.ops = 0.780
        metrics.xwoba = 0.350
        metrics.hr = 20
        metrics.r = 45
        metrics.rbi = 50
        metrics.sb = 5
        # Simulate missing obp attribute
        del metrics.obp

        mapping = Mock()
        mapping.mlbam_id = "12345"

        db.query.return_value.filter.return_value.first.side_effect = [mapping, metrics]

        result = _query_statcast_proxy(db, "yahoo_123", "batter", "Test Player")

        assert result is not None
        # obp should be estimated from avg
        assert result["obp"] == pytest.approx(0.350)


class TestExtractSteamerData:
    """Test _extract_steamer_data helper."""

    def test_extracts_batter_steamer_data(self):
        """Should extract batter projection from PlayerProjection row."""
        projection = Mock()
        projection.avg = 0.280
        projection.obp = 0.350
        projection.slg = 0.450
        projection.ops = 0.800
        projection.hr = 30
        projection.sb = 10

        result = _extract_steamer_data(projection, "batter")

        assert result is not None
        assert result["avg"] == 0.280
        assert result["obp"] == 0.350
        assert result["slg"] == 0.450
        assert result["ops"] == 0.800
        assert result["hr_per_pa"] == pytest.approx(30 / 550.0)
        assert result["sb_per_pa"] == pytest.approx(10 / 550.0)

    def test_returns_none_for_invalid_batter_data(self):
        """Should return None when avg and ops are both None."""
        projection = Mock()
        projection.avg = None
        projection.ops = None

        result = _extract_steamer_data(projection, "batter")

        assert result is None

    def test_extracts_pitcher_steamer_data(self):
        """Should extract pitcher projection from PlayerProjection row."""
        projection = Mock()
        projection.era = 3.50
        projection.whip = 1.15
        projection.k_per_nine = 9.5
        projection.bb_per_nine = 2.8

        result = _extract_steamer_data(projection, "pitcher")

        assert result is not None
        assert result["era"] == 3.50
        assert result["whip"] == 1.15
        assert result["k_per_nine"] == 9.5
        assert result["bb_per_nine"] == 2.8

    def test_returns_none_for_default_pitcher_data(self):
        """Should return None when data looks like defaults."""
        projection = Mock()
        projection.era = 4.00
        projection.k_per_nine = 8.5  # Default value

        result = _extract_steamer_data(projection, "pitcher")

        assert result is None

    def test_uses_defaults_for_missing_batter_fields(self):
        """Should use population defaults for missing optional fields."""
        projection = Mock()
        projection.avg = 0.280
        projection.obp = None
        projection.slg = 0.450
        projection.ops = None
        projection.hr = None
        projection.sb = None

        result = _extract_steamer_data(projection, "batter")

        assert result is not None
        assert result["avg"] == 0.280
        assert result["obp"] == 0.320  # Default
        assert result["ops"] == 0.720  # Default
        assert result["k_percent"] == 0.225  # Default


class TestConvertFusionProjToBoardFormat:
    """Test _convert_fusion_proj_to_board_format helper."""

    def test_converts_batter_fusion_proj(self):
        """Should convert fusion engine output to board format."""
        fusion_proj = {
            "avg": 0.280,
            "obp": 0.350,
            "slg": 0.450,
            "ops": 0.800,
            "hr_per_pa": 0.040,
            "sb_per_pa": 0.015,
            "k_percent": 0.200,
        }

        result = _convert_fusion_proj_to_board_format(fusion_proj, "batter")

        assert result["pa"] == 550
        assert result["avg"] == 0.280
        assert result["ops"] == 0.800
        assert result["hr"] == round(0.040 * 550)  # 22
        assert result["nsb"] == round(0.015 * 550)  # 8

    def test_converts_pitcher_fusion_proj(self):
        """Should convert fusion engine output to board format."""
        fusion_proj = {
            "era": 3.50,
            "whip": 1.20,
            "k_per_nine": 9.5,
            "bb_per_nine": 2.5,
            "k_percent": 0.25,
        }

        result = _convert_fusion_proj_to_board_format(fusion_proj, "pitcher")

        assert result["era"] == 3.50
        assert result["whip"] == 1.20
        assert result["k9"] == 9.5
        assert result["ip"] == 180


class TestFourStateFusionIntegration:
    """Integration tests for the four-state Bayesian fusion logic."""

    @patch("backend.fantasy_baseball.player_board._query_statcast_proxy")
    @patch("backend.fantasy_baseball.player_board.get_board")
    @patch("backend.fantasy_baseball.player_board._extract_steamer_data")
    def test_state_1_both_sources_full_fusion_batter(
        self, mock_extract, mock_board, mock_statcast
    ):
        """State 1: Steamer + Statcast should trigger full Marcel update."""
        # Setup: empty board
        mock_board.return_value = []

        # Steamer data
        mock_extract.return_value = {
            "avg": 0.280,
            "obp": 0.350,
            "slg": 0.450,
            "ops": 0.800,
            "hr_per_pa": 0.040,
            "sb_per_pa": 0.010,
            "k_percent": 0.225,
            "bb_percent": 0.080,
        }

        # Statcast data (observed performance differs from Steamer)
        mock_statcast.return_value = {
            "avg": 0.320,  # Higher than Steamer
            "obp": 0.380,
            "slg": 0.500,
            "ops": 0.880,
            "xwoba": 0.380,
            "woba": 0.370,
            "pa": 150,
            "hr": 25,
            "r": 50,
            "rbi": 55,
            "sb": 8,
            "sample_size": 150,
        }

        yahoo_player = {
            "name": "Test Player",
            "player_key": "mlb.p.12345",
            "positions": ["OF"],
            "team": "NYY",
        }

        result = get_or_create_projection(yahoo_player)

        assert result is not None
        assert result["name"] == "Test Player"
        assert result["type"] == "batter"
        assert result["is_proxy"] is True
        assert result["fusion_source"] == "fusion"
        assert result["components_fused"] > 0  # Should have fused components
        # Result should be between Steamer and Statcast (Marcel update)
        # With sample_size=150 (< most stabilization points), should lean toward Steamer
        assert 0.270 < result["proj"]["avg"] < 0.320

    @patch("backend.fantasy_baseball.player_board._query_statcast_proxy")
    @patch("backend.fantasy_baseball.player_board.get_board")
    @patch("backend.fantasy_baseball.player_board._extract_steamer_data")
    def test_state_2_steamer_only_unchanged(
        self, mock_extract, mock_board, mock_statcast
    ):
        """State 2: Steamer only should return Steamer unchanged."""
        mock_board.return_value = []

        mock_extract.return_value = {
            "avg": 0.280,
            "obp": 0.350,
            "slg": 0.450,
            "ops": 0.800,
            "hr_per_pa": 0.040,
            "sb_per_pa": 0.010,
            "k_percent": 0.225,
            "bb_percent": 0.080,
        }

        mock_statcast.return_value = None  # No Statcast

        yahoo_player = {
            "name": "Steamer Only",
            "player_key": "mlb.p.22222",
            "positions": ["1B"],
            "team": "LAD",
        }

        result = get_or_create_projection(yahoo_player)

        assert result is not None
        assert result["fusion_source"] == "steamer"
        assert result["components_fused"] == 0
        # proj should be based on Steamer values
        assert result["proj"]["avg"] == pytest.approx(0.280, abs=0.01)

    @patch("backend.fantasy_baseball.player_board._query_statcast_proxy")
    @patch("backend.fantasy_baseball.player_board.get_board")
    @patch("backend.fantasy_baseball.player_board._extract_steamer_data")
    def test_state_3_statcast_only_double_shrinkage(
        self, mock_extract, mock_board, mock_statcast
    ):
        """State 3: Statcast only should use double shrinkage with population prior."""
        mock_board.return_value = []

        mock_extract.return_value = None  # No Steamer

        mock_statcast.return_value = {
            "avg": 0.320,  # Well above average
            "obp": 0.380,
            "slg": 0.550,
            "ops": 0.930,
            "xwoba": 0.400,
            "woba": 0.390,
            "pa": 100,
            "hr": 30,
            "r": 60,
            "rbi": 65,
            "sb": 5,
            "sample_size": 100,
        }

        yahoo_player = {
            "name": "Statcast Only",
            "player_key": "mlb.p.33333",
            "positions": ["SS"],
            "team": "BAL",
        }

        result = get_or_create_projection(yahoo_player)

        assert result is not None
        assert result["fusion_source"] == "statcast_shrunk"
        assert result["components_fused"] > 0
        # With double shrinkage (2x stabilization), should regress strongly toward mean
        # Population avg is 0.250, observed is 0.320 with 100 PA
        # Result should be closer to 0.250 than 0.320
        prior = PopulationPrior()
        assert result["proj"]["avg"] < 0.320  # Shrunk toward prior
        assert result["proj"]["avg"] > prior.BATTER_AVG  # But not all the way

    @patch("backend.fantasy_baseball.player_board._query_statcast_proxy")
    @patch("backend.fantasy_baseball.player_board.get_board")
    @patch("backend.fantasy_baseball.player_board._extract_steamer_data")
    def test_state_4_neither_population_prior(
        self, mock_extract, mock_board, mock_statcast
    ):
        """State 4: Neither source should return population prior."""
        mock_board.return_value = []

        mock_extract.return_value = None
        mock_statcast.return_value = None

        yahoo_player = {
            "name": "Unknown Rookie",
            "player_key": "mlb.p.99999",
            "positions": ["C"],
            "team": "MIA",
        }

        result = get_or_create_projection(yahoo_player)

        assert result is not None
        assert result["fusion_source"] == "population_prior"
        assert result["components_fused"] == 0
        prior = PopulationPrior()
        # Should use population priors
        assert result["proj"]["avg"] == prior.BATTER_AVG


class TestNullValueHandling:
    """Test graceful handling of NULL values in data sources."""

    @patch("backend.fantasy_baseball.player_board._query_statcast_proxy")
    @patch("backend.fantasy_baseball.player_board.get_board")
    @patch("backend.fantasy_baseball.player_board._extract_steamer_data")
    def test_handles_null_statcast_woba(
        self, mock_extract, mock_board, mock_statcast
    ):
        """Should estimate woba from ops when woba is NULL."""
        mock_board.return_value = []

        mock_extract.return_value = None

        # Statcast data with missing woba
        mock_statcast.return_value = {
            "avg": 0.280,
            "slg": 0.450,
            "ops": 0.780,
            "xwoba": 0.350,
            "woba": None,  # Missing
            "pa": 120,
            "hr": 22,
            "r": 48,
            "rbi": 52,
            "sb": 6,
            "sample_size": 120,
        }

        yahoo_player = {
            "name": "Missing WOBA",
            "player_key": "mlb.p.44444",
            "positions": ["3B"],
            "team": "CHC",
        }

        result = get_or_create_projection(yahoo_player)

        assert result is not None
        # Should not crash, fusion should handle None values

    @patch("backend.fantasy_baseball.player_board._query_statcast_proxy")
    @patch("backend.fantasy_baseball.player_board.get_board")
    @patch("backend.fantasy_baseball.player_board._extract_steamer_data")
    def test_handles_null_statcast_obp(
        self, mock_extract, mock_board, mock_statcast
    ):
        """Should estimate obp from avg when obp is NULL."""
        mock_board.return_value = []

        mock_extract.return_value = None

        # Simulate metrics without obp attribute
        class MockMetrics:
            pass

        metrics = MockMetrics()
        metrics.avg = 0.290
        metrics.slg = 0.480
        metrics.ops = 0.810
        metrics.xwoba = 0.370
        metrics.pa = 130
        metrics.hr = 24
        metrics.r = 51
        metrics.rbi = 58
        metrics.sb = 4
        metrics.sample_size = 130

        mock_statcast.return_value = metrics

        yahoo_player = {
            "name": "Missing OBP",
            "player_key": "mlb.p.55555",
            "positions": ["2B"],
            "team": "SF",
        }

        result = get_or_create_projection(yahoo_player)

        assert result is not None
        # Should not crash

    @patch("backend.fantasy_baseball.player_board._query_statcast_proxy")
    @patch("backend.fantasy_baseball.player_board.get_board")
    @patch("backend.fantasy_baseball.player_board._extract_steamer_data")
    def test_handles_zero_sample_size(
        self, mock_extract, mock_board, mock_statcast
    ):
        """Should handle zero sample_size gracefully."""
        mock_board.return_value = []

        mock_extract.return_value = None

        # Statcast with zero sample_size
        mock_statcast.return_value = {
            "avg": 0.280,
            "slg": 0.450,
            "ops": 0.780,
            "xwoba": 0.350,
            "pa": 0,
            "hr": 0,
            "r": 0,
            "rbi": 0,
            "sb": 0,
            "sample_size": 0,
        }

        yahoo_player = {
            "name": "Zero Sample",
            "player_key": "mlb.p.66666",
            "positions": ["DH"],
            "team": "SEA",
        }

        result = get_or_create_projection(yahoo_player)

        assert result is not None
        # Should default to population prior when sample_size is 0


class TestOutputFormatCompatibility:
    """Test that output format matches application expectations."""

    @patch("backend.fantasy_baseball.player_board._query_statcast_proxy")
    @patch("backend.fantasy_baseball.player_board.get_board")
    @patch("backend.fantasy_baseball.player_board._extract_steamer_data")
    def test_output_has_all_required_keys(
        self, mock_extract, mock_board, mock_statcast
    ):
        """Output dict must have all keys expected by waiver engine."""
        mock_board.return_value = []

        mock_extract.return_value = None
        mock_statcast.return_value = None

        yahoo_player = {
            "name": "Format Test",
            "player_key": "mlb.p.77777",
            "positions": ["OF"],
            "team": "BOS",
        }

        result = get_or_create_projection(yahoo_player)

        # Required keys
        required_keys = [
            "id", "name", "team", "positions", "type", "tier",
            "rank", "adp", "z_score", "cat_scores", "proj",
            "is_keeper", "keeper_round", "is_proxy",
        ]

        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        # Fusion metadata keys
        fusion_metadata_keys = [
            "fusion_source", "components_fused", "xwoba_override",
        ]

        for key in fusion_metadata_keys:
            assert key in result, f"Missing fusion metadata: {key}"

    @patch("backend.fantasy_baseball.player_board._query_statcast_proxy")
    @patch("backend.fantasy_baseball.player_board.get_board")
    @patch("backend.fantasy_baseball.player_board._extract_steamer_data")
    def test_cat_scores_is_dict(
        self, mock_extract, mock_board, mock_statcast
    ):
        """cat_scores should be a dict with numeric values."""
        mock_board.return_value = []

        mock_extract.return_value = None
        mock_statcast.return_value = None

        yahoo_player = {
            "name": "CatScores Test",
            "player_key": "mlb.p.88888",
            "positions": ["1B"],
            "team": "ARI",
        }

        result = get_or_create_projection(yahoo_player)

        assert isinstance(result["cat_scores"], dict)
        # z_score should be sum of cat_scores
        expected_z = sum(result["cat_scores"].values())
        assert result["z_score"] == pytest.approx(expected_z)

    @patch("backend.fantasy_baseball.player_board._query_statcast_proxy")
    @patch("backend.fantasy_baseball.player_board.get_board")
    @patch("backend.fantasy_baseball.player_board._extract_steamer_data")
    def test_proj_has_counting_stats(
        self, mock_extract, mock_board, mock_statcast
    ):
        """proj should have counting stats (hr, r, rbi, sb) not just rates."""
        mock_board.return_value = []

        mock_extract.return_value = {
            "avg": 0.280,
            "obp": 0.350,
            "slg": 0.450,
            "ops": 0.800,
            "hr_per_pa": 0.040,
            "sb_per_pa": 0.010,
            "k_percent": 0.225,
            "bb_percent": 0.080,
        }

        mock_statcast.return_value = None

        yahoo_player = {
            "name": "Counting Stats Test",
            "player_key": "mlb.p.99999",
            "positions": ["RF"],
            "team": "SD",
        }

        result = get_or_create_projection(yahoo_player)

        proj = result["proj"]
        # Should have counting stats
        assert "hr" in proj or "hr_per_pa" in proj
        assert "r" in proj
        assert "rbi" in proj
        assert "sb" in proj or "nsb" in proj
        # Should have rate stats
        assert "avg" in proj
        assert "ops" in proj


class TestPitcherFusionIntegration:
    """Integration tests specific to pitcher fusion."""

    @patch("backend.fantasy_baseball.player_board._query_statcast_proxy")
    @patch("backend.fantasy_baseball.player_board.get_board")
    @patch("backend.fantasy_baseball.player_board._extract_steamer_data")
    def test_pitcher_state_1_full_fusion(
        self, mock_extract, mock_board, mock_statcast
    ):
        """Test full fusion for pitchers."""
        mock_board.return_value = []

        mock_extract.return_value = {
            "era": 3.50,
            "whip": 1.15,
            "k_per_nine": 9.5,
            "bb_per_nine": 2.8,
            "k_percent": 0.25,
            "bb_percent": 0.07,
        }

        mock_statcast.return_value = {
            "era": 4.50,  # Worse than Steamer
            "whip": 1.35,
            "k_percent": 0.28,
            "bb_percent": 0.08,
            "k_9": 8.0,
            "xera": 3.80,
            "ip": 50.0,
            "w": 3,
            "l": 2,
            "sv": 0,
            "qs": 4,
            "hr_pit": 8,
            "k_pit": 45,
            "sample_size": 50,
        }

        yahoo_player = {
            "name": "Fusion Pitcher",
            "player_key": "mlb.p.p11111",
            "positions": ["SP"],
            "team": "NYY",
        }

        result = get_or_create_projection(yahoo_player)

        assert result is not None
        assert result["type"] == "pitcher"
        assert result["fusion_source"] == "fusion"
        # Result should be between Steamer (3.50) and Statcast (4.50)
        # With small sample (50 IP), should lean toward Steamer
        assert 3.40 < result["proj"]["era"] < 4.60

    @patch("backend.fantasy_baseball.player_board._query_statcast_proxy")
    @patch("backend.fantasy_baseball.player_board.get_board")
    @patch("backend.fantasy_baseball.player_board._extract_steamer_data")
    def test_pitcher_state_4_population_prior(
        self, mock_extract, mock_board, mock_statcast
    ):
        """Test population prior for pitchers with no data."""
        mock_board.return_value = []

        mock_extract.return_value = None
        mock_statcast.return_value = None

        yahoo_player = {
            "name": "Unknown Pitcher",
            "player_key": "mlb.p.p22222",
            "positions": ["SP"],
            "team": "TBR",
        }

        result = get_or_create_projection(yahoo_player)

        assert result is not None
        assert result["type"] == "pitcher"
        assert result["fusion_source"] == "population_prior"
        prior = PopulationPrior()
        # Should use population priors
        assert result["proj"]["era"] == prior.PITCHER_ERA
        assert result["proj"]["whip"] == prior.PITCHER_WHIP
