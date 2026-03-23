"""
K-15 Oracle Validation — unit tests.

Run with:
    pytest tests/test_oracle_validator.py -v
"""

import pytest
from backend.services.oracle_validator import (
    OracleResult,
    calculate_oracle_divergence,
    _select_threshold,
    ORACLE_THRESHOLD_Z_EARLY,
    ORACLE_THRESHOLD_Z_MID,
    ORACLE_THRESHOLD_Z_LATE,
    ORACLE_SD,
)


# ---------------------------------------------------------------------------
# _select_threshold
# ---------------------------------------------------------------------------

class TestSelectThreshold:
    def test_none_hours_returns_early(self):
        assert _select_threshold(None) == ORACLE_THRESHOLD_Z_EARLY

    def test_24_hours_returns_early(self):
        assert _select_threshold(24.0) == ORACLE_THRESHOLD_Z_EARLY

    def test_48_hours_returns_early(self):
        assert _select_threshold(48.0) == ORACLE_THRESHOLD_Z_EARLY

    def test_12_hours_returns_mid(self):
        assert _select_threshold(12.0) == ORACLE_THRESHOLD_Z_MID

    def test_4_hours_returns_mid(self):
        assert _select_threshold(4.0) == ORACLE_THRESHOLD_Z_MID

    def test_3_hours_returns_late(self):
        assert _select_threshold(3.0) == ORACLE_THRESHOLD_Z_LATE

    def test_0_hours_returns_late(self):
        assert _select_threshold(0.0) == ORACLE_THRESHOLD_Z_LATE


# ---------------------------------------------------------------------------
# calculate_oracle_divergence — happy paths
# ---------------------------------------------------------------------------

class TestCalculateOracleDivergence:
    """Known-input tests covering the divergence math exactly."""

    def test_returns_none_with_no_ratings(self):
        result = calculate_oracle_divergence(
            model_spread=5.0,
            kenpom_home=None,
            kenpom_away=None,
            barttorvik_home=None,
            barttorvik_away=None,
        )
        assert result is None

    def test_returns_none_with_partial_kenpom(self):
        """One side of KenPom available but not both — should skip that source."""
        result = calculate_oracle_divergence(
            model_spread=5.0,
            kenpom_home=10.0,
            kenpom_away=None,
            barttorvik_home=None,
            barttorvik_away=None,
        )
        assert result is None

    def test_single_source_kenpom(self):
        """Only KenPom available — oracle_spread = home - away."""
        # KenPom margin: 8.0 (home AdjEM) - 2.0 (away AdjEM) = +6.0
        # Model spread = 5.0, divergence = |5.0 - 6.0| = 1.0
        result = calculate_oracle_divergence(
            model_spread=5.0,
            kenpom_home=8.0,
            kenpom_away=2.0,
            barttorvik_home=None,
            barttorvik_away=None,
        )
        assert result is not None
        assert result.oracle_spread == pytest.approx(6.0)
        assert result.model_spread == pytest.approx(5.0)
        assert result.divergence_points == pytest.approx(1.0)
        assert result.divergence_z == pytest.approx(1.0 / ORACLE_SD)
        assert "kenpom" in result.sources
        assert "barttorvik" not in result.sources

    def test_two_source_consensus(self):
        """Both KenPom and BartTorvik — oracle is the average."""
        # KenPom: 10 - 4 = 6.0
        # BartTorvik: 12 - 4 = 8.0
        # Consensus: (6.0 + 8.0) / 2 = 7.0
        # Model = 5.0, divergence = |5.0 - 7.0| = 2.0
        result = calculate_oracle_divergence(
            model_spread=5.0,
            kenpom_home=10.0,
            kenpom_away=4.0,
            barttorvik_home=12.0,
            barttorvik_away=4.0,
        )
        assert result is not None
        assert result.oracle_spread == pytest.approx(7.0)
        assert result.divergence_points == pytest.approx(2.0)
        assert result.divergence_z == pytest.approx(2.0 / ORACLE_SD)
        assert set(result.sources) == {"kenpom", "barttorvik"}

    def test_flagged_when_z_exceeds_threshold(self):
        """Divergence large enough to trigger flag at default early threshold (2.0)."""
        # Make divergence = ORACLE_SD * 2.5 = 10.0 points → z = 2.5 ≥ 2.0
        result = calculate_oracle_divergence(
            model_spread=0.0,
            kenpom_home=10.0,
            kenpom_away=0.0,   # oracle = 10.0
            barttorvik_home=None,
            barttorvik_away=None,
            hours_to_tipoff=None,  # uses early threshold = 2.0
            oracle_sd=4.0,
        )
        # model=0.0, oracle=10.0, divergence=10.0, z=10/4=2.5 ≥ 2.0 → flagged
        assert result is not None
        assert result.divergence_z == pytest.approx(2.5)
        assert result.flagged is True

    def test_not_flagged_below_threshold(self):
        """Divergence below threshold — not flagged."""
        # oracle=6.0, model=5.0, divergence=1.0, z=0.25 < 2.0
        result = calculate_oracle_divergence(
            model_spread=5.0,
            kenpom_home=8.0,
            kenpom_away=2.0,
            barttorvik_home=None,
            barttorvik_away=None,
            hours_to_tipoff=None,
            oracle_sd=4.0,
        )
        assert result is not None
        assert result.flagged is False

    def test_time_window_late_higher_threshold(self):
        """Late window uses higher threshold — same divergence may not flag."""
        # z = 2.2 → flags at early (2.0) but not at late (3.0)
        divergence = 2.2 * ORACLE_SD  # points needed for z=2.2
        result_early = calculate_oracle_divergence(
            model_spread=0.0,
            kenpom_home=divergence,
            kenpom_away=0.0,
            barttorvik_home=None,
            barttorvik_away=None,
            hours_to_tipoff=25.0,  # early window
        )
        result_late = calculate_oracle_divergence(
            model_spread=0.0,
            kenpom_home=divergence,
            kenpom_away=0.0,
            barttorvik_home=None,
            barttorvik_away=None,
            hours_to_tipoff=1.0,  # late window
        )
        assert result_early is not None and result_early.flagged is True
        assert result_late is not None and result_late.flagged is False

    def test_negative_divergence_uses_abs(self):
        """Model below oracle — divergence is still positive."""
        # oracle = 6.0, model = 2.0, divergence_points = 4.0
        result = calculate_oracle_divergence(
            model_spread=2.0,
            kenpom_home=8.0,
            kenpom_away=2.0,
            barttorvik_home=None,
            barttorvik_away=None,
        )
        assert result is not None
        assert result.divergence_points > 0

    def test_zero_divergence_not_flagged(self):
        """Model exactly matches oracle — zero divergence, never flagged."""
        result = calculate_oracle_divergence(
            model_spread=6.0,
            kenpom_home=8.0,
            kenpom_away=2.0,
            barttorvik_home=None,
            barttorvik_away=None,
        )
        assert result is not None
        assert result.divergence_points == pytest.approx(0.0)
        assert result.divergence_z == pytest.approx(0.0)
        assert result.flagged is False


# ---------------------------------------------------------------------------
# OracleResult.to_dict()
# ---------------------------------------------------------------------------

class TestOracleResultToDict:
    def test_to_dict_keys(self):
        result = calculate_oracle_divergence(
            model_spread=5.0,
            kenpom_home=8.0,
            kenpom_away=2.0,
            barttorvik_home=10.0,
            barttorvik_away=2.0,
        )
        assert result is not None
        d = result.to_dict()
        expected_keys = {
            "oracle_spread", "model_spread", "divergence_points",
            "divergence_z", "threshold_z", "flagged", "sources",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_values_are_rounded(self):
        result = calculate_oracle_divergence(
            model_spread=5.1234567,
            kenpom_home=8.9999999,
            kenpom_away=2.1111111,
            barttorvik_home=None,
            barttorvik_away=None,
        )
        assert result is not None
        d = result.to_dict()
        # All float values should have ≤ 3 decimal places
        for key in ("oracle_spread", "model_spread", "divergence_points", "divergence_z"):
            val = d[key]
            assert isinstance(val, float)
            assert len(str(val).split(".")[-1]) <= 3

    def test_to_dict_sources_is_list(self):
        result = calculate_oracle_divergence(
            model_spread=5.0,
            kenpom_home=8.0,
            kenpom_away=2.0,
            barttorvik_home=10.0,
            barttorvik_away=2.0,
        )
        assert result is not None
        d = result.to_dict()
        assert isinstance(d["sources"], list)
        assert sorted(d["sources"]) == ["barttorvik", "kenpom"]
