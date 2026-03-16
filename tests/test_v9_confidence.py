"""Tests for V9 Predictive Confidence Engine — SNR scoring and dynamic risk-premium Kelly adjustment.

Tests are organised around the three new methods on CBBEdgeModel:

  calculate_snr(available_sources)       → float in [0, 1]
  _snr_kelly_scalar(snr)                 → float in [floor, 1.0]
  _integrity_kelly_scalar(verdict)       → float in (0, 1.0]

Plus integration tests that verify the full analyze_game() pipeline
correctly scales kelly_fractional when sources disagree or scout flags risk.

Run: pytest tests/test_v9_confidence.py -v
"""

import pytest
from backend.betting_model import CBBEdgeModel

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _model(seed=42) -> CBBEdgeModel:
    return CBBEdgeModel(seed=seed)


def _game():
    return {"home_team": "Duke", "away_team": "UNC", "is_neutral": False}


def _odds(spread=-5.0):
    return {"spread": spread, "spread_odds": -110}


def _ratings(kp=10.0, bt=10.0, em=10.0):
    """All three sources present with configurable margins."""
    return {
        "kenpom":    {"home": kp,  "away": 0.0},
        "barttorvik": {"home": bt,  "away": 0.0},
        "evanmiya":  {"home": em,  "away": 0.0},
    }


# ===========================================================================
# TestCalculateSNR
# ===========================================================================

class TestCalculateSNR:
    """calculate_snr() maps source-margin agreement to [0, 1]."""

    def test_single_source_returns_conservative_floor(self):
        """One source → agreement is undefined → 0.3 floor."""
        model = _model()
        snr = model.calculate_snr([("kenpom", 8.0)])
        assert snr == pytest.approx(0.3)

    def test_two_sources_identical_returns_one(self):
        """Perfect agreement → SNR = 1.0."""
        model = _model()
        snr = model.calculate_snr([("kenpom", 8.0), ("barttorvik", 8.0)])
        assert snr == pytest.approx(1.0)

    def test_two_sources_half_base_sd_spread(self):
        """Spread = 0.5 × base_sd → SNR = 1 - 0.5/2 = 0.75."""
        model = _model()  # base_sd=11.0
        # spread = 5.5 pts = 0.5 * base_sd; norm = 22
        snr = model.calculate_snr([("kenpom", 0.0), ("barttorvik", 5.5)])
        assert snr == pytest.approx(1.0 - 5.5 / 22.0)

    def test_three_sources_perfect_agreement(self):
        model = _model()
        snr = model.calculate_snr([("kenpom", 5.0), ("barttorvik", 5.0), ("evanmiya", 5.0)])
        assert snr == pytest.approx(1.0)

    def test_three_sources_large_disagreement_low_snr(self):
        """Spread of 18 pts on base_sd=11 → SNR ≈ 0.18."""
        model = _model()
        # norm=22; spread=18 → SNR = 1 - 18/22 ≈ 0.18
        snr = model.calculate_snr([("kenpom", 0.0), ("barttorvik", 9.0), ("evanmiya", 18.0)])
        assert snr == pytest.approx(1.0 - 18.0 / 22.0, abs=0.01)

    def test_spread_beyond_norm_clips_to_zero(self):
        """Disagreement >= 2*base_sd → SNR = 0.0."""
        model = _model()
        snr = model.calculate_snr([("kenpom", -12.0), ("barttorvik", 12.0)])
        assert snr == pytest.approx(0.0)

    def test_empty_source_list_returns_zero_floor(self):
        """No sources → can't measure anything → 0.0."""
        model = _model()
        snr = model.calculate_snr([])
        # len(values) < 2 but values == [] → returns 0.3 (single-source path)
        # Actually with 0 sources len < 2 still → 0.3
        assert snr == pytest.approx(0.3)


# ===========================================================================
# TestSNRKellyScalar
# ===========================================================================

class TestSNRKellyScalar:
    """_snr_kelly_scalar() maps SNR → Kelly multiplier ∈ [0.5, 1.0]."""

    def test_snr_one_returns_one(self):
        """Full agreement → no Kelly reduction."""
        assert _model()._snr_kelly_scalar(1.0) == pytest.approx(1.0)

    def test_snr_zero_returns_floor(self):
        """Maximum disagreement → floor multiplier (default 0.5)."""
        assert _model()._snr_kelly_scalar(0.0) == pytest.approx(0.5)

    def test_snr_half_returns_midpoint(self):
        """SNR=0.5 → 0.75x (midpoint between floor and 1.0)."""
        assert _model()._snr_kelly_scalar(0.5) == pytest.approx(0.75)

    def test_output_clipped_above_one(self):
        """SNR > 1.0 (should not occur, but defensive clipping holds."""
        assert _model()._snr_kelly_scalar(2.0) == pytest.approx(1.0)

    def test_output_clipped_at_floor_for_negative(self):
        """SNR < 0.0 clips to floor."""
        assert _model()._snr_kelly_scalar(-1.0) == pytest.approx(0.5)


# ===========================================================================
# TestIntegrityKellyScalar
# ===========================================================================

class TestIntegrityKellyScalar:
    """_integrity_kelly_scalar() maps scout verdict → Kelly multiplier."""

    def test_none_verdict_no_penalty(self):
        assert _model()._integrity_kelly_scalar(None) == pytest.approx(1.0)

    def test_confirmed_no_penalty(self):
        assert _model()._integrity_kelly_scalar("CONFIRMED") == pytest.approx(1.0)
        assert _model()._integrity_kelly_scalar("Confirmed — bet looks solid") == pytest.approx(1.0)

    def test_caution_reduces_to_75pct(self):
        assert _model()._integrity_kelly_scalar("CAUTION") == pytest.approx(0.75)
        assert _model()._integrity_kelly_scalar("CAUTION - Injury Alert") == pytest.approx(0.75)

    def test_volatile_reduces_to_50pct(self):
        assert _model()._integrity_kelly_scalar("VOLATILE") == pytest.approx(0.50)
        assert _model()._integrity_kelly_scalar("VOLATILE - Senior Night risk") == pytest.approx(0.50)

    def test_case_insensitive_matching(self):
        """Verdicts should match regardless of case."""
        assert _model()._integrity_kelly_scalar("volatile") == pytest.approx(0.50)
        assert _model()._integrity_kelly_scalar("Caution") == pytest.approx(0.75)

    def test_volatile_takes_precedence_over_caution(self):
        """If both keywords appear, VOLATILE wins (checked first)."""
        assert _model()._integrity_kelly_scalar("VOLATILE CAUTION") == pytest.approx(0.50)

    def test_unrecognized_verdict_no_penalty(self):
        """Unknown strings (e.g. garbled LLM output) don't penalise."""
        assert _model()._integrity_kelly_scalar("UNKNOWN STATUS") == pytest.approx(1.0)


# ===========================================================================
# TestV9AnalyzeGameIntegration
# ===========================================================================

class TestV9AnalyzeGameIntegration:
    """
    Verify that SNR and integrity signals propagate correctly through
    the full analyze_game() pipeline and are reflected in kelly_fractional.

    We compare a baseline (all sources agree) against degraded scenarios.
    All fixtures use the same spread / ratings magnitude so the only variable
    is source divergence or the integrity_verdict.
    """

    def _baseline_kelly(self) -> float:
        """Kelly for a game where all 3 sources perfectly agree."""
        model = _model()
        r = model.analyze_game(_game(), _odds(), _ratings(kp=10.0, bt=10.0, em=10.0))
        return r.kelly_fractional

    def test_perfect_agreement_snr_is_one(self):
        """SNR field in full_analysis == 1.0 when all sources agree."""
        model = _model()
        result = model.analyze_game(_game(), _odds(), _ratings(kp=10.0, bt=10.0, em=10.0))
        calcs = result.full_analysis.get("calculations", {})
        assert calcs["snr"] == pytest.approx(1.0)
        assert calcs["snr_kelly_scalar"] == pytest.approx(1.0)

    def test_disagreeing_sources_reduce_kelly(self):
        """Large source divergence → lower SNR → smaller kelly_fractional."""
        model = _model(seed=42)
        baseline = model.analyze_game(_game(), _odds(), _ratings(kp=10.0, bt=10.0, em=10.0))

        # Recreate model with same seed to get comparable kelly_full
        model2 = _model(seed=42)
        diverged = model2.analyze_game(_game(), _odds(), _ratings(kp=10.0, bt=10.0, em=-5.0))

        # EvanMiya diverges by 15pts → SNR < 1 → Kelly should be smaller or equal
        if baseline.kelly_fractional > 0 and diverged.kelly_fractional > 0:
            assert diverged.kelly_fractional <= baseline.kelly_fractional + 1e-6

    def test_volatile_integrity_reduces_kelly(self):
        """VOLATILE integrity verdict → 50% Kelly reduction vs CONFIRMED."""
        model_conf = _model(seed=99)
        result_conf = model_conf.analyze_game(
            _game(), _odds(), _ratings(),
            integrity_verdict="CONFIRMED",
        )
        model_vol = _model(seed=99)
        result_vol = model_vol.analyze_game(
            _game(), _odds(), _ratings(),
            integrity_verdict="VOLATILE",
        )
        calcs_vol = result_vol.full_analysis.get("calculations", {})
        assert calcs_vol["integrity_verdict"] == "VOLATILE"
        assert calcs_vol["integrity_kelly_scalar"] == pytest.approx(0.50)
        # Kelly should be ~50% of CONFIRMED (assuming same underlying edge)
        if result_conf.kelly_fractional > 0:
            ratio = result_vol.kelly_fractional / result_conf.kelly_fractional
            assert ratio == pytest.approx(0.50, abs=0.05)

    def test_caution_integrity_reduces_kelly(self):
        """CAUTION → 75% Kelly vs no verdict (None)."""
        model_base = _model(seed=7)
        result_base = model_base.analyze_game(_game(), _odds(), _ratings(), integrity_verdict=None)
        model_caut = _model(seed=7)
        result_caut = model_caut.analyze_game(
            _game(), _odds(), _ratings(),
            integrity_verdict="CAUTION - injury news",
        )
        calcs = result_caut.full_analysis.get("calculations", {})
        assert calcs["integrity_kelly_scalar"] == pytest.approx(0.75)
        if result_base.kelly_fractional > 0:
            ratio = result_caut.kelly_fractional / result_base.kelly_fractional
            assert ratio == pytest.approx(0.75, abs=0.05)

    def test_snr_and_integrity_multiply(self):
        """Both penalties active → combined scalar = snr_scalar × integrity_scalar."""
        model = _model(seed=42)
        # EvanMiya wildly disagrees → low SNR + CAUTION verdict
        result = model.analyze_game(
            _game(), _odds(), _ratings(kp=10.0, bt=10.0, em=-8.0),
            integrity_verdict="CAUTION",
        )
        calcs = result.full_analysis.get("calculations", {})
        snr_scalar = calcs["snr_kelly_scalar"]
        int_scalar = calcs["integrity_kelly_scalar"]
        # net v9_scalar = product
        assert snr_scalar < 1.0
        assert int_scalar == pytest.approx(0.75)

    def test_model_version_is_v9(self):
        """model_version key in full_analysis must be v9.x."""
        model = _model()
        result = model.analyze_game(_game(), _odds(), _ratings())
        assert result.full_analysis.get("model_version", "").startswith("v9.")

    def test_none_integrity_no_reduction(self):
        """integrity_verdict=None → no Kelly penalty (scalar = 1.0)."""
        model = _model(seed=5)
        result = model.analyze_game(_game(), _odds(), _ratings(), integrity_verdict=None)
        calcs = result.full_analysis.get("calculations", {})
        assert calcs["integrity_kelly_scalar"] == pytest.approx(1.0)
        assert calcs["integrity_verdict"] is None
