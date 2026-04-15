"""
Tests for P19 Explainability Layer (backend/services/explainability_layer.py).

All tests are pure-function, no DB required.
Shared helper _make_input(**overrides) supplies valid defaults; each test
overrides only the fields it cares about.
"""

import pytest
from datetime import date

from backend.services.explainability_layer import (
    ExplanationInput,
    ExplanationFactor,
    ExplanationResult,
    _label_z,
    _factor_weight,
    _build_hitter_factors,
    _build_pitcher_factors,
    _build_summary,
    _build_confidence_narrative,
    _build_risk_narrative,
    _build_track_record_narrative,
    explain,
    explain_batch,
)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _make_input(**overrides) -> ExplanationInput:
    """Return a valid ExplanationInput with sensible defaults."""
    defaults = dict(
        decision_id=1,
        as_of_date=date(2026, 4, 5),
        decision_type="lineup",
        bdl_player_id=12345,
        player_name="Aaron Judge",
        target_slot="OF",
        drop_player_id=None,
        drop_player_name=None,
        lineup_score=78.5,
        value_gain=None,
        decision_confidence=0.85,
        player_type="hitter",
        score_0_100=78.5,
        composite_z=1.2,
        z_hr=1.8,
        z_rbi=1.1,
        z_sb=0.3,
        z_nsb=0.25,     # NSB correlates closely with SB; slight divergence for realism
        z_avg=0.8,
        z_obp=0.9,
        z_era=None,
        z_whip=None,
        z_k_per_9=None,
        score_confidence=0.85,
        games_in_window=12,
        signal="HOT",
        delta_z=0.4,
        proj_hr_p50=22.5,
        proj_rbi_p50=65.0,
        proj_sb_p50=3.0,
        proj_avg_p50=0.285,
        proj_k_p50=None,
        proj_era_p50=None,
        proj_whip_p50=None,
        prob_above_median=0.65,
        downside_p25=55.0,
        upside_p75=82.0,
        backtest_composite_mae=1.5,
        backtest_games=20,
    )
    defaults.update(overrides)
    return ExplanationInput(**defaults)


# ---------------------------------------------------------------------------
# _label_z tests
# ---------------------------------------------------------------------------

def test_label_z_elite():
    assert _label_z(2.0) == "ELITE"


def test_label_z_strong():
    assert _label_z(1.0) == "STRONG"


def test_label_z_average():
    assert _label_z(0.0) == "AVERAGE"


def test_label_z_weak():
    assert _label_z(-1.0) == "WEAK"


def test_label_z_poor():
    assert _label_z(-2.0) == "POOR"


def test_label_z_none():
    assert _label_z(None) == "UNKNOWN"


# ---------------------------------------------------------------------------
# _factor_weight tests
# ---------------------------------------------------------------------------

def test_factor_weight_cap():
    """abs(z) >= 3.0 -> weight exactly 1.0"""
    assert _factor_weight(3.0) == 1.0
    assert _factor_weight(-3.5) == 1.0


def test_factor_weight_none():
    assert _factor_weight(None) == 0.0


def test_factor_weight_partial():
    """z=1.5 -> abs(1.5)/3.0 = 0.5"""
    assert abs(_factor_weight(1.5) - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# _build_hitter_factors tests
# ---------------------------------------------------------------------------

def test_build_hitter_factors_skips_none_z():
    """A factor with a None Z-score must NOT appear in the output list."""
    # P27: both z_sb AND z_nsb must be None for the basestealing factor to be skipped,
    # since the factor prefers z_nsb and falls back to z_sb.
    inp = _make_input(z_sb=None, z_nsb=None)
    factors = _build_hitter_factors(inp)
    names = [f.name for f in factors]
    assert all(
        "SB" not in n and "Speed" not in n and "NSB" not in n and "basestealing" not in n.lower()
        for n in names
    ), "Basestealing factor should be skipped when both z_sb and z_nsb are None"


def test_build_hitter_factors_ranked_by_weight():
    """Factors must be sorted by abs(weight) descending."""
    inp = _make_input(z_hr=2.5, z_rbi=0.2, z_avg=1.0, z_sb=0.1, z_obp=0.5)
    factors = _build_hitter_factors(inp)
    weights = [abs(f.weight) for f in factors]
    assert weights == sorted(weights, reverse=True)


def test_build_hitter_factors_narrative_contains_label():
    """Each factor narrative must include its label string."""
    inp = _make_input()
    factors = _build_hitter_factors(inp)
    for f in factors:
        assert f.label in f.narrative, (
            "Narrative '{}' does not contain label '{}'".format(f.narrative, f.label)
        )


# ---------------------------------------------------------------------------
# _build_summary tests
# ---------------------------------------------------------------------------

def test_explain_lineup_summary():
    """Lineup summary must contain player name, slot, score, and signal."""
    inp = _make_input(decision_type="lineup", player_name="Judge", target_slot="OF",
                      score_0_100=80.0, signal="SURGING")
    factors = _build_hitter_factors(inp)
    summary = _build_summary(inp, factors)
    assert "Judge" in summary
    assert "OF" in summary
    assert "80" in summary
    assert "SURGING" in summary


def test_explain_waiver_summary():
    """Waiver summary must contain player name, drop name, and value gain."""
    inp = _make_input(
        decision_type="waiver",
        player_name="Shohei Ohtani",
        drop_player_name="Bench Player",
        value_gain=4.75,
    )
    factors = _build_hitter_factors(inp)
    summary = _build_summary(inp, factors)
    assert "Shohei Ohtani" in summary
    assert "Bench Player" in summary
    assert "4.75" in summary


# ---------------------------------------------------------------------------
# _build_confidence_narrative tests
# ---------------------------------------------------------------------------

def test_confidence_narrative_high():
    inp = _make_input(score_confidence=0.9, games_in_window=12)
    result = _build_confidence_narrative(inp)
    assert result.startswith("High confidence")
    assert "12" in result


def test_confidence_narrative_moderate():
    inp = _make_input(score_confidence=0.6, games_in_window=7)
    result = _build_confidence_narrative(inp)
    assert result.startswith("Moderate confidence")


def test_confidence_narrative_low():
    inp = _make_input(score_confidence=0.3, games_in_window=3)
    result = _build_confidence_narrative(inp)
    assert result.startswith("Low confidence")
    assert "3" in result


# ---------------------------------------------------------------------------
# _build_risk_narrative tests
# ---------------------------------------------------------------------------

def test_risk_narrative_strong_upside():
    inp = _make_input(prob_above_median=0.7, upside_p75=82.0)
    result = _build_risk_narrative(inp)
    assert result is not None
    assert "Strong upside" in result
    assert "82.0" in result


def test_risk_narrative_balanced():
    inp = _make_input(prob_above_median=0.5, upside_p75=70.0)
    result = _build_risk_narrative(inp)
    assert result is not None
    assert "Balanced" in result


def test_risk_narrative_downside():
    inp = _make_input(prob_above_median=0.3, downside_p25=40.0)
    result = _build_risk_narrative(inp)
    assert result is not None
    assert "downside risk" in result


def test_risk_narrative_none_when_no_sim():
    inp = _make_input(prob_above_median=None)
    result = _build_risk_narrative(inp)
    assert result is None


# ---------------------------------------------------------------------------
# _build_track_record_narrative tests
# ---------------------------------------------------------------------------

def test_track_record_strong():
    inp = _make_input(backtest_composite_mae=1.5, backtest_games=20)
    result = _build_track_record_narrative(inp)
    assert result is not None
    assert "Strong track record" in result
    assert "1.50" in result
    assert "20" in result


def test_track_record_acceptable():
    inp = _make_input(backtest_composite_mae=3.5, backtest_games=15)
    result = _build_track_record_narrative(inp)
    assert result is not None
    assert "Acceptable track record" in result


def test_track_record_poor():
    inp = _make_input(backtest_composite_mae=6.0, backtest_games=10)
    result = _build_track_record_narrative(inp)
    assert result is not None
    assert "Poor track record" in result
    assert "caution" in result


def test_track_record_none_when_no_backtest():
    inp = _make_input(backtest_composite_mae=None, backtest_games=None)
    result = _build_track_record_narrative(inp)
    assert result is None


# ---------------------------------------------------------------------------
# explain() integration tests
# ---------------------------------------------------------------------------

def test_explain_returns_explanation_result():
    inp = _make_input()
    result = explain(inp)
    assert isinstance(result, ExplanationResult)
    assert result.decision_id == 1
    assert result.bdl_player_id == 12345
    assert isinstance(result.summary, str)
    assert len(result.factors) > 0


def test_explain_pitcher_uses_pitcher_factors():
    """pitcher player_type must produce ERA/WHIP/K9 factors, not HR/RBI."""
    inp = _make_input(
        player_type="pitcher",
        z_era=-1.2,
        z_whip=-0.8,
        z_k_per_9=1.5,
        z_hr=None,
        z_rbi=None,
        z_sb=None,
        z_nsb=None,
        z_avg=None,
        z_obp=None,
        proj_era_p50=3.45,
        proj_whip_p50=1.15,
        proj_k_p50=145.0,
    )
    result = explain(inp)
    factor_names = [f.name for f in result.factors]
    assert any("ERA" in n or "Run prevention" in n for n in factor_names)
    assert not any("HR" in n or "Power" in n for n in factor_names)


def test_explain_two_way_uses_hitter_factors():
    """two_way player_type must use hitter factors (batting is primary)."""
    inp = _make_input(player_type="two_way")
    result = explain(inp)
    factor_names = [f.name for f in result.factors]
    assert any("HR" in n or "Power" in n for n in factor_names)


# ---------------------------------------------------------------------------
# explain_batch() tests
# ---------------------------------------------------------------------------

def test_explain_batch_skips_unknown_type():
    """player_type='unknown' must be excluded from batch output."""
    known = _make_input(decision_id=1, bdl_player_id=1, player_type="hitter")
    unknown = _make_input(decision_id=2, bdl_player_id=2, player_type="unknown")
    results = explain_batch([known, unknown])
    assert len(results) == 1
    assert results[0].decision_id == 1


# ---------------------------------------------------------------------------
# P27 NSB basestealing factor tests
# ---------------------------------------------------------------------------

def _find_basestealing_factor(factors):
    """Return the basestealing factor (NSB or legacy SB), or None if absent."""
    for f in factors:
        if "NSB" in f.name or "basestealing" in f.name.lower() or "SB" in f.name or "Speed" in f.name:
            return f
    return None


def test_basestealing_factor_prefers_z_nsb_when_both_present():
    """
    When both z_sb and z_nsb are populated, the factor must use z_nsb
    (the canonical H2H One Win category) and be labeled as NSB.
    """
    # Make z_nsb and z_sb clearly different so we can tell which was used
    inp = _make_input(z_sb=0.10, z_nsb=1.80)   # elite NSB, weak SB
    factors = _build_hitter_factors(inp)
    bs = _find_basestealing_factor(factors)
    assert bs is not None, "Expected a basestealing factor"
    assert "NSB" in bs.name, f"Expected NSB in factor name, got: {bs.name!r}"
    assert bs.value == pytest.approx(1.80), "Factor value must be z_nsb, not z_sb"
    assert bs.label == "ELITE", f"Label should reflect z_nsb=1.80 (ELITE), got {bs.label}"
    assert "NSB Z-score" in bs.narrative
    # Weight follows z_nsb magnitude
    expected_weight = min(1.0, 1.80 / 3.0)
    assert bs.weight == pytest.approx(expected_weight)


def test_basestealing_factor_falls_back_to_z_sb_when_nsb_none():
    """
    Transition-period handling: pre-v27 rows only have z_sb. The factor
    must still render using z_sb (labeled 'Speed') so nothing silently
    drops from the narrative.
    """
    inp = _make_input(z_sb=1.2, z_nsb=None)
    factors = _build_hitter_factors(inp)
    bs = _find_basestealing_factor(factors)
    assert bs is not None, "Expected a fallback SB factor when z_nsb is None"
    assert "NSB" not in bs.name, "Must NOT label as NSB when falling back to z_sb"
    assert "Speed" in bs.name or "SB" in bs.name
    assert bs.value == pytest.approx(1.2)
    assert bs.label == "STRONG"
    # Narrative must indicate SB (not NSB) in fallback mode
    assert "SB Z-score" in bs.narrative
    assert "NSB Z-score" not in bs.narrative


def test_basestealing_factor_skipped_when_both_none():
    """Both z_sb and z_nsb None -> no basestealing factor at all."""
    inp = _make_input(z_sb=None, z_nsb=None)
    factors = _build_hitter_factors(inp)
    bs = _find_basestealing_factor(factors)
    assert bs is None, f"Expected no basestealing factor, got {bs}"


def test_basestealing_narrative_includes_proj_sb_when_available():
    """
    The simulator projects raw SB (not NSB), so even the NSB narrative
    must reference 'Projects X.X SB ROS' for continuity with projection tables.
    """
    inp = _make_input(z_nsb=0.9, proj_sb_p50=18.5)
    factors = _build_hitter_factors(inp)
    bs = _find_basestealing_factor(factors)
    assert bs is not None
    assert "18.5 SB ROS" in bs.narrative
    assert "NSB Z-score" in bs.narrative


def test_basestealing_narrative_omits_projection_when_none():
    """No proj_sb_p50 -> narrative falls back to Z-score-only message."""
    inp = _make_input(z_nsb=-0.3, proj_sb_p50=None)
    factors = _build_hitter_factors(inp)
    bs = _find_basestealing_factor(factors)
    assert bs is not None
    assert "SB ROS" not in bs.narrative
    assert "NSB Z-score" in bs.narrative
    # z=-0.3 falls in AVERAGE band (-0.5 < z <= 0.5 per _label_z)
    assert bs.label == "AVERAGE"


def test_explain_populates_nsb_factor_end_to_end():
    """Full explain() for a hitter must surface the NSB factor in the result."""
    inp = _make_input(player_type="hitter", z_nsb=2.1, z_sb=0.3)
    result = explain(inp)
    names = [f.name for f in result.factors]
    assert any("NSB" in n for n in names), f"Expected NSB factor, got: {names}"
    # And z_sb-labeled factor must NOT appear (superseded by NSB)
    assert not any("Speed (SB" in n for n in names), (
        f"Speed(SB) factor must not appear when z_nsb is present, got: {names}"
    )


def test_explanation_input_has_z_nsb_field():
    """Dataclass contract: z_nsb field exists."""
    inp = _make_input()
    assert hasattr(inp, "z_nsb")
    assert inp.z_nsb == pytest.approx(0.25)


def test_explain_batch_processes_all_known():
    """All non-unknown inputs must appear in the output."""
    inputs = [
        _make_input(decision_id=i, bdl_player_id=i, player_type="hitter")
        for i in range(1, 6)
    ]
    results = explain_batch(inputs)
    assert len(results) == 5


def test_explain_batch_empty_input():
    """Empty input list produces empty output."""
    assert explain_batch([]) == []


# ---------------------------------------------------------------------------
# ASCII-only output guard
# ---------------------------------------------------------------------------

def test_summary_ascii_only():
    """Summary string must contain no characters outside ASCII range (0-127)."""
    inp = _make_input()
    result = explain(inp)
    for char in result.summary:
        assert ord(char) < 128, "Non-ASCII char found in summary: {!r}".format(char)


def test_factor_narratives_ascii_only():
    """All factor narratives must contain no characters outside ASCII range."""
    inp = _make_input()
    result = explain(inp)
    for factor in result.factors:
        for char in factor.narrative:
            assert ord(char) < 128, (
                "Non-ASCII char {!r} in factor narrative: {!r}".format(char, factor.narrative)
            )
