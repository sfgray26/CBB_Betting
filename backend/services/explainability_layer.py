"""
P19 -- Explainability Layer.

Pure-computation module (no DB imports, no side effects).
All imports are at module top level -- no imports inside functions.

Generates human-readable decision traces for every lineup and waiver action.
Consumes pre-joined data from daily_ingestion._run_explainability() and
emits ExplanationResult dataclasses that are persisted to decision_explanations.

ADR-004: Never import betting_model or analysis.
Layer 1 contract: no I/O, no DB, no logging -- pure deterministic transforms.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional


# ---------------------------------------------------------------------------
# Input dataclass -- all fields assembled upstream by the DB orchestrator
# ---------------------------------------------------------------------------

@dataclass
class ExplanationInput:
    # From decision_results
    decision_id: int
    as_of_date: date
    decision_type: str          # "lineup" | "waiver"
    bdl_player_id: int
    player_name: str            # looked up upstream, passed in here
    target_slot: Optional[str]
    drop_player_id: Optional[int]
    drop_player_name: Optional[str]
    lineup_score: Optional[float]
    value_gain: Optional[float]
    decision_confidence: float

    # From player_scores (window_days=14)
    player_type: str
    score_0_100: float
    composite_z: float
    z_hr: Optional[float]
    z_rbi: Optional[float]
    z_sb: Optional[float]
    z_avg: Optional[float]
    z_obp: Optional[float]
    z_era: Optional[float]
    z_whip: Optional[float]
    z_k_per_9: Optional[float]
    score_confidence: float
    games_in_window: int

    # From player_momentum
    signal: str                 # SURGING/HOT/STABLE/COLD/COLLAPSING

    delta_z: float

    # From simulation_results
    proj_hr_p50: Optional[float]
    proj_rbi_p50: Optional[float]
    proj_sb_p50: Optional[float]
    proj_avg_p50: Optional[float]
    proj_k_p50: Optional[float]
    proj_era_p50: Optional[float]
    proj_whip_p50: Optional[float]
    prob_above_median: Optional[float]
    downside_p25: Optional[float]
    upside_p75: Optional[float]

    # From backtest_results (may be None if no history yet)
    backtest_composite_mae: Optional[float]
    backtest_games: Optional[int]


# ---------------------------------------------------------------------------
# Output dataclasses -- pure computation output, NOT the ORM model
# In daily_ingestion.py import the ORM as:
#   from backend.models import DecisionExplanation as DecisionExplanationORM
# ---------------------------------------------------------------------------

@dataclass
class ExplanationFactor:
    name: str       # e.g. "Power (HR Z-score)"
    value: float    # the raw metric value
    label: str      # "STRONG", "AVERAGE", "WEAK", etc.
    weight: float   # contribution to the decision (0.0-1.0)
    narrative: str  # one-sentence human-readable fragment


@dataclass
class ExplanationResult:
    decision_id: int
    bdl_player_id: int
    as_of_date: date
    decision_type: str
    summary: str                        # one-sentence headline
    factors: list                       # list[ExplanationFactor], ranked by abs(weight) desc
    confidence_narrative: str
    risk_narrative: Optional[str]
    track_record_narrative: Optional[str]


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

def _label_z(z: Optional[float]) -> str:
    """
    Classify a Z-score into a label string.
    z > 1.5  -> "ELITE"
    z > 0.5  -> "STRONG"
    z > -0.5 -> "AVERAGE"
    z > -1.5 -> "WEAK"
    else     -> "POOR"
    None     -> "UNKNOWN"
    """
    if z is None:
        return "UNKNOWN"
    if z > 1.5:
        return "ELITE"
    if z > 0.5:
        return "STRONG"
    if z > -0.5:
        return "AVERAGE"
    if z > -1.5:
        return "WEAK"
    return "POOR"


def _factor_weight(z: Optional[float]) -> float:
    """
    Map Z-score to a contribution weight [0.0, 1.0].
    Uses abs(z) normalized: abs(z) / 3.0, capped at 1.0.
    None -> 0.0.
    """
    if z is None:
        return 0.0
    return min(1.0, abs(z) / 3.0)


def _build_hitter_factors(inp: ExplanationInput) -> list:
    """Build ranked ExplanationFactor list for hitters. Includes HR, RBI, SB, AVG, OBP."""
    candidates = []

    # HR factor
    if inp.z_hr is not None:
        label = _label_z(inp.z_hr)
        if inp.proj_hr_p50 is not None:
            narrative = (
                "Projects {:.1f} HR ROS; Z-score {:+.2f} ({})".format(
                    inp.proj_hr_p50, inp.z_hr, label
                )
            )
        else:
            narrative = "Z-score {:+.2f} ({})".format(inp.z_hr, label)
        candidates.append(ExplanationFactor(
            name="Power (HR Z-score)",
            value=inp.z_hr,
            label=label,
            weight=_factor_weight(inp.z_hr),
            narrative=narrative,
        ))

    # RBI factor
    if inp.z_rbi is not None:
        label = _label_z(inp.z_rbi)
        if inp.proj_rbi_p50 is not None:
            narrative = (
                "Projects {:.1f} RBI ROS; Z-score {:+.2f} ({})".format(
                    inp.proj_rbi_p50, inp.z_rbi, label
                )
            )
        else:
            narrative = "Z-score {:+.2f} ({})".format(inp.z_rbi, label)
        candidates.append(ExplanationFactor(
            name="Run production (RBI Z-score)",
            value=inp.z_rbi,
            label=label,
            weight=_factor_weight(inp.z_rbi),
            narrative=narrative,
        ))

    # SB factor
    if inp.z_sb is not None:
        label = _label_z(inp.z_sb)
        if inp.proj_sb_p50 is not None:
            narrative = (
                "Projects {:.1f} SB ROS; Z-score {:+.2f} ({})".format(
                    inp.proj_sb_p50, inp.z_sb, label
                )
            )
        else:
            narrative = "Z-score {:+.2f} ({})".format(inp.z_sb, label)
        candidates.append(ExplanationFactor(
            name="Speed (SB Z-score)",
            value=inp.z_sb,
            label=label,
            weight=_factor_weight(inp.z_sb),
            narrative=narrative,
        ))

    # AVG factor
    if inp.z_avg is not None:
        label = _label_z(inp.z_avg)
        if inp.proj_avg_p50 is not None:
            # Format avg as .XXX style
            avg_str = "{:.0f}".format(inp.proj_avg_p50 * 1000)
            narrative = (
                "Batting avg projection .{}; Z-score {:+.2f} ({})".format(
                    avg_str, inp.z_avg, label
                )
            )
        else:
            narrative = "Z-score {:+.2f} ({})".format(inp.z_avg, label)
        candidates.append(ExplanationFactor(
            name="Batting average (AVG Z-score)",
            value=inp.z_avg,
            label=label,
            weight=_factor_weight(inp.z_avg),
            narrative=narrative,
        ))

    # OBP factor
    if inp.z_obp is not None:
        label = _label_z(inp.z_obp)
        narrative = "OBP Z-score {:+.2f} ({})".format(inp.z_obp, label)
        candidates.append(ExplanationFactor(
            name="On-base ability (OBP Z-score)",
            value=inp.z_obp,
            label=label,
            weight=_factor_weight(inp.z_obp),
            narrative=narrative,
        ))

    # Rank by abs(weight) descending
    candidates.sort(key=lambda f: abs(f.weight), reverse=True)
    return candidates


def _build_pitcher_factors(inp: ExplanationInput) -> list:
    """Build ranked ExplanationFactor list for pitchers. Includes ERA, WHIP, K/9."""
    candidates = []

    # ERA factor
    if inp.z_era is not None:
        label = _label_z(inp.z_era)
        if inp.proj_era_p50 is not None:
            narrative = (
                "ERA Z-score {:+.2f} ({}); projects {:.2f} ERA ROS".format(
                    inp.z_era, label, inp.proj_era_p50
                )
            )
        else:
            narrative = "Z-score {:+.2f} ({})".format(inp.z_era, label)
        candidates.append(ExplanationFactor(
            name="Run prevention (ERA Z-score)",
            value=inp.z_era,
            label=label,
            weight=_factor_weight(inp.z_era),
            narrative=narrative,
        ))

    # WHIP factor
    if inp.z_whip is not None:
        label = _label_z(inp.z_whip)
        if inp.proj_whip_p50 is not None:
            narrative = (
                "WHIP Z-score {:+.2f} ({}); projects {:.2f} WHIP ROS".format(
                    inp.z_whip, label, inp.proj_whip_p50
                )
            )
        else:
            narrative = "Z-score {:+.2f} ({})".format(inp.z_whip, label)
        candidates.append(ExplanationFactor(
            name="Baserunner control (WHIP Z-score)",
            value=inp.z_whip,
            label=label,
            weight=_factor_weight(inp.z_whip),
            narrative=narrative,
        ))

    # K/9 factor
    if inp.z_k_per_9 is not None:
        label = _label_z(inp.z_k_per_9)
        if inp.proj_k_p50 is not None:
            narrative = (
                "K/9 Z-score {:+.2f} ({}); projects {:.0f} K ROS".format(
                    inp.z_k_per_9, label, inp.proj_k_p50
                )
            )
        else:
            narrative = "Z-score {:+.2f} ({})".format(inp.z_k_per_9, label)
        candidates.append(ExplanationFactor(
            name="Strikeout rate (K/9 Z-score)",
            value=inp.z_k_per_9,
            label=label,
            weight=_factor_weight(inp.z_k_per_9),
            narrative=narrative,
        ))

    # Rank by abs(weight) descending
    candidates.sort(key=lambda f: abs(f.weight), reverse=True)
    return candidates


def _build_summary(inp: ExplanationInput, factors: list) -> str:
    """
    Generate a one-sentence decision summary. ASCII only -- no Unicode symbols.
    For lineup: "{player_name} starts at {target_slot}: score {score:.0f}/100, {signal} momentum."
    For waiver: "Add {player_name} (drop {drop_player_name}): +{value_gain:.2f} projected value gain."
    """
    if inp.decision_type == "lineup":
        slot = inp.target_slot or "FLEX"
        score = inp.score_0_100 if inp.score_0_100 is not None else 0.0
        return (
            "{} starts at {}: score {:.0f}/100, {} momentum.".format(
                inp.player_name, slot, score, inp.signal
            )
        )
    else:
        # waiver
        drop_name = inp.drop_player_name or "roster cut"
        gain = inp.value_gain if inp.value_gain is not None else 0.0
        return (
            "Add {} (drop {}): +{:.2f} projected value gain.".format(
                inp.player_name, drop_name, gain
            )
        )


def _build_confidence_narrative(inp: ExplanationInput) -> str:
    """
    Generate confidence narrative from games_in_window and score_confidence.
    confidence >= 0.8 and games >= 10 -> "High confidence ({games} games, strong data)"
    confidence >= 0.5 -> "Moderate confidence ({games} games)"
    else -> "Low confidence ({games} games, limited data)"
    """
    games = inp.games_in_window
    conf = inp.score_confidence
    if conf >= 0.8 and games >= 10:
        return "High confidence ({} games, strong data)".format(games)
    if conf >= 0.5:
        return "Moderate confidence ({} games)".format(games)
    return "Low confidence ({} games, limited data)".format(games)


def _build_risk_narrative(inp: ExplanationInput) -> Optional[str]:
    """
    Generate risk narrative from simulation risk metrics.
    Return None if prob_above_median is None.
    prob_above_median >= 0.6 -> "Strong upside (P75={upside:.1f}, {prob:.0%} chance above median)"
    prob_above_median >= 0.4 -> "Balanced risk/reward (P75={upside:.1f})"
    else -> "Elevated downside risk (P25={downside:.1f})"
    """
    if inp.prob_above_median is None:
        return None
    prob = inp.prob_above_median
    upside = inp.upside_p75 if inp.upside_p75 is not None else 0.0
    downside = inp.downside_p25 if inp.downside_p25 is not None else 0.0
    if prob >= 0.6:
        return "Strong upside (P75={:.1f}, {:.0%} chance above median)".format(upside, prob)
    if prob >= 0.4:
        return "Balanced risk/reward (P75={:.1f})".format(upside)
    return "Elevated downside risk (P25={:.1f})".format(downside)


def _build_track_record_narrative(inp: ExplanationInput) -> Optional[str]:
    """
    Generate track record narrative from backtest_composite_mae.
    Return None if backtest_composite_mae is None or backtest_games is None.
    mae < 2.0 -> "Strong track record (MAE={mae:.2f} over {games} games)"
    mae < 5.0 -> "Acceptable track record (MAE={mae:.2f} over {games} games)"
    else      -> "Poor track record (MAE={mae:.2f} over {games} games) -- use with caution"
    """
    if inp.backtest_composite_mae is None or inp.backtest_games is None:
        return None
    mae = inp.backtest_composite_mae
    games = inp.backtest_games
    if mae < 2.0:
        return "Strong track record (MAE={:.2f} over {} games)".format(mae, games)
    if mae < 5.0:
        return "Acceptable track record (MAE={:.2f} over {} games)".format(mae, games)
    return "Poor track record (MAE={:.2f} over {} games) -- use with caution".format(mae, games)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain(inp: ExplanationInput) -> ExplanationResult:
    """
    Generate a full ExplanationResult for one player decision.
    Dispatches to _build_hitter_factors or _build_pitcher_factors based on player_type.
    Two-way players use hitter factors (batting is primary).
    """
    ptype = inp.player_type.lower() if inp.player_type else "unknown"

    if ptype == "pitcher":
        factors = _build_pitcher_factors(inp)
    else:
        # hitter, two_way, or any other type defaults to hitter factors
        factors = _build_hitter_factors(inp)

    summary = _build_summary(inp, factors)
    confidence_narrative = _build_confidence_narrative(inp)
    risk_narrative = _build_risk_narrative(inp)
    track_record_narrative = _build_track_record_narrative(inp)

    return ExplanationResult(
        decision_id=inp.decision_id,
        bdl_player_id=inp.bdl_player_id,
        as_of_date=inp.as_of_date,
        decision_type=inp.decision_type,
        summary=summary,
        factors=factors,
        confidence_narrative=confidence_narrative,
        risk_narrative=risk_narrative,
        track_record_narrative=track_record_narrative,
    )


def explain_batch(inputs: list) -> list:
    """Run explain() for each input. Skip any with player_type='unknown'."""
    results = []
    for inp in inputs:
        ptype = inp.player_type.lower() if inp.player_type else "unknown"
        if ptype == "unknown":
            continue
        results.append(explain(inp))
    return results
