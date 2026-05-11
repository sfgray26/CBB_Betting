"""
Savant Pitch Quality scoring for waiver and breakout detection.

This is an in-house, Savant-native pitcher signal. It is deliberately separate
from FanGraphs Stuff+/Location+ fields.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date


@dataclass(frozen=True)
class SavantPitcherInput:
    player_id: str
    player_name: str
    season: int
    team: str | None = None
    as_of_date: date | None = None
    xera: float | None = None
    xwoba: float | None = None
    barrel_percent_allowed: float | None = None
    hard_hit_percent_allowed: float | None = None
    avg_exit_velocity_allowed: float | None = None
    k_percent: float | None = None
    bb_percent: float | None = None
    k_9: float | None = None
    whiff_percent: float | None = None
    ip: float | None = None
    pa: int | None = None   # batters faced — used as IP proxy when ip is NULL
    pitches: int | None = None
    era: float | None = None
    whip: float | None = None
    fastball_velocity: float | None = None
    spin_rate: float | None = None


@dataclass(frozen=True)
class SavantPitchQualityScore:
    player_id: str
    player_name: str
    season: int
    as_of_date: date | None
    savant_pitch_quality: float
    arsenal_quality: float
    bat_missing_skill: float
    contact_suppression: float
    command_stability: float
    trend_adjustment: float
    sample_confidence: float
    signals: list[str] = field(default_factory=list)
    inputs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_savant_pitch_quality(pitcher: SavantPitcherInput) -> SavantPitchQualityScore:
    """Return a 100-centered pitcher quality score and waiver-oriented signals."""
    arsenal_quality = _mean_score(
        _higher_is_better(pitcher.fastball_velocity, poor=90.0, avg=93.5, elite=97.0),
        _higher_is_better(pitcher.spin_rate, poor=1950.0, avg=2200.0, elite=2500.0),
        _higher_is_better(pitcher.k_9, poor=6.8, avg=8.4, elite=11.2),
        _higher_is_better(pitcher.whiff_percent, poor=18.0, avg=23.0, elite=32.0),
    )
    bat_missing_skill = _mean_score(
        _higher_is_better(pitcher.whiff_percent, poor=18.0, avg=23.0, elite=32.0),
        _higher_is_better(pitcher.k_percent, poor=17.0, avg=22.0, elite=31.0),
        _higher_is_better(pitcher.k_9, poor=6.8, avg=8.4, elite=11.2),
    )
    contact_suppression = _mean_score(
        _lower_is_better(pitcher.xwoba, poor=0.350, avg=0.315, elite=0.275),
        _lower_is_better(pitcher.barrel_percent_allowed, poor=12.0, avg=8.5, elite=5.0),
        _lower_is_better(pitcher.hard_hit_percent_allowed, poor=45.0, avg=39.0, elite=31.0),
        _lower_is_better(pitcher.avg_exit_velocity_allowed, poor=91.0, avg=88.8, elite=86.0),
        _lower_is_better(pitcher.xera, poor=4.80, avg=4.05, elite=2.85),
    )
    command_stability = _mean_score(
        _lower_is_better(pitcher.bb_percent, poor=12.0, avg=8.3, elite=5.5),
        _lower_is_better(pitcher.whip, poor=1.42, avg=1.28, elite=1.08),
        _lower_is_better(pitcher.xera, poor=4.80, avg=4.05, elite=2.85),
    )
    trend_adjustment = _trend_adjustment(pitcher)
    sample_confidence = _sample_confidence(pitcher)

    raw_score = (
        arsenal_quality * 0.35
        + bat_missing_skill * 0.30
        + contact_suppression * 0.20
        + command_stability * 0.15
        + trend_adjustment
    )
    final_score = 100.0 + ((raw_score - 100.0) * sample_confidence)
    final_score = _clamp(final_score, 70.0, 130.0)

    signals = _build_signals(
        final_score=final_score,
        raw_score=raw_score,
        bat_missing_skill=bat_missing_skill,
        contact_suppression=contact_suppression,
        command_stability=command_stability,
        trend_adjustment=trend_adjustment,
        sample_confidence=sample_confidence,
    )

    return SavantPitchQualityScore(
        player_id=str(pitcher.player_id),
        player_name=pitcher.player_name,
        season=pitcher.season,
        as_of_date=pitcher.as_of_date,
        savant_pitch_quality=round(final_score, 1),
        arsenal_quality=round(arsenal_quality, 1),
        bat_missing_skill=round(bat_missing_skill, 1),
        contact_suppression=round(contact_suppression, 1),
        command_stability=round(command_stability, 1),
        trend_adjustment=round(trend_adjustment, 1),
        sample_confidence=round(sample_confidence, 3),
        signals=signals,
        inputs=_input_snapshot(pitcher),
    )


def score_pitcher_population(pitchers: list[SavantPitcherInput]) -> list[SavantPitchQualityScore]:
    """Score a pitcher list and return highest-quality arms first."""
    scores = [calculate_savant_pitch_quality(pitcher) for pitcher in pitchers]
    return sorted(scores, key=lambda score: score.savant_pitch_quality, reverse=True)


def _higher_is_better(value: float | int | None, *, poor: float, avg: float, elite: float) -> float | None:
    if value is None:
        return None
    value = float(value)
    if value <= avg:
        return _clamp(70.0 + ((value - poor) / (avg - poor)) * 30.0, 70.0, 100.0)
    return _clamp(100.0 + ((value - avg) / (elite - avg)) * 30.0, 100.0, 130.0)


def _lower_is_better(value: float | int | None, *, poor: float, avg: float, elite: float) -> float | None:
    if value is None:
        return None
    value = float(value)
    if value >= avg:
        return _clamp(70.0 + ((poor - value) / (poor - avg)) * 30.0, 70.0, 100.0)
    return _clamp(100.0 + ((avg - value) / (avg - elite)) * 30.0, 100.0, 130.0)


def _mean_score(*scores: float | None) -> float:
    present = [score for score in scores if score is not None]
    if not present:
        return 100.0
    return sum(present) / len(present)


def _trend_adjustment(pitcher: SavantPitcherInput) -> float:
    if pitcher.era is None or pitcher.xera is None:
        return 0.0
    # ERA above xERA suggests the visible ratios may not yet reflect skill.
    return _clamp((float(pitcher.era) - float(pitcher.xera)) * 3.0, -6.0, 8.0)


def _sample_confidence(pitcher: SavantPitcherInput) -> float:
    ip_val = pitcher.ip
    if ip_val is None and pitcher.pa is not None and pitcher.pa > 0:
        ip_val = pitcher.pa / 4.3  # ~4.3 batters faced per inning
    ip_conf = _clamp((ip_val or 0.0) / 40.0, 0.0, 1.0)
    pitch_conf = _clamp((pitcher.pitches or 0) / 650.0, 0.0, 1.0)
    return _clamp((ip_conf * 0.45) + (pitch_conf * 0.55), 0.0, 1.0)


def _build_signals(
    *,
    final_score: float,
    raw_score: float,
    bat_missing_skill: float,
    contact_suppression: float,
    command_stability: float,
    trend_adjustment: float,
    sample_confidence: float,
) -> list[str]:
    signals: list[str] = []
    ratio_risk = bat_missing_skill >= 108.0 and (
        command_stability < 95.0 or contact_suppression < 95.0
    )
    if ratio_risk:
        signals.append("RATIO_RISK")
    if sample_confidence < 0.50 and (raw_score >= 108.0 or trend_adjustment >= 2.5):
        signals.append("WATCHLIST")
    if raw_score >= 108.0 or trend_adjustment >= 3.0:
        signals.append("SKILL_CHANGE")
    if final_score >= 112.0 and sample_confidence >= 0.70 and not ratio_risk:
        signals.append("BREAKOUT_ARM")
    if final_score >= 104.0 and sample_confidence >= 0.55 and not ratio_risk:
        signals.append("STREAMER_UPSIDE")
    return signals


def _input_snapshot(pitcher: SavantPitcherInput) -> dict:
    values = asdict(pitcher)
    values["as_of_date"] = pitcher.as_of_date.isoformat() if pitcher.as_of_date else None
    return values


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
