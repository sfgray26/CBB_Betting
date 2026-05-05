"""
PR 5.2/5.3/5.4 — Matchup Context Engine

Pure computation module for per-player-per-game environmental context signals.
No module-level I/O. All DB queries live inside collect_matchup_context() and
the _fetch_* helpers — the explicit I/O boundary.

Signals computed (5 factors):
  handedness_score  : hitter wOBA vs pitcher hand vs league average   (35%)
  pitcher_score     : opponent starter quality penalty/boost           (25%)
  park_score        : run environment adjustment                       (15%)
  weather_bonus     : temperature + wind favorability                  (10%)
  bullpen_score     : opponent bullpen quality                         (15%)

  matchup_z         : weighted combination, re-normalised when data absent
  matchup_confidence: data completeness sigmoid [0.0, 1.0]
  matchup_score     : final 0-100 signal (50 = neutral)

Consumed by:
  - daily_ingestion._compute_matchup_context() (lock 100_039, 10:30 AM ET)
  - daily_lineup_optimizer.rank_batters()  (PR 5.5, feature-flagged OFF by default)

Architecture note: matchup_score is a MODIFIER on lineup_score, not a
primary signal.  Max boost: clamp(matchup_z * 0.1, -0.2, +0.2) × confidence.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.services.config_service import get_threshold as _get_threshold

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config-driven weights (loaded at module level with safe defaults)
# ---------------------------------------------------------------------------

_W_HAND:    float = _get_threshold("matchup.weight.handedness", default=0.35)
_W_PITCHER: float = _get_threshold("matchup.weight.pitcher",    default=0.25)
_W_PARK:    float = _get_threshold("matchup.weight.park",       default=0.15)
_W_WEATHER: float = _get_threshold("matchup.weight.weather",    default=0.10)
_W_BULLPEN: float = _get_threshold("matchup.weight.bullpen",    default=0.15)

_BOOST_CAP:     float = _get_threshold("matchup.boost.cap",     default=0.2)
_BOOST_Z_SCALE: float = _get_threshold("matchup.boost.z_scale", default=0.1)

# Confidence gate: below this, matchup_z is halved before score conversion
_CONF_GATE: float = _get_threshold("matchup.confidence_gate", default=0.4)

# MLB 2026 season baselines (hardcoded fallbacks; overridden by compute_baselines)
_MLB_BASELINES_DEFAULT: dict = {
    "mean_era":          4.20,
    "std_era":           0.85,
    "mean_whip":         1.28,
    "std_whip":          0.18,
    "mean_bullpen_era":  4.35,
    "std_bullpen_era":   0.75,
    "std_woba_gap":      0.045,  # typical std of hand-split wOBA gap across hitters
    "min_split_pa":      30,     # minimum PA vs hand for meaningful confidence
}


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class PitcherStats:
    """Opponent probable starter metrics."""

    name: str
    hand: Optional[str]       # 'L' | 'R' | None (not yet tracked → confidence gate)
    era: Optional[float]
    whip: Optional[float]
    k_per_nine: Optional[float]
    mlbam_id: Optional[int] = None


@dataclass
class HitterSplits:
    """
    Hitter split stats vs a specific pitcher hand (365-day lookback).
    woba_overall is the hand-agnostic baseline for computing the gap.
    """

    woba_vs_hand: Optional[float]
    woba_overall: Optional[float]
    k_pct_vs_hand: Optional[float]
    iso_vs_hand: Optional[float]
    pa_vs_hand: int = 0


@dataclass
class BullpenStats:
    """Opponent bullpen aggregate (excluding probable starter)."""

    era: Optional[float]
    whip: Optional[float]
    pitcher_count: int = 0


@dataclass
class WeatherData:
    """Weather conditions at the park."""

    temp_f: Optional[float]
    wind_mph: Optional[float]
    wind_direction: Optional[str]  # e.g. "out", "in", "cross"
    precip_chance: Optional[float]  # 0-100


@dataclass
class MatchupContext:
    """All inputs needed to compute matchup_z for one player on one game date."""

    bdl_player_id: int
    game_date: date
    opponent_team: Optional[str]
    home_team: Optional[str]
    pitcher: Optional[PitcherStats]
    splits: Optional[HitterSplits]
    bullpen: Optional[BullpenStats]
    weather: Optional[WeatherData]
    park_factor_runs: float = 1.0
    park_factor_hr: float = 1.0


@dataclass
class MatchupResult:
    """
    Matchup signal output for a single player on a single date.

    matchup_score       -- 0-100 signal (50 = neutral)
    matchup_z           -- raw weighted z-score
    matchup_confidence  -- data completeness [0.0, 1.0]
    component_weights   -- weights actually used after None-exclusion renorm
    """

    matchup_score:      float
    matchup_z:          float
    matchup_confidence: float
    component_weights:  dict = field(default_factory=dict)


# Neutral result returned when data is entirely absent
_NEUTRAL_RESULT = MatchupResult(
    matchup_score=50.0,
    matchup_z=0.0,
    matchup_confidence=0.0,
    component_weights={},
)


# ---------------------------------------------------------------------------
# Baseline computation (DB-backed with hardcoded fallback)
# ---------------------------------------------------------------------------

def compute_baselines(db: Optional[Session] = None) -> dict:
    """
    Compute 2026 season-to-date baselines from statcast_pitcher_metrics.

    Falls back to _MLB_BASELINES_DEFAULT on any DB failure or insufficient data.
    Exposed at module level (no leading underscore) so tests and callers can
    inject synthetic baselines without touching the DB.
    """
    if db is None:
        return dict(_MLB_BASELINES_DEFAULT)

    try:
        row = db.execute(text("""
            SELECT
                AVG(era)           AS mean_era,
                STDDEV_POP(era)    AS std_era,
                AVG(whip)          AS mean_whip,
                STDDEV_POP(whip)   AS std_whip
            FROM statcast_pitcher_metrics
            WHERE season = 2026
              AND era  IS NOT NULL
              AND whip IS NOT NULL
              AND CAST(ip AS FLOAT) > 5
        """)).fetchone()

        bullpen_row = db.execute(text("""
            SELECT
                AVG(era)           AS mean_bullpen_era,
                STDDEV_POP(era)    AS std_bullpen_era
            FROM statcast_pitcher_metrics
            WHERE season = 2026
              AND era IS NOT NULL
              AND CAST(ip AS FLOAT) BETWEEN 5 AND 50
        """)).fetchone()

        baselines = dict(_MLB_BASELINES_DEFAULT)

        if row and row[0] is not None:
            baselines["mean_era"]  = float(row[0])
            baselines["std_era"]   = float(row[1]) if row[1] else _MLB_BASELINES_DEFAULT["std_era"]
            baselines["mean_whip"] = float(row[2])
            baselines["std_whip"]  = float(row[3]) if row[3] else _MLB_BASELINES_DEFAULT["std_whip"]

        if bullpen_row and bullpen_row[0] is not None:
            baselines["mean_bullpen_era"] = float(bullpen_row[0])
            baselines["std_bullpen_era"]  = (
                float(bullpen_row[1]) if bullpen_row[1]
                else _MLB_BASELINES_DEFAULT["std_bullpen_era"]
            )

        return baselines

    except Exception as exc:
        logger.warning("matchup_engine: baseline query failed (%s), using defaults", exc)
        return dict(_MLB_BASELINES_DEFAULT)


# ---------------------------------------------------------------------------
# Data fetch helpers (all I/O here; pure math below)
# ---------------------------------------------------------------------------

def _fetch_pitcher_stats(
    opponent_team: str,
    game_date: date,
    db: Session,
) -> Optional[PitcherStats]:
    """
    Look up probable starter for opponent_team on game_date.

    Joins probable_pitchers → statcast_pitcher_metrics for ERA/WHIP/K9.
    Returns None on any failure (gracefully excluded from scoring).
    """
    try:
        row = db.execute(text("""
            SELECT
                pp.pitcher_name,
                pp.mlbam_id,
                spm.era,
                spm.whip,
                spm.k_9
            FROM probable_pitchers pp
            LEFT JOIN statcast_pitcher_metrics spm
                   ON CAST(spm.mlbam_id AS INTEGER) = pp.mlbam_id
                  AND spm.season = 2026
            WHERE pp.team     = :team
              AND pp.game_date = :gd
            ORDER BY pp.fetched_at DESC
            LIMIT 1
        """), {"team": opponent_team, "gd": game_date}).fetchone()

        if row is None:
            return None

        name, mlbam_id, era, whip, k9 = row
        return PitcherStats(
            name=name or "Unknown",
            hand=None,  # hand not tracked yet; confidence gates handedness component
            era=float(era) if era is not None else None,
            whip=float(whip) if whip is not None else None,
            k_per_nine=float(k9) if k9 is not None else None,
            mlbam_id=int(mlbam_id) if mlbam_id is not None else None,
        )

    except Exception as exc:
        logger.debug("matchup_engine._fetch_pitcher_stats: %s", exc)
        return None


def _fetch_hitter_splits(
    bdl_player_id: int,
    pitcher_hand: Optional[str],
    db: Session,
    lookback_days: int = 365,
) -> Optional[HitterSplits]:
    """
    Compute hitter split stats vs pitcher_hand from mlb_player_stats.

    wOBA proxy:  (H + 0.7*BB + 1.4*HR) / (AB + BB)
    ISO  proxy:  (2*2B + 3*3B + 4*HR)  / AB
    K%   proxy:  K / PA

    Returns None when pitcher_hand is None (can't split by unknown hand).
    Returns HitterSplits with pa_vs_hand=0 when fewer than 5 qualifying rows.
    """
    if pitcher_hand is None:
        return None

    try:
        split_row = db.execute(text("""
            SELECT
                COUNT(*)            AS pa_vs_hand,
                SUM(hits)           AS h_vs,
                SUM(walks)          AS bb_vs,
                SUM(home_runs)      AS hr_vs,
                SUM(ab)             AS ab_vs,
                SUM(strikeouts_bat) AS k_vs,
                SUM(doubles)        AS d_vs,
                SUM(triples)        AS t_vs
            FROM mlb_player_stats
            WHERE bdl_player_id        = :pid
              AND opponent_starter_hand = :hand
              AND game_date            >= CURRENT_DATE - :days * INTERVAL '1 day'
              AND ab IS NOT NULL AND ab > 0
        """), {"pid": bdl_player_id, "hand": pitcher_hand, "days": lookback_days}).fetchone()

        overall_row = db.execute(text("""
            SELECT
                SUM(hits)      AS total_h,
                SUM(walks)     AS total_bb,
                SUM(home_runs) AS total_hr,
                SUM(ab)        AS total_ab
            FROM mlb_player_stats
            WHERE bdl_player_id = :pid
              AND game_date     >= CURRENT_DATE - :days * INTERVAL '1 day'
              AND ab IS NOT NULL AND ab > 0
        """), {"pid": bdl_player_id, "days": lookback_days}).fetchone()

        # No plate appearances vs this hand
        if split_row is None or int(split_row[0] or 0) == 0:
            return HitterSplits(
                woba_vs_hand=None,
                woba_overall=None,
                k_pct_vs_hand=None,
                iso_vs_hand=None,
                pa_vs_hand=0,
            )

        pa_vs = int(split_row[0] or 0)
        h_vs  = int(split_row[1] or 0)
        bb_vs = int(split_row[2] or 0)
        hr_vs = int(split_row[3] or 0)
        ab_vs = int(split_row[4] or 0)
        k_vs  = int(split_row[5] or 0)
        d_vs  = int(split_row[6] or 0)
        t_vs  = int(split_row[7] or 0)

        denom_vs = ab_vs + bb_vs
        woba_vs  = (h_vs + 0.7 * bb_vs + 1.4 * hr_vs) / denom_vs if denom_vs > 0 else None
        k_pct_vs = k_vs / pa_vs if pa_vs > 0 else None
        iso_vs   = (2 * d_vs + 3 * t_vs + 4 * hr_vs) / ab_vs if ab_vs > 0 else None

        woba_overall = None
        if overall_row and int(overall_row[3] or 0) > 0:
            tot_h  = int(overall_row[0] or 0)
            tot_bb = int(overall_row[1] or 0)
            tot_hr = int(overall_row[2] or 0)
            tot_ab = int(overall_row[3] or 0)
            denom_tot = tot_ab + tot_bb
            if denom_tot > 0:
                woba_overall = (tot_h + 0.7 * tot_bb + 1.4 * tot_hr) / denom_tot

        return HitterSplits(
            woba_vs_hand=woba_vs,
            woba_overall=woba_overall,
            k_pct_vs_hand=k_pct_vs,
            iso_vs_hand=iso_vs,
            pa_vs_hand=pa_vs,
        )

    except Exception as exc:
        logger.debug("matchup_engine._fetch_hitter_splits: %s", exc)
        return None


def _fetch_bullpen_stats(
    opponent_team: str,
    exclude_starter_mlbam: Optional[int],
    db: Session,
) -> Optional[BullpenStats]:
    """
    Aggregate ERA/WHIP for opponent's bullpen from statcast_pitcher_metrics.

    Excludes the probable starter by mlbam_id.
    Returns None on failure or when fewer than 2 pitchers qualify.
    """
    try:
        params: dict = {"team": opponent_team, "season": 2026}
        exclude_clause = ""
        if exclude_starter_mlbam is not None:
            exclude_clause = " AND CAST(mlbam_id AS INTEGER) != :starter_mlbam "
            params["starter_mlbam"] = exclude_starter_mlbam

        row = db.execute(text(
            "SELECT AVG(era) AS bullpen_era, AVG(whip) AS bullpen_whip, COUNT(*) AS pitcher_count"
            " FROM statcast_pitcher_metrics"
            " WHERE team = :team AND season = :season AND era IS NOT NULL"
            " AND CAST(ip AS FLOAT) BETWEEN 1 AND 50"
            + exclude_clause
        ), params).fetchone()

        if row is None or row[2] is None or int(row[2]) < 2:
            return None

        return BullpenStats(
            era=float(row[0]) if row[0] is not None else None,
            whip=float(row[1]) if row[1] is not None else None,
            pitcher_count=int(row[2]),
        )

    except Exception as exc:
        logger.debug("matchup_engine._fetch_bullpen_stats: %s", exc)
        return None


# Park name lookup (best-effort — unknown teams fall back to park_factor 1.0)
_TEAM_TO_PARK: dict = {
    "ARI": "Chase Field",
    "ATL": "Truist Park",
    "BAL": "Camden Yards",
    "BOS": "Fenway Park",
    "CHC": "Wrigley Field",
    "CIN": "Great American Ball Park",
    "CLE": "Progressive Field",
    "COL": "Coors Field",
    "CWS": "Guaranteed Rate Field",
    "DET": "Comerica Park",
    "HOU": "Minute Maid Park",
    "KC":  "Kauffman Stadium",
    "LAA": "Angel Stadium",
    "LAD": "Dodger Stadium",
    "MIA": "LoanDepot Park",
    "MIL": "American Family Field",
    "MIN": "Target Field",
    "NYM": "Citi Field",
    "NYY": "Yankee Stadium",
    "OAK": "Oakland Coliseum",
    "PHI": "Citizens Bank Park",
    "PIT": "PNC Park",
    "SD":  "Petco Park",
    "SEA": "T-Mobile Park",
    "SF":  "Oracle Park",
    "STL": "Busch Stadium",
    "TB":  "Tropicana Field",
    "TEX": "Globe Life Field",
    "TOR": "Rogers Centre",
    "WSH": "Nationals Park",
}


def _fetch_park_factor(home_team: str) -> float:
    """
    Fetch run park factor for home_team via ballpark_factors module.

    Returns 1.0 (neutral) on any failure.
    """
    try:
        from backend.fantasy_baseball.ballpark_factors import get_park_factor  # lazy import
        return get_park_factor(home_team, "run")
    except Exception as exc:
        logger.debug("matchup_engine._fetch_park_factor(%s): %s", home_team, exc)
        return 1.0


def _fetch_weather(
    home_team: str,
    game_date: date,
    db: Session,
) -> Optional[WeatherData]:
    """
    Look up weather forecast from weather_forecasts table.

    Converts Celsius → Fahrenheit and km/h → mph.
    Returns None when no forecast found or team-to-park mapping is unknown.
    """
    park = _TEAM_TO_PARK.get(home_team)
    if park is None:
        return None

    try:
        row = db.execute(text("""
            SELECT
                temperature_high,
                wind_speed,
                wind_direction,
                precipitation_probability
            FROM weather_forecasts
            WHERE game_date = :gd
              AND park_name = :park
            ORDER BY fetched_at DESC
            LIMIT 1
        """), {"gd": game_date, "park": park}).fetchone()

        if row is None:
            return None

        temp_c, wind_kmh, wind_dir, precip = row
        temp_f   = (float(temp_c) * 9.0 / 5.0 + 32.0) if temp_c is not None else None
        wind_mph = float(wind_kmh) / 1.609 if wind_kmh is not None else None

        return WeatherData(
            temp_f=temp_f,
            wind_mph=wind_mph,
            wind_direction=wind_dir,
            precip_chance=float(precip) if precip is not None else None,
        )

    except Exception as exc:
        logger.debug("matchup_engine._fetch_weather(%s): %s", home_team, exc)
        return None


def collect_matchup_context(
    bdl_player_id: int,
    game_date: date,
    opponent_team: str,
    home_team: str,
    db: Session,
) -> MatchupContext:
    """
    Assemble all context for a single hitter + game.

    Silently swallows failures on each individual fetch — caller gets a
    partial MatchupContext and scoring degrades gracefully through the
    confidence gate.  No I/O happens outside this function and _fetch_*
    helpers.
    """
    pitcher = _fetch_pitcher_stats(opponent_team, game_date, db)
    pitcher_hand = pitcher.hand if pitcher is not None else None
    splits  = _fetch_hitter_splits(bdl_player_id, pitcher_hand, db)
    bullpen = _fetch_bullpen_stats(
        opponent_team,
        pitcher.mlbam_id if pitcher is not None else None,
        db,
    )
    park_factor = _fetch_park_factor(home_team)
    weather     = _fetch_weather(home_team, game_date, db)

    return MatchupContext(
        bdl_player_id=bdl_player_id,
        game_date=game_date,
        opponent_team=opponent_team,
        home_team=home_team,
        pitcher=pitcher,
        splits=splits,
        bullpen=bullpen,
        weather=weather,
        park_factor_runs=park_factor,
        park_factor_hr=park_factor,
    )


# ---------------------------------------------------------------------------
# Pure scoring functions (no I/O — all testable in isolation)
# ---------------------------------------------------------------------------

def compute_handedness_score(
    splits: Optional[HitterSplits],
    baselines: dict,
) -> float:
    """
    Handedness advantage: (wOBA_vs_hand - wOBA_overall) / std_woba_gap.

    Positive = hitter advantages vs this pitcher's hand.
    Negative = handedness disadvantage (e.g. RHB vs elite RHP).
    Returns 0.0 when splits are missing or wOBA values unavailable.
    """
    if splits is None:
        return 0.0
    if splits.woba_vs_hand is None or splits.woba_overall is None:
        return 0.0

    std = baselines.get("std_woba_gap", _MLB_BASELINES_DEFAULT["std_woba_gap"])
    if std <= 0.0:
        return 0.0

    return (splits.woba_vs_hand - splits.woba_overall) / std


def compute_pitcher_score(
    pitcher: Optional[PitcherStats],
    baselines: dict,
) -> float:
    """
    Pitcher quality penalty/boost (hitter perspective).

    Elite pitcher (low ERA/WHIP) → negative z → drags matchup_z down.
    Weak pitcher  (high ERA/WHIP) → positive z → boosts matchup_z.

    Returns average of available z-scores (ERA + WHIP), or 0.0 when unavailable.
    """
    if pitcher is None:
        return 0.0

    scores: list = []

    mean_era  = baselines.get("mean_era",  _MLB_BASELINES_DEFAULT["mean_era"])
    std_era   = baselines.get("std_era",   _MLB_BASELINES_DEFAULT["std_era"])
    mean_whip = baselines.get("mean_whip", _MLB_BASELINES_DEFAULT["mean_whip"])
    std_whip  = baselines.get("std_whip",  _MLB_BASELINES_DEFAULT["std_whip"])

    if pitcher.era is not None and std_era > 0:
        scores.append((pitcher.era - mean_era) / std_era)

    if pitcher.whip is not None and std_whip > 0:
        scores.append((pitcher.whip - mean_whip) / std_whip)

    return sum(scores) / len(scores) if scores else 0.0


def compute_park_score(park_factor_runs: float) -> float:
    """
    Park run environment score.

    Scales deviation from neutral × 20 to produce a z-like score:
      Coors Field  (~1.35) → +7.0
      Petco Park   (~0.88) → -2.4
      Neutral park (1.00)  →  0.0
    """
    return (park_factor_runs - 1.0) * 20.0


def compute_weather_bonus(weather: Optional[WeatherData]) -> float:
    """
    Weather favorability for hitters (raw additive score).

    Wind blowing out (>15 mph)      → +3.0
    Wind blowing in  (>15 mph)      → -1.5
    Hot day          (>85°F)        → +1.5
    High precip      (>50%)         → -2.0
    Combined range:  ≈ [-3.5, +4.5]

    Returns 0.0 when weather data is unavailable.
    """
    if weather is None:
        return 0.0

    bonus = 0.0
    wind_dir = (weather.wind_direction or "").lower()
    wind_mph = weather.wind_mph or 0.0

    if wind_mph > 15:
        if any(kw in wind_dir for kw in ("out", "l_to_r", "ltr")):
            bonus += 3.0
        elif any(kw in wind_dir for kw in ("in", "r_to_l", "rtl")):
            bonus -= 1.5

    if (weather.temp_f or 0.0) > 85.0:
        bonus += 1.5

    if (weather.precip_chance or 0.0) > 50.0:
        bonus -= 2.0

    return bonus


def compute_bullpen_score(
    bullpen: Optional[BullpenStats],
    baselines: dict,
) -> float:
    """
    Opponent bullpen quality score (hitter perspective).

    Weak bullpen (high ERA)   → positive z → good for hitter.
    Strong bullpen (low ERA)  → negative z → bad for hitter.

    Returns z-score vs league mean, or 0.0 when bullpen data missing.
    """
    if bullpen is None or bullpen.era is None:
        return 0.0

    mean_bp = baselines.get("mean_bullpen_era", _MLB_BASELINES_DEFAULT["mean_bullpen_era"])
    std_bp  = baselines.get("std_bullpen_era",  _MLB_BASELINES_DEFAULT["std_bullpen_era"])

    if std_bp <= 0.0:
        return 0.0

    return (bullpen.era - mean_bp) / std_bp


def compute_matchup_confidence(
    splits: Optional[HitterSplits],
    pitcher: Optional[PitcherStats],
    baselines: dict,
) -> float:
    """
    Confidence in matchup_z signal [0.0, 1.0].

    Based on:
      - Sigmoid over PA-vs-hand (saturates ~1.0 at 3× min_split_pa)
      - Data completeness penalty (−0.20 for missing pitcher, −0.15 for missing splits)

    Result is clamped to [0.0, 1.0].
    """
    min_pa = int(baselines.get("min_split_pa", _MLB_BASELINES_DEFAULT["min_split_pa"]))
    pa = splits.pa_vs_hand if splits is not None else 0

    if pa <= 0:
        base_conf = 0.10
    else:
        # Logistic sigmoid centred at min_pa; equals ~0.5 at pa == min_pa
        x = pa / max(min_pa, 1)
        base_conf = 1.0 / (1.0 + math.exp(-3.0 * (x - 1.0)))

    missing_penalty = 0.0
    if pitcher is None:
        missing_penalty += 0.20
    if splits is None or splits.woba_vs_hand is None:
        missing_penalty += 0.15

    return max(0.0, min(1.0, base_conf - missing_penalty))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_matchup_z(
    context: MatchupContext,
    baselines: Optional[dict] = None,
) -> MatchupResult:
    """
    Compute matchup_z and matchup_score from all 5 context factors.

    Weights are dynamically re-normalised when a factor is unavailable,
    so the remaining factors still sum to 1.0.

    Returns a neutral MatchupResult (z=0, score=50, conf=0) when all
    factors return zero weight (complete data absence).
    """
    if baselines is None:
        baselines = dict(_MLB_BASELINES_DEFAULT)

    # --- Component scores ---
    h_score = compute_handedness_score(context.splits, baselines)
    p_score = compute_pitcher_score(context.pitcher, baselines)
    k_score = compute_park_score(context.park_factor_runs)
    # Normalize weather raw bonus to a ±1 scale (max ≈ ±4.5 → divide by 4.5)
    w_raw   = compute_weather_bonus(context.weather)
    w_score = w_raw / 4.5 if context.weather is not None else 0.0
    b_score = compute_bullpen_score(context.bullpen, baselines)

    # --- Data availability flags ---
    has_splits = (
        context.splits is not None
        and context.splits.woba_vs_hand is not None
        and context.pitcher is not None
        and context.pitcher.hand is not None  # hand needed to give splits meaning
    )
    has_pitcher = context.pitcher is not None
    has_weather = context.weather is not None
    has_bullpen = context.bullpen is not None and context.bullpen.era is not None

    weights = {
        "handedness": _W_HAND    if has_splits  else 0.0,
        "pitcher":    _W_PITCHER if has_pitcher else 0.0,
        "park":       _W_PARK,  # always available (defaults to 1.0 neutral)
        "weather":    _W_WEATHER if has_weather else 0.0,
        "bullpen":    _W_BULLPEN if has_bullpen else 0.0,
    }
    total_w = sum(weights.values())

    if total_w <= 0.0:
        return MatchupResult(
            matchup_score=50.0,
            matchup_z=0.0,
            matchup_confidence=0.0,
            component_weights=weights,
        )

    # Re-normalise so weights sum to 1.0
    norm = {k: v / total_w for k, v in weights.items()}

    scores_map = {
        "handedness": h_score,
        "pitcher":    p_score,
        "park":       k_score,
        "weather":    w_score,
        "bullpen":    b_score,
    }
    matchup_z = sum(norm[k] * scores_map[k] for k in scores_map)

    confidence = compute_matchup_confidence(context.splits, context.pitcher, baselines)

    # Apply confidence gate: dampen signal when data is thin
    if confidence < _CONF_GATE:
        matchup_z *= 0.5

    # Convert z → 0-100 score (50 = neutral; ±1 σ ≈ ±20 points)
    matchup_score = max(0.0, min(100.0, 50.0 + matchup_z * 20.0))

    return MatchupResult(
        matchup_score=round(matchup_score, 2),
        matchup_z=round(matchup_z, 4),
        matchup_confidence=round(confidence, 3),
        component_weights=norm,
    )


# ---------------------------------------------------------------------------
# CBB legacy compatibility stubs
# These symbols were exported by the old CBB matchup engine and are still
# imported by backend/services/analysis.py (frozen CBB module, season over).
# Stubs prevent ImportError without re-introducing CBB runtime logic.
# ---------------------------------------------------------------------------

class _LegacyTeamPlayStyle:
    """Minimal stub — CBB season is over; analysis.py may call this but results unused."""
    def __init__(self, team: str = "", **kwargs):
        self.team = team
        # PBP-derived fields that default to 0.0 when not supplied
        self.drop_coverage_pct = 0.0
        self.zone_pct = 0.0
        for k, v in kwargs.items():
            setattr(self, k, v)

    def has_profiles(self) -> bool:  # noqa: D102
        return False


class _LegacyMatchupEngine:
    """Stub CBB matchup engine — all methods are no-ops."""

    def has_profiles(self) -> bool:
        return False

    def analyze_matchup(self, home, away):
        class _Adj:
            margin_adj = 0.0
            sd_adj = 0.0
            factors: dict = {}
            notes: list = []
        return _Adj()


class _LegacyProfileCache:
    """Stub team profile cache."""

    def has_profiles(self) -> bool:
        return False

    def load_from_barttorvik(self) -> int:
        return 0

    def get(self, team: str):
        return None

    def set(self, team: str, profile) -> None:
        pass


# Module-level singleton stubs
_LEGACY_ENGINE = _LegacyMatchupEngine()
_LEGACY_CACHE = _LegacyProfileCache()

# Public aliases used by analysis.py (CBB legacy)
TeamPlayStyle = _LegacyTeamPlayStyle


def get_matchup_engine() -> _LegacyMatchupEngine:
    """Return stub CBB matchup engine (CBB season over; all ops are no-ops)."""
    return _LEGACY_ENGINE


def get_profile_cache() -> _LegacyProfileCache:
    """Return stub team profile cache (CBB season over)."""
    return _LEGACY_CACHE

