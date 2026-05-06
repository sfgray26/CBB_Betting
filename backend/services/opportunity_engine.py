"""
PR 3.2/3.3 — Opportunity Engine

Pure computation module for playing-time intelligence signals.
No module-level I/O. All DB queries live inside aggregate_player_opportunity()
and compute_opportunity_baselines() — the explicit I/O boundary.

Signals computed:
  - pa_per_game          : plate appearances / games in 14d window
  - games_started_pct    : fraction of games where player started
  - lineup_slot_entropy  : Shannon entropy of batting order (None if data absent)
  - platoon_risk_score   : degree of platoon split (None if hand-split absent)
  - role_certainty_score : pitcher role consistency (closers~1.0, swingmen~0.0)
  - opportunity_score    : 0-1 weighted composite of available signals
  - opportunity_z        : cohort-relative z-score
  - opportunity_confidence: sigmoid of PA count in window

Consumed by:
  - daily_ingestion._compute_opportunity() (lock 100_037, 6 AM ET)
  - scoring_engine opportunity modifier (feature-flagged, off by default)

Design: cohort deltas required for z-score — raises ValueError if absent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Optional

from backend.services.config_service import get_threshold as _get_threshold

# ---------------------------------------------------------------------------
# Config-driven weights
# ---------------------------------------------------------------------------

_W_PA_PER_GAME: float     = _get_threshold("opportunity.weight.pa_per_game",     default=0.40)
_W_STARTS_PCT: float      = _get_threshold("opportunity.weight.games_started_pct",default=0.25)
_W_ROLE_CERT: float       = _get_threshold("opportunity.weight.role_certainty",   default=0.20)
_W_ENTROPY_INV: float     = _get_threshold("opportunity.weight.lineup_stability", default=0.15)

# League-average baselines used when DB baseline query is unavailable
_DEFAULT_PA_PER_GAME: float = _get_threshold("opportunity.baseline.pa_per_game", default=3.2)
_DEFAULT_PA_STD: float      = _get_threshold("opportunity.baseline.pa_std",      default=1.1)

# Minimum PA in the 14-day window before confidence is meaningful
_MIN_PA_CONFIDENCE: int = _get_threshold("opportunity.min_pa_confidence", default=20)

WINDOW_DAYS: int = 14


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

@dataclass
class OpportunityMetrics:
    """
    Raw metrics for a single player on a single date.

    Fields that require data not yet ingested (lineup_slot_*, platoon_*)
    will be None. The score computation skips None components and
    re-normalizes weights across available signals.
    """
    bdl_player_id:      int
    as_of_date:         date
    player_type:        str   # "hitter" | "pitcher" | "two_way"

    # Hitter volume
    pa_per_game:           Optional[float] = None
    ab_per_game:           Optional[float] = None
    games_played_14d:      Optional[int]   = None
    games_started_14d:     Optional[int]   = None
    games_started_pct:     Optional[float] = None

    # Batting order stability (None until batting_order column available)
    lineup_slot_avg:       Optional[float] = None
    lineup_slot_mode:      Optional[int]   = None
    lineup_slot_entropy:   Optional[float] = None

    # Platoon exposure (None until opponent_starter_hand available)
    pa_vs_lhp_14d:         Optional[int]   = None
    pa_vs_rhp_14d:         Optional[int]   = None
    platoon_ratio:         Optional[float] = None
    platoon_risk_score:    Optional[float] = None

    # Pitcher role
    appearances_14d:       Optional[int]   = None
    saves_14d:             Optional[int]   = None
    holds_14d:             Optional[int]   = None
    role_certainty_score:  Optional[float] = None

    # Availability
    days_since_last_game:  Optional[int]   = None
    il_stint_flag:         bool            = False

    # Composite output (computed by compute_opportunity_score)
    opportunity_score:      float = 0.0
    opportunity_z:          float = 0.0
    opportunity_confidence: float = 0.0


# ---------------------------------------------------------------------------
# Pure math functions (no I/O)
# ---------------------------------------------------------------------------

def compute_lineup_entropy(slots: list[int]) -> Optional[float]:
    """
    Shannon entropy of batting-order slots, normalized to [0, 1].

    Returns None if slots is empty (data not available).
    0.0 = always same slot; 1.0 = equally distributed across all 9 spots.
    """
    if not slots:
        return None

    n = len(slots)
    counts: dict[int, int] = {}
    for s in slots:
        counts[s] = counts.get(s, 0) + 1

    entropy = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            entropy -= p * math.log2(p)

    max_entropy = math.log2(9)  # 9 batting slots
    return entropy / max_entropy if max_entropy > 0 else 0.0


def compute_platoon_risk(pa_vs_lhp: Optional[int], pa_vs_rhp: Optional[int]) -> Optional[float]:
    """
    Platoon risk score: 0.0 = faces both hands equally, 1.0 = strict platoon.

    Returns None if either input is None (split data absent).
    """
    if pa_vs_lhp is None or pa_vs_rhp is None:
        return None

    total = pa_vs_lhp + pa_vs_rhp
    if total == 0:
        return None

    # Imbalance = how far the split is from 50/50
    majority = max(pa_vs_lhp, pa_vs_rhp)
    return (majority / total - 0.5) * 2.0  # maps [0.5, 1.0] → [0.0, 1.0]


def compute_role_certainty(
    appearances: Optional[int],
    saves: Optional[int],
    holds: Optional[int],
    player_type: str,
    season_sv_pct: Optional[float] = None,
) -> Optional[float]:
    """
    Pitcher role certainty score: 0.0 = pure swingman, 1.0 = locked-in closer.

    For hitters: returns None (not applicable).

    Inputs:
      appearances   -- games pitched in last 14d
      saves         -- saves recorded in last 14d
      holds         -- holds recorded in last 14d
      player_type   -- "pitcher" | "two_way" | "hitter"
      season_sv_pct -- season save% = (saves + holds) / save_opportunities; optional
    """
    if player_type == "hitter":
        return None

    if appearances is None or appearances == 0:
        return 0.0

    saves_val = saves or 0
    holds_val = holds or 0
    high_lev = saves_val + holds_val

    # Fraction of appearances in high-leverage roles
    role_rate = high_lev / appearances

    # Blend with season save% if available
    if season_sv_pct is not None:
        role_rate = 0.6 * role_rate + 0.4 * season_sv_pct

    return min(1.0, max(0.0, role_rate))


def _sigmoid(x: float, scale: float = 1.0) -> float:
    """Logistic sigmoid: maps any real to (0, 1)."""
    return 1.0 / (1.0 + math.exp(-x / scale))


def compute_opportunity_confidence(total_pa_14d: Optional[int]) -> float:
    """
    Sigmoid-based confidence from PA count in 14-day window.

    _MIN_PA_CONFIDENCE (default 20) → ~0.5 confidence.
    40 PA → ~0.88. 10 PA → ~0.27.
    """
    if total_pa_14d is None or total_pa_14d <= 0:
        return 0.0

    center = _MIN_PA_CONFIDENCE
    return _sigmoid(total_pa_14d - center, scale=center * 0.5)


def compute_opportunity_score(metrics: OpportunityMetrics) -> float:
    """
    Compute opportunity_score (0-1) from a filled OpportunityMetrics.

    Weights are config-driven. Components that are None are skipped and
    the remaining weights are re-normalized so the score stays in [0, 1].

    Hitter formula (weighted components):
      1. pa_per_game normalized (0-1 via sigmoid around league mean)
      2. games_started_pct (already 0-1)
      3. 1 - lineup_slot_entropy (stability; None if absent)
      4. 1 - platoon_risk_score (everyday player; None if absent)

    Pitcher formula:
      1. appearances_14d normalized (0-1, capped at 14 appearances = daily)
      2. role_certainty_score (0-1)
    """
    if metrics.player_type == "pitcher":
        return _pitcher_opportunity_score(metrics)

    components: list[tuple[float, float]] = []

    # Component 1: pa_per_game (sigmoid around 3.2 league mean, scale=1.5)
    if metrics.pa_per_game is not None:
        pa_norm = _sigmoid(metrics.pa_per_game - _DEFAULT_PA_PER_GAME, scale=1.5)
        components.append((pa_norm, _W_PA_PER_GAME))

    # Component 2: games_started_pct
    if metrics.games_started_pct is not None:
        components.append((metrics.games_started_pct, _W_STARTS_PCT))

    # Component 3: lineup stability (1 - entropy; high entropy = unstable lineup spot)
    if metrics.lineup_slot_entropy is not None:
        components.append((1.0 - metrics.lineup_slot_entropy, _W_ENTROPY_INV))

    # Component 4: everyday vs platoon (1 - platoon_risk)
    if metrics.platoon_risk_score is not None:
        weight = _W_PA_PER_GAME * 0.5  # platoon is secondary to raw PA volume
        components.append((1.0 - metrics.platoon_risk_score, weight))

    if not components:
        return 0.0

    total_weight = sum(w for _, w in components)
    score = sum(v * w for v, w in components) / total_weight
    return min(1.0, max(0.0, score))


def _pitcher_opportunity_score(metrics: OpportunityMetrics) -> float:
    components: list[tuple[float, float]] = []

    # Pitcher volume: appearances in 14d (14 = pitch every day = 1.0)
    if metrics.appearances_14d is not None:
        app_norm = min(1.0, metrics.appearances_14d / 14.0)
        components.append((app_norm, _W_PA_PER_GAME))

    # Role certainty
    if metrics.role_certainty_score is not None:
        components.append((metrics.role_certainty_score, _W_ROLE_CERT))

    if not components:
        return 0.0

    total_weight = sum(w for _, w in components)
    return sum(v * w for v, w in components) / total_weight


# ---------------------------------------------------------------------------
# Cohort z-score computation
# ---------------------------------------------------------------------------

def compute_opportunity_z(
    opportunity_score: float,
    cohort_scores: list[float],
) -> float:
    """
    Compute z-score of opportunity_score relative to cohort.

    Raises ValueError if cohort_scores is empty.
    """
    if not cohort_scores:
        raise ValueError("cohort_scores is required for compute_opportunity_z")

    n = len(cohort_scores)
    mean = sum(cohort_scores) / n
    variance = sum((s - mean) ** 2 for s in cohort_scores) / n
    std = math.sqrt(variance) if variance > 0 else 1.0

    return (opportunity_score - mean) / std


# ---------------------------------------------------------------------------
# I/O boundary: DB aggregation functions
# ---------------------------------------------------------------------------

def aggregate_player_opportunity(
    bdl_player_id: int,
    as_of_date: date,
    player_type: str,
    db: Any,  # SQLAlchemy Session
) -> OpportunityMetrics:
    """
    Query DB for last 14 days of performance data and build raw OpportunityMetrics.

    I/O BOUNDARY: This function hits the DB. All callers must pass a live session.
    The returned OpportunityMetrics has opportunity_score/z/confidence = 0 until
    the caller invokes compute_opportunity_score() and compute_opportunity_z().

    Queries:
      - statcast_performances: per-game PA data (14d window)
      - statcast_pitcher_metrics: season save totals for role certainty
      - player_id_mapping: bdl_id → player_id (MLBAM)
    """
    from sqlalchemy import text

    cutoff = as_of_date - timedelta(days=WINDOW_DAYS)
    metrics = OpportunityMetrics(
        bdl_player_id=bdl_player_id,
        as_of_date=as_of_date,
        player_type=player_type,
    )

    try:
        # Resolve bdl_player_id → MLBAM player_id for statcast lookup
        # Query per-game stats from mlb_player_stats (bdl_player_id direct — no MLBAM join needed).
        # PA = ab + walks (BDL doesn't expose PA directly; misses HBP/SF/SH but <5% error).
        # ip_flag = 1.0 if pitcher appeared, 0.0 otherwise (innings_pitched is a string e.g. "6.2").
        rows = db.execute(
            text(
                """
                SELECT game_date,
                       COALESCE(ab, 0) + COALESCE(walks, 0) AS pa,
                       COALESCE(ab, 0) AS ab,
                       CASE WHEN innings_pitched IS NOT NULL
                                 AND innings_pitched NOT IN ('', '0', '0.0')
                            THEN 1.0 ELSE 0.0 END AS ip_flag
                FROM mlb_player_stats
                WHERE bdl_player_id = :bdl_id
                  AND game_date > :cutoff
                  AND game_date <= :as_of
                ORDER BY game_date DESC
                """
            ),
            {"bdl_id": bdl_player_id, "cutoff": cutoff, "as_of": as_of_date},
        ).fetchall()

        if rows:
            games_played = len(rows)
            total_pa = sum(r[1] or 0 for r in rows)
            total_ab = sum(r[2] or 0 for r in rows)
            last_game_date = rows[0][0]

            metrics.games_played_14d = games_played
            metrics.pa_per_game = total_pa / games_played if games_played else 0.0
            metrics.ab_per_game = total_ab / games_played if games_played else 0.0
            metrics.days_since_last_game = (as_of_date - last_game_date).days

            # Hitter: games_started if PA > 0 (proxy; no lineup_position column)
            if player_type in ("hitter", "two_way"):
                started = sum(1 for r in rows if (r[1] or 0) > 0)
                metrics.games_started_14d = started
                metrics.games_started_pct = started / games_played if games_played else 0.0

            # Pitcher: appearances where ip > 0
            if player_type in ("pitcher", "two_way"):
                appearances = sum(1 for r in rows if (r[3] or 0.0) > 0)
                metrics.appearances_14d = appearances

        # Pitcher role certainty from season save totals (needs MLBAM ID for statcast_pitcher_metrics)
        if player_type in ("pitcher", "two_way"):
            id_row = db.execute(
                text("SELECT mlbam_id FROM player_id_mapping WHERE bdl_id = :bdl"),
                {"bdl": bdl_player_id},
            ).fetchone()
            mlbam_id = str(id_row[0]) if (id_row and id_row[0]) else None

            sv_row = None
            if mlbam_id:
                sv_row = db.execute(
                    text(
                        """
                        SELECT sv, k_pit, ip
                        FROM statcast_pitcher_metrics
                        WHERE mlbam_id = :pid
                          AND season = :season
                        LIMIT 1
                        """
                    ),
                    {"pid": mlbam_id, "season": as_of_date.year},
                ).fetchone()

            if sv_row and sv_row[2] and sv_row[2] > 0:
                # season_sv_pct = sv / (ip / 1.0) as a proxy for closure rate
                season_sv = sv_row[0] or 0
                season_ip = sv_row[2] or 0.0
                # Closer proxy: saves per IP ratio (closers pitch short, save often)
                sv_rate = min(1.0, season_sv / max(1.0, season_ip) * 5.0)
                metrics.role_certainty_score = compute_role_certainty(
                    appearances=metrics.appearances_14d,
                    saves=metrics.saves_14d,
                    holds=metrics.holds_14d,
                    player_type=player_type,
                    season_sv_pct=sv_rate,
                )
            else:
                metrics.role_certainty_score = compute_role_certainty(
                    appearances=metrics.appearances_14d,
                    saves=None,
                    holds=None,
                    player_type=player_type,
                )

    except Exception:
        # Restore session to clean state so subsequent players can still query
        try:
            db.rollback()
        except Exception:
            pass
        import logging
        logging.getLogger(__name__).warning(
            "opportunity_engine: DB query failed for bdl_player_id=%s as_of=%s",
            bdl_player_id, as_of_date,
        )

    return metrics


def compute_opportunity_baselines(
    as_of_date: date,
    player_type: str,
    db: Any,
) -> dict[str, float]:
    """
    Compute league-wide mean and std for opportunity_score across all active players.

    Used to normalize opportunity_score into opportunity_z.

    Returns {"mean": float, "std": float} or defaults if query fails.
    """
    from sqlalchemy import text

    cutoff = as_of_date - timedelta(days=WINDOW_DAYS)

    try:
        if player_type in ("hitter", "two_way"):
            result = db.execute(
                text(
                    """
                    SELECT
                        AVG(pa_per_game)   AS mean_pa,
                        STDDEV_POP(pa_per_game) AS std_pa
                    FROM (
                        SELECT player_id, AVG(pa) AS pa_per_game
                        FROM statcast_performances
                        WHERE game_date > :cutoff AND game_date <= :as_of AND pa > 0
                        GROUP BY player_id
                        HAVING COUNT(*) >= 5
                    ) sub
                    """
                ),
                {"cutoff": cutoff, "as_of": as_of_date},
            ).fetchone()

            if result and result[0] is not None:
                return {
                    "mean": float(result[0]),
                    "std": float(result[1]) if result[1] and result[1] > 0 else _DEFAULT_PA_STD,
                }
    except Exception:
        import logging
        logging.getLogger(__name__).warning(
            "opportunity_engine: baseline query failed for %s as_of=%s",
            player_type, as_of_date,
        )

    return {"mean": _DEFAULT_PA_PER_GAME, "std": _DEFAULT_PA_STD}


# ---------------------------------------------------------------------------
# Batch computation (main entry point for daily_ingestion)
# ---------------------------------------------------------------------------

def compute_all_opportunity(
    player_rows: list[tuple[int, str]],
    as_of_date: date,
    db: Any,
) -> list[OpportunityMetrics]:
    """
    Compute OpportunityMetrics for all players in player_rows.

    Parameters
    ----------
    player_rows : list of (bdl_player_id, player_type) tuples
    as_of_date  : date to compute as-of
    db          : SQLAlchemy session

    Returns
    -------
    list[OpportunityMetrics] with opportunity_score, opportunity_z,
    opportunity_confidence populated.
    """
    if not player_rows:
        return []

    # Separate into hitter and pitcher cohorts for z-score computation
    hitter_metrics: list[OpportunityMetrics] = []
    pitcher_metrics: list[OpportunityMetrics] = []

    for bdl_id, player_type in player_rows:
        m = aggregate_player_opportunity(bdl_id, as_of_date, player_type, db)
        m.opportunity_score = compute_opportunity_score(m)

        total_pa = None
        if player_type == "pitcher":
            total_pa = (m.appearances_14d or 0) * 3  # proxy: avg 3 batters/appearance
            pitcher_metrics.append(m)
        else:
            total_pa = int((m.pa_per_game or 0.0) * (m.games_played_14d or 0))
            hitter_metrics.append(m)

        m.opportunity_confidence = compute_opportunity_confidence(total_pa)

    # Compute cohort z-scores separately for hitters and pitchers
    def _apply_z_scores(cohort: list[OpportunityMetrics]) -> None:
        if len(cohort) < 2:
            for m in cohort:
                m.opportunity_z = 0.0
            return
        scores = [m.opportunity_score for m in cohort]
        for m in cohort:
            try:
                m.opportunity_z = compute_opportunity_z(m.opportunity_score, scores)
            except ValueError:
                m.opportunity_z = 0.0

    _apply_z_scores(hitter_metrics)
    _apply_z_scores(pitcher_metrics)

    return hitter_metrics + pitcher_metrics
