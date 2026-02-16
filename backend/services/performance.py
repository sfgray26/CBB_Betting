"""
Performance analytics computation.

All public functions receive a SQLAlchemy Session and return plain dicts
so they can be called from FastAPI endpoints or background jobs without
importing any web-layer code.
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sqlalchemy.orm import Session, joinedload

from backend.models import BetLog, Game, PerformanceSnapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_roi(profit: float, risked: float) -> float:
    return round(profit / risked, 4) if risked > 0 else 0.0


def _win_rate(wins: int, total: int) -> float:
    return round(wins / total, 4) if total > 0 else 0.0


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _std(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _clv_status(mean_clv: Optional[float]) -> str:
    if mean_clv is None:
        return "UNKNOWN"
    if mean_clv > 0.005:
        return "HEALTHY"
    if mean_clv >= -0.005:
        return "WARNING"
    return "STOP"


def _settled_bets(db: Session, cutoff: Optional[datetime] = None) -> List[BetLog]:
    """Return settled (non-push) bets, optionally filtered by timestamp."""
    q = (
        db.query(BetLog)
        .join(Game)
        .options(joinedload(BetLog.game))
        .filter(BetLog.outcome.isnot(None), BetLog.outcome != -1)
    )
    if cutoff:
        q = q.filter(BetLog.timestamp >= cutoff)
    return q.order_by(Game.game_date.asc()).all()


# ---------------------------------------------------------------------------
# calculate_summary_stats
# ---------------------------------------------------------------------------

def calculate_summary_stats(db: Session) -> Dict:
    """
    Full performance summary including:
      - overall metrics (win rate, ROI, CLV, drawdown)
      - by_bet_type breakdown
      - by_edge_bucket breakdown (conservative_edge)
      - rolling_windows (last 10 / 50 / 100)
    """
    bets = _settled_bets(db)

    if not bets:
        return {"message": "No settled bets yet", "total_bets": 0}

    total = len(bets)
    wins = sum(1 for b in bets if b.outcome == 1)
    total_risked = sum(b.bet_size_dollars or 0.0 for b in bets)
    total_pl = sum(b.profit_loss_dollars or 0.0 for b in bets)
    clv_vals = [b.clv_prob for b in bets if b.clv_prob is not None]
    mean_clv = _mean(clv_vals)

    # Peak-to-trough drawdown
    running = peak = max_dd = 0.0
    for b in bets:
        running += b.profit_loss_dollars or 0.0
        if running > peak:
            peak = running
        if peak > 0:
            dd = (peak - running) / peak
            if dd > max_dd:
                max_dd = dd

    overall = {
        "total_bets": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": _win_rate(wins, total),
        "roi": _safe_roi(total_pl, total_risked),
        "total_profit_dollars": round(total_pl, 2),
        "total_risked_dollars": round(total_risked, 2),
        "mean_clv": round(mean_clv, 5) if mean_clv is not None else None,
        "median_clv": round(_median(clv_vals), 5) if clv_vals else None,
        "current_drawdown": round(max_dd, 4),
        "status": _clv_status(mean_clv),
    }

    # --- By bet type ---
    by_type: Dict[str, list] = {}
    for b in bets:
        by_type.setdefault(b.bet_type or "unknown", []).append(b)

    by_bet_type = {
        bt: {
            "bets": len(lst),
            "win_rate": _win_rate(sum(1 for b in lst if b.outcome == 1), len(lst)),
            "roi": _safe_roi(
                sum(b.profit_loss_dollars or 0.0 for b in lst),
                sum(b.bet_size_dollars or 0.0 for b in lst),
            ),
        }
        for bt, lst in by_type.items()
    }

    # --- By edge bucket (conservative_edge) ---
    edge_buckets = {
        "2-4%": (0.02, 0.04),
        "4-6%": (0.04, 0.06),
        "6%+":  (0.06, 1.00),
    }
    by_edge_bucket = {}
    for label, (lo, hi) in edge_buckets.items():
        grp = [
            b for b in bets
            if b.conservative_edge is not None and lo <= b.conservative_edge < hi
        ]
        if not grp:
            continue
        g_clv = [b.clv_prob for b in grp if b.clv_prob is not None]
        by_edge_bucket[label] = {
            "bets": len(grp),
            "win_rate": _win_rate(sum(1 for b in grp if b.outcome == 1), len(grp)),
            "roi": _safe_roi(
                sum(b.profit_loss_dollars or 0.0 for b in grp),
                sum(b.bet_size_dollars or 0.0 for b in grp),
            ),
            "mean_clv": round(_mean(g_clv), 5) if g_clv else None,
        }

    # --- Rolling windows ---
    def _window(n: int) -> Dict:
        w = bets[-n:]
        w_clv = [b.clv_prob for b in w if b.clv_prob is not None]
        return {
            "bets": len(w),
            "win_rate": _win_rate(sum(1 for b in w if b.outcome == 1), len(w)),
            "mean_clv": round(_mean(w_clv), 5) if w_clv else None,
        }

    rolling_windows = {
        "last_10": _window(10),
        "last_50": _window(50),
        "last_100": _window(100),
    }

    return {
        "overall": overall,
        "by_bet_type": by_bet_type,
        "by_edge_bucket": by_edge_bucket,
        "rolling_windows": rolling_windows,
    }


# ---------------------------------------------------------------------------
# calculate_clv_analysis
# ---------------------------------------------------------------------------

def calculate_clv_analysis(db: Session) -> Dict:
    """Detailed CLV breakdown: distribution, by-confidence, status."""
    bets = _settled_bets(db)
    clv_vals = [b.clv_prob for b in bets if b.clv_prob is not None]

    if not clv_vals:
        return {"message": "No CLV data yet (requires closing lines)", "bets_with_clv": 0}

    mean_clv = _mean(clv_vals)
    positive_rate = sum(1 for v in clv_vals if v > 0) / len(clv_vals)

    distribution = {
        "strong_negative": sum(1 for v in clv_vals if v < -0.03),
        "negative":        sum(1 for v in clv_vals if -0.03 <= v < -0.01),
        "neutral":         sum(1 for v in clv_vals if -0.01 <= v < 0.01),
        "positive":        sum(1 for v in clv_vals if 0.01 <= v < 0.03),
        "strong_positive": sum(1 for v in clv_vals if v >= 0.03),
    }

    def _conf_bucket(lo: float, hi: float) -> Dict:
        grp = [
            b for b in bets
            if b.clv_prob is not None
            and b.conservative_edge is not None
            and lo <= b.conservative_edge < hi
        ]
        vals = [b.clv_prob for b in grp]
        return {"mean_clv": round(_mean(vals), 5) if vals else None, "count": len(grp)}

    clv_by_confidence = {
        "low_edge":    _conf_bucket(0.00, 0.04),
        "medium_edge": _conf_bucket(0.04, 0.06),
        "high_edge":   _conf_bucket(0.06, 1.00),
    }

    # Top/bottom 10 by CLV
    bets_with_clv = sorted(
        [b for b in bets if b.clv_prob is not None],
        key=lambda b: b.clv_prob,
        reverse=True,
    )
    def _bet_row(b: BetLog) -> Dict:
        return {
            "bet_id": b.id,
            "pick": b.pick,
            "game_date": b.game.game_date.date().isoformat() if b.game else None,
            "clv_prob": b.clv_prob,
            "clv_points": b.clv_points,
            "outcome": b.outcome,
        }

    top_10 = [_bet_row(b) for b in bets_with_clv[:10]]
    bottom_10 = [_bet_row(b) for b in bets_with_clv[-10:]]

    status = _clv_status(mean_clv)
    rec_map = {
        "HEALTHY": "Continue betting — positive CLV sustained.",
        "WARNING": "Monitor closely — CLV near zero. Consider reducing size.",
        "STOP":    "Stop betting — CLV consistently negative. Review model.",
        "UNKNOWN": "Not enough data yet.",
    }

    return {
        "bets_with_clv": len(clv_vals),
        "mean_clv": round(mean_clv, 5),
        "median_clv": round(_median(clv_vals), 5),
        "std_clv": round(_std(clv_vals), 5) if len(clv_vals) >= 2 else None,
        "positive_clv_rate": round(positive_rate, 4),
        "distribution": distribution,
        "clv_by_confidence": clv_by_confidence,
        "top_10_clv": top_10,
        "bottom_10_clv": bottom_10,
        "status": status,
        "recommendation": rec_map.get(status, ""),
    }


# ---------------------------------------------------------------------------
# calculate_calibration
# ---------------------------------------------------------------------------

def calculate_calibration(db: Session) -> Dict:
    """
    Calibration: predicted win probability vs actual win rate per bucket.
    Also computes mean calibration error and Brier score.
    """
    bets = _settled_bets(db)

    bins_config = [
        (0.50, 0.55, "50-55%"),
        (0.55, 0.60, "55-60%"),
        (0.60, 0.65, "60-65%"),
        (0.65, 0.70, "65-70%"),
        (0.70, 0.75, "70-75%"),
        (0.75, 1.00, "75%+"),
    ]

    buckets: Dict[str, List[int]] = {label: [] for _, _, label in bins_config}

    for b in bets:
        if b.model_prob is None:
            continue
        p = b.model_prob
        for lo, hi, label in bins_config:
            if lo <= p < hi:
                buckets[label].append(b.outcome)
                break

    calib_buckets = []
    errors = []
    brier_components = []

    for lo, hi, label in bins_config:
        outcomes = buckets[label]
        if not outcomes:
            continue
        mid = (lo + hi) / 2.0
        actual = sum(outcomes) / len(outcomes)
        err = abs(mid - actual)
        errors.append(err)
        brier_components.extend((mid - o) ** 2 for o in outcomes)
        calib_buckets.append({
            "bin": label,
            "predicted_prob": round(mid, 3),
            "actual_win_rate": round(actual, 4),
            "count": len(outcomes),
            "error": round(err, 4),
        })

    mean_error = _mean(errors)
    brier = _mean(brier_components)

    return {
        "calibration_buckets": calib_buckets,
        "mean_calibration_error": round(mean_error, 4) if mean_error is not None else None,
        "is_well_calibrated": (mean_error < 0.07) if mean_error is not None else None,
        "brier_score": round(brier, 4) if brier is not None else None,
    }


# ---------------------------------------------------------------------------
# calculate_timeline
# ---------------------------------------------------------------------------

def calculate_timeline(db: Session, days: int = 30) -> Dict:
    """
    Daily performance over the last `days` days, plus cumulative series.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)
    bets = _settled_bets(db, cutoff=cutoff)

    if not bets:
        return {"timeline": [], "cumulative_profit": [], "cumulative_roi": []}

    by_date: Dict[str, List[BetLog]] = {}
    for b in bets:
        d = (
            b.game.game_date.date().isoformat()
            if b.game
            else b.timestamp.date().isoformat()
        )
        by_date.setdefault(d, []).append(b)

    timeline = []
    cumulative_profit = []
    cumulative_roi = []
    cum_pl = 0.0
    cum_risked = 0.0

    for date_str in sorted(by_date):
        day = by_date[date_str]
        day_wins = sum(1 for b in day if b.outcome == 1)
        day_pl = sum(b.profit_loss_dollars or 0.0 for b in day)
        day_risked = sum(b.bet_size_dollars or 0.0 for b in day)
        day_clv = [b.clv_prob for b in day if b.clv_prob is not None]

        cum_pl += day_pl
        cum_risked += day_risked
        cumulative_profit.append(round(cum_pl, 2))
        cumulative_roi.append(_safe_roi(cum_pl, cum_risked))

        timeline.append({
            "date": date_str,
            "bets": len(day),
            "wins": day_wins,
            "roi": _safe_roi(day_pl, day_risked),
            "clv": round(_mean(day_clv), 5) if day_clv else None,
            "profit": round(day_pl, 2),
        })

    return {
        "timeline": timeline,
        "cumulative_profit": cumulative_profit,
        "cumulative_roi": cumulative_roi,
    }


# ---------------------------------------------------------------------------
# generate_daily_snapshot
# ---------------------------------------------------------------------------

def generate_daily_snapshot(db: Session) -> PerformanceSnapshot:
    """
    Aggregate yesterday's settled bets into a PerformanceSnapshot row.
    Called by the scheduler at 4 AM daily.
    """
    from datetime import date as _date
    yesterday = datetime.utcnow().date() - timedelta(days=1)
    period_start = datetime(yesterday.year, yesterday.month, yesterday.day)
    period_end = period_start + timedelta(days=1)

    bets = (
        db.query(BetLog)
        .join(Game)
        .options(joinedload(BetLog.game))
        .filter(
            BetLog.outcome.isnot(None),
            BetLog.outcome != -1,
            Game.game_date >= period_start,
            Game.game_date < period_end,
        )
        .all()
    )

    total = len(bets)
    wins = sum(1 for b in bets if b.outcome == 1)
    risked = sum(b.bet_size_dollars or 0.0 for b in bets)
    pl = sum(b.profit_loss_dollars or 0.0 for b in bets)
    clv_vals = [b.clv_prob for b in bets if b.clv_prob is not None]

    snap = PerformanceSnapshot(
        snapshot_date=datetime.utcnow(),
        period_type="daily",
        period_start=period_start,
        period_end=period_end,
        total_bets=total,
        total_wins=wins,
        total_losses=total - wins,
        win_rate=_win_rate(wins, total),
        total_risked=risked,
        total_profit_loss=pl,
        roi=_safe_roi(pl, risked),
        mean_clv=_mean(clv_vals),
        median_clv=_median(clv_vals),
    )
    db.add(snap)
    db.commit()

    logger.info(
        "Daily snapshot %s: %d bets W%d-L%d ROI %.1f%%",
        yesterday, total, wins, total - wins, snap.roi * 100,
    )
    return snap
