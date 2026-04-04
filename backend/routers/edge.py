"""
Edge router — all betting/analysis API routes.

Strangler-fig extraction from backend/main.py.
Do NOT import from other backend.routers modules here.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import text, func
from typing import List, Optional
import logging
import os
import re
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from dataclasses import asdict

from backend.models import (
    Game,
    Prediction,
    BetLog,
    ClosingLine,
    PerformanceSnapshot,
    ModelParameter,
    TeamProfile,
    DBAlert,
    DataIngestionLog,
    DataFetch,
    SessionLocal,
    get_db,
)
from backend.auth import verify_api_key, verify_admin_api_key
from backend.services.clv import calculate_clv_full
from backend.services.performance import (
    calculate_summary_stats,
    calculate_clv_analysis,
    calculate_calibration,
    calculate_model_accuracy,
    calculate_timeline,
    calculate_financial_metrics,
)
from backend.services.alerts import check_performance_alerts
from backend.services.recalibration import compute_dynamic_weights
from backend.services.odds_monitor import get_odds_monitor
from backend.services.portfolio import get_portfolio_manager
from backend.utils.env_utils import get_float_env
from backend.schemas import (
    TodaysPredictionsResponse,
    BetLogResponse,
    OutcomeResponse,
    AnalysisTriggerResponse,
    OracleFlaggedResponse,
    OraclePredictionDetail,
    BetLogCreate,
    OutcomeUpdate,
    PredictionResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# HELPERS
# ============================================================================

def _get_model_param(db: Session, name: str) -> Optional[ModelParameter]:
    return (
        db.query(ModelParameter)
        .filter(ModelParameter.parameter_name == name)
        .order_by(ModelParameter.effective_date.desc())
        .first()
    )


def get_effective_bankroll(db: Session) -> float:
    """Return the active bankroll: DB override if set, else STARTING_BANKROLL env var."""
    row = _get_model_param(db, "current_bankroll")
    if row and row.parameter_value and row.parameter_value > 0:
        return row.parameter_value
    return get_float_env("STARTING_BANKROLL", "1000")


# ============================================================================
# TOURNAMENT
# ============================================================================

@router.get("/api/tournament/bracket-projection")
async def get_bracket_projection(
    n_sims: int = Query(default=10000, ge=1000, le=50000),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """
    Monte Carlo NCAA Tournament bracket projection.

    Uses the V9.1 tournament module (composite KenPom+BartTorvik ratings,
    round-specific SD multipliers, historical seed upset blending).

    Falls back to data/bracket_2026.json when BALLDONTLIE_API_KEY is not set,
    so the endpoint always returns a result during tournament weeks.

    Query parameters:
      n_sims: Number of Monte Carlo simulations (1,000 - 50,000; default 10,000).
    """
    import json as _json
    from pathlib import Path as _Path
    from backend.tournament.matchup_predictor import TournamentTeam
    from backend.tournament.bracket_simulator import run_monte_carlo

    REGIONS = ["east", "south", "west", "midwest"]
    BRACKET_JSON = _Path(__file__).resolve().parent.parent.parent / "data" / "bracket_2026.json"

    if not BRACKET_JSON.exists():
        raise HTTPException(
            status_code=503,
            detail="bracket_2026.json not found. Re-deploy or run build_bracket_from_db.py.",
        )

    with open(BRACKET_JSON, encoding="utf-8") as _f:
        raw = _json.load(_f)

    bracket: dict = {}
    for region in REGIONS:
        if region not in raw:
            continue
        bracket[region] = [
            TournamentTeam(
                name=t["name"],
                seed=t["seed"],
                region=region,
                composite_rating=t.get("composite_rating", 0.0),
                kp_adj_em=t.get("kp_adj_em"),
                bt_adj_em=t.get("bt_adj_em"),
                pace=t.get("pace", 68.0),
                three_pt_rate=t.get("three_pt_rate", 0.35),
                def_efg_pct=t.get("def_efg_pct", 0.50),
                conference=t.get("conference", ""),
                tournament_exp=t.get("tournament_exp", 0.70),
            )
            for t in raw[region]
        ]

    if len(bracket) < 4:
        raise HTTPException(
            status_code=503,
            detail=f"bracket_2026.json is incomplete ({len(bracket)}/4 regions).",
        )

    try:
        results = run_monte_carlo(bracket, n_sims=n_sims, n_workers=2, base_seed=42)
    except Exception as exc:
        logger.error("Bracket Monte Carlo failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Simulation error: {exc}")

    all_teams = [t for teams in bracket.values() for t in teams]
    seed_map = {t.name: t.seed for t in all_teams}
    region_map = {t.name: t.region for t in all_teams}

    sorted_champ = sorted(results.championship.items(), key=lambda x: -x[1])
    projected_champion = sorted_champ[0][0] if sorted_champ else None

    top_f4 = sorted(results.final_four.items(), key=lambda x: -x[1])[:4]
    projected_final_four = [t for t, _ in top_f4]

    upset_alerts = [
        {
            "team": t.name,
            "seed": t.seed,
            "region": t.region,
            "r64_win_prob": round(results.round_of_32.get(t.name, 0) * 100, 1),
        }
        for t in all_teams
        if t.seed >= 10 and results.round_of_32.get(t.name, 0) >= 0.35
    ]

    advancement_probs = {
        t: {
            "seed": seed_map.get(t, 0),
            "region": region_map.get(t, ""),
            "r32_pct": round(results.round_of_32.get(t, 0) * 100, 1),
            "s16_pct": round(results.sweet_sixteen.get(t, 0) * 100, 1),
            "e8_pct": round(results.elite_eight.get(t, 0) * 100, 1),
            "f4_pct": round(results.final_four.get(t, 0) * 100, 1),
            "runner_up_pct": round(results.runner_up.get(t, 0) * 100, 1),
            "champion_pct": round(results.championship.get(t, 0) * 100, 1),
        }
        for t in results.championship
    }

    return {
        "n_sims": results.n_sims,
        "data_source": "bracket_2026.json (V9.1 composite ratings)",
        "projected_champion": projected_champion,
        "projected_final_four": projected_final_four,
        "upset_alerts": upset_alerts,
        "advancement_probs": advancement_probs,
        "avg_upsets_per_tournament": round(results.avg_upsets_per_tournament, 1),
        "avg_championship_margin": round(results.avg_championship_margin, 1),
    }


# ============================================================================
# PREDICTIONS
# ============================================================================

@router.get("/api/predictions/today", response_model=TodaysPredictionsResponse)
async def get_todays_predictions(
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """Get all UPCOMING predictions from the latest analysis batch, deduplicated by game."""
    today_utc = datetime.utcnow().date()
    now_utc = datetime.utcnow()

    _TIER_PRIORITY = {"nightly": 0, "opener": 1}

    predictions = (
        db.query(Prediction)
        .join(Game)
        .filter(Prediction.prediction_date == today_utc)
        .filter(Game.game_date > now_utc)
        .order_by(Game.game_date.asc())
        .options(joinedload(Prediction.game))
        .all()
    )

    seen: dict = {}
    for p in predictions:
        gid = p.game_id
        if gid not in seen:
            seen[gid] = p
        else:
            cur_pri = _TIER_PRIORITY.get(seen[gid].run_tier or "", 99)
            new_pri = _TIER_PRIORITY.get(p.run_tier or "", 99)
            if new_pri < cur_pri or (
                new_pri == cur_pri
                and (p.edge_conservative or 0) > (seen[gid].edge_conservative or 0)
            ):
                seen[gid] = p

    deduped = sorted(seen.values(), key=lambda p: p.game.game_date)

    return TodaysPredictionsResponse(
        date=today_utc,
        total_games=len(deduped),
        bets_recommended=len([p for p in deduped if p.verdict.startswith("Bet")]),
        predictions=deduped,
    )


@router.get("/api/predictions/today/all", response_model=TodaysPredictionsResponse)
async def get_todays_predictions_all(
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Get ALL predictions for today (including games that have started).
    Used to review bets after games have begun.
    """
    today_utc = datetime.utcnow().date()

    _TIER_PRIORITY = {"nightly": 0, "opener": 1}

    predictions = (
        db.query(Prediction)
        .join(Game)
        .filter(Prediction.prediction_date == today_utc)
        .order_by(Game.game_date.asc())
        .options(joinedload(Prediction.game))
        .all()
    )

    seen: dict = {}
    for p in predictions:
        gid = p.game_id
        if gid not in seen:
            seen[gid] = p
        else:
            cur_pri = _TIER_PRIORITY.get(seen[gid].run_tier or "", 99)
            new_pri = _TIER_PRIORITY.get(p.run_tier or "", 99)
            if new_pri < cur_pri or (
                new_pri == cur_pri
                and (p.edge_conservative or 0) > (seen[gid].edge_conservative or 0)
            ):
                seen[gid] = p

    deduped = sorted(seen.values(), key=lambda p: p.game.game_date)

    return TodaysPredictionsResponse(
        date=today_utc,
        total_games=len(deduped),
        bets_recommended=len([p for p in deduped if p.verdict.startswith("Bet")]),
        predictions=deduped,
    )


@router.get("/api/predictions/bets")
async def get_recommended_bets(
    days: int = Query(default=7, ge=1, le=30),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """Get all recommended bets from the last N days"""
    cutoff = datetime.utcnow() - timedelta(days=days)

    bets = (
        db.query(Prediction)
        .join(Game)
        .filter(
            Prediction.created_at >= cutoff,
            Prediction.verdict.like("Bet%")
        )
        .order_by(Prediction.created_at.desc())
        .all()
    )

    return {
        "period_days": days,
        "total_bets": len(bets),
        "bets": [
            {
                "game_id": b.game_id,
                "date": b.game.game_date.isoformat(),
                "matchup": f"{b.game.away_team} @ {b.game.home_team}",
                "verdict": b.verdict,
                "edge_point": b.edge_point,
                "edge_conservative": b.edge_conservative,
                "recommended_units": b.recommended_units,
            }
            for b in bets
        ]
    }


@router.get("/api/predictions/game/{game_id}")
async def get_game_prediction(
    game_id: int,
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Get the most recent prediction for a specific game.
    Returns the latest prediction regardless of run_tier.
    """
    prediction = (
        db.query(Prediction)
        .filter(Prediction.game_id == game_id)
        .order_by(Prediction.created_at.desc())
        .first()
    )

    if not prediction:
        return {"message": "No prediction found for this game", "game_id": game_id}

    bet_details = {
        "has_bet": prediction.verdict.startswith("Bet"),
        "pick": None,
        "bet_type": None,
        "odds": None,
        "units": prediction.recommended_units,
    }

    if bet_details["has_bet"]:
        match = re.search(r'Bet\s+[\d.]+u\s+([^@]+)\s+@\s+([-+]?\d+)', prediction.verdict)
        if match:
            pick_str = match.group(1).strip()
            odds_str = match.group(2).strip()

            bet_details["pick"] = pick_str
            bet_details["odds"] = int(odds_str)

            if "/" in pick_str and ("U" in pick_str or "O" in pick_str):
                bet_details["bet_type"] = "total"
            elif "-" in pick_str or "+" in pick_str:
                bet_details["bet_type"] = "spread"
            else:
                bet_details["bet_type"] = "moneyline"
        else:
            bet_details["pick"] = ""
            bet_details["bet_type"] = "spread"
            bet_details["odds"] = -110

    return {
        "game_id": game_id,
        "prediction_id": prediction.id,
        "verdict": prediction.verdict,
        "projected_margin": prediction.projected_margin,
        "point_prob": prediction.point_prob,
        "edge_point": prediction.edge_point,
        "edge_conservative": prediction.edge_conservative,
        "recommended_units": prediction.recommended_units,
        "bet_details": bet_details,
    }


@router.get("/api/predictions/parlays")
async def get_optimal_parlays(
    max_legs: int = Query(default=3, ge=2, le=4),
    max_parlays: int = Query(default=10, ge=1, le=50),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """Build optimal cross-game parlays from today's +EV straight bets."""
    from backend.services.parlay_engine import build_optimal_parlays

    starting_bankroll = get_effective_bankroll(db)
    max_daily_pct     = get_float_env("MAX_DAILY_EXPOSURE_PCT", "20.0")
    max_daily_dollars = starting_bankroll * max_daily_pct / 100.0

    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    already_allocated: float = (
        db.query(func.sum(BetLog.bet_size_dollars))
        .filter(BetLog.timestamp >= today_start)
        .filter(BetLog.is_paper_trade.is_(True))
        .scalar()
        or 0.0
    )

    true_remaining_capacity = max(0.0, max_daily_dollars - already_allocated)

    if true_remaining_capacity <= 0.0:
        return {
            "date": datetime.utcnow().date().isoformat(),
            "message": "Portfolio capacity exhausted - no room for parlay sizing",
            "capital_allocated_dollars": round(already_allocated, 2),
            "max_daily_dollars": round(max_daily_dollars, 2),
            "parlays": [],
        }

    today_utc = datetime.utcnow().date()
    predictions = (
        db.query(Prediction)
        .join(Game)
        .filter(Prediction.prediction_date == today_utc)
        .filter(Prediction.verdict.like("Bet%"))
        .options(joinedload(Prediction.game))
        .all()
    )

    if not predictions:
        return {
            "message": "No +EV bets available today for parlay construction",
            "parlays": [],
        }

    slate_bets = []
    for pred in predictions:
        game  = pred.game
        calcs = (pred.full_analysis or {}).get("calculations", {})
        bet_side = calcs.get("bet_side", "home")
        spread   = calcs.get("spread") or (
            (pred.full_analysis or {}).get("inputs", {}).get("odds", {}).get("spread")
        ) or 0.0
        if bet_side == "away":
            away_spread = -spread
            sign = "+" if away_spread > 0 else ""
            pick = f"{game.away_team} {sign}{away_spread:.1f}"
        else:
            sign = "+" if spread > 0 else ""
            pick = f"{game.home_team} {sign}{spread:.1f}"

        slate_bets.append({
            "game_id":           pred.game_id,
            "pick":              pick,
            "edge_conservative": pred.edge_conservative,
            "lower_ci_prob":     pred.lower_ci_prob,
            "full_analysis":     pred.full_analysis or {},
        })

    parlays = build_optimal_parlays(
        slate_bets,
        max_legs=max_legs,
        max_parlays=max_parlays,
        remaining_capacity_dollars=true_remaining_capacity,
        bankroll=starting_bankroll,
    )

    return {
        "date":                        today_utc.isoformat(),
        "capital_allocated_dollars":   round(already_allocated, 2),
        "remaining_capacity_dollars":  round(true_remaining_capacity, 2),
        "max_daily_dollars":           round(max_daily_dollars, 2),
        "straight_bets_available":     len(slate_bets),
        "parlays_generated":           len(parlays),
        "max_legs":                    max_legs,
        "parlays":                     parlays,
    }


# ============================================================================
# PERFORMANCE
# ============================================================================

@router.get("/api/performance/summary")
async def get_performance_summary(
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Full performance summary: overall metrics, by-type, by-edge-bucket, and rolling windows."""
    return calculate_summary_stats(db)


@router.get("/api/performance/clv-analysis")
async def get_clv_analysis(
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Detailed CLV breakdown: distribution, by-confidence, top/bottom 10."""
    return calculate_clv_analysis(db)


@router.get("/api/performance/calibration")
async def get_calibration_data(
    days: int = Query(default=90, ge=1, le=365),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Model calibration: predicted probability vs actual win rate + Brier score."""
    return calculate_calibration(db, days=days)


@router.get("/api/performance/model-accuracy")
async def get_model_accuracy(
    days: int = Query(default=90, ge=1, le=365),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Model accuracy metrics over resolved predictions."""
    return calculate_model_accuracy(db, days=days)


@router.get("/api/performance/timeline")
async def get_performance_timeline(
    days: int = Query(default=30, ge=1, le=365),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Daily performance timeline with cumulative P&L and ROI series."""
    return calculate_timeline(db, days=days)


@router.get("/api/performance/financial-metrics")
async def get_financial_metrics(
    days: int = Query(default=90, ge=7, le=365),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Sharpe ratio, Sortino ratio, expected Kelly growth, max drawdown, Calmar."""
    return calculate_financial_metrics(db, days=days)


@router.get("/api/performance/by-team")
async def get_performance_by_team(
    days: int = Query(default=90, ge=7, le=365),
    min_bets: int = Query(default=2, ge=1, le=20, description="Minimum bets to include a team"),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Per-team win/loss breakdown for settled bets."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    bets = (
        db.query(BetLog)
        .join(Game)
        .options(joinedload(BetLog.game))
        .filter(
            BetLog.outcome.isnot(None),
            BetLog.outcome != -1,
            BetLog.timestamp >= cutoff,
        )
        .all()
    )

    if not bets:
        return {"teams": [], "total_bets": 0, "days": days}

    import re as _re

    def _extract_pick_team(pick: str) -> str:
        if not pick:
            return "Unknown"
        stripped = pick.strip()
        m = _re.match(r"^(.+?)\s+[+-]?\d+\.?\d*\s*$", stripped)
        if m:
            return m.group(1).strip()
        return stripped

    team_stats: dict = {}
    for b in bets:
        team_name = _extract_pick_team(b.pick or "")
        if not team_name:
            continue

        if team_name not in team_stats:
            team_stats[team_name] = {
                "team": team_name,
                "bets": 0,
                "wins": 0,
                "losses": 0,
                "total_pl_dollars": 0.0,
                "total_risked": 0.0,
                "edges": [],
            }
        s = team_stats[team_name]
        s["bets"] += 1
        s["total_pl_dollars"] += b.profit_loss_dollars or 0.0
        s["total_risked"] += b.bet_size_dollars or 0.0
        if b.outcome == 1:
            s["wins"] += 1
        else:
            s["losses"] += 1
        if b.conservative_edge is not None:
            s["edges"].append(b.conservative_edge)

    results = []
    for team_name, s in team_stats.items():
        if s["bets"] < min_bets:
            continue
        win_rate = s["wins"] / s["bets"] if s["bets"] > 0 else 0.0
        roi = s["total_pl_dollars"] / s["total_risked"] if s["total_risked"] > 0 else 0.0
        mean_edge = sum(s["edges"]) / len(s["edges"]) if s["edges"] else None
        anomaly = s["bets"] >= 3 and (win_rate >= 0.80 or win_rate <= 0.20)
        results.append({
            "team": team_name,
            "bets": s["bets"],
            "wins": s["wins"],
            "losses": s["losses"],
            "win_rate": round(win_rate, 4),
            "roi": round(roi, 4),
            "total_pl_dollars": round(s["total_pl_dollars"], 2),
            "mean_edge": round(mean_edge, 4) if mean_edge is not None else None,
            "anomaly_flag": anomaly,
        })

    results.sort(key=lambda x: x["win_rate"], reverse=True)
    anomalies = [r for r in results if r["anomaly_flag"]]

    return {
        "teams": results,
        "anomalies": anomalies,
        "total_bets": len(bets),
        "total_teams": len(results),
        "days": days,
    }


@router.get("/api/performance/source-weights")
async def get_source_weights(
    history_days: int = Query(default=30, ge=1, le=365),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Current dynamic source weights and 30-day change history."""
    from backend.services.recalibration import load_current_params
    current = load_current_params(db)
    weights = {
        "weight_kenpom":     current.get("weight_kenpom",     0.342),
        "weight_barttorvik": current.get("weight_barttorvik",  0.333),
        "weight_evanmiya":   current.get("weight_evanmiya",    0.325),
    }
    cutoff = datetime.utcnow() - timedelta(days=history_days)
    history = (
        db.query(ModelParameter)
        .filter(
            ModelParameter.parameter_name.in_(
                ["weight_kenpom", "weight_barttorvik", "weight_evanmiya"]
            ),
            ModelParameter.effective_date >= cutoff,
        )
        .order_by(ModelParameter.effective_date.desc())
        .limit(300)
        .all()
    )
    history_data = [
        {
            "date":      h.effective_date.isoformat() if h.effective_date else None,
            "parameter": h.parameter_name,
            "value":     h.parameter_value,
            "reason":    h.reason,
        }
        for h in history
    ]
    return {
        "current_weights": weights,
        "history":         history_data,
        "history_days":    history_days,
    }


@router.get("/api/performance/alerts")
async def get_performance_alerts(
    include_acknowledged: bool = Query(default=False),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Return active system health alerts from the database."""
    query = db.query(DBAlert).order_by(DBAlert.created_at.desc())
    if not include_acknowledged:
        query = query.filter(DBAlert.acknowledged == False)

    db_alerts = query.limit(50).all()

    live_alerts = check_performance_alerts(db)

    overall_severity = "OK"
    for a in live_alerts:
        if a.severity == "CRITICAL":
            overall_severity = "CRITICAL"
            break
        if a.severity == "WARNING":
            overall_severity = "WARNING"

    return {
        "alerts": [
            {
                "id": a.id,
                "alert_type": a.alert_type,
                "severity": a.severity,
                "message": a.message,
                "threshold": a.threshold,
                "current_value": a.current_value,
                "acknowledged": a.acknowledged,
                "created_at": a.created_at.isoformat(),
            }
            for a in db_alerts
        ],
        "live_alerts": [a.to_dict() for a in live_alerts],
        "status": overall_severity,
    }


# ============================================================================
# BET LOGS
# ============================================================================

@router.post("/api/bets/log", response_model=BetLogResponse)
async def log_bet(
    bet_data: BetLogCreate,
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Manually log a paper trade or real bet."""
    bet = BetLog(
        game_id=bet_data.game_id,
        prediction_id=bet_data.prediction_id,
        pick=bet_data.pick,
        bet_type=bet_data.bet_type,
        odds_taken=bet_data.odds_taken,
        bankroll_at_bet=bet_data.bankroll_at_bet,
        kelly_full=bet_data.kelly_full,
        kelly_fractional=bet_data.kelly_fractional,
        bet_size_pct=bet_data.bet_size_pct,
        bet_size_units=bet_data.bet_size_units,
        bet_size_dollars=bet_data.bet_size_dollars,
        model_prob=bet_data.model_prob,
        lower_ci_prob=bet_data.lower_ci_prob,
        point_edge=bet_data.point_edge,
        conservative_edge=bet_data.conservative_edge,
        is_paper_trade=bet_data.is_paper_trade,
        is_backfill=bet_data.is_backfill,
        notes=bet_data.notes,
    )
    db.add(bet)
    db.commit()
    db.refresh(bet)

    logger.info("Bet logged: %s %.2fu by %s", bet.pick, bet.bet_size_units, user)

    return BetLogResponse(
        message="Bet logged successfully",
        bet_id=bet.id,
        pick=bet.pick,
        bet_size_units=bet.bet_size_units,
        is_paper_trade=bet.is_paper_trade,
    )


@router.put("/api/bets/{bet_id}/outcome", response_model=OutcomeResponse)
async def update_bet_outcome(
    bet_id: int,
    payload: OutcomeUpdate,
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Settle a bet: record outcome, compute P&L, and calculate CLV."""
    bet = db.query(BetLog).filter(BetLog.id == bet_id).first()
    if not bet:
        raise HTTPException(status_code=404, detail="Bet not found")

    bet.outcome = payload.outcome

    if payload.outcome == 1:  # Win
        if bet.odds_taken > 0:
            profit = bet.bet_size_dollars * (bet.odds_taken / 100.0)
        else:
            profit = bet.bet_size_dollars * (100.0 / abs(bet.odds_taken))
        bet.profit_loss_dollars = round(profit, 2)
    else:  # Loss
        bet.profit_loss_dollars = round(-bet.bet_size_dollars, 2)

    if bet.bankroll_at_bet and bet.bankroll_at_bet > 0:
        unit_value = bet.bankroll_at_bet / 100.0
        bet.profit_loss_units = round(bet.profit_loss_dollars / unit_value, 4)
    else:
        bet.profit_loss_units = None

    clv_grade: Optional[str] = None

    if payload.closing_odds is not None:
        try:
            opening_spread: Optional[float] = None
            if bet.prediction_id:
                pred = db.query(Prediction).filter(Prediction.id == bet.prediction_id).first()
                if pred and pred.full_analysis:
                    opening_spread = pred.full_analysis.get("inputs", {}).get("odds", {}).get("spread")

            base_sd = get_float_env("BASE_SD", "11.0")

            clv = calculate_clv_full(
                opening_odds=bet.odds_taken,
                closing_odds=payload.closing_odds,
                opening_spread=opening_spread,
                closing_spread=payload.closing_spread,
                other_side_closing_odds=payload.closing_odds_other_side,
                base_sd=base_sd,
            )

            bet.closing_line = payload.closing_odds
            bet.clv_points = round(clv.clv_points, 3)
            bet.clv_prob = round(clv.clv_prob, 4)
            clv_grade = clv.grade()

            logger.info(
                "CLV for bet %d: %.3f pts / %.2f%% (%s)",
                bet_id,
                clv.clv_points,
                clv.clv_prob * 100,
                clv_grade,
            )
        except Exception as exc:
            logger.warning("CLV calculation failed for bet %d: %s", bet_id, exc)

    db.commit()

    logger.info(
        "Bet %d settled: %s, P&L $%.2f",
        bet_id,
        "WIN" if payload.outcome else "LOSS",
        bet.profit_loss_dollars,
    )

    return OutcomeResponse(
        message="Outcome updated",
        bet_id=bet.id,
        outcome=bet.outcome,
        profit_loss_dollars=bet.profit_loss_dollars,
        profit_loss_units=bet.profit_loss_units,
        clv_points=bet.clv_points,
        clv_prob=bet.clv_prob,
        clv_grade=clv_grade,
    )


@router.post("/api/bets/{bet_id}/placed")
async def mark_bet_placed(
    bet_id: int,
    placed: bool = True,
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Mark an existing BetLog as placed (executed=True) or unplaced (executed=False)."""
    bet = db.query(BetLog).filter(BetLog.id == bet_id).first()
    if not bet:
        raise HTTPException(status_code=404, detail="Bet not found")
    bet.executed = placed
    db.commit()
    logger.info("Bet %d marked as %s by %s", bet_id, "placed" if placed else "unplaced", user)
    return {"success": True, "bet_id": bet_id, "placed": placed}


# ============================================================================
# GAMES
# ============================================================================

@router.get("/api/games/recent")
async def get_recent_games(
    days_back: int = Query(default=7, ge=1, le=30),
    days_ahead: int = Query(default=3, ge=1, le=7),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Return recent and upcoming games, used to populate the bet-entry game selector."""
    start = datetime.utcnow() - timedelta(days=days_back)
    end = datetime.utcnow() + timedelta(days=days_ahead)

    games = (
        db.query(Game)
        .filter(Game.game_date >= start, Game.game_date <= end)
        .order_by(Game.game_date.desc())
        .all()
    )

    return {
        "games": [
            {
                "id": g.id,
                "matchup": f"{g.away_team} @ {g.home_team}",
                "home_team": g.home_team,
                "away_team": g.away_team,
                "game_date": g.game_date.isoformat(),
                "completed": g.completed,
                "home_score": g.home_score,
                "away_score": g.away_score,
            }
            for g in games
        ]
    }


# ============================================================================
# BET LOG QUERIES
# ============================================================================

@router.get("/api/bets")
async def get_bet_logs(
    status: str = Query(default="all", description="all | pending | settled | cancelled | placed"),
    days: int = Query(default=60, ge=1, le=365),
    dedup: bool = Query(default=True, description="Keep only the first BetLog per game per day"),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Return bet logs with optional status filter and date window."""
    cutoff = datetime.utcnow() - timedelta(days=days)

    query = (
        db.query(BetLog)
        .join(Game)
        .filter(BetLog.timestamp >= cutoff)
        .options(joinedload(BetLog.game))
    )

    if status == "pending":
        query = query.filter(BetLog.outcome.is_(None))
    elif status == "settled":
        query = query.filter(BetLog.outcome.isnot(None), BetLog.outcome != -1)
    elif status == "cancelled":
        query = query.filter(BetLog.outcome == -1)
    elif status == "placed":
        query = query.filter(BetLog.executed.is_(True), BetLog.outcome != -1)
    else:
        query = query.filter(BetLog.outcome != -1)

    bets = query.order_by(BetLog.id.asc()).all()

    if dedup:
        seen: set = set()
        deduped: list = []
        for b in bets:
            bet_date = b.timestamp.date() if b.timestamp else None
            key = (b.game_id, bet_date)
            if key not in seen:
                seen.add(key)
                deduped.append(b)
        bets = deduped

    bets = sorted(bets, key=lambda b: b.timestamp or datetime.min, reverse=True)

    return {
        "total": len(bets),
        "bets": [
            {
                "id": b.id,
                "game_id": b.game_id,
                "matchup": f"{b.game.away_team} @ {b.game.home_team}",
                "game_date": b.game.game_date.isoformat(),
                "pick": b.pick,
                "bet_type": b.bet_type,
                "odds_taken": b.odds_taken,
                "bet_size_units": b.bet_size_units,
                "bet_size_dollars": b.bet_size_dollars,
                "model_prob": b.model_prob,
                "outcome": b.outcome,
                "profit_loss_dollars": b.profit_loss_dollars,
                "profit_loss_units": b.profit_loss_units,
                "clv_points": b.clv_points,
                "clv_prob": b.clv_prob,
                "is_paper_trade": b.is_paper_trade,
                "timestamp": b.timestamp.isoformat() if b.timestamp else None,
                "notes": b.notes,
            }
            for b in bets
        ],
    }


@router.get("/api/closing-lines")
async def get_closing_lines_batch(
    game_ids: str = Query(description="Comma-separated game IDs"),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Return closing lines for multiple games in one query. Returns {game_id: data_or_null}."""
    try:
        ids = [int(x.strip()) for x in game_ids.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="game_ids must be comma-separated integers")

    rows = (
        db.query(ClosingLine)
        .filter(ClosingLine.game_id.in_(ids))
        .order_by(ClosingLine.captured_at.desc())
        .all()
    )
    seen: dict = {}
    for cl in rows:
        if cl.game_id not in seen:
            seen[cl.game_id] = {
                "game_id": cl.game_id,
                "captured_at": cl.captured_at.isoformat() if cl.captured_at else None,
                "spread": cl.spread,
                "spread_odds": cl.spread_odds,
                "total": cl.total,
                "total_odds": cl.total_odds,
                "moneyline_home": cl.moneyline_home,
                "moneyline_away": cl.moneyline_away,
            }
    result = {gid: seen.get(gid) for gid in ids}
    return result


@router.get("/api/closing-lines/{game_id}")
async def get_closing_lines(
    game_id: int,
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Return the most recent closing line capture for a game."""
    cl = (
        db.query(ClosingLine)
        .filter(ClosingLine.game_id == game_id)
        .order_by(ClosingLine.captured_at.desc())
        .first()
    )
    if not cl:
        raise HTTPException(status_code=404, detail="No closing line found for this game")
    return {
        "game_id": cl.game_id,
        "captured_at": cl.captured_at.isoformat() if cl.captured_at else None,
        "spread": cl.spread,
        "spread_odds": cl.spread_odds,
        "total": cl.total,
        "total_odds": cl.total_odds,
        "moneyline_home": cl.moneyline_home,
        "moneyline_away": cl.moneyline_away,
    }


@router.get("/api/performance/history")
async def get_performance_history(
    days: int = Query(default=90, ge=1, le=365),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Return time-series data for cumulative P&L and rolling win-rate charts."""
    cutoff = datetime.utcnow() - timedelta(days=days)

    bets = (
        db.query(BetLog)
        .join(Game)
        .filter(
            BetLog.outcome.isnot(None),
            BetLog.timestamp >= cutoff,
        )
        .options(joinedload(BetLog.game))
        .order_by(Game.game_date.asc())
        .all()
    )

    if not bets:
        return {"data_points": []}

    cumulative_pl = 0.0
    cumulative_units = 0.0
    wins = 0

    data_points = []
    for i, b in enumerate(bets, start=1):
        if b.outcome == 1:
            wins += 1
        cumulative_pl += b.profit_loss_dollars or 0.0
        cumulative_units += b.profit_loss_units or 0.0

        data_points.append(
            {
                "bet_number": i,
                "date": b.game.game_date.date().isoformat(),
                "bet_id": b.id,
                "pick": b.pick,
                "outcome": b.outcome,
                "pl_dollars": b.profit_loss_dollars,
                "cumulative_pl_dollars": round(cumulative_pl, 2),
                "cumulative_pl_units": round(cumulative_units, 4),
                "win_rate": round(wins / i, 4),
                "clv_prob": b.clv_prob,
            }
        )

    return {"data_points": data_points}


# ============================================================================
# FEATURE FLAGS (read side)
# ============================================================================

_ALLOWED_FLAGS = {"draft_board_enabled"}


@router.get("/api/feature-flags")
async def get_feature_flags(
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Return feature flag values. Unset flags return their default."""
    defaults = {"draft_board_enabled": True}
    result = dict(defaults)
    for flag in _ALLOWED_FLAGS:
        row = _get_model_param(db, flag)
        if row is not None and row.parameter_value_json is not None:
            result[flag] = bool(row.parameter_value_json)
    return result
