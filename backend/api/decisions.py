"""
API endpoints for decision tracking and accuracy reporting.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from backend.fantasy_baseball.decision_tracker import (
    DecisionTracker,
    PlayerDecision,
    get_decision_tracker,
    DailyAccuracy,
    TrendReport,
)

router = APIRouter(prefix="/api/fantasy/decisions", tags=["decisions"])


@router.post("/record")
async def record_decision(
    decision: Dict,
    tracker: DecisionTracker = Depends(get_decision_tracker),
):
    """Record a new lineup decision."""
    try:
        player_decision = PlayerDecision(
            decision_id=decision.get("decision_id") or f"{decision['player_id']}_{decision['date']}",
            date=decision["date"],
            player_name=decision["player_name"],
            player_id=decision["player_id"],
            team=decision["team"],
            recommended_action=decision["recommended_action"],
            confidence=decision.get("confidence", 50),
            factors=decision.get("factors", []),
            opponent=decision.get("opponent", ""),
            opposing_pitcher=decision.get("opposing_pitcher"),
            venue=decision.get("venue", ""),
            weather_factor=decision.get("weather_factor", 1.0),
            projected_stats=decision.get("projected_stats", {}),
        )
        tracker.record_decision(player_decision)
        return {"status": "recorded", "decision_id": player_decision.decision_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/override/{decision_id}")
async def record_override(
    decision_id: str,
    user_action: str,
    reason: Optional[str] = None,
    tracker: DecisionTracker = Depends(get_decision_tracker),
):
    """Record that user overrode a recommendation."""
    try:
        tracker.record_override(decision_id, user_action, reason)
        return {"status": "override_recorded", "decision_id": decision_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resolve/{decision_id}")
async def resolve_decision(
    decision_id: str,
    actual_stats: Dict[str, float],
    game_happened: bool = True,
    tracker: DecisionTracker = Depends(get_decision_tracker),
):
    """Record what actually happened for a decision."""
    try:
        tracker.resolve_decision(decision_id, actual_stats, game_happened)
        return {"status": "resolved", "decision_id": decision_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/accuracy/{date}")
async def get_daily_accuracy(
    date: str,
    tracker: DecisionTracker = Depends(get_decision_tracker),
):
    """Get accuracy report for a specific date."""
    try:
        accuracy = tracker.get_daily_accuracy(date)
        if not accuracy:
            return {"error": "No data for date", "date": date}
        
        return {
            "date": accuracy.date,
            "summary": {
                "total_decisions": accuracy.total_decisions,
                "followed_recommendations": accuracy.followed_recommendations,
                "overrides": accuracy.overrides,
                "accuracy": {
                    "correct": accuracy.correct_predictions,
                    "incorrect": accuracy.incorrect_predictions,
                    "rate": round(
                        accuracy.correct_predictions / 
                        (accuracy.correct_predictions + accuracy.incorrect_predictions), 2
                    ) if (accuracy.correct_predictions + accuracy.incorrect_predictions) > 0 else 0
                }
            },
            "by_confidence": {
                "high_80_100": accuracy.high_conf_accuracy,
                "medium_60_79": accuracy.med_conf_accuracy,
                "low_0_59": accuracy.low_conf_accuracy,
            },
            "by_action": {
                "start_success_rate": accuracy.start_success_rate,
                "bench_success_rate": accuracy.bench_success_rate,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends")
async def get_trend_report(
    days: int = 14,
    tracker: DecisionTracker = Depends(get_decision_tracker),
):
    """Get trend analysis over N days."""
    try:
        report = tracker.get_trend_report(days)
        return {
            "period": {
                "days": report.period_days,
                "start": report.start_date,
                "end": report.end_date,
            },
            "overall_accuracy": report.overall_accuracy,
            "trend_direction": report.trend_direction,
            "best_confidence_threshold": report.best_confidence_threshold,
            "factor_analysis": {
                "overvalued": report.overvalued_factors,
                "undervalued": report.undervalued_factors,
            },
            "suggestions": report.suggested_adjustments,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_decision_dashboard(
    tracker: DecisionTracker = Depends(get_decision_tracker),
):
    """Get complete decision dashboard."""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Today's pending decisions
        today_acc = tracker.get_daily_accuracy(today)
        
        # Yesterday's results
        yesterday_acc = tracker.get_daily_accuracy(yesterday)
        
        # 14-day trend
        trend = tracker.get_trend_report(14)
        
        return {
            "today": {
                "date": today,
                "pending_decisions": today_acc.total_decisions if today_acc else 0,
                "resolved": today_acc.correct_predictions + today_acc.incorrect_predictions if today_acc else 0,
            },
            "yesterday": {
                "date": yesterday,
                "accuracy": round(
                    yesterday_acc.correct_predictions / 
                    (yesterday_acc.correct_predictions + yesterday_acc.incorrect_predictions), 2
                ) if yesterday_acc and (yesterday_acc.correct_predictions + yesterday_acc.incorrect_predictions) > 0 else None,
                "followed_vs_overrides": {
                    "followed": yesterday_acc.followed_recommendations if yesterday_acc else 0,
                    "overrides": yesterday_acc.overrides if yesterday_acc else 0,
                }
            },
            "trend_14d": {
                "accuracy": trend.overall_accuracy,
                "direction": trend.trend_direction,
                "confidence_threshold": trend.best_confidence_threshold,
            },
            "insights": trend.suggested_adjustments[:3] if trend.suggested_adjustments else [],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pending/{date}")
async def get_pending_decisions(
    date: str,
):
    """Get all pending decisions for a date that need resolution."""
    try:
        from backend.fantasy_baseball.nightly_resolution import get_pending_resolutions
        pending = get_pending_resolutions(date)
        return {
            "date": date,
            "pending_count": len(pending),
            "decisions": pending
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resolve-bulk/{date}")
async def resolve_bulk_decisions(
    date: str,
    resolutions: List[Dict],  # [{"decision_id": "...", "actual_stats": {...}}]
    tracker: DecisionTracker = Depends(get_decision_tracker),
):
    """
    Resolve multiple decisions at once.
    
    Example:
    [
      {"decision_id": "pete_alonso_2025-03-27", "actual_stats": {"hr": 1, "r": 2}},
      {"decision_id": "mookie_betts_2025-03-27", "actual_stats": {"hr": 0, "r": 1}}
    ]
    """
    results = []
    for res in resolutions:
        try:
            tracker.resolve_decision(
                res["decision_id"],
                res.get("actual_stats", {}),
                res.get("game_happened", True)
            )
            results.append({"decision_id": res["decision_id"], "status": "resolved"})
        except Exception as e:
            results.append({"decision_id": res["decision_id"], "status": "error", "error": str(e)})
    
    return {
        "date": date,
        "processed": len(results),
        "successful": sum(1 for r in results if r["status"] == "resolved"),
        "results": results
    }


@router.post("/run-nightly-resolution")
async def run_nightly_resolution():
    """Trigger the nightly resolution job (for cron)."""
    try:
        from backend.fantasy_baseball.nightly_resolution import resolve_yesterdays_decisions
        result = resolve_yesterdays_decisions()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
