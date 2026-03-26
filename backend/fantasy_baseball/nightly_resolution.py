"""
Nightly job to resolve fantasy baseball decisions from the previous day.

This should run after all games have completed (around midnight ET).
"""

import logging
from datetime import datetime, timedelta

from backend.fantasy_baseball.decision_tracker import get_decision_tracker
from backend.fantasy_baseball.yahoo_client import YahooFantasyClient

logger = logging.getLogger(__name__)


def resolve_yesterdays_decisions() -> dict:
    """
    Resolve all pending decisions from yesterday with actual results.
    
    Returns summary of resolutions.
    """
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    tracker = get_decision_tracker()
    
    # Load pending decisions
    from backend.fantasy_baseball.decision_tracker import PlayerDecision
    
    pending = []
    for decision in tracker._load_decisions_for_date(yesterday):
        if decision.outcome == "pending":
            pending.append(decision)
    
    if not pending:
        logger.info(f"No pending decisions to resolve for {yesterday}")
        return {"date": yesterday, "resolved": 0, "skipped": 0}
    
    logger.info(f"Resolving {len(pending)} decisions for {yesterday}")
    
    # Try to get actual stats from Yahoo
    try:
        client = YahooFantasyClient()
        # TODO: Implement fetching actual stats from Yahoo
        # For now, we'll need manual resolution
        logger.info("Automatic stat fetching not yet implemented - decisions remain pending")
    except Exception as e:
        logger.warning(f"Could not fetch Yahoo stats: {e}")
    
    return {
        "date": yesterday,
        "pending_count": len(pending),
        "message": "Use manual resolution endpoint with actual stats",
        "players": [d.player_name for d in pending[:10]]  # First 10
    }


def get_pending_resolutions(date: str = None) -> list:
    """
    Get list of decisions pending resolution for a date.
    
    Args:
        date: YYYY-MM-DD, defaults to yesterday
    """
    if date is None:
        date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    tracker = get_decision_tracker()
    decisions = tracker._load_decisions_for_date(date)
    
    pending = [
        {
            "decision_id": d.decision_id,
            "player_name": d.player_name,
            "team": d.team,
            "recommended_action": d.recommended_action,
            "confidence": d.confidence,
            "opponent": d.opponent,
        }
        for d in decisions
        if d.outcome == "pending"
    ]
    
    return pending


if __name__ == "__main__":
    # Run resolution
    result = resolve_yesterdays_decisions()
    print(f"Resolution result: {result}")
