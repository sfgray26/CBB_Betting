"""
Line Movement Monitor — O-10

Periodically checks for significant movement in consensus lines vs.
original spreads recorded in active bet logs.

Logic:
  1. Poll The Odds API for current consensus lines.
  2. Query database for pending bets (outcome is NULL).
  3. Compare current consensus spread with original spread from bet.pick.
  4. If abs(delta) >= 1.5 points:
     - Re-analyze the game using CachedGameContext from Prediction record.
     - If fresh edge < MIN_BET_EDGE, log LINE_MOVEMENT_ABANDON.
     - Send Discord alert via coordinator.py.
"""

import logging
import math
from datetime import datetime
from typing import Dict, List, Optional

from backend.betting_model import CBBEdgeModel, ReanalysisEngine, CachedGameContext
from backend.models import BetLog, Game, Prediction, SessionLocal
from backend.services.odds import OddsAPIClient
from backend.services.bet_tracker import parse_pick
from backend.services.coordinator import send_line_movement_alert
from backend.utils.env_utils import get_float_env

logger = logging.getLogger(__name__)


def check_line_movements() -> Dict:
    """
    Main job function called by the scheduler every 30 minutes.
    """
    if OddsAPIClient.quota_is_low():
        logger.warning("Skipping line movement check -- quota low")
        return {"bets_checked": 0, "significant_moves": 0, "abandonments": 0, "errors": ["quota_paused"]}

    logger.info("Starting check_line_movements job")

    db = SessionLocal()
    client = OddsAPIClient()
    model = CBBEdgeModel()
    
    # 1. Load threshold and constants
    min_bet_edge = get_float_env("MIN_BET_EDGE", "1.8") / 100.0
    move_threshold = 1.5
    
    summary = {
        "bets_checked": 0,
        "significant_moves": 0,
        "abandonments": 0,
        "errors": []
    }
    
    try:
        # 2. Fetch current consensus lines
        try:
            current_games = client.get_todays_games()
            # Map by external_id for fast lookup
            current_odds_map = {g["game_id"]: g for g in current_games if g.get("game_id")}
        except Exception as e:
            logger.error(f"Failed to fetch current odds: {e}")
            return {"status": "error", "error": f"Odds fetch failed: {e}"}
            
        # 3. Get pending bets
        pending_bets = (
            db.query(BetLog)
            .filter(BetLog.outcome.is_(None))
            .all()
        )
        
        summary["bets_checked"] = len(pending_bets)
        
        for bet in pending_bets:
            try:
                game = bet.game
                if not game or not game.external_id:
                    continue
                    
                # NEW: Skip games that have already started
                if game.game_date and game.game_date < datetime.utcnow():
                    logger.debug(f"Skipping started game: {game.away_team} @ {game.home_team}")
                    continue
                    
                curr_odds = current_odds_map.get(game.external_id)
                if not curr_odds:
                    continue
                    
                # Get current consensus spread (home side)
                curr_home_spread = curr_odds.get("sharp_consensus_spread")
                if curr_home_spread is None:
                    # Fallback to best available if sharp consensus missing
                    curr_home_spread = curr_odds.get("best_spread")
                    
                if curr_home_spread is None:
                    continue
                    
                # 4. Parse original spread from pick
                team_name, original_spread_team = parse_pick(bet.pick)
                if original_spread_team is None:
                    continue # Skip moneyline for now as per mission focus on "points"
                    
                # Normalize current consensus to team perspective
                team_is_home = team_name.lower() == game.home_team.lower()
                curr_spread_team = curr_home_spread if team_is_home else -curr_home_spread
                
                delta = curr_spread_team - original_spread_team
                
                # 5. Check threshold
                if abs(delta) >= move_threshold:
                    summary["significant_moves"] += 1
                    game_key = f"{game.away_team}@{game.home_team}"
                    logger.warning(
                        f"SIGNIFICANT_MOVE detected for {game_key}: "
                        f"{bet.pick} -> current consensus {team_name} {curr_spread_team:+.1f} "
                        f"(delta={delta:+.1f} pts)"
                    )
                    
                    # 6. Re-analyze
                    if not bet.prediction_id:
                        logger.warning(f"No prediction_id for bet {bet.id}, skipping re-analysis")
                        continue
                        
                    pred = db.query(Prediction).filter(Prediction.id == bet.prediction_id).first()
                    if not pred or not pred.full_analysis:
                        continue
                        
                    # Reconstruct context from Prediction
                    inputs = pred.full_analysis.get("inputs", {})
                    
                    # CBBEdgeModel needs weights from env or DB
                    # ReanalysisEngine will use current model's weights
                    ctx = CachedGameContext(
                        game_data={
                            "home_team": game.home_team,
                            "away_team": game.away_team,
                            "is_neutral": game.is_neutral
                        },
                        ratings=inputs.get("ratings", {}),
                        base_odds=inputs.get("odds", {}),
                        injuries=inputs.get("injuries"),
                        home_style=inputs.get("home_style"),
                        away_style=inputs.get("away_style"),
                        sharp_books_available=inputs.get("odds", {}).get("sharp_books_available", 0),
                        original_verdict=pred.verdict
                    )
                    
                    engine = ReanalysisEngine(model, ctx)
                    updated = engine.reanalyze(new_spread=curr_home_spread)
                    
                    abandoned = updated.edge_conservative < min_bet_edge
                    if abandoned:
                        summary["abandonments"] += 1
                        logger.error(
                            f"LINE_MOVEMENT_ABANDON [{game_key}]: Fresh edge {updated.edge_conservative:.1%} "
                            f"< MIN_BET_EDGE {min_bet_edge:.1%}"
                        )
                        
                    # 7. Notify Discord
                    game_time_str = game.game_date.strftime("%b %d, %I:%M %p UTC") if game.game_date else None
                    send_line_movement_alert(
                        game_key=game_key,
                        away_team=game.away_team,
                        home_team=game.home_team,
                        old_spread=original_spread_team,
                        new_spread=curr_spread_team,
                        delta=delta,
                        new_edge=updated.edge_conservative,
                        abandoned=abandoned,
                        game_time=game_time_str,
                        min_bet_edge=min_bet_edge
                    )
                    
            except Exception as e:
                err_msg = f"Error checking bet {bet.id}: {e}"
                logger.error(err_msg)
                summary["errors"].append(err_msg)
                
        db.commit() # In case any notes were added (none yet, but good practice)
        
    except Exception as e:
        logger.error(f"Fatal error in check_line_movements: {e}")
        summary["errors"].append(f"Fatal: {e}")
    finally:
        db.close()
        
    return summary
