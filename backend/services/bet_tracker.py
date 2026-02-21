"""
Automated bet lifecycle management.

Scheduled jobs:
  update_completed_games()  - every 2 hours: fetch scores, settle pending bets
  capture_closing_lines()   - every 30 min: store closing odds, update CLV
"""

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests

from backend.models import BetLog, ClosingLine, DataFetch, Game, Prediction, SessionLocal
from backend.services.clv import calculate_clv_full

logger = logging.getLogger(__name__)

API_KEY = os.getenv("THE_ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4"


# ---------------------------------------------------------------------------
# Pick parsing and outcome calculation (pure functions — no DB)
# ---------------------------------------------------------------------------

def parse_pick(pick: str) -> Tuple[str, Optional[float]]:
    """
    Parse 'Duke -4.5' → ('Duke', -4.5).
    Parse 'Kansas +3'  → ('Kansas', 3.0).
    Parse 'Duke'       → ('Duke', None)   (moneyline).

    The spread is from the perspective of the picked team:
      negative → favourite, positive → underdog.
    """
    match = re.match(r"^(.+?)\s+([+-]?\d+\.?\d*)$", pick.strip())
    if match:
        team = match.group(1).strip()
        spread = float(match.group(2))
        return team, spread
    return pick.strip(), None


@dataclass
class OutcomeResult:
    outcome: int            # 1 = win, 0 = loss, -1 = push
    profit_loss_dollars: float
    profit_loss_units: float


def calculate_bet_outcome(
    bet: "BetLog",
    game: "Game",
    starting_bankroll: float = 1000.0,
) -> Optional[OutcomeResult]:
    """
    Determine outcome and P&L for a settled game.

    Cover condition (spread bet):
        team_actual_margin + spread_taken > 0  →  covers
        = 0                                     →  push
        < 0                                     →  loses

    Returns None if scores or pick cannot be resolved.
    """
    if game.home_score is None or game.away_score is None:
        return None

    home_score: int = game.home_score
    away_score: int = game.away_score
    team, spread = parse_pick(bet.pick)

    # Determine whether the picked team is home or away (case-insensitive)
    team_is_home = team.lower() == game.home_team.lower()

    if spread is None:
        # Moneyline: picked team must win outright
        won = (home_score > away_score) if team_is_home else (away_score > home_score)
    else:
        # Spread: actual margin from picked team's perspective + spread taken
        if team_is_home:
            margin = home_score - away_score
        else:
            margin = away_score - home_score

        cover_margin = margin + spread

        if abs(cover_margin) < 0.01:
            # Push — return stake, zero P&L
            unit_value = (bet.bankroll_at_bet or starting_bankroll) / 100.0
            return OutcomeResult(outcome=-1, profit_loss_dollars=0.0, profit_loss_units=0.0)

        won = cover_margin > 0

    # --- P&L ---
    bet_dollars = bet.bet_size_dollars or 0.0
    odds = bet.odds_taken

    if won:
        profit = (
            bet_dollars * (odds / 100.0)
            if odds > 0
            else bet_dollars * (100.0 / abs(odds))
        )
        outcome = 1
    else:
        profit = -bet_dollars
        outcome = 0

    unit_value = (bet.bankroll_at_bet or starting_bankroll) / 100.0
    profit_units = round(profit / unit_value, 4) if unit_value > 0 else 0.0

    return OutcomeResult(
        outcome=outcome,
        profit_loss_dollars=round(profit, 2),
        profit_loss_units=profit_units,
    )


# ---------------------------------------------------------------------------
# The Odds API — scores
# ---------------------------------------------------------------------------

def _fetch_scores(days_from: int = 2) -> List[Dict]:
    """Return completed NCAAB games from the last `days_from` days."""
    if not API_KEY:
        raise ValueError("THE_ODDS_API_KEY not set")

    url = f"{BASE_URL}/sports/basketball_ncaab/scores"
    params = {"apiKey": API_KEY, "daysFrom": days_from}

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    completed = [g for g in data if g.get("completed")]
    logger.info(
        "Scores API: %d total games, %d completed (daysFrom=%d)",
        len(data), len(completed), days_from,
    )
    return completed


def _parse_scores(score_data: Dict) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract (home_score, away_score) from a scores API entry.

    The API returns:
        {"home_team": "Duke", "scores": [{"name": "Duke", "score": "83"}, ...]}
    """
    home_name = score_data.get("home_team")
    home_score: Optional[int] = None
    away_score: Optional[int] = None

    for entry in score_data.get("scores") or []:
        try:
            val = int(entry.get("score", 0))
        except (TypeError, ValueError):
            continue
        if entry.get("name") == home_name:
            home_score = val
        else:
            away_score = val

    return home_score, away_score


# ---------------------------------------------------------------------------
# Job 1: update_completed_games
# ---------------------------------------------------------------------------

def update_completed_games() -> Dict:
    """
    Fetch completed game scores from The Odds API and settle all
    pending BetLog entries whose game is now complete.

    Called by scheduler every 2 hours.
    """
    logger.info("Starting update_completed_games")
    db = SessionLocal()

    games_updated = 0
    bets_settled = 0
    pushes = 0
    errors: List[str] = []
    starting_bankroll = float(os.getenv("STARTING_BANKROLL", "1000"))

    try:
        try:
            completed = _fetch_scores(days_from=2)
            db.add(DataFetch(
                data_source="odds_api_scores",
                success=True,
                records_fetched=len(completed),
            ))
            db.commit()
        except Exception as exc:
            db.add(DataFetch(
                data_source="odds_api_scores",
                success=False,
                error_message=str(exc)[:500],
                records_fetched=0,
            ))
            db.commit()
            return _job_summary(0, 0, 0, [f"Scores API failed: {exc}"])

        for score_data in completed:
            external_id = score_data.get("id")
            if not external_id:
                continue

            game = db.query(Game).filter(Game.external_id == external_id).first()
            if not game:
                continue  # Not a game we track

            home_s, away_s = _parse_scores(score_data)
            if home_s is None or away_s is None:
                continue

            # Update game record (idempotent)
            if not game.completed or game.home_score != home_s or game.away_score != away_s:
                game.home_score = home_s
                game.away_score = away_s
                game.completed = True
                db.flush()
                games_updated += 1
                logger.info(
                    "Score updated: game %d | %s %d – %s %d",
                    game.id, game.home_team, home_s, game.away_team, away_s,
                )

                # Backfill actual_margin on every Prediction linked to this game.
                # This enables margin-prediction MAE and calibration tracking
                # for ALL model predictions, not just ones where a bet was placed.
                actual_margin = home_s - away_s  # positive = home team won
                unresolved_preds = (
                    db.query(Prediction)
                    .filter(
                        Prediction.game_id == game.id,
                        Prediction.actual_margin.is_(None),
                    )
                    .all()
                )
                for pred in unresolved_preds:
                    pred.actual_margin = actual_margin
                if unresolved_preds:
                    logger.info(
                        "Backfilled actual_margin=%.1f on %d prediction(s) for game %d",
                        actual_margin, len(unresolved_preds), game.id,
                    )

            # Settle pending bets
            pending = (
                db.query(BetLog)
                .filter(BetLog.game_id == game.id, BetLog.outcome.is_(None))
                .all()
            )
            for bet in pending:
                try:
                    result = calculate_bet_outcome(bet, game, starting_bankroll)
                    if result is None:
                        errors.append(f"Bet {bet.id} ({bet.pick}): could not determine outcome")
                        continue

                    bet.outcome = result.outcome
                    bet.profit_loss_dollars = result.profit_loss_dollars
                    bet.profit_loss_units = result.profit_loss_units

                    if result.outcome == -1:
                        pushes += 1
                        logger.info("PUSH: bet %d (%s)", bet.id, bet.pick)
                    else:
                        bets_settled += 1
                        logger.info(
                            "%s: bet %d (%s) | P&L $%.2f",
                            "WIN" if result.outcome == 1 else "LOSS",
                            bet.id, bet.pick, result.profit_loss_dollars,
                        )
                except Exception as exc:
                    errors.append(f"Bet {bet.id}: {exc}")
                    logger.error("Error settling bet %d: %s", bet.id, exc)

            db.commit()

    except Exception as exc:
        logger.error("Fatal error in update_completed_games: %s", exc, exc_info=True)
        db.rollback()
        errors.append(f"Fatal: {exc}")
    finally:
        db.close()

    summary = _job_summary(games_updated, bets_settled, pushes, errors)
    logger.info("update_completed_games done: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Job 2: capture_closing_lines
# ---------------------------------------------------------------------------

def capture_closing_lines() -> Dict:
    """
    For games starting within the next 90 minutes, capture current lines
    as closing lines and update CLV on any pending bets.

    Called by scheduler every 30 minutes.
    """
    logger.info("Starting capture_closing_lines")
    db = SessionLocal()

    lines_captured = 0
    clv_updated = 0
    errors: List[str] = []

    try:
        if not API_KEY:
            return _job_summary(0, 0, 0, ["THE_ODDS_API_KEY not set"])

        now = datetime.utcnow()
        window_end = now + timedelta(minutes=90)

        upcoming = (
            db.query(Game)
            .filter(
                Game.game_date >= now,
                Game.game_date <= window_end,
                Game.completed == False,
            )
            .all()
        )

        if not upcoming:
            logger.info("No games in closing-line window")
            return _job_summary(0, 0, 0, [])

        ext_id_to_game = {g.external_id: g for g in upcoming if g.external_id}

        # Fetch live odds
        url = f"{BASE_URL}/sports/basketball_ncaab/odds"
        params = {
            "apiKey": API_KEY,
            "regions": "us",
            "markets": "spreads,totals,h2h",
            "oddsFormat": "american",
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        raw_odds = resp.json()

        base_sd = float(os.getenv("BASE_SD", "11.0"))

        for odds_data in raw_odds:
            ext_id = odds_data.get("id")
            if ext_id not in ext_id_to_game:
                continue

            game = ext_id_to_game[ext_id]

            # Extract best lines
            best_spread = None
            best_spread_odds = None
            best_total = None
            best_ml_home = None
            best_ml_away = None

            for bookmaker in odds_data.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    key = market.get("key")
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name")
                        point = outcome.get("point")
                        price = outcome.get("price")

                        if key == "spreads" and name == game.home_team and best_spread is None:
                            best_spread = point
                            best_spread_odds = price
                        elif key == "totals" and point and best_total is None:
                            best_total = point
                        elif key == "h2h" and price:
                            if name == game.home_team:
                                if best_ml_home is None or price > best_ml_home:
                                    best_ml_home = price
                            else:
                                if best_ml_away is None or price > best_ml_away:
                                    best_ml_away = price

            # Persist the closing line snapshot
            cl = ClosingLine(
                game_id=game.id,
                captured_at=now,
                spread=best_spread,
                spread_odds=best_spread_odds,
                total=best_total,
                moneyline_home=best_ml_home,
                moneyline_away=best_ml_away,
            )
            db.add(cl)
            lines_captured += 1

            # Update CLV on pending bets for this game
            if best_spread is not None and best_spread_odds is not None:
                pending = (
                    db.query(BetLog)
                    .filter(BetLog.game_id == game.id, BetLog.outcome.is_(None))
                    .all()
                )
                for bet in pending:
                    try:
                        team_name, opening_spread_team = parse_pick(bet.pick)

                        # Normalize opening spread to home perspective.
                        # Closing spread from the API is already home-side;
                        # opening spread from parse_pick is team-side.
                        team_is_home = team_name.lower() == game.home_team.lower()
                        if opening_spread_team is not None:
                            opening_spread_home = (
                                opening_spread_team if team_is_home
                                else -opening_spread_team
                            )
                        else:
                            opening_spread_home = None

                        clv = calculate_clv_full(
                            opening_odds=bet.odds_taken,
                            closing_odds=best_spread_odds,
                            opening_spread=opening_spread_home,
                            closing_spread=best_spread,
                            base_sd=base_sd,
                        )
                        bet.closing_line = best_spread_odds
                        bet.clv_points = round(clv.clv_points, 3)
                        bet.clv_prob = round(clv.clv_prob, 4)
                        clv_updated += 1
                    except Exception as exc:
                        errors.append(f"CLV bet {bet.id}: {exc}")

            db.commit()

    except Exception as exc:
        logger.error("Fatal error in capture_closing_lines: %s", exc, exc_info=True)
        db.rollback()
        errors.append(f"Fatal: {exc}")
    finally:
        db.close()

    summary = _job_summary(lines_captured, clv_updated, 0, errors)
    logger.info("capture_closing_lines done: %s", summary)
    return summary


def _job_summary(updated: int, settled: int, pushes: int, errors: List[str]) -> Dict:
    return {
        "games_updated": updated,
        "bets_settled": settled,
        "pushes": pushes,
        "errors": errors,
        "timestamp": datetime.utcnow().isoformat(),
    }
