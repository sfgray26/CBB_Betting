"""
Nightly analysis orchestration.

Workflow:
    1. Fetch current odds from The Odds API (sharp + retail lines)
    2. Fetch ratings from KenPom / BartTorvik / EvanMiya
    3. For each game:
         - compute dynamic SD from game total: SD ≈ sqrt(Total) × 0.85
         - run model with dynamic SD override
         - persist Game + Prediction records
    4. Auto-create a paper-trade BetLog for every "Bet" verdict
    5. Single bulk commit at the end (savepoints guard per-game errors)
    6. Log DataFetch records for monitoring

Quantitative improvements
--------------------------
  Dynamic SD:   Each game's base_sd is computed from its sharp-consensus
                total (or best_total) as sqrt(total) × 0.85 instead of
                the hardcoded 11.0.  This is passed as base_sd_override to
                model.analyze_game().

  Bulk insert:  Predictions and BetLogs are flushed per-prediction (to
                satisfy the FK chain) but only committed once at the end
                of the loop.  SQLAlchemy savepoints (begin_nested) isolate
                per-game errors so a single bad game never aborts the batch.
"""

import logging
import math
import os
from datetime import datetime, date
from typing import Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models import SessionLocal, Game, Prediction, BetLog, DataFetch
from backend.betting_model import CBBEdgeModel
from backend.services.odds import fetch_current_odds
from backend.services.ratings import fetch_current_ratings, get_ratings_service
from backend.services.recalibration import load_current_params

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_or_create_game(db: Session, game_data: Dict) -> Game:
    """Return existing Game row or create a new one (idempotent)."""
    external_id = game_data.get("game_id")

    game = db.query(Game).filter(Game.external_id == external_id).first()
    if game:
        return game

    commence_time = game_data.get("commence_time")
    if isinstance(commence_time, str):
        game_date = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
    else:
        game_date = datetime.utcnow()

    game = Game(
        external_id=external_id,
        game_date=game_date,
        home_team=game_data.get("home_team"),
        away_team=game_data.get("away_team"),
        is_neutral=game_data.get("is_neutral", False),
    )
    db.add(game)
    db.flush()  # Populate game.id without committing the outer transaction
    return game


def _daily_exposure(db: Session) -> float:
    """
    Total dollars committed in pending paper trades placed today.

    Used to enforce the portfolio daily cap so independent Kelly bets
    don't silently stack into runaway bankroll exposure on busy days.
    """
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    result = (
        db.query(func.sum(BetLog.bet_size_dollars))
        .filter(
            BetLog.timestamp >= today_start,
            BetLog.outcome.is_(None),           # still pending
            BetLog.is_paper_trade.is_(True),
        )
        .scalar()
    )
    return float(result or 0.0)


def _create_paper_bet(db: Session, game: Game, prediction: Prediction, daily_exposure: float = 0.0) -> BetLog:
    """
    Auto-create a paper-trade BetLog from a Bet verdict.

    Called immediately after the Prediction row is flushed so prediction.id
    is available as a foreign key.
    """
    starting_bankroll = float(os.getenv("STARTING_BANKROLL", "1000"))
    recommended_units = prediction.recommended_units or 0.0
    bet_dollars = recommended_units * (starting_bankroll / 100.0)

    # --- Portfolio daily cap -------------------------------------------------
    # Scale down if total daily exposure would exceed MAX_DAILY_EXPOSURE_PCT.
    # This guards against a busy slate of games stacking independent Kelly bets
    # into a combined exposure far larger than intended.
    max_daily_pct = float(os.getenv("MAX_DAILY_EXPOSURE_PCT", "5.0"))
    max_daily_dollars = starting_bankroll * max_daily_pct / 100.0
    remaining_capacity = max(0.0, max_daily_dollars - daily_exposure)

    if bet_dollars > remaining_capacity:
        original = bet_dollars
        bet_dollars = remaining_capacity
        recommended_units = (bet_dollars / starting_bankroll) * 100.0
        logger.info(
            "Daily cap: scaled %.2f → %.2f (capacity=%.2f, max=%.2f)",
            original, bet_dollars, remaining_capacity, max_daily_dollars,
        )
    # -------------------------------------------------------------------------

    spread_odds: float = -110
    spread_value: Optional[float] = None
    if prediction.full_analysis:
        odds_block = prediction.full_analysis.get("inputs", {}).get("odds", {})
        spread_odds = odds_block.get("spread_odds", -110) or -110
        spread_value = odds_block.get("spread")

    if prediction.projected_margin is not None and prediction.projected_margin > 0:
        if spread_value is not None:
            pick = f"{game.home_team} {spread_value:+.1f}"
        else:
            pick = game.home_team
    else:
        if spread_value is not None:
            pick = f"{game.away_team} {-spread_value:+.1f}"
        else:
            pick = game.away_team

    bet = BetLog(
        game_id=game.id,
        prediction_id=prediction.id,
        pick=pick,
        bet_type="spread",
        odds_taken=spread_odds,
        bankroll_at_bet=starting_bankroll,
        kelly_full=prediction.kelly_full,
        kelly_fractional=prediction.kelly_fractional,
        bet_size_pct=recommended_units,
        bet_size_units=recommended_units,
        bet_size_dollars=bet_dollars,
        model_prob=prediction.point_prob,
        lower_ci_prob=prediction.lower_ci_prob,
        point_edge=prediction.edge_point,
        conservative_edge=prediction.edge_conservative,
        is_paper_trade=True,
        executed=False,
        notes=f"Auto paper trade — nightly analysis {prediction.model_version}",
    )
    db.add(bet)
    return bet


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_nightly_analysis() -> Dict:
    """
    Main nightly analysis job (called by APScheduler and /admin/run-analysis).

    Returns a summary dict:
        {
            'games_analyzed': int,
            'bets_recommended': int,
            'paper_trades_created': int,
            'errors': List[str],
            'timestamp': str,
            'duration_seconds': float,
        }
    """
    logger.info("Starting nightly analysis")
    start_time = datetime.utcnow()

    db = SessionLocal()
    errors: List[str] = []
    games_analyzed = 0
    bets_recommended = 0
    paper_trades_created = 0

    try:
        # ----------------------------------------------------------------
        # STEP 1: Fetch odds (sharp + retail)
        # ----------------------------------------------------------------
        logger.info("Fetching odds from The Odds API...")
        try:
            odds_games, odds_freshness = fetch_current_odds()
            db.add(DataFetch(
                data_source="the_odds_api",
                success=True,
                records_fetched=len(odds_games),
            ))
            db.commit()
            logger.info("Fetched %d games from The Odds API", len(odds_games))
        except Exception as exc:
            logger.error("Odds API failed: %s", exc, exc_info=True)
            db.add(DataFetch(
                data_source="the_odds_api",
                success=False,
                error_message=str(exc)[:500],
                records_fetched=0,
            ))
            db.commit()
            return _summary(start_time, 0, 0, 0, [f"Odds API failed: {exc}"])

        if not odds_games:
            logger.warning("No games returned from Odds API")
            return _summary(start_time, 0, 0, 0, [], status="no_games")

        # ----------------------------------------------------------------
        # STEP 2: Fetch ratings
        # ----------------------------------------------------------------
        logger.info("Fetching ratings from all sources...")
        try:
            all_ratings = fetch_current_ratings()
            ratings_service = get_ratings_service()

            for source in ("kenpom", "barttorvik", "evanmiya"):
                count = len(all_ratings.get(source, {}))
                db.add(DataFetch(
                    data_source=source,
                    success=count > 0,
                    records_fetched=count,
                    error_message=None if count > 0 else f"Zero records for {source}",
                ))
            db.commit()

            logger.info(
                "Ratings loaded — KenPom: %d, BartTorvik: %d, EvanMiya: %d",
                len(all_ratings.get("kenpom", {})),
                len(all_ratings.get("barttorvik", {})),
                len(all_ratings.get("evanmiya", {})),
            )
        except Exception as exc:
            logger.error("Ratings fetch failed: %s", exc, exc_info=True)
            db.add(DataFetch(
                data_source="ratings_aggregate",
                success=False,
                error_message=str(exc)[:500],
            ))
            db.commit()
            errors.append(f"Ratings error: {exc}")
            return _summary(start_time, 0, 0, 0, errors)

        # ----------------------------------------------------------------
        # STEP 3: Initialise model — prefer calibrated params from DB,
        #         fall back to env vars
        # ----------------------------------------------------------------
        calibrated = load_current_params(db)
        sd_multiplier = calibrated.get(
            "sd_multiplier", float(os.getenv("SD_MULTIPLIER", "0.85"))
        )

        model = CBBEdgeModel(
            base_sd=float(os.getenv("BASE_SD", "11.0")),
            weights={
                "kenpom":    float(os.getenv("WEIGHT_KENPOM",    "0.342")),
                "barttorvik": float(os.getenv("WEIGHT_BARTTORVIK", "0.333")),
                "evanmiya":  float(os.getenv("WEIGHT_EVANMIYA",  "0.325")),
            },
            home_advantage=calibrated.get(
                "home_advantage",
                float(os.getenv("HOME_ADVANTAGE", "3.09")),
            ),
            max_kelly=float(os.getenv("MAX_KELLY_FRACTION",             "0.20")),
            fractional_kelly_divisor=float(os.getenv("FRACTIONAL_KELLY_DIVISOR", "2.0")),
        )
        logger.info(
            "Model initialised — home_adv=%.3f, sd_multiplier=%.4f",
            model.home_advantage, sd_multiplier,
        )

        # ----------------------------------------------------------------
        # STEP 4: Analyse each game and persist results
        #
        # Strategy: SQLAlchemy savepoints (begin_nested) isolate per-game
        # errors.  All successful games are committed in a single call at
        # the end, reducing N database round-trips to 1.
        # ----------------------------------------------------------------
        logger.info("Analysing %d games...", len(odds_games))

        for game_data in odds_games:
            home_team = game_data.get("home_team", "Unknown")
            away_team = game_data.get("away_team", "Unknown")

            try:
                with db.begin_nested():  # Savepoint — rolls back only this game on error
                    # ---- Dynamic SD from game total ----------------------
                    # Prefer sharp consensus total; fall back to retail best.
                    game_total = (
                        game_data.get("sharp_consensus_total")
                        or game_data.get("best_total")
                    )
                    if game_total and game_total > 0:
                        # Use calibrated sd_multiplier (default 0.85)
                        dynamic_base_sd = math.sqrt(game_total) * sd_multiplier
                    else:
                        dynamic_base_sd = None  # model uses its own base_sd

                    # ---- Build model inputs ------------------------------
                    ratings_input = {
                        "kenpom": {
                            "home": ratings_service.get_team_rating(
                                home_team, all_ratings.get("kenpom", {})
                            ),
                            "away": ratings_service.get_team_rating(
                                away_team, all_ratings.get("kenpom", {})
                            ),
                        },
                        "barttorvik": {
                            "home": ratings_service.get_team_rating(
                                home_team, all_ratings.get("barttorvik", {})
                            ),
                            "away": ratings_service.get_team_rating(
                                away_team, all_ratings.get("barttorvik", {})
                            ),
                        },
                        "evanmiya": {
                            "home": ratings_service.get_team_rating(
                                home_team, all_ratings.get("evanmiya", {})
                            ),
                            "away": ratings_service.get_team_rating(
                                away_team, all_ratings.get("evanmiya", {})
                            ),
                        },
                    }

                    odds_input = {
                        "spread": game_data.get("best_spread"),
                        "spread_odds": game_data.get("best_spread_odds", -110),
                        "total": game_total,
                        # Sharp consensus for CLV benchmarking
                        "sharp_consensus_spread": game_data.get("sharp_consensus_spread"),
                        "sharp_consensus_total": game_data.get("sharp_consensus_total"),
                        "sharp_books_available": game_data.get("sharp_books_available", 0),
                    }

                    game_input = {
                        "home_team": home_team,
                        "away_team": away_team,
                        "is_neutral": game_data.get("is_neutral", False),
                    }

                    # ---- Run model with dynamic SD -----------------------
                    analysis = model.analyze_game(
                        game_data=game_input,
                        odds=odds_input,
                        ratings=ratings_input,
                        injuries=None,
                        data_freshness=odds_freshness,
                        base_sd_override=dynamic_base_sd,
                    )

                    # ---- Persist Game (idempotent) -----------------------
                    game = get_or_create_game(db, game_data)

                    # Skip if already analysed today
                    today = datetime.utcnow().date()
                    existing_prediction = db.query(Prediction).filter(
                        Prediction.game_id == game.id,
                        Prediction.prediction_date == today,
                    ).first()

                    if existing_prediction:
                        logger.info(
                            "Skipping %s @ %s: already analyzed today.",
                            away_team, home_team,
                        )
                        continue

                    # ---- Persist Prediction ------------------------------
                    prediction = Prediction(
                        game_id=game.id,
                        prediction_date=today,
                        model_version="v7.0",
                        kenpom_home=ratings_input["kenpom"]["home"],
                        kenpom_away=ratings_input["kenpom"]["away"],
                        barttorvik_home=ratings_input["barttorvik"]["home"],
                        barttorvik_away=ratings_input["barttorvik"]["away"],
                        evanmiya_home=ratings_input["evanmiya"]["home"],
                        evanmiya_away=ratings_input["evanmiya"]["away"],
                        projected_margin=analysis.projected_margin,
                        adjusted_sd=analysis.adjusted_sd,
                        point_prob=analysis.point_prob,
                        lower_ci_prob=analysis.lower_ci_prob,
                        upper_ci_prob=analysis.upper_ci_prob,
                        edge_point=analysis.edge_point,
                        edge_conservative=analysis.edge_conservative,
                        kelly_full=analysis.kelly_full,
                        kelly_fractional=analysis.kelly_fractional,
                        recommended_units=analysis.recommended_units,
                        verdict=analysis.verdict,
                        pass_reason=analysis.pass_reason,
                        full_analysis=analysis.full_analysis,
                        data_freshness_tier=analysis.data_freshness_tier,
                        penalties_applied=analysis.penalties_applied,
                    )
                    db.add(prediction)
                    db.flush()  # Populate prediction.id for BetLog FK

                    games_analyzed += 1

                    if analysis.verdict.startswith("Bet"):
                        bets_recommended += 1
                        current_exposure = _daily_exposure(db)
                        paper_bet = _create_paper_bet(
                            db, game, prediction, daily_exposure=current_exposure
                        )
                        paper_trades_created += 1
                        logger.info(
                            "BET: %s @ %s — %s (paper: %s)",
                            away_team, home_team, analysis.verdict, paper_bet.pick,
                        )
                    else:
                        logger.debug(
                            "PASS: %s @ %s (%s)",
                            away_team, home_team, analysis.pass_reason,
                        )

            except Exception as exc:
                logger.error(
                    "Error analysing %s @ %s: %s",
                    away_team, home_team, exc, exc_info=True,
                )
                errors.append(f"{away_team} @ {home_team}: {str(exc)[:100]}")
                # Savepoint was automatically rolled back; outer transaction intact
                continue

        # ----------------------------------------------------------------
        # STEP 5: Single bulk commit for all successful games
        # ----------------------------------------------------------------
        db.commit()
        logger.info(
            "Bulk commit: %d games, %d bets, %d paper trades",
            games_analyzed, bets_recommended, paper_trades_created,
        )

        return _summary(start_time, games_analyzed, bets_recommended, paper_trades_created, errors)

    except Exception as exc:
        logger.error("Fatal error in nightly analysis: %s", exc, exc_info=True)
        try:
            db.rollback()
        except Exception:
            pass
        return _summary(
            start_time,
            games_analyzed,
            bets_recommended,
            paper_trades_created,
            errors + [f"Fatal: {exc}"],
        )
    finally:
        db.close()


def _summary(
    start_time: datetime,
    games_analyzed: int,
    bets_recommended: int,
    paper_trades_created: int,
    errors: List[str],
    status: str = "ok",
) -> Dict:
    duration = (datetime.utcnow() - start_time).total_seconds()
    result = {
        "status": status,
        "games_analyzed": games_analyzed,
        "bets_recommended": bets_recommended,
        "paper_trades_created": paper_trades_created,
        "errors": errors,
        "timestamp": datetime.utcnow().isoformat(),
        "duration_seconds": round(duration, 2),
    }
    logger.info(
        "Analysis complete in %.1fs — %d games, %d bets, %d paper trades, %d errors",
        duration,
        games_analyzed,
        bets_recommended,
        paper_trades_created,
        len(errors),
    )
    return result
