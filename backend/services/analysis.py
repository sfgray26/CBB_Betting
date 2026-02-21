"""
Nightly analysis orchestration — Version 8.

Workflow:
    1. Fetch current odds from The Odds API (sharp + retail lines)
    2. Fetch ratings from KenPom / BartTorvik / EvanMiya
    3. For each game:
         - compute dynamic SD from game total: SD ≈ sqrt(Total) × sd_multiplier
         - run model with dynamic SD override
         - persist Game + Prediction records
    4. Auto-create a paper-trade BetLog for every "Bet" verdict
    5. Single bulk commit at the end (savepoints guard per-game errors)
    6. Log DataFetch records for monitoring

Quantitative improvements
--------------------------
  Dynamic SD:   Each game's base_sd is computed from its sharp-consensus
                total (or best_total) as sqrt(total) × sd_multiplier instead
                of the hardcoded 11.0.  Passed as base_sd_override to
                model.analyze_game().

  Calibrated:   home_advantage and sd_multiplier are read from the
                model_parameters DB table on each run (via load_current_params).
                If no DB override exists, env-var values are used.

  Bulk insert:  Predictions and BetLogs are flushed per-prediction (to
                satisfy the FK chain) but only committed once at the end
                of the loop.  SQLAlchemy savepoints (begin_nested) isolate
                per-game errors so a single bad game never aborts the batch.

  Daily cap:    MAX_DAILY_EXPOSURE_PCT limits total paper-trade dollars per day
                to prevent independent Kelly bets from stacking on busy slates.
"""

import logging
import math
import os
from collections import defaultdict
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models import SessionLocal, Game, Prediction, BetLog, DataFetch
from backend.betting_model import CBBEdgeModel, GameAnalysis
from backend.services.odds import fetch_current_odds
from backend.services.ratings import fetch_current_ratings, get_ratings_service
from backend.services.recalibration import load_current_params
from backend.services.injuries import get_injury_service
from backend.services.matchup_engine import get_profile_cache, get_matchup_engine

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


def _apply_simultaneous_kelly(
    pending_bets: List[Dict],
    max_total_exposure_pct: float = 15.0,
) -> List[Dict]:
    """
    Apply simultaneous Kelly covariance penalty to a slate of recommended bets.

    Standard Kelly assumes sequential resolution — bet, resolve, resize.
    On a college basketball Saturday 10+ games may resolve concurrently,
    so independent Kelly sizing over-allocates total bankroll exposure.

    **Algorithm:**

    1. *Conference correlation penalty*: Bets sharing a conference have
       weakly correlated outcomes (shared officiating crews, travel
       patterns, and conference strength-of-schedule effects).  For each
       additional same-conference bet beyond the first, multiply the
       individual Kelly size by ``(1 - 0.05 * n_prior)`` (floor 0.70).

    2. *Favourite-fade clustering penalty*: Multiple bets fading heavy
       favourites (spread > +5) are correlated through public money
       patterns.  Apply 0.90x per additional fade-favourite bet.

    3. *Greedy exposure allocation*: After penalties, bets are sorted by
       conservative edge (descending) and greedily allocated until the
       cumulative total reaches ``max_total_exposure_pct``.  Any bet that
       would push the total over the cap is reduced to the remaining
       capacity; subsequent bets are zeroed out.  This concentrates
       capital on the highest-edge opportunities instead of diluting
       across the full slate via proportional scaling.

    Args:
        pending_bets: List of dicts with keys:
            - 'index': int (position in the game loop)
            - 'conference': Optional[str]
            - 'spread': float (home spread; positive = home is underdog)
            - 'bet_side': str ("home" or "away")
            - 'recommended_units': float
            - 'kelly_fractional': float
            - 'edge_conservative': float (lower CI edge — sort key)
        max_total_exposure_pct: Portfolio-level cap on total units deployed.

    Returns:
        The same list with 'recommended_units' adjusted in-place and
        'slate_adjustment_reason' added.  Bets that were dropped will
        have ``recommended_units = 0``.
    """
    if not pending_bets:
        return pending_bets

    # --- Step 1: Conference correlation penalty ---
    conf_counts: Dict[str, int] = defaultdict(int)
    for bet in pending_bets:
        conf = (bet.get("conference") or "").strip().lower()
        if conf:
            conf_counts[conf] += 1

    conf_seen: Dict[str, int] = defaultdict(int)
    for bet in pending_bets:
        conf = (bet.get("conference") or "").strip().lower()
        reasons = []
        if conf and conf_counts[conf] > 1:
            n_prior = conf_seen[conf]
            if n_prior > 0:
                penalty = max(0.70, 1.0 - 0.05 * n_prior)
                bet["recommended_units"] *= penalty
                reasons.append(f"conf_corr={penalty:.2f} ({conf}, #{n_prior+1})")
            conf_seen[conf] += 1
        else:
            if conf:
                conf_seen[conf] += 1

        # --- Step 2: Favourite-fade clustering ---
        spread = bet.get("spread", 0) or 0
        side = bet.get("bet_side", "home")
        is_fade_fav = (
            (side == "away" and spread < -5.0)
            or (side == "home" and spread > 5.0)
        )
        bet["_is_fade_fav"] = is_fade_fav

        bet["_reasons"] = reasons

    # Count total fade-favourite bets
    n_fades = sum(1 for b in pending_bets if b.get("_is_fade_fav"))
    if n_fades > 1:
        fade_seen = 0
        for bet in pending_bets:
            if bet.get("_is_fade_fav"):
                if fade_seen > 0:
                    fade_penalty = max(0.80, 0.90 ** fade_seen)
                    bet["recommended_units"] *= fade_penalty
                    bet["_reasons"].append(
                        f"fade_fav={fade_penalty:.2f} (#{fade_seen+1}/{n_fades})"
                    )
                fade_seen += 1

    # --- Step 3: Greedy exposure allocation (sorted by edge) ---
    total_units = sum(b["recommended_units"] for b in pending_bets)
    if total_units > max_total_exposure_pct:
        # Sort by conservative edge (highest first) to prioritize best bets
        sorted_bets = sorted(
            pending_bets,
            key=lambda b: b.get("edge_conservative", 0),
            reverse=True,
        )
        cumulative = 0.0
        allocated_indices = set()
        for bet in sorted_bets:
            units = bet["recommended_units"]
            remaining = max_total_exposure_pct - cumulative
            if remaining <= 0:
                # Cap reached — drop this bet entirely
                bet["recommended_units"] = 0.0
                bet["_reasons"].append(
                    f"greedy_drop (cap={max_total_exposure_pct:.1f}% reached)"
                )
            elif units > remaining:
                # Partial fill — reduce to remaining capacity
                bet["_reasons"].append(
                    f"greedy_partial={remaining:.2f}/{units:.2f} "
                    f"(cap={max_total_exposure_pct:.1f}%)"
                )
                bet["recommended_units"] = remaining
                cumulative += remaining
                allocated_indices.add(id(bet))
            else:
                # Full allocation
                cumulative += units
                allocated_indices.add(id(bet))

        logger.info(
            "Greedy Kelly: %d/%d bets allocated (%.1f%% → %.1f%% cap)",
            len(allocated_indices), len(pending_bets),
            total_units, max_total_exposure_pct,
        )

    # Finalise reasons and clean up temp keys
    for bet in pending_bets:
        bet["slate_adjustment_reason"] = "; ".join(bet.pop("_reasons", []))
        bet.pop("_is_fade_fav", None)

    return pending_bets


def _create_paper_bet(db: Session, game: Game, prediction: Prediction, daily_exposure: float = 0.0) -> BetLog:
    """
    Auto-create a paper-trade BetLog from a Bet verdict.

    Called immediately after the Prediction row is flushed so prediction.id
    is available as a foreign key.

    daily_exposure: dollars already committed today (for daily cap enforcement).
    """
    starting_bankroll = float(os.getenv("STARTING_BANKROLL", "1000"))
    recommended_units = prediction.recommended_units or 0.0
    bet_dollars = recommended_units * (starting_bankroll / 100.0)

    # --- Portfolio daily cap with EV displacement logic ----------------------
    # Scale down if total daily exposure would exceed MAX_DAILY_EXPOSURE_PCT.
    # This guards against a busy slate of games stacking independent Kelly bets
    # into a combined exposure far larger than intended.
    #
    # **EV Displacement**: If the daily cap is full but a new high-EV bet
    # arrives (e.g., late-breaking line movement), allow it to displace
    # an inferior pending bet instead of dropping it entirely.
    max_daily_pct = float(os.getenv("MAX_DAILY_EXPOSURE_PCT", "5.0"))
    max_daily_dollars = starting_bankroll * max_daily_pct / 100.0
    remaining_capacity = max(0.0, max_daily_dollars - daily_exposure)

    ev_displacement_applied = False
    if bet_dollars > remaining_capacity and remaining_capacity < bet_dollars * 0.5:
        # Capacity is tight — check for EV displacement opportunity
        new_edge = prediction.edge_conservative or 0.0

        # Query pending paper trades from earlier today
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        pending_bets = db.query(BetLog).filter(
            BetLog.timestamp >= today_start,
            BetLog.outcome.is_(None),
            BetLog.is_paper_trade.is_(True),
            BetLog.conservative_edge.isnot(None),
        ).order_by(BetLog.conservative_edge.asc()).all()

        if pending_bets:
            lowest_ev_bet = pending_bets[0]
            lowest_edge = lowest_ev_bet.conservative_edge or 0.0

            # Displacement threshold: new bet must have >= 1.5% higher edge
            if new_edge >= (lowest_edge + 0.015):
                ev_displacement_applied = True
                # Flag the inferior bet
                old_notes = lowest_ev_bet.notes or ""
                lowest_ev_bet.notes = (
                    f"{old_notes} | WARNING: EV Displacement - Hedge Recommended "
                    f"(displaced by {new_edge:.3f} > {lowest_edge:.3f})"
                )
                logger.warning(
                    "EV Displacement: New bet edge=%.3f displaces %s (edge=%.3f). "
                    "Temporary overcap allowed.",
                    new_edge, lowest_ev_bet.pick, lowest_edge,
                )
                # Allow the new bet at full size (temporary overcap)
                # bet_dollars unchanged

    if not ev_displacement_applied and bet_dollars > remaining_capacity:
        # Standard cap enforcement — scale down
        original = bet_dollars
        bet_dollars = remaining_capacity
        recommended_units = (bet_dollars / starting_bankroll) * 100.0
        logger.info(
            "Daily cap: scaled %.2f → %.2f (capacity=%.2f, max=%.2f)",
            original, bet_dollars, remaining_capacity, max_daily_dollars,
        )
    # -------------------------------------------------------------------------

    spread_value: Optional[float] = None
    calcs = {}
    if prediction.full_analysis:
        odds_block = prediction.full_analysis.get("inputs", {}).get("odds", {})
        calcs = prediction.full_analysis.get("calculations", {})
        spread_value = odds_block.get("spread")

    # Use bet_side from model analysis (determined by edge direction, not margin sign)
    bet_side = calcs.get("bet_side", "home")
    bet_odds = calcs.get("bet_odds", -110)

    if bet_side == "home":
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
        odds_taken=bet_odds,
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

def run_nightly_analysis(run_tier: str = "nightly") -> Dict:
    """
    Main analysis job (called by APScheduler and /admin/run-analysis).

    Args:
        run_tier: Distinguishes analysis runs on the same day.
                  "opener" for early-day runs, "nightly" for the 3 AM job,
                  "closing" for near-tipoff re-analysis.  Each tier can
                  create its own Prediction row for the same game + date.

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
    logger.info("Starting analysis (v8, tier=%s)", run_tier)
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

        # ---- Injury + profile + matchup services -------------------------
        injury_service = get_injury_service()
        profile_cache = get_profile_cache()
        matchup_engine = get_matchup_engine()
        if not profile_cache.has_profiles():
            try:
                loaded = profile_cache.load_from_barttorvik()
                logger.info("Loaded %d team profiles from BartTorvik", loaded)
            except Exception as exc:
                logger.warning("Could not load team profiles: %s", exc)

        # ----------------------------------------------------------------
        # STEP 4: Analyse each game and persist predictions
        #
        # Strategy: SQLAlchemy savepoints (begin_nested) isolate per-game
        # errors.  Bet candidates are collected first; simultaneous Kelly
        # adjustments are applied AFTER the full slate is known.
        # ----------------------------------------------------------------
        logger.info("Analysing %d games...", len(odds_games))

        # Collect bet candidates for simultaneous Kelly post-processing
        bet_candidates: List[Dict] = []

        for game_data in odds_games:
            home_team = game_data.get("home_team", "Unknown")
            away_team = game_data.get("away_team", "Unknown")

            try:
                with db.begin_nested():  # Savepoint — rolls back only this game on error
                    # ---- Dynamic SD from game total ----------------------
                    game_total = (
                        game_data.get("sharp_consensus_total")
                        or game_data.get("best_total")
                    )
                    if game_total and game_total > 0:
                        dynamic_base_sd = math.sqrt(game_total) * sd_multiplier
                    else:
                        dynamic_base_sd = None

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
                        "spread_away_odds": game_data.get("best_spread_away_odds", -110),
                        "total": game_total,
                        "sharp_consensus_spread": game_data.get("sharp_consensus_spread"),
                        "sharp_consensus_total": game_data.get("sharp_consensus_total"),
                        "sharp_books_available": game_data.get("sharp_books_available", 0),
                    }

                    game_input = {
                        "home_team": home_team,
                        "away_team": away_team,
                        "is_neutral": game_data.get("is_neutral", False),
                    }

                    # ---- Injuries + team profiles --------------------------
                    try:
                        game_injuries = injury_service.get_game_injuries(home_team, away_team)
                    except Exception as exc:
                        logger.warning("Injury lookup failed for %s @ %s: %s", away_team, home_team, exc)
                        game_injuries = None

                    home_profile = profile_cache.get(home_team)
                    away_profile = profile_cache.get(away_team)
                    home_style = (
                        {"pace": home_profile.pace, "three_par": home_profile.three_par,
                         "ft_rate": home_profile.ft_rate, "to_pct": home_profile.to_pct}
                        if home_profile else None
                    )
                    away_style = (
                        {"pace": away_profile.pace, "three_par": away_profile.three_par,
                         "ft_rate": away_profile.ft_rate, "to_pct": away_profile.to_pct}
                        if away_profile else None
                    )

                    # ---- Matchup engine analysis -------------------------
                    matchup_margin_adj = 0.0
                    if home_profile and away_profile:
                        try:
                            matchup_adj = matchup_engine.analyze_matchup(
                                home_profile, away_profile,
                            )
                            matchup_margin_adj = matchup_adj.margin_adj
                            if matchup_adj.notes:
                                logger.debug(
                                    "Matchup %s @ %s: adj=%.2f, sd_adj=%.2f",
                                    away_team, home_team,
                                    matchup_adj.margin_adj, matchup_adj.sd_adj,
                                )
                        except Exception as exc:
                            logger.warning(
                                "Matchup analysis failed for %s @ %s: %s",
                                away_team, home_team, exc,
                            )

                    # ---- Run model with dynamic SD -----------------------
                    analysis = model.analyze_game(
                        game_data=game_input,
                        odds=odds_input,
                        ratings=ratings_input,
                        injuries=game_injuries,
                        data_freshness=odds_freshness,
                        base_sd_override=dynamic_base_sd,
                        home_style=home_style,
                        away_style=away_style,
                        matchup_margin_adj=matchup_margin_adj,
                    )

                    # ---- Persist Game (idempotent) -----------------------
                    game = get_or_create_game(db, game_data)

                    # Check for existing prediction with same run_tier today.
                    # If found, update in-place when the analysis has materially
                    # changed (verdict flip or edge shift > 1%).  This prevents
                    # the opener_attack job from locking out later nightly runs.
                    today = datetime.utcnow().date()
                    existing_prediction = db.query(Prediction).filter(
                        Prediction.game_id == game.id,
                        Prediction.prediction_date == today,
                        Prediction.run_tier == run_tier,
                    ).first()

                    if existing_prediction:
                        old_verdict = existing_prediction.verdict or ""
                        new_verdict = analysis.verdict
                        edge_diff = abs(
                            (existing_prediction.edge_conservative or 0)
                            - analysis.edge_conservative
                        )
                        verdict_changed = (
                            old_verdict.startswith("Bet") != new_verdict.startswith("Bet")
                        )
                        material_change = verdict_changed or edge_diff > 0.01

                        if not material_change:
                            logger.debug(
                                "No material change for %s @ %s (tier=%s), skipping update.",
                                away_team, home_team, run_tier,
                            )
                            games_analyzed += 1
                            continue

                        # Material change — update existing prediction in-place
                        logger.info(
                            "Updating prediction for %s @ %s (tier=%s): "
                            "%s → %s (Δedge=%.3f)",
                            away_team, home_team, run_tier,
                            old_verdict[:20], new_verdict[:20], edge_diff,
                        )
                        prediction = existing_prediction
                        prediction.model_version = "v8.0"
                    else:
                        # No existing prediction for this tier — create new
                        prediction = Prediction(
                            game_id=game.id,
                            prediction_date=today,
                            run_tier=run_tier,
                            model_version="v8.0",
                        )
                        db.add(prediction)
                        old_verdict = None

                    # ---- Populate Prediction fields ----------------------
                    prediction.kenpom_home = ratings_input["kenpom"]["home"]
                    prediction.kenpom_away = ratings_input["kenpom"]["away"]
                    prediction.barttorvik_home = ratings_input["barttorvik"]["home"]
                    prediction.barttorvik_away = ratings_input["barttorvik"]["away"]
                    prediction.evanmiya_home = ratings_input["evanmiya"]["home"]
                    prediction.evanmiya_away = ratings_input["evanmiya"]["away"]
                    prediction.projected_margin = analysis.projected_margin
                    prediction.adjusted_sd = analysis.adjusted_sd
                    prediction.point_prob = analysis.point_prob
                    prediction.lower_ci_prob = analysis.lower_ci_prob
                    prediction.upper_ci_prob = analysis.upper_ci_prob
                    prediction.edge_point = analysis.edge_point
                    prediction.edge_conservative = analysis.edge_conservative
                    prediction.kelly_full = analysis.kelly_full
                    prediction.kelly_fractional = analysis.kelly_fractional
                    prediction.recommended_units = analysis.recommended_units
                    prediction.verdict = analysis.verdict
                    prediction.pass_reason = analysis.pass_reason
                    prediction.full_analysis = analysis.full_analysis
                    prediction.data_freshness_tier = analysis.data_freshness_tier
                    prediction.penalties_applied = analysis.penalties_applied
                    db.flush()

                    games_analyzed += 1

                    if analysis.verdict.startswith("Bet"):
                        bets_recommended += 1
                        # Collect for simultaneous Kelly — don't create paper trades yet
                        calcs = analysis.full_analysis.get("calculations", {})
                        bet_candidates.append({
                            "index": len(bet_candidates),
                            "game": game,
                            "prediction": prediction,
                            "analysis": analysis,
                            "old_verdict": old_verdict,
                            "conference": game_data.get("conference"),
                            "spread": odds_input.get("spread", 0) or 0,
                            "bet_side": calcs.get("bet_side", "home"),
                            "recommended_units": analysis.recommended_units,
                            "kelly_fractional": analysis.kelly_fractional,
                            "edge_conservative": analysis.edge_conservative,
                            "home_team": home_team,
                            "away_team": away_team,
                        })
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
                continue

        # ----------------------------------------------------------------
        # STEP 4b: Simultaneous Kelly adjustment across the full slate
        # ----------------------------------------------------------------
        if bet_candidates:
            max_exposure = float(os.getenv("MAX_DAILY_EXPOSURE_PCT", "5.0"))
            _apply_simultaneous_kelly(bet_candidates, max_total_exposure_pct=max_exposure)

            current_exposure = _daily_exposure(db)
            for candidate in bet_candidates:
                prediction = candidate["prediction"]
                game = candidate["game"]
                adjusted_units = candidate["recommended_units"]
                slate_reason = candidate.get("slate_adjustment_reason", "")
                prev_verdict = candidate.get("old_verdict")

                # Update prediction with slate-adjusted units
                prediction.recommended_units = adjusted_units
                if slate_reason:
                    prediction.notes = f"Slate adj: {slate_reason}"

                # BetLog guard: only create paper trades when flipping
                # from PASS (or first analysis) to Bet.  If old_verdict
                # was already a Bet, we updated the Prediction in-place
                # but do NOT duplicate the BetLog.
                if prev_verdict is not None and prev_verdict.startswith("Bet"):
                    logger.debug(
                        "Skipping duplicate paper trade for %s @ %s "
                        "(old verdict already Bet)",
                        candidate["away_team"], candidate["home_team"],
                    )
                    continue

                paper_bet = _create_paper_bet(
                    db, game, prediction, daily_exposure=current_exposure,
                )
                current_exposure += paper_bet.bet_size_dollars or 0
                paper_trades_created += 1
                logger.info(
                    "BET: %s @ %s — %.2fu%s (paper: %s)",
                    candidate["away_team"],
                    candidate["home_team"],
                    adjusted_units,
                    f" [{slate_reason}]" if slate_reason else "",
                    paper_bet.pick,
                )

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
