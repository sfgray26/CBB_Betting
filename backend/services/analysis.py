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
from datetime import datetime, date, timezone
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
from backend.services.matchup_engine import get_profile_cache, get_matchup_engine, TeamPlayStyle
from backend.services.parlay_engine import build_optimal_parlays, format_parlay_ticket
from backend.services.team_mapping import normalize_team_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# D1 average four-factor baselines used when profile cache misses.
_D1_AVG_ADJ_O   = 105.0   # Adjusted offensive efficiency (pts / 100 poss)
_D1_AVG_ADJ_DE  = 105.0   # D1 average adjusted defensive efficiency (pts / 100 poss)
_D1_AVG_EFG     = 0.505   # Effective FG%
_D1_AVG_TO_PCT  = 0.175   # Turnover rate
_D1_AVG_ORB     = 0.280   # Offensive rebound rate
_D1_AVG_FT_RATE = 0.320   # Free-throw attempt rate (FTA/FGA)
_D1_AVG_PACE    = 68.0    # Possessions per 40 min
_D1_AVG_THREE   = 0.360   # 3PA / FGA


def _heuristic_style_from_rating(
    off_rating_raw: Optional[float],
    def_rating_raw: Optional[float] = None,
) -> Optional[Dict]:
    """
    Estimate a Markov-engine style dict from top-line efficiency numbers.

    Handles two different scales in the wild:
      - BartTorvik AdjOE: absolute offensive efficiency (~90–130, D1 avg ≈ 105)
      - KenPom / EvanMiya AdjEM: margin relative to average (~-30 to +40)

    When only a margin is available, it is converted to an estimated AdjO by
    assuming offense contributes ~55% and defense ~45% of the margin:
        AdjO ≈ D1_AVG_ADJ_O + margin × 0.55
        AdjDE ≈ D1_AVG_ADJ_DE − margin × 0.45

    Passing ``def_rating_raw`` enables two improvements over pure D1 defaults:
      1. Defensive eFG% allowed: ``0.505 × (AdjDE / 105.0)``
         (lower AdjDE = better defence = lower eFG% allowed)
      2. Forced turnover rate: ``0.175 × (105.0 / AdjDE)``
         (lower AdjDE = better defence = more TOs forced)
      3. ``is_heuristic`` is set to ``False``, allowing the Markov engine to
         use these synthetic profiles rather than the Gaussian fallback.

    Convention for ``def_rating_raw`` when on the margin scale:
      - Pass the **negated** overall AdjEM so that a positive-AdjEM team gets a
        lower estimated AdjDE (better defence):
            AdjDE_est ≈ 105 + (−AdjEM) × 0.45 = 105 − AdjEM × 0.45
      - Pass the raw AdjDE directly when on the absolute (~90-130) scale.

    Args:
        off_rating_raw: Offensive efficiency or margin from any ratings source.
        def_rating_raw: Negated AdjEM margin, or raw AdjDE.  When not supplied,
                        defensive four-factor baselines default to D1 averages
                        and ``is_heuristic`` remains ``True``.

    Returns:
        Style dict compatible with ``betting_model.analyze_game()`` or ``None``
        if ``off_rating_raw`` is ``None`` (no data available at all).
    """
    if off_rating_raw is None:
        return None

    # Detect scale: absolute efficiencies are > 50; margins are typically ≤ 45
    if abs(off_rating_raw) < 50:
        # Margin scale (KenPom AdjEM, EvanMiya BPR) → convert to estimated AdjO
        adj_o = _D1_AVG_ADJ_O + off_rating_raw * 0.55
    else:
        adj_o = off_rating_raw  # Already on absolute-efficiency scale

    # Scale eFG% from offensive efficiency; clamp to realistic CBB range
    estimated_efg = _D1_AVG_EFG * (adj_o / _D1_AVG_ADJ_O)
    estimated_efg = max(0.350, min(0.650, estimated_efg))

    # Offensive turnover rate: elite offenses protect the ball better
    # Higher adj_o → lower TO rate; inverse relationship with efficiency
    estimated_to = _D1_AVG_TO_PCT * (_D1_AVG_ADJ_O / adj_o)
    estimated_to = max(0.10, min(0.30, estimated_to))

    # Defensive four-factor estimates.
    # When def_rating_raw is provided we can compute AdjDE-scaled values:
    #   def_efg_pct = 0.505 × (AdjDE / 105.0)   — lower AdjDE → lower eFG% allowed
    #   def_to_pct  = 0.175 × (105.0 / AdjDE)   — lower AdjDE → higher TOs forced
    # Both are clamped to physically plausible CBB ranges.
    has_def_data = def_rating_raw is not None
    if has_def_data:
        if abs(def_rating_raw) < 50:  # type: ignore[arg-type]
            # Margin scale (negated AdjEM is passed by callers):
            #   adj_d = 105 + (−AdjEM) × 0.45 = 105 − AdjEM × 0.45
            adj_d = _D1_AVG_ADJ_DE + def_rating_raw * 0.45  # type: ignore[operator]
        else:
            adj_d = def_rating_raw  # Already on absolute AdjDE scale (~90-130)
        # Guard against degenerate adj_d values (e.g. ±∞ from bad inputs)
        adj_d = max(75.0, min(130.0, adj_d))

        def_efg = _D1_AVG_EFG * (adj_d / _D1_AVG_ADJ_DE)
        def_efg = max(0.390, min(0.620, def_efg))

        def_to = _D1_AVG_TO_PCT * (_D1_AVG_ADJ_DE / adj_d)
        def_to = max(0.120, min(0.260, def_to))
    else:
        # No defensive data — fall back to D1 averages and keep the heuristic
        # flag so the Markov guard can choose to use the Gaussian path.
        def_efg = _D1_AVG_EFG
        def_to  = _D1_AVG_TO_PCT

    return {
        "pace":        _D1_AVG_PACE,           # Tempo unknown without granular data
        "efg_pct":     round(estimated_efg, 4),
        "to_pct":      round(estimated_to, 4), # Dynamic: elite offenses protect ball
        "ft_rate":     _D1_AVG_FT_RATE,
        "three_par":   _D1_AVG_THREE,
        "def_efg_pct": round(def_efg, 4),
        "def_to_pct":  round(def_to, 4),
        # is_heuristic=False unlocks the Markov engine; True keeps Gaussian fallback.
        "is_heuristic": False,
    }


def _build_profile_from_style(style: Dict, team_name: str) -> TeamPlayStyle:
    """
    Construct a minimal TeamPlayStyle from a four-factor style dict when the
    BartTorvik profile cache returns None for a team.

    Populates only the fields available from KenPom/BartTorvik four-factor
    data.  PBP-derived fields (drop_coverage_pct, zone_pct, transition_freq,
    rim_rate, etc.) are left at TeamPlayStyle defaults — the factors that
    depend on them (_three_point_vs_drop, _zone_vs_three, _transition_gap)
    require non-default thresholds to fire (drop_coverage_pct > 0.30, etc.),
    so they remain dormant.  The four-factor-driven factors (_efg_pressure_gap,
    _turnover_pressure_gap) WILL fire when real eFG%/TO% data is present.
    """
    return TeamPlayStyle(
        team=team_name,
        pace=style.get("pace", 68.0),
        efg_pct=style.get("efg_pct"),          # None when heuristic path
        to_pct=style.get("to_pct", 0.175),
        ft_rate=style.get("ft_rate", 0.280),
        three_par=style.get("three_par", 0.360),
        def_efg_pct=style.get("def_efg_pct", 0.505),
        def_to_pct=style.get("def_to_pct", 0.175),
        def_ft_rate=style.get("def_ft_rate", 0.280),
        def_three_par=style.get("def_three_par", 0.360),
        # PBP-derived fields intentionally at defaults — no data available:
        # drop_coverage_pct=0.0, zone_pct=0.0, transition_freq=0.15, etc.
    )


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


def _create_paper_bet(
    db: Session,
    game: Game,
    prediction: Prediction,
    daily_exposure: float = 0.0,
    scaled_bet_dollars: Optional[float] = None
) -> tuple[BetLog, float]:
    """
    Auto-create a paper-trade BetLog from a Bet verdict.

    Called immediately after the Prediction row is flushed so prediction.id
    is available as a foreign key.

    Args:
        db: Database session
        game: Game object
        prediction: Prediction object
        daily_exposure: dollars already committed today (for daily cap enforcement)
        scaled_bet_dollars: If provided, use this exact dollar amount (Global Scaling).
                           If None, calculate from prediction.recommended_units.

    Returns:
        (BetLog, net_exposure_change) where net_exposure_change is the
        actual change in deployed capital, accounting for any EV displacement.
        If displacement occurred, net_change = new_bet_size - displaced_capital.
    """
    starting_bankroll = float(os.getenv("STARTING_BANKROLL", "1000"))

    if scaled_bet_dollars is not None:
        bet_dollars = scaled_bet_dollars
        # Sync recommended_units to match the scaled dollar amount
        recommended_units = (bet_dollars / starting_bankroll) * 100.0
    else:
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
    max_daily_pct = float(os.getenv("MAX_DAILY_EXPOSURE_PCT", "20.0"))
    max_daily_dollars = starting_bankroll * max_daily_pct / 100.0
    remaining_capacity = max(0.0, max_daily_dollars - daily_exposure)

    ev_displacement_applied = False
    displaced_capital = 0.0  # Track freed capital from displacement

    # Only attempt displacement if we haven't already provided a scaled amount
    # OR if the scaled amount still exceeds remaining capacity.
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

            # Volatility-scaled displacement threshold
            game_sd = prediction.adjusted_sd or 11.0
            threshold_bump = 0.001 * game_sd
            threshold_bump = max(0.005, min(threshold_bump, 0.025))

            # Displacement condition: new edge must exceed lowest edge + threshold
            if new_edge >= (lowest_edge + threshold_bump):
                ev_displacement_applied = True
                displaced_capital = lowest_ev_bet.bet_size_dollars or 0.0

                # Cancel the displaced bet
                lowest_ev_bet.outcome = -1  # -1 = Cancelled/Push
                lowest_ev_bet.profit_loss_dollars = 0.0
                lowest_ev_bet.profit_loss_units = 0.0

                # Update notes to reflect cancellation
                old_notes = lowest_ev_bet.notes or ""
                lowest_ev_bet.notes = (
                    f"{old_notes} | CANCELLED via EV Displacement "
                    f"(displaced by new bet: edge={new_edge:.3f} > {lowest_edge:.3f})"
                )

                # Free the cancelled bet's capital so the new bet can use it
                remaining_capacity += displaced_capital

                logger.warning(
                    "EV Displacement: New bet edge=%.3f displaces %s (edge=%.3f). "
                    "Cancelled bet freed %.2f capacity.",
                    new_edge, lowest_ev_bet.pick, lowest_edge, displaced_capital,
                )

    # NOTE: In Global Scaling mode (scaled_bet_dollars is not None), we skip
    # the sequential 'remaining_capacity' scale-down because the caller has
    # already ensured the total fits the cap.
    if scaled_bet_dollars is None and not ev_displacement_applied and bet_dollars > remaining_capacity:
        # Standard sequential cap enforcement — scale down
        original = bet_dollars
        bet_dollars = remaining_capacity
        recommended_units = (bet_dollars / starting_bankroll) * 100.0
        logger.info(
            "Daily cap (sequential): scaled %.2f -> %.2f (capacity=%.2f, max=%.2f)",
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

    # Return net exposure change: new bet size minus any displaced capital.
    # If no displacement occurred, this equals bet_dollars.
    # If displacement occurred, this equals the incremental increase in exposure.
    net_exposure_change = bet_dollars - displaced_capital
    return bet, net_exposure_change


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
    games_considered = 0   # CONSIDER-verdict games (edge positive but below MIN_BET_EDGE)
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

        # Persist profiles to the DB so CLV / performance pages have real data.
        # Always refresh on each nightly run (not guarded by has_profiles) so
        # that a stale in-memory cache from a previous run doesn't prevent a
        # DB update when BartTorvik publishes fresher data.
        try:
            saved = ratings_service.save_team_profiles(db)
            logger.info("Persisted %d TeamProfile rows to DB", saved)
        except Exception as exc:
            logger.warning("save_team_profiles failed (non-fatal): %s", exc)

        # ----------------------------------------------------------------
        # STEP 4: Analyse each game and persist predictions
        #
        # Strategy: SQLAlchemy savepoints (begin_nested) isolate per-game
        # errors.  Bet candidates are collected first; simultaneous Kelly
        # adjustments are applied AFTER the full slate is known.
        # ----------------------------------------------------------------
        logger.info("Analysing %d games...", len(odds_games))

        # Pre-compute valid team names from the profile cache once so that
        # normalize_team_name can fuzzy-match raw Odds API names against
        # the BartTorvik vocabulary on every game without re-building the list.
        profile_valid_teams: List[str] = profile_cache.teams()

        # Collect bet candidates for simultaneous Kelly post-processing
        bet_candidates: List[Dict] = []

        # Running exposure tracker for Pass 2.  Updated each time a game
        # produces a BET verdict so that analyze_game() sees a realistic
        # concurrent_exposure for every subsequent game in the slate.
        _pass2_concurrent_exposure: float = 0.0

        # ================================================================
        # TWO-PASS SLATE: PASS 1 — pre-score by raw edge
        # ================================================================
        # Goal: sort the slate so the main Pass-2 loop processes the
        # highest-EV games first.  The portfolio Kelly divisor in
        # analyze_game() grows with concurrent_exposure, so if low-edge
        # games are evaluated first they consume exposure budget before the
        # high-EV opportunities are reached — causing greedy_drop verdicts
        # on the best bets.
        #
        # Pass 1 calls analyze_game(concurrent_exposure=0.0) for every
        # game using only ratings + odds (no injuries / profiles / DB ops).
        # The resulting edge_conservative is used purely as a sort key;
        # final Kelly sizing with proper concurrent_exposure happens in
        # Pass 2.
        # ================================================================
        logger.info(
            "Two-pass pre-sort: Pass 1 scoring %d games for slate ordering...",
            len(odds_games),
        )
        _prescore_edges: Dict[str, float] = {}
        for _gd in odds_games:
            _ht = _gd.get("home_team", "")
            _at = _gd.get("away_team", "")
            _game_key = f"{_at}@{_ht}"
            try:
                _gt = _gd.get("sharp_consensus_total") or _gd.get("best_total")
                _dyn_sd = math.sqrt(float(_gt)) * sd_multiplier if _gt else None
                _ri: Dict = {
                    "kenpom": {
                        "home": ratings_service.get_team_rating(
                            _ht, all_ratings.get("kenpom", {})
                        ),
                        "away": ratings_service.get_team_rating(
                            _at, all_ratings.get("kenpom", {})
                        ),
                    },
                    "barttorvik": {
                        "home": ratings_service.get_team_rating(
                            _ht, all_ratings.get("barttorvik", {})
                        ),
                        "away": ratings_service.get_team_rating(
                            _at, all_ratings.get("barttorvik", {})
                        ),
                    },
                    "evanmiya": {
                        "home": ratings_service.get_team_rating(
                            _ht, all_ratings.get("evanmiya", {})
                        ),
                        "away": ratings_service.get_team_rating(
                            _at, all_ratings.get("evanmiya", {})
                        ),
                    },
                    "_meta": all_ratings.get("_meta", {}),
                }
                _oi: Dict = {
                    "spread":                  _gd.get("best_spread"),
                    "spread_odds":             _gd.get("best_spread_odds", -110),
                    "spread_away_odds":        _gd.get("best_spread_away_odds", -110),
                    "total":                   _gt,
                    "sharp_consensus_spread":  _gd.get("sharp_consensus_spread"),
                    "sharp_consensus_total":   _gd.get("sharp_consensus_total"),
                    "sharp_books_available":   _gd.get("sharp_books_available", 0),
                }
                _gi: Dict = {
                    "home_team":  _ht,
                    "away_team":  _at,
                    "is_neutral": _gd.get("is_neutral", False),
                }
                _ps = model.analyze_game(
                    game_data=_gi,
                    odds=_oi,
                    ratings=_ri,
                    base_sd_override=_dyn_sd,
                    concurrent_exposure=0.0,
                )
                _prescore_edges[_game_key] = _ps.edge_conservative
            except Exception as _exc:
                logger.debug(
                    "Pass-1 pre-score skipped for %s: %s", _game_key, _exc
                )
                _prescore_edges[_game_key] = -999.0  # sink to end of sorted slate

        # Sort slate descending by raw edge so the main loop processes
        # the highest-EV games first and the portfolio receives capital
        # in priority order.
        odds_games = sorted(
            odds_games,
            key=lambda g: _prescore_edges.get(
                f"{g.get('away_team', '')}@{g.get('home_team', '')}",
                -999.0,
            ),
            reverse=True,
        )
        if _prescore_edges:
            _n_pos_edge = sum(1 for e in _prescore_edges.values() if e > 0.0)
            _top_edge   = max(_prescore_edges.values(), default=0.0)
            logger.info(
                "Two-pass pre-sort complete: %d games sorted by edge "
                "(%d with positive raw edge, top=%.2f%%)",
                len(odds_games), _n_pos_edge, _top_edge * 100,
            )

        # ================================================================
        # TWO-PASS SLATE: PASS 2 — main loop (edge-sorted order)
        # ================================================================
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
                    _meta = all_ratings.get("_meta", {})
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
                        # Propagate auto-drop flag so the model can suppress
                        # the missing-evanmiya SD penalty when it was
                        # dropped due to Cloudflare blocking, not bad data.
                        "_meta": _meta,
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
                    # Normalize raw Odds API names to the profile-cache vocabulary
                    # (BartTorvik names) before querying.  Without this step,
                    # profile_cache.get() always returns None for Odds API teams
                    # because the key space differs (e.g. "Saint Mary's" vs
                    # "Saint Mary's (CA)").  Falls back to the raw name if no
                    # match is found so existing code paths are unaffected.
                    norm_home = (
                        normalize_team_name(home_team, profile_valid_teams) or home_team
                    )
                    norm_away = (
                        normalize_team_name(away_team, profile_valid_teams) or away_team
                    )
                    if norm_home != home_team or norm_away != away_team:
                        logger.debug(
                            "Team normalization: '%s'→'%s', '%s'→'%s'",
                            home_team, norm_home, away_team, norm_away,
                        )

                    try:
                        game_injuries = injury_service.get_game_injuries(norm_home, norm_away)
                    except Exception as exc:
                        logger.warning("Injury lookup failed for %s @ %s: %s", away_team, home_team, exc)
                        game_injuries = None

                    home_profile = profile_cache.get(norm_home)
                    away_profile = profile_cache.get(norm_away)

                    # Build style dicts from the BartTorvik Four-Factors cache
                    # (most accurate path).  If the cache misses — because the
                    # BartTorvik scrape failed, the team name didn't normalize,
                    # or the season data isn't loaded yet — fall back to a
                    # heuristic estimate from top-line efficiency ratings so the
                    # Markov engine can ALWAYS run when we have any rating data.
                    #
                    # Architecture: generate heuristic baseline first (gives a
                    # physically valid efg_pct from AdjEM since TeamPlayStyle
                    # has no direct efg_pct field), then overlay the real
                    # BartTorvik four-factor data where the profile cache has
                    # an entry.  Validity clamps in save_team_profiles() and
                    # Markov safety guards in possession_sim.py ensure the
                    # overlaid values are in D1-plausible ranges.
                    _home_em = (
                        ratings_input.get("barttorvik", {}).get("home")
                        or ratings_input.get("kenpom", {}).get("home")
                    )
                    _away_em = (
                        ratings_input.get("barttorvik", {}).get("away")
                        or ratings_input.get("kenpom", {}).get("away")
                    )

                    home_style = _heuristic_style_from_rating(
                        off_rating_raw=_home_em,
                        def_rating_raw=-_home_em if _home_em is not None else None,
                    )
                    away_style = _heuristic_style_from_rating(
                        off_rating_raw=_away_em,
                        def_rating_raw=-_away_em if _away_em is not None else None,
                    )

                    # Overlay real BartTorvik stats from profile cache.
                    # All values have been range-validated inside load_from_barttorvik()
                    # (same clamps as save_team_profiles), so they are D1-plausible
                    # before reaching the Markov engine.
                    # efg_pct: use real BartTorvik value when the scraper found it
                    # (profile.efg_pct is not None); otherwise keep the heuristic.
                    #
                    # four_factors_heuristic: propagated from profile_cache so the
                    # adaptive circuit breaker in betting_model.py can widen its
                    # threshold when pricing is based on AdjOE/AdjDE derivatives
                    # rather than scraped four-factor columns.
                    _ff_heuristic = profile_cache.four_factors_heuristic
                    if home_profile and home_style:
                        home_style["pace"]                   = home_profile.pace
                        home_style["ft_rate"]                = home_profile.ft_rate
                        home_style["three_par"]              = home_profile.three_par
                        home_style["to_pct"]                 = home_profile.to_pct
                        home_style["def_efg_pct"]            = home_profile.def_efg_pct
                        home_style["def_to_pct"]             = home_profile.def_to_pct
                        home_style["four_factors_heuristic"] = _ff_heuristic
                        if home_profile.efg_pct is not None:
                            home_style["efg_pct"] = home_profile.efg_pct

                    if away_profile and away_style:
                        away_style["pace"]                   = away_profile.pace
                        away_style["ft_rate"]                = away_profile.ft_rate
                        away_style["three_par"]              = away_profile.three_par
                        away_style["to_pct"]                 = away_profile.to_pct
                        away_style["def_efg_pct"]            = away_profile.def_efg_pct
                        away_style["def_to_pct"]             = away_profile.def_to_pct
                        away_style["four_factors_heuristic"] = _ff_heuristic
                        if away_profile.efg_pct is not None:
                            away_style["efg_pct"] = away_profile.efg_pct

                    # ---- KenPom four-factor overlay (highest priority) -------
                    # KenPom API data is authoritative for Markov inputs because:
                    #   (a) it is fetched via an authenticated token — not scraped
                    #   (b) field names are stable and documented
                    #   (c) it removes the primary source of phantom edges that
                    #       arose from corrupted BartTorvik CSV columns
                    #
                    # This overlay runs AFTER the BartTorvik profile-cache overlay
                    # so that any KenPom field that is non-None overwrites the
                    # potentially-heuristic BartTorvik value.  When KenPom is
                    # unavailable (API down), kenpom_ff_home / kenpom_ff_away will
                    # be empty dicts and this block is a no-op — BartTorvik / heuristic
                    # data from the block above remains in effect and the degraded-mode
                    # SD penalty in betting_model.py provides the risk buffer.
                    _kp_ff: Dict = all_ratings.get("kenpom_four_factors", {})

                    # Try both the normalized name (BartTorvik vocabulary) and
                    # the raw Odds API name so KenPom's name space is searched.
                    kenpom_ff_home: Dict = (
                        _kp_ff.get(norm_home) or _kp_ff.get(home_team) or {}
                    )
                    kenpom_ff_away: Dict = (
                        _kp_ff.get(norm_away) or _kp_ff.get(away_team) or {}
                    )

                    _KP_FF_FIELDS = (
                        "efg_pct", "to_pct", "ft_rate", "three_par",
                        "def_efg_pct", "def_to_pct", "def_ft_rate", "def_three_par",
                        "pace",
                    )
                    if kenpom_ff_home and home_style:
                        for _f in _KP_FF_FIELDS:
                            _v = kenpom_ff_home.get(_f)
                            if _v is not None:
                                home_style[_f] = _v
                        # KenPom data is verified — clear the heuristic flag so the
                        # degraded-mode SD penalty and circuit-breaker lift are suppressed.
                        home_style["four_factors_heuristic"] = False
                        home_style["kenpom_four_factors"]    = True
                        logger.debug(
                            "KenPom four-factor overlay applied for %s", home_team
                        )

                    if kenpom_ff_away and away_style:
                        for _f in _KP_FF_FIELDS:
                            _v = kenpom_ff_away.get(_f)
                            if _v is not None:
                                away_style[_f] = _v
                        away_style["four_factors_heuristic"] = False
                        away_style["kenpom_four_factors"]    = True
                        logger.debug(
                            "KenPom four-factor overlay applied for %s", away_team
                        )

                    matchup_margin_adj = 0.0

                    # ---- Profile fallback: synthesize TeamPlayStyle from style dicts --
                    # profile_cache.get() returns None when:
                    #   (a) BartTorvik scrape failed / returned empty data
                    #   (b) Team name normalization missed (profile_valid_teams=[])
                    #   (c) Odds API name differs from BartTorvik vocabulary
                    # In all cases, build a minimal TeamPlayStyle from the enriched
                    # home_style / away_style dicts (which have KenPom + BartTorvik
                    # four-factor overlays applied above). PBP-derived fields default
                    # to 0.0 / D1 averages — the four-factor factors (_efg_pressure_gap,
                    # _turnover_pressure_gap) will still fire; PBP factors won't.
                    if home_profile is None and home_style:
                        home_profile = _build_profile_from_style(home_style, home_team)
                        logger.debug(
                            "Profile fallback: synthesized TeamPlayStyle for %s from style dict",
                            home_team,
                        )
                    if away_profile is None and away_style:
                        away_profile = _build_profile_from_style(away_style, away_team)
                        logger.debug(
                            "Profile fallback: synthesized TeamPlayStyle for %s from style dict",
                            away_team,
                        )

                    # ---- Matchup engine ----------------------------------------
                    if home_profile and away_profile:
                        try:
                            matchup_adj = matchup_engine.analyze_matchup(
                                home_profile, away_profile,
                                game_total=game_total,
                                base_sd=dynamic_base_sd,
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

                    # ---- Compute hours to tipoff (used for dynamic Kelly/SD) --
                    hours_to_tipoff: Optional[float] = None
                    _commence_raw = game_data.get("commence_time")
                    if _commence_raw:
                        try:
                            _game_time = datetime.fromisoformat(
                                _commence_raw.replace("Z", "+00:00")
                            )
                            # Guard: fromisoformat can return a naive datetime when
                            # the string has no offset (rare but possible if the Odds
                            # API omits the trailing Z).  Subtraction against the
                            # timezone-aware datetime.now(timezone.utc) would raise
                            # a TypeError, so explicitly localise to UTC first.
                            if _game_time.tzinfo is None:
                                _game_time = _game_time.replace(tzinfo=timezone.utc)
                            _delta = _game_time - datetime.now(timezone.utc)
                            hours_to_tipoff = _delta.total_seconds() / 3600.0
                        except (ValueError, TypeError) as _exc:
                            logger.debug(
                                "Could not parse commence_time '%s': %s",
                                _commence_raw, _exc,
                            )

                    # ---- Live-game guard ------------------------------------
                    # Skip games that have already tipped off.  A negative
                    # hours_to_tipoff means the scheduled start has passed;
                    # -0.1 gives a 6-minute grace window for late tip-offs.
                    if hours_to_tipoff is not None and hours_to_tipoff < -0.1:
                        logger.debug(
                            "Skipping %s @ %s — game already started "
                            "(%.1fh past tipoff).",
                            away_team, home_team, abs(hours_to_tipoff),
                        )
                        continue

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
                        hours_to_tipoff=hours_to_tipoff,
                        # Pass running portfolio exposure so _portfolio_kelly_divisor
                        # scales each game's Kelly fraction relative to capital already
                        # committed to higher-EV games earlier in the sorted slate.
                        concurrent_exposure=_pass2_concurrent_exposure,
                        # Sharp-book count — widens MC CI when Pinnacle/Circa absent.
                        sharp_books_available=odds_input.get("sharp_books_available", 0),
                        # Soft-proxy flag — applies half SE penalty (0.15) instead of
                        # full NO_SHARP_BOOKS_SE_ADDEND (0.30).
                        sharp_proxy_used=bool(game_data.get("sharp_proxy_used", False)),
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
                            "bet_odds": calcs.get("bet_odds"),
                            "recommended_units": analysis.recommended_units,
                            "kelly_fractional": analysis.kelly_fractional,
                            "edge_conservative": analysis.edge_conservative,
                            "home_team": home_team,
                            "away_team": away_team,
                        })
                        # Advance the running exposure so games evaluated later
                        # in this edge-sorted slate see a realistic portfolio load.
                        # recommended_units is in percentage-point units (e.g. 2.5
                        # = 2.5% of bankroll); concurrent_exposure uses 0-1 fractions.
                        if analysis.recommended_units > 0:
                            _max_exp_frac = (
                                float(os.getenv("MAX_DAILY_EXPOSURE_PCT", "20.0")) / 100.0
                            )
                            _pass2_concurrent_exposure = min(
                                _pass2_concurrent_exposure
                                + analysis.recommended_units / 100.0,
                                _max_exp_frac,
                            )
                            logger.debug(
                                "Pass-2 exposure: +%.2f%% → cumulative=%.2f%% "
                                "after %s @ %s",
                                analysis.recommended_units,
                                _pass2_concurrent_exposure * 100,
                                away_team,
                                home_team,
                            )
                    elif analysis.verdict.startswith("CONSIDER"):
                        # CONSIDER: positive edge below MIN_BET_EDGE floor.
                        # Stored in Prediction for CLV tracking; no paper trade created.
                        games_considered += 1
                        logger.info(
                            "CONSIDER [%s @ %s]: %s (no paper trade)",
                            away_team, home_team, analysis.verdict,
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
                continue

        # ----------------------------------------------------------------
        # STEP 4b: Simultaneous Kelly adjustment & Global Proportional Scaling
        # ----------------------------------------------------------------
        if bet_candidates:
            starting_bankroll = float(os.getenv("STARTING_BANKROLL", "1000"))
            max_daily_pct = float(os.getenv("MAX_DAILY_EXPOSURE_PCT", "20.0"))
            max_daily_dollars = starting_bankroll * max_daily_pct / 100.0

            # 1. Apply simultaneous Kelly covariance penalty
            _apply_simultaneous_kelly(bet_candidates, max_total_exposure_pct=max_daily_pct)

            # 2. Calculate Global Scaling Factor
            # We check how many dollars we WANT to bet across all candidates
            # after Kelly adjustments, and compare to REMAINING daily capacity.
            current_exposure_before = _daily_exposure(db)
            remaining_capacity = max(0.0, max_daily_dollars - current_exposure_before)

            total_requested_dollars = sum(
                (c["recommended_units"] * starting_bankroll / 100.0)
                for c in bet_candidates
            )

            scaling_factor = 1.0
            if total_requested_dollars > remaining_capacity and total_requested_dollars > 0:
                scaling_factor = remaining_capacity / total_requested_dollars
                logger.warning(
                    "Global Scaling: Total requested $%.2f exceeds capacity $%.2f. "
                    "Applying scaling factor %.4f to all %d bets.",
                    total_requested_dollars, remaining_capacity, scaling_factor, len(bet_candidates)
                )

            # 3. Create paper trades with scaled amounts
            current_exposure_acc = current_exposure_before
            for candidate in bet_candidates:
                prediction = candidate["prediction"]
                game = candidate["game"]
                orig_units = candidate["recommended_units"]
                
                # Apply global scaling factor
                scaled_units = orig_units * scaling_factor
                scaled_dollars = (scaled_units * starting_bankroll / 100.0)
                
                # Floor check: if scaled below 0.25u, keep at 0.25u UNLESS that would 
                # violate the cap again (highly unlikely with large slates).
                if 0 < scaled_units < 0.25:
                    scaled_units = 0.25
                    scaled_dollars = (scaled_units * starting_bankroll / 100.0)

                candidate["recommended_units"] = scaled_units
                prediction.recommended_units = scaled_units

                slate_reason = candidate.get("slate_adjustment_reason", "")
                if scaling_factor < 1.0:
                    scaling_note = f"Global scale: {scaling_factor:.2f}x"
                    slate_reason = f"{slate_reason} | {scaling_note}" if slate_reason else scaling_note
                
                if slate_reason:
                    prediction.notes = f"Slate adj: {slate_reason}"

                # Sync verdict string
                bet_odds = candidate.get("bet_odds")
                if scaled_units == 0.0 and (prediction.verdict or "").startswith("Bet"):
                    prediction.verdict = "PASS - Slate Cap Reached"
                    prediction.pass_reason = f"Global Scaling: {slate_reason}"
                elif (
                    scaled_units > 0.0
                    and (prediction.verdict or "").startswith("Bet")
                    and bet_odds is not None
                ):
                    prediction.verdict = f"Bet {scaled_units:.2f}u @ {bet_odds:+.0f}"

                # BetLog guard
                prev_verdict = candidate.get("old_verdict")
                if prev_verdict is not None and prev_verdict.startswith("Bet"):
                    logger.debug(
                        "Skipping duplicate paper trade for %s @ %s "
                        "(old verdict already Bet)",
                        candidate["away_team"], candidate["home_team"],
                    )
                    continue

                # Create the bet using our pre-calculated scaled dollar amount
                paper_bet, net_change = _create_paper_bet(
                    db, game, prediction, 
                    daily_exposure=current_exposure_acc,
                    scaled_bet_dollars=scaled_dollars
                )
                
                current_exposure_acc += net_change
                paper_trades_created += 1
                logger.info(
                    "BET: %s @ %s — %.2fu%s (paper: %s)",
                    candidate["away_team"],
                    candidate["home_team"],
                    scaled_units,
                    f" [{slate_reason}]" if slate_reason else "",
                    paper_bet.pick,
                )

            # Update current_exposure for parlay logic below
            current_exposure = current_exposure_acc
        else:
            # Still need to define current_exposure if no candidates
            current_exposure = _daily_exposure(db)

        # ----------------------------------------------------------------
        # STEP 4c: Cross-game parlay recommendations
        # Build a parlay slate from qualifying straight bets (units > 0),
        # then enforce the remaining portfolio capacity so parlay sizing
        # never overflows what straight bets have already consumed.
        # ----------------------------------------------------------------
        parlay_slate = []
        for candidate in bet_candidates:
            if candidate["recommended_units"] <= 0:
                continue  # Zeroed by slate cap — skip for parlays too
            game = candidate["game"]
            calcs = candidate["analysis"].full_analysis.get("calculations", {})
            bet_side = candidate["bet_side"]
            spread = candidate["spread"]
            home_team = candidate["home_team"]
            away_team = candidate["away_team"]

            # Build human-readable pick string (e.g. "Duke -4.5", "UNC +3.5")
            if bet_side == "away":
                away_spread = -spread
                sign = "+" if away_spread > 0 else ""
                pick = f"{away_team} {sign}{away_spread:.1f}"
            else:
                sign = "+" if spread > 0 else ""
                pick = f"{home_team} {sign}{spread:.1f}"

            parlay_slate.append({
                "game_id": game.id,
                "pick": pick,
                "edge_conservative": candidate["edge_conservative"],
                "full_analysis": {
                    "calculations": {
                        "bet_odds": calcs.get("bet_odds", -110),
                        "market_prob": calcs.get("market_prob", 0.5),
                        "edge_conservative": calcs.get("edge_conservative", 0.0),
                    }
                },
            })

        if len(parlay_slate) >= 2:
            _starting_bankroll = float(os.getenv("STARTING_BANKROLL", "1000"))
            _max_daily_pct = float(os.getenv("MAX_DAILY_EXPOSURE_PCT", "20.0"))
            _max_daily_dollars = _starting_bankroll * _max_daily_pct / 100.0
            _remaining_capacity = max(0.0, _max_daily_dollars - current_exposure)

            recommended_parlays = build_optimal_parlays(
                parlay_slate,
                max_legs=3,
                max_parlays=5,
                remaining_capacity_dollars=_remaining_capacity,
                bankroll=_starting_bankroll,
            )

            if recommended_parlays:
                logger.info(
                    "=== PARLAY RECOMMENDATIONS (%d tickets, $%.2f remaining capacity) ===",
                    len(recommended_parlays), _remaining_capacity,
                )
                for parlay in recommended_parlays:
                    logger.info(format_parlay_ticket(parlay))
            else:
                logger.info("No parlay opportunities met the edge threshold today.")

        # ----------------------------------------------------------------
        # STEP 5: Single bulk commit for all successful games
        # ----------------------------------------------------------------
        db.commit()
        logger.info(
            "Bulk commit: %d games, %d bets, %d paper trades",
            games_analyzed, bets_recommended, paper_trades_created,
        )

        return _summary(
            start_time, games_analyzed, bets_recommended,
            paper_trades_created, errors, games_considered=games_considered,
        )

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
            games_considered=games_considered,
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
    games_considered: int = 0,
) -> Dict:
    duration = (datetime.utcnow() - start_time).total_seconds()
    n = max(games_analyzed, 1)
    result = {
        "status": status,
        "games_analyzed": games_analyzed,
        "bets_recommended": bets_recommended,
        "games_considered": games_considered,
        "paper_trades_created": paper_trades_created,
        "bet_rate": round(bets_recommended / n, 3),
        "consider_rate": round(games_considered / n, 3),
        "pass_rate": round((games_analyzed - bets_recommended - games_considered) / n, 3),
        "errors": errors,
        "timestamp": datetime.utcnow().isoformat(),
        "duration_seconds": round(duration, 2),
    }
    logger.info(
        "Analysis complete in %.1fs — %d games | %d BET (%.0f%%) | "
        "%d CONSIDER (%.0f%%) | %d PASS | %d paper trades | %d errors",
        duration,
        games_analyzed,
        bets_recommended, 100 * bets_recommended / n,
        games_considered, 100 * games_considered / n,
        games_analyzed - bets_recommended - games_considered,
        paper_trades_created,
        len(errors),
    )
    return result
