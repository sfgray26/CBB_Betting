"""
Daily Fantasy Baseball Lineup Optimizer

Uses sportsbook odds from The Odds API to compute implied team run totals,
then ranks batters and pitchers for daily lineup decisions.

Key logic:
  - Game total + spread -> implied runs per team
  - High implied runs -> stack batters from that team
  - Low opponent implied runs + high K/9 -> stream SP
  - Injury filter -> skip IL/DTD players
  - Park factor adjustment (via ballpark_factors.py)

Usage:
    from backend.fantasy_baseball.daily_lineup_optimizer import DailyLineupOptimizer
    opt = DailyLineupOptimizer()
    report = opt.build_daily_report("2026-04-01")
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests

from backend.models import SessionLocal
from backend.services.config_service import get_threshold as _thresh_cfg, is_flag_enabled as _is_flag
from backend.services.probable_pitcher_fallback import (
    infer_probable_pitcher_map,
    load_probable_pitchers_from_snapshot,
)
from backend.utils.env_utils import get_float_env

logger = logging.getLogger(__name__)

# Additional aliases for common alternate abbreviations
_TEAM_ALIASES = {
    "TBR": "TB",  # Tampa Bay Rays (ESPN/Odds API style)
    "KCR": "KC",  # Kansas City Royals (ESPN/Odds API style)
    "SFG": "SF",  # San Francisco Giants (ESPN/Odds API style)
    "SDP": "SD",  # San Diego Padres (ESPN/Odds API style)
    "WSN": "WSH", # Washington Nationals (ESPN style)
    "AZ": "ARI",  # Arizona Diamondbacks (Yahoo style)
    "CHW": "CWS", # Chicago White Sox (Yahoo/ESPN style -> standard)
}


def normalize_team_abbr(abbr: str) -> str:
    """Normalize team abbreviation to Yahoo standard."""
    if not abbr:
        return ""
    abbr_upper = abbr.upper()
    # First check if it's an alias
    if abbr_upper in _TEAM_ALIASES:
        return _TEAM_ALIASES[abbr_upper]
    # Otherwise return as-is (already in standard form)
    return abbr_upper


# Park run factors (1.0 = neutral; > 1.0 = hitter-friendly)
_PARK_FACTORS: Dict[str, float] = {
    "COL": 1.25, "CIN": 1.10, "TEX": 1.08, "PHI": 1.07, "ARI": 1.06,
    "MIL": 1.05, "NYY": 1.04, "BOS": 1.03, "CHC": 1.02, "TOR": 1.02,
    "STL": 1.00, "ATL": 1.00, "LAD": 1.00, "NYM": 0.99, "DET": 0.99,
    "KC":  0.98, "BAL": 0.98, "WSH": 0.97, "MIN": 0.97, "CWS": 0.97,
    "SEA": 0.96, "CLE": 0.96, "HOU": 0.95, "MIA": 0.95, "OAK": 0.94,
    "TB":  0.93, "SF":  0.92, "SD":  0.92, "PIT": 0.91, "LAA": 0.99,
}


@dataclass
class MLBGameOdds:
    """Parsed odds for one MLB game."""
    game_id: str
    commence_time: str
    home_team: str          # full name
    away_team: str          # full name
    home_abbrev: str
    away_abbrev: str
    spread_home: Optional[float] = None     # negative = home favored
    total: Optional[float] = None
    moneyline_home: Optional[float] = None
    moneyline_away: Optional[float] = None
    # Derived
    implied_home_runs: Optional[float] = None
    implied_away_runs: Optional[float] = None
    park_factor: float = 1.0
    # Run environment for pitcher streaming (PR-22)
    home_win_prob: Optional[float] = None   # implied win probability (0-1)
    away_win_prob: Optional[float] = None   # implied win probability (0-1)
    game_total: Optional[float] = None      # alias for total, explicit naming


@dataclass
class BatterRanking:
    """Daily batter ranking for lineup decisions."""
    name: str
    team: str                   # abbreviation
    positions: List[str]
    implied_team_runs: float    # team's expected runs today
    park_factor: float
    projected_r: float = 0.0
    projected_hr: float = 0.0
    projected_rbi: float = 0.0
    projected_avg: float = 0.0
    is_home: bool = False
    status: Optional[str] = None
    lineup_score: float = 0.0   # composite daily score
    reason: str = ""
    has_game: bool = False      # Whether team plays today
    composite_z: float = 0.0   # Live 14-day rolling z-score (from player_scores)


@dataclass
class PitcherRanking:
    """Daily SP streaming ranking."""
    name: str
    team: str
    opponent: str
    implied_opp_runs: float     # lower = better for pitcher
    park_factor: float
    projected_k: float = 0.0
    projected_era: float = 0.0
    projected_ip: float = 0.0
    is_home: bool = False
    status: Optional[str] = None
    stream_score: float = 0.0
    reason: str = ""


@dataclass
class LineupSlotResult:
    """One filled lineup slot from the constraint solver."""
    slot: str               # "C", "1B", "2B", "3B", "SS", "OF", "Util", "SP", "RP", "BN"
    player_name: str
    player_team: str
    positions: List[str]
    lineup_score: float
    implied_runs: float
    park_factor: float
    has_game: bool          # True if team plays today
    status: Optional[str]  # injury status from Yahoo
    reason: str             # human-readable explanation


# ---------------------------------------------------------------------------
# Yahoo H2H standard slot config: fill scarcest positions first so that
# multi-eligible players (e.g. Castro 2B/3B) cover whatever gap remains.
# ---------------------------------------------------------------------------
_DEFAULT_BATTER_SLOTS: List[Tuple[str, List[str]]] = [
    # Scarcity-first order: C and SS are rarest, fill them before 1B.
    # Multi-eligible players (e.g. 1B/SS) are then free to cover the scarcer slots
    # rather than being wasted on the abundant 1B position.
    ("C",    ["C"]),
    ("SS",   ["SS"]),
    ("2B",   ["2B"]),
    ("3B",   ["3B"]),
    ("1B",   ["1B"]),
    ("OF",   ["OF", "LF", "CF", "RF"]),
    ("OF",   ["OF", "LF", "CF", "RF"]),
    ("OF",   ["OF", "LF", "CF", "RF"]),
    ("Util", ["C", "1B", "2B", "3B", "SS", "OF", "LF", "CF", "RF", "DH"]),
]

# Statuses that mean "occupying an IL slot, not an active roster spot"
_INACTIVE_STATUSES = frozenset({"IL", "IL10", "IL60", "NA", "OUT"})

# Static scarcity rank: lower = scarcer. Mirrors POSITION_SCARCITY in
# daily_ingestion._sync_position_eligibility (kept in sync manually).
_POSITION_SCARCITY: dict[str, int] = {
    "C": 1, "SS": 2, "2B": 3, "3B": 4, "CF": 5,
    "SP": 6, "RP": 7, "LF": 8, "RF": 9, "1B": 10, "DH": 11, "OF": 12,
}


def _get_scarcity_rank(db, primary_position: str) -> int:
    """
    Return the scarcity_rank for a position, lowest = most scarce (C=1).

    Strategy:
      1. Query position_eligibility for the median scarcity_rank among all
         players at this primary_position (single query, cached per call site).
      2. Fall back to _POSITION_SCARCITY static dict if DB returns nothing.
      3. Fall back to 13 (least scarce) if position unknown.

    Integration point: call this from assign_lineup_slots() when two players
    are equally eligible for a slot and one should prefer their natural position
    over Util. Example:
        if score_a == score_b:
            rank_a = _get_scarcity_rank(db, player_a.primary_position)
            rank_b = _get_scarcity_rank(db, player_b.primary_position)
            winner = player_a if rank_a < rank_b else player_b
    """
    if primary_position in _POSITION_SCARCITY:
        static_rank = _POSITION_SCARCITY[primary_position]
    else:
        static_rank = 13

    if db is None:
        return static_rank

    try:
        from backend.models import PositionEligibility
        from sqlalchemy import func
        result = (
            db.query(func.min(PositionEligibility.scarcity_rank))
            .filter(
                PositionEligibility.primary_position == primary_position,
                PositionEligibility.scarcity_rank.isnot(None),
            )
            .scalar()
        )
        return int(result) if result is not None else static_rank
    except Exception:
        return static_rank


class DailyLineupOptimizer:
    """
    Combines sportsbook odds with projection data to rank
    batters and pitchers for daily fantasy lineup decisions.
    """

    def __init__(self):
        self._odds_cache: Dict[str, List[MLBGameOdds]] = {}

    # ------------------------------------------------------------------
    # Odds fetching
    # ------------------------------------------------------------------

    def fetch_mlb_odds(self, game_date: Optional[str] = None) -> List[MLBGameOdds]:
        """
        Fetch MLB game odds from mlb_odds_snapshot table.

        Returns list of MLBGameOdds with implied run totals computed.
        Falls back to _load_schedule_fallback_games() on errors.
        """
        from backend.models import SessionLocal, MLBOddsSnapshot, MLBGameLog, MLBTeam
        from sqlalchemy import func, case
        from sqlalchemy.orm import aliased

        target_date = self._parse_game_date(game_date)
        if target_date is None:
            return []

        cache_key = game_date or "today"
        if cache_key in self._odds_cache:
            return self._odds_cache[cache_key]

        db = SessionLocal()
        try:
            AwayTeam = aliased(MLBTeam)
            HomeTeam = aliased(MLBTeam)

            max_window_subq = (
                db.query(
                    MLBOddsSnapshot.game_id,
                    func.max(MLBOddsSnapshot.snapshot_window).label("max_window")
                )
                .group_by(MLBOddsSnapshot.game_id)
                .subquery()
            )

            vendor_priority = case(
                {"pinnacle": 0, "draftkings": 1, "fanduel": 2, "betmgm": 3, "caesars": 4},
                value=MLBOddsSnapshot.vendor,
                else_=5
            )

            rows = (
                db.query(
                    MLBOddsSnapshot,
                    AwayTeam.abbreviation.label("away_abbr"),
                    AwayTeam.display_name.label("away_name"),
                    HomeTeam.abbreviation.label("home_abbr"),
                    HomeTeam.display_name.label("home_name"),
                    MLBGameLog.raw_payload['date'].astext.label("game_time_raw"),
                    vendor_priority.label("vp")
                )
                .join(
                    max_window_subq,
                    (MLBOddsSnapshot.game_id == max_window_subq.c.game_id) &
                    (MLBOddsSnapshot.snapshot_window == max_window_subq.c.max_window)
                )
                .join(MLBGameLog, MLBOddsSnapshot.game_id == MLBGameLog.game_id)
                .join(AwayTeam, MLBGameLog.away_team_id == AwayTeam.team_id)
                .join(HomeTeam, MLBGameLog.home_team_id == HomeTeam.team_id)
                .filter(MLBGameLog.game_date == target_date)
                .order_by(MLBOddsSnapshot.game_id, "vp")
                .all()
            )

            seen: set[int] = set()
            games: List[MLBGameOdds] = []
            for odds, away_abbr, away_name, home_abbr, home_name, game_time_raw, _ in rows:
                if odds.game_id in seen:
                    continue
                seen.add(odds.game_id)

                total_f = float(odds.total) if odds.total else None
                spread_f = float(odds.spread_home) if odds.spread_home else None
                implied_h, implied_a = None, None
                if total_f is not None:
                    implied_h, implied_a = self._implied_runs(total_f, spread_f or 0.0)
                
                # Compute win probabilities for run environment scoring (PR-22)
                ml_home = float(odds.ml_home_odds) if odds.ml_home_odds else None
                ml_away = float(odds.ml_away_odds) if odds.ml_away_odds else None
                win_h, win_a = self._compute_win_probabilities(ml_home, ml_away)

                games.append(MLBGameOdds(
                    game_id=str(odds.game_id),
                    commence_time=game_time_raw or str(target_date),
                    home_team=home_name,
                    away_team=away_name,
                    home_abbrev=home_abbr,
                    away_abbrev=away_abbr,
                    spread_home=spread_f,
                    total=total_f,
                    moneyline_home=float(odds.ml_home_odds) if odds.ml_home_odds else None,
                    moneyline_away=float(odds.ml_away_odds) if odds.ml_away_odds else None,
                    implied_home_runs=implied_h,
                    implied_away_runs=implied_a,
                    park_factor=_PARK_FACTORS.get(home_abbr, 1.0),
                    # Run environment data for pitcher streaming (PR-22)
                    home_win_prob=win_h,
                    away_win_prob=win_a,
                    game_total=total_f,
                ))

            logger.info(
                "lineup_optimizer: loaded odds for %d games from DB (%s)",
                len(games),
                target_date.isoformat()
            )
            self._odds_cache[cache_key] = games
            return games

        except Exception as exc:
            logger.warning("lineup_optimizer: DB odds fetch failed: %s", exc)
            fallback_games = self._load_schedule_fallback_games(game_date)
            if fallback_games:
                self._odds_cache[cache_key] = fallback_games
            return fallback_games
        finally:
            db.close()

    def _load_schedule_fallback_games(self, game_date: Optional[str]) -> List[MLBGameOdds]:
        """Build synthetic game context from persisted probable-pitcher snapshots."""
        from backend.models import ProbablePitcherSnapshot

        target_date = self._parse_game_date(game_date)
        if target_date is None:
            return []

        db = SessionLocal()
        try:
            rows = (
                db.query(
                    ProbablePitcherSnapshot.team,
                    ProbablePitcherSnapshot.opponent,
                    ProbablePitcherSnapshot.is_home,
                    ProbablePitcherSnapshot.park_factor,
                )
                .filter(ProbablePitcherSnapshot.game_date == target_date)
                .all()
            )
        except Exception as exc:
            logger.warning("Failed to load probable-pitcher schedule fallback: %s", exc)
            return []
        finally:
            db.close()

        synthetic_games: Dict[str, MLBGameOdds] = {}
        for row in rows:
            if hasattr(row, "team"):
                team = row.team
                opponent = row.opponent
                is_home = row.is_home
                park_factor = row.park_factor
            else:
                team, opponent, is_home, park_factor = row
            team_norm = normalize_team_abbr(team)
            opp_norm = normalize_team_abbr(opponent)
            if not team_norm or not opp_norm or is_home is None:
                continue

            home_team = team_norm if is_home else opp_norm
            away_team = opp_norm if is_home else team_norm
            game_key = f"{away_team}@{home_team}"
            home_park_factor = park_factor or _PARK_FACTORS.get(home_team, 1.0)
            neutral_total = max(7.0, min(11.5, round(9.0 * home_park_factor, 2)))
            implied_home_runs = round(min(7.0, max(2.5, neutral_total / 2.0 + 0.15)), 2)
            implied_away_runs = round(min(7.0, max(2.5, neutral_total - implied_home_runs)), 2)

            synthetic_games[game_key] = MLBGameOdds(
                game_id=f"snapshot:{target_date.isoformat()}:{game_key}",
                commence_time="",
                home_team=home_team,
                away_team=away_team,
                home_abbrev=home_team,
                away_abbrev=away_team,
                implied_home_runs=implied_home_runs,
                implied_away_runs=implied_away_runs,
                park_factor=home_park_factor,
            )

        games = list(synthetic_games.values())
        if games:
            logger.info(
                "Lineup optimizer using probable-pitcher schedule fallback for %s (%d games)",
                target_date.isoformat(),
                len(games),
            )
        return games

    @staticmethod
    def _parse_game_date(game_date: Optional[str]) -> Optional[date]:
        """Parse a game date string in YYYY-MM-DD format using ET as default."""
        if game_date:
            try:
                return datetime.fromisoformat(game_date).date()
            except ValueError:
                logger.warning("Could not parse game_date '%s' for schedule fallback", game_date)
                return None
        return datetime.now(ZoneInfo("America/New_York")).date()


    @staticmethod
    def _implied_runs(total: float, spread_home: float) -> Tuple[float, float]:
        """
        Convert game total + spread to per-team implied runs.

        Spread reflects run differential, so:
          home_runs = (total - spread_home) / 2
          away_runs = total - home_runs

        spread_home is negative when home team is favored (e.g., -1.5).
        """
        home_runs = (total - spread_home) / 2.0
        away_runs = total - home_runs
        # Clamp to realistic range
        home_runs = max(1.0, min(12.0, home_runs))
        away_runs = max(1.0, min(12.0, away_runs))
        return round(home_runs, 2), round(away_runs, 2)

    @staticmethod
    def _win_prob_from_moneyline(ml: Optional[float]) -> Optional[float]:
        """
        Convert American moneyline odds to implied win probability (0-1).
        
        Handles vig removal by normalizing both sides to sum to 1.0.
        Positive odds (underdog): prob = 100 / (odds + 100)
        Negative odds (favorite): prob = abs(odds) / (abs(odds) + 100)
        """
        if ml is None:
            return None
        if ml > 0:
            raw_prob = 100 / (ml + 100)
        else:
            raw_prob = abs(ml) / (abs(ml) + 100)
        return round(raw_prob, 3)

    def _compute_win_probabilities(
        self, ml_home: Optional[float], ml_away: Optional[float]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute vig-adjusted win probabilities for both teams.
        
        Returns (home_prob, away_prob) that sum to 1.0, or (None, None) if data missing.
        """
        if ml_home is None or ml_away is None:
            return None, None
        
        raw_home = self._win_prob_from_moneyline(ml_home)
        raw_away = self._win_prob_from_moneyline(ml_away)
        
        if raw_home is None or raw_away is None:
            return None, None
        
        # Remove vig by normalizing
        total = raw_home + raw_away
        if total == 0:
            return None, None
        
        return round(raw_home / total, 3), round(raw_away / total, 3)

    # ------------------------------------------------------------------
    # Batter ranking
    # ------------------------------------------------------------------

    def rank_batters(
        self,
        roster: List[dict],
        projections: List[dict],
        game_date: Optional[str] = None,
    ) -> List[BatterRanking]:
        """
        Rank batters by daily lineup value.

        Args:
            roster: List of player dicts from YahooFantasyClient.get_roster()
            projections: List of player projection dicts from projections_loader
            game_date: YYYY-MM-DD (defaults to today)

        Returns:
            Sorted list of BatterRanking (best first).
        """
        games = self.fetch_mlb_odds(game_date)
        team_odds = self._build_team_odds_map(games)

        # Pre-load pitcher quality scores for today's games
        pitcher_quality: Dict[str, float] = {}
        target_date = self._parse_game_date(game_date)
        if target_date is not None:
            from backend.models import ProbablePitcherSnapshot
            _pp_db = SessionLocal()
            try:
                pp_rows = _pp_db.query(
                    ProbablePitcherSnapshot.team,
                    ProbablePitcherSnapshot.quality_score,
                ).filter(
                    ProbablePitcherSnapshot.game_date == target_date,
                    ProbablePitcherSnapshot.quality_score.isnot(None),
                ).all()
                pitcher_quality = {r.team: r.quality_score for r in pp_rows}
            except Exception:
                pass  # neutral fallback for all batters
            finally:
                _pp_db.close()

        # Pre-load composite_z live bonus from player_scores (14-day rolling window)
        composite_z_lookup: Dict[str, float] = {}
        _cz_db = SessionLocal()
        try:
            from sqlalchemy import text as _text
            cz_rows = _cz_db.execute(_text(
                "SELECT LOWER(pe.player_name) AS name_key, ps.composite_z "
                "FROM position_eligibility pe "
                "JOIN player_scores ps ON pe.bdl_player_id = ps.bdl_player_id "
                "WHERE ps.as_of_date = (SELECT MAX(as_of_date) FROM player_scores) "
                "  AND ps.window_days = 14 "
                "  AND pe.bdl_player_id IS NOT NULL"
            )).fetchall()
            composite_z_lookup = {r.name_key: r.composite_z for r in cz_rows
                                  if r.composite_z is not None}
        except Exception:
            pass  # neutral fallback for all players
        finally:
            _cz_db.close()

        # Pre-load matchup context boost (PR 5.5, gated by feature.matchup_enabled)
        matchup_lookup: Dict[str, Tuple[float, float]] = {}  # name.lower() → (m_z, m_conf)
        _use_matchup = False
        _boost_cap = 0.2
        _z_scale = 0.1
        try:
            _use_matchup = _is_flag("feature.matchup_enabled")
        except Exception:
            pass

        if _use_matchup and target_date is not None:
            _boost_cap = _thresh_cfg("matchup.boost.cap", default=0.2)
            _z_scale = _thresh_cfg("matchup.boost.z_scale", default=0.1)
            _mq_db = SessionLocal()
            try:
                from sqlalchemy import text as _mq_text
                mq_rows = _mq_db.execute(_mq_text("""
                    SELECT LOWER(pim.full_name) AS name_key,
                           mc.matchup_z,
                           mc.matchup_confidence
                    FROM matchup_context mc
                    JOIN player_id_mapping pim ON mc.bdl_player_id = pim.bdl_id
                    WHERE mc.game_date = :gd
                      AND mc.matchup_confidence IS NOT NULL
                """), {"gd": target_date}).fetchall()
                matchup_lookup = {
                    r.name_key: (r.matchup_z, r.matchup_confidence)
                    for r in mq_rows
                }
            except Exception:
                pass  # neutral fallback — no boost applied
            finally:
                _mq_db.close()

        proj_by_name = {p["name"].lower(): p for p in projections
                        if p.get("type") == "batter" or p.get("player_type") == "batter"}

        rankings = []
        for player in roster:
            positions = player.get("positions", [])
            # Skip pitchers - if ANY position is SP/RP/P, they're a pitcher
            # This handles two-way players (e.g., Shohei Ohtani with SP + Util)
            if any(p in ("SP", "RP", "P") for p in positions):
                continue
            status = player.get("status")
            if status in ("IL", "IL60", "NA"):
                continue

            name = player.get("name", "")
            team_raw = player.get("team", "")
            team = normalize_team_abbr(team_raw)
            proj = proj_by_name.get(name.lower(), {})

            # Get team's implied runs from odds
            odds_data = team_odds.get(team, {})
            implied_runs = odds_data.get("implied_runs", 4.5)   # league avg fallback
            is_home = odds_data.get("is_home", False)
            park_factor = odds_data.get("park_factor", 1.0)
            has_game = team in team_odds  # True if team has a game today

            # 70/30 Talent-Prior scoring model
            # TALENT PRIOR (70%): per-game normalized ROS projections + live composite_z
            # MATCHUP MODIFIER (30%): daily environment (run environment, park, home, opp pitcher)
            _GAMES_ROS = 130  # approx remaining games (mid-April baseline)
            cz_val = composite_z_lookup.get(name.lower(), 0.0)
            has_proj = bool(proj)
            if has_proj:
                # Per-game normalized Steamer counting stats + rate contribution
                # avg * 5.0 is NOT per-game — intentional: rate stat contributes ~13-15 of ~20 range
                talent_prior = (
                    proj.get("hr", 0) * 2.0 / _GAMES_ROS
                    + proj.get("r", 0) * 0.3 / _GAMES_ROS
                    + proj.get("rbi", 0) * 0.3 / _GAMES_ROS
                    + proj.get("nsb", 0) * 0.5 / _GAMES_ROS
                    + proj.get("avg", 0.0) * 5.0
                ) * 10 + cz_val * 1.0
            else:
                # No Steamer data: composite_z is the sole talent signal.
                # Floor of 12.0 so that elite players (cz≥2.5) beat replacement-level
                # players who do have projections (~14-16 range).
                talent_prior = 12.0 + cz_val * 2.0

            # MATCHUP MODIFIER (30%): daily environment
            opp_team = team_odds.get(team, {}).get("opponent", "")
            opp_qs = pitcher_quality.get(opp_team)  # None = no data
            matchup_modifier = (
                (implied_runs - 4.5) * 0.5       # run environment vs league avg
                + (park_factor - 1.0) * 2.0      # hitter-friendly park bonus
                + (0.2 if is_home else 0.0)       # marginal home-field edge
            )
            if opp_qs is not None:
                # Positive quality_score = better pitcher = reduces batter score.
                # Negative = weak pitcher = boosts batter score.
                matchup_modifier -= opp_qs * 0.15

            lineup_score = talent_prior * 0.7 + matchup_modifier * 0.3 + 6.0

            # PR 5.5 — Bounded matchup context boost (feature-flagged OFF by default)
            if _use_matchup:
                m_z, m_conf = matchup_lookup.get(name.lower(), (0.0, 0.0))
                matchup_boost = (
                    max(-_boost_cap, min(_boost_cap, m_z * _z_scale)) * m_conf
                )
                lineup_score = lineup_score * (1 + matchup_boost)

            reason_parts = [f"team implied {implied_runs:.1f}R"]
            if park_factor > 1.05:
                reason_parts.append(f"hitter park ({park_factor:.2f}x)")
            if is_home:
                reason_parts.append("home")
            if status and status not in ("", "DTD"):
                reason_parts.append(f"status: {status}")

            rankings.append(BatterRanking(
                name=name,
                team=team,
                positions=positions,
                implied_team_runs=implied_runs,
                park_factor=park_factor,
                projected_r=proj.get("r", 0),
                projected_hr=proj.get("hr", 0),
                projected_rbi=proj.get("rbi", 0),
                projected_avg=proj.get("avg", 0.0),
                is_home=is_home,
                status=status,
                lineup_score=round(lineup_score, 3),
                reason=", ".join(reason_parts),
                has_game=has_game,
                composite_z=cz_val,
            ))

        rankings.sort(key=lambda x: x.lineup_score, reverse=True)
        return rankings

    # ------------------------------------------------------------------
    # SP streaming
    # ------------------------------------------------------------------

    def rank_streamers(
        self,
        free_agents: List[dict],
        projections: List[dict],
        game_date: Optional[str] = None,
        min_k9: float = 7.5,
        max_era: float = 4.50,
    ) -> List[PitcherRanking]:
        """
        Rank streaming SP candidates by daily matchup quality.

        Args:
            free_agents: Players from YahooFantasyClient.get_free_agents('SP')
            projections: Pitcher projections from projections_loader
            game_date: YYYY-MM-DD
            min_k9: Minimum K/9 for consideration
            max_era: Maximum projected ERA for consideration
        """
        games = self.fetch_mlb_odds(game_date)
        team_odds = self._build_team_odds_map(games)

        proj_by_name = {p["name"].lower(): p for p in projections
                        if (p.get("type") or p.get("player_type", "")) == "pitcher"}

        # Pre-load composite_z live bonus for pitcher FAs (14-day rolling window)
        composite_z_lookup: Dict[str, float] = {}
        _cz_db = SessionLocal()
        try:
            from sqlalchemy import text as _text
            cz_rows = _cz_db.execute(_text(
                "SELECT LOWER(pe.player_name) AS name_key, ps.composite_z "
                "FROM position_eligibility pe "
                "JOIN player_scores ps ON pe.bdl_player_id = ps.bdl_player_id "
                "WHERE ps.as_of_date = (SELECT MAX(as_of_date) FROM player_scores) "
                "  AND ps.window_days = 14 "
                "  AND pe.bdl_player_id IS NOT NULL"
            )).fetchall()
            composite_z_lookup = {r.name_key: r.composite_z for r in cz_rows
                                  if r.composite_z is not None}
        except Exception:
            pass  # neutral fallback for all players
        finally:
            _cz_db.close()

        rankings = []
        for player in free_agents:
            status = player.get("status")
            if status in ("IL", "IL60", "NA"):
                continue
            name = player.get("name", "")
            team_raw = player.get("team", "")
            team = normalize_team_abbr(team_raw)
            proj = proj_by_name.get(name.lower(), {})

            k9 = proj.get("k9", 0.0)
            era = proj.get("era", 5.0)
            if k9 < min_k9 or era > max_era:
                continue

            # Pitcher wants LOW opponent implied runs
            odds_data = team_odds.get(team, {})
            opp_team = odds_data.get("opponent", "")
            opp_odds = team_odds.get(opp_team, {})
            implied_opp_runs = opp_odds.get("implied_runs", 4.5)
            is_home = odds_data.get("is_home", False)
            park_factor = odds_data.get("park_factor", 1.0)

            # Stream score: lower opponent runs = better; higher K/9 = better
            # Normalize: 3.5 opp runs = best, 5.5 = worst
            env_score = max(0.0, (5.5 - implied_opp_runs) / 2.0) * 10  # 0-10
            k_score = min(10.0, k9 - 5.0)  # 0-10 for 5-15 K/9
            park_score = (2.0 - park_factor) * 5  # pitcher parks get bonus
            stream_score = env_score * 0.5 + k_score * 0.3 + park_score * 0.2

            # Live rolling bonus: composite_z for pitchers reflects z_era, z_whip, z_k_p
            cz_val = composite_z_lookup.get(name.lower(), 0.0)
            stream_score += cz_val * 0.5  # ±1.5 on a ~-1 to +9 scale; fallback 0.0

            reason_parts = [f"opp {implied_opp_runs:.1f}R", f"K/9 {k9:.1f}"]
            if is_home:
                reason_parts.append("home")
            if park_factor < 0.97:
                reason_parts.append(f"pitcher park ({park_factor:.2f}x)")

            rankings.append(PitcherRanking(
                name=name,
                team=team,
                opponent=opp_team,
                implied_opp_runs=implied_opp_runs,
                park_factor=park_factor,
                projected_k=proj.get("k", 0.0),
                projected_era=era,
                projected_ip=proj.get("ip", 0.0),
                is_home=is_home,
                status=status,
                stream_score=round(stream_score, 3),
                reason=", ".join(reason_parts),
            ))

        rankings.sort(key=lambda x: x.stream_score, reverse=True)
        return rankings

    # ------------------------------------------------------------------
    # Constraint-aware lineup solver
    # ------------------------------------------------------------------

    def solve_lineup(
        self,
        roster: List[dict],
        projections: List[dict],
        game_date: Optional[str] = None,
        slot_config: Optional[List[Tuple[str, List[str]]]] = None,
        db=None,
    ) -> Tuple[List[LineupSlotResult], List[str]]:
        """
        Fill Yahoo lineup slots using greedy scarcity-first constraint solving.

        Slots are filled in order of scarcity (C → SS → 2B → 3B → 1B → OF×3 → Util)
        so that multi-eligible flex players (e.g. Castro 2B/3B) naturally cover
        whichever scarce position is left uncovered, rather than being wasted on OF.

        Off-day detection: when the Odds API returns data for 10+ teams (≥5 games),
        players whose team has no game are deprioritised — they fill slots only if
        no in-game player is available.

        Returns:
            (slot_results, warnings)
            slot_results — one LineupSlotResult per slot + BN entries for bench
            warnings     — human-readable alerts (empty slot, off-day start, etc.)
        """
        if game_date is None:
            game_date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

        games = self.fetch_mlb_odds(game_date)
        team_odds = self._build_team_odds_map(games)

        # Only apply off-day filtering when we have a credible slate (≥5 games worth
        # of teams).  Sparse/missing odds data must not bench healthy players.
        apply_offday_filter = len(team_odds) >= 10

        slots = slot_config if slot_config is not None else _DEFAULT_BATTER_SLOTS

        # Ranked by lineup_score descending; IL/pitcher rows already excluded by rank_batters
        ranked: List[BatterRanking] = self.rank_batters(roster, projections, game_date)

        # Belt-and-suspenders: also exclude any IL10/OUT rows rank_batters may have kept
        ranked = [b for b in ranked if b.status not in _INACTIVE_STATUSES]

        def _has_game(team: str) -> bool:
            return team in team_odds

        assigned: set = set()
        slot_results: List[LineupSlotResult] = []
        warnings: List[str] = []

        # Sort key: lineup_score DESC, scarcity_rank ASC (tiebreaker only).
        # _get_scarcity_rank falls back to _POSITION_SCARCITY when db is None.
        def _slot_sort_key(b: BatterRanking) -> tuple:
            primary = b.positions[0] if b.positions else "OF"
            return (-b.lineup_score, _get_scarcity_rank(db, primary))

        for slot_label, eligible_positions in slots:
            # Collect all eligible candidates, split by off-day status
            in_game: List[BatterRanking] = []
            off_day_eligible: List[BatterRanking] = []

            for b in ranked:
                if b.name in assigned:
                    continue
                if not any(pos in b.positions for pos in eligible_positions):
                    continue
                if apply_offday_filter and not _has_game(b.team):
                    # Log data quality issue: player has no game when expected
                    import json
                    logger.info(json.dumps({
                        "event": "data_quality_issue",
                        "issue_type": "matchup_detection_miss",
                        "player_name": b.name,
                        "team": b.team,
                        "game_date": game_date,
                        "team_odds_keys": list(team_odds.keys()),
                        "odds_api_games_count": len(games) if games else 0
                    }))
                    off_day_eligible.append(b)
                    continue
                in_game.append(b)

            # Pass 1 → prefer in-game; Pass 2 → fall back to off-day
            candidates = in_game if in_game else off_day_eligible
            using_offday_fallback = not in_game and bool(off_day_eligible) and apply_offday_filter

            # Pick best: lineup_score DESC, scarcity_rank ASC as tiebreaker
            candidates.sort(key=_slot_sort_key)
            best: Optional[BatterRanking] = candidates[0] if candidates else None

            if best is not None and using_offday_fallback:
                warnings.append(
                    f"{slot_label}: {best.name} ({best.team}) has no game today — verify schedule"
                )

            if best is None:
                warnings.append(f"No eligible active player found for {slot_label} slot")
                slot_results.append(LineupSlotResult(
                    slot=slot_label, player_name="EMPTY", player_team="",
                    positions=[], lineup_score=0.0, implied_runs=0.0,
                    park_factor=1.0, has_game=False, status=None,
                    reason=f"No eligible player for {slot_label}",
                ))
            else:
                assigned.add(best.name)
                slot_results.append(LineupSlotResult(
                    slot=slot_label,
                    player_name=best.name,
                    player_team=best.team,
                    positions=best.positions,
                    lineup_score=best.lineup_score,
                    implied_runs=best.implied_team_runs,
                    park_factor=best.park_factor,
                    has_game=_has_game(best.team),
                    status=best.status,
                    reason=best.reason,
                ))

        # All remaining eligible players → bench
        for b in ranked:
            if b.name not in assigned:
                slot_results.append(LineupSlotResult(
                    slot="BN",
                    player_name=b.name,
                    player_team=b.team,
                    positions=b.positions,
                    lineup_score=b.lineup_score,
                    implied_runs=b.implied_team_runs,
                    park_factor=b.park_factor,
                    has_game=_has_game(b.team),
                    status=b.status,
                    reason=b.reason,
                ))

        # ---- POST-GREEDY SWAP-IMPROVEMENT PASS ---------------------------------
        # After the greedy assigns slots, check if any bench player scores higher
        # than the player currently in an active slot they are eligible for.
        # Repeatedly swapping such pairs ensures no high-value player sits on bench
        # when a weaker player occupies a slot they could fill.
        #
        # This corrects cases where scarcity-first ordering still wastes a
        # multi-eligible player (e.g. a 1B/SS assigned to 1B, leaving a weaker
        # player at SS while a better 1B-eligible player sits on bench).
        #
        # Only active→bench swaps change total score (active↔active swaps cancel).
        # Run until convergence (typically ≤2 iterations for a 25-player roster).
        slot_label_to_eligible: dict = {
            label: eligible_pos for label, eligible_pos in slots
        }
        ranked_by_name: dict = {b.name: b for b in ranked}

        swap_improved = True
        while swap_improved:
            swap_improved = False
            for i, active in enumerate(slot_results):
                if active.slot == "BN" or active.player_name == "EMPTY":
                    continue
                active_eligible_pos = slot_label_to_eligible.get(active.slot, [])

                for j, bench in enumerate(slot_results):
                    if bench.slot != "BN":
                        continue
                    if bench.lineup_score <= active.lineup_score:
                        continue
                    # Don't promote an off-day bench player over an in-game active player
                    if apply_offday_filter and not bench.has_game and active.has_game:
                        continue

                    bench_player = ranked_by_name.get(bench.player_name)
                    if bench_player is None:
                        continue
                    if not any(pos in bench_player.positions for pos in active_eligible_pos):
                        continue

                    # Perform swap
                    logger.debug(
                        "swap-improve: %s (%.2f) → %s, %s (%.2f) → BN",
                        bench_player.name, bench_player.lineup_score, active.slot,
                        active.player_name, active.lineup_score,
                    )
                    slot_results[i] = LineupSlotResult(
                        slot=active.slot,
                        player_name=bench_player.name,
                        player_team=bench_player.team,
                        positions=bench_player.positions,
                        lineup_score=bench_player.lineup_score,
                        implied_runs=bench_player.implied_team_runs,
                        park_factor=bench_player.park_factor,
                        has_game=_has_game(bench_player.team),
                        status=bench_player.status,
                        reason=bench_player.reason,
                    )
                    slot_results[j] = LineupSlotResult(
                        slot="BN",
                        player_name=active.player_name,
                        player_team=active.player_team,
                        positions=active.positions,
                        lineup_score=active.lineup_score,
                        implied_runs=active.implied_runs,
                        park_factor=active.park_factor,
                        has_game=active.has_game,
                        status=active.status,
                        reason=active.reason,
                    )
                    swap_improved = True
                    break
                if swap_improved:
                    break

        return slot_results, warnings

    # ------------------------------------------------------------------
    # SP off-day detection
    # ------------------------------------------------------------------

    def flag_pitcher_starts(
        self,
        roster: List[dict],
        game_date: Optional[str] = None,
    ) -> List[dict]:
        """
        Return each pitcher from roster annotated with has_start: bool.

        SP with has_start=False should sit (no start today).
        RP always has_start=True (they can pitch any day).
        """
        if game_date is None:
            game_date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

        team_odds = self._build_team_odds_map(self.fetch_mlb_odds(game_date))
        has_slate = len(team_odds) >= 10
        
        # Fetch probable pitchers for accurate start detection
        probable_pitchers = self._fetch_probable_pitchers_for_date(game_date)

        result = []
        for p in roster:
            positions = p.get("positions", [])
            status = p.get("status")
            player_name = p.get("name", "")
            
            logger.debug(f"[PITCHER_DEBUG] {player_name}: positions={positions}, status={status}")
            
            if not any(pos in ("SP", "RP", "P") for pos in positions):
                continue
            if status in _INACTIVE_STATUSES:
                continue
            
            is_sp = "SP" in positions
            team_raw = p.get("team", "")
            team = normalize_team_abbr(team_raw)
            
            logger.debug(f"[PITCHER_DEBUG] {player_name}: is_sp={is_sp}, team={team}")
            
            # Check if this specific pitcher is the probable starter
            if is_sp:
                # Get expected opponent if team has a game
                has_game = team in team_odds if has_slate else True
                opponent = team_odds.get(team, {}).get("opponent", "") if has_slate else ""
                
                # Check if this player matches a probable starter
                is_probable = self._is_probable_starter(player_name, team, opponent, probable_pitchers)
                
                # FALLBACK 1: dict completely empty — no data at all (spring training / offseason)
                if not probable_pitchers and has_game:
                    is_probable = True
                    logger.debug(f"No probable pitchers available, assuming {player_name} ({team}) is a starter")
                # FALLBACK 2: dict has data for some teams but NOT this team — no signal,
                # default to start rather than falsely showing NO_START
                elif probable_pitchers and team not in probable_pitchers and has_game:
                    is_probable = True
                    logger.debug(f"No probable pitcher data for {team} — defaulting {player_name} to has_start=True")
                
                has_start = has_game and is_probable
            else:
                has_start = True  # RP can pitch any day
                opponent = ""

            result.append({
                **p,
                "has_start": has_start,
                "pitcher_slot": "SP" if is_sp else "RP",
                "opponent": opponent,
            })
        return result
    
    def _fetch_probable_pitchers_for_date(self, game_date: str) -> dict:
        """
        Fetch probable pitchers for a date.

        Resolution order:
          1. persisted `probable_pitchers` snapshot table
          2. MLB Stats API live lookup
          3. conservative 5-day rotation inference from recent pitcher stats

        Returns dict mapping team abbrev to pitcher name.
        """
        try:
            from datetime import date as date_type

            parsed_date = date_type.fromisoformat(game_date)
        except ValueError:
            parsed_date = None

        if parsed_date is not None:
            db = SessionLocal()
            try:
                persisted = load_probable_pitchers_from_snapshot(db, parsed_date)
                if persisted:
                    return persisted
            except Exception as exc:
                logger.warning(f"Failed to load probable pitchers from snapshot: {exc}")
            finally:
                db.close()

        url = "https://statsapi.mlb.com/api/v1/schedule"
        params = {
            "sportId": 1,
            "date": game_date,
            "hydrate": "probablePitcher",
        }
        
        probable = {}
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            for date_info in data.get("dates", []):
                for game in date_info.get("games", []):
                    teams = game.get("teams", {})
                    
                    # Home pitcher - normalize team abbreviation to Yahoo standard
                    home_team_raw = teams.get("home", {}).get("team", {}).get("abbreviation", "")
                    home_team = normalize_team_abbr(home_team_raw)
                    home_pitcher = teams.get("home", {}).get("probablePitcher", {})
                    if home_pitcher and home_team:
                        probable[home_team] = home_pitcher.get("fullName", "").lower()
                    
                    # Away pitcher - normalize team abbreviation to Yahoo standard
                    away_team_raw = teams.get("away", {}).get("team", {}).get("abbreviation", "")
                    away_team = normalize_team_abbr(away_team_raw)
                    away_pitcher = teams.get("away", {}).get("probablePitcher", {})
                    if away_pitcher and away_team:
                        probable[away_team] = away_pitcher.get("fullName", "").lower()
                        
        except Exception as e:
            logger.warning(f"Failed to fetch probable pitchers: {e}")

        if probable or parsed_date is None:
            return probable

        db = SessionLocal()
        try:
            inferred = infer_probable_pitcher_map(db, parsed_date)
            return {team: candidate.pitcher_name.lower() for team, candidate in inferred.items()}
        except Exception as exc:
            logger.warning(f"Failed to infer probable pitchers from recent stats: {exc}")
            return probable
        finally:
            db.close()

    
    def _is_probable_starter(self, player_name: str, team: str, opponent: str, probable: dict) -> bool:
        """
        Check if a player is the probable starter.
        Uses fuzzy matching on names.
        """
        if not player_name:
            return False
            
        # Direct match
        player_lower = player_name.lower()
        if team in probable:
            if probable[team] == player_lower:
                return True
            # Partial match (e.g., "Shota Imanaga" matches "Shota Imanaga")
            if player_lower in probable[team] or probable[team] in player_lower:
                return True
                
        return False

    # ------------------------------------------------------------------
    # Full daily report
    # ------------------------------------------------------------------

    def build_daily_report(
        self,
        game_date: Optional[str] = None,
        roster: Optional[List[dict]] = None,
        projections: Optional[List[dict]] = None,
    ) -> dict:
        """
        Build a full daily report: game environment + batter/pitcher rankings.

        Returns dict with:
            - game_date
            - games: list of game odds with implied runs
            - batter_rankings: sorted list of BatterRanking dicts
            - pitcher_rankings: sorted list of PitcherRanking dicts
            - best_stacks: teams with highest implied runs (stack candidates)
            - avoid_pitchers: opponents of high-implied-run teams
        """
        if game_date is None:
            game_date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

        games = self.fetch_mlb_odds(game_date)
        team_odds = self._build_team_odds_map(games)

        # Identify best stacks (teams with >5.0 implied runs)
        stack_candidates = sorted(
            [(team, data["implied_runs"]) for team, data in team_odds.items()
             if data.get("implied_runs", 0) >= 5.0],
            key=lambda x: x[1],
            reverse=True,
        )

        # Identify environments to avoid for pitchers
        high_offense_teams = [t for t, _ in stack_candidates[:4]]

        batter_rankings = []
        pitcher_rankings = []
        if roster and projections:
            batter_rankings = [
                {
                    "name": b.name, "team": b.team, "positions": b.positions,
                    "implied_runs": b.implied_team_runs, "park_factor": b.park_factor,
                    "score": b.lineup_score, "reason": b.reason, "status": b.status,
                }
                for b in self.rank_batters(roster, projections, game_date)
            ]

        return {
            "game_date": game_date,
            "games": [
                {
                    "home": g.home_abbrev, "away": g.away_abbrev,
                    "total": g.total, "spread_home": g.spread_home,
                    "implied_home": g.implied_home_runs,
                    "implied_away": g.implied_away_runs,
                    "park_factor": g.park_factor,
                }
                for g in games
            ],
            "stack_candidates": [
                {"team": t, "implied_runs": round(r, 2)} for t, r in stack_candidates
            ],
            "avoid_pitcher_matchups": high_offense_teams,
            "batter_rankings": batter_rankings,
            "pitcher_rankings": pitcher_rankings,
            "games_found": len(games),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_team_odds_map(self, games: List[MLBGameOdds]) -> Dict[str, dict]:
        """
        Build a dict: team_abbrev -> {implied_runs, is_home, opponent, park_factor,
        win_prob, game_total, run_environment_score}
        
        Run environment score combines game total and win probability to identify
        favorable pitching conditions (low total + high win prob = good stream).
        """
        result: Dict[str, dict] = {}
        logger.debug(f"[BUILD_MAP] Building team odds map from {len(games)} games")
        for g in games:
            # Always add teams to the map, even without implied runs
            # This ensures has_game detection works
            # Normalize team abbreviations to Yahoo standard
            home_norm = normalize_team_abbr(g.home_abbrev)
            away_norm = normalize_team_abbr(g.away_abbrev)
            
            if home_norm and away_norm:
                # Calculate run environment favorability for pitchers
                # Lower total = better for pitchers (fewer runs expected)
                # Higher win prob = better for pitchers (more likely to get the W)
                game_total = g.game_total or 9.0
                home_win_prob = g.home_win_prob or 0.5
                away_win_prob = g.away_win_prob or 0.5
                
                # Normalize game total to 0-1 scale (7 = best, 11.5 = worst)
                home_total_score = max(0, min(1, (11.5 - game_total) / 4.5))
                away_total_score = max(0, min(1, (11.5 - game_total) / 4.5))
                
                # Combined run environment score (0-10 scale)
                # 60% weight on game total (lower is better), 40% on win prob (higher is better)
                home_run_env = round((home_total_score * 6) + (home_win_prob * 4), 2)
                away_run_env = round((away_total_score * 6) + (away_win_prob * 4), 2)
                
                result[home_norm] = {
                    "implied_runs": g.implied_home_runs if g.implied_home_runs is not None else 4.5,
                    "is_home": True,
                    "opponent": away_norm,
                    "park_factor": g.park_factor,
                    # Run environment for pitcher streaming (PR-22)
                    "win_prob": home_win_prob,
                    "game_total": game_total,
                    "run_environment_score": home_run_env,
                }
                result[away_norm] = {
                    "implied_runs": g.implied_away_runs if g.implied_away_runs is not None else 4.5,
                    "is_home": False,
                    "opponent": home_norm,
                    "park_factor": g.park_factor,
                    # Run environment for pitcher streaming (PR-22)
                    "win_prob": away_win_prob,
                    "game_total": game_total,
                    "run_environment_score": away_run_env,
                }
                logger.debug(f"[BUILD_MAP] Added {home_norm} vs {away_norm} (implied: {g.implied_home_runs}, {g.implied_away_runs}, run_env: {home_run_env}/{away_run_env})")
        logger.info(f"[BUILD_MAP] Final team_odds keys: {list(result.keys())}")
        return result


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_optimizer: Optional[DailyLineupOptimizer] = None


def get_lineup_optimizer() -> DailyLineupOptimizer:
    """Get singleton optimizer instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = DailyLineupOptimizer()
    return _optimizer
