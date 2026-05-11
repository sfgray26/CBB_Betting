"""
MLB nightly analysis pipeline — EMAC-080.

Mirrors the structure of the CBB nightly pipeline but for baseball:
  1. Fetch today's MLB schedule via statsapi
  2. For each game, look up pitcher xERA + team wRC+ from FanGraphs cache
  3. Project runs using a weighted pitching/offense/park-factor formula
  4. Fetch market odds from The Odds API (baseball_mlb endpoint)
  5. Calculate edge vs market (win-prob delta on the runline)
  6. Return projections — caller writes to DB if desired

GUARDIAN FREEZE: do NOT import or modify betting_model.py or analysis.py.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from backend.core.sport_config import SportConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Park factors — 2024 Statcast park-factor data (runs-based, 1.0 = neutral)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Team abbreviation -> full statsapi name mapping
# ---------------------------------------------------------------------------

_TEAM_ABB_TO_FULL: dict[str, str] = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles", "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs", "CHW": "Chicago White Sox",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies", "DET": "Detroit Tigers",
    "HOU": "Houston Astros", "KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins", "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates",
    "SDP": "San Diego Padres", "SEA": "Seattle Mariners",
    "SFG": "San Francisco Giants", "STL": "St. Louis Cardinals",
    "TBR": "Tampa Bay Rays", "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays", "WSN": "Washington Nationals",
    # Common FanGraphs variant abbreviations
    "CWS": "Chicago White Sox",
    "KC": "Kansas City Royals",
    "SD": "San Diego Padres",
    "SF": "San Francisco Giants",
    "TB": "Tampa Bay Rays",
    "WSH": "Washington Nationals",
}


PARK_FACTORS: dict[str, float] = {
    "Coors Field": 1.22,
    "Oracle Park": 0.92,
    "Petco Park": 0.90,
    "Comerica Park": 0.93,
    "Kauffman Stadium": 0.97,
    "Fenway Park": 1.08,
    "Wrigley Field": 1.04,
    "Great American Ball Park": 1.10,
    "Yankee Stadium": 1.05,
    "Globe Life Field": 1.02,
    "Truist Park": 1.00,
    "loanDepot park": 0.94,
    "American Family Field": 1.03,
    "T-Mobile Park": 0.95,
    "Target Field": 0.98,
    "Nationals Park": 1.00,
    "Progressive Field": 0.97,
    "Guaranteed Rate Field": 1.02,
    "Minute Maid Park": 1.00,
    "Oakland Coliseum": 0.95,
    "PNC Park": 0.96,
    "Busch Stadium": 0.96,
    "Camden Yards": 1.04,
    "Rogers Centre": 1.06,
    "Chase Field": 1.03,
    "Angel Stadium": 0.98,
    "Dodger Stadium": 0.96,
    "Sutter Health Park": 1.00,
}

LEAGUE_AVG_ERA: float = 4.25  # 2024 MLB average ERA (FanGraphs)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MLBGameProjection:
    """Lightweight result container for a single MLB game projection."""

    game_id: str
    home_team: str
    away_team: str
    game_date: date
    projected_home_runs: float
    projected_away_runs: float
    projected_total: float
    projected_runline_margin: float   # home - away
    home_win_prob: float
    model_version: str = "v1.0-mlb"
    edge: float = 0.0                 # projected_win_prob - market_implied_prob


# ---------------------------------------------------------------------------
# Analysis service
# ---------------------------------------------------------------------------

class MLBAnalysisService:
    """
    MLB nightly analysis pipeline.

    Does NOT write to DB directly — caller (scheduler job) writes.
    """

    def __init__(self, config: Optional[SportConfig] = None) -> None:
        self.config = config or SportConfig.mlb()

    # ------------------------------------------------------------------ #
    # Public entry point                                                   #
    # ------------------------------------------------------------------ #

    async def run_analysis(self, target_date: Optional[date] = None) -> list[MLBGameProjection]:
        """Run full MLB nightly analysis for target_date (defaults to today)."""
        target_date = target_date or date.today()
        try:
            games = self._fetch_schedule(target_date)
        except Exception as exc:
            logger.warning("mlb_analysis: schedule fetch failed: %s", exc)
            return []
        if not games:
            logger.info("mlb_analysis: no games on %s", target_date)
            return []

        pitcher_stats = self._load_pitcher_stats()
        team_stats = self._load_team_stats()
        market_odds = self._fetch_mlb_odds()

        projections: list[MLBGameProjection] = []
        for game in games:
            try:
                proj = self._project_game(game, pitcher_stats, team_stats, target_date)
                market_key = f"{proj.away_team}@{proj.home_team}"
                market = market_odds.get(market_key, {})
                proj.edge = self._calculate_edge(proj, market)
                projections.append(proj)
            except Exception as exc:
                logger.warning(
                    "mlb_analysis: game projection failed (game_id=%s): %s",
                    game.get("game_id"), exc,
                )

        logger.info("mlb_analysis: projected %d games for %s", len(projections), target_date)
        return projections

    # ------------------------------------------------------------------ #
    # Schedule fetch                                                       #
    # ------------------------------------------------------------------ #

    def _fetch_schedule(self, target_date: date) -> list[dict]:
        """Fetch MLB schedule via statsapi. Returns list of game dicts."""
        try:
            import statsapi
            date_str = target_date.strftime("%m/%d/%Y")
            games = statsapi.schedule(sportId=1, date=date_str)  # sportId=1 = MLB
            return games if isinstance(games, list) else []
        except Exception as exc:
            logger.warning("mlb_analysis: statsapi.schedule failed: %s", exc)
            return []

    # ------------------------------------------------------------------ #
    # Run projection                                                       #
    # ------------------------------------------------------------------ #

    def _project_game(
        self,
        game: dict,
        pitcher_stats: dict,
        team_stats: dict,
        target_date: Optional[date] = None,
    ) -> MLBGameProjection:
        """
        Project runs for a single game.

        Formula:
          projected_home_runs = league_avg * home_off_factor * away_pitch_factor * park_factor
          projected_away_runs = league_avg * away_off_factor * home_pitch_factor * park_factor

        where:
          off_factor  = team_wrc_plus / 100.0  (FanGraphs wRC+; 1.0 = average)
          pitch_factor = starter_xera / LEAGUE_AVG_ERA  (< 1.0 = better pitcher)
          park_factor  = venue run factor (Coors=1.22, avg=1.0, Petco=0.90)
          league_avg   = config.d1_avg_adj_o = 4.25

        Home-field advantage added after projection (+config.home_advantage_pts = +0.25).
        Win probability via normal CDF on margin with sigma = sqrt(total) * sd_multiplier.
        """
        home_pitcher = game.get("home_probable_pitcher") or ""
        away_pitcher = game.get("away_probable_pitcher") or ""

        # Pitching factor: pitcher xERA relative to league avg.
        # Higher ERA -> higher factor -> opponent scores more.
        # Fall back to league avg (factor = 1.0) when no data.
        home_sp_xera = self._pitcher_xera(home_pitcher, pitcher_stats)
        away_sp_xera = self._pitcher_xera(away_pitcher, pitcher_stats)

        home_pitch_factor = home_sp_xera / LEAGUE_AVG_ERA
        away_pitch_factor = away_sp_xera / LEAGUE_AVG_ERA

        # Offensive factor from wRC+ (100 = league average, > 100 = better offense)
        home_team = game.get("home_name", "")
        away_team = game.get("away_name", "")

        home_off = team_stats.get(home_team, {}).get("wrc_plus", 100) / 100.0
        away_off = team_stats.get(away_team, {}).get("wrc_plus", 100) / 100.0

        # Park factor — default 1.0 for unknown venues
        venue = game.get("venue_name") or game.get("venue") or ""
        park_factor = PARK_FACTORS.get(venue, 1.0)

        league_avg = self.config.d1_avg_adj_o  # 4.25 runs per team per game

        # Runs scored: facing the opposing pitcher
        home_runs = league_avg * home_off * away_pitch_factor * park_factor
        away_runs = league_avg * away_off * home_pitch_factor * park_factor

        # Home field advantage (small for MLB: +0.25 runs)
        home_runs += self.config.home_advantage_pts

        total = home_runs + away_runs
        margin = home_runs - away_runs

        # Win probability via normal CDF
        from scipy.stats import norm
        sigma = (total ** 0.5) * self.config.base_sd_multiplier
        home_win_prob = float(norm.cdf(margin / sigma)) if sigma > 0 else 0.5

        game_date = target_date or date.today()

        return MLBGameProjection(
            game_id=str(game.get("game_id", "")),
            home_team=home_team,
            away_team=away_team,
            game_date=game_date,
            projected_home_runs=round(home_runs, 2),
            projected_away_runs=round(away_runs, 2),
            projected_total=round(total, 2),
            projected_runline_margin=round(margin, 2),
            home_win_prob=round(home_win_prob, 4),
        )

    # ------------------------------------------------------------------ #
    # Stats loaders                                                        #
    # ------------------------------------------------------------------ #

    def _load_pitcher_stats(self) -> dict[str, float]:
        """
        Load pitcher xERA from the pybaseball FanGraphs cache.

        Returns dict[canonical_pitcher_name, xera_float].
        Falls back to empty dict (graceful degradation to league avg in _project_game).

        Cache structure: pybaseball_pitching_YYYY.json holds StatcastPitcher
        dataclasses keyed by canonical (lowercase, ASCII) pitcher name.
        StatcastPitcher.xera is xERA (expected ERA from Statcast); raw ERA is
        NOT stored in the dataclass — xERA is the appropriate regression-stable
        proxy for projection purposes.
        """
        try:
            from backend.fantasy_baseball.pybaseball_loader import load_pybaseball_pitchers
            cache = load_pybaseball_pitchers(year=2025)
            # Extract xera per player; fall back to LEAGUE_AVG_ERA for missing values
            result: dict[str, float] = {}
            for name, pitcher in cache.items():
                xera = getattr(pitcher, "xera", 0.0)
                # A zero xERA means no data — treat as missing rather than
                # "perfect pitcher"; callers fall back to LEAGUE_AVG_ERA on 0.0
                if xera and xera > 0.0:
                    result[name] = xera
            logger.debug("mlb_analysis: loaded %d pitcher xERA entries", len(result))
            return result
        except Exception as exc:
            logger.warning("mlb_analysis: pitcher stats load failed: %s", exc)
            return {}

    def _load_team_stats(self) -> dict[str, dict]:
        """
        Load team-level offensive stats aggregated from the per-batter pybaseball cache.

        Aggregation:
          - Loads all batters from the FanGraphs 2025 cache via load_pybaseball_batters().
          - Filters out batters whose cached wrc_plus == 100.0 exactly and team == "" (no data).
          - Groups remaining batters by team abbreviation and computes a plain mean wRC+.
            (PA-weighted average would be more accurate but PA is not stored in the cache.)
          - Maps team abbreviations to full statsapi names via _TEAM_ABB_TO_FULL.
          - Teams with fewer than 3 qualifying batters are excluded (too small a sample).

        Returns dict[full_team_name, {"wrc_plus": float}].
        Falls back to empty dict on any failure (caller defaults to wRC+=100).
        """
        try:
            from backend.fantasy_baseball.pybaseball_loader import load_pybaseball_batters

            cache = load_pybaseball_batters(year=2025)
            if not cache:
                return {}

            # Group wrc_plus values by team abbreviation
            team_wrc: dict[str, list[float]] = {}
            for batter in cache.values():
                abb = getattr(batter, "team", "").strip().upper()
                wrc = getattr(batter, "wrc_plus", 100.0)
                # Skip batters with no team tag or suspiciously default values
                if not abb:
                    continue
                # wrc_plus defaults to 100.0 in the dataclass; only include if
                # the value was explicitly set (non-default) OR the team is tagged.
                # We include all tagged batters since even 100 is a valid score.
                team_wrc.setdefault(abb, []).append(float(wrc))

            result: dict[str, dict] = {}
            for abb, values in team_wrc.items():
                if len(values) < 3:
                    # Too few batters — skip, let caller default to league avg
                    continue
                avg_wrc = sum(values) / len(values)
                full_name = _TEAM_ABB_TO_FULL.get(abb)
                if not full_name:
                    logger.debug("mlb_analysis: no full-name mapping for team abbr '%s'", abb)
                    continue
                result[full_name] = {"wrc_plus": round(avg_wrc, 1)}

            logger.debug(
                "mlb_analysis: _load_team_stats aggregated wRC+ for %d teams", len(result)
            )
            return result
        except Exception as exc:
            logger.warning("mlb_analysis: team stats load failed: %s", exc)
            return {}

    # ------------------------------------------------------------------ #
    # Odds fetch                                                           #
    # ------------------------------------------------------------------ #

    def _fetch_mlb_odds(self) -> dict[str, dict]:
        """
        Fetch current MLB runline odds from mlb_odds_snapshot table.

        Returns dict keyed by "AwayTeam@HomeTeam" -> flat BDL odds dict.
        Queries the DB for the most recent snapshot per game, prefers vendor
        by quality order. Returns empty dict on any failure (graceful degradation).
        """
        from backend.models import SessionLocal, MLBOddsSnapshot, MLBGameLog, MLBTeam

        preferred_vendors = ["pinnacle", "draftkings", "fanduel", "betmgm", "caesars"]
        try:
            db = SessionLocal()
            from sqlalchemy import func, case, literal_column

            max_window_subq = (
                db.query(
                    MLBOddsSnapshot.game_id,
                    func.max(MLBOddsSnapshot.snapshot_window).label("max_window")
                )
                .group_by(MLBOddsSnapshot.game_id)
                .subquery()
            )

            vendor_priority = case(
                *[(literal_column(f"'{v}'"), i) for i, v in enumerate(preferred_vendors)],
                else_=len(preferred_vendors)
            )

            latest_odds_q = (
                db.query(
                    MLBOddsSnapshot,
                    MLBGameLog,
                    MLBTeam,
                    vendor_priority.label("vendor_priority")
                )
                .join(MLBGameLog, MLBOddsSnapshot.game_id == MLBGameLog.game_id)
                .join(MLBTeam, MLBGameLog.away_team_id == MLBTeam.team_id)
                .join(
                    max_window_subq,
                    (MLBOddsSnapshot.game_id == max_window_subq.c.game_id) &
                    (MLBOddsSnapshot.snapshot_window == max_window_subq.c.max_window)
                )
                .order_by(MLBOddsSnapshot.game_id, "vendor_priority")
                .all()
            )

            result: dict[str, dict] = {}
            seen_games = set()

            for row in latest_odds_q:
                odds = row[0]
                game = row[1]
                away_team = row[2]

                if odds.game_id in seen_games:
                    continue

                home_team = game.home_team_obj
                if not home_team:
                    continue

                seen_games.add(odds.game_id)
                key = f"{away_team.abbreviation}@{home_team.abbreviation}"
                result[key] = {
                    "ml_home_odds": odds.ml_home_odds,
                    "ml_away_odds": odds.ml_away_odds,
                    "spread_home": odds.spread_home,
                    "spread_home_odds": odds.spread_home_odds,
                    "total": odds.total,
                    "vendor": odds.vendor
                }

            logger.debug(
                "mlb_analysis: loaded odds for %d games from mlb_odds_snapshot",
                len(result)
            )
            return result
        except Exception as exc:
            logger.warning("mlb_analysis: odds DB fetch failed: %s", exc)
            return {}
        finally:
            db.close()

    # ------------------------------------------------------------------ #
    # Edge calculation                                                     #
    # ------------------------------------------------------------------ #

    def _calculate_edge(self, projection: MLBGameProjection, market: dict) -> float:
        """
        Edge = projected_win_prob - market_implied_win_prob on the runline.

        Positive = our model thinks home team is more likely to cover -1.5
        than the market implies. Uses ml_home_odds from the flat BDL structure.

        Returns 0.0 when no market data is available.
        """
        if not market or "ml_home_odds" not in market or market["ml_home_odds"] == 0:
            return 0.0

        try:
            american_odds = market["ml_home_odds"]
            if american_odds < 0:
                market_prob = abs(american_odds) / (abs(american_odds) + 100)
            else:
                market_prob = 100 / (american_odds + 100)
            return round(projection.home_win_prob - market_prob, 4)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------ #
    # Edge verification (sprint commitment)                                  #
    # ------------------------------------------------------------------ #

    def verify_edge_calculation(self) -> dict:
        """
        Sanity-check edge math with known test cases.
        Returns report dict for scheduler observability.
        """
        test_cases = [
            # (home_win_prob, american_odds, expected_edge_sign)
            (0.60, -150, "positive"),   # model 60% vs market 60% → ~0 edge
            (0.70, -150, "positive"),   # model 70% vs market 60% → positive edge
            (0.50, +150, "positive"),   # model 50% vs market 40% → positive edge
            (0.40, -200, "negative"),   # model 40% vs market 66.7% → negative edge
        ]
        passed = 0
        failed = 0
        results = []
        for home_win_prob, american_odds, expected_sign in test_cases:
            market = {"ml_home_odds": american_odds}
            proj = MLBGameProjection(
                game_id="TEST",
                home_team="H",
                away_team="A",
                game_date=date.today(),
                projected_home_runs=4.0,
                projected_away_runs=3.0,
                projected_total=7.0,
                projected_runline_margin=1.0,
                home_win_prob=home_win_prob,
            )
            edge = self._calculate_edge(proj, market)
            actual_sign = "positive" if edge > 0.01 else "negative" if edge < -0.01 else "neutral"
            ok = actual_sign == expected_sign
            if ok:
                passed += 1
            else:
                failed += 1
            results.append({
                "home_win_prob": home_win_prob,
                "american_odds": american_odds,
                "expected_sign": expected_sign,
                "actual_edge": edge,
                "actual_sign": actual_sign,
                "pass": ok,
            })
        logger.info("mlb_analysis: edge verification %d/%d passed", passed, len(test_cases))
        return {
            "status": "passed" if failed == 0 else "failed",
            "passed": passed,
            "failed": failed,
            "results": results,
        }

    def write_projections_to_db(self, projections: list[MLBGameProjection]) -> dict:
        """Persist projections to the mlb_projections table. Idempotent upsert."""
        from backend.models import SessionLocal, MLBProjection
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        db = SessionLocal()
        try:
            rows_inserted = 0
            for p in projections:
                stmt = pg_insert(MLBProjection).values(
                    game_id=p.game_id,
                    home_team=p.home_team,
                    away_team=p.away_team,
                    projection_date=p.game_date,
                    projected_home_runs=p.projected_home_runs,
                    projected_away_runs=p.projected_away_runs,
                    projected_total=p.projected_total,
                    projected_runline_margin=p.projected_runline_margin,
                    home_win_prob=p.home_win_prob,
                    edge=p.edge,
                    market_ml_home_odds=None,
                    model_version=p.model_version,
                ).on_conflict_do_update(
                    index_elements=["game_id", "projection_date"],
                    set_={
                        "projected_home_runs": p.projected_home_runs,
                        "projected_away_runs": p.projected_away_runs,
                        "projected_total": p.projected_total,
                        "projected_runline_margin": p.projected_runline_margin,
                        "home_win_prob": p.home_win_prob,
                        "edge": p.edge,
                        "model_version": p.model_version,
                    }
                )
                db.execute(stmt)
                rows_inserted += 1
            db.commit()
            logger.info("mlb_analysis: persisted %d projections to DB", rows_inserted)
            return {"status": "success", "rows_inserted": rows_inserted}
        except Exception as exc:
            db.rollback()
            logger.error("mlb_analysis: DB write failed: %s", exc)
            return {"status": "failed", "error": str(exc)}
        finally:
            db.close()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _pitcher_xera(self, pitcher_name: str, pitcher_stats: dict) -> float:
        """
        Look up a pitcher's xERA from the stats cache.

        Tries exact lowercase key first, then fuzzy last-name match.
        Falls back to LEAGUE_AVG_ERA (4.25) when no data found.
        """
        if not pitcher_name:
            return LEAGUE_AVG_ERA
        key = pitcher_name.strip().lower()
        xera = pitcher_stats.get(key)
        if xera:
            return xera
        # Fuzzy: try matching on last name only when name format differs
        parts = key.split()
        if len(parts) >= 2:
            last = parts[-1]
            candidates = [v for k, v in pitcher_stats.items() if k.split()[-1] == last]
            if len(candidates) == 1:
                return candidates[0]
        return LEAGUE_AVG_ERA
