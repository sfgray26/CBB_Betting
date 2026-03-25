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
        Load team-level offensive stats.

        wRC+ is not stored per-team in the current FanGraphs cache (only per-player).
        This method returns an empty dict for now — the projection formula
        defaults to wrc_plus=100 (league average offense) for all teams.

        ADR note: wRC+ team aggregation is a planned enhancement (EMAC-081).
        A future version will aggregate per-batter wRC+ into a team lineup score
        using the FanGraphs team batting leaderboard via pybaseball.team_batting().
        """
        return {}

    # ------------------------------------------------------------------ #
    # Odds fetch                                                           #
    # ------------------------------------------------------------------ #

    def _fetch_mlb_odds(self) -> dict[str, dict]:
        """
        Fetch current MLB runline odds from The Odds API.

        Returns dict keyed by "AwayTeam@HomeTeam" -> full bookmaker payload.
        Reuses the same baseball_mlb endpoint as daily_ingestion.py.
        Returns empty dict on any failure (graceful degradation).
        """
        api_key = os.getenv("THE_ODDS_API_KEY")
        if not api_key:
            return {}
        try:
            import requests
            resp = requests.get(
                "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds",
                params={
                    "apiKey": api_key,
                    "regions": os.getenv("ODDS_API_REGIONS", "us,eu"),
                    "markets": "spreads,totals,h2h",
                    "oddsFormat": "american",
                },
                timeout=10,
            )
            resp.raise_for_status()
            games = resp.json()
            result: dict[str, dict] = {}
            for g in (games if isinstance(games, list) else []):
                key = f"{g.get('away_team', '')}@{g.get('home_team', '')}"
                result[key] = g
            return result
        except Exception as exc:
            logger.warning("mlb_analysis: odds fetch failed: %s", exc)
            return {}

    # ------------------------------------------------------------------ #
    # Edge calculation                                                     #
    # ------------------------------------------------------------------ #

    def _calculate_edge(self, projection: MLBGameProjection, market: dict) -> float:
        """
        Edge = projected_win_prob - market_implied_win_prob on the runline.

        Positive = our model thinks home team is more likely to cover -1.5
        than the market implies. Uses the first bookmaker's spreads market
        for the home team outcome.

        Returns 0.0 when no market data is available.
        """
        if not market:
            return 0.0
        try:
            for bookmaker in market.get("bookmakers", []):
                for mkt in bookmaker.get("markets", []):
                    if mkt.get("key") == "spreads":
                        for outcome in mkt.get("outcomes", []):
                            if outcome.get("name") == projection.home_team:
                                american_odds = outcome.get("price", -110)
                                if american_odds < 0:
                                    market_prob = abs(american_odds) / (abs(american_odds) + 100)
                                else:
                                    market_prob = 100 / (american_odds + 100)
                                return round(projection.home_win_prob - market_prob, 4)
        except Exception:
            pass
        return 0.0

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
