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

import requests

from backend.utils.env_utils import get_float_env

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The Odds API — MLB games
# ---------------------------------------------------------------------------
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
MLB_SPORT = "baseball_mlb"

# MLB team abbreviation -> full name normalization map (Odds API uses full names)
_TEAM_ABBREV = {
    "NYY": "New York Yankees", "BOS": "Boston Red Sox", "TOR": "Toronto Blue Jays",
    "BAL": "Baltimore Orioles", "TBR": "Tampa Bay Rays", "TB": "Tampa Bay Rays",
    "CLE": "Cleveland Guardians", "CWS": "Chicago White Sox", "DET": "Detroit Tigers",
    "KCR": "Kansas City Royals", "KC": "Kansas City Royals", "MIN": "Minnesota Twins",
    "HOU": "Houston Astros", "TEX": "Texas Rangers", "SEA": "Seattle Mariners",
    "OAK": "Oakland Athletics", "LAA": "Los Angeles Angels",
    "NYM": "New York Mets", "PHI": "Philadelphia Phillies", "ATL": "Atlanta Braves",
    "MIA": "Miami Marlins", "WSH": "Washington Nationals", "WSN": "Washington Nationals",
    "MIL": "Milwaukee Brewers", "CHC": "Chicago Cubs", "STL": "St. Louis Cardinals",
    "CIN": "Cincinnati Reds", "PIT": "Pittsburgh Pirates",
    "LAD": "Los Angeles Dodgers", "SFG": "San Francisco Giants", "SF": "San Francisco Giants",
    "SDP": "San Diego Padres", "SD": "San Diego Padres", "COL": "Colorado Rockies",
    "ARI": "Arizona Diamondbacks", "AZ": "Arizona Diamondbacks",
}
# Reverse: full name -> abbreviation
_FULL_TO_ABBREV: Dict[str, str] = {v: k for k, v in _TEAM_ABBREV.items()}

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


class DailyLineupOptimizer:
    """
    Combines sportsbook odds with projection data to rank
    batters and pitchers for daily fantasy lineup decisions.
    """

    def __init__(self):
        self._api_key = os.getenv("THE_ODDS_API_KEY", "")
        self._odds_cache: Dict[str, List[MLBGameOdds]] = {}

    # ------------------------------------------------------------------
    # Odds fetching
    # ------------------------------------------------------------------

    def fetch_mlb_odds(self, game_date: Optional[str] = None) -> List[MLBGameOdds]:
        """
        Fetch today's MLB game odds from The Odds API.

        Returns list of MLBGameOdds with implied run totals computed.
        Falls back to empty list if API key missing or request fails.
        """
        if not self._api_key:
            logger.warning("THE_ODDS_API_KEY not set — lineup optimizer running without odds data")
            return []

        cache_key = game_date or "today"
        if cache_key in self._odds_cache:
            return self._odds_cache[cache_key]

        try:
            resp = requests.get(
                f"{ODDS_API_BASE}/sports/{MLB_SPORT}/odds",
                params={
                    "apiKey": self._api_key,
                    "regions": "us",
                    "markets": "spreads,totals,h2h",
                    "oddsFormat": "american",
                    "commenceTimeTo": f"{game_date}T23:59:59Z" if game_date else None,
                    "commenceTimeFrom": f"{game_date}T00:00:00Z" if game_date else None,
                },
                timeout=10,
            )
            if resp.status_code != 200:
                logger.warning("Odds API returned %d for MLB odds", resp.status_code)
                return []

            games_raw = resp.json()
            games = []
            for g in games_raw:
                game = self._parse_game_odds(g)
                if game:
                    games.append(game)

            self._odds_cache[cache_key] = games
            logger.info("Fetched %d MLB games from Odds API for %s", len(games), game_date or "today")
            return games

        except Exception as exc:
            logger.warning("Failed to fetch MLB odds: %s", exc)
            return []

    def _parse_game_odds(self, raw: dict) -> Optional[MLBGameOdds]:
        """Parse raw Odds API game dict into MLBGameOdds."""
        home_name = raw.get("home_team", "")
        away_name = raw.get("away_team", "")
        home_abbrev = _FULL_TO_ABBREV.get(home_name, home_name[:3].upper())
        away_abbrev = _FULL_TO_ABBREV.get(away_name, away_name[:3].upper())

        game = MLBGameOdds(
            game_id=raw.get("id", ""),
            commence_time=raw.get("commence_time", ""),
            home_team=home_name,
            away_team=away_name,
            home_abbrev=home_abbrev,
            away_abbrev=away_abbrev,
            park_factor=_PARK_FACTORS.get(home_abbrev, 1.0),
        )

        # Parse bookmaker odds — prefer DraftKings > FanDuel > first available
        bookmakers = raw.get("bookmakers", [])
        preferred_order = ["draftkings", "fanduel", "bovada"]
        bm_data = None
        for pref in preferred_order:
            bm_data = next((b for b in bookmakers if b.get("key") == pref), None)
            if bm_data:
                break
        if not bm_data and bookmakers:
            bm_data = bookmakers[0]

        if bm_data:
            for market in bm_data.get("markets", []):
                mtype = market.get("key")
                outcomes = market.get("outcomes", [])
                if mtype == "totals" and outcomes:
                    # Find "Over" — total is the point value
                    for o in outcomes:
                        if o.get("name") == "Over":
                            game.total = float(o.get("point", 0))
                            break
                elif mtype == "spreads":
                    for o in outcomes:
                        if o.get("name") == home_name:
                            game.spread_home = float(o.get("point", 0))
                            break
                elif mtype == "h2h":
                    for o in outcomes:
                        if o.get("name") == home_name:
                            game.moneyline_home = float(o.get("price", 0))
                        elif o.get("name") == away_name:
                            game.moneyline_away = float(o.get("price", 0))

        # Compute implied team runs
        if game.total is not None:
            game.implied_home_runs, game.implied_away_runs = self._implied_runs(
                game.total, game.spread_home or 0.0
            )

        return game

    @staticmethod
    def _implied_runs(total: float, spread_home: float) -> Tuple[float, float]:
        """
        Convert game total + spread to per-team implied runs.

        Spread reflects run differential, so:
          home_runs = (total - spread_home) / 2 + spread_home
                    = (total + spread_home) / 2
          away_runs = total - home_runs

        spread_home is negative when home team is favored (e.g., -1.5).
        """
        home_runs = (total + spread_home) / 2.0
        away_runs = total - home_runs
        # Clamp to realistic range
        home_runs = max(1.0, min(12.0, home_runs))
        away_runs = max(1.0, min(12.0, away_runs))
        return round(home_runs, 2), round(away_runs, 2)

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

        proj_by_name = {p["name"].lower(): p for p in projections if p.get("player_type") == "batter"}

        rankings = []
        for player in roster:
            positions = player.get("positions", [])
            # Skip pitchers and bench/IL
            if all(p in ("SP", "RP", "P") for p in positions):
                continue
            status = player.get("status")
            if status in ("IL", "IL60", "NA"):
                continue

            name = player.get("name", "")
            team = player.get("team", "")
            proj = proj_by_name.get(name.lower(), {})

            # Get team's implied runs from odds
            odds_data = team_odds.get(team, {})
            implied_runs = odds_data.get("implied_runs", 4.5)   # league avg fallback
            is_home = odds_data.get("is_home", False)
            park_factor = odds_data.get("park_factor", 1.0)

            # Composite lineup score
            # Weights: implied_runs (environment) + projected stats
            base_score = implied_runs * park_factor
            stat_bonus = (
                proj.get("hr", 0) * 2.0
                + proj.get("r", 0) * 0.3
                + proj.get("rbi", 0) * 0.3
                + proj.get("nsb", 0) * 0.5
                + proj.get("avg", 0.250) * 5.0
            )
            lineup_score = base_score + stat_bonus * 0.1

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

        proj_by_name = {p["name"].lower(): p for p in projections if p.get("player_type") == "pitcher"}

        rankings = []
        for player in free_agents:
            status = player.get("status")
            if status in ("IL", "IL60", "NA"):
                continue
            name = player.get("name", "")
            team = player.get("team", "")
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
            game_date = datetime.utcnow().strftime("%Y-%m-%d")

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
        Build a dict: team_abbrev -> {implied_runs, is_home, opponent, park_factor}
        """
        result: Dict[str, dict] = {}
        for g in games:
            if g.implied_home_runs is not None:
                result[g.home_abbrev] = {
                    "implied_runs": g.implied_home_runs,
                    "is_home": True,
                    "opponent": g.away_abbrev,
                    "park_factor": g.park_factor,
                }
                result[g.away_abbrev] = {
                    "implied_runs": g.implied_away_runs,
                    "is_home": False,
                    "opponent": g.home_abbrev,
                    "park_factor": g.park_factor,
                }
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
