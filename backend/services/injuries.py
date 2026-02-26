"""
Injury scraping and roster availability service.

Provides a real-time injury matrix so the model never trades blind
into a market that has already priced in a key absence.

Sources (in priority order):
    1. Manual overrides via API / database
    2. CBS Sports college basketball injury page (primary scrape source)

Impact tiers quantify the expected margin swing when a player
is out, based on their usage rate and team depth.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InjuryReport:
    """Single player injury entry."""

    team: str
    player: str
    status: str  # "Out", "Doubtful", "Questionable", "Probable"
    impact_tier: str  # "star", "starter", "role", "bench"
    usage_rate: Optional[float] = None  # 0-100 scale if known
    margin_impact: float = 0.0  # Estimated point swing
    source: str = "manual"
    updated_at: Optional[datetime] = None


@dataclass
class TeamInjuryMatrix:
    """Aggregate injury state for a single team."""

    team: str
    injuries: List[InjuryReport] = field(default_factory=list)
    total_margin_impact: float = 0.0
    confidence: str = "low"  # "high" if from live feed, "low" if stale/manual

    def recalculate(self) -> None:
        """Sum up individual margin impacts, weighting by status probability."""
        status_weights = {
            "Out": 1.0,
            "Doubtful": 0.75,
            "Questionable": 0.40,
            "Probable": 0.10,
        }
        self.total_margin_impact = sum(
            inj.margin_impact * status_weights.get(inj.status, 0.5)
            for inj in self.injuries
        )


# ---------------------------------------------------------------------------
# Impact estimation
# ---------------------------------------------------------------------------

# Default margin impact by tier (points, when the player is fully OUT).
# Derived from replacement-level analysis of CBB usage rates.
TIER_IMPACT = {
    "star": 3.5,       # > 28% usage, primary creator
    "starter": 1.8,    # Regular starter, 18-28% usage
    "role": 0.7,       # Rotation player, 10-18% usage
    "bench": 0.2,      # Deep bench, < 10% usage
}


def estimate_impact(tier: str, usage_rate: Optional[float] = None) -> float:
    """Estimate the margin swing (points) from a player being out.

    When usage_rate is provided, scales the tier base by the player's usage
    relative to the D1 starter average (~22%).  The multiplier is clamped to
    [0.5, 1.8] so outlier usage rates never produce absurd point swings.

    Examples:
        star (3.5 base) at 22% usage  → 3.5 * 1.00 = 3.50
        star (3.5 base) at 35% usage  → 3.5 * 1.59 = 5.57
        star (3.5 base) at 44%+ usage → 3.5 * 1.80 = 6.30  (capped)
        star (3.5 base) at 10% usage  → 3.5 * 0.50 = 1.75  (floored)
    """
    base = TIER_IMPACT.get(tier, 0.2)
    if usage_rate is not None and usage_rate > 0:
        D1_STARTER_USAGE = 22.0
        multiplier = min(1.8, max(0.5, usage_rate / D1_STARTER_USAGE))
        return base * multiplier
    return base


def classify_tier(usage_rate: Optional[float] = None) -> str:
    """Auto-classify injury tier from usage rate."""
    if usage_rate is None:
        return "role"
    if usage_rate >= 28:
        return "star"
    if usage_rate >= 18:
        return "starter"
    if usage_rate >= 10:
        return "role"
    return "bench"


# ---------------------------------------------------------------------------
# Scrapers
# ---------------------------------------------------------------------------

def _normalize_status(raw: str) -> Optional[str]:
    """
    Map a raw CBS Sports status string to one of our four canonical values.

    Returns None for statuses that indicate a healthy player (Active, Full,
    Day-To-Day with no imminent game risk) so callers can skip them cleanly.

    CBS Sports uses these status labels (non-exhaustive):
        Out, Doubtful, Questionable, Probable, GTD (Game Time Decision),
        Active, Full, Expected, Healthy
    """
    s = raw.strip().lower()
    if not s:
        return None
    if "out" in s:
        return "Out"
    if "doubtful" in s:
        return "Doubtful"
    # "GTD" and "game time" are operationally equivalent to Questionable.
    if "questionable" in s or "gtd" in s or "game time" in s:
        return "Questionable"
    if "probable" in s:
        return "Probable"
    # Healthy / Active / Full — not a meaningful injury for the model.
    return None


def scrape_injuries() -> List[InjuryReport]:
    """
    Scrape college basketball injury reports from CBS Sports.

    Target: https://www.cbssports.com/college-basketball/injuries/

    CBS Sports renders each team's injury data in a ``div.TableBaseWrapper``
    block that contains:
      - A heading element (``h3.TableBase-title`` or ``h4``) with the team
        name, optionally wrapped in an anchor tag.
      - A ``TableBase-table`` table whose body rows follow the column order:
        Player (0) | Pos (1) | Injury (2) | Status (3) | Expected Return (4)

    A secondary CSS selector strategy (``div.PlayerInjuries``) is tried when
    the primary TableBaseWrapper pattern yields zero teams, providing forward
    compatibility if CBS restructures their layout.

    Output format is identical to the previous ESPN scraper so no callers
    need to change: a list of ``InjuryReport`` objects with the same fields.
    The ``source`` field changes from ``"espn"`` to ``"cbssports"``.

    Returns an empty list on any HTTP or parse failure; callers should apply
    a stale-injury penalty when the list is empty.
    """
    url = "https://www.cbssports.com/college-basketball/injuries/"
    headers = {
        # Modern Chrome UA is required — CBS Sports aggressively blocks default
        # Python requests UAs and older browser strings.
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8,"
            "application/signed-exchange;v=b3;q=0.7"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.cbssports.com/",
        "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Upgrade-Insecure-Requests": "1",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        # 404 / 4xx: endpoint moved or Cloudflare-blocked — suppress recurring
        # log spam by using INFO rather than WARNING.
        logger.info(
            "CBS Sports injury scrape HTTP error (endpoint may have moved): %s", exc
        )
        return []
    except requests.RequestException as exc:
        # Network-level failures are transient and warrant operator attention.
        logger.warning("CBS Sports injury scrape failed (network error): %s", exc)
        return []

    injuries: List[InjuryReport] = []
    soup = BeautifulSoup(resp.text, "lxml")

    # ---- Primary strategy: TableBaseWrapper blocks (standard CBS layout) ---
    wrappers = soup.select("div.TableBaseWrapper")

    # ---- Secondary strategy: PlayerInjuries sections (alt CBS layout) ------
    if not wrappers:
        wrappers = soup.select("section.PlayerInjuries, div.PlayerInjuries")

    for wrapper in wrappers:
        # Extract team name from the nearest heading / link element.
        # CBS uses a few different patterns depending on the sport/page:
        #   <h3 class="TableBase-title"><a href="...">Duke Blue Devils</a></h3>
        #   <span class="TeamName">Duke Blue Devils</span>
        #   <h4 class="team-name">Duke Blue Devils</h4>
        team_el = wrapper.select_one(
            "h3.TableBase-title a, h3.TableBase-title, "
            "span.TeamName, .PlayerInjuries-teamName, "
            "h4, h3, h2"
        )
        if team_el is None:
            continue
        team_name = team_el.get_text(strip=True)
        if not team_name:
            continue

        # CBS marks data rows with class "TableBase-bodyTr".  Try that selector
        # first (precise), fall back to generic tbody tr for forward-compat.
        data_rows = (
            wrapper.select("tr.TableBase-bodyTr")
            or wrapper.select("tbody tr")
        )
        for row in data_rows:
            cells = row.select("td")
            # Expect at minimum: Player | Pos | Injury | Status
            if len(cells) < 4:
                continue

            player = cells[0].get_text(strip=True)
            # Skip any header rows that sneak into tbody
            if not player or player.lower() in ("player", "name", "athlete"):
                continue

            # CBS column order: Player(0) Pos(1) Injury(2) Status(3) Return(4)
            # Also check for an explicit status class in case CBS highlights
            # the status cell with a dedicated element (e.g. injury-status span).
            status_el = (
                cells[3].select_one(".injury-status, [class*='status']")
                or cells[3]
            )
            status_raw = status_el.get_text(strip=True)

            status = _normalize_status(status_raw)
            if status is None:
                # Healthy / Active — not relevant for the model.
                continue

            tier = "role"  # Default; enriched with usage data in later pipeline stages
            impact = estimate_impact(tier)

            injuries.append(
                InjuryReport(
                    team=team_name,
                    player=player,
                    status=status,
                    impact_tier=tier,
                    margin_impact=impact,
                    source="cbssports",
                    updated_at=datetime.utcnow(),
                )
            )

    if not injuries and wrappers:
        # Page loaded but no rows parsed — log the first 500 chars for debugging.
        logger.warning(
            "CBS Sports injury page loaded (%d wrapper blocks) but no injury "
            "rows were extracted. Page snippet: %.500s",
            len(wrappers), resp.text,
        )
    else:
        logger.info(
            "CBS Sports injury scrape: %d entries across teams", len(injuries)
        )
    return injuries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class InjuryService:
    """Aggregates injury data from all sources and builds per-game matrices."""

    def __init__(self):
        self._cache: List[InjuryReport] = []
        self._cache_time: Optional[datetime] = None
        self._manual_overrides: List[InjuryReport] = []

    def add_manual_override(self, report: InjuryReport) -> None:
        """Add or update a manual injury entry (highest priority)."""
        # Remove existing entry for same team+player
        self._manual_overrides = [
            r
            for r in self._manual_overrides
            if not (r.team == report.team and r.player == report.player)
        ]
        report.source = "manual"
        report.margin_impact = estimate_impact(report.impact_tier, report.usage_rate)
        self._manual_overrides.append(report)

    def fetch_injuries(self, max_age_minutes: int = 30) -> List[InjuryReport]:
        """Return current injury data, refreshing from scrapers if stale."""
        now = datetime.utcnow()
        if (
            self._cache_time
            and (now - self._cache_time).total_seconds() < max_age_minutes * 60
        ):
            return self._merge_with_overrides(self._cache)

        scraped = scrape_injuries()
        if scraped:
            self._cache = scraped
            self._cache_time = now
        else:
            logger.warning("Injury cache stale — scrape returned 0 entries")

        return self._merge_with_overrides(self._cache)

    def get_game_injuries(
        self,
        home_team: str,
        away_team: str,
        max_age_minutes: int = 30,
    ) -> List[Dict]:
        """
        Get injury reports formatted for ``CBBEdgeModel.analyze_game()``.

        Returns a list of dicts with keys: team, player, impact_tier,
        status, margin_impact.
        """
        all_injuries = self.fetch_injuries(max_age_minutes)

        game_injuries = []
        for inj in all_injuries:
            # Match by team name (case-insensitive substring)
            inj_team_lower = inj.team.lower()
            if home_team.lower() in inj_team_lower or inj_team_lower in home_team.lower():
                matched_team = home_team
            elif away_team.lower() in inj_team_lower or inj_team_lower in away_team.lower():
                matched_team = away_team
            else:
                continue

            # Only include players actually expected to miss the game
            if inj.status in ("Out", "Doubtful", "Questionable"):
                game_injuries.append(
                    {
                        "team": matched_team,
                        "player": inj.player,
                        "impact_tier": inj.impact_tier,
                        "status": inj.status,
                        "margin_impact": inj.margin_impact,
                    }
                )

        return game_injuries

    def _merge_with_overrides(
        self, base: List[InjuryReport]
    ) -> List[InjuryReport]:
        """Manual overrides take priority over scraped data."""
        override_keys = {
            (r.team.lower(), r.player.lower()) for r in self._manual_overrides
        }
        merged = [
            r
            for r in base
            if (r.team.lower(), r.player.lower()) not in override_keys
        ]
        merged.extend(self._manual_overrides)
        return merged


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_injury_service: Optional[InjuryService] = None


def get_injury_service() -> InjuryService:
    global _injury_service
    if _injury_service is None:
        _injury_service = InjuryService()
    return _injury_service
