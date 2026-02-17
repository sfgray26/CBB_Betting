"""
CBB Ratings data from multiple sources.

Sources
-------
KenPom    (Official API, Bearer token) — AdjEM (adjusted efficiency margin)
BartTorvik (barttorvik.com CSV)        — adjusted efficiency column
EvanMiya   (evanmiya.com HTML table)   — BPR (Box Plus-Minus for teams)

Weights in CBBEdgeModel: KenPom 34.2%, BartTorvik 33.3%, EvanMiya 32.5%.
Missing weights are renormalized at model runtime — see betting_model.py.

EvanMiya scraper notes
-----------------------
The public evanmiya.com site renders a sortable HTML table of team BPR values.
Set EVANMIYA_URL in .env to override the endpoint if the URL changes.
If the scrape fails for any reason the function returns {} and the model
falls back to KenPom+BartTorvik with renormalized weights + a small SD penalty.

BartTorvik robustness
----------------------
Column detection uses CSV header names rather than hardcoded column indexes.
If the site restructures its CSV, the scraper warns and falls back to positional
parsing (original column 1 = team, column 5 = efficiency).
"""

import csv
import io
import logging
import os
from datetime import datetime
from typing import Dict, Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from backend.services.team_mapping import normalize_team_name

load_dotenv()

logger = logging.getLogger(__name__)

_KENPOM_URL   = "https://kenpom.com/api.php"
_CURRENT_YEAR = int(os.getenv("SEASON_YEAR", "2026"))

_BARTTORVIK_URL = os.getenv(
    "BARTTORVIK_URL",
    f"https://barttorvik.com/{_CURRENT_YEAR}_team_results.csv",
)
_EVANMIYA_URL = os.getenv("EVANMIYA_URL", "https://evanmiya.com/")


class RatingsService:
    """Multi-source ratings aggregator with automated team name matching."""

    def __init__(self):
        self.kenpom_key      = os.getenv("KENPOM_API_KEY")
        self.cache: Dict     = {}
        self.cache_timestamp: Optional[datetime] = None

    # -----------------------------------------------------------------------
    # Team name lookup
    # -----------------------------------------------------------------------

    def get_team_rating(
        self, team_name: str, source_data: Dict[str, float]
    ) -> Optional[float]:
        """Safely extract a rating using the centralised normalisation service."""
        if not source_data:
            return None

        normalized = normalize_team_name(team_name, list(source_data.keys()))
        if normalized is None:
            logger.warning("Mismatch: '%s' not found in source data.", team_name)
            return None

        rating = source_data.get(normalized)
        if rating is None:
            logger.warning(
                "Mismatch: '%s' (normalized: '%s') absent post-normalisation.",
                team_name, normalized,
            )
        return rating

    # -----------------------------------------------------------------------
    # KenPom
    # -----------------------------------------------------------------------

    def get_kenpom_ratings(self) -> Dict[str, float]:
        """Fetch AdjEM ratings from the official KenPom API."""
        if not self.kenpom_key:
            logger.warning("KenPom API key not set — skipping KenPom.")
            return {}

        headers = {
            "Authorization": f"Bearer {self.kenpom_key}",
            "Accept":        "application/json",
        }
        params = {"endpoint": "ratings", "y": _CURRENT_YEAR}

        try:
            response = requests.get(
                _KENPOM_URL, headers=headers, params=params, timeout=15
            )
            response.raise_for_status()
            data = response.json()

            ratings: Dict[str, float] = {}
            for team in data:
                name   = team.get("TeamName")
                adj_em = team.get("AdjEM")
                if name and adj_em is not None:
                    try:
                        ratings[name.strip()] = float(adj_em)
                    except (ValueError, TypeError):
                        logger.warning(
                            "Could not parse AdjEM for %s: %r", name, adj_em
                        )

            logger.info("KenPom: loaded %d teams", len(ratings))
            return ratings

        except Exception as exc:
            logger.error("KenPom API error: %s", exc)
            return {}

    # -----------------------------------------------------------------------
    # BartTorvik
    # -----------------------------------------------------------------------

    def get_barttorvik_ratings(self) -> Dict[str, float]:
        """
        Fetch adjusted efficiency ratings from the BartTorvik CSV export.

        Column detection uses header names first, falling back to original
        positional indexes (team=1, efficiency=5) so that minor CSV
        restructuring does not silently corrupt values.
        """
        url = _BARTTORVIK_URL

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            reader = csv.reader(io.StringIO(response.text))
            rows   = [row for row in reader if any(cell.strip() for cell in row)]

            if not rows:
                logger.warning("BartTorvik CSV is empty")
                return {}

            header = [h.strip().lower() for h in rows[0]]

            # Locate team-name column
            team_col: Optional[int] = None
            for candidate in ("team", "teamname", "team_name", "school"):
                if candidate in header:
                    team_col = header.index(candidate)
                    break

            # Locate efficiency column (several naming conventions used over the years)
            eff_col: Optional[int] = None
            for candidate in ("adjoe", "adj_oe", "adjem", "adj_em", "barthag",
                               "adj oe", "adj em", "eff margin"):
                if candidate in header:
                    eff_col = header.index(candidate)
                    break

            # Fallback to hard-coded positions
            if team_col is None:
                team_col = 1
                logger.debug("BartTorvik: team column not in header; using index 1")
            if eff_col is None:
                eff_col = 5
                logger.debug("BartTorvik: eff column not in header; using index 5")

            ratings: Dict[str, float] = {}
            for row in rows[1:]:
                if len(row) <= max(team_col, eff_col):
                    continue
                name = row[team_col].strip()
                if not name:
                    continue
                try:
                    ratings[name] = float(row[eff_col])
                except (ValueError, TypeError):
                    continue

            logger.info(
                "BartTorvik: loaded %d teams (team_col=%d, eff_col=%d)",
                len(ratings), team_col, eff_col,
            )
            return ratings

        except Exception as exc:
            logger.warning("BartTorvik CSV error: %s", exc)
            return {}

    # -----------------------------------------------------------------------
    # EvanMiya
    # -----------------------------------------------------------------------

    def get_evanmiya_ratings(self) -> Dict[str, float]:
        """
        Scrape BPR (Box Plus-Minus) ratings from evanmiya.com.

        The public site renders a sortable HTML table.  We locate columns
        labelled 'Team' and 'BPR' (or sum 'OBPR' + 'DBPR' if separate).

        BPR is on a similar scale to KenPom AdjEM; positive = above-average.

        Returns {} on any failure so the model gracefully falls back to
        KenPom + BartTorvik with renormalized weights.
        """
        try:
            response = requests.get(
                _EVANMIYA_URL,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
                },
                timeout=15,
            )
            response.raise_for_status()
        except Exception as exc:
            logger.warning("EvanMiya fetch error: %s", exc)
            return {}

        try:
            soup   = BeautifulSoup(response.text, "lxml")
            tables = soup.find_all("table")

            if not tables:
                logger.warning("EvanMiya: no <table> elements found on page")
                return {}

            ratings: Dict[str, float] = {}

            for table in tables:
                header_row = table.find("tr")
                if header_row is None:
                    continue

                col_headers = [
                    th.get_text(strip=True).lower()
                    for th in header_row.find_all(["th", "td"])
                ]

                # Identify columns
                team_col = bpr_col = obpr_col = dbpr_col = None
                for i, h in enumerate(col_headers):
                    if h in ("team", "school", "teamname"):
                        team_col = i
                    if h == "bpr":
                        bpr_col = i
                    if h in ("obpr", "off bpr", "off_bpr", "o bpr"):
                        obpr_col = i
                    if h in ("dbpr", "def bpr", "def_bpr", "d bpr"):
                        dbpr_col = i

                if team_col is None:
                    continue
                if bpr_col is None and (obpr_col is None or dbpr_col is None):
                    continue

                for row in table.find_all("tr")[1:]:
                    cells = row.find_all(["td", "th"])
                    if len(cells) <= team_col:
                        continue

                    team_name = cells[team_col].get_text(strip=True)
                    if not team_name:
                        continue

                    try:
                        if bpr_col is not None and bpr_col < len(cells):
                            bpr = float(cells[bpr_col].get_text(strip=True))
                        elif obpr_col is not None and dbpr_col is not None:
                            obpr = float(cells[obpr_col].get_text(strip=True))
                            dbpr = float(cells[dbpr_col].get_text(strip=True))
                            bpr  = obpr - dbpr      # net efficiency, same sign convention as KenPom
                        else:
                            continue

                        ratings[team_name] = bpr
                    except (ValueError, TypeError, IndexError):
                        continue

                if ratings:
                    break   # Found a usable table — stop searching

            if ratings:
                logger.info(
                    "EvanMiya: loaded %d teams from %s", len(ratings), _EVANMIYA_URL
                )
            else:
                logger.warning(
                    "EvanMiya: page loaded but no BPR data parsed. "
                    "Verify EVANMIYA_URL and table structure."
                )

            return ratings

        except Exception as exc:
            logger.warning("EvanMiya parse error: %s", exc)
            return {}

    # -----------------------------------------------------------------------
    # Aggregator
    # -----------------------------------------------------------------------

    def get_all_ratings(self, use_cache: bool = True) -> Dict[str, Dict[str, float]]:
        """Fetch all three sources, caching results for up to 6 hours."""
        if use_cache and self.cache and self.cache_timestamp:
            age_hours = (
                datetime.utcnow() - self.cache_timestamp
            ).total_seconds() / 3600
            if age_hours < 6:
                return self.cache

        ratings = {
            "kenpom":    self.get_kenpom_ratings(),
            "barttorvik": self.get_barttorvik_ratings(),
            "evanmiya":  self.get_evanmiya_ratings(),
        }

        self.cache           = ratings
        self.cache_timestamp = datetime.utcnow()

        logger.info(
            "Ratings loaded — KenPom: %d, BartTorvik: %d, EvanMiya: %d",
            len(ratings["kenpom"]),
            len(ratings["barttorvik"]),
            len(ratings["evanmiya"]),
        )
        return ratings


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_ratings_service: Optional[RatingsService] = None


def get_ratings_service() -> RatingsService:
    global _ratings_service
    if _ratings_service is None:
        _ratings_service = RatingsService()
    return _ratings_service


def fetch_current_ratings() -> Dict:
    return get_ratings_service().get_all_ratings()
