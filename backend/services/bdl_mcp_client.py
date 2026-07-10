"""
BDLPlayerResolver -- MCP-shaped player resolution API over the BDL REST client.

Naming note: this module carries the "mcp" name because it mirrors the method
surface of the official BallDontLie MCP server (https://mcp.balldontlie.io/mcp):
search_players / get_player_by_name / get_player_stats / get_projections.
Per CLAUDE.md, production runtime MUST use direct REST + Pydantic contracts --
the hosted MCP is for agent research and validation only. This module therefore
delegates all transport to BallDontLieClient (backend/services/balldontlie.py).

Projection note: BDL GOAT exposes season aggregates (/mlb/v1/season_stats) but
has NO forward-looking projection endpoint. get_projections() returns the
season-to-date aggregate line as the projection basis, tagged source="bdl".
Callers that need true ROS projections combine this with Yahoo ROS or the
Statcast pipeline (see projection source priority in daily_ingestion).

Name matching is accent-insensitive: BDL stores unaccented names
("Cristopher Sanchez") while Yahoo sends accented ones ("Cristopher Sánchez").
"""

import logging
import unicodedata
from typing import List, Optional

from backend.data_contracts import MLBPlayer, MLBSeasonStats
from backend.services.balldontlie import BallDontLieClient, get_bdl_client

logger = logging.getLogger(__name__)


def _normalize_name(name: str) -> str:
    """Accent-strip + lowercase a player name for cross-system comparison."""
    if not name:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(name))
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    normalized = stripped.lower().strip()
    for suffix in (" jr.", " sr.", " ii", " iii", " iv", " jr", " sr"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].strip()
    normalized = normalized.replace(".", "")
    while "  " in normalized:
        normalized = normalized.replace("  ", " ")
    return normalized


class BDLPlayerResolver:
    """
    Player-identity resolution against BallDontLie.

    Usage:
        resolver = BDLPlayerResolver()               # shared REST client
        resolver = BDLPlayerResolver(client=client)  # injected (tests)
    """

    def __init__(self, client: Optional[BallDontLieClient] = None):
        self._client = client

    @property
    def client(self) -> BallDontLieClient:
        if self._client is None:
            self._client = get_bdl_client()
        return self._client

    # ------------------------------------------------------------------
    # MCP-shaped surface
    # ------------------------------------------------------------------

    def search_players(self, query: str) -> List[MLBPlayer]:
        """
        Search BDL players by name fragment.

        BDL's search parameter does not handle accents -- the query is
        accent-stripped before sending. Returns [] on error (never raises).
        """
        if not query or not query.strip():
            return []
        clean = "".join(
            c for c in unicodedata.normalize("NFKD", query) if not unicodedata.combining(c)
        ).strip()
        return self.client.search_mlb_players(clean)

    def get_player_by_name(self, name: str) -> Optional[MLBPlayer]:
        """
        Resolve a full player name to a single BDL player.

        Strategy:
          1. Search on the last name token (widest reliable BDL search).
          2. Exact accent-insensitive full-name match.
          3. If exactly one active candidate shares the last name and first
             initial, accept it (handles nickname/spelling drift).

        Returns None when zero or ambiguous matches -- never guesses between
        multiple plausible players.
        """
        if not name or not name.strip():
            return None

        target = _normalize_name(name)
        last_token = target.split()[-1] if target.split() else ""
        candidates = self.search_players(last_token or name)
        if not candidates:
            # Retry with the full name (covers single-token and suffix cases)
            candidates = self.search_players(name)
        if not candidates:
            return None

        exact = [p for p in candidates if _normalize_name(p.full_name) == target]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            active_exact = [p for p in exact if p.active]
            if len(active_exact) == 1:
                return active_exact[0]
            logger.warning("get_player_by_name(%r): %d exact matches, ambiguous", name, len(exact))
            return None

        first_initial = target[0] if target else ""
        loose = [
            p for p in candidates
            if p.active
            and _normalize_name(p.last_name) == last_token
            and _normalize_name(p.first_name)[:1] == first_initial
        ]
        if len(loose) == 1:
            logger.info(
                "get_player_by_name(%r): loose match -> bdl_id=%d (%s)",
                name, loose[0].id, loose[0].full_name,
            )
            return loose[0]

        logger.warning(
            "get_player_by_name(%r): no unambiguous match (%d candidates)",
            name, len(candidates),
        )
        return None

    def get_player_stats(self, bdl_id: int, season: int = 2026) -> Optional[MLBSeasonStats]:
        """
        Fetch the season-aggregate stat line for one player.

        Returns None if BDL has no stats for that player/season.
        """
        rows = self.client.get_mlb_season_stats(season=season, player_ids=[bdl_id])
        for row in rows:
            if row.bdl_player_id == bdl_id:
                return row
        return None

    def get_projections(self, bdl_id: int, season: int = 2026) -> Optional[dict]:
        """
        Return the BDL-derived projection basis for a player.

        BDL has no forward-looking projection endpoint; this packages the
        season-to-date aggregate as projection inputs with explicit source
        tagging so downstream priority logic (BDL -> Yahoo ROS -> Statcast)
        can log which source won.
        """
        stats = self.get_player_stats(bdl_id, season=season)
        if stats is None:
            return None
        return {
            "source": "bdl",
            "bdl_id": bdl_id,
            "season": stats.season,
            "is_pitcher": stats.is_pitcher_line,
            "is_batter": stats.is_batter_line,
            "stats": stats.model_dump(exclude={"player"}),
        }


_resolver: Optional[BDLPlayerResolver] = None


def get_bdl_resolver() -> BDLPlayerResolver:
    """Return the shared BDLPlayerResolver (lazy-init)."""
    global _resolver
    if _resolver is None:
        _resolver = BDLPlayerResolver()
    return _resolver
