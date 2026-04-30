"""
Player Mapper — Phase 4 Workstream B

Maps Yahoo roster data + PlayerRollingStats to CanonicalPlayerRow.
Orchestrates L0 (contracts) and L2 (Yahoo data) to produce UI-ready player rows.
"""

from datetime import datetime
from typing import Dict, List, Optional, Set
from zoneinfo import ZoneInfo

from sqlalchemy import or_
from sqlalchemy.orm import Session

from backend.stat_contract import (
    SCORING_CATEGORY_CODES,
    BATTING_CODES,
    PITCHING_CODES,
    CONTRACT,
)
from backend.contracts import (
    CanonicalPlayerRow,
    CategoryStats,
    PlayerGameContext,
    FreshnessMetadata,
)
from backend.models import PlayerRollingStats


# Mapping from PlayerRollingStats fields to canonical category codes
_ROLLING_BATTING_MAP: Dict[str, str] = {
    "w_runs": "R",
    "w_hits": "H",
    "w_home_runs": "HR_B",
    "w_rbi": "RBI",
    "w_strikeouts_bat": "K_B",
    "w_tb": "TB",
    "w_avg": "AVG",
    "w_ops": "OPS",
    "w_net_stolen_bases": "NSB",
}

_ROLLING_PITCHING_MAP: Dict[str, str] = {
    "w_strikeouts_pit": "K_P",
    "w_era": "ERA",
    "w_whip": "WHIP",
    "w_k_per_9": "K_9",
    "w_qs": "QS",
}


def _map_rolling_to_category_stats(
    rolling_stats: Optional[PlayerRollingStats],
) -> Optional[CategoryStats]:
    """Map PlayerRollingStats row to CategoryStats for rolling_14d window."""
    if rolling_stats is None:
        return None

    values: Dict[str, Optional[float]] = {cat: None for cat in SCORING_CATEGORY_CODES}

    # Map batting stats
    for rolling_field, canon_code in _ROLLING_BATTING_MAP.items():
        val = getattr(rolling_stats, rolling_field, None)
        if val is not None:
            values[canon_code] = float(val)

    # Map pitching stats
    for rolling_field, canon_code in _ROLLING_PITCHING_MAP.items():
        val = getattr(rolling_stats, rolling_field, None)
        if val is not None:
            values[canon_code] = float(val)

    return CategoryStats(values=values)


def _map_yahoo_stats_to_category_stats(
    yahoo_player: Dict,
) -> Optional[CategoryStats]:
    """
    Map Yahoo player stats dict to CategoryStats for season_stats.

    Yahoo stats use stat_id keys. We map them via CONTRACT.yahoo_id_index.
    """
    stats_raw = yahoo_player.get("stats", {})
    if not stats_raw:
        return None

    values: Dict[str, Optional[float]] = {cat: None for cat in SCORING_CATEGORY_CODES}

    # CONTRACT.yahoo_id_index maps stat_id -> canonical_code
    yahoo_index = CONTRACT.yahoo_id_index

    for stat_id_str, value in stats_raw.items():
        stat_id = str(stat_id_str)
        if stat_id in yahoo_index:
            canon_code = yahoo_index[stat_id]
            if canon_code in SCORING_CATEGORY_CODES:
                try:
                    values[canon_code] = float(value) if value is not None else None
                except (ValueError, TypeError):
                    values[canon_code] = None

    return CategoryStats(values=values)


def _build_player_game_context(yahoo_player: Dict) -> Optional[PlayerGameContext]:
    """Build PlayerGameContext from Yahoo player data if available."""
    # Check if player has upcoming game info
    # Yahoo provides this in various formats - this is a minimal implementation

    # For MVP: return None (game context requires additional Yahoo API calls)
    # TODO: K-XX: Wire up Yahoo matchup data for game context
    return None


def _normalize_status(yahoo_player: Dict) -> str:
    """
    Normalize Yahoo status to CanonicalPlayerRow.status enum.

    Mapping:
    - "IL", "IL15", "IL60", "IL10" -> "IL"
    - "DL" (legacy) -> "IL"
    - "DTD", "Day-to-Day" -> "probable"
    - "O", "Out" -> "not_playing"
    - "NA", "N/A" -> "minors"
    - "Active" (or no status) -> "playing"
    """
    # Use get with default None, then check explicitly for each key
    # to avoid issues with falsy values like False, 0, ""
    raw_status = yahoo_player.get("status")
    if raw_status is None:
        raw_status = yahoo_player.get("injury_status")

    if isinstance(raw_status, bool):
        return "playing" if raw_status else "not_playing"

    if not raw_status:
        return "playing"

    status_str = str(raw_status).upper()

    # Injured list variants
    if status_str.startswith("IL") or status_str == "DL":
        return "IL"

    # Day-to-day
    if status_str in ("DTD", "DAY-TO-DAY"):
        return "probable"

    # Out
    if status_str in ("O", "OUT", "SUSPENDED"):
        return "not_playing"

    # Minors / NA
    if status_str in ("NA", "N/A", "MINORS"):
        return "minors"

    # Default to playing
    return "playing"


def map_yahoo_player_to_canonical_row(
    yahoo_player: Dict,
    rolling_stats: Optional[PlayerRollingStats] = None,
    rolling_stats_7d: Optional[PlayerRollingStats] = None,
    rolling_stats_14d: Optional[PlayerRollingStats] = None,
    rolling_stats_15d: Optional[PlayerRollingStats] = None,
    rolling_stats_30d: Optional[PlayerRollingStats] = None,
    computed_at: Optional[datetime] = None,
    ros_projection: Optional[CategoryStats] = None,
) -> CanonicalPlayerRow:
    """
    Map Yahoo player data + PlayerRollingStats to CanonicalPlayerRow.

    Args:
        yahoo_player: Parsed dict from YahooFantasyClient.get_roster()
        rolling_stats: (Deprecated, use rolling_stats_14d) Optional PlayerRollingStats row for 14-day window
        rolling_stats_7d: Optional PlayerRollingStats row for 7-day window
        rolling_stats_14d: Optional PlayerRollingStats row for 14-day window
        rolling_stats_15d: Optional PlayerRollingStats row for 15-day window
        rolling_stats_30d: Optional PlayerRollingStats row for 30-day window
        computed_at: Timestamp for freshness metadata
        ros_projection: Pre-built ROS CategoryStats from PlayerProjection.cat_scores batch query

    Returns:
        CanonicalPlayerRow with all PR-1 through PR-22 fields populated
    """
    now_et = computed_at or datetime.now(ZoneInfo("America/New_York"))

    # Backward compatibility: rolling_stats -> rolling_stats_14d
    if rolling_stats is not None and rolling_stats_14d is None:
        rolling_stats_14d = rolling_stats

    # PR-1 through PR-4: Identity and status
    player_name = yahoo_player.get("name") or yahoo_player.get("full_name", "")
    team = yahoo_player.get("team", "")
    eligible_positions = yahoo_player.get("positions") or []
    if isinstance(eligible_positions, str):
        eligible_positions = [eligible_positions]

    status = _normalize_status(yahoo_player)

    # PR-5 through PR-12: Game context (TODO: wire up Yahoo matchup data)
    game_context = _build_player_game_context(yahoo_player)

    # PR-13: Season stats from Yahoo
    season_stats = _map_yahoo_stats_to_category_stats(yahoo_player)

    # PR-14 through PR-17: Rolling windows from PlayerRollingStats
    rolling_7d = _map_rolling_to_category_stats(rolling_stats_7d)
    rolling_14d = _map_rolling_to_category_stats(rolling_stats_14d)
    rolling_15d = _map_rolling_to_category_stats(rolling_stats_15d)
    rolling_30d = _map_rolling_to_category_stats(rolling_stats_30d)

    # ros_projection: passed in by caller from PlayerProjection.cat_scores batch query
    # row_projection: Phase 2 deliverable — remains None until RoW data is wired
    row_projection = None

    # PR-20: Ownership percentage from Yahoo
    ownership_pct = yahoo_player.get("ownership_pct", yahoo_player.get("percent_owned", 0.0))
    if isinstance(ownership_pct, str):
        try:
            ownership_pct = float(ownership_pct)
        except ValueError:
            ownership_pct = 0.0

    # PR-21: Injury status
    injury_status = yahoo_player.get("injury_note") or yahoo_player.get("injury_status")

    # PR-22: Freshness metadata
    freshness = FreshnessMetadata(
        primary_source="yahoo",
        fetched_at=None,  # TODO: track from Yahoo client
        computed_at=now_et,
        staleness_threshold_minutes=60,
        is_stale=False,  # TODO: compute from fetched_at
    )

    # Internal IDs
    yahoo_player_key = yahoo_player.get("player_key", "")
    bdl_player_id = yahoo_player.get("bdl_player_id")  # If available from join
    mlbam_id = yahoo_player.get("mlbam_id")  # If available from join

    return CanonicalPlayerRow(
        player_name=player_name,
        team=team,
        eligible_positions=eligible_positions,
        status=status,
        game_context=game_context,
        season_stats=season_stats,
        rolling_7d=rolling_7d,
        rolling_14d=rolling_14d,
        rolling_15d=rolling_15d,
        rolling_30d=rolling_30d,
        ros_projection=ros_projection,
        row_projection=row_projection,
        ownership_pct=ownership_pct,
        injury_status=injury_status,
        freshness=freshness,
        yahoo_player_key=yahoo_player_key,
        bdl_player_id=bdl_player_id,
        mlbam_id=mlbam_id,
    )


def fetch_rolling_stats_for_players(
    db: Session,
    yahoo_player_keys: List[str],
    as_of_date: Optional[str] = None,
    window_days: int = 14,
) -> Dict[str, PlayerRollingStats]:
    """
    Fetch PlayerRollingStats for a list of Yahoo player keys.

    Returns a dict mapping yahoo_player_key -> PlayerRollingStats.
    """
    from backend.models import PlayerIDMapping

    if not yahoo_player_keys:
        return {}

    query_keys = {str(key) for key in yahoo_player_keys if key}
    yahoo_ids = {key.rsplit(".", 1)[-1] for key in query_keys}

    # Map Yahoo keys to BDL IDs via PlayerIDMapping.
    bdl_ids_query = (
        db.query(PlayerIDMapping.bdl_id, PlayerIDMapping.yahoo_key, PlayerIDMapping.yahoo_id)
        .filter(
            or_(
                PlayerIDMapping.yahoo_key.in_(query_keys),
                PlayerIDMapping.yahoo_id.in_(yahoo_ids),
            )
        )
        .all()
    )

    yahoo_to_bdl: Dict[str, int] = {}
    for row in bdl_ids_query:
        if row.bdl_id is None:
            continue
        if row.yahoo_key:
            yahoo_to_bdl[str(row.yahoo_key)] = row.bdl_id
        if row.yahoo_id:
            yahoo_to_bdl[str(row.yahoo_id)] = row.bdl_id

    resolved_map = {}
    for yahoo_key in query_keys:
        bdl_id = yahoo_to_bdl.get(yahoo_key)
        if bdl_id is None:
            bdl_id = yahoo_to_bdl.get(yahoo_key.rsplit(".", 1)[-1])
        if bdl_id is not None:
            resolved_map[yahoo_key] = bdl_id

    if not resolved_map:
        return {}

    # Fetch rolling stats — fall back to latest available date if target_date has no data
    target_date = as_of_date or datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

    rolling_stats_query = (
        db.query(PlayerRollingStats)
        .filter(
            PlayerRollingStats.bdl_player_id.in_(resolved_map.values()),
            PlayerRollingStats.as_of_date == target_date,
            PlayerRollingStats.window_days == window_days,
        )
        .all()
    )

    # Fallback: if no stats for target_date, use the most recent available date
    if not rolling_stats_query:
        from sqlalchemy import func

        latest_date_result = (
            db.query(func.max(PlayerRollingStats.as_of_date))
            .filter(
                PlayerRollingStats.bdl_player_id.in_(resolved_map.values()),
                PlayerRollingStats.window_days == window_days,
            )
            .scalar()
        )
        if latest_date_result:
            target_date = latest_date_result.strftime("%Y-%m-%d")
            rolling_stats_query = (
                db.query(PlayerRollingStats)
                .filter(
                    PlayerRollingStats.bdl_player_id.in_(resolved_map.values()),
                    PlayerRollingStats.as_of_date == target_date,
                    PlayerRollingStats.window_days == window_days,
                )
                .all()
            )

    # Map back to Yahoo player keys
    bdl_to_yahoo = {bdl_id: yahoo_key for yahoo_key, bdl_id in resolved_map.items()}
    result = {}
    for rs in rolling_stats_query:
        yahoo_key = bdl_to_yahoo.get(rs.bdl_player_id)
        if yahoo_key:
            result[yahoo_key] = rs

    return result
