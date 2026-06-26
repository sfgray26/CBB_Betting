"""
Fantasy router — all /api/fantasy/*, /api/dashboard/*, /api/user/preferences* routes.

Strangler-fig extraction from backend/main.py.
Do NOT import from other backend.routers modules here.
"""

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import text, func, or_, and_, inspect, cast, Text
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import aliased
from typing import List, Optional, Literal, Dict
import logging
import os
import difflib as _difflib
import json as _json
import asyncio
import unicodedata
import uuid
from zoneinfo import ZoneInfo
from datetime import date, datetime, timedelta, timezone
from dataclasses import asdict

from backend.models import (
    FantasyDraftSession,
    FantasyDraftPick,
    FantasyLineup,
    PlayerDailyMetric,
    StatcastPerformance,
    PlayerProjection,
    PatternDetectionAlert,
    UserPreferences,
    ProjectionCacheEntry,
    ProjectionSnapshot,
    PlayerValuationCache,
    DataIngestionLog,
    DBAlert,
    PlayerScore,
    PlayerIDMapping,
    PositionEligibility,
    DecisionResult,
    DecisionExplanation,
    SessionLocal,
    get_db,
)
from backend.auth import verify_api_key, verify_admin_api_key
from backend.schemas import (
    DailyLineupResponse,
    WaiverWireResponse,
    RosterResponse,
    MatchupResponse,
    LineupPlayerOut,
    StartingPitcherOut,
    RosterPlayerOut,
    MatchupTeamOut,
    LineupApplyPlayer,
    LineupApplyRequest,
    MatchupSimulateRequest,
    PlayerScoresResponse,
    PlayerScoreOut,
    PlayerScoreCategoryBreakdown,
    DecisionsResponse,
    DecisionWithExplanation,
    DecisionResultOut,
    DecisionExplanationOut,
    FactorDetail,
    DecisionPipelineStatus,
    RosterActionError,
    RosterActionRequest,
    RosterActionResponse,
    RosterActionWarning,
)
from backend.contracts import (
    BulkRosterMove,
    BulkRosterMoveRequest,
    BulkRosterMoveResponse,
    CanonicalRosterResponse,
    DecisionAccuracyResponse,
    DecisionAccuracyTrendPoint,
    FreshnessMetadata,
    MatchupPreviewCategoryProjection,
    MatchupPreviewResponse,
    ScheduleAdvantage,
    WeakCategory,
    RosterMoveRequest,
    RosterMoveResponse,
    RosterOptimizeRequest,
    RosterOptimizeResponse,
    PlayerSlotAssignment,
    AutoStreamConfigureRequest,
)
from backend.stat_contract import YAHOO_ID_INDEX, SCORING_CATEGORY_CODES
from backend.utils.time_utils import today_et
from backend.fantasy_baseball.yahoo_client_resilient import (
    YahooAuthError,
    YahooAPIError,
    ResilientYahooClient,
    get_yahoo_client,
    get_resilient_yahoo_client,
)
from backend.fantasy_baseball.daily_lineup_optimizer import get_lineup_optimizer
from backend.services.injury_overlay import (
    apply_injury_penalty,
    load_injury_overlays_for_yahoo_players,
)
from backend.services.job_queue_service import submit_job as jq_submit, get_job_status as jq_status
from backend.services.player_mapper import (
    map_yahoo_player_to_canonical_row,
    fetch_rolling_stats_for_players,
    fetch_rolling_stats_for_players_all_windows,
)
from backend.services.category_comparator import (
    compare_category,
    get_canonical_category,
)

# Auto-Stream service (lazy import to avoid circular imports)
_auto_stream_service = None

def get_auto_stream_service():
    """Get singleton AutoStreamService instance."""
    global _auto_stream_service
    if _auto_stream_service is None:
        from backend.services.auto_stream import AutoStreamService
        _auto_stream_service = AutoStreamService()
    return _auto_stream_service

logger = logging.getLogger(__name__)
router = APIRouter()

# Module-level MLB probable-starts cache (shared state — same pattern as main.py)
_STARTS_CACHE: dict = {}

# Matchup response cache removed (Loop 9) — always fetch live data from Yahoo.
# Previous 5-minute cache caused staleness: 0W-0L-18T vs 4W-11L-3T across modules.
# Timeout safety valve added (5s) to prevent hangs on slow Yahoo responses.

# League settings (stat ID map) — 2-hour TTL; never changes mid-season.
_LEAGUE_SETTINGS_CACHE: dict = {}

# Per-player need_score history for volatility detection.
# Keyed by player_key → (score: float, ts: datetime).
# Only entries within the last 60 minutes trigger a volatile flag.
_NEED_SCORE_HISTORY: dict = {}

# Yahoo H2H weeks run Monday–Sunday. Week 1 = March 24–30, 2026 (first Monday on or after Opening Day).
# Using Opening Day (Mar 20, Thursday) as epoch produces a 4-day offset that causes queries to
# hit the wrong Yahoo matchup week — Week 10 instead of Week 9 on May 22.
_MLB_FIRST_MATCHUP_MONDAY = date(2026, 3, 24)
_FANTASY_TOTAL_WEEKS = 25


def _compute_mlb_current_week(today: date) -> int:
    days_elapsed = max(0, (today - _MLB_FIRST_MATCHUP_MONDAY).days)
    return max(1, min(_FANTASY_TOTAL_WEEKS, (days_elapsed // 7) + 1))


def _fetch_probable_starts_map(start_date: str, end_date: str) -> dict:
    """Return {pitcher_full_name_lower: starts_count} via public MLB Stats API (6h cached).

    Module-level so both the waiver-wire and recommendations endpoints share the
    same cache.  Keys are lowercase full names; values are integer start counts.
    """
    import httpx as _httpx
    _now = datetime.now(timezone.utc)
    if _STARTS_CACHE.get("data") and _STARTS_CACHE.get("fetched_at"):
        _fetched = _STARTS_CACHE["fetched_at"]
        if _fetched.tzinfo is None:
            _fetched = _fetched.replace(tzinfo=timezone.utc)
        age_h = (_now - _fetched).total_seconds() / 3600
        if age_h < 6:
            return _STARTS_CACHE["data"]
    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&startDate={start_date}&endDate={end_date}"
        "&gameType=R&hydrate=probablePitcher"
    )
    try:
        resp = _httpx.get(url, timeout=8.0)
        resp.raise_for_status()
    except Exception as _e:
        logger.warning("MLB Stats API schedule fetch failed: %s", _e)
        return _STARTS_CACHE.get("data") or {}
    starts: dict = {}
    for date_entry in resp.json().get("dates", []):
        for game in date_entry.get("games", []):
            for side in ("home", "away"):
                pitcher = (game.get("teams", {})
                           .get(side, {})
                           .get("probablePitcher", {}))
                pname = (pitcher.get("fullName") or "").strip().lower()
                if pname:
                    starts[pname] = starts.get(pname, 0) + 1
    _STARTS_CACHE["data"] = starts
    _STARTS_CACHE["fetched_at"] = _now
    return starts


def _populate_starts_this_week(players: list, starts_map: dict) -> None:
    """Mutate each SP dict in *players* to add ``starts_this_week`` from *starts_map*.

    Uses difflib fuzzy matching at ≥0.90 ratio as a fallback for name variations.
    Non-SP positions are left unchanged.
    """
    import difflib as _difflib
    for _fa in players:
        if "SP" not in (_fa.get("positions") or []):
            continue
        _name = (_fa.get("name") or "").strip().lower()
        _starts = starts_map.get(_name, 0)
        if _starts == 0 and starts_map:
            _best = max(
                starts_map.keys(),
                key=lambda k: _difflib.SequenceMatcher(None, _name, k).ratio(),
                default=None,
            )
            if _best and _difflib.SequenceMatcher(None, _name, _best).ratio() >= 0.90:
                _starts = starts_map[_best]
        _fa["starts_this_week"] = _starts

# Shared fallback for Yahoo numeric stat category IDs.
_YAHOO_STAT_FALLBACK: dict = dict(YAHOO_ID_INDEX)

# Maps SCORING_CATEGORY_CODES (uppercase, from YAHOO_ID_INDEX values) to the
# lowercase board keys used in PlayerProjection.cat_scores and player_board.py.
# cat.lower() covers most cases; only non-trivial remappings are listed here.
_CANONICAL_TO_BOARD: dict = {
    "HR_B":  "hr",
    "HR_P":  "hr_pit",
    "K_9":   "k9",
    "K_P":   "k_pit",
    "K_B":   "k_bat",
}


# ============================================================================
# HELPERS
# ============================================================================

def _get_ingestion_orchestrator():
    """Lazy-import ingestion orchestrator from main to avoid circular imports."""
    try:
        from backend.main import _ingestion_orchestrator
        return _ingestion_orchestrator
    except Exception:
        return None


def _enforce_projection_freshness(consumer: str, force_stale: bool = False) -> list:
    """Block stale fantasy decisions unless the caller explicitly overrides the guard."""
    try:
        from backend.main import _get_projection_freshness_report
        report = _get_projection_freshness_report()
    except Exception:
        return []

    violations = list(report.get("violations") or [])
    if not violations:
        return []

    message = f"Projection freshness gate triggered for {consumer}"
    detail = {
        "error": "projection_freshness_violation",
        "message": message,
        "consumer": consumer,
        "force_stale_available": True,
        "freshness": report,
    }

    if not force_stale:
        raise HTTPException(status_code=503, detail=detail)

    logger.warning("%s -- proceeding due to force_stale override: %s", message, violations)
    return [f"Stale-data override active: {'; '.join(violations)}"]


def _flatten_scoreboard_team_entry(team_entry) -> tuple:
    """Recursively flatten a team entry from Yahoo scoreboard response.

    Yahoo returns matchup team payloads as irregular nested list/dict
    structures. A shallow 2-level descent — as the waiver handler
    historically used — misses team_key and team_stats when they are
    one level deeper, leaving matchup_opponent as "TBD" and
    category_deficits empty. The matchup endpoint already uses a
    recursive walker; this is the shared implementation.

    Returns (team_key, team_name, raw_stats_list).
    """
    t_meta: dict = {}
    stats_raw: list = []

    def _walk(node, depth: int = 0) -> None:
        nonlocal stats_raw
        if depth > 5:
            return
        if isinstance(node, list):
            for item in node:
                _walk(item, depth + 1)
        elif isinstance(node, dict):
            if "team_stats" in node and not stats_raw:
                inner = node["team_stats"].get("stats", [])
                if isinstance(inner, list):
                    stats_raw = inner
            for key in ("team_key", "name", "team_id", "nickname"):
                if key in node and key not in t_meta:
                    t_meta[key] = node[key]

    _walk(team_entry)

    team_key = t_meta.get("team_key", "") or ""
    team_name = t_meta.get("name", "") or t_meta.get("nickname", "") or ""
    return (team_key, team_name, stats_raw)


def _iter_scoreboard_matchup_teams(matchups: list) -> list:
    """Return list of matchups; each matchup is a list of flattened team tuples."""
    result: list = []
    for m in matchups or []:
        if not isinstance(m, dict):
            continue
        teams = m.get("teams") or m.get("0", {}).get("teams", {})
        team_entries: list = []
        if isinstance(teams, list):
            for item in teams:
                if isinstance(item, dict):
                    team_entries.append(
                        _flatten_scoreboard_team_entry(item.get("team", item))
                    )
        elif isinstance(teams, dict):
            try:
                count_t = int(teams.get("count", 0))
            except (TypeError, ValueError):
                count_t = 0
            for ti in range(count_t):
                entry = teams.get(str(ti), {})
                if isinstance(entry, dict):
                    team_entries.append(
                        _flatten_scoreboard_team_entry(entry.get("team", entry))
                    )
        if team_entries:
            result.append(team_entries)
    return result


def _normalize_identity_name(name: str) -> str:
    """Normalize player names before cross-system identity comparisons."""
    if not name:
        return ""

    normalized = unicodedata.normalize("NFKD", str(name)).lower().strip()
    for suffix in (" jr.", " sr.", " ii", " iii", " iv", " jr", " sr"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].strip()
    normalized = normalized.replace(".", "")
    while "  " in normalized:
        normalized = normalized.replace("  ", " ")
    return normalized.strip()


def _yahoo_key_variants(player_key: str) -> set[str]:
    """Return lookup variants for Yahoo roster keys across legacy formats."""
    if not player_key:
        return set()

    variants = {player_key.strip()}
    if ".p." not in player_key:
        return variants

    prefix, yahoo_id = player_key.split(".p.", 1)
    if ".l." in prefix:
        game_id = prefix.split(".l.", 1)[0]
        variants.add(f"{game_id}.p.{yahoo_id}")
    return {variant for variant in variants if variant}


def _mapping_name_matches(player_name: str, mapping_name: str) -> bool:
    """Reject stale identity rows that point a Yahoo key at the wrong player."""
    roster_normalized = _normalize_identity_name(player_name)
    mapping_normalized = _normalize_identity_name(mapping_name)
    if not roster_normalized or not mapping_normalized:
        return True
    return roster_normalized == mapping_normalized


def _projection_fallback_score(yahoo_player: dict) -> tuple[float, str]:
    """Return a lineup score from board projections for players missing from player_scores.

    This path is reached when a player has no BDL ID in player_id_mapping OR has a
    BDL ID but no row in the PlayerScore table (ingestion not yet run today).
    In a 10-team league every rostered player should have a PlayerScore — if many
    players hit this path it signals a data pipeline gap in job 100_019/100_029.

    Scoring tiers:
      - Steamer/Statcast data found (fusion_source != "population_prior"):
            score from z_score, source = "projection_fallback"
      - Truly no data (population prior only, z_score == 0.0):
            score = 0.0, source = "no_score" — ranks below all real-scored players
    """
    from backend.fantasy_baseball.player_board import get_or_create_projection

    projection = get_or_create_projection(yahoo_player)
    z_score = float(projection.get("z_score") or 0.0)
    ownership_pct = float(
        yahoo_player.get("percent_owned", yahoo_player.get("owned_pct", 0.0)) or 0.0
    )
    fusion_source = projection.get("fusion_source", "population_prior")

    # Population prior means player_board found no Steamer or Statcast data at all.
    # z_score will be 0.0. Returning 0.0 ranks them below every real-scored player.
    if fusion_source == "population_prior" and z_score == 0.0:
        return 0.0, "no_score"

    # Has real projection data (Steamer, steamer_db, SAVANT_ADJUSTED, etc.) — use it.
    score = 50.0 + (z_score * 8.0) + min(ownership_pct, 100.0) * 0.05
    return max(20.0, min(95.0, round(score, 2))), "projection_fallback"


def _resolve_roster_player_bdl_ids(db: Session, raw_players: list[dict]) -> dict[str, dict]:
    """Resolve Yahoo roster players to BDL and MLBAM IDs using key-first identity matching.

    Returns dict mapping player_key -> {"bdl_id": int, "mlbam_id": int | None}.
    """
    roster_lookup_keys: set[str] = set()
    yahoo_ids: set[str] = set()

    for player in raw_players:
        player_key = player.get("player_key") or ""
        roster_lookup_keys.update(_yahoo_key_variants(player_key))
        if ".p." in player_key:
            yahoo_ids.add(player_key.split(".p.", 1)[-1])

    predicates = []
    if roster_lookup_keys:
        predicates.append(PlayerIDMapping.yahoo_key.in_(list(roster_lookup_keys)))
    if yahoo_ids:
        predicates.append(PlayerIDMapping.yahoo_id.in_(list(yahoo_ids)))
    if not predicates:
        return {}

    mapping_rows = (
        db.query(
            PlayerIDMapping.yahoo_key,
            PlayerIDMapping.yahoo_id,
            PlayerIDMapping.bdl_id,
            PlayerIDMapping.mlbam_id,
            PlayerIDMapping.normalized_name,
            PlayerIDMapping.full_name,
        )
        .filter(
            PlayerIDMapping.bdl_id.isnot(None),
            or_(*predicates),
        )
        .all()
    )

    mapping_by_key = {}
    mapping_by_yahoo_id = {}
    for row in mapping_rows:
        if row.yahoo_key:
            mapping_by_key[row.yahoo_key] = row
        if row.yahoo_id:
            mapping_by_yahoo_id[row.yahoo_id] = row

    player_key_to_ids: dict[str, dict] = {}
    unresolved_players: list[dict] = []

    for player in raw_players:
        player_key = player.get("player_key") or ""
        player_name = player.get("name") or ""

        matched_row = None
        for key_variant in _yahoo_key_variants(player_key):
            candidate = mapping_by_key.get(key_variant)
            if candidate and _mapping_name_matches(
                player_name,
                candidate.normalized_name or candidate.full_name or "",
            ):
                matched_row = candidate
                break

        if matched_row is None and ".p." in player_key:
            yahoo_id = player_key.split(".p.", 1)[-1]
            candidate = mapping_by_yahoo_id.get(yahoo_id)
            if candidate and _mapping_name_matches(
                player_name,
                candidate.normalized_name or candidate.full_name or "",
            ):
                matched_row = candidate

        if matched_row is not None and matched_row.bdl_id is not None:
            player_key_to_ids[player_key] = {
                "bdl_id": matched_row.bdl_id,
                "mlbam_id": matched_row.mlbam_id,
            }
        else:
            unresolved_players.append(player)

    unresolved_names = {
        _normalize_identity_name(player.get("name") or "")
        for player in unresolved_players
        if player.get("name")
    }
    unresolved_names.discard("")

    if unresolved_names:
        name_rows = (
            db.query(
                PlayerIDMapping.normalized_name,
                PlayerIDMapping.bdl_id,
                PlayerIDMapping.mlbam_id,
                PlayerIDMapping.full_name,
            )
            .filter(
                PlayerIDMapping.normalized_name.in_(list(unresolved_names)),
                PlayerIDMapping.bdl_id.isnot(None),
            )
            .all()
        )

        candidates_by_name: dict[str, list] = {}
        for row in name_rows:
            candidates_by_name.setdefault(row.normalized_name, []).append(row)

        for player in unresolved_players:
            normalized_name = _normalize_identity_name(player.get("name") or "")
            candidates = candidates_by_name.get(normalized_name, [])
            if len(candidates) == 1:
                player_key = player.get("player_key") or ""
                if player_key:
                    player_key_to_ids[player_key] = {
                        "bdl_id": candidates[0].bdl_id,
                        "mlbam_id": candidates[0].mlbam_id,
                    }

    return player_key_to_ids

def build_pitcher_quality_map(db: Session, today: date) -> dict[str, float]:
    """
    Build a name→quality_score map from ProbablePitcherSnapshot for today through today+7 days.

    Non-fatal: returns empty dict on any exception so the waiver wire still
    renders when the probable_pitchers table is empty or unavailable.
    Keeps the highest quality_score when a pitcher has multiple starts in the window.
    """
    result: dict[str, float] = {}
    try:
        from backend.models import ProbablePitcherSnapshot
        rows = (
            db.query(
                ProbablePitcherSnapshot.pitcher_name,
                ProbablePitcherSnapshot.quality_score,
            )
            .filter(
                ProbablePitcherSnapshot.game_date >= today,
                ProbablePitcherSnapshot.game_date <= today + timedelta(days=7),
                ProbablePitcherSnapshot.quality_score.isnot(None),
            )
            .all()
        )
        for r in rows:
            if r.pitcher_name and r.quality_score is not None:
                key = r.pitcher_name.strip().lower()
                if key not in result or r.quality_score > result[key]:
                    result[key] = float(r.quality_score)
    except Exception:
        pass
    return result


# ============================================================================
# PROJECTION STATUS
# ============================================================================

@router.get("/api/fantasy/projection-status")
async def get_projection_status(db: Session = Depends(get_db)):
    """Return freshness metadata for the active projection source."""
    from backend.fantasy_baseball import player_board as _pb
    from backend.models import PlayerProjection
    from datetime import datetime, timezone

    try:
        newest = (
            db.query(PlayerProjection.updated_at)
            .filter(PlayerProjection.player_type.isnot(None))
            .order_by(PlayerProjection.updated_at.desc())
            .first()
        )
        db_updated_at = newest[0] if newest else None
        db_count = db.query(PlayerProjection).filter(
            PlayerProjection.player_type.isnot(None)
        ).count()
    except Exception:
        db_updated_at = None
        db_count = 0

    now = datetime.now(timezone.utc)
    age_hours = None
    if db_updated_at:
        ts = db_updated_at.replace(tzinfo=timezone.utc) if db_updated_at.tzinfo is None else db_updated_at
        age_hours = round((now - ts).total_seconds() / 3600, 1)

    board = _pb._BOARD
    active_source = "unknown"
    board_count = 0
    if board:
        board_count = len(board)
        active_source = (board[0] or {}).get("source", "unknown")

    return {
        "db_updated_at": db_updated_at.isoformat() if db_updated_at else None,
        "db_player_count": db_count,
        "age_hours": age_hours,
        "is_stale": age_hours is None or age_hours > 26,
        "active_source": active_source,
        "board_player_count": board_count,
    }


# ============================================================================
# DRAFT BOARD
# ============================================================================

@router.get("/api/fantasy/draft-board")
async def fantasy_draft_board(
    position: Optional[str] = Query(None, description="Filter by position (C, 1B, SP, RP, OF, ...)"),
    player_type: Optional[str] = Query(None, description="Filter by type: batter or pitcher"),
    tier_max: Optional[int] = Query(None, description="Only show players at or below this tier"),
    limit: int = Query(200, ge=1, le=500),
    user: str = Depends(verify_api_key),
):
    """Return the full ranked fantasy draft board (Steamer/ZiPS projections)."""
    try:
        from backend.fantasy_baseball.player_board import get_board
        board = get_board()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Player board unavailable: {exc}")

    if position:
        pos_upper = position.upper()
        board = [p for p in board if pos_upper in p.get("positions", [])]
    if player_type:
        board = [p for p in board if p.get("type", "").lower() == player_type.lower()]
    if tier_max is not None:
        board = [p for p in board if p.get("tier", 99) <= tier_max]

    board = board[:limit]
    return {"count": len(board), "players": board}


@router.get("/api/fantasy/player/{player_id}")
async def fantasy_player_detail(
    player_id: str,
    user: str = Depends(verify_api_key),
):
    """Return a single player's full detail from the draft board."""
    try:
        from backend.fantasy_baseball.player_board import get_board
        board = get_board()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Player board unavailable: {exc}")

    player = next((p for p in board if p["id"] == player_id), None)
    if player is None:
        raise HTTPException(status_code=404, detail=f"Player '{player_id}' not found")
    return player


# ============================================================================
# DRAFT SESSIONS
# ============================================================================

@router.post("/api/fantasy/draft-session")
async def create_draft_session(
    my_draft_position: int = Query(..., ge=1, le=20),
    num_teams: int = Query(12, ge=4, le=20),
    num_rounds: int = Query(23, ge=10, le=30),
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """Create a new live-draft tracking session. Keepers are pre-inserted."""
    import secrets
    from backend.fantasy_baseball.player_board import get_board, MY_KEEPERS

    session_key = secrets.token_hex(8)
    session = FantasyDraftSession(
        session_key=session_key,
        my_draft_position=my_draft_position,
        num_teams=num_teams,
        num_rounds=num_rounds,
        current_pick=1,
        is_active=True,
    )
    db.add(session)
    db.flush()

    board_by_id = {p["id"]: p for p in get_board()}
    for player_id, keeper_round in MY_KEEPERS.items():
        player = board_by_id.get(player_id)
        if not player:
            continue
        db.add(FantasyDraftPick(
            session_id=session.id,
            pick_number=0,
            round_number=keeper_round,
            drafter_position=my_draft_position,
            is_my_pick=True,
            player_id=player_id,
            player_name=player["name"],
            player_team=player.get("team"),
            player_positions=player.get("positions"),
            player_tier=player.get("tier"),
            player_adp=player.get("adp"),
            player_z_score=player.get("z_score"),
        ))

    db.commit()
    db.refresh(session)
    return {
        "session_key": session_key,
        "my_draft_position": my_draft_position,
        "num_teams": num_teams,
        "num_rounds": num_rounds,
        "keepers_preloaded": list(MY_KEEPERS.keys()),
        "message": "Draft session created with keepers pre-loaded.",
    }


@router.post("/api/fantasy/draft-session/{session_key}/pick")
async def record_draft_pick(
    session_key: str,
    player_id: str = Query(...),
    drafter_position: int = Query(..., ge=1, le=20),
    is_my_pick: bool = Query(False),
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """Record a pick in a live draft session and return recommendations for the next pick."""
    from backend.fantasy_baseball.player_board import get_board
    from backend.fantasy_baseball.draft_engine import DraftState, DraftRecommender

    session = db.query(FantasyDraftSession).filter_by(
        session_key=session_key, is_active=True
    ).with_for_update().first()
    if session is None:
        raise HTTPException(status_code=404, detail="Draft session not found or inactive")

    board = get_board()
    player = next((p for p in board if p["id"] == player_id), None)
    if player is None:
        raise HTTPException(status_code=404, detail=f"Player '{player_id}' not found in board")

    pick_number = session.current_pick
    round_number = ((pick_number - 1) // session.num_teams) + 1

    pick = FantasyDraftPick(
        session_id=session.id,
        pick_number=pick_number,
        round_number=round_number,
        drafter_position=drafter_position,
        is_my_pick=is_my_pick,
        player_id=player["id"],
        player_name=player["name"],
        player_team=player.get("team"),
        player_positions=player.get("positions"),
        player_tier=player.get("tier"),
        player_adp=player.get("adp"),
        player_z_score=player.get("z_score"),
    )
    db.add(pick)
    session.current_pick = pick_number + 1
    db.commit()

    drafted_ids = {p.player_id for p in session.picks}
    try:
        state = DraftState(
            my_draft_position=session.my_draft_position,
            num_teams=session.num_teams,
            num_rounds=session.num_rounds,
        )
        state.pick_number = session.current_pick
        state.my_picks = [p.player_id for p in session.picks if p.is_my_pick]
        recs = DraftRecommender(state, board).recommend(top_n=5, drafted_ids=drafted_ids)
    except Exception:
        recs = [p for p in board if p["id"] not in drafted_ids][:5]

    return {
        "message": "Pick recorded",
        "pick_number": pick_number,
        "player_name": player["name"],
        "is_my_pick": is_my_pick,
        "next_recommendations": recs,
    }


@router.get("/api/fantasy/draft-session/{session_key}")
async def get_draft_session(
    session_key: str,
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """Return the current state and all picks for a draft session."""
    session = db.query(FantasyDraftSession).filter_by(session_key=session_key).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Draft session not found")

    picks = [
        {
            "pick_number": p.pick_number,
            "round": p.round_number,
            "drafter_position": p.drafter_position,
            "is_my_pick": p.is_my_pick,
            "is_keeper": p.pick_number == 0,
            "player_id": p.player_id,
            "player_name": p.player_name,
            "player_team": p.player_team,
            "player_positions": p.player_positions,
            "player_tier": p.player_tier,
            "player_adp": p.player_adp,
        }
        for p in sorted(session.picks, key=lambda x: x.pick_number)
    ]
    my_picks = [p for p in picks if p["is_my_pick"]]

    return {
        "session_key": session_key,
        "my_draft_position": session.my_draft_position,
        "num_teams": session.num_teams,
        "num_rounds": session.num_rounds,
        "current_pick": session.current_pick,
        "total_picks": len(picks),
        "my_picks_count": len(my_picks),
        "is_active": session.is_active,
        "picks": picks,
        "my_picks": my_picks,
    }


@router.delete("/api/fantasy/draft-session/{session_key}")
async def delete_draft_session(
    session_key: str,
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """Delete a draft session and all its picks (for resetting a test session)."""
    session = db.query(FantasyDraftSession).filter_by(session_key=session_key).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Draft session not found")

    db.query(FantasyDraftPick).filter_by(session_id=session.id).delete()
    db.delete(session)
    db.commit()
    return {"message": "Draft session deleted", "session_key": session_key}


@router.get("/api/fantasy/draft-sessions")
async def list_draft_sessions(
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """List all draft sessions (active and inactive)."""
    sessions = db.query(FantasyDraftSession).order_by(FantasyDraftSession.created_at.desc()).all()
    result = []
    for s in sessions:
        pick_count = db.query(FantasyDraftPick).filter_by(session_id=s.id).count()
        my_pick_count = db.query(FantasyDraftPick).filter_by(session_id=s.id, is_my_pick=True).count()
        result.append({
            "session_key": s.session_key,
            "my_draft_position": s.my_draft_position,
            "num_teams": s.num_teams,
            "num_rounds": s.num_rounds,
            "current_pick": s.current_pick,
            "total_picks_recorded": pick_count,
            "my_picks_recorded": my_pick_count,
            "is_active": s.is_active,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "updated_at": s.updated_at.isoformat() if s.updated_at else None,
        })
    return {"sessions": result, "count": len(result)}


@router.post("/api/fantasy/draft-session/{session_key}/reset")
async def reset_draft_session(
    session_key: str,
    my_draft_position: Optional[int] = Query(None, ge=1, le=20,
        description="Update draft position (e.g. 6). Omit to keep existing."),
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """Reset a draft session: clears all picks and resets current_pick to 1."""
    session = db.query(FantasyDraftSession).filter_by(session_key=session_key).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Draft session not found")

    picks_deleted = db.query(FantasyDraftPick).filter_by(session_id=session.id).delete()
    session.current_pick = 1
    session.is_active = True
    if my_draft_position is not None:
        session.my_draft_position = my_draft_position
    db.commit()

    return {
        "message": "Draft session reset",
        "session_key": session_key,
        "picks_cleared": picks_deleted,
        "my_draft_position": session.my_draft_position,
        "ready_for_draft": True,
    }


@router.get("/api/fantasy/draft-session/value-board")
async def fantasy_value_board(
    drafted_ids: str = Query("", description="Comma-separated player IDs already drafted"),
    position: Optional[str] = Query(None, description="Filter by position (C, 1B, SP, RP, OF, ...)"),
    player_type: Optional[str] = Query(None, description="batter or pitcher"),
    tier_max: Optional[int] = Query(None, description="Exclude players below this tier"),
    limit: int = Query(100, ge=1, le=300),
    user: str = Depends(verify_api_key),
):
    """Advanced-metrics value board - ranks AVAILABLE players by a composite value_score."""
    from backend.fantasy_baseball.player_board import get_board
    from backend.fantasy_baseball.draft_analytics import (
        inject_advanced_analytics,
        compute_value_score,
    )

    try:
        raw_board = get_board()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Player board unavailable: {exc}")

    board = [dict(p) for p in raw_board]

    inject_advanced_analytics(board)

    from backend.fantasy_baseball.player_board import ALL_LEAGUE_KEEPERS
    exclude = ALL_LEAGUE_KEEPERS | {pid.strip() for pid in drafted_ids.split(",") if pid.strip()}
    board = [p for p in board if p["id"] not in exclude]

    if position:
        pos_upper = position.upper()
        board = [p for p in board if pos_upper in p.get("positions", [])]
    if player_type:
        board = [p for p in board if p.get("type", "").lower() == player_type.lower()]
    if tier_max is not None:
        board = [p for p in board if 0 < p.get("tier", 99) <= tier_max]

    board.sort(key=compute_value_score, reverse=True)

    for p in board:
        p["value_score"] = round(compute_value_score(p), 3)

    board = board[:limit]
    return {
        "count": len(board),
        "analytics_overlay": any(bool(p.get("statcast")) for p in board),
        "players": board,
    }


@router.post("/api/fantasy/draft-session/{session_key}/sync-yahoo")
async def sync_draft_from_yahoo(
    session_key: str,
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """Poll Yahoo's draftresults endpoint and sync any new picks into the session."""
    from backend.fantasy_baseball.player_board import get_board

    session = db.query(FantasyDraftSession).filter_by(session_key=session_key).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Draft session not found")

    try:
        client = get_yahoo_client()
    except YahooAuthError as exc:
        raise HTTPException(status_code=401, detail=f"Yahoo auth not configured: {exc}")

    try:
        yahoo_picks = client.get_draft_results()
    except YahooAPIError as exc:
        raise HTTPException(status_code=502, detail=f"Yahoo API error: {exc}")

    if not yahoo_picks:
        return {"picks_synced": 0, "total_yahoo_picks": 0, "message": "Draft not started or no picks yet"}

    board = get_board()

    def _norm(s: str) -> str:
        return s.lower().replace(" ", "_").replace(".", "").replace("'", "").replace("-", "_")

    board_by_id = {p["id"]: p for p in board}
    board_by_name = {_norm(p["name"]): p for p in board}

    existing_pick_numbers = {
        pk.pick_number for pk in db.query(FantasyDraftPick).filter_by(session_id=session.id).all()
    }

    picks_synced = 0
    for raw in yahoo_picks:
        pick_num = int(raw.get("pick", 0))
        if pick_num == 0 or pick_num in existing_pick_numbers:
            continue

        round_num = int(raw.get("round", ((pick_num - 1) // session.num_teams) + 1))
        player_key = raw.get("player_key", "")

        from backend.fantasy_baseball.draft_engine import build_full_pick_order
        full_order = build_full_pick_order(session.num_teams, session.num_rounds)
        pick_pos = 0
        if pick_num <= len(full_order):
            _, _, pick_pos = full_order[pick_num - 1]
        is_my_pick = (pick_pos == session.my_draft_position)

        player_name = player_key
        player_id = _norm(player_key)
        positions: list = []
        player_type = "batter"
        tier = 0
        adp = 999.0

        try:
            yahoo_player = client.get_player(player_key)
            player_name = yahoo_player.get("name") or player_key
            positions = yahoo_player.get("positions") or []
            player_id = _norm(player_name)
        except Exception:
            pass

        board_match = board_by_id.get(player_id) or board_by_name.get(player_id)
        if board_match:
            player_type = board_match.get("type", "batter")
            tier = board_match.get("tier", 0)
            adp = board_match.get("adp", 999.0)
            if not positions:
                positions = board_match.get("positions", [])

        pick_record = FantasyDraftPick(
            session_id=session.id,
            pick_number=pick_num,
            round_number=round_num,
            player_id=player_id,
            player_name=player_name,
            player_team=yahoo_player.get("team", "") if "yahoo_player" in dir() else "",
            player_positions=",".join(positions),
            player_type=player_type,
            player_tier=tier,
            player_adp=adp,
            is_my_pick=is_my_pick,
        )
        db.add(pick_record)
        picks_synced += 1

    if picks_synced:
        session.current_pick = max(session.current_pick, len(yahoo_picks) + 1)
        db.commit()

    my_picks = [
        pk.player_name for pk in
        db.query(FantasyDraftPick).filter_by(session_id=session.id, is_my_pick=True)
        .order_by(FantasyDraftPick.pick_number).all()
    ]

    return {
        "picks_synced": picks_synced,
        "total_yahoo_picks": len(yahoo_picks),
        "session_current_pick": session.current_pick,
        "my_roster_so_far": my_picks,
    }


@router.post("/api/fantasy/draft-session/{session_key}/sync-keepers")
async def sync_keepers_pre_draft(
    session_key: str,
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """Pre-draft keeper sweep. Fetches all 12 teams' current rosters and inserts keepers."""
    from backend.fantasy_baseball.player_board import get_board

    session = db.query(FantasyDraftSession).filter_by(session_key=session_key).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Draft session not found")

    try:
        client = get_yahoo_client()
    except YahooAuthError as exc:
        raise HTTPException(status_code=401, detail=f"Yahoo auth not configured: {exc}")

    try:
        my_team_key = client.get_my_team_key()
        all_teams = client.get_all_teams()
        all_rosters = client.get_all_rosters()
    except YahooAPIError as exc:
        raise HTTPException(status_code=502, detail=f"Yahoo API error: {exc}")

    board = get_board()

    def _norm(s: str) -> str:
        return s.lower().replace(" ", "_").replace(".", "").replace("'", "").replace("-", "_")

    board_by_name = {_norm(p["name"]): p for p in board}

    existing = {
        row.player_id
        for row in db.query(FantasyDraftPick.player_id)
        .filter_by(session_id=session.id, pick_number=0)
        .all()
    }

    team_pos = {t["team_key"]: (i + 1) for i, t in enumerate(all_teams)}

    keepers_found = 0
    keepers_inserted = 0
    keepers_skipped = 0
    my_keepers: list = []

    for team_key, roster in all_rosters.items():
        is_my_team = (team_key == my_team_key)
        drafter_pos = team_pos.get(team_key, 0)

        for player in roster:
            keepers_found += 1
            raw_name = player.get("name") or ""
            player_id = _norm(raw_name) if raw_name else player.get("player_key", "unknown")

            if player_id in existing:
                keepers_skipped += 1
                if is_my_team:
                    my_keepers.append(raw_name)
                continue

            board_match = board_by_name.get(player_id)
            positions = player.get("positions") or (board_match.get("positions") if board_match else []) or []

            db.add(FantasyDraftPick(
                session_id=session.id,
                pick_number=0,
                round_number=None,
                drafter_position=drafter_pos,
                is_my_pick=is_my_team,
                player_id=player_id,
                player_name=raw_name,
                player_team=player.get("team"),
                player_positions=",".join(positions) if isinstance(positions, list) else positions,
                player_tier=board_match.get("tier") if board_match else None,
                player_adp=board_match.get("adp") if board_match else None,
                player_z_score=board_match.get("z_score") if board_match else None,
            ))
            existing.add(player_id)
            keepers_inserted += 1
            if is_my_team:
                my_keepers.append(raw_name)

    db.commit()

    return {
        "status": "ok",
        "keepers_found": keepers_found,
        "keepers_inserted": keepers_inserted,
        "keepers_skipped": keepers_skipped,
        "my_keepers": my_keepers,
        "player_pool_ready": True,
        "message": (
            f"Keeper sweep complete. {keepers_inserted} new keepers loaded "
            f"across {len(all_rosters)} teams. Player pool clean for draft."
        ),
    }


# ============================================================================
# LINEUP
# ============================================================================

@router.get("/api/fantasy/lineup/current", response_model=DailyLineupResponse)
async def get_fantasy_lineup_current(
    use_smart_selector: bool = True,
    force_stale: bool = Query(True, description="Allow lineup generation even when projection freshness SLA is violated."),
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """Return daily lineup recommendations for today (ET)."""
    from datetime import date as date_type
    _today = today_et().strftime("%Y-%m-%d")
    return await get_fantasy_lineup_recommendations(
        lineup_date=_today,
        use_smart_selector=use_smart_selector,
        force_stale=force_stale,
        db=db,
        user=user,
    )


@router.get("/api/fantasy/lineup-cards", tags=["lineups"])
def get_lineup_cards(
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format; defaults to today ET"),
    db: Session = Depends(get_db),
):
    """
    Return confirmed MLB starting lineups for the given date.

    Fetches from MLB Stats API (/api/v1/schedule with lineup hydration).
    SLA: Data should be fresh by 10:00 AM ET daily.

    Returns lineup cards keyed by game_pk plus freshness/SLA metadata.
    """
    from backend.fantasy_baseball.probable_pitcher_fallback import (
        fetch_daily_lineups,
        get_lineup_card_freshness,
    )

    target_date = date or today_et().strftime("%Y-%m-%d")

    try:
        datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format: {target_date}. Use YYYY-MM-DD.",
        ) from exc

    lineups = fetch_daily_lineups(target_date)
    freshness = get_lineup_card_freshness(target_date)

    return {
        "date": target_date,
        "game_count": len(lineups),
        "lineup_cards": lineups,
        "freshness": freshness,
    }


@router.get("/api/fantasy/lineup/{lineup_date}", response_model=DailyLineupResponse)
async def get_fantasy_lineup_recommendations(
    lineup_date: str,
    use_smart_selector: bool = True,
    force_stale: bool = Query(True, description="Allow lineup generation even when projection freshness SLA is violated."),
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """Return daily lineup recommendations for a given date."""
    from datetime import date as date_type
    try:
        ld = date_type.fromisoformat(lineup_date)
    except ValueError:
        raise HTTPException(status_code=422, detail="lineup_date must be YYYY-MM-DD")

    lineup_warnings: list = _enforce_projection_freshness(
        consumer=f"lineup optimizer for {lineup_date}",
        force_stale=force_stale,
    )

    _lineup_roster: list = []
    try:
        _lineup_client = get_yahoo_client()
        _lineup_roster = _lineup_client.get_roster()
    except Exception as _exc:
        logger.warning("Could not fetch Yahoo roster for lineup optimizer: %s", _exc)

    if _lineup_roster:
        _lineup_ids = _resolve_roster_player_bdl_ids(db, _lineup_roster)
        for _player in _lineup_roster:
            _player_key = _player.get("player_key")
            if _player_key in _lineup_ids:
                _player["bdl_player_id"] = _lineup_ids[_player_key].get("bdl_id")
                _player["mlbam_id"] = _lineup_ids[_player_key].get("mlbam_id")

    from backend.fantasy_baseball.daily_lineup_optimizer import normalize_team_abbr

    _lineup_projections: list = []
    if _lineup_roster:
        roster_teams = [normalize_team_abbr(p.get("team", "")) for p in _lineup_roster if p.get("team")]
        logger.info("[LINEUP_DEBUG] Yahoo roster teams: %s", roster_teams)
        try:
            from backend.fantasy_baseball.player_board import get_or_create_projection as _get_lineup_proj
            _lineup_projections = [_get_lineup_proj(p) for p in _lineup_roster]
        except Exception as _exc:
            logger.warning("Could not load player board projections for lineup: %s", _exc)

    _injury_overlays_by_key = load_injury_overlays_for_yahoo_players(db, _lineup_roster) if _lineup_roster else {}
    _injury_lookup: dict = {}
    for p in _lineup_roster:
        _pname = p.get("name")
        if not _pname:
            continue
        _overlay = _injury_overlays_by_key.get(p.get("player_key") or "")
        _raw_status = p.get("status") if isinstance(p.get("status"), str) else None
        _injury_lookup[_pname.lower()] = getattr(_overlay, "status", None) or _raw_status

    _name_to_player_key: dict = {
        p.get("name", "").strip().lower(): p.get("player_key", "")
        for p in _lineup_roster
        if p.get("name") and p.get("player_key")
    }

    # Lookup tables for lineup schema enrichment fields.
    # eligible_positions comes from Yahoo roster metadata parsed by _parse_player().
    # bdl_player_id / mlbam_id are only present if the roster payload was enriched
    # by the DB hydration step (they are not in raw Yahoo data), so most entries
    # will be None — consistent with Optional[int] typing.
    _eligible_positions_lookup: dict = {
        p.get("name", "").strip().lower(): p.get("positions") or []
        for p in _lineup_roster
        if p.get("name")
    }
    _bdl_id_lookup: dict = {
        p.get("name", "").strip().lower(): p.get("bdl_player_id")
        for p in _lineup_roster
        if p.get("name") and p.get("bdl_player_id")
    }
    _mlbam_id_lookup: dict = {
        p.get("name", "").strip().lower(): p.get("mlbam_id")
        for p in _lineup_roster
        if p.get("name") and p.get("mlbam_id")
    }

    batters: list = []

    team_odds: dict = {}
    _games: list = []
    try:
        from backend.fantasy_baseball.smart_lineup_selector import get_smart_selector
        _smart_sel = get_smart_selector()

        _games = _smart_sel.base_optimizer.fetch_mlb_odds(lineup_date)

        if not _games:
            try:
                next_day = (datetime.fromisoformat(lineup_date) + timedelta(days=1)).strftime("%Y-%m-%d")
                _games = _smart_sel.base_optimizer.fetch_mlb_odds(next_day)
                if _games:
                    logger.info("[LINEUP_DEBUG] Found games on next day (%s) due to UTC timezone", next_day)
            except Exception:
                pass

        logger.info("[LINEUP_DEBUG] Fetched %d games from Odds API", len(_games))
        team_odds = _smart_sel.base_optimizer._build_team_odds_map(_games)
        logger.info("[LINEUP_DEBUG] team_odds keys: %s", list(team_odds.keys()))
        for g in _games:
            if hasattr(g, 'commence_time') and g.commence_time:
                try:
                    start_dt = datetime.fromisoformat(g.commence_time.replace('Z', '+00:00'))
                    if g.home_abbrev in team_odds:
                        team_odds[g.home_abbrev]['start_time'] = start_dt
                    if g.away_abbrev in team_odds:
                        team_odds[g.away_abbrev]['start_time'] = start_dt
                except Exception:
                    pass
    except Exception as e:
        logger.warning("Could not fetch game odds: %s", e)

    def _get_game_context(team: str):
        team_norm = normalize_team_abbr(team)
        if team_norm in team_odds:
            opp = team_odds[team_norm].get("opponent", "")
            start = team_odds[team_norm].get("start_time")
            opp_impl = 4.5
            if opp and opp in team_odds:
                opp_impl = team_odds[opp].get("implied_runs", 4.5)
            return opp, start, opp_impl
        if team_odds:
            logger.info("[LINEUP_DEBUG] Team '%s' has no game today (not in %d teams with games)", team_norm, len(team_odds))
        else:
            logger.warning("[LINEUP_DEBUG] Team '%s' not found - no games data available", team_norm)
        return "", None, 4.5

    def _resolve_assignment_position(assignment: dict) -> str:
        positions = [str(pos) for pos in (assignment.get("positions") or []) if pos]
        slot = assignment.get("slot")
        if positions:
            return positions[0]
        if slot and slot not in {"BN", "Util"}:
            return str(slot)
        return "?"

    if use_smart_selector and _lineup_roster and _lineup_projections:
        try:
            from backend.fantasy_baseball import SmartLineupSelector, get_smart_selector
            from backend.fantasy_baseball.category_tracker import get_category_tracker

            smart_selector = get_smart_selector()

            category_needs = []
            try:
                tracker = get_category_tracker()
                category_needs = tracker.get_category_needs()
                if category_needs:
                    logger.info("Category needs: %s", [(c.category, c.needed) for c in category_needs])
            except Exception as e:
                logger.warning("Could not fetch category needs: %s", e)

            assignments, lineup_warnings = smart_selector.solve_smart_lineup(
                roster=_lineup_roster,
                projections=_lineup_projections,
                game_date=lineup_date,
                category_needs=category_needs,
            )

            batters = []
            for a in assignments:
                team = a.get("team", "")
                opp, start, opp_impl = _get_game_context(team)
                team_info = team_odds.get(normalize_team_abbr(team), {})
                _pname = a["player_name"]
                _slot_status = "START" if a["slot"] != "BN" else "BENCH"
                batters.append(LineupPlayerOut(
                    player_id=a["player_id"] or _pname,
                    player_key=_name_to_player_key.get(_pname.lower().strip(), "") or None,
                    name=_pname,
                    team=team,
                    position=_resolve_assignment_position(a),
                    implied_runs=round(float(team_info.get("implied_runs", a.get("implied_runs", 4.5))), 2),
                    park_factor=round(float(team_info.get("park_factor", a.get("park_factor", 1.0))), 3),
                    lineup_score=round(a.get("smart_score", 0), 3),
                    start_time=start,
                    opponent=opp,
                    status=_slot_status,
                    lineup_status=_slot_status,
                    assigned_slot=a["slot"],
                    has_game=bool(team_info) or a.get("has_game", False),
                    eligible_positions=_eligible_positions_lookup.get(_pname.lower().strip()) or None,
                    game_time=start,
                    bdl_player_id=_bdl_id_lookup.get(_pname.lower().strip()),
                    mlbam_id=_mlbam_id_lookup.get(_pname.lower().strip()),
                    injury_status=_injury_lookup.get(_pname.lower()),
                ))

            logger.info("SmartLineupSelector produced %d assignments for %s", len(batters), lineup_date)

        except Exception as _exc:
            logger.warning("SmartLineupSelector failed, falling back to base optimizer: %s", _exc)
            lineup_warnings.append(f"Smart selector unavailable: {_exc}")
            batters = []

    if not batters and _lineup_roster and _lineup_projections:
        try:
            optimizer = get_lineup_optimizer()
            solved_slots, lineup_warnings = optimizer.solve_lineup(
                roster=_lineup_roster,
                projections=_lineup_projections,
                game_date=lineup_date,
                db=db,
            )
            batters = []
            for s in solved_slots:
                opp, start, opp_impl = _get_game_context(s.player_team)
                team_info = team_odds.get(normalize_team_abbr(s.player_team), {})
                _sname = s.player_name
                _s_slot_status = "START" if s.slot != "BN" else "BENCH"
                batters.append(LineupPlayerOut(
                    player_id=_sname,
                    player_key=_name_to_player_key.get(_sname.lower().strip(), "") or None,
                    name=_sname,
                    team=s.player_team,
                    position=s.positions[0] if s.positions else "?",
                    implied_runs=round(float(team_info.get("implied_runs", s.implied_runs)), 2),
                    park_factor=round(float(team_info.get("park_factor", s.park_factor)), 3),
                    lineup_score=round(s.lineup_score, 3),
                    start_time=start,
                    opponent=opp,
                    status=_s_slot_status,
                    lineup_status=_s_slot_status,
                    assigned_slot=s.slot,
                    has_game=bool(team_info) or s.has_game,
                    eligible_positions=_eligible_positions_lookup.get(_sname.lower().strip()) or None,
                    game_time=start,
                    bdl_player_id=_bdl_id_lookup.get(_sname.lower().strip()),
                    mlbam_id=_mlbam_id_lookup.get(_sname.lower().strip()),
                    injury_status=_injury_lookup.get(_sname.lower()),
                ))
        except Exception as _exc:
            logger.warning("solve_lineup failed, falling back to score-rank: %s", _exc)
            lineup_warnings.append(f"Constraint solver unavailable: {_exc}")

    if not batters:
        optimizer = get_lineup_optimizer()
        report = optimizer.build_daily_report(
            game_date=lineup_date,
            roster=_lineup_roster or None,
            projections=_lineup_projections or None,
        )
        batters = []
        for i, b in enumerate(report.get("batter_rankings", [])):
            team = b.get("team", "")
            opp, start, opp_impl = _get_game_context(team)
            team_info = team_odds.get(normalize_team_abbr(team), {})
            _b_name = b.get("name", "")
            _rank_slot_status = "START" if i < 9 else "BENCH"
            batters.append(LineupPlayerOut(
                player_id=str(b.get("player_id", _b_name)),
                player_key=_name_to_player_key.get(_b_name.lower().strip(), "") or None,
                name=_b_name,
                team=team,
                position=(b.get("positions") or ["OF"])[0],
                implied_runs=round(float(team_info.get("implied_runs", b.get("implied_team_runs", 4.5))), 2),
                park_factor=float(team_info.get("park_factor", b.get("park_factor", 1.0))),
                lineup_score=float(b.get("score", 0)),
                start_time=start,
                opponent=opp,
                status=_rank_slot_status,
                lineup_status=_rank_slot_status,
                assigned_slot=None,
                has_game=bool(team_info) or b.get("has_game", True),
                eligible_positions=_eligible_positions_lookup.get(_b_name.lower().strip()) or None,
                game_time=start,
                bdl_player_id=_bdl_id_lookup.get(_b_name.lower().strip()),
                mlbam_id=_mlbam_id_lookup.get(_b_name.lower().strip()),
                injury_status=_injury_lookup.get(_b_name.lower()),
            ))

    for _b in batters:
        if _b.status == "START" and not _b.opponent:
            _b.status = "BENCH"
            _b.lineup_status = "BENCH"
            _msg = f"{_b.name} moved to BENCH -- no game data for {_b.team} on {lineup_date} (Odds API coverage gap)"
            if _msg not in lineup_warnings:
                lineup_warnings.append(_msg)
            logger.warning("No-game START override: %s (%s)", _b.name, _b.team)

    report = {"games": []}
    if _games:
        report["games"] = [
            {
                "home": g.home_abbrev,
                "away": g.away_abbrev,
                "start_time": g.commence_time,
            }
            for g in _games
        ]
    elif 'optimizer' in locals():
        try:
            _fallback_report = optimizer.build_daily_report(game_date=lineup_date)
            report = {"games": _fallback_report.get("games", [])}
        except Exception:
            pass

    if 'optimizer' not in locals():
        optimizer = get_lineup_optimizer()

    pitchers: list = []
    try:
        flagged_pitchers = optimizer.flag_pitcher_starts(
            roster=_lineup_roster,
            game_date=lineup_date,
        )
        logger.info("flag_pitcher_starts returned %d pitchers", len(flagged_pitchers))

        for p in flagged_pitchers:
            team_raw = p.get("team", "")
            team = normalize_team_abbr(team_raw)
            is_sp = p.get("pitcher_slot") == "SP"
            has_start = p.get("has_start", False)

            opponent = ""
            opp_implied = 4.5
            park_factor = 1.0
            sp_score = 0.0
            start_time = None

            if team in team_odds:
                opp = team_odds[team].get("opponent", "")
                opponent = opp
                if opp in team_odds:
                    opp_implied = team_odds[opp].get("implied_runs", 4.5)
                park_factor = team_odds[team].get("park_factor", 1.0)
                start_time = team_odds[team].get("start_time")

            if is_sp and has_start:
                # Enhanced sp_score using run environment data (PR-22)
                # Components:
                # 1. Opponent weakness (40%): Lower opp implied runs = better
                # 2. Park factor (25%): Pitcher-friendly park = better  
                # 3. Run environment (35%): Low total + high win prob = better
                
                opp_factor = max(0, 5.0 - opp_implied) * 0.8  # 0-4 points
                park_factor_score = max(0, (2.0 - park_factor) * 2.5)  # 0-2.5 points
                
                # Get run environment score from team_odds (0-10 scale)
                run_env_score = team_odds.get(team, {}).get("run_environment_score", 5.0)
                # Normalize to 0-3.5 points (pitcher wants HIGH run_env_score)
                run_env_component = (run_env_score / 10.0) * 3.5
                
                sp_score = round(opp_factor + park_factor_score + run_env_component, 2)
                sp_score = min(10, max(0, sp_score))  # Clamp to 0-10

            status = "START" if has_start else "NO_START"
            if not is_sp:
                status = "RP"

            pitcher_type = "SP" if is_sp else "RP"

            _pname_raw = p.get("name", "")
            pitchers.append(StartingPitcherOut(
                player_id=p.get("player_key") or _pname_raw,
                player_key=p.get("player_key") or _name_to_player_key.get(_pname_raw.lower().strip(), "") or None,
                name=_pname_raw,
                team=team,
                pitcher_type=pitcher_type,
                opponent=opponent,
                opponent_implied_runs=round(opp_implied, 2),
                park_factor=round(park_factor, 3),
                sp_score=round(sp_score, 3),
                start_time=start_time,
                status=status,
                has_game=has_start or bool(opponent),
                is_two_start=bool(p.get("is_two_start", False)),
                game_time=start_time,
                bdl_player_id=_bdl_id_lookup.get(_pname_raw.lower().strip()),
                mlbam_id=_mlbam_id_lookup.get(_pname_raw.lower().strip()),
            ))

        sp_no_start = [p for p in pitchers if p.status == "NO_START"]
        if sp_no_start:
            lineup_warnings.append(
                f"{len(sp_no_start)} SP(s) have no start today: "
                + ", ".join(p.name for p in sp_no_start[:3])
            )

    except Exception as _exc:
        logger.warning("flag_pitcher_starts failed: %s", _exc)

    if 'report' not in locals():
        report = {"games": []}

    games_list = report.get("games", [])
    if len(games_list) == 0:
        if not _games:
            lineup_warnings.insert(0,
                "No games found for this date -- Odds API may not have data yet "
                f"(requested: {lineup_date}). Lineup ranked by projections only."
            )
        else:
            lineup_warnings.insert(0,
                f"Odds API returned {_games} games but no game data available. "
                "Lineup ranked by projections only."
            )

    _BENCH_SLOTS = {"BN", None}
    _batter_active = [b for b in batters if b.assigned_slot not in _BENCH_SLOTS]
    _pitcher_active = [p for p in pitchers if p.status == "START"]
    if len(_batter_active) < 6:
        lineup_warnings.append(
            f"Only {len(_batter_active)} active batter slots filled -- "
            "check bench/IL for promotable players."
        )
    if len(_pitcher_active) < 2:
        lineup_warnings.append(
            f"Only {len(_pitcher_active)} active pitcher slots filled -- "
            "consider streaming a SP."
        )

    _POSITION_ORDER = {
        "C": 0, "1B": 1, "2B": 2, "3B": 3, "SS": 4,
        "OF": 5, "Util": 6, "BN": 7, None: 8, "?": 9
    }
    batters = sorted(
        batters,
        key=lambda b: (
            _POSITION_ORDER.get(b.assigned_slot, 99),
            -b.lineup_score
        )
    )

    try:
        pos_map: dict = {}
        for _b in batters:
            if _b.assigned_slot and _b.assigned_slot != "BN":
                pos_map[_b.assigned_slot] = _b.name
        _sp_idx = 0
        for _p in pitchers:
            if _p.status == "START":
                _sp_idx += 1
                pos_map[f"SP{_sp_idx}"] = _p.name
        _projected = float(
            sum(_b.lineup_score for _b in batters if _b.assigned_slot and _b.assigned_slot != "BN")
            + sum(_p.sp_score for _p in pitchers if _p.status == "START")
        )
        _notes_str = "; ".join(lineup_warnings) if lineup_warnings else None
        _existing_rec = db.query(FantasyLineup).filter_by(
            lineup_date=ld, platform="yahoo_recommendation"
        ).first()
        if _existing_rec:
            _existing_rec.positions = pos_map
            _existing_rec.projected_points = _projected
            _existing_rec.notes = _notes_str
            _existing_rec.updated_at = datetime.now(ZoneInfo("America/New_York"))
        else:
            db.add(FantasyLineup(
                lineup_date=ld,
                platform="yahoo_recommendation",
                positions=pos_map,
                projected_points=_projected,
                notes=_notes_str,
            ))
        db.commit()
    except Exception as _persist_err:
        db.rollback()
        logger.warning("lineup recommendation persistence failed: %s", _persist_err)

    return DailyLineupResponse(
        date=ld,
        batters=batters,
        pitchers=pitchers,
        games_count=len(games_list),
        no_games_today=len(games_list) == 0,
        lineup_warnings=lineup_warnings,
    )


@router.get("/api/fantasy/briefing/{briefing_date}")
async def get_daily_briefing(
    briefing_date: str,
    record_decisions: bool = Query(True, description="Record decisions for accuracy tracking"),
    user: str = Depends(verify_api_key),
):
    """Get elite manager daily briefing."""
    from datetime import date as date_type
    try:
        _ = date_type.fromisoformat(briefing_date)
    except ValueError:
        raise HTTPException(status_code=422, detail="briefing_date must be YYYY-MM-DD")

    roster: list = []
    try:
        client = get_yahoo_client()
        roster = client.get_roster()
    except Exception as e:
        logger.warning("Could not fetch roster: %s", e)
        raise HTTPException(status_code=503, detail="Yahoo API unavailable")

    projections: list = []
    try:
        from backend.fantasy_baseball.player_board import get_or_create_projection
        projections = [get_or_create_projection(p) for p in roster]
    except Exception as e:
        logger.warning("Could not load projections: %s", e)

    try:
        from backend.fantasy_baseball.daily_briefing import get_briefing_generator, CATEGORY_DISPLAY_NAMES
        generator = get_briefing_generator(record_decisions=record_decisions)
        briefing = generator.generate(
            roster=roster,
            projections=projections,
            game_date=briefing_date,
        )

        # Statcast enrichment for briefing cards (PR-15, non-blocking)
        _br_bat: dict = {}
        _br_pit: dict = {}
        try:
            from backend.fantasy_baseball.pybaseball_loader import (
                load_pybaseball_batters as _lb,
                load_pybaseball_pitchers as _lp,
                match_yahoo_to_statcast as _ms,
            )
            from backend.fantasy_baseball.statcast_loader import build_statcast_signals as _bss
            _br_bat = _lb(2026)
            _br_pit = _lp(2026)
        except Exception:
            pass

        def _enrich_card(card: dict, roster_player_positions: list) -> dict:
            """Add statcast_stats and statcast_signals to a briefing card."""
            name = card.get("name", "")
            if not name or (not _br_bat and not _br_pit):
                return card
            is_pit = bool(roster_player_positions) and roster_player_positions[0] in ("SP", "RP", "P")
            try:
                if not is_pit and _br_bat:
                    ck = _ms(name, _br_bat)
                    if ck:
                        sb = _br_bat[ck]
                        card["statcast_stats"] = {
                            "xwoba": round(sb.xwoba, 3) if sb.xwoba else None,
                            "barrel_pct": round(sb.barrel_pct, 1) if sb.barrel_pct else None,
                            "exit_velo_avg": round(sb.exit_velo_avg, 1) if sb.exit_velo_avg else None,
                            "hard_hit_pct": round(sb.hard_hit_pct, 1) if sb.hard_hit_pct else None,
                            "wrc_plus": round(sb.wrc_plus, 0) if sb.wrc_plus else None,
                            "sprint_speed": round(sb.sprint_speed, 1) if sb.sprint_speed else None,
                        }
                elif is_pit and _br_pit:
                    ck = _ms(name, _br_pit)
                    if ck:
                        sp = _br_pit[ck]
                        card["statcast_stats"] = {
                            "xera": round(sp.xera, 2) if sp.xera else None,
                            "stuff_plus": round(sp.stuff_plus, 0) if sp.stuff_plus else None,
                            "location_plus": round(sp.location_plus, 0) if sp.location_plus else None,
                            "whiff_pct": round(sp.whiff_pct, 1) if sp.whiff_pct else None,
                            "fb_velo_avg": round(sp.fb_velo_avg, 1) if sp.fb_velo_avg else None,
                        }
                sigs, _ = _bss(name, is_pit)
                if sigs:
                    card["statcast_signals"] = sigs
            except Exception:
                pass
            return card

        # Build position lookup from roster for enrichment
        _pos_by_name: dict = {}
        for rp in roster:
            _pos_by_name[(rp.get("name") or "").strip()] = rp.get("positions") or []

        def _enrich_cards(cards: list) -> list:
            return [_enrich_card(c, _pos_by_name.get(c.get("name", ""), [])) for c in cards]

        return {
            "date": briefing_date,
            "generated_at": briefing.generated_at.isoformat(),
            "strategy": briefing.strategy,
            "risk_profile": briefing.risk_profile,
            "overall_confidence": round(briefing.overall_confidence / 100, 4),
            "summary": {
                "total_decisions": briefing.total_decisions,
                "easy_decisions": briefing.easy_decisions,
                "tough_decisions": briefing.tough_decisions,
                "monitor_count": briefing.monitor_count,
            },
            "categories": [
                {
                    "name": CATEGORY_DISPLAY_NAMES.get(c.category, c.category),
                    "category": c.category,
                    "current": c.current,
                    "opponent": c.opponent,
                    "status": c.status,
                    "urgency": c.urgency,
                }
                for c in briefing.categories
            ],
            "starters": _enrich_cards([p.to_card() for p in briefing.start_recommendations]),
            "bench": _enrich_cards([p.to_card() for p in briefing.bench_recommendations[:5]]),
            "monitor": _enrich_cards([p.to_card() for p in briefing.monitor_list]),
            "alerts": briefing.alerts,
            "_meta": {
                "decisions_recorded": record_decisions,
                "decisions_count": briefing.total_decisions,
            }
        }
    except Exception as e:
        logger.exception("Failed to generate briefing")
        raise HTTPException(status_code=500, detail=f"Briefing generation failed: {str(e)}")


# ============================================================================
# WAIVER WIRE
# ============================================================================

@router.get("/api/fantasy/waiver", response_model=WaiverWireResponse)
async def get_fantasy_waiver_recommendations(
    position: Optional[str] = Query(None),
    sort: str = Query("need_score"),
    min_z_score: Optional[float] = Query(None),
    max_percent_owned: float = Query(100.0),
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=10, le=100),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Return waiver wire recommendations. Pulls real free agents from Yahoo API."""
    from datetime import date as date_type, timedelta
    from backend.schemas import WaiverPlayerOut

    today = date_type.today()
    week_end = today + timedelta(days=(6 - today.weekday()))

    matchup_opponent = "TBD"
    top_available: list = []
    two_start_pitchers: list = []
    category_deficits: list = []
    _closer_alert: Optional[str] = None
    _il_info: dict = {"used": 0, "total": 2, "available": 0}
    _faab_balance: Optional[float] = None
    _roster_context: dict = {}

    try:
        client = get_yahoo_client()
        my_team_key = os.getenv("YAHOO_TEAM_KEY", "")
        if not my_team_key:
            try:
                my_team_key = client.get_my_team_key()
            except Exception:
                my_team_key = ""

        my_roster: list = []
        if my_team_key:
            try:
                my_roster = client.get_roster()
            except Exception:
                pass

        try:
            _faab_balance = client.get_faab_balance()
        except Exception:
            pass

        _fa_start = (page - 1) * per_page
        _yahoo_pos = position if position and position.upper() != "ALL" else ""

        # Fetch free agents with defensive error handling
        try:
            free_agents = client.get_free_agents(
                position=_yahoo_pos, start=_fa_start, count=per_page
            )
            logger.info("waiver: fetched %d free agents for position=%s start=%d", len(free_agents) if free_agents else 0, _yahoo_pos, _fa_start)
        except Exception as fa_err:
            logger.error("waiver: get_free_agents failed: %s", fa_err, exc_info=True)
            free_agents = []

        if not free_agents:
            logger.warning("waiver: no free agents returned from Yahoo API, returning empty response")

        # Augment with dedicated batter pool when no position filter is active.
        # Yahoo's sort=AR returns pitchers first (5× more pitchers than batters
        # are available in any league), so page 1 without augmentation is ~90%
        # pitchers regardless of team needs.
        if not _yahoo_pos and free_agents is not None:
            try:
                _batter_fas = client.get_free_agents(position="OF", start=0, count=per_page)
                if _batter_fas:
                    _existing_keys = {p.get("player_key") for p in free_agents}
                    _new_batters = [p for p in _batter_fas if p.get("player_key") not in _existing_keys]
                    free_agents = free_agents + _new_batters
                    logger.info(
                        "waiver: augmented with %d batters (total pool=%d)",
                        len(_new_batters),
                        len(free_agents),
                    )
            except Exception as _bat_err:
                logger.warning("waiver: batter augment failed (non-fatal): %s", _bat_err)

        # Fetch scoreboard once and reuse for both opponent resolution and
        # category_deficits — the previous implementation fetched twice and
        # used a shallow 2-level parser that missed nested team_key/team_stats,
        # causing matchup_opponent="TBD" and category_deficits=[] regressions.
        try:
            matchups_scoreboard = client.get_scoreboard()
        except Exception as _sb_err:
            logger.warning("waiver get_scoreboard failed (non-fatal): %s", _sb_err)
            matchups_scoreboard = []

        my_matchup_teams: list = []
        for matchup_teams in _iter_scoreboard_matchup_teams(matchups_scoreboard):
            my_tuple = None
            for t in matchup_teams:
                t_key = t[0]
                if not t_key:
                    continue
                if t_key == my_team_key or (
                    my_team_key and (t_key in my_team_key or my_team_key in t_key)
                ):
                    my_tuple = t
                    break
            if my_tuple is not None:
                opp_tuple = next((t for t in matchup_teams if t[0] != my_tuple[0]), None)
                if opp_tuple is not None:
                    matchup_opponent = opp_tuple[1] or "TBD"
                    my_matchup_teams = [my_tuple, opp_tuple]
                break

        sid_map: dict = dict(_YAHOO_STAT_FALLBACK)
        try:
            _settings_waiver = client.get_league_settings()
            _stat_cats_waiver = (
                _settings_waiver
                .get("settings", [{}])[0]
                .get("stat_categories", {})
                .get("stats", [])
            )
            _waiver_stat_entries: list = []
            for _entry_w in _stat_cats_waiver:
                if isinstance(_entry_w, dict):
                    _s_w = _entry_w.get("stat", {})
                    _sid_w = str(_s_w.get("stat_id", ""))
                    _abbr_w = (
                        _s_w.get("display_name")
                        or _s_w.get("abbreviation")
                        or _s_w.get("name")
                        or _sid_w
                    )
                    _pos_w = _s_w.get("position_type", "")
                    if _sid_w:
                        _waiver_stat_entries.append((_sid_w, _abbr_w, _pos_w))
            _waiver_abbr_pos: dict = {}
            for _sid_w, _abbr_w, _pos_w in _waiver_stat_entries:
                _waiver_abbr_pos.setdefault(_abbr_w, set()).add(_pos_w)
            _P_RENAME = {"HR": "HRA", "K": "K(P)"}
            _B_RENAME = {"K": "K(B)", "HR": "HR"}
            for _sid_w, _abbr_w, _pos_w in _waiver_stat_entries:
                _final = _abbr_w
                if len(_waiver_abbr_pos.get(_abbr_w, set())) > 1:
                    if _pos_w == "P" and _abbr_w in _P_RENAME:
                        _final = _P_RENAME[_abbr_w]
                    elif _pos_w == "B" and _abbr_w in _B_RENAME:
                        _final = _B_RENAME[_abbr_w]
                sid_map[_sid_w] = _final
        except Exception as _e_sid:
            logger.warning("get_league_settings failed in waiver sid_map build (using fallback): %s", _e_sid)

        try:
            from backend.schemas import CategoryDeficitOut

            if my_matchup_teams:
                my_tuple, opp_tuple = my_matchup_teams

                # Display-only Yahoo stats that are NOT scoring categories.
                # H_AB is stat ID 60 ("8/20" string), IP and GS are volume
                # indicators only — none of these are H2H matchup cats.
                _NON_SCORING_DISPLAY = frozenset({"H_AB", "IP", "GS"})

                def _stats_dict_from_raw(raw_stats_list: list) -> dict:
                    out: dict = {}
                    for st in raw_stats_list:
                        if not isinstance(st, dict):
                            continue
                        stobj = st.get("stat", {})
                        if not isinstance(stobj, dict):
                            continue
                        sid_k = str(stobj.get("stat_id", ""))
                        if not sid_k:
                            continue
                        key2 = sid_map.get(sid_k, sid_k)
                        # Drop non-scoring Yahoo stat_ids that fall through
                        # translation as bare numeric strings.
                        if isinstance(key2, str) and key2.isdigit():
                            continue
                        # Drop display-only stats that are not H2H scoring cats.
                        if key2 in _NON_SCORING_DISPLAY:
                            continue
                        try:
                            out[key2] = float(stobj.get("value", 0) or 0)
                        except (TypeError, ValueError):
                            out[key2] = 0.0
                    return out

                my_stats = _stats_dict_from_raw(my_tuple[2])
                opp_stats = _stats_dict_from_raw(opp_tuple[2])

                for cat, my_val in my_stats.items():
                    opp_val = opp_stats.get(cat, 0.0)
                    result = compare_category(cat, my_val, opp_val)
                    category_deficits.append(
                        CategoryDeficitOut(
                            category=cat,
                            my_total=my_val,
                            opponent_total=opp_val,
                            deficit=result.gap,
                            winning=(result.verdict == "W"),
                        )
                    )
        except Exception as _cd_err:
            logger.warning("waiver category_deficits build failed (non-fatal): %s", _cd_err)
            category_deficits = []

        from backend.fantasy_baseball.player_board import get_or_create_projection as _get_proj

        # Load Statcast from database (uses fixed queries from Bugs 2 & 3)
        from backend.fantasy_baseball.statcast_loader import (
            get_statcast_batter as _get_sc_bat,
            get_statcast_pitcher as _get_sc_pit,
            build_statcast_signals as _build_sc_sig,
        )

        def _hot_cold_flag(cat_contributions: dict) -> Optional[str]:
            scores = list(cat_contributions.values())
            if not scores:
                return None
            avg = sum(scores) / len(scores)
            if avg > 0.75:
                return "HOT"
            if avg < -0.5:
                return "COLD"
            return None

        # Daily availability blacklist: admin-seeded player_keys confirmed OUT or on day-off.
        _blacklist_keys: set = set()
        try:
            from backend.models import DailyAvailabilityOverride as _DAO
            from zoneinfo import ZoneInfo as _ZI
            _bl_today = datetime.now(_ZI("America/New_York")).date()
            _blacklist_keys = {
                r.player_key
                for r in db.query(_DAO.player_key)
                    .filter(_DAO.game_date == _bl_today, _DAO.status.in_(["OUT", "DAY_OFF"]))
                    .all()
            }
        except Exception:
            pass  # non-fatal: empty blacklist on any error

        def _to_waiver_player(p: dict) -> WaiverPlayerOut:
            positions = p.get("positions") or []
            name = (p.get("name") or "").strip()
            board_player = _get_proj(p)

            _raw_stats: dict = p.get("stats") or {}
            _translated_stats: dict = {}
            for k, v in _raw_stats.items():
                _translated_key = sid_map.get(k, k)
                if _translated_key == "K(P)":
                    _translated_key = "K"
                # Drop untranslated numeric stat_ids (e.g. "38") — Yahoo's
                # stats batch can include non-scoring stat_ids not present
                # in YAHOO_ID_INDEX or league settings, which would surface
                # as opaque "38": "0" entries in the API response.
                if isinstance(_translated_key, str) and _translated_key.isdigit():
                    continue
                _translated_stats[_translated_key] = v

            # April 21 Issue 5 fix: Remove position-inappropriate stats to prevent
            # batters from carrying pitcher-only stats (IP, W, GS, ERA, WHIP, etc.)
            # and pitchers from carrying batter-only stats (R, H, HR_B, RBI, etc.)
            _PITCHER_ONLY_STATS = frozenset({
                "IP", "W", "L", "ERA", "WHIP", "K_9", "QS", "HR_P", "K_P", "NSV", "GS"
            })
            _BATTER_ONLY_STATS = frozenset({
                "R", "H", "HR_B", "RBI", "TB", "AVG", "OPS", "NSB", "K_B", "SB"
            })
            _primary_pos = positions[0] if positions else None
            if _primary_pos in ("SP", "RP", "P"):
                # Pitcher: remove batter-only stats
                _translated_stats = {
                    k: v for k, v in _translated_stats.items()
                    if k not in _BATTER_ONLY_STATS
                }
            elif _primary_pos not in ("P",):
                # Batter (or utility/etc. without explicit pitcher position): remove pitcher-only stats
                _translated_stats = {
                    k: v for k, v in _translated_stats.items()
                    if k not in _PITCHER_ONLY_STATS
                }

            _is_reliever = "RP" in positions
            _raw_nsv = 0.0
            if _is_reliever:
                if "NSV" in _translated_stats:
                    try:
                        _raw_nsv = round(float(_translated_stats["NSV"]), 1)
                    except (TypeError, ValueError):
                        pass
                elif "83" in _raw_stats:
                    try:
                        _raw_nsv = round(float(_raw_stats["83"]), 1)
                    except (TypeError, ValueError):
                        pass
                else:
                    _raw_nsv = round(float((board_player.get("proj") or {}).get("nsv", 0.0)), 1)

            # ── Player data extraction ────────────────────────────────────────────
            cat_scores = board_player.get("cat_scores", {})
            player_z = board_player.get("z_score", 0.0)

            # Use raw cat_scores for hot/cold flag (consistent with recommendations endpoint)
            contributions = {k: float(v) for k, v in cat_scores.items() if isinstance(v, (int, float))}

            # Hot/cold from BDL momentum (preferred) or z-score average (fallback)
            _hc: Optional[str] = None
            _bdl_id: Optional[int] = None
            try:
                _player_name_key = name.lower().strip()
                _bdl_id = _fa_name_to_bdl_id.get(_player_name_key)
                if _bdl_id and _bdl_id in _momentum_signal_by_bdl_id:
                    _mom_sig = _momentum_signal_by_bdl_id[_bdl_id]
                    _hc = "HOT" if _mom_sig in ("SURGING", "HOT") else (
                        "COLD" if _mom_sig in ("COLD", "COLLAPSING") else None
                    )
                else:
                    # Fallback: season z-score average (less accurate)
                    _hc = _hot_cold_flag(contributions) if contributions else None
            except Exception:
                pass

            # ── Injury status & availability ──────────────────────────────────────
            _status = p.get("status") or None
            _injury_note = p.get("injury_note") or None
            _injury_status = p.get("injury_status")
            _injury_timeline = None
            _overlay = _injury_overlays_by_key.get(p.get("player_key") or "")
            if _overlay is not None:
                _status = getattr(_overlay, "status", None) or _status
                _injury_note = _injury_note or getattr(_overlay, "note", None)
                _injury_status = getattr(_overlay, "status", None) or _injury_status
                _injury_timeline = getattr(_overlay, "return_timeline", None)

            _pkey = p.get("player_key") or ""
            if _pkey and _pkey in _blacklist_keys:
                _avail_note: Optional[str] = "NOT AVAILABLE TODAY — day off confirmed"
            elif _injury_status and any(kw in (_injury_status or "").upper() for kw in ("IL", "DL", "60-DAY", "15-DAY", "10-DAY")):
                _avail_note = "On IL — check IL slot availability"
            elif _injury_status and "DTD" in (_injury_status or "").upper():
                _avail_note = "DTD — confirm before adding"
            else:
                _avail_note = None

            # ── Statcast enrichment (BEFORE need_score calculation) ────────────────
            _sc_dict: dict | None = None
            _sc_sigs: list[str] = []
            _fa_is_pitcher = positions[0] in ("SP", "RP", "P") if positions else False
            try:
                if not _fa_is_pitcher:
                    _sb = _get_sc_bat(name)
                    if _sb:
                        _sc_dict = {
                            "xwoba": round(_sb.xwoba, 3) if _sb.xwoba else None,
                            "xwoba_diff": round(_sb.xwoba_diff, 3) if _sb.xwoba_diff is not None else None,
                            "barrel_pct": round(_sb.barrel_pct, 1) if _sb.barrel_pct is not None else None,
                            "exit_velo_avg": round(_sb.exit_velo_avg, 1) if _sb.exit_velo_avg is not None else None,
                            "hard_hit_pct": round(_sb.hard_hit_pct, 1) if _sb.hard_hit_pct is not None else None,
                            "wrc_plus": round(_sb.wrc_plus, 0) if _sb.wrc_plus is not None else None,
                            "sprint_speed": round(_sb.sprint_speed, 1) if _sb.sprint_speed is not None else None,
                        }
                else:
                    _sp = _get_sc_pit(name)
                    if _sp:
                        _sc_dict = {
                            "xera": round(_sp.xera, 2) if _sp.xera is not None else None,
                            "xera_diff": round(_sp.xera_diff, 2) if _sp.xera_diff is not None else None,
                            "stuff_plus": round(_sp.stuff_plus, 0) if _sp.stuff_plus is not None else None,
                            "location_plus": round(_sp.location_plus, 0) if _sp.location_plus is not None else None,
                            "whiff_pct": round(_sp.whiff_pct, 1) if _sp.whiff_pct is not None else None,
                            "barrel_allowed_pct": round(_sp.barrel_allowed_pct, 1) if _sp.barrel_allowed_pct is not None else None,
                            "xwoba_allowed": round(_sp.xwoba_allowed, 3) if _sp.xwoba_allowed is not None else None,
                        }
                if name:
                    _sigs, _ = _build_sc_sig(name, _fa_is_pitcher, p.get("percent_owned", 100.0))
                    _sc_sigs = _sigs
            except Exception:
                pass

            # Add HIGH_INJURY_RISK signal if injury penalty applies
            if _overlay:
                _penalty_base, _penalty_note = apply_injury_penalty(0.0, _overlay)
                if _penalty_note and "HIGH_INJURY_RISK" not in _sc_sigs:
                    _sc_sigs.append("HIGH_INJURY_RISK")

            # ── Need-score calculation (unified service with Statcast boost) ─────────
            from backend.services.need_score import (
                calculate_need_score as _calc_need_score,
                NeedScoreResult,
            )

            need_score_result: NeedScoreResult = _calc_need_score(
                player_cat_scores=cat_scores,
                player_z_score=player_z,
                category_deficits=category_deficits,
                n_cats=max(1, len(category_deficits)) if category_deficits else 1,
                statcast_signals=_sc_sigs,
                team_context=None,
                marginal_stats=None,
            )

            # Base need_score (pure category-aware, no Statcast)
            need_score = need_score_result.base_need_score
            # Statcast boost (transparent separate calculation)
            statcast_boost = need_score_result.statcast_boost
            # Adjusted need_score (base + Statcast boost)
            adjusted_need_score = need_score_result.adjusted_need_score

            # Apply injury penalty to final need_score
            need_score, _penalty_note = apply_injury_penalty(need_score, _overlay)

            # Blacklisted players get zero need_score
            if _pkey and _pkey in _blacklist_keys:
                need_score = 0.0

            # ── Stability metadata ──────────────────────────────────────────
            _pkey_stab = p.get("player_key") or ""
            _volatile = False
            _now_stab = datetime.now(ZoneInfo("America/New_York"))
            if _pkey_stab and _pkey_stab in _NEED_SCORE_HISTORY:
                _prev_score, _prev_ts = _NEED_SCORE_HISTORY[_pkey_stab]
                _age_min = (_now_stab - _prev_ts).total_seconds() / 60.0
                if _age_min < 60.0 and _prev_score > 0.1:
                    _delta_pct = abs(need_score - _prev_score) / _prev_score
                    if _delta_pct > 0.20:
                        _volatile = True
                        logger.warning(
                            "[waiver_volatility] %s: %.2f→%.2f (%.0f%% swing in %.0fm)",
                            name, _prev_score, need_score, _delta_pct * 100.0, _age_min,
                        )
            if _pkey_stab:
                _NEED_SCORE_HISTORY[_pkey_stab] = (need_score, _now_stab)

            # CI factor: steamer+statcast=15%, steamer=25%, proxy/draft=40%
            _proj_src = board_player.get("fusion_source") or (
                "proxy" if board_player.get("is_proxy") else "draft_board"
            )
            _ci_factors = {"steamer+statcast": 0.15, "steamer": 0.25}
            _need_ci: Optional[float] = round(need_score * _ci_factors.get(_proj_src, 0.40), 2) if need_score > 0 else None
            logger.debug(
                "[waiver_score] %s need=%.3f ci=±%.2f src=%s volatile=%s n_defs=%d",
                name, need_score, _need_ci or 0.0, _proj_src, _volatile, len(category_deficits),
            )

            # Suppress HOT for any injured/unavailable player (IL already filtered out;
            # DTD suppression prevents a misleading badge on players who may not play)
            _injury_upper = (_injury_status or "").upper()
            if _hc == "HOT" and (
                "DTD" in _injury_upper
                or "IL" in _injury_upper
                or "DL" in _injury_upper
            ):
                _hc = None

            # Small-sample flag: warn when season-to-date stats are too thin to trust.
            # Pitchers: use IP (Yahoo stat ID "50"); batters: estimate PA from H (stat "8").
            _small_sample: Optional[bool] = None
            if _fa_is_pitcher:
                _ip_ytd = float(_raw_stats.get("50") or 0.0)
                _small_sample = 0.0 < _ip_ytd < 30.0
            else:
                _h_ytd = float(_raw_stats.get("8") or 0.0)
                _pa_est = _h_ytd / 0.265 if _h_ytd > 0.0 else 0.0
                _small_sample = 0.0 < _pa_est < 100.0

            _starts = _pitcher_starts_by_name.get(name.lower().strip(), [])
            _start1_opp = _starts[0] if len(_starts) > 0 else None
            _start2_opp = _starts[1] if len(_starts) > 1 else None

            # Closer role classification from season saves (stat 57 / NSV)
            _closer_role: Optional[str] = None
            if _fa_is_pitcher and "RP" in positions:
                if _raw_nsv >= 2.0:
                    _closer_role = "CLOSER"
                else:
                    _closer_role = "NO_SAVE_ROLE"

            return WaiverPlayerOut(
                player_id=p.get("player_key") or "",
                name=name,
                team=p.get("team") or "",
                position=positions[0] if positions else "?",
                need_score=round(need_score, 3),  # Base need_score (no Statcast adjustments)
                statcast_boost=round(statcast_boost, 3) if statcast_boost is not None else None,  # NEW: Statcast adjustment factor
                adjusted_need_score=round(adjusted_need_score, 3) if adjusted_need_score is not None else None,  # NEW: Base + Boost
                z_score=round(player_z, 3),
                category_contributions=contributions,
                owned_pct=p.get("percent_owned", 0.0),
                starts_this_week=p.get("starts_this_week", 0),
                two_start=len(_starts) >= 2 or p.get("starts_this_week", 0) >= 2,
                two_start_this_week=len(_starts) >= 2 or p.get("starts_this_week", 0) >= 2,
                start1_opp=_start1_opp,
                start2_opp=_start2_opp,
                projected_saves=_raw_nsv,
                hot_cold=_hc,
                status=_status,
                injury_note=_injury_note,
                injury_status=_injury_status,
                injury_return_timeline=_injury_timeline,
                stats=_translated_stats,
                statcast_stats=_sc_dict,
                statcast_signals=_sc_sigs,
                quality_score=_pitcher_quality_map.get(name.lower()) if _fa_is_pitcher else None,
                rank_percentile=None,
                small_sample=_small_sample,
                momentum_signal=_momentum_signal_by_bdl_id.get(_bdl_id) if _bdl_id else None,
                park_factor=round(_get_park_factor(p.get("team") or "", "run"), 3),
                closer_role=_closer_role,
                availability_note=_avail_note,
                need_score_ci=_need_ci,
                need_score_volatile=_volatile,
                projection_source=_proj_src,
            )

        # Bulk quality_score lookup for pitcher FA candidates (enrichment only).
        # Queries probable_pitchers for today+next 7 days, keyed by pitcher_name.
        # Non-fatal: any exception leaves the dict empty (quality_score stays None).
        from datetime import date as _date_type
        _pitcher_quality_map = build_pitcher_quality_map(db, _date_type.today())

        def _apply_ownership_fallback(players: list[dict]) -> None:
            """Fill missing Yahoo ownership from the persisted eligibility snapshot."""
            player_keys = [p.get("player_key") for p in players if p.get("player_key")]
            if not player_keys:
                return
            try:
                rows = (
                    db.query(
                        PositionEligibility.yahoo_player_key,
                        PositionEligibility.league_rostered_pct,
                    )
                    .filter(PositionEligibility.yahoo_player_key.in_(player_keys))
                    .all()
                )
                ownership_by_key = {}
                for key, raw_pct in rows:
                    if raw_pct is None:
                        continue
                    pct = float(raw_pct)
                    if 0.0 < pct <= 1.0:
                        pct *= 100.0
                    if pct > 0.0:
                        ownership_by_key[key] = min(pct, 100.0)
                filled = 0
                for p in players:
                    if float(p.get("percent_owned") or 0.0) > 0.0:
                        continue
                    pct = ownership_by_key.get(p.get("player_key"))
                    if pct is not None:
                        p["percent_owned"] = pct
                        p["percent_owned_source"] = "position_eligibility"
                        filled += 1
                if filled:
                    logger.info("waiver: filled ownership fallback for %s players", filled)
            except Exception as exc:
                logger.warning("waiver ownership fallback failed (non-fatal): %s", exc)

            try:
                adp_data = client._load_adp_data()
                if not adp_data:
                    return
                filled = 0
                for p in players:
                    if float(p.get("percent_owned") or 0.0) > 0.0:
                        continue
                    name = (p.get("name") or "").strip()
                    pct = client._estimate_ownership_from_adp(name, adp_data)
                    if pct > 0.0:
                        p["percent_owned"] = pct
                        p["percent_owned_estimated"] = True
                        p["percent_owned_source"] = "adp_proxy"
                        filled += 1
                if filled:
                    logger.info("waiver: filled ADP ownership proxy for %s players", filled)
            except Exception as exc:
                logger.warning("waiver ADP ownership fallback failed (non-fatal): %s", exc)

        # Fetch MLB probable starts and populate starts_this_week for ALL SP pitchers
        # BEFORE creating top_available — April 21 Issue 2 fix (starts_this_week was 0 for all)
        from datetime import date as _dt, timedelta as _td
        _today = date_type.today()
        _week_end_ts = _today + _td(days=6)

        starts_map = _fetch_probable_starts_map(
            _today.strftime("%Y-%m-%d"), _week_end_ts.strftime("%Y-%m-%d")
        )

        # Populate starts_this_week for ALL SP pitchers in free_agents
        _populate_starts_this_week(free_agents, starts_map)
        _apply_ownership_fallback(free_agents)
        _injury_overlays_by_key = load_injury_overlays_for_yahoo_players(db, free_agents) if free_agents else {}

        # Park factor lookup (non-fatal import guard)
        try:
            from backend.fantasy_baseball.ballpark_factors import get_park_factor as _get_park_factor
        except ImportError:
            def _get_park_factor(team: str, factor: str = "run") -> float:  # type: ignore[misc]
                return 1.0

        # Bulk-load latest PlayerMomentum signals for hot/cold badge accuracy.
        # Uses 14-day delta-Z signal (SURGING/HOT/STABLE/COLD/COLLAPSING) instead of
        # season z-score average, which produced false "HOT" labels on slumping players.
        _momentum_signal_by_bdl_id: dict[int, str] = {}
        try:
            from backend.models import PlayerMomentum as _PM
            from sqlalchemy import func as _sqlfunc
            _latest_date = (
                db.query(_sqlfunc.max(_PM.as_of_date)).scalar()
            )
            if _latest_date:
                _mom_rows = (
                    db.query(_PM.bdl_player_id, _PM.signal)
                    .filter(_PM.as_of_date == _latest_date)
                    .all()
                )
                _momentum_signal_by_bdl_id = {r.bdl_player_id: r.signal for r in _mom_rows}
        except Exception:
            pass  # non-fatal: falls back to season z-score logic below

        # Build normalized_name→bdl_id lookup for free agents using PlayerIDMapping.
        # normalized_name is lowercase + ASCII-normalized (matches name.lower().strip() for most players).
        _fa_name_to_bdl_id: dict[str, int] = {}
        try:
            from backend.models import PlayerIDMapping as _PIM
            _fa_names = [p.get("name", "").strip().lower() for p in free_agents if p.get("name")]
            if _fa_names:
                _pim_rows = (
                    db.query(_PIM.normalized_name, _PIM.bdl_id)
                    .filter(_PIM.normalized_name.in_(_fa_names))
                    .all()
                )
                _fa_name_to_bdl_id = {r.normalized_name: r.bdl_id for r in _pim_rows if r.bdl_id}
        except Exception:
            pass

        # Load probable pitcher start opponents for the current scoring week
        # to populate start1_opp/start2_opp on two-start pitchers.
        _pitcher_starts_by_name: dict[str, list[str]] = {}
        try:
            from backend.models import ProbablePitcherSnapshot as _PPS
            from datetime import timedelta as _td
            _today = datetime.now(ZoneInfo("America/New_York")).date()
            _week_end = _today + _td(days=7)
            _pp_rows = (
                db.query(_PPS.pitcher_name, _PPS.opponent, _PPS.game_date)
                .filter(
                    _PPS.game_date >= _today,
                    _PPS.game_date <= _week_end,
                    _PPS.pitcher_name.isnot(None),
                )
                .order_by(_PPS.game_date)
                .all()
            )
            for row in _pp_rows:
                key = (row.pitcher_name or "").strip().lower()
                if key:
                    _pitcher_starts_by_name.setdefault(key, []).append(row.opponent or "?")
        except Exception:
            pass

        top_available = [_to_waiver_player(p) for p in free_agents]

        # Reconcile tag contradictions: IL status always overrides LOW_INJURY_RISK
        _IL_STATUS_VALUES = frozenset({
            "il", "il10", "il15", "il60", "dl", "dl10", "dl15", "dl60",
            "10-day-il", "15-day-il", "60-day-il", "injured list", "15dayil",
        })

        def _is_on_il(player) -> bool:
            for field in (player.injury_status, player.status):
                if field and field.lower().replace(" ", "-") in _IL_STATUS_VALUES:
                    return True
            return False

        for p in top_available:
            if _is_on_il(p) and "LOW_INJURY_RISK" in p.statcast_signals:
                # IL + LOW_INJURY_RISK is a contradiction — remove LOW_INJURY_RISK
                p.statcast_signals = [s for s in p.statcast_signals if s != "LOW_INJURY_RISK"]
                if "HIGH_INJURY_RISK" not in p.statcast_signals:
                    p.statcast_signals.append("HIGH_INJURY_RISK")

        # Separate IL players into a watch list — they should not be in active recommendations
        il_watch = [p for p in top_available if _is_on_il(p)]
        top_available = [p for p in top_available if not _is_on_il(p)]

        # Annotate waiver candidates with within-league drop intelligence (non-fatal)
        try:
            from backend.services.league_transaction_feed import (
                get_recent_league_drops,
                build_drop_lookup,
            )
            _drops = get_recent_league_drops(client)
            _drop_lk = build_drop_lookup(_drops)
            _by_key = _drop_lk["by_key"]
            _by_name = _drop_lk["by_name"]
            for _p in top_available + il_watch:
                _drop = _by_key.get(_p.player_id) or _by_name.get(_p.name.lower())
                if _drop:
                    _p.league_drop = {
                        "dropped_by": _drop.dropped_by_name,
                        "days_ago": _drop.days_ago,
                        "team_key": _drop.dropped_by_team,
                    }
                    _p.dropped_by_team = _drop.dropped_by_name or "Unknown Team"
        except Exception as _txn_err:
            logger.warning("waiver: league_transaction_feed non-fatal: %s", _txn_err)

        if min_z_score is not None:
            top_available = [p for p in top_available if p.need_score >= min_z_score]
        top_available = [p for p in top_available if p.owned_pct <= max_percent_owned]
        if sort == "percent_owned":
            top_available.sort(key=lambda x: x.owned_pct, reverse=True)
        else:
            top_available.sort(key=lambda x: x.need_score, reverse=True)

        total_ranked = len(top_available)
        if total_ranked:
            for idx, player in enumerate(top_available):
                player.rank_percentile = round(
                    ((total_ranked - idx) / total_ranked) * 100.0,
                    1,
                )

        # two_start_pitchers — filter from top_available (starts_this_week already populated)
        two_start_pitchers = sorted(
            [p for p in top_available if p.starts_this_week >= 2],
            key=lambda x: x.need_score, reverse=True
        )[:5]

        _closer_fas = [f for f in top_available if f.category_contributions.get("nsv", 0) > 0.5]
        _closer_alert = None
        if len(_closer_fas) == 0:
            _closer_alert = "NO_CLOSERS"
        elif len(_closer_fas) < 2:
            _closer_alert = "LOW_CLOSERS"

        from backend.services.waiver_edge_detector import il_capacity_info as _il_cap
        _il_info = _il_cap(my_roster) if my_roster else {"used": 0, "total": 2, "available": 0}

        # Build roster context: weakest player per canonical position (for upgrade comparison UI)
        # IL players skipped — they don't occupy droppable roster spots
        _CONTEXT_POSITIONS = frozenset({"SP", "RP", "OF", "1B", "2B", "3B", "SS", "C"})
        for _rp in (my_roster or []):
            try:
                _rp_selected = (_rp.get("selected_position") or "").upper()
                if _rp_selected in ("IL", "IL10", "IL60"):
                    continue
                _rp_proj = _get_proj(_rp)
                _rp_z = round(_rp_proj.get("z_score", 0.0) if _rp_proj else 0.0, 3)
                _rp_positions = _rp.get("positions") or []
                _rp_name = (_rp.get("name") or "").strip()
                _rp_key = _rp.get("player_key") or _rp_name
                _rp_team = _rp.get("team") or ""
                for _pos in _rp_positions:
                    _canon = "OF" if _pos in ("LF", "CF", "RF") else _pos
                    if _canon not in _CONTEXT_POSITIONS:
                        continue
                    # Keep weakest z_score per position (most droppable)
                    if _canon not in _roster_context or _rp_z < _roster_context[_canon]["z_score"]:
                        _rp_cat_scores = {}
                        if _rp_proj:
                            _rp_cat_scores = {k: round(float(v), 3) for k, v in (_rp_proj.get("cat_scores") or {}).items() if isinstance(v, (int, float))}
                        _roster_context[_canon] = {
                            "player_id": _rp_key,
                            "name": _rp_name,
                            "z_score": _rp_z,
                            "team": _rp_team,
                            "positions": _rp_positions,
                            "cat_scores": _rp_cat_scores,
                        }
            except Exception:
                continue

    except YahooAuthError as exc:
        logger.error("Waiver endpoint -- Yahoo auth error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Yahoo auth failed - refresh token may be expired. ({exc})",
        ) from exc
    except YahooAPIError as exc:
        logger.error("Waiver endpoint -- Yahoo API error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Yahoo API error: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Waiver endpoint failed unexpectedly: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Unexpected error fetching waiver data: {exc}",
        ) from exc

    from backend.schemas import PaginationOut
    return WaiverWireResponse(
        week_end=week_end,
        matchup_opponent=matchup_opponent,
        category_deficits=category_deficits,
        top_available=top_available,
        two_start_pitchers=two_start_pitchers,
        pagination=PaginationOut(
            page=page,
            per_page=per_page,
            has_next=len(free_agents) == per_page,
        ),
        closer_alert=_closer_alert,
        il_slots_used=_il_info["used"],
        il_slots_available=_il_info["available"],
        faab_balance=_faab_balance,
        roster_context=_roster_context,
        il_watch=il_watch,
        data_as_of=datetime.now(ZoneInfo("America/New_York")),
        scored_at=datetime.now(ZoneInfo("America/New_York")),
        scoring_model_version="2.1",
    )


@router.post("/api/fantasy/waiver/add")
async def add_fantasy_waiver_player(
    add_player_key: str = Query(..., description="Yahoo player key to add (mlb.p.XXXXX)"),
    drop_player_key: Optional[str] = Query(None, description="Yahoo player key to drop (mlb.p.XXXXX)"),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Execute a direct Yahoo add/drop transaction from waiver UI."""
    try:
        client = get_yahoo_client()
    except YahooAuthError as exc:
        raise HTTPException(
            status_code=503,
            detail="Yahoo not configured -- set YAHOO_REFRESH_TOKEN",
        ) from exc

    add_key = add_player_key.strip()
    drop_key = (drop_player_key or "").strip() or None

    if not add_key.startswith("mlb.p."):
        raise HTTPException(status_code=422, detail="add_player_key must be mlb.p.XXXXX")
    if drop_key and not drop_key.startswith("mlb.p."):
        raise HTTPException(status_code=422, detail="drop_player_key must be mlb.p.XXXXX")

    team_key = os.getenv("YAHOO_TEAM_KEY") or client.get_my_team_key()
    try:
        ok = client.add_drop_player(
            add_player_key=add_key,
            drop_player_key=drop_key,
            team_key=team_key,
        )
    except YahooAPIError as exc:
        raise HTTPException(status_code=502, detail=f"Yahoo add/drop failed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Waiver add failed: {exc}") from exc

    if ok:
        from backend.models import RosterAcquisition
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo
        _now_et = datetime.now(ZoneInfo("America/New_York"))
        _days_since_monday = _now_et.weekday()
        _week_start = _now_et.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=_days_since_monday)
        try:
            db.add(RosterAcquisition(
                team_key=team_key,
                player_added_key=add_key,
                player_dropped_key=drop_key,
                executed_at=_now_et,
                week_start=_week_start,
            ))
            db.commit()
        except Exception as _db_err:
            db.rollback()
            logger.warning("waiver/add: failed to persist RosterAcquisition: %s", _db_err)

    return {
        "success": bool(ok),
        "added": add_key,
        "dropped": drop_key,
        "team_key": team_key,
    }


# ---------------------------------------------------------------------------
# Roster Action Endpoint — validated Yahoo roster mutations
# Loop Iteration 10 (2026-06-24)
# ---------------------------------------------------------------------------

@router.post("/api/fantasy/roster/action", response_model=RosterActionResponse)
async def execute_roster_action(
    request: RosterActionRequest,
    user: str = Depends(verify_api_key),
):
    """
    Execute a roster action after a side-effect-free validation phase.

    Two-phase semantics:
    1. Phase 1: Validate roster space, player availability, position eligibility
    2. Phase 2: Submit one Yahoo mutation

    ADD_DROP uses Yahoo's atomic add/drop transaction. It is never decomposed
    into separate mutations, so application-level rollback is not required.
    """
    from backend.services.yahoo_actions import (
        get_yahoo_actions_service,
        ActionResult,
    )

    try:
        service = get_yahoo_actions_service()

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Yahoo Actions Service unavailable: {str(exc)}"
        ) from exc

    # Execute the action
    result: ActionResult = await service.execute_action(
        action=request.action,
        add_player_id=request.add_player_id,
        drop_player_id=request.drop_player_id,
        position=request.position,
    )

    # Convert service result to response model
    errors = [
        RosterActionError(code=e.code, message=e.message)
        for e in result.errors
    ] if result.errors else None

    warnings = [
        RosterActionWarning(code=w.code, message=w.message)
        for w in result.warnings
    ] if result.warnings else None

    return RosterActionResponse(
        success=result.success,
        transaction_id=result.transaction_id,
        roster_state=result.roster_state,
        errors=errors,
        warnings=warnings,
        rollback_attempted=result.rollback_attempted,
        rollback_succeeded=result.rollback_succeeded,
        manual_action_required=result.manual_action_required,
        execution_time_et=result.execution_time_et,
    )


# ---------------------------------------------------------------------------


@router.get("/api/fantasy/waiver/recommendations")
async def get_waiver_recommendations(
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Actionable ADD/DROP/ADD_DROP recommendations."""
    from datetime import date as date_type, timedelta
    from backend.schemas import (
        RosterMoveRecommendation, WaiverRecommendationsResponse,
        WaiverPlayerOut, CategoryDeficitOut, DropPlayerOut,
    )
    from backend.fantasy_baseball.player_board import get_or_create_projection as _get_proj
    from backend.fantasy_baseball.statcast_loader import (
        build_statcast_signals, statcast_need_score_boost,
    )

    today = date_type.today()
    week_end = today + timedelta(days=(6 - today.weekday()))

    matchup_opponent = "TBD"
    opponent_team_key = ""
    category_deficits: list = []
    recommendations: list = []

    try:
        client = get_yahoo_client()
        my_team_key = os.getenv("YAHOO_TEAM_KEY", "")
        if not my_team_key:
            try:
                my_team_key = client.get_my_team_key()
            except Exception:
                my_team_key = ""

        my_roster: list = []
        if my_team_key:
            try:
                my_roster = client.get_roster()
            except Exception:
                pass

        try:
            _sb = client.get_scoreboard()
            _my_matchup_teams: list = []
            for _matchup_teams in _iter_scoreboard_matchup_teams(_sb):
                _my_tuple = None
                for _t in _matchup_teams:
                    _tk = _t[0]
                    if _tk and (
                        _tk == my_team_key
                        or (my_team_key and (_tk in my_team_key or my_team_key in _tk))
                    ):
                        _my_tuple = _t
                        break
                if _my_tuple is not None:
                    _opp_tuple = next(
                        (_t for _t in _matchup_teams if _t[0] != _my_tuple[0]), None
                    )
                    if _opp_tuple is not None:
                        matchup_opponent = _opp_tuple[1] or "TBD"
                        opponent_team_key = _opp_tuple[0] or ""
                        _my_matchup_teams = [_my_tuple, _opp_tuple]
                    break

            _sid_map: dict = dict(_YAHOO_STAT_FALLBACK)
            if _my_matchup_teams:

                # Display-only Yahoo stats that are NOT scoring categories.
                _NON_SCORING_DISPLAY_2 = frozenset({"H_AB", "IP", "GS"})

                def _rec_stats_dict(raw_stats_list: list) -> dict:
                    out: dict = {}
                    for _st in raw_stats_list:
                        if not isinstance(_st, dict):
                            continue
                        _so = _st.get("stat", {})
                        if not isinstance(_so, dict):
                            continue
                        _sid_k = str(_so.get("stat_id", ""))
                        if not _sid_k:
                            continue
                        _key2 = _sid_map.get(_sid_k, _sid_k)
                        if isinstance(_key2, str) and _key2.isdigit():
                            continue
                        # Drop display-only stats that are not H2H scoring cats.
                        if _key2 in _NON_SCORING_DISPLAY_2:
                            continue
                        try:
                            out[_key2] = float(_so.get("value", 0) or 0)
                        except (TypeError, ValueError):
                            out[_key2] = 0.0
                    return out

                _my_stats = _rec_stats_dict(_my_matchup_teams[0][2])
                _opp_stats = _rec_stats_dict(_my_matchup_teams[1][2])
                for _cat, _my_val in _my_stats.items():
                    _opp_val = _opp_stats.get(_cat, 0.0)
                    _result = compare_category(_cat, _my_val, _opp_val)
                    category_deficits.append(
                        CategoryDeficitOut(
                            category=_cat,
                            my_total=_my_val,
                            opponent_total=_opp_val,
                            deficit=_result.gap,
                            winning=(_result.verdict == "W"),
                        )
                    )
        except Exception as _rec_sb_err:
            logger.warning("recommendations scoreboard failed (non-fatal): %s", _rec_sb_err)

        free_agents = client.get_free_agents(count=40)
        _recommendation_injury_overlays = (
            load_injury_overlays_for_yahoo_players(db, free_agents)
            if free_agents
            else {}
        )

        # Populate starts_this_week for SP free agents before scoring.
        # Uses the module-level _STARTS_CACHE so the MLB Stats API is called
        # at most once per 6 hours across both waiver endpoints.
        try:
            from datetime import date as _fa_dt, timedelta as _fa_td
            _fa_today = _fa_dt.today()
            _fa_week_end = _fa_today + _fa_td(days=6)
            _fa_starts_map = _fetch_probable_starts_map(
                _fa_today.strftime("%Y-%m-%d"), _fa_week_end.strftime("%Y-%m-%d")
            )
            _populate_starts_this_week(free_agents, _fa_starts_map)
        except Exception as _fa_se:
            logger.warning("starts_this_week population failed in recommendations (non-fatal): %s", _fa_se)

        # Park factor lookup for recommendations endpoint (non-fatal import guard)
        try:
            from backend.fantasy_baseball.ballpark_factors import get_park_factor as _get_park_factor_rec
        except ImportError:
            def _get_park_factor_rec(team: str, factor: str = "run") -> float:  # type: ignore[misc]
                return 1.0

        # Bulk-load latest PlayerMomentum signals for hot/cold badge accuracy.
        # Uses 14-day delta-Z signal (SURGING/HOT/STABLE/COLD/COLLAPSING) instead of
        # season z-score average, which produced false "HOT" labels on slumping players.
        _rec_momentum_signal_by_bdl_id: dict[int, str] = {}
        try:
            from backend.models import PlayerMomentum as _PM_rec
            from sqlalchemy import func as _sqlfunc_rec
            _rec_latest_date = (
                db.query(_sqlfunc_rec.max(_PM_rec.as_of_date)).scalar()
            )
            if _rec_latest_date:
                _rec_mom_rows = (
                    db.query(_PM_rec.bdl_player_id, _PM_rec.signal)
                    .filter(_PM_rec.as_of_date == _rec_latest_date)
                    .all()
                )
                _rec_momentum_signal_by_bdl_id = {r.bdl_player_id: r.signal for r in _rec_mom_rows}
        except Exception:
            pass  # non-fatal: falls back to season z-score logic below

        # Build normalized_name→bdl_id lookup for free agents using PlayerIDMapping.
        _rec_fa_name_to_bdl_id: dict[str, int] = {}
        try:
            from backend.models import PlayerIDMapping as _PIM_rec
            _rec_fa_names = [p.get("name", "").strip().lower() for p in free_agents if p.get("name")]
            if _rec_fa_names:
                _rec_pim_rows = (
                    db.query(_PIM_rec.normalized_name, _PIM_rec.bdl_id)
                    .filter(_PIM_rec.normalized_name.in_(_rec_fa_names))
                    .all()
                )
                _rec_fa_name_to_bdl_id = {r.normalized_name: r.bdl_id for r in _rec_pim_rows if r.bdl_id}
        except Exception:
            pass

        # Bulk quality_score lookup for pitcher FA candidates (enrichment only).
        # Queries probable_pitchers for today+next 7 days, keyed by pitcher_name.
        # Non-fatal: any exception leaves the dict empty (quality_score stays None).
        _pitcher_quality: dict[str, float] = {}
        try:
            from backend.models import ProbablePitcherSnapshot
            from datetime import date as _date_type, timedelta as _td
            _today = _date_type.today()
            _qs_rows = (
                db.query(
                    ProbablePitcherSnapshot.pitcher_name,
                    ProbablePitcherSnapshot.quality_score,
                )
                .filter(
                    ProbablePitcherSnapshot.game_date >= _today,
                    ProbablePitcherSnapshot.game_date <= _today + _td(days=7),
                    ProbablePitcherSnapshot.quality_score.isnot(None),
                )
                .all()
            )
            for _r in _qs_rows:
                if _r.pitcher_name and _r.quality_score is not None:
                    # Keep highest quality_score per name (multiple starts possible)
                    key = _r.pitcher_name.strip().lower()
                    if key not in _pitcher_quality or _r.quality_score > _pitcher_quality[key]:
                        _pitcher_quality[key] = float(_r.quality_score)
        except Exception:
            pass

        # Daily availability blacklist for recommendations endpoint (mirrors main waiver path).
        _rec_blacklist_keys: set = set()
        try:
            from backend.models import DailyAvailabilityOverride as _DAO_rec
            from zoneinfo import ZoneInfo as _ZI_rec
            _rec_bl_today = datetime.now(_ZI_rec("America/New_York")).date()
            _rec_blacklist_keys = {
                r.player_key
                for r in db.query(_DAO_rec.player_key)
                    .filter(_DAO_rec.game_date == _rec_bl_today, _DAO_rec.status.in_(["OUT", "DAY_OFF"]))
                    .all()
            }
        except Exception:
            pass  # non-fatal

        def _score_fa(p: dict) -> WaiverPlayerOut:
            """Score free agent using unified need-score service with transparent Statcast boost."""
            positions = p.get("positions") or []
            name = (p.get("name") or "").strip()
            bp = _get_proj(p)
            z_score = bp.get("z_score", 0.0)
            if isinstance(z_score, (tuple, list)):
                z_score = float(z_score[0]) if z_score else 0.0
            else:
                z_score = float(z_score) if z_score is not None else 0.0
            cat_scores = bp.get("cat_scores") or {}

            # ── Injury status & availability ──────────────────────────────────────
            _status = p.get("status") or None
            _injury_note = p.get("injury_note") or None
            _injury_status = p.get("injury_status")
            _injury_timeline = None
            _overlay = _recommendation_injury_overlays.get(p.get("player_key") or "")
            if _overlay is not None:
                _status = getattr(_overlay, "status", None) or _status
                _injury_note = _injury_note or getattr(_overlay, "note", None)
                _injury_status = getattr(_overlay, "status", None) or _injury_status
                _injury_timeline = getattr(_overlay, "return_timeline", None)

            _pkey_rec = p.get("player_key") or ""
            if _pkey_rec and _pkey_rec in _rec_blacklist_keys:
                _avail_note_rec: Optional[str] = "NOT AVAILABLE TODAY — day off confirmed"
            elif _injury_status and any(kw in (_injury_status or "").upper() for kw in ("IL", "DL", "60-DAY", "15-DAY", "10-DAY")):
                _avail_note_rec = "On IL — check IL slot availability"
            elif _injury_status and "DTD" in (_injury_status or "").upper():
                _avail_note_rec = "DTD — confirm before adding"
            else:
                _avail_note_rec = None

            # Translate raw Yahoo stat_ids → display names using _sid_map.
            _raw_stats: dict = p.get("stats") or {}
            _translated_stats: dict = {}
            for _sk, _sv in _raw_stats.items():
                _tk = _sid_map.get(_sk, _sk)
                if isinstance(_tk, str) and _tk.isdigit():
                    continue
                _translated_stats[_tk] = _sv

            # ── Statcast enrichment (BEFORE need_score calculation) ────────────────
            _fa_is_pitcher = positions[0] in ("SP", "RP", "P") if positions else False
            _sc_sigs: list = []
            _sc_dict: Optional[dict] = None
            try:
                from backend.fantasy_baseball.statcast_loader import (
                    get_statcast_batter as _get_sc_bat,
                    get_statcast_pitcher as _get_sc_pit,
                )
                if not _fa_is_pitcher:
                    _sb_sc = _get_sc_bat(name)
                    if _sb_sc:
                        _sc_dict = {
                            "xwoba": round(_sb_sc.xwoba, 3) if _sb_sc.xwoba else None,
                            "xwoba_diff": round(_sb_sc.xwoba_diff, 3) if _sb_sc.xwoba_diff is not None else None,
                            "barrel_pct": round(_sb_sc.barrel_pct, 1) if _sb_sc.barrel_pct is not None else None,
                            "exit_velo_avg": round(_sb_sc.exit_velo_avg, 1) if _sb_sc.exit_velo_avg is not None else None,
                            "hard_hit_pct": round(_sb_sc.hard_hit_pct, 1) if _sb_sc.hard_hit_pct is not None else None,
                            "wrc_plus": round(_sb_sc.wrc_plus, 0) if _sb_sc.wrc_plus is not None else None,
                        }
                else:
                    _sp_sc = _get_sc_pit(name)
                    if _sp_sc:
                        _sc_dict = {
                            "xera": round(_sp_sc.xera, 2) if _sp_sc.xera is not None else None,
                            "xera_diff": round(_sp_sc.xera_diff, 2) if _sp_sc.xera_diff is not None else None,
                            "stuff_plus": round(_sp_sc.stuff_plus, 0) if _sp_sc.stuff_plus is not None else None,
                            "whiff_pct": round(_sp_sc.whiff_pct, 1) if _sp_sc.whiff_pct is not None else None,
                            "barrel_allowed_pct": round(_sp_sc.barrel_allowed_pct, 1) if _sp_sc.barrel_allowed_pct is not None else None,
                            "xwoba_allowed": round(_sp_sc.xwoba_allowed, 3) if _sp_sc.xwoba_allowed is not None else None,
                        }
                if name:
                    _sc_sigs, _ = build_statcast_signals(name, _fa_is_pitcher, p.get("percent_owned", 100.0))
            except Exception:
                pass

            # Add HIGH_INJURY_RISK signal if injury penalty applies
            if _overlay:
                _penalty_base, _penalty_note = apply_injury_penalty(0.0, _overlay)
                if _penalty_note and "HIGH_INJURY_RISK" not in _sc_sigs:
                    _sc_sigs.append("HIGH_INJURY_RISK")

            # ── Need-score calculation (unified service with Statcast boost) ─────────
            from backend.services.need_score import (
                calculate_need_score as _calc_need_score_rec,
                NeedScoreResult,
            )

            need_score_result: NeedScoreResult = _calc_need_score_rec(
                player_cat_scores=cat_scores,
                player_z_score=z_score,
                category_deficits=category_deficits,
                n_cats=max(1, len(category_deficits)) if category_deficits else 1,
                statcast_signals=_sc_sigs,
                team_context=None,
                marginal_stats=None,
            )

            # Base need_score (pure category-aware, no Statcast)
            need_score = need_score_result.base_need_score
            # Statcast boost (transparent separate calculation)
            statcast_boost = need_score_result.statcast_boost
            # Adjusted need_score (base + Statcast boost)
            adjusted_need_score = need_score_result.adjusted_need_score

            # Apply injury penalty to final need_score
            need_score, _penalty_note = apply_injury_penalty(need_score, _overlay)

            # Blacklisted players get zero need_score
            if _pkey_rec and _pkey_rec in _rec_blacklist_keys:
                need_score = 0.0

            # ── Stability metadata ──────────────────────────────────────────
            _pkey_stab_rec = p.get("player_key") or ""
            _volatile_rec = False
            _now_stab_rec = datetime.now(ZoneInfo("America/New_York"))
            if _pkey_stab_rec and _pkey_stab_rec in _NEED_SCORE_HISTORY:
                _prev_score_rec, _prev_ts_rec = _NEED_SCORE_HISTORY[_pkey_stab_rec]
                _age_min_rec = (_now_stab_rec - _prev_ts_rec).total_seconds() / 60.0
                if _age_min_rec < 60.0 and _prev_score_rec > 0.1:
                    _delta_pct_rec = abs(need_score - _prev_score_rec) / _prev_score_rec
                    if _delta_pct_rec > 0.20:
                        _volatile_rec = True
                        logger.warning(
                            "[waiver_volatility_rec] %s: %.2f→%.2f (%.0f%% swing in %.0fm)",
                            name, _prev_score_rec, need_score, _delta_pct_rec * 100.0, _age_min_rec,
                        )
            if _pkey_stab_rec:
                _NEED_SCORE_HISTORY[_pkey_stab_rec] = (need_score, _now_stab_rec)

            _proj_src_rec = bp.get("fusion_source") or (
                "proxy" if bp.get("is_proxy") else "draft_board"
            )
            _ci_factors_rec = {"steamer+statcast": 0.15, "steamer": 0.25}
            _need_ci_rec: Optional[float] = round(need_score * _ci_factors_rec.get(_proj_src_rec, 0.40), 2) if need_score > 0 else None
            logger.debug(
                "[waiver_score_rec] %s need=%.3f ci=±%.2f src=%s volatile=%s",
                name, need_score, _need_ci_rec or 0.0, _proj_src_rec, _volatile_rec,
            )

            # hot_cold derived from PlayerMomentum 14d signal (falls back to season z-score average)
            _hc: Optional[str] = None
            _rec_bdl_id: Optional[int] = None
            try:
                _rec_player_name_key = name.lower().strip()
                _rec_bdl_id = _rec_fa_name_to_bdl_id.get(_rec_player_name_key)
                if _rec_bdl_id and _rec_bdl_id in _rec_momentum_signal_by_bdl_id:
                    _mom_sig = _rec_momentum_signal_by_bdl_id[_rec_bdl_id]
                    _hc = "HOT" if _mom_sig in ("SURGING", "HOT") else (
                        "COLD" if _mom_sig in ("COLD", "COLLAPSING") else None
                    )
                elif cat_scores:
                    # Fallback: season z-score average (less accurate)
                    _contribs = {k: float(v) for k, v in cat_scores.items() if isinstance(v, (int, float))}
                    if _contribs:
                        _avg = sum(_contribs.values()) / len(_contribs)
                        _hc = "HOT" if _avg > 0.75 else ("COLD" if _avg < -0.5 else None)
            except Exception:
                pass

            # Populate quality_score for pitcher FA candidates only
            is_pitcher_fa = positions and positions[0] in ("SP", "RP", "P")
            qs = _pitcher_quality.get(name.lower()) if is_pitcher_fa else None
            return WaiverPlayerOut(
                player_id=p.get("player_key") or "",
                name=name,
                team=p.get("team") or "",
                position=positions[0] if positions else "?",
                need_score=round(need_score, 3),  # Base need_score (no Statcast adjustments)
                statcast_boost=round(statcast_boost, 3) if statcast_boost is not None else None,  # NEW: Statcast adjustment factor
                adjusted_need_score=round(adjusted_need_score, 3) if adjusted_need_score is not None else None,  # NEW: Base + Boost
                category_contributions=cat_scores,
                owned_pct=p.get("percent_owned", 0.0),
                starts_this_week=p.get("starts_this_week", 0),
                quality_score=qs,
                stats=_translated_stats,
                statcast_signals=_sc_sigs,
                statcast_stats=_sc_dict,
                hot_cold=_hc,
                status=_status,
                injury_note=_injury_note,
                injury_status=_injury_status,
                injury_return_timeline=_injury_timeline,
                momentum_signal=_rec_momentum_signal_by_bdl_id.get(_rec_bdl_id) if _rec_bdl_id else None,
                park_factor=round(_get_park_factor_rec(p.get("team") or "", "run"), 3),
                availability_note=_avail_note_rec,
                need_score_ci=_need_ci_rec,
                need_score_volatile=_volatile_rec,
                projection_source=_proj_src_rec,
            )

        scored_fas = sorted(
            [_score_fa(p) for p in free_agents],
            key=lambda x: x.need_score,
            reverse=True,
        )

        try:
            from backend.fantasy_baseball.pybaseball_loader import (
                log_statcast_coverage,
                load_pybaseball_batters,
                load_pybaseball_pitchers,
            )
            fa_names = [p.get("name", "") for p in free_agents]
            _sc = {**load_pybaseball_batters(2025), **load_pybaseball_pitchers(2025)}
            if _sc:
                log_statcast_coverage(fa_names, _sc, "waiver FAs")
        except Exception:
            pass

        from backend.services.waiver_edge_detector import (
            drop_candidate_value as _drop_candidate_value,
            is_protected_drop_candidate as _is_protected_drop_candidate,
        )

        my_roster_scored: list = []
        _IL_STATUSES = {"IL", "IL10", "IL15", "IL60", "NA", "OUT", "DL"}
        _STARTING_SLOTS = {"C", "1B", "2B", "3B", "SS", "OF", "Util", "SP", "RP", "P"}
        for rp in my_roster:
            bp = _get_proj(rp)
            # Derive effective IL status from both status field and selected_position slot.
            # Yahoo sometimes returns status=None for IL players but sets selected_position=IL.
            raw_status = rp.get("status")
            sel_pos = rp.get("selected_position") or ""
            effective_status = raw_status if raw_status else (sel_pos if sel_pos in _IL_STATUSES else raw_status)
            my_roster_scored.append({
                "name": (rp.get("name") or "").strip(),
                "player_key": rp.get("player_key") or "",
                "positions": rp.get("positions") or [],
                "id": bp.get("id") or rp.get("player_key") or "",
                "z_score": bp.get("z_score", 0.0),
                "is_proxy": bp.get("is_proxy", False),
                "cat_scores": bp.get("cat_scores") or {},
                "tier": bp.get("tier"),
                "adp": bp.get("adp"),
                "starts_this_week": int(rp.get("starts_this_week", 1)),
                "status": effective_status,
                "injury_note": rp.get("injury_note"),
                "is_undroppable": bool(rp.get("is_undroppable", 0)),
                "is_keeper": bool(bp.get("is_keeper", False)),
                "percent_owned": rp.get("percent_owned", rp.get("owned_pct", 0.0)),
            })

        opponent_roster_scored: list = []
        if opponent_team_key:
            try:
                _opp_players = client.get_roster(opponent_team_key)
                for _rp in _opp_players:
                    _bp = _get_proj(_rp)
                    opponent_roster_scored.append({
                        "name": (_rp.get("name") or "").strip(),
                        "positions": _rp.get("positions") or [],
                        "cat_scores": _bp.get("cat_scores") or {},
                        "starts_this_week": int(_rp.get("starts_this_week", 1)),
                    })
            except Exception as exc:
                logger.warning("opponent_roster fetch failed (non-fatal): %s", exc)

        def _weakest_safe_to_drop(target_positions: list) -> Optional[dict]:
            candidates = [
                rp for rp in my_roster_scored
                if not rp.get("is_undroppable", False)
                and any(pos in rp["positions"] for pos in target_positions)
                and not _is_protected_drop_candidate(rp)
            ]
            if not candidates:
                return None
            active = [p for p in candidates if p.get("status") not in _IL_STATUSES]
            pool = active if len(active) >= 2 else (candidates if len(active) == 0 else None)
            if pool is None:
                return None

            # Refuse to pick a drop when ALL candidates have default/missing scoring
            # data (empty cat_scores + z_score=0 + default ADP). This prevents the
            # "always Seiya Suzuki" universal-drop bug where the tiebreaker became
            # the only discriminator across identically-scored players.
            from backend.services.waiver_edge_detector import _coerce_float as _wv_cf, _coerce_int as _wv_ci
            all_data_missing = all(
                not p.get("cat_scores")
                and _wv_cf(p.get("z_score"), 0.0) == 0.0
                and _wv_cf(p.get("adp"), 9999.0) >= 9000.0
                and _wv_ci(p.get("tier"), 999) >= 999
                for p in pool
            )
            if all_data_missing:
                return None

            return min(pool, key=_drop_candidate_value)

        def _fmt_signals(signals: list, reg_delta: float, is_pitcher: bool) -> str:
            parts = []
            if "BUY_LOW" in signals:
                metric = "xERA" if is_pitcher else "xwOBA"
                parts.append(f"BUY_LOW ({metric} delta={reg_delta:+.3f})")
            if "BREAKOUT" in signals:
                parts.append("BREAKOUT candidate")
            if not parts:
                return ""
            return " [" + "; ".join(parts) + "]"

        def _fmt_drop_signals(signals: list, reg_delta: float, is_pitcher: bool) -> str:
            parts = []
            if "SELL_HIGH" in signals:
                metric = "xERA" if is_pitcher else "xwOBA"
                parts.append(f"drop is SELL_HIGH ({metric} delta={reg_delta:+.3f})")
            if "HIGH_INJURY_RISK" in signals:
                parts.append("drop has HIGH_INJURY_RISK")
            if not parts:
                return ""
            return " [" + "; ".join(parts) + "]"

        def _build_drop_out(candidate: dict) -> DropPlayerOut:
            raw_cats = candidate.get("cat_scores") or {}
            try:
                safe_cats = {k: float(v) for k, v in raw_cats.items()}
            except (ValueError, TypeError):
                safe_cats = {}
            return DropPlayerOut(
                player_id=candidate.get("player_key", ""),
                name=candidate["name"],
                position=candidate["positions"][0] if candidate.get("positions") else "?",
                positions=candidate.get("positions") or [],
                z_score=round(candidate.get("z_score", 0.0), 3),
                cat_scores=safe_cats,
                tier=candidate.get("tier", 5),
                adp=candidate.get("adp", 999.0),
                percent_owned=round(candidate.get("percent_owned", 0.0), 1),
                status=candidate.get("status"),
                injury_note=candidate.get("injury_note"),
                starts_this_week=candidate.get("starts_this_week", 0),
            )

        def _compute_positional_impact(candidate: dict, add_position: str, roster: list) -> list:
            active_after = [
                p for p in roster
                if p["name"] != candidate["name"]
                and p.get("status") not in _IL_STATUSES
            ]
            cand_positions = set(candidate.get("positions") or [])
            add_positions = {add_position} if add_position and add_position != "?" else set()
            warnings = []
            for pos in cand_positions:
                if pos not in _STARTING_SLOTS:
                    continue
                if pos in add_positions:
                    continue
                coverage = sum(1 for p in active_after if pos in p.get("positions", []))
                if coverage == 0:
                    warnings.append(f"Drops last {pos}-eligible player")
            return warnings

        # Constraint pre-computations for the recommendation loop.
        from backend.services.waiver_edge_detector import il_capacity_info as _il_cap
        _il_slots_available = _il_cap(my_roster)["available"] if my_roster else 0

        # FAAB balance for budget constraint check (non-fatal).
        _rec_faab_balance: Optional[float] = None
        try:
            _rec_faab_balance = client.get_faab_balance()
        except Exception:
            pass

        for fa in scored_fas[:15]:
            if len(recommendations) >= 5:
                break

            # Skip zero-evidence proxy players — no projections or cat_scores means
            # the gain calculation is unreliable (all such players share z_score=0.0,
            # producing clone recommendations driven solely by the drop candidate score
            # rather than any genuine player quality signal).
            if not fa.category_contributions and fa.need_score == 0.0:
                logger.debug(
                    "[waiver_recs] skipping zero-evidence FA: %s (%s) — no cat_scores or z_score",
                    fa.name, fa.position,
                )
                continue

            fa_positions = [fa.position] if fa.position != "?" else []
            if fa.position in ("SP", "RP", "P"):
                pos_group = ["SP", "RP", "P"]
                pos_label = "pitching"
            elif fa.position in ("C",):
                pos_group = ["C"]
                pos_label = "catcher"
            elif fa.position in ("SS",):
                pos_group = ["SS"]
                pos_label = "shortstop"
            elif fa.position in ("2B",):
                pos_group = ["2B"]
                pos_label = "second base"
            elif fa.position in ("3B",):
                pos_group = ["3B"]
                pos_label = "third base"
            elif fa.position in ("1B",):
                pos_group = ["1B"]
                pos_label = "first base"
            elif fa.position in ("OF", "LF", "CF", "RF"):
                pos_group = ["OF", "LF", "CF", "RF"]
                pos_label = "outfield"
            else:
                pos_group = fa_positions
                pos_label = fa.position

            fa_is_pitcher = fa.position in ("SP", "RP", "P")
            fa_signals, fa_reg_delta = build_statcast_signals(
                fa.name, fa_is_pitcher, fa.owned_pct
            )
            statcast_boost = statcast_need_score_boost(fa_signals)
            adjusted_need = fa.need_score + statcast_boost

            drop_candidate = _weakest_safe_to_drop(pos_group)
            if not drop_candidate:
                if adjusted_need >= 2.0:
                    signal_text = _fmt_signals(fa_signals, fa_reg_delta, fa_is_pitcher)
                    recommendations.append(RosterMoveRecommendation(
                        action="ADD",
                        add_player=fa,
                        drop_player_name=None,
                        drop_player_position=None,
                        rationale=(
                            f"Add {fa.name} ({fa.position}, {fa.team}) - "
                            f"projected z={fa.need_score:+.1f}{signal_text}. "
                            f"No {pos_label} to drop suggested; check bench."
                        ),
                        category_targets=[
                            k for k, v in (fa.category_contributions or {}).items()
                            if isinstance(v, (int, float)) and v > 0
                        ],
                        need_score=round(adjusted_need, 3),
                        confidence=0.5 if not _get_proj({"player_key": fa.player_id, "name": fa.name}).get("is_proxy") else 0.3,
                        statcast_signals=fa_signals,
                        regression_delta=fa_reg_delta,
                    ))
                continue

            drop_is_pitcher = drop_candidate["positions"][0] in ("SP", "RP", "P") if drop_candidate["positions"] else False
            drop_signals, drop_reg_delta = build_statcast_signals(
                drop_candidate["name"], drop_is_pitcher
            )
            # Defensive type coercion: ensure both sides of comparison are floats
            # drop_candidate["z_score"] might leak as tuple from board projection data
            drop_z_score = drop_candidate.get("z_score", 0.0)
            if isinstance(drop_z_score, (tuple, list)):
                drop_z_score = float(drop_z_score[0]) if drop_z_score else 0.0
            else:
                drop_z_score = float(drop_z_score)
            
            drop_score_adj = max(
                _drop_candidate_value(drop_candidate)[0],
                drop_z_score + statcast_need_score_boost(drop_signals),
            )

            if drop_score_adj >= adjusted_need:
                continue

            gain = adjusted_need - drop_score_adj
            if gain < 0.5:
                continue

            fa_proj = _get_proj({"player_key": fa.player_id, "name": fa.name, "positions": [fa.position]})
            is_proxy = fa_proj.get("is_proxy", False)
            confidence = 0.75 if not is_proxy else 0.45

            signal_text = _fmt_signals(fa_signals, fa_reg_delta, fa_is_pitcher)
            drop_signal_text = _fmt_drop_signals(drop_signals, drop_reg_delta, drop_is_pitcher)

            rationale = (
                f"Add {fa.name} ({fa.position}, {fa.team}, {fa.owned_pct:.0f}% owned), "
                f"drop {drop_candidate['name']} ({drop_candidate['positions'][0] if drop_candidate['positions'] else '?'}). "
                f"Net gain: {gain:+.1f} ({drop_score_adj:+.1f} -> {adjusted_need:+.1f}){signal_text}{drop_signal_text}."
            )
            if is_proxy:
                rationale += " [Call-up - projections estimated.]"
            if drop_candidate.get("status") in _IL_STATUSES:
                from backend.services.waiver_edge_detector import il_capacity_info as _il_cap2
                if my_roster and _il_cap2(my_roster)["available"] > 0:
                    rationale = (
                        f"[IL slot free - move {drop_candidate['name']} to IL first] " + rationale
                    )
                else:
                    rationale += f" [Note: {drop_candidate['name']} is {drop_candidate['status']} - consider IL slot if available]"

            _mcmc = {}
            try:
                from backend.fantasy_baseball.mcmc_simulator import simulate_roster_move as _sim_move
                _add_for_mcmc = {
                    "name": fa.name,
                    "positions": [fa.position],
                    "cat_scores": dict(fa.category_contributions),
                    "starts_this_week": fa.starts_this_week,
                }
                logger.debug(
                    "[MCMC_DEBUG] fa.name=%s, cat_scores=%s, starts=%s",
                    fa.name, fa.category_contributions, fa.starts_this_week
                )
                logger.debug(
                    "[MCMC_DEBUG] my_roster players with cat_scores: %d/%d",
                    sum(1 for p in my_roster_scored if p.get("cat_scores")),
                    len(my_roster_scored)
                )
                _mcmc = _sim_move(
                    my_roster=my_roster_scored,
                    opponent_roster=opponent_roster_scored,
                    add_player=_add_for_mcmc,
                    drop_player_name=drop_candidate["name"],
                    n_sims=1000,
                )
                logger.info(
                    "[MCMC] %s: enabled=%s win_prob %.3f->%.3f gain=%.3f opp_roster=%d",
                    fa.name,
                    _mcmc.get("mcmc_enabled"),
                    _mcmc.get("win_prob_before", 0.0),
                    _mcmc.get("win_prob_after", 0.0),
                    _mcmc.get("win_prob_gain", 0.0),
                    len(opponent_roster_scored),
                )
                if _mcmc.get("mcmc_enabled") and abs(_mcmc["win_prob_gain"]) >= 0.005:
                    wp_before_pct = round(_mcmc["win_prob_before"] * 100)
                    wp_after_pct = round(_mcmc["win_prob_after"] * 100)
                    wp_gain_pct = round(_mcmc["win_prob_gain"] * 100)
                    rationale += (
                        f" Win prob: {wp_before_pct}% -> {wp_after_pct}%"
                        f" ({wp_gain_pct:+d}%)."
                    )
            except Exception as exc:
                logger.warning("[MCMC] sim failed for %s: %s", fa.name, exc)

            # When MCMC produced a verdict, never surface a move that the
            # simulator says hurts the matchup. z-score-driven gains can
            # disagree with simulated win probability (e.g. category
            # tradeoffs); trust the simulator when it ran.
            # When MCMC ran and produced a non-positive win-probability gain,
            # never surface the move: a zero-gain swap wastes a roster spot and
            # a negative-gain swap actively hurts the matchup.
            if _mcmc.get("mcmc_enabled") and _mcmc.get("win_prob_gain", 0.0) <= 0:
                continue

            # Build rich drop context fields
            cat_win_probs_after = _mcmc.get("category_win_probs_after") or {}
            mcmc_has_cats = bool(cat_win_probs_after)
            add_cats = fa.category_contributions or {}
            drop_cats = drop_candidate.get("cat_scores") or {}
            all_cats = set(add_cats) | set(drop_cats)
            category_deltas: dict = {}
            for _cat in all_cats:
                _add_val = add_cats.get(_cat, 0.0)
                _drop_val = drop_cats.get(_cat, 0.0)
                if not isinstance(_add_val, (int, float)) or not isinstance(_drop_val, (int, float)):
                    continue
                category_deltas[_cat] = {
                    "add": round(float(_add_val), 3),
                    "drop": round(float(_drop_val), 3),
                    "net": round(float(_add_val) - float(_drop_val), 3),
                    "cat_win_prob": round(float(cat_win_probs_after[_cat]), 4) if mcmc_has_cats and _cat in cat_win_probs_after else None,
                }
            if mcmc_has_cats:
                _missing_cats = all_cats - set(cat_win_probs_after)
                if _missing_cats:
                    logger.warning("category key mismatch: z-score cats %s not in MCMC cat_win_probs_after", _missing_cats)

            positional_impact = _compute_positional_impact(drop_candidate, fa.position, my_roster_scored)
            drop_out = _build_drop_out(drop_candidate)
            drop_out.positional_impact = positional_impact

            all_droppable = sorted(
                [
                    p for p in my_roster_scored
                    if not p.get("is_undroppable", False)
                    and not _is_protected_drop_candidate(p)
                    and p["name"] != drop_candidate["name"]
                ],
                key=_drop_candidate_value,
            )
            alternative_drops = []
            for _alt in all_droppable[:2]:
                _alt_out = _build_drop_out(_alt)
                _alt_out.positional_impact = _compute_positional_impact(_alt, fa.position, my_roster_scored)
                alternative_drops.append(_alt_out)

            _constraint: Optional[str] = None
            _fa_injury_upper = (fa.injury_status or "").upper()
            _IL_KW = ("IL", "DL", "60-DAY", "15-DAY", "10-DAY")
            if any(kw in _fa_injury_upper for kw in _IL_KW) and _il_slots_available == 0:
                _constraint = "IL slots full — move an injured player to IL first"
            elif _rec_faab_balance is not None and _rec_faab_balance < 1:
                _constraint = "FAAB budget exhausted — free agents only"

            roster_context = {
                "active_player_count": sum(1 for p in my_roster_scored if p.get("status") not in _IL_STATUSES),
                "add_weekly_starts": fa.starts_this_week,
                "drop_weekly_starts": drop_candidate.get("starts_this_week", 0),
            }

            recommendations.append(RosterMoveRecommendation(
                action="ADD_DROP",
                add_player=fa,
                drop_player_name=drop_candidate["name"],
                drop_player_position=drop_candidate["positions"][0] if drop_candidate["positions"] else "?",
                rationale=rationale,
                category_targets=[
                    k for k, v in (fa.category_contributions or {}).items()
                    if isinstance(v, (int, float)) and v > 0
                ],
                need_score=round(gain, 3),
                confidence=confidence,
                statcast_signals=fa_signals,
                regression_delta=fa_reg_delta,
                win_prob_before=_mcmc.get("win_prob_before", 0.0),
                win_prob_after=_mcmc.get("win_prob_after", 0.0),
                win_prob_gain=_mcmc.get("win_prob_gain", 0.0),
                category_win_probs=_mcmc.get("category_win_probs_after", {}),
                mcmc_enabled=_mcmc.get("mcmc_enabled", False),
                drop_player=drop_out,
                category_deltas=category_deltas,
                alternative_drops=alternative_drops,
                positional_impact=positional_impact,
                roster_context=roster_context,
                constraint_warning=_constraint,
            ))

    except YahooAuthError as exc:
        logger.error("Waiver recommendations endpoint -- Yahoo auth error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Yahoo auth failed - refresh token may be expired. ({exc})",
        ) from exc
    except YahooAPIError as exc:
        logger.error("Waiver recommendations endpoint -- Yahoo API error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Yahoo API error: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Waiver recommendations endpoint failed unexpectedly: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Unexpected error: {exc}",
        ) from exc

    def _safe_need_score(r):
        ns = r.need_score
        if isinstance(ns, (int, float)):
            return float(ns)
        if isinstance(ns, tuple) and ns:
            logger.warning(
                "Waiver rec leaked tuple need_score for %s; using first element",
                getattr(getattr(r, "add_player", None), "name", "?"),
            )
            return float(ns[0])
        logger.warning("Waiver rec non-numeric need_score %r; treating as 0.0", ns)
        return 0.0

    return WaiverRecommendationsResponse(
        week_end=week_end,
        matchup_opponent=matchup_opponent,
        recommendations=sorted(recommendations, key=_safe_need_score, reverse=True),
        category_deficits=category_deficits,
    )


# ============================================================================
# SAVED LINEUP
# ============================================================================

@router.post("/api/fantasy/lineup")
async def save_fantasy_lineup(
    payload: dict,
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """Save a daily lineup. Body: {lineup_date, platform, positions, projected_points, notes}"""
    from datetime import date as date_type

    lineup_date_raw = payload.get("lineup_date")
    if not lineup_date_raw:
        raise HTTPException(status_code=422, detail="lineup_date is required")
    try:
        lineup_date = date_type.fromisoformat(str(lineup_date_raw))
    except ValueError:
        raise HTTPException(status_code=422, detail="lineup_date must be YYYY-MM-DD")

    platform = payload.get("platform", "yahoo")
    positions = payload.get("positions", {})
    if not positions:
        raise HTTPException(status_code=422, detail="positions dict is required")

    existing = db.query(FantasyLineup).filter_by(
        lineup_date=lineup_date, platform=platform
    ).first()
    if existing:
        existing.positions = positions
        existing.projected_points = payload.get("projected_points")
        existing.notes = payload.get("notes")
        db.commit()
        return {"message": "Lineup updated", "id": existing.id}

    lineup = FantasyLineup(
        lineup_date=lineup_date,
        platform=platform,
        positions=positions,
        projected_points=payload.get("projected_points"),
        notes=payload.get("notes"),
    )
    db.add(lineup)
    db.commit()
    db.refresh(lineup)
    return {"message": "Lineup saved", "id": lineup.id}


@router.get("/api/fantasy/saved-lineup/{lineup_date}")
async def get_fantasy_lineup(
    lineup_date: str,
    platform: str = Query("yahoo"),
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """Retrieve a previously saved DK/Yahoo lineup for a given date."""
    from datetime import date as date_type

    try:
        ld = date_type.fromisoformat(lineup_date)
    except ValueError:
        raise HTTPException(status_code=422, detail="lineup_date must be YYYY-MM-DD")

    lineup = db.query(FantasyLineup).filter_by(lineup_date=ld, platform=platform).first()
    if lineup is None:
        raise HTTPException(status_code=404, detail="No lineup saved for this date")

    return {
        "lineup_date": lineup_date,
        "platform": lineup.platform,
        "positions": lineup.positions,
        "projected_points": lineup.projected_points,
        "actual_points": lineup.actual_points,
        "notes": lineup.notes,
    }


# ============================================================================
# YAHOO ROSTER / MATCHUP / LINEUP APPLY
# ============================================================================

@router.get("/api/fantasy/yahoo-diag")
async def yahoo_diag(user: str = Depends(verify_api_key)):
    """Diagnostic endpoint - returns Yahoo config status without making API calls."""
    client_id = os.getenv("YAHOO_CLIENT_ID", "")
    client_secret = os.getenv("YAHOO_CLIENT_SECRET", "")
    refresh_token = os.getenv("YAHOO_REFRESH_TOKEN", "")
    access_token = os.getenv("YAHOO_ACCESS_TOKEN", "")
    league_id = os.getenv("YAHOO_LEAGUE_ID", "72586")

    constructor_ok = False
    constructor_error = None
    try:
        _c = get_yahoo_client()
        constructor_ok = True
    except YahooAuthError as e:
        constructor_error = str(e)
    except Exception as e:
        constructor_error = f"Unexpected: {e}"

    token_ok = False
    token_error = None
    if constructor_ok:
        try:
            _c._ensure_token()
            token_ok = True
        except YahooAuthError as e:
            token_error = str(e)
        except Exception as e:
            token_error = f"Unexpected: {e}"

    return {
        "env_vars_present": {
            "YAHOO_CLIENT_ID": bool(client_id),
            "YAHOO_CLIENT_SECRET": bool(client_secret),
            "YAHOO_REFRESH_TOKEN": bool(refresh_token),
            "YAHOO_ACCESS_TOKEN": bool(access_token),
            "YAHOO_LEAGUE_ID": league_id,
        },
        "client_id_length": len(client_id),
        "client_secret_length": len(client_secret),
        "refresh_token_length": len(refresh_token),
        "constructor_ok": constructor_ok,
        "constructor_error": constructor_error,
        "token_refresh_ok": token_ok,
        "token_refresh_error": token_error,
    }


@router.get("/api/fantasy/roster", response_model=CanonicalRosterResponse)
async def get_fantasy_roster(
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Return the authenticated user's current Yahoo roster in CanonicalPlayerRow format.

    Phase 4 Workstream B: Returns CanonicalPlayerRow with rolling_14d stats from
    PlayerRollingStats table and season stats from Yahoo.
    """
    now_et = datetime.now(ZoneInfo("America/New_York"))

    try:
        client = get_yahoo_client()
    except YahooAuthError as exc:
        raise HTTPException(
            status_code=503,
            detail="Yahoo not configured -- set YAHOO_REFRESH_TOKEN",
        ) from exc

    team_key = os.getenv("YAHOO_TEAM_KEY", "469.l.72586.t.7")

    try:
        raw_players = client.get_roster(team_key=team_key)
    except YahooAuthError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except YahooAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # Enrich roster players with ownership % — get_roster() omits /ownership subresource;
    # _enrich_ownership_batch() fetches it via players;player_keys=.../ownership (same as
    # get_free_agents does). Without this call all roster players show 0.0% owned.
    client._enrich_ownership_batch(raw_players)

    # Extract player keys for rolling stats lookup
    player_keys = [p.get("player_key") for p in raw_players if p.get("player_key")]

    # Fetch rolling stats for all players across all window sizes in one query.
    # fetch_rolling_stats_for_players_all_windows does a single PlayerIDMapping
    # lookup and a single PlayerRollingStats query covering all three windows,
    # replacing the six DB round-trips that resulted from three separate calls.
    _rolling_all = fetch_rolling_stats_for_players_all_windows(
        db=db,
        yahoo_player_keys=player_keys,
        as_of_date=now_et.strftime("%Y-%m-%d"),
        window_sizes=[7, 14, 30],
    )
    rolling_stats_7d = _rolling_all.get(7, {})
    rolling_stats_14d = _rolling_all.get(14, {})
    rolling_stats_30d = _rolling_all.get(30, {})

    # Fetch Yahoo season stats for all roster players — the canonical router
    # mapper reads yahoo_player["stats"] to populate CanonicalPlayerRow
    # season_stats, but get_roster() intentionally does not batch-fetch
    # stats (Yahoo's team/roster subresource does not support stats=...).
    # Without this call, season_stats is null for every roster row.
    season_stats_by_key: dict = {}
    try:
        if player_keys:
            season_stats_by_key = client.get_players_stats_batch(
                player_keys, stat_type="season"
            )
    except Exception as _season_err:
        logger.warning(
            "Roster season stats batch fetch failed (non-fatal): %s", _season_err
        )

    # Fallback: load ownership % from PositionEligibility if Yahoo live fetch
    # returned 0.0 (roster endpoint does not include ownership blocks).
    _pe_ownership: dict[str, float] = {}
    try:
        from backend.models import PositionEligibility as _PositionEligibility
        if player_keys:
            _pe_rows = db.query(_PositionEligibility).filter(
                _PositionEligibility.yahoo_player_key.in_(player_keys)
            ).all()
            _pe_ownership = {
                r.yahoo_player_key: r.league_rostered_pct
                for r in _pe_rows
                if r.league_rostered_pct is not None
            }
    except Exception as _pe_err:
        logger.warning(
            "Roster PositionEligibility ownership fallback failed (non-fatal): %s",
            _pe_err,
        )

    # Resolve BDL and MLBAM IDs via PlayerIDMapping — required for rolling stats
    # and for populating bdl_player_id/mlbam_id in CanonicalPlayerRow output.
    player_key_to_ids = _resolve_roster_player_bdl_ids(db, raw_players)
    injury_overlays_by_key = load_injury_overlays_for_yahoo_players(
        db,
        raw_players,
        now_et=now_et,
    )

    # Batch-query PlayerProjection for all roster players (ros_projection hydration)
    from backend.models import PlayerProjection as _PlayerProjection
    from backend.stat_contract import SCORING_CATEGORY_CODES as _SCC
    # Primary lookup: by MLBAM ID via PlayerIDMapping crosswalk.
    # PlayerProjection.player_id stores MLBAM IDs after Steamer ingestion.
    _proj_mlbam_ids = [
        str(ids["mlbam_id"])
        for pk, ids in player_key_to_ids.items()
        if ids.get("mlbam_id") is not None
    ]
    _projections_by_mlbam: dict[str, _PlayerProjection] = {}
    if _proj_mlbam_ids:
        _proj_rows = db.query(_PlayerProjection).filter(
            _PlayerProjection.player_id.in_(_proj_mlbam_ids)
        ).all()
        _projections_by_mlbam = {p.player_id: p for p in _proj_rows}

    # Secondary lookup: by normalized player_name for Steamer rows whose
    # player_id is a Steamer internal key (not Yahoo numeric ID).
    # Single ORM query (replaces the previous raw-SQL-then-re-query pattern).
    _projections_by_name: dict[str, _PlayerProjection] = {}
    try:
        _name_rows = (
            db.query(_PlayerProjection)
            .filter(
                _PlayerProjection.cat_scores.isnot(None),
                cast(_PlayerProjection.cat_scores, Text) != "{}",
            )
            .all()
        )
        for _nr in _name_rows:
            if _nr.player_name:
                _projections_by_name[_normalize_identity_name(_nr.player_name)] = _nr
    except Exception as _nq_err:
        logger.warning("roster: name-projection index build failed: %s", _nq_err)

    # Build a canonical-code upper→lower lookup map once, outside the per-player loop.
    # cat_scores_builder stores keys lowercase (e.g. "hr", "k_bat", "era").
    # SCORING_CATEGORY_CODES are uppercase (e.g. "HR_B", "K_B", "ERA").
    # Map: uppercase_code -> lowercase_cat_scores_key
    _UPPER_TO_LOWER: dict[str, str] = {
        "R": "r", "H": "h", "HR_B": "hr", "RBI": "rbi",
        "K_B": "k_bat", "TB": "tb", "AVG": "avg", "OPS": "ops", "NSB": "nsb",
        "W": "w", "L": "l", "HR_P": "hr_pit", "K_P": "k_pit",
        "ERA": "era", "WHIP": "whip", "K_9": "k9", "QS": "qs", "NSV": "nsv",
        "IP": "ip", "SV": "sv", "HLD": "hld",
    }

    def _build_ros_proj(proj_row: _PlayerProjection):
        """Convert a PlayerProjection row into a CategoryStats object."""
        if not proj_row or not proj_row.cat_scores:
            return None
        raw: dict = proj_row.cat_scores
        _values: dict[str, float | None] = {code: None for code in _SCC}
        for upper_code in _SCC:
            # Try direct uppercase key first (future-proofing)
            if upper_code in raw:
                _values[upper_code] = float(raw[upper_code]) if raw[upper_code] is not None else None
                continue
            # Try lowercase mapping
            lower_key = _UPPER_TO_LOWER.get(upper_code)
            if lower_key and lower_key in raw:
                _values[upper_code] = float(raw[lower_key]) if raw[lower_key] is not None else None
        # Only return if at least some values were populated
        if not any(v is not None for v in _values.values()):
            return None
        try:
            from backend.contracts import CategoryStats as _CategoryStats
            return _CategoryStats(values=_values)
        except Exception:
            return None

    # Map each Yahoo player to CanonicalPlayerRow
    canonical_players = []
    for p in raw_players:
        player_key = p.get("player_key")
        if not player_key:
            continue

        # Start with the raw player dict and enrich it.
        merged_player = dict(p)

        # Merge ownership fallback from PositionEligibility if Yahoo returned 0.0
        if player_key in _pe_ownership and merged_player.get("percent_owned", 0.0) == 0.0:
            merged_player["percent_owned"] = _pe_ownership[player_key]

        # Merge season stats from batch so the mapper can translate them.
        if player_key in season_stats_by_key:
            merged_player["stats"] = season_stats_by_key[player_key]

        # Merge BDL/MLBAM IDs from PlayerIDMapping lookup.
        if player_key in player_key_to_ids:
            ids = player_key_to_ids[player_key]
            merged_player["bdl_player_id"] = ids.get("bdl_id")
            merged_player["mlbam_id"] = ids.get("mlbam_id")

        # Fetch all rolling windows for this player
        rs_7d = rolling_stats_7d.get(player_key)
        rs_14d = rolling_stats_14d.get(player_key)
        rs_15d = None
        rs_30d = rolling_stats_30d.get(player_key)

        # Build ros_projection from PlayerProjection.cat_scores if available.
        # Priority 1: MLBAM ID lookup via PlayerIDMapping crosswalk.
        # Priority 2: Normalized name lookup (catches unmapped call-ups).
        _ros_proj = None
        _proj_row = None
        _ids = player_key_to_ids.get(player_key) or {}
        _mlbam = _ids.get("mlbam_id")
        if _mlbam is not None:
            _proj_row = _projections_by_mlbam.get(str(_mlbam))
        if _proj_row is None:
            _pname = p.get("name", "")
            if _pname:
                _proj_row = _projections_by_name.get(_normalize_identity_name(_pname))
        _ros_proj = _build_ros_proj(_proj_row) if _proj_row else None
        if _ros_proj is not None:
            logger.debug("roster: ros_projection populated for %s", p.get("name"))
        else:
            logger.debug("roster: ros_projection missing for %s (no matching projection)", p.get("name"))

        overlay = injury_overlays_by_key.get(player_key)
        if overlay is not None:
            merged_player["injury_status"] = getattr(overlay, "status", None) or merged_player.get("injury_status")
            merged_player["injury_note"] = merged_player.get("injury_note") or getattr(overlay, "note", None)

        canonical_row = map_yahoo_player_to_canonical_row(
            yahoo_player=merged_player,
            rolling_stats_7d=rs_7d,
            rolling_stats_14d=rs_14d,
            rolling_stats_15d=rs_15d,
            rolling_stats_30d=rs_30d,
            computed_at=now_et,
            ros_projection=_ros_proj,
        )
        if overlay is not None:
            canonical_row = canonical_row.model_copy(
                update={
                    "injury_status": overlay.status,
                    "injury_return_timeline": overlay.return_timeline,
                }
            )
        canonical_players.append(canonical_row)

    # Post-serialization guard: log and coerce any boolean injury_status that bypassed validators
    _guarded = []
    for _p in canonical_players:
        if isinstance(_p.injury_status, bool):
            logger.error(
                "BOOLEAN LEAK: injury_status for %s is bool %r; coercing to string",
                _p.player_name,
                _p.injury_status,
            )
            _p = _p.model_copy(update={"injury_status": "IL" if _p.injury_status else None})
        _guarded.append(_p)
    canonical_players = _guarded

    # Build freshness metadata
    freshness = FreshnessMetadata(
        primary_source="yahoo",
        fetched_at=None,  # TODO: track from Yahoo client
        computed_at=now_et,
        staleness_threshold_minutes=60,
        is_stale=False,  # TODO: compute from fetched_at
    )

    return CanonicalRosterResponse(
        team_key=team_key,
        players=canonical_players,
        count=len(canonical_players),
        freshness=freshness,
    )


@router.post("/api/fantasy/roster/move", response_model=RosterMoveResponse)
async def move_roster_player(
    request: RosterMoveRequest,
):
    """
    Move a player to a new roster slot.

    Validates the move, builds the full lineup with the player moved,
    and submits to Yahoo via set_lineup.

    Valid target positions: C, 1B, 2B, 3B, SS, LF, CF, RF, OF, Util, SP, RP, P, BN, IL, IL60.
    """
    from pydantic import ValidationError

    now_et = datetime.now(ZoneInfo("America/New_York"))

    # Valid roster slots (LF/CF/RF are Yahoo's granular outfield positions)
    valid_positions = {
        "C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "OF", "Util",
        "SP", "RP", "P",
        "BN", "IL", "IL60",
    }

    # Validate target position
    if request.target_position not in valid_positions:
        return RosterMoveResponse(
            success=False,
            player_key=request.player_key,
            to_position=request.target_position,
            message=f"Invalid position: {request.target_position}. Valid: {sorted(valid_positions)}",
            freshness=FreshnessMetadata(
                primary_source="yahoo",
                fetched_at=None,
                computed_at=now_et,
                staleness_threshold_minutes=60,
                is_stale=False,
            ),
        )

    try:
        client = get_yahoo_client()
    except YahooAuthError as exc:
        raise HTTPException(
            status_code=503,
            detail="Yahoo not configured -- set YAHOO_REFRESH_TOKEN",
        ) from exc

    team_key = os.getenv("YAHOO_TEAM_KEY", "469.l.72586.t.7")

    # Fetch current roster
    try:
        raw_players = client.get_roster(team_key=team_key)
    except YahooAuthError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except YahooAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # Find the player being moved
    player_to_move = None
    from_position = None
    for p in raw_players:
        if p.get("player_key") == request.player_key:
            player_to_move = p
            from_position = p.get("selected_position")
            break

    if not player_to_move:
        return RosterMoveResponse(
            success=False,
            player_key=request.player_key,
            to_position=request.target_position,
            message=f"Player {request.player_key} not found on roster",
            freshness=FreshnessMetadata(
                primary_source="yahoo",
                fetched_at=None,
                computed_at=now_et,
                staleness_threshold_minutes=60,
                is_stale=False,
            ),
        )

    # IL guard: players with an active IL designation cannot be placed in active slots.
    if _is_il_designated(player_to_move) and request.target_position not in _IL_SLOTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{player_to_move.get('name', request.player_key)} has IL designation "
                f"({player_to_move.get('status')}) — must be placed in IL or IL60 slot, "
                f"not {request.target_position!r}"
            ),
        )

    # Position eligibility guard: player must be eligible for target slot.
    if request.target_position not in _EXEMPT_SLOTS:
        _player_eligible = player_to_move.get("eligible_positions") or player_to_move.get("positions") or []
        if not _can_fill_slot(_player_eligible, request.target_position, player_to_move.get("name", request.player_key)):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{player_to_move.get('name', request.player_key)} is not eligible for "
                    f"{request.target_position!r} slot — eligible: "
                    f"{', '.join(_player_eligible) or 'unknown'}"
                ),
            )

    # Build lineup list: all players with the moved player's position updated
    lineup = []
    for p in raw_players:
        player_key = p.get("player_key")
        if not player_key:
            continue

        if player_key == request.player_key:
            # Move to target position
            lineup.append({
                "player_key": player_key,
                "position": request.target_position,
            })
        else:
            # Keep existing position
            existing_pos = p.get("selected_position", "BN")
            lineup.append({
                "player_key": player_key,
                "position": existing_pos,
            })

    # Apply the lineup change
    try:
        result = client.set_lineup(team_key=team_key, lineup=lineup)
        applied = result.get("applied", [])
        warnings = result.get("warnings", [])
    except YahooAPIError as exc:
        return RosterMoveResponse(
            success=False,
            player_key=request.player_key,
            from_position=from_position,
            to_position=request.target_position,
            message=f"Yahoo API error: {str(exc)}",
            freshness=FreshnessMetadata(
                primary_source="yahoo",
                fetched_at=None,
                computed_at=now_et,
                staleness_threshold_minutes=60,
                is_stale=False,
            ),
        )

    success = request.player_key in applied
    if success:
        message = f"Moved {player_to_move.get('name', request.player_key)} from {from_position} to {request.target_position}"
    else:
        message = f"Failed to move {player_to_move.get('name', request.player_key)} to {request.target_position}"

    return RosterMoveResponse(
        success=success,
        player_key=request.player_key,
        from_position=from_position,
        to_position=request.target_position,
        message=message,
        warnings=warnings,
        freshness=FreshnessMetadata(
            primary_source="yahoo",
            fetched_at=None,
            computed_at=now_et,
            staleness_threshold_minutes=60,
            is_stale=False,
        ),
    )


@router.post("/api/fantasy/roster/bulk-apply", response_model=BulkRosterMoveResponse)
async def bulk_apply_roster_moves(
    request: BulkRosterMoveRequest,
):
    """
    Apply multiple roster moves atomically via a single set_lineup call.

    All moves are validated before any Yahoo API call is made.
    Returns 400 if any move fails validation.
    Returns 200 with applied_count/failed_count/errors after Yahoo processes the lineup.
    """
    valid_positions = {
        "C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "OF", "Util",
        "SP", "RP", "P",
        "BN", "IL", "IL60",
    }

    if not request.moves:
        raise HTTPException(status_code=400, detail={"errors": ["No moves provided"]})

    # Phase 1: validate positions before touching Yahoo
    validation_errors: list[str] = []
    for move in request.moves:
        if move.target_position not in valid_positions:
            validation_errors.append(
                f"Invalid position {move.target_position!r} for player {move.player_key}"
            )

    if validation_errors:
        raise HTTPException(status_code=400, detail={"errors": validation_errors})

    try:
        client = get_yahoo_client()
    except YahooAuthError as exc:
        raise HTTPException(
            status_code=503,
            detail="Yahoo not configured -- set YAHOO_REFRESH_TOKEN",
        ) from exc

    team_key = os.getenv("YAHOO_TEAM_KEY", "469.l.72586.t.7")

    try:
        raw_players = client.get_roster(team_key=team_key)
    except YahooAuthError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except YahooAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # Index current roster
    roster_by_key = {p["player_key"]: p for p in raw_players if p.get("player_key")}

    # Phase 2: validate all player keys exist on roster, then IL guard
    for move in request.moves:
        if move.player_key not in roster_by_key:
            validation_errors.append(f"Player {move.player_key} not found on roster")

    if validation_errors:
        raise HTTPException(status_code=400, detail={"errors": validation_errors})

    # Phase 3: IL guard — IL-designated players may only go to IL/IL60 slots
    for move in request.moves:
        player = roster_by_key.get(move.player_key, {})
        if _is_il_designated(player) and move.target_position not in _IL_SLOTS:
            validation_errors.append(
                f"{player.get('name', move.player_key)} has IL designation "
                f"({player.get('status')}) — cannot move to {move.target_position!r}"
            )

    if validation_errors:
        raise HTTPException(status_code=400, detail={"errors": validation_errors})

    # Phase 4: Position eligibility guard
    for move in request.moves:
        if move.target_position not in _EXEMPT_SLOTS:
            player = roster_by_key.get(move.player_key, {})
            _eligible = player.get("eligible_positions") or player.get("positions") or []
            if not _can_fill_slot(_eligible, move.target_position, player.get("name", move.player_key)):
                validation_errors.append(
                    f"{player.get('name', move.player_key)} is not eligible for "
                    f"{move.target_position!r} slot — eligible: {', '.join(_eligible) or 'unknown'}"
                )

    if validation_errors:
        raise HTTPException(status_code=400, detail={"errors": validation_errors})

    # Build full lineup with all moves applied at once (atomic)
    move_map = {m.player_key: m.target_position for m in request.moves}
    lineup = []
    for player in raw_players:
        pk = player.get("player_key")
        if not pk:
            continue
        target = move_map.get(pk, player.get("selected_position", "BN"))
        lineup.append({"player_key": pk, "position": target})

    try:
        result = client.set_lineup(team_key=team_key, lineup=lineup)
    except YahooAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    applied = set(result.get("applied", []))
    move_keys = {m.player_key for m in request.moves}
    applied_count = len(move_keys & applied)
    failed_count = len(move_keys) - applied_count

    errors = [
        f"Player {pk} was not confirmed in Yahoo's applied list"
        for pk in sorted(move_keys - applied)
    ]

    return BulkRosterMoveResponse(
        applied_count=applied_count,
        failed_count=failed_count,
        errors=errors,
    )


@router.get("/api/fantasy/matchup-preview", response_model=MatchupPreviewResponse)
async def get_matchup_preview(
    db: Session = Depends(get_db),
):
    """
    Next-week H2H matchup preview using MCMC simulation.

    Fetches next week's Yahoo matchup opponent, simulates category win probabilities
    against a league-average opponent baseline, and surfaces streaming recommendations
    for categories projected to lose. Falls back to current-week opponent when
    next week's matchup isn't published yet.
    """
    from backend.fantasy_baseball.mcmc_simulator import simulate_weekly_matchup
    from zoneinfo import ZoneInfo

    try:
        client = get_yahoo_client()
    except YahooAuthError as exc:
        raise HTTPException(
            status_code=503,
            detail="Yahoo not configured — set YAHOO_REFRESH_TOKEN",
        ) from exc

    # Determine current fantasy week and compute next week
    current_week_num = 1
    try:
        league_meta = client.get_league()
        current_week_num = int(league_meta.get("current_week", 1))
    except Exception as _w_err:
        logger.warning("matchup_preview: league current_week fetch failed: %s", _w_err)

    next_week_num = current_week_num + 1

    # Resolve my team key once for scoreboard parsing.
    # Use the same default as all other roster endpoints to prevent empty-string fallback
    # that silently breaks team-key matching in _extract_opponent_from_scoreboard.
    _preview_my_team_key = os.getenv("YAHOO_TEAM_KEY", "469.l.72586.t.7")
    try:
        # get_my_team_key() is authoritative — prefer it over the env default
        _live_key = client.get_my_team_key()
        if _live_key:
            _preview_my_team_key = _live_key
    except Exception as _key_err:
        logger.debug(
            "matchup_preview: get_my_team_key() failed (%s), using env/default %s",
            _key_err, _preview_my_team_key,
        )

    def _extract_opponent_from_scoreboard(week: int) -> str:
        """Use the same proven get_scoreboard() path as the scoreboard endpoint."""
        try:
            raw_sb = client.get_scoreboard(week=week)
        except Exception as _sb_err:
            logger.warning("matchup_preview: get_scoreboard(week=%d) failed: %s", week, _sb_err)
            return "Unknown"
        matchups = _iter_scoreboard_matchup_teams(raw_sb or [])
        logger.debug(
            "matchup_preview: week=%d scoreboard has %d matchup(s), my_key=%r",
            week, len(matchups), _preview_my_team_key,
        )
        for _matchup_teams in matchups:
            _my_t = None
            for _t in _matchup_teams:
                _tk = _t[0]
                if _tk and _preview_my_team_key and (
                    _tk == _preview_my_team_key
                    or _tk in _preview_my_team_key
                    or _preview_my_team_key in _tk
                ):
                    _my_t = _t
                    break
            if _my_t is not None:
                _opp = next((_t for _t in _matchup_teams if _t[0] != _my_t[0]), None)
                if _opp and _opp[1]:
                    return _opp[1]
        if matchups:
            logger.warning(
                "matchup_preview: week=%d had %d matchup(s) but none matched my_key=%r — "
                "team keys in scoreboard: %s",
                week, len(matchups),
                _preview_my_team_key,
                [t[0] for m in matchups for t in m],
            )
        return "Unknown"

    # Get next week's opponent name — fall back to current week if not yet published
    opponent_name = "Unknown"
    preview_week = next_week_num
    opponent_name = _extract_opponent_from_scoreboard(next_week_num)
    if opponent_name == "Unknown":
        logger.warning(
            "matchup_preview: next-week (week %d) opponent not found in scoreboard, "
            "falling back to current week %d",
            next_week_num, current_week_num,
        )
        preview_week = current_week_num
        opponent_name = _extract_opponent_from_scoreboard(current_week_num)

    # Build my roster with cat_scores from player_projections table
    try:
        my_roster, _ = _fetch_rosters_for_simulate(db)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Roster fetch failed: {exc}") from exc

    if not my_roster:
        raise HTTPException(
            status_code=422,
            detail="No roster data available — check Yahoo API connection",
        )

    # TBD opponent check: suppress ALL projections when opponent is undetermined
    # This includes All-Star breaks, bye weeks, and cases where Yahoo hasn't published
    # the next matchup yet. Return a simplified TBD response instead.
    _TBD_INDICATORS = {"Unknown", "TBD", "All-Star Break", "Bye Week", "None"}
    if opponent_name in _TBD_INDICATORS:
        return MatchupPreviewResponse(
            week_number=preview_week,
            opponent_name=opponent_name,
            opponent_logo=None,
            overall_win_prob=None,  # No win % without opponent
            category_projections=[],  # Empty category table
            weak_categories=[],  # No streaming recommendations
            schedule_advantage=ScheduleAdvantage(my_games=0, opponent_games=0),
            message="MATCHUP TBD: Opponent not yet published. Projections unavailable until matchup is confirmed.",
        )

    # Simulate full-week projection against league-average opponent (empty = z=0 baseline)
    try:
        sim = simulate_weekly_matchup(
            my_roster=my_roster,
            opponent_roster=[],
            n_sims=2000,
            remaining_fraction=1.0,
        )
    except Exception as exc:
        logger.error("matchup_preview: simulation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Simulation failed: {exc}") from exc

    _CAT_LABELS: dict[str, str] = {
        "r": "Runs", "h": "Hits", "hr_b": "HR", "rbi": "RBI",
        "k_b": "K", "tb": "Total Bases", "avg": "AVG", "ops": "OPS", "nsb": "NSB",
        "w": "W", "l": "L", "hr_p": "HR", "k_p": "K",
        "era": "ERA", "whip": "WHIP", "k_9": "K/9", "qs": "QS", "nsv": "NSV",
    }

    cat_win_probs: dict[str, float] = sim.get("category_win_probs", {})
    category_projections: list[MatchupPreviewCategoryProjection] = []
    weak_categories: list[WeakCategory] = []

    for cat, win_prob in sorted(cat_win_probs.items()):
        category_projections.append(
            MatchupPreviewCategoryProjection(
                category=cat,
                win_prob=round(win_prob, 4),
            )
        )
        if win_prob < 0.4:
            label = _CAT_LABELS.get(cat, cat.upper())
            weak_categories.append(
                WeakCategory(
                    category=cat,
                    label=label,
                    win_prob=round(win_prob, 4),
                    reason=(
                        f"Projected to lose {label} ({win_prob:.0%} win rate) — "
                        f"target streamers with {label} upside"
                    ),
                )
            )

    return MatchupPreviewResponse(
        week_number=preview_week,
        opponent_name=opponent_name,
        opponent_logo=None,
        overall_win_prob=round(float(sim.get("win_prob", 0.5)), 4),
        category_projections=category_projections,
        weak_categories=weak_categories,
        schedule_advantage=ScheduleAdvantage(my_games=0, opponent_games=0),
        message=None,
    )


@router.post("/api/fantasy/roster/optimize", response_model=RosterOptimizeResponse)
async def optimize_roster(
    request: RosterOptimizeRequest,
    db: Session = Depends(get_db),
):
    """
    Optimize roster lineup assignment.

    Uses rolling_14d stats to score players and recommend optimal
    starter/bench assignments based on roster slots.

    Valid roster slots: C, 1B, 2B, 3B, SS, OF (x3, accepts LF/CF/RF), Util, SP (x2), RP (x2), P, BN (x5).
    """
    now_et = datetime.now(ZoneInfo("America/New_York"))
    target_date = request.target_date or now_et.strftime("%Y-%m-%d")

    # Roster slot configuration (Yahoo H2H standard)
    # Yahoo uses LF/CF/RF but our league has 3 generic OF slots.
    # _can_fill_slot() handles the LF/CF/RF → OF eligibility mapping.
    slot_capacity = {
        "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3, "Util": 1,
        "SP": 2, "RP": 2, "P": 1, "BN": 5,
    }

    # Schedule gate: check if any MLB games exist for target_date before running solver.
    # Uses its own SessionLocal so it never consumes the route's db mock in tests.
    # Fail open (schedule_available=True) if the check itself errors — don't block user.
    schedule_available = True
    try:
        from backend.models import SessionLocal as _SL, ProbablePitcherSnapshot as _PPS
        _sched_db = _SL()
        try:
            _snap_count = _sched_db.query(_PPS).filter(
                _PPS.game_date == target_date
            ).count()
        finally:
            _sched_db.close()
        if _snap_count == 0:
            import requests as _sreq
            _sr = _sreq.get(
                "https://statsapi.mlb.com/api/v1/schedule",
                params={"sportId": 1, "date": target_date, "gameType": "R"},
                timeout=5,
            )
            if _sr.ok:
                _api_games = sum(
                    len(d.get("games", []))
                    for d in _sr.json().get("dates", [])
                )
                schedule_available = _api_games > 0
            else:
                schedule_available = False
    except Exception as _sched_err:
        logger.debug("Schedule gate check failed (non-fatal): %s", _sched_err)

    try:
        client = get_yahoo_client()
    except YahooAuthError as exc:
        raise HTTPException(
            status_code=503,
            detail="Yahoo not configured -- set YAHOO_REFRESH_TOKEN",
        ) from exc

    team_key = os.getenv("YAHOO_TEAM_KEY", "469.l.72586.t.7")

    # Fetch current roster
    try:
        raw_players = client.get_roster(team_key=team_key)
    except YahooAuthError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except YahooAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # Validate roster player data structure before processing
    # This catches data corruption early and provides actionable error messages
    for i, p in enumerate(raw_players):
        if not isinstance(p, dict):
            logger.error("Roster player at index %d is not a dict: %s", i, type(p))
            raise HTTPException(
                status_code=500,
                detail={
                    "error_code": "ROSTER_DATA_CORRUPTED",
                    "message": f"Roster player at index {i} has invalid type",
                    "index": i,
                    "type": str(type(p)),
                    "recovery_hint": "Check Yahoo API response format and player data structure"
                }
            )

        # Log warning for unexpected status types (helps debug data source)
        status_val = p.get("status")
        if status_val is not None and not isinstance(status_val, (str, bool)):
            logger.warning(
                "Player %s has unexpected status type=%s (value=%s)",
                p.get("name", "unknown"), type(status_val), status_val
            )

    # Resolve Yahoo roster keys to BDL IDs using canonical yahoo_key linkage first.
    player_key_to_ids = _resolve_roster_player_bdl_ids(db, raw_players)

    # Fetch player_scores for each player's most recent date (not exact target_date match)
    # Phase 4.5a Priority 2 fix: Use actual as_of_date from player_scores, not requested date
    bdl_ids = list({
        ids["bdl_id"] for ids in player_key_to_ids.values()
        if ids.get("bdl_id") is not None
    })
    player_scores_map = {}
    as_of_dates = set()  # Track actual as_of_dates found
    staleness_warning = False
    actual_data_date = target_date  # Initialize with requested date
    fallback_count = 0

    if bdl_ids:
        # Get most recent score for each player at or before target_date
        subq = (
            db.query(
                PlayerScore.bdl_player_id,
                func.max(PlayerScore.as_of_date).label("max_date"),
            )
            .filter(
                PlayerScore.bdl_player_id.in_(bdl_ids),
                PlayerScore.window_days == 14,
                PlayerScore.as_of_date <= target_date,
            )
            .group_by(PlayerScore.bdl_player_id)
            .subquery()
        )

        scores = (
            db.query(PlayerScore)
            .join(
                subq,
                (PlayerScore.bdl_player_id == subq.c.bdl_player_id) &
                (PlayerScore.as_of_date == subq.c.max_date),
            )
            .filter(PlayerScore.window_days == 14)
            .all()
        )
        # Map bdl_id -> score_0_100 and track as_of_dates
        for s in scores:
            player_scores_map[s.bdl_player_id] = s.score_0_100
            as_of_dates.add(s.as_of_date)

        # Check for staleness: if fewer than 50% of players have scores, warn
        if len(player_scores_map) < len(bdl_ids) * 0.5:
            staleness_warning = True

        # Use the most recent as_of_date in response, or target_date if none found
        actual_data_date = max(as_of_dates) if as_of_dates else target_date

    # Build player data with scores; skip IL-designated players (they stay in IL slots)
    player_data = []
    il_player_count = 0
    for p in raw_players:
        player_key = p.get("player_key")
        if not player_key:
            continue

        if _is_il_designated(p):
            il_player_count += 1
            continue

        score = 50.0
        score_source = "default"
        if player_key in player_key_to_ids:
            bdl_id = player_key_to_ids[player_key].get("bdl_id")
            if bdl_id in player_scores_map:
                score = player_scores_map[bdl_id]
                score_source = "player_scores"
            else:
                score, score_source = _projection_fallback_score(p)
                fallback_count += 1
        else:
            score, score_source = _projection_fallback_score(p)
            fallback_count += 1

        player_data.append({
            "player_key": player_key,
            "name": p.get("name", "Unknown"),
            "eligible_positions": p.get("positions") or [],
            "current_position": p.get("selected_position", "BN"),
            "lineup_score": score,
            "score_source": score_source,
        })

    # TASK-5: Normalize hitter and pitcher scores to a common 0-100 scale
    # before routing to separate solver tracks.
    #
    # score_0_100 from player_scores is a within-cohort percentile rank:
    #   - A pitcher at 97 means 97th percentile among pitchers.
    #   - A hitter at 87 means 87th percentile among hitters.
    # These are NOT comparable across positions — do not use raw score_0_100 for
    # cross-group display.  We min-max normalize each group independently so that
    # the displayed lineup_score is "position-relative rank" (0-100) for both groups.
    # Players with source "no_score" (no real data) always stay at 0.0 and are
    # excluded from the normalization range to avoid anchoring the floor.
    _PITCHER_POSITIONS_SET = {"SP", "RP", "P"}

    def _normalize_group_scores(group: list) -> None:
        """In-place min-max normalize lineup_score within a group, preserving no_score=0."""
        real_scores = [p["lineup_score"] for p in group if p.get("score_source") != "no_score"]
        if not real_scores:
            return
        lo, hi = min(real_scores), max(real_scores)
        if hi == lo:
            # All players have identical scores — preserve as-is (already normalized)
            return
        for p in group:
            if p.get("score_source") == "no_score":
                p["lineup_score"] = 0.0  # stays below any real-scored player
            else:
                p["lineup_score"] = round((p["lineup_score"] - lo) / (hi - lo) * 100.0, 2)

    # Split into groups, normalize, then re-merge.  This ensures the ILP solver and
    # the greedy pitcher sort both use position-relative scores, making the response
    # scores directly comparable across groups.
    _hitter_group = [p for p in player_data if not (bool(p.get("eligible_positions")) and {pos.upper() for pos in p["eligible_positions"]}.issubset(_PITCHER_POSITIONS_SET))]
    _pitcher_group = [p for p in player_data if (bool(p.get("eligible_positions")) and {pos.upper() for pos in p["eligible_positions"]}.issubset(_PITCHER_POSITIONS_SET))]
    _normalize_group_scores(_hitter_group)
    _normalize_group_scores(_pitcher_group)
    # Rebuild player_data in original order with normalized scores
    _norm_map = {p["player_key"]: p["lineup_score"] for p in _hitter_group + _pitcher_group}
    for p in player_data:
        if p["player_key"] in _norm_map:
            p["lineup_score"] = _norm_map[p["player_key"]]
    hitter_data = [
        p for p in player_data
        if not (
            bool(p.get("eligible_positions"))
            and {pos.upper() for pos in p["eligible_positions"]}.issubset(_PITCHER_POSITIONS_SET)
        )
    ]
    pitcher_data = [
        p for p in player_data
        if (
            bool(p.get("eligible_positions"))
            and {pos.upper() for pos in p["eligible_positions"]}.issubset(_PITCHER_POSITIONS_SET)
        )
    ]

    # Build solver inputs (player_key serves as player_id)
    players_for_solver = [{"player_id": p["player_key"], "name": p["name"]} for p in hitter_data]
    elite_scores = {
        p["player_key"]: EliteScore(
            total_score=p["lineup_score"],
            environment_score=0.0,
            matchup_multiplier=1.0,
            platoon_multiplier=1.0,
            form_adjusted_woba=0.0,
            regression_boost=0.0,
            lineup_spot_bonus=0.0,
            confidence=1.0,
            data_quality=p.get("score_source", "default"),
            reasoning=f"Score {p['lineup_score']:.1f} ({p.get('score_source', 'default')})",
        )
        for p in hitter_data
    }
    eligibility_map = {
        p["player_key"]: [pos.upper() for pos in (p.get("eligible_positions") or [])]
        for p in hitter_data
    }

    hitter_assignments = []
    if players_for_solver:
        solver = get_lineup_solver()
        optimized = solver.solve(players_for_solver, elite_scores, eligibility_map)
        if not optimized.assignments:
            # ILP infeasible (partial roster) — greedy fills what it can
            optimized = solver._solve_greedy(players_for_solver, elite_scores, eligibility_map, None)
        hitter_assignments = optimized.assignments

    # Map solver batting assignments → response format; normalize OF1/OF2/OF3 → "OF"
    _OF_SOLVER_SLOTS = {PositionSlot.OUTFIELD_1, PositionSlot.OUTFIELD_2, PositionSlot.OUTFIELD_3}
    placed_keys: set = set()
    starters = []
    for sa in hitter_assignments:
        slot_str = "OF" if sa.slot in _OF_SOLVER_SLOTS else sa.slot.value
        starters.append(PlayerSlotAssignment(
            player_key=sa.player_id,
            player_name=sa.player_name,
            assigned_slot=slot_str,
            lineup_score=round(sa.score, 2),
            reasoning=sa.reason or f"Score {sa.score:.1f}",
        ))
        placed_keys.add(sa.player_id)

    # Pitcher slots: simple greedy (SP → RP → P; no position-scarcity concern)
    pitcher_slot_fill = {"SP": 0, "RP": 0, "P": 0}
    pitcher_data.sort(key=lambda p: p["lineup_score"], reverse=True)
    for player in pitcher_data:
        if player["player_key"] in placed_keys:
            continue
        for slot in ["SP", "RP", "P"]:
            if pitcher_slot_fill[slot] >= slot_capacity[slot]:
                continue
            if _can_fill_slot(player["eligible_positions"], slot, player["name"]):
                starters.append(PlayerSlotAssignment(
                    player_key=player["player_key"],
                    player_name=player["name"],
                    assigned_slot=slot,
                    lineup_score=round(player["lineup_score"], 2),
                    reasoning=f"Score {player['lineup_score']:.1f} ({player.get('score_source', 'default')}), eligible for {slot}",
                ))
                pitcher_slot_fill[slot] += 1
                placed_keys.add(player["player_key"])
                break

    # Bench: hitters unassigned by solver + excess pitchers
    bench_assignments = []
    unrostered = []
    for player in player_data:
        if player["player_key"] in placed_keys:
            continue
        if len(bench_assignments) < slot_capacity["BN"]:
            bench_assignments.append(PlayerSlotAssignment(
                player_key=player["player_key"],
                player_name=player["name"],
                assigned_slot="BN",
                lineup_score=round(player["lineup_score"], 2),
                reasoning=f"Bench: score {player['lineup_score']:.1f}",
            ))
            placed_keys.add(player["player_key"])
        else:
            unrostered.append(player["player_key"])

    total_score = sum(a.lineup_score for a in starters) if starters else 0.0

    # Build message with staleness warning if needed
    # Use actual_data_date to reflect real data freshness, not requested date
    base_msg = f"Optimized lineup for {actual_data_date}"
    if staleness_warning:
        base_msg += " (Warning: Using projection fallback for some players - player_scores stale)"
    elif actual_data_date != target_date:
        base_msg += f" (Note: Data from {actual_data_date}, not requested {target_date})"
    elif fallback_count:
        base_msg += f" (Used projection fallback for {fallback_count} player{'s' if fallback_count != 1 else ''})"
    if il_player_count:
        base_msg += f" ({il_player_count} IL player{'s' if il_player_count != 1 else ''} excluded from active slots)"
    if not schedule_available:
        base_msg += f" — WARNING: No MLB games found for {target_date} (off-day or schedule not yet available)"

    return RosterOptimizeResponse(
        success=True,
        message=base_msg,
        target_date=target_date,
        starters=starters,
        bench=bench_assignments,
        unrostered=unrostered,
        total_lineup_score=round(total_score, 2),
        schedule_available=schedule_available,
        freshness=FreshnessMetadata(
            primary_source="yahoo",
            fetched_at=None,
            computed_at=now_et,
            staleness_threshold_minutes=60,
            is_stale=False,
        ),
    )


# Outfield positions: Yahoo returns LF/CF/RF as distinct positions.
# Any of LF, CF, RF, OF can fill an "OF" slot.
_OUTFIELD_POSITIONS = {"OF", "LF", "CF", "RF"}
_HITTER_POSITIONS = {"C", "1B", "2B", "3B", "SS", "OF", "LF", "CF", "RF", "DH"}

# IL slot names accepted by Yahoo's set_lineup API
_IL_SLOTS = frozenset({"IL", "IL60"})

# Slots that accept any player regardless of position eligibility
_EXEMPT_SLOTS = frozenset({"BN", "IL", "IL60"})


def _is_il_designated(player: dict) -> bool:
    """Return True if Yahoo status indicates an active IL designation.

    Matches: "IL", "IL10", "IL15", "IL60", "10-Day IL", "15-Day IL", "60-Day IL".

    Defensive: handles status as string, boolean, or None.
    Logs warning for unexpected types to aid data debugging.
    """
    raw_status = player.get("status")

    # Defensive: handle boolean status values (data corruption/API change)
    if isinstance(raw_status, bool):
        logger.warning(
            "Player %s has boolean status=%s - expected string. "
            "Treating as NOT IL-designated. Data may be corrupted.",
            player.get("name", "unknown"), raw_status
        )
        return False

    # Defensive: handle None or missing status
    if raw_status is None:
        return False

    # Expected case: status is string
    status = str(raw_status).upper().strip()
    return status.startswith("IL") or "-IL" in status


def _can_fill_slot(eligible_positions, slot, player_name) -> bool:
    """Check if player can fill the given roster slot."""
    if not eligible_positions:
        return False

    positions = [p.upper() for p in eligible_positions] if isinstance(eligible_positions, list) else [eligible_positions.upper()]

    # Direct position match
    if slot in positions:
        return True

    # OF slot accepts any outfield position (LF, CF, RF)
    if slot == "OF":
        return any(p in _OUTFIELD_POSITIONS for p in positions)

    # Util accepts any hitter position (including LF/CF/RF)
    if slot == "Util":
        return any(p in _HITTER_POSITIONS for p in positions)

    # P accepts any pitcher position
    if slot == "P":
        pitcher_positions = {"SP", "RP"}
        return any(p in pitcher_positions for p in positions)

    return False


@router.get("/api/fantasy/players/valuations")
async def get_player_valuations(
    date_str: Optional[str] = Query(None, alias="date"),
    league_key: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Return cached PlayerValuationReports for a given date and league."""
    target_date = date_str or datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

    try:
        query = """
            SELECT player_id, player_name, target_date, report, computed_at
            FROM player_valuation_cache
            WHERE invalidated_at IS NULL
        """
        params: dict = {}

        if league_key:
            query += " AND league_key = :league_key"
            params["league_key"] = league_key

        exact_rows = db.execute(
            text(query + " AND target_date = :tdate ORDER BY computed_at DESC"),
            {**params, "tdate": target_date},
        ).fetchall()

        if exact_rows:
            rows = exact_rows
            cache_status = "fresh"
        else:
            fallback_cutoff = (
                datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=7)
            ).isoformat()
            rows = db.execute(
                text(query + " AND target_date >= :cutoff ORDER BY target_date DESC, computed_at DESC"),
                {**params, "cutoff": fallback_cutoff},
            ).fetchall()
            cache_status = "stale" if rows else "empty"

        return {
            "cache_status": cache_status,
            "target_date": target_date,
            "league_key": league_key,
            "count": len(rows),
            "valuations": [
                {
                    "player_id": r.player_id,
                    "player_name": r.player_name,
                    "target_date": r.target_date.isoformat() if r.target_date else None,
                    "computed_at": r.computed_at.isoformat() if r.computed_at else None,
                    "report": r.report,
                }
                for r in rows
            ],
        }
    except Exception as exc:
        logger.error("get_player_valuations: DB error (%s)", exc)
        return {
            "cache_status": "error",
            "target_date": target_date,
            "league_key": league_key,
            "count": 0,
            "valuations": [],
            "error": "Cache unavailable -- try again later",
        }


@router.get("/api/fantasy/matchup", response_model=MatchupResponse)
async def get_fantasy_matchup(user: str = Depends(verify_api_key)):
    """Return current week's matchup: opponent name + category-by-category breakdown.

    Live data only (no cache) — 5-second timeout safety valve returns 504 on Yahoo hang.
    """
    import time as _time
    import asyncio

    try:
        client = get_yahoo_client()
    except YahooAuthError as exc:
        raise HTTPException(
            status_code=503,
            detail="Yahoo not configured -- set YAHOO_REFRESH_TOKEN",
        ) from exc

    my_team_key = os.getenv("YAHOO_TEAM_KEY", "")
    if not my_team_key:
        try:
            my_team_key = client.get_my_team_key()
        except Exception:
            my_team_key = ""
    logger.info("Matchup: resolved my_team_key=%s", my_team_key)

    _stub_my = MatchupTeamOut(team_key=my_team_key, team_name="My Team", stats={})
    _stub_opp = MatchupTeamOut(team_key="", team_name="TBD", stats={})

    stat_id_map: dict = dict(_YAHOO_STAT_FALLBACK)
    active_stat_abbrs: set = set()

    # League settings cache (2-hour TTL) — stat ID map never changes mid-season
    _ls_cached = _LEAGUE_SETTINGS_CACHE.get("entry")
    if _ls_cached and (_time.monotonic() - _ls_cached["built_at"]) < 7200:
        stat_id_map = _ls_cached["stat_id_map"]
        active_stat_abbrs = _ls_cached["active_stat_abbrs"]
    else:
        try:
            settings = client.get_league_settings()
            stat_cats = (
                settings
                .get("settings", [{}])[0]
                .get("stat_categories", {})
                .get("stats", [])
            )
            _stat_entries: list = []
            for entry in stat_cats:
                if isinstance(entry, dict):
                    s = entry.get("stat", {})
                    sid = str(s.get("stat_id", ""))
                    abbr = s.get("display_name") or s.get("abbreviation") or s.get("name") or sid
                    pos_type = s.get("position_type", "")
                    is_display = bool(s.get("is_only_display_stat", 0))
                    if sid:
                        _stat_entries.append((sid, abbr, pos_type, is_display))

            _abbr_positions: dict = {}
            for sid, abbr, pos_type, _ in _stat_entries:
                _abbr_positions.setdefault(abbr, set()).add(pos_type)

            _PITCHER_RENAME = {"HR": "HR_P", "K": "K_P"}
            _BATTER_RENAME = {"K": "K_B", "HR": "HR_B"}

            for sid, abbr, pos_type, is_display in _stat_entries:
                final_abbr = abbr
                if len(_abbr_positions.get(abbr, set())) > 1:
                    if pos_type == "P" and abbr in _PITCHER_RENAME:
                        final_abbr = _PITCHER_RENAME[abbr]
                    elif pos_type == "B" and abbr in _BATTER_RENAME:
                        final_abbr = _BATTER_RENAME[abbr]
                stat_id_map[sid] = final_abbr
                if not is_display:
                    active_stat_abbrs.add(final_abbr)

            _LEAGUE_SETTINGS_CACHE["entry"] = {
                "stat_id_map": stat_id_map,
                "active_stat_abbrs": active_stat_abbrs,
                "built_at": _time.monotonic(),
            }
        except Exception as _e:
            logger.warning("get_league_settings failed, using fallback stat_id_map: %s", _e)

    if not active_stat_abbrs and SCORING_CATEGORY_CODES:
        active_stat_abbrs = set(SCORING_CATEGORY_CODES)

    try:
        # Timeout safety valve: 5-second limit prevents hanging on slow Yahoo responses
        matchups = await asyncio.wait_for(
            asyncio.to_thread(client.get_scoreboard),
            timeout=5.0,
        )
    except asyncio.TimeoutError:
        logger.error("Matchup fetch timeout (>5s) — Yahoo API slow or unresponsive")
        raise HTTPException(
            status_code=504,
            detail="Yahoo API timeout — matchup data temporarily unavailable",
        )
    except (YahooAuthError, YahooAPIError) as exc:
        logger.error("Matchup scoreboard fetch failed: %s", exc)
        return MatchupResponse(my_team=_stub_my, opponent=_stub_opp, message="Scoreboard unavailable -- Yahoo API error.")

    if not matchups:
        return MatchupResponse(my_team=_stub_my, opponent=_stub_opp, message="No matchup data yet -- season may be starting.")

    week: Optional[int] = None
    is_playoffs = False

    def _extract_team_stats(team_entry) -> tuple:
        t_meta: dict = {}
        stats_raw: list = []

        def flatten_entry(entry, depth=0):
            nonlocal stats_raw
            if depth > 5:
                return
            if isinstance(entry, list):
                for item in entry:
                    flatten_entry(item, depth + 1)
            elif isinstance(entry, dict):
                if "team_stats" in entry and not stats_raw:
                    inner = entry["team_stats"].get("stats", [])
                    if isinstance(inner, list):
                        stats_raw = inner
                for key in ["team_key", "name", "team_id", "nickname"]:
                    if key in entry:
                        t_meta[key] = entry[key]

        flatten_entry(team_entry)

        stats_dict: dict = {}
        for s in stats_raw:
            if isinstance(s, dict):
                stat = s.get("stat", {})
                if isinstance(stat, dict):
                    sid = str(stat.get("stat_id", ""))
                    key = stat_id_map.get(sid, sid)
                    val = stat.get("value", "")
                    if active_stat_abbrs and key not in active_stat_abbrs:
                        continue
                    _NON_NEGATIVE_STATS = frozenset({
                        "GS", "W", "SV", "K", "HR", "R", "RBI", "H",
                        "IP", "HLD", "QS", "BB", "NSV", "62",
                    })
                    if key in _NON_NEGATIVE_STATS:
                        try:
                            if float(val) < 0:
                                val = "0"
                        except (TypeError, ValueError):
                            pass
                    if key == "BB":
                        try:
                            fval = float(val)
                            if fval != int(fval):
                                continue
                        except (TypeError, ValueError):
                            pass
                    if key:
                        # NSB (and any fraction stat like "X/Y") — Yahoo returns
                        # "successful_steals/attempts". Use only the numerator.
                        if isinstance(val, str) and "/" in val:
                            try:
                                val = str(int(val.split("/")[0]))
                            except (ValueError, IndexError):
                                pass
                        stats_dict[key] = val

        team_key = t_meta.get("team_key", "")
        team_name = t_meta.get("name", "") or t_meta.get("nickname", "")
        return (team_key, team_name, stats_dict)

    for m in matchups:
        if not isinstance(m, dict):
            continue
        w = m.get("week")
        if w:
            try:
                week = int(w)
            except (TypeError, ValueError):
                pass
        raw_playoffs = bool(m.get("is_playoffs", 0))
        is_playoffs = raw_playoffs and (week is not None and week >= 20)

        teams = m.get("teams") or m.get("0", {}).get("teams", {})
        team_data: list = []

        if isinstance(teams, list):
            for item in teams:
                if isinstance(item, dict):
                    if "team" in item:
                        team_data.append(_extract_team_stats(item["team"]))
                    else:
                        team_data.append(_extract_team_stats(item))
        elif isinstance(teams, dict):
            count_t = int(teams.get("count", 0))
            for ti in range(count_t):
                entry = teams.get(str(ti), {})
                if isinstance(entry, dict):
                    team_entry = entry.get("team", entry)
                    team_data.append(_extract_team_stats(team_entry))

        if not team_data and "team" in m:
            direct_teams = m.get("team", [])
            if isinstance(direct_teams, list):
                for t in direct_teams:
                    team_data.append(_extract_team_stats(t))
            elif isinstance(direct_teams, dict):
                team_data.append(_extract_team_stats(direct_teams))

        my_entry = None
        for t in team_data:
            # CRITICAL: Use exact match ONLY. Substring matching causes false positives
            # when team IDs are numeric (e.g., "t.7" matches "t.71", "t.72", etc.)
            if t[0] == my_team_key:
                my_entry = t
                break

        if my_entry is None:
            continue

        opp_entry = next((t for t in team_data if t[0] != my_entry[0]), None)
        if opp_entry is None:
            opp_entry = ("", "Unknown", {})

        def _filter_stats(stats: dict) -> dict:
            if active_stat_abbrs:
                _ALWAYS_KEEP = {"IP", "GS", "H/AB", "21", "50", "62"}
                return {
                    k: v for k, v in stats.items()
                    if k in active_stat_abbrs or k in _ALWAYS_KEEP
                }
            return {k: v for k, v in stats.items() if v not in ("", "-", None)}

        my_stats = _filter_stats(my_entry[2])
        opp_stats = _filter_stats(opp_entry[2])

        _result = MatchupResponse(
            week=week,
            my_team=MatchupTeamOut(
                team_key=my_entry[0],
                team_name=my_entry[1],
                stats=my_stats,
            ),
            opponent=MatchupTeamOut(
                team_key=opp_entry[0],
                team_name=opp_entry[1],
                stats=opp_stats,
            ),
            is_playoffs=is_playoffs,
        )
        return _result

    return MatchupResponse(week=week, my_team=_stub_my, opponent=_stub_opp, message="Your team was not found in the current week's matchup.")


# ============================================================================
# DASHBOARD
# ============================================================================

from backend.services.dashboard_service import get_dashboard_service


@router.get("/api/dashboard")
async def get_dashboard(
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Phase B: Enhanced Dashboard"""
    from backend.models import PlayerScore, MLBGameLog
    service = get_dashboard_service()
    dashboard = await service.get_dashboard(user_id=user)

    # --- Data freshness indicator ---
    now_et = datetime.now(ZoneInfo("America/New_York"))
    last_sync_dt: Optional[datetime] = None
    try:
        row = (
            db.query(func.max(PlayerScore.computed_at))
            .scalar()
        )
        if row is not None:
            last_sync_dt = row if row.tzinfo else row.replace(tzinfo=ZoneInfo("America/New_York"))
    except Exception as _fs_err:
        logger.debug("dashboard: last_sync query failed (non-fatal): %s", _fs_err)

    is_stale = False
    games_today = False
    stale_warning: Optional[dict] = None
    if last_sync_dt is not None:
        age_seconds = (now_et - last_sync_dt).total_seconds()
        is_stale = age_seconds > 7200  # >2 hours
        if is_stale:
            try:
                today_et = now_et.date()
                games_today = (
                    db.query(MLBGameLog)
                    .filter(MLBGameLog.game_date == today_et)
                    .count()
                ) > 0
            except Exception:
                games_today = True  # conservative: assume games if we can't check
            if games_today:
                stale_warning = {
                    "stale": True,
                    "last_sync": last_sync_dt.isoformat(),
                    "recommendation": "Lineup data may not reflect today's starting lineups",
                }

    return {
        "success": True,
        "timestamp": dashboard.timestamp,
        "data": {
            "lineup_gaps": [asdict(g) for g in dashboard.lineup_gaps],
            "lineup_filled_count": dashboard.lineup_filled_count,
            "lineup_total_count": dashboard.lineup_total_count,
            "hot_streaks": [asdict(s) for s in dashboard.hot_streaks],
            "cold_streaks": [asdict(s) for s in dashboard.cold_streaks],
            "waiver_targets": [asdict(t) for t in dashboard.waiver_targets],
            "injury_flags": [asdict(i) for i in dashboard.injury_flags],
            "healthy_count": dashboard.healthy_count,
            "injured_count": dashboard.injured_count,
            "matchup_preview": asdict(dashboard.matchup_preview) if dashboard.matchup_preview else None,
            "probable_pitchers": [asdict(p) for p in dashboard.probable_pitchers],
            "two_start_pitchers": [asdict(p) for p in dashboard.two_start_pitchers],
        },
        "last_sync": last_sync_dt.isoformat() if last_sync_dt else None,
        "stale_warning": stale_warning,
        "preferences": dashboard.preferences,
    }


@router.get("/api/user/preferences")
async def get_user_preferences(user: str = Depends(verify_api_key)):
    """Get current user preferences."""
    service = get_dashboard_service()
    prefs = service.get_preferences(user_id=user)
    return {"success": True, "preferences": prefs}


@router.post("/api/user/preferences")
async def update_user_preferences(
    updates: dict,
    user: str = Depends(verify_api_key),
):
    """Update user preferences."""
    service = get_dashboard_service()
    updated = service.update_preferences(user_id=user, updates=updates)
    return {"success": True, "preferences": updated}


@router.get("/api/dashboard/streaks")
async def get_dashboard_streaks(user: str = Depends(verify_api_key)):
    """Get hot/cold streaks for rostered players."""
    service = get_dashboard_service()
    hot, cold = await service._get_streaks(user_id=user)
    return {
        "success": True,
        "hot_streaks": [asdict(s) for s in hot[:5]],
        "cold_streaks": [asdict(s) for s in cold[:5]],
    }


@router.get("/api/dashboard/waiver-targets")
async def get_dashboard_waiver_targets(user: str = Depends(verify_api_key)):
    """Get prioritized waiver wire targets."""
    service = get_dashboard_service()
    db = SessionLocal()
    try:
        prefs = service._get_or_create_preferences(db, user)
        targets = await service._get_waiver_targets(user_id=user, prefs=prefs)
        return {
            "success": True,
            "targets": [asdict(t) for t in targets[:10]],
        }
    finally:
        db.close()


# ============================================================================
# SSE DASHBOARD STREAM
# ============================================================================

_ALL_PANELS = frozenset({"waiver_targets", "matchup_preview", "probable_pitchers", "streaks"})


@router.get("/api/fantasy/dashboard/stream")
async def dashboard_stream(
    panels: Optional[str] = Query(
        default=None,
        description="Comma-separated panel names to stream. "
        "Defaults to all panels: waiver_targets,matchup_preview,probable_pitchers,streaks",
    ),
    user: str = Depends(verify_api_key),
):
    """Server-Sent Events stream for the fantasy dashboard."""
    requested: frozenset
    if panels:
        requested = frozenset(p.strip() for p in panels.split(",") if p.strip()) & _ALL_PANELS
    else:
        requested = _ALL_PANELS

    service = get_dashboard_service()

    def _fmt_event(name: str, payload: dict) -> str:
        return f"event: {name}\ndata: {_json.dumps(payload)}\n\n"

    def _fmt_error(panel: str, msg: str) -> str:
        return f"event: error\ndata: {_json.dumps({'panel': panel, 'error': msg})}\n\n"

    async def generate():
        while True:
            if "waiver_targets" in requested:
                try:
                    _db = SessionLocal()
                    try:
                        prefs = service._get_or_create_preferences(_db, user)
                        targets = await service._get_waiver_targets(user_id=user, prefs=prefs)
                    finally:
                        _db.close()
                    yield _fmt_event("waiver_targets", {
                        "targets": [asdict(t) for t in targets[:10]],
                    })
                except Exception as exc:
                    logger.warning("SSE waiver_targets error: %s", exc)
                    yield _fmt_error("waiver_targets", str(exc))

            if "matchup_preview" in requested:
                try:
                    matchup = await service._get_matchup_preview(user_id=user, team_key=None)
                    yield _fmt_event("matchup_preview", {
                        "matchup": asdict(matchup) if matchup else None,
                    })
                except Exception as exc:
                    logger.warning("SSE matchup_preview error: %s", exc)
                    yield _fmt_error("matchup_preview", str(exc))

            if "probable_pitchers" in requested:
                try:
                    pitchers, two_starts = await service._get_probable_pitchers(user_id=user)
                    yield _fmt_event("probable_pitchers", {
                        "pitchers": [asdict(p) for p in pitchers],
                        "two_start_pitchers": [asdict(p) for p in two_starts],
                    })
                except Exception as exc:
                    logger.warning("SSE probable_pitchers error: %s", exc)
                    yield _fmt_error("probable_pitchers", str(exc))

            if "streaks" in requested:
                try:
                    hot, cold = await service._get_streaks(user_id=user)
                    yield _fmt_event("streaks", {
                        "hot": [asdict(s) for s in hot[:5]],
                        "cold": [asdict(s) for s in cold[:5]],
                    })
                except Exception as exc:
                    logger.warning("SSE streaks error: %s", exc)
                    yield _fmt_error("streaks", str(exc))

            yield ": keep-alive\n\n"
            await asyncio.sleep(60)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# ELITE LINEUP OPTIMIZER
# ============================================================================

from backend.fantasy_baseball.elite_lineup_scorer import (
    EliteScore,
    get_elite_scorer,
    BatterProfile,
    PitcherProfile,
    GameContext,
)
from backend.fantasy_baseball.lineup_constraint_solver import (
    get_lineup_solver,
    PositionSlot,
)


@router.post("/api/fantasy/lineup/async-optimize")
async def async_optimize_lineup(
    target_date: str,
    risk_tolerance: str = "balanced",
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """Submit lineup optimization as an async job. Returns immediately with job_id."""
    job_id = jq_submit(
        db,
        job_type="lineup_optimization",
        payload={"target_date": target_date, "risk_tolerance": risk_tolerance},
        priority=3,
    )
    return {"job_id": job_id, "status": "queued", "poll_url": f"/api/fantasy/jobs/{job_id}"}


@router.get("/api/fantasy/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """Poll async job status. Returns status + result when complete."""
    result = jq_status(db, job_id)
    if result.get("error") == "not_found":
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return result


@router.get("/api/fantasy/yahoo-health")
async def yahoo_health():
    """
    Health check for Yahoo Fantasy API connectivity.

    Returns structured status for frontend diagnostics and proactive UI disabling.
    Frontend can poll this to disable the 'Optimize Lineup' button when Yahoo is unavailable.
    """
    from backend.fantasy_baseball.yahoo_client_resilient import _client, _client_lock
    from datetime import datetime
    from zoneinfo import ZoneInfo

    health_status = {
        "status": "unknown",  # healthy | degraded | down
        "circuit_state": None,  # closed | open | half_open
        "last_success_at": None,
        "error": None,
        "recovery_hint": None,
    }

    # Check client singleton status
    if _client is None:
        health_status.update({
            "status": "down",
            "error": "Yahoo client not initialized",
            "recovery_hint": "Check YAHOO_CLIENT_ID, YAHOO_CLIENT_SECRET, YAHOO_REFRESH_TOKEN environment variables"
        })
        return health_status

    # Check circuit breaker
    try:
        if hasattr(_client, 'circuit'):
            cb_stats = _client.circuit.get_stats()
            health_status["circuit_state"] = cb_stats["state"]
            if cb_stats["state"] == "open":
                health_status["status"] = "degraded"
                health_status["error"] = "Circuit breaker is OPEN after repeated failures"
                health_status["recovery_hint"] = "Wait 5 minutes for circuit recovery or investigate API failures"

        # Try a lightweight API call (get league metadata)
        _client.get_league()
        health_status["status"] = "healthy"
        health_status["last_success_at"] = datetime.now(ZoneInfo("America/New_York")).isoformat()

    except YahooAuthError as exc:
        health_status["status"] = "down"
        health_status["error"] = f"Authentication failed: {str(exc)}"
        health_status["recovery_hint"] = "Re-run OAuth flow: python -m backend.fantasy_baseball.yahoo_client_resilient --auth"
    except Exception as exc:
        health_status["status"] = "degraded"
        health_status["error"] = f"API check failed: {str(exc)}"

    return health_status


@router.get("/api/fantasy/global-freshness")
async def global_freshness():
    """
    Global data freshness for all fantasy data sources.

    Returns aggregated freshness info for frontend health indicators.
    Severity mapping: "fresh" (< 5 min) → "warning" (5-60 min) → "critical" (> 60 min).
    Never returns "unknown" as terminal state — uses "warning" with message instead.
    """
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("America/New_York"))
    sources = []

    # Check Yahoo Fantasy freshness (if client initialized)
    try:
        from backend.fantasy_baseball.yahoo_client_resilient import _client
        if _client is not None:
            # Check circuit breaker last success time if available
            last_yahoo_at = None
            try:
                if hasattr(_client, 'circuit'):
                    stats = _client.circuit.get_stats()
                    last_yahoo_at = stats.get('last_success_time')
            except Exception:
                # Circuit breaker stats unavailable — continue with last_yahoo_at = None
                pass

            yahoo_minutes_ago = None
            yahoo_severity = "warning"  # Default to warning (STALE) when no timestamp
            yahoo_message = "Yahoo auth required — no recent fetch"
            if last_yahoo_at:
                delta = now - last_yahoo_at
                yahoo_minutes_ago = int(delta.total_seconds() / 60)
                # Use 5-minute threshold for LIVE, 60-minute for STALE, >60 for OFFLINE
                yahoo_severity = "fresh" if yahoo_minutes_ago < 5 else "warning" if yahoo_minutes_ago < 60 else "critical"
                yahoo_message = f"Last successful call {yahoo_minutes_ago} minutes ago"

            sources.append({
                "name": "yahoo",
                "severity": yahoo_severity,
                "minutes_ago": yahoo_minutes_ago,
                "message": yahoo_message
            })
        else:
            # Client not initialized — return STALE, not unknown
            sources.append({
                "name": "yahoo",
                "severity": "warning",
                "minutes_ago": None,
                "message": "Yahoo client not initialized — auth required"
            })
    except ImportError:
        # Import failed — client module unavailable, treat as not initialized (STALE)
        sources.append({
            "name": "yahoo",
            "severity": "warning",
            "minutes_ago": None,
            "message": "Yahoo client not available — auth required"
        })
    except Exception as e:
        # Actual error — return critical (OFFLINE)
        sources.append({
            "name": "yahoo",
            "severity": "critical",
            "minutes_ago": None,
            "message": f"Failed to check Yahoo: {str(e)}"
        })

    # Aggregate severity (worst source determines overall)
    severity_order = {"critical": 3, "warning": 2, "fresh": 1}
    worst_severity = max(
        (severity_order.get(s.get("severity", "warning"), 1) for s in sources),
        default=1
    )
    severity_map = {3: "critical", 2: "warning", 1: "fresh"}
    overall_severity = severity_map.get(worst_severity, "warning")

    # Compute minutes_ago as the worst (max) among sources
    minutes_ago = None
    for s in sources:
        if s.get("minutes_ago") is not None:
            if minutes_ago is None or s["minutes_ago"] > minutes_ago:
                minutes_ago = s["minutes_ago"]

    warning_text = None
    if overall_severity == "critical":
        warning_text = "One or more data sources are critically stale or unavailable"
    elif overall_severity == "warning":
        warning_text = "Some data sources are stale but still usable"

    return {
        "severity": overall_severity,
        "minutes_ago": minutes_ago,
        "warning_text": warning_text,
        "sources": sources
    }


@router.get("/api/fantasy/streaming/recommendations")
async def streaming_recommendations(
    target_date: str = Query(..., description="Target date (YYYY-MM-DD) to analyze for streaming recommendations"),
    days_ahead: int = Query(7, description="Number of days ahead to analyze for 2-start pitchers"),
    db: Session = Depends(get_db),
):
    """
    Streaming recommendations for fantasy baseball.

    Returns 2-start starting pitchers in the scoring period with quality ratings.
    Uses ProbablePitcherSnapshot table (populated daily by ingestion job).

    Response includes:
    - Two-start pitchers with matchup details
    - Quality scores (ERA + park factor)
    - EXCELLENT/GOOD/AVERAGE/AVOID recommendations
    - Transparency fields for each recommendation
    """
    from zoneinfo import ZoneInfo
    from datetime import datetime, timedelta
    from backend.models import ProbablePitcherSnapshot

    # Parse target date
    try:
        target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    now_et = datetime.now(ZoneInfo("America/New_York"))

    # Query probable pitchers for the window
    end_dt = target_dt + timedelta(days=days_ahead)

    query = db.query(
        ProbablePitcherSnapshot.bdl_player_id,
        ProbablePitcherSnapshot.pitcher_name,
        ProbablePitcherSnapshot.team,
        ProbablePitcherSnapshot.handedness,
        ProbablePitcherSnapshot.game_date,
        ProbablePitcherSnapshot.opponent,
        ProbablePitcherSnapshot.is_home,
        ProbablePitcherSnapshot.quality_score,
        ProbablePitcherSnapshot.is_confirmed,
        ProbablePitcherSnapshot.game_time_et,
    ).filter(
        ProbablePitcherSnapshot.game_date >= target_dt,
        ProbablePitcherSnapshot.game_date <= end_dt,
        ProbablePitcherSnapshot.bdl_player_id.isnot(None),
        ProbablePitcherSnapshot.quality_score.isnot(None),
    ).order_by(ProbablePitcherSnapshot.game_date, ProbablePitcherSnapshot.team)

    rows = query.all()

    # Group by pitcher
    pitcher_starts: dict[int, list] = {}
    for r in rows:
        pid = r.bdl_player_id
        if pid not in pitcher_starts:
            pitcher_starts[pid] = []
        pitcher_starts[pid].append({
            "pitcher_name": r.pitcher_name,
            "team": r.team,
            "handedness": r.handedness,
            "date": r.game_date.isoformat(),
            "opponent": r.opponent,
            "is_home": r.is_home,
            "quality_score": round(float(r.quality_score or 0), 2),
            "is_confirmed": r.is_confirmed,
            "game_time_et": r.game_time_et,
        })

    # Identify 2-start pitchers
    two_starters = []
    for pid, starts in pitcher_starts.items():
        if len(starts) >= 2:
            # Calculate overall quality (average of quality scores)
            avg_quality = sum(s["quality_score"] for s in starts[:2]) / min(len(starts), 2)

            # Determine confidence level
            confirmed_count = sum(1 for s in starts[:2] if s.get("is_confirmed"))
            if confirmed_count == 2:
                confidence = "HIGH"
            elif confirmed_count == 1:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            # Determine recommendation tier with confidence-weighted matrix
            if avg_quality >= 1.0 and confidence == "HIGH":
                recommendation = "EXCELLENT"
            elif avg_quality >= 0.3 and confidence in ("HIGH", "MEDIUM"):
                recommendation = "GOOD"
            elif confidence == "LOW":
                recommendation = "AVOID"
            elif avg_quality >= -0.3:
                recommendation = "AVERAGE"
            else:  # avg_quality < -0.3
                recommendation = "AVOID"

            # Build transparency factors
            factors = []
            factors.append(f"starts_count: {len(starts)}")
            factors.append(f"avg_quality: {avg_quality:.2f}")
            if confirmed_count > 0:
                factors.append(f"confirmed_starts: {confirmed_count}")

            # Build risk note based on confirmation status
            if confirmed_count == 2:
                risk_note = "Both starts confirmed — safe stream"
            elif confirmed_count == 1:
                risk_note = "One start projected — monitor for scratches"
            else:
                risk_note = "Both starts projected — high variance, have backup ready"

            two_starters.append({
                "bdl_player_id": pid,
                "name": starts[0].get("pitcher_name", "Unknown"),
                "team": starts[0].get("team", ""),
                "handedness": starts[0].get("handedness", ""),
                "starts": starts[:2],  # First 2 starts for the period
                "overall_quality": round(avg_quality, 2),
                "recommendation": recommendation,
                "risk_note": risk_note,
                "transparency": {
                    "quality_score": round(avg_quality, 2),
                    "factors": factors,
                    "confidence": confidence,
                }
            })

    # Sort by overall_quality descending
    two_starters.sort(key=lambda x: x["overall_quality"], reverse=True)

    # Get data freshness
    freshness_query = db.query(
        func.max(ProbablePitcherSnapshot.fetched_at)
    ).filter(
        ProbablePitcherSnapshot.game_date >= target_dt,
        ProbablePitcherSnapshot.game_date <= end_dt,
    )

    last_refresh_at = freshness_query.scalar()
    staleness_ms = None
    if last_refresh_at:
        staleness_ms = int((now_et - last_refresh_at.replace(tzinfo=ZoneInfo("America/New_York"))).total_seconds() * 1000)

    return {
        "target_date": target_date,
        "analysis_window_days": days_ahead,
        "two_start_pitchers": two_starters,
        "freshness": {
            "last_refresh_at": last_refresh_at.isoformat() if last_refresh_at else None,
            "staleness_ms": staleness_ms,
            "query_time_et": now_et.isoformat(),
        },
        "data_sources": ["ProbablePitcherSnapshot", "StatcastPerformances (quality_score)"]
    }


# ---------------------------------------------------------------------------
# Auto-Stream Configuration
# ---------------------------------------------------------------------------

@router.post("/api/fantasy/auto-stream/configure")
async def configure_auto_stream(
    request: AutoStreamConfigureRequest,
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """
    Configure Auto-Stream settings for the authenticated user.

    Auto-Stream runs daily at 6 AM ET to execute streaming recommendations
    based on user configuration.

    Args:
        request: Configuration with enabled flag, drop priority, thresholds
        db: Database session
        user: Authenticated user ID

    Returns:
        Updated AutoStreamConfig
    """
    service = get_auto_stream_service()
    # Use league ID from env var (same as Auto-Stream job), not API user
    league_user_id = os.getenv("YAHOO_LEAGUE_ID", "default")

    try:
        config = await service.update_config(
            user_id=league_user_id,
            enabled=request.enabled,
            drop_priority=request.drop_priority,
            min_confidence=request.min_confidence,
            min_recommendation=request.min_recommendation,
            max_adds_per_week=request.max_adds_per_week,
            db=db,
        )

        return {
            "success": True,
            "config": {
                "enabled": config.enabled,
                "drop_priority": config.drop_priority,
                "min_confidence": config.min_confidence,
                "min_recommendation": config.min_recommendation,
                "max_adds_per_week": config.max_adds_per_week,
                "executed_this_week": config.executed_this_week,
                "last_run_at": config.last_run_at,
                "next_run_at": config.next_run_at,
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to configure Auto-Stream: {e}")
        raise HTTPException(status_code=500, detail="Failed to configure Auto-Stream")


@router.get("/api/fantasy/auto-stream/status")
async def get_auto_stream_status(
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """
    Get current Auto-Stream status for the authenticated user.

    Returns configuration, recent activity, and pending actions.

    Args:
        db: Database session
        user: Authenticated user ID

    Returns:
        AutoStreamStatusResponse with current state
    """
    service = get_auto_stream_service()
    # Use league ID from env var (same as Auto-Stream job), not API user
    league_user_id = os.getenv("YAHOO_LEAGUE_ID", "default")

    try:
        status = await service.get_status(user_id=league_user_id, db=db)

        return {
            "enabled": status.enabled,
            "config": {
                "enabled": status.config.enabled,
                "drop_priority": status.config.drop_priority,
                "min_confidence": status.config.min_confidence,
                "min_recommendation": status.config.min_recommendation,
                "max_adds_per_week": status.config.max_adds_per_week,
                "executed_this_week": status.config.executed_this_week,
                "last_run_at": status.config.last_run_at,
                "next_run_at": status.config.next_run_at,
            },
            "last_run_at": status.last_run_at,
            "next_run_at": status.next_run_at,
            "executed_this_week": status.executed_this_week,
            "pending_actions": status.pending_actions,
            "recent_log": status.recent_log,
        }
    except Exception as e:
        logger.error(f"Failed to get Auto-Stream status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get Auto-Stream status")


@router.get("/api/fantasy/lineup/elite-optimize/{lineup_date}")
async def elite_optimize_lineup(
    lineup_date: str,
    use_ortools: bool = True,
    user: str = Depends(verify_api_key),
):
    """ELITE lineup optimizer with multi-factor scoring."""
    from backend.fantasy_baseball.daily_lineup_optimizer import DailyLineupOptimizer

    try:
        yahoo = get_yahoo_client()
    except YahooAuthError:
        raise HTTPException(status_code=503, detail="Yahoo not configured")

    try:
        roster = yahoo.get_roster()
    except Exception as e:
        logger.error("Failed to fetch roster: %s", e)
        raise HTTPException(status_code=500, detail="Failed to fetch roster")

    optimizer = DailyLineupOptimizer()
    games = optimizer.fetch_mlb_odds(lineup_date)
    team_odds = optimizer._build_team_odds_map(games)

    probable_pitchers = optimizer._fetch_probable_pitchers_for_date(lineup_date)

    elite_scorer = get_elite_scorer()
    solver = get_lineup_solver()

    player_scores = {}
    eligibility = {}

    for player in roster:
        pid = player.get("player_id", "")
        name = player.get("name", "")
        team = player.get("team", "")
        positions = player.get("positions", [])

        if any(p in ("SP", "RP", "P") for p in positions):
            continue

        eligibility[pid] = positions

        batter = BatterProfile(
            player_id=pid,
            name=name,
            team=team,
            positions=positions,
            season_woba=0.320,
            woba_vs_lhp=0.320,
            woba_vs_rhp=0.320,
        )

        odds = team_odds.get(team, {})
        context = GameContext(
            implied_runs=odds.get("implied_runs", 4.5),
            park_factor=odds.get("park_factor", 1.0),
            is_home=odds.get("is_home", False),
        )

        opp_team = odds.get("opponent", "")
        pitcher_name = probable_pitchers.get(opp_team.lower(), "")
        pitcher = PitcherProfile(
            name=pitcher_name or "Unknown",
            team=opp_team,
            handedness="R",
            xera=4.25,
        )

        score = elite_scorer.calculate_batter_score(batter, pitcher, context)
        player_scores[pid] = score

    lineup = solver.solve(
        players=roster,
        player_scores=player_scores,
        eligibility=eligibility,
    )

    return {
        "success": True,
        "lineup_date": lineup_date,
        "solver_type": lineup.solver_type,
        "is_optimal": lineup.is_optimal,
        "total_score": lineup.total_score,
        "lineup": [
            {
                "slot": a.slot.value,
                "player_id": a.player_id,
                "player_name": a.player_name,
                "score": a.score,
                "eligibility": a.eligibility,
                "reason": a.reason,
            }
            for a in lineup.assignments
        ],
        "bench": lineup.unassigned_players,
    }


@router.post("/api/fantasy/lineup/analyze-scarcity")
async def analyze_lineup_scarcity(user: str = Depends(verify_api_key)):
    """Analyze roster scarcity by position."""
    from backend.fantasy_baseball.lineup_constraint_solver import get_lineup_solver

    try:
        yahoo = get_yahoo_client()
    except YahooAuthError:
        raise HTTPException(status_code=503, detail="Yahoo not configured")

    roster = yahoo.get_roster()
    eligibility = {p.get("player_id"): p.get("positions", []) for p in roster}

    solver = get_lineup_solver()
    analysis = solver.analyze_scarcity(roster, eligibility)

    return {
        "success": True,
        "analysis": analysis,
    }


@router.post("/api/fantasy/lineup/compare-scoring")
async def compare_scoring_methods(
    player_name: str,
    opponent_pitcher: str = "",
    user: str = Depends(verify_api_key),
):
    """Compare elite multi-factor scoring to simple implied-runs scoring."""
    from backend.fantasy_baseball.elite_lineup_scorer import (
        get_elite_scorer,
        BatterProfile,
        PitcherProfile,
        GameContext,
    )

    scorer = get_elite_scorer()

    batter = BatterProfile(
        player_id="test",
        name=player_name,
        team="NYY",
        positions=["1B"],
        season_woba=0.350,
        woba_vs_lhp=0.320,
        woba_vs_rhp=0.360,
        xwoba=0.355,
    )

    pitcher = PitcherProfile(
        name=opponent_pitcher or "Average Pitcher",
        team="BOS",
        handedness="R",
        xera=4.25 if not opponent_pitcher else 5.50,
    )

    context = GameContext(
        implied_runs=4.5,
        park_factor=1.0,
        is_home=True,
    )

    comparison = scorer.compare_to_simple_score(batter, pitcher, context)

    return {
        "success": True,
        "comparison": comparison,
    }


# ============================================================================
# LINEUP APPLY
# ============================================================================

@router.put("/api/fantasy/lineup/apply")
async def apply_fantasy_lineup(
    payload: LineupApplyRequest,
    user: str = Depends(verify_api_key),
    auto_correct: bool = True,
):
    """Push a lineup to Yahoo Fantasy with game-aware validation."""
    from datetime import datetime as _dt

    try:
        client = get_resilient_yahoo_client()
    except YahooAuthError as exc:
        raise HTTPException(
            status_code=503,
            detail="Yahoo not configured -- set YAHOO_REFRESH_TOKEN",
        ) from exc

    apply_date = payload.date or _dt.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    team_key = os.getenv("YAHOO_TEAM_KEY", "469.l.72586.t.7")

    roster_lookup_by_key: dict = {}
    roster_lookup_by_name: dict = {}
    try:
        roster_players = client.get_roster(team_key=team_key)
        for rp in roster_players:
            rp_key = str(rp.get("player_key") or "").strip()
            rp_name = str(rp.get("name") or "").strip().lower()
            if rp_key:
                roster_lookup_by_key[rp_key] = rp
            if rp_name:
                roster_lookup_by_name[rp_name] = rp
    except Exception as _exc:
        logger.warning("Could not prefetch Yahoo roster for lineup apply sanitization: %s", _exc)

    apply_warnings: list = []
    try:
        mlb_games = get_lineup_optimizer().fetch_mlb_odds(apply_date)
        if not mlb_games:
            apply_warnings.append(
                f"No MLB games scheduled for {apply_date} -- applying lineup in preseason mode."
            )
    except Exception as _exc:
        logger.warning("Could not pre-check MLB schedule for %s: %s", apply_date, _exc)

    def _is_valid_yahoo_key(key: str) -> bool:
        if not key:
            return False
        parts = key.split(".")
        return len(parts) == 3 and parts[1] == "p" and parts[2].isdigit()

    invalid_players: list = []
    sanitized_players: list = []
    for p in payload.players:
        raw_identifier = (p.player_key or "").strip()
        requested_position = (p.position or "").strip()

        rp = roster_lookup_by_key.get(raw_identifier)
        if rp is None and raw_identifier:
            rp = roster_lookup_by_name.get(raw_identifier.lower())

        resolved_player_key = raw_identifier if _is_valid_yahoo_key(raw_identifier) else ""
        if rp:
            resolved_player_key = str(rp.get("player_key") or resolved_player_key or "").strip()

        if not _is_valid_yahoo_key(resolved_player_key):
            invalid_players.append(raw_identifier or "<missing>")
            continue

        eligible_positions = [str(x).strip() for x in (rp.get("positions") or [])] if rp else []
        if not eligible_positions:
            eligible_positions = [requested_position] if requested_position else []

        resolved_position = requested_position
        if resolved_position == "OF" and "OF" not in eligible_positions:
            for of_pos in ("LF", "CF", "RF"):
                if of_pos in eligible_positions:
                    resolved_position = of_pos
                    break
        if not resolved_position:
            resolved_position = eligible_positions[0] if eligible_positions else "BN"

        sanitized_players.append(
            {
                "id": resolved_player_key,
                "player_key": resolved_player_key,
                "position": resolved_position,
                "positions": eligible_positions,
                "name": (rp.get("name") if rp else None),
            }
        )

    if invalid_players:
        raise HTTPException(
            status_code=422,
            detail={
                "success": False,
                "error": (
                    "Invalid player identifiers in lineup payload. Expected Yahoo player keys "
                    "formatted as mlb.p.XXXXX or 469.p.XXXXX, or exact roster names."
                ),
                "invalid_players": invalid_players,
            },
        )

    optimized_lineup = {"starters": sanitized_players}

    try:
        result = await client.set_lineup_resilient(
            team_id=team_key,
            optimized_lineup=optimized_lineup,
            auto_correct=auto_correct,
        )
    except Exception as exc:
        logger.exception("Error applying lineup with ResilientYahooClient")
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not result.success:
        detail = {
            "success": False,
            "error": result.errors,
            "warnings": result.warnings,
            "suggested_action": result.suggested_action,
            "retry_possible": result.retry_possible,
        }
        raise HTTPException(status_code=422, detail=detail)

    applied_count = len(result.changes) if result.changes else len(payload.players)

    return {
        "success": True,
        "applied": applied_count,
        "skipped": 0,
        "date": apply_date,
        "warnings": apply_warnings + result.warnings,
        "changes": result.changes,
        "auto_correct": auto_correct,
    }


# ============================================================================
# MATCHUP SIMULATE
# ============================================================================

# cat_scores keys stored in player_projections → v2 lowercase mcmc keys
_PROJ_TO_SIM_KEY: dict[str, str] = {
    "r": "r", "h": "h", "hr": "hr_b", "rbi": "rbi",
    "k_bat": "k_b", "tb": "tb", "avg": "avg", "ops": "ops", "nsb": "nsb",
    "w": "w", "l": "l", "hr_pit": "hr_p", "k_pit": "k_p",
    "era": "era", "whip": "whip", "k9": "k_9", "qs": "qs", "nsv": "nsv",
}


def _fetch_rosters_for_simulate(db: Session) -> tuple[list, list]:
    """Build my_roster and opponent_roster dicts for simulate_weekly_matchup.

    Fetches both Yahoo rosters from the scoreboard, then enriches each player
    with cat_scores from the player_projections table. Falls back to empty
    cat_scores when a player has no projection row.
    """
    try:
        client = get_yahoo_client()
    except YahooAuthError as exc:
        raise HTTPException(status_code=503, detail="Yahoo not configured") from exc

    my_team_key = os.getenv("YAHOO_TEAM_KEY", "") or client.get_my_team_key()

    # Discover opponent team key from the scoreboard
    opponent_team_key: Optional[str] = None
    try:
        matchups = client.get_scoreboard()

        def _extract_team_keys(obj, depth=0) -> list[str]:
            """Recursively find all team_key values in a Yahoo response structure."""
            if depth > 5:
                return []
            keys: list[str] = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if (k == "team_key" or k == "winner_team_key") and isinstance(v, str):
                        keys.append(v)
                    else:
                        keys.extend(_extract_team_keys(v, depth + 1))
            elif isinstance(obj, list):
                for item in obj:
                    keys.extend(_extract_team_keys(item, depth + 1))
            return keys

        for m in matchups or []:
            if not isinstance(m, dict):
                continue
            team_keys = _extract_team_keys(m)
            if not team_keys:
                continue
            is_my_matchup = any(
                tk == my_team_key or (tk and my_team_key and (tk in my_team_key or my_team_key in tk))
                for tk in team_keys
            )
            if is_my_matchup:
                opponent_team_key = next(
                    (
                        tk for tk in team_keys
                        if tk != my_team_key and not (tk and my_team_key and my_team_key in tk)
                    ),
                    None,
                )
                logger.info("simulate: found opponent_team_key=%s from scoreboard", opponent_team_key)
                break
    except Exception as _sb_err:
        logger.warning("simulate: scoreboard lookup failed: %s", _sb_err)

    # Build name→PlayerProjection map for cat_scores enrichment
    _proj_by_name: dict[str, PlayerProjection] = {}
    try:
        _proj_rows = db.query(PlayerProjection).filter(
            PlayerProjection.cat_scores.isnot(None)
        ).all()
        for _pr in _proj_rows:
            if _pr.player_name:
                _proj_by_name[_normalize_identity_name(_pr.player_name)] = _pr
    except Exception as _db_err:
        logger.warning("simulate: projection name-map build failed: %s", _db_err)

    def _player_dict(p: dict) -> dict:
        name = p.get("name", "")
        positions = p.get("eligible_positions", []) or p.get("positions", [])
        if isinstance(positions, str):
            positions = [positions]
        cat_scores: dict[str, float] = {}
        _norm_name = _normalize_identity_name(name)
        proj = _proj_by_name.get(_norm_name)
        if proj is None and _proj_by_name:
            _fuzzy = _difflib.get_close_matches(_norm_name, _proj_by_name.keys(), n=1, cutoff=0.85)
            if _fuzzy:
                proj = _proj_by_name[_fuzzy[0]]
        if proj and proj.cat_scores:
            for src, dest in _PROJ_TO_SIM_KEY.items():
                val = proj.cat_scores.get(src)
                if val is not None:
                    cat_scores[dest] = float(val)
        is_pitcher = any(pos in ("SP", "RP", "P") for pos in positions)
        return {
            "name": name,
            "positions": positions,
            "cat_scores": cat_scores,
            "starts_this_week": 1 if is_pitcher else 0,
        }

    # Fetch my roster with detailed error logging
    logger.info("simulate: fetching my roster with team_key=%s", my_team_key)
    my_raw = client.get_roster(team_key=my_team_key)
    logger.info("simulate: get_roster returned %d players", len(my_raw) if my_raw else 0)
    my_roster = [_player_dict(p) for p in my_raw]
    logger.info("simulate: my_roster has %d players after _player_dict conversion", len(my_roster))

    opp_roster: list = []
    if opponent_team_key:
        try:
            logger.info("simulate: fetching opponent roster with team_key=%s", opponent_team_key)
            opp_raw = client.get_roster(team_key=opponent_team_key)
            logger.info("simulate: opponent get_roster returned %d players", len(opp_raw) if opp_raw else 0)
            opp_roster = [_player_dict(p) for p in opp_raw]
            logger.info("simulate: opp_roster has %d players after _player_dict conversion", len(opp_roster))
        except Exception as _opp_err:
            logger.warning("simulate: opponent roster fetch failed: %s", _opp_err)

    return my_roster, opp_roster


@router.post("/api/fantasy/matchup/simulate")
async def simulate_matchup(
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
    payload: Optional[MatchupSimulateRequest] = Body(None),
):
    """Monte Carlo simulation of a weekly H2H matchup.

    When called with no body (or empty rosters), fetches both rosters from
    Yahoo and enriches with cat_scores from player_projections.
    """
    from backend.fantasy_baseball.mcmc_simulator import simulate_weekly_matchup

    if payload is None:
        payload = MatchupSimulateRequest()

    my_roster = payload.my_roster
    opponent_roster = payload.opponent_roster

    if not my_roster or not opponent_roster:
        try:
            my_roster, opponent_roster = _fetch_rosters_for_simulate(db)
        except Exception as _fetch_exc:
            logger.warning("simulate_matchup: roster fetch failed: %s", _fetch_exc)
            raise HTTPException(status_code=503, detail=f"Failed to fetch roster data: {_fetch_exc}")

    if not my_roster or not opponent_roster:
        my_count = len(my_roster) if my_roster else 0
        opp_count = len(opponent_roster) if opponent_roster else 0
        raise HTTPException(
            status_code=422,
            detail=f"Roster data unavailable — Yahoo returned {my_count} my players, {opp_count} opponent players. Check Yahoo API connection and team key configuration.",
        )

    n = min(max(100, payload.n_sims), 5000)
    try:
        # Auto-fetch rosters from Yahoo if not provided and auto_fetch is enabled
        if payload.auto_fetch_rosters and (not my_roster or not opponent_roster):
            fetched_my, fetched_opp = _fetch_rosters_for_simulate(db)
            if not my_roster and fetched_my:
                my_roster = fetched_my
            if not opponent_roster and fetched_opp:
                opponent_roster = fetched_opp
        if not my_roster or not opponent_roster:
            my_count = len(my_roster) if my_roster else 0
            opp_count = len(opponent_roster) if opponent_roster else 0
            raise HTTPException(
                status_code=400,
                detail=f"Roster data required — currently have {my_count} my players, {opp_count} opponent players. Either provide my_roster/opponent_roster or set auto_fetch_rosters=true with Yahoo auth."
            )
        # Fetch current scoreboard stats to anchor simulation to mid-week reality
        _current_my: dict = {}
        _current_opp: dict = {}
        _remaining_frac: float = 1.0
        try:
            _sb_client = get_yahoo_client()
            _my_tk = os.getenv("YAHOO_TEAM_KEY", "") or _sb_client.get_my_team_key()
            _week_stats = _sb_client.get_matchup_stats(my_team_key=_my_tk)
            if _week_stats:
                _raw_my = _week_stats.get("my_stats", {})
                _raw_opp = _week_stats.get("opp_stats", {})
                _SCORE_TO_SIM = {
                    "HR": "hr_b", "R": "r", "RBI": "rbi", "H": "h",
                    "TB": "tb", "K": "k_b", "SB": "nsb", "AVG": "avg", "OPS": "ops",
                    "W": "w", "L": "l", "ERA": "era", "WHIP": "whip",
                    "K9": "k_9", "QS": "qs", "SV": "nsv",
                }
                for _bk, _sk in _SCORE_TO_SIM.items():
                    if _bk in _raw_my:
                        _current_my[_sk] = float(_raw_my[_bk] or 0.0)
                    if _bk in _raw_opp:
                        _current_opp[_sk] = float(_raw_opp[_bk] or 0.0)
            from datetime import datetime
            from zoneinfo import ZoneInfo
            _dow = datetime.now(ZoneInfo("America/New_York")).weekday()  # 0=Mon, 6=Sun
            _remaining_frac = max(0.05, (6 - _dow) / 7.0)
        except Exception as _anchor_err:
            logger.warning("simulate_matchup: scoreboard anchor fetch failed: %s", _anchor_err)

        result = simulate_weekly_matchup(
            my_roster=my_roster,
            opponent_roster=opponent_roster,
            n_sims=n,
            my_current_stats=_current_my or None,
            opp_current_stats=_current_opp or None,
            remaining_fraction=_remaining_frac,
        )
        return result
    except Exception as exc:
        logger.error("simulate_matchup failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Simulation failed: {exc}")


# ============================================================================
# LAYER 3: PLAYER SCORES (P14 League Z-Scores)
# ============================================================================

_ALLOWED_WINDOWS = {7, 14, 30}


@router.get("/api/fantasy/players/{bdl_player_id}/scores", response_model=PlayerScoresResponse)
async def get_player_scores(
    bdl_player_id: int,
    window_days: int = Query(14, description="Rolling window size: 7, 14, or 30"),
    as_of_date: Optional[date] = Query(None, description="Score date (defaults to latest available)"),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """
    Get authoritative Layer 3 scoring output for a player.

    Returns player_scores (league Z-scores) sourced from the P14 scoring pipeline.
    Default window_days=14. If as_of_date is omitted, returns the latest available
    score for the requested player and window.

    Raises 400 if window_days is not 7, 14, or 30.
    Raises 404 if no score exists for the requested player/window/date.
    """
    # Validate window_days
    if window_days not in _ALLOWED_WINDOWS:
        raise HTTPException(
            status_code=400,
            detail=f"window_days must be one of: {', '.join(map(str, sorted(_ALLOWED_WINDOWS)))}",
        )

    def _get_score_value(source, key: str, default=None):
        if source is None:
            return default
        mapping = getattr(source, "_mapping", None)
        if mapping is not None and key in mapping:
            return mapping.get(key, default)
        if isinstance(source, dict):
            return source.get(key, default)
        return getattr(source, key, default)

    def _infer_player_type(source) -> str:
        hitter_fields = ("z_hr", "z_rbi", "z_nsb", "z_avg", "z_obp")
        pitcher_fields = ("z_era", "z_whip", "z_k_per_9")
        has_hitter = any(_get_score_value(source, field) is not None for field in hitter_fields)
        has_pitcher = any(_get_score_value(source, field) is not None for field in pitcher_fields)
        if has_hitter and has_pitcher:
            return "two_way"
        if has_pitcher:
            return "pitcher"
        return "hitter"

    def _normalize_score_date(raw_value):
        if isinstance(raw_value, datetime):
            return raw_value.date()
        if isinstance(raw_value, str):
            return date.fromisoformat(raw_value)
        return raw_value

    def _load_score_via_schema_fallback(target_date: Optional[date]):
        try:
            inspector = inspect(db.bind)
            if "player_scores" not in inspector.get_table_names():
                return None
            available_columns = {
                column["name"] for column in inspector.get_columns("player_scores")
            }
        except Exception:
            return None

        required_columns = {"bdl_player_id", "as_of_date", "window_days"}
        if not required_columns.issubset(available_columns):
            return None

        select_columns = [
            column
            for column in (
                "bdl_player_id",
                "as_of_date",
                "window_days",
                "player_type",
                "games_in_window",
                "composite_z",
                "score_0_100",
                "confidence",
                "z_hr",
                "z_rbi",
                "z_nsb",
                "z_avg",
                "z_obp",
                "z_era",
                "z_whip",
                "z_k_per_9",
            )
            if column in available_columns
        ]
        sql = (
            f"SELECT {', '.join(select_columns)} FROM player_scores "
            "WHERE bdl_player_id = :bdl_player_id AND window_days = :window_days"
        )
        params = {"bdl_player_id": bdl_player_id, "window_days": window_days}
        if target_date is not None:
            sql += " AND as_of_date = :as_of_date"
            params["as_of_date"] = target_date
        sql += " ORDER BY as_of_date DESC LIMIT 1"
        return db.execute(text(sql), params).mappings().first()

    # Build query
    try:
        query = db.query(PlayerScore).filter(
            PlayerScore.bdl_player_id == bdl_player_id,
            PlayerScore.window_days == window_days,
        )

        # Resolve as_of_date
        if as_of_date is None:
            latest = query.order_by(PlayerScore.as_of_date.desc()).first()
            if not latest:
                raise HTTPException(
                    status_code=404,
                    detail=f"No player_scores found for bdl_player_id={bdl_player_id} window_days={window_days}",
                )
            as_of_date = latest.as_of_date
        else:
            latest = query.filter(PlayerScore.as_of_date == as_of_date).first()
            if not latest:
                raise HTTPException(
                    status_code=404,
                    detail=f"No player_scores found for bdl_player_id={bdl_player_id} window_days={window_days} as_of_date={as_of_date}",
                )
    except (ProgrammingError, OperationalError):
        latest = _load_score_via_schema_fallback(as_of_date)
        if not latest:
            detail = (
                f"No player_scores found for bdl_player_id={bdl_player_id} window_days={window_days}"
                if as_of_date is None
                else f"No player_scores found for bdl_player_id={bdl_player_id} window_days={window_days} as_of_date={as_of_date}"
            )
            raise HTTPException(status_code=404, detail=detail)
        as_of_date = _normalize_score_date(_get_score_value(latest, "as_of_date"))

    # Build category_scores
    category_scores = PlayerScoreCategoryBreakdown(
        z_hr=_get_score_value(latest, "z_hr"),
        z_rbi=_get_score_value(latest, "z_rbi"),
        z_nsb=_get_score_value(latest, "z_nsb"),
        z_avg=_get_score_value(latest, "z_avg"),
        z_obp=_get_score_value(latest, "z_obp"),
        z_era=_get_score_value(latest, "z_era"),
        z_whip=_get_score_value(latest, "z_whip"),
        z_k_per_9=_get_score_value(latest, "z_k_per_9"),
    )

    # Build score output
    score_out = PlayerScoreOut(
        bdl_player_id=_get_score_value(latest, "bdl_player_id"),
        as_of_date=_normalize_score_date(_get_score_value(latest, "as_of_date")),
        window_days=_get_score_value(latest, "window_days"),
        player_type=_get_score_value(latest, "player_type") or _infer_player_type(latest),
        games_in_window=_get_score_value(latest, "games_in_window", 0) or 0,
        composite_z=_get_score_value(latest, "composite_z", 0.0) or 0.0,
        score_0_100=_get_score_value(latest, "score_0_100", 0.0) or 0.0,
        confidence=_get_score_value(latest, "confidence", 0.0) or 0.0,
        category_scores=category_scores,
    )

    return PlayerScoresResponse(
        bdl_player_id=bdl_player_id,
        requested_window_days=window_days,
        as_of_date=as_of_date,
        score=score_out,
    )


@router.get("/api/fantasy/decisions", response_model=DecisionsResponse)
async def get_decisions(
    decision_type: Optional[Literal["lineup", "waiver"]] = Query(
        None, description="Filter by decision type: lineup or waiver"
    ),
    as_of_date: Optional[date] = Query(None, description="Decision date (defaults to latest available)"),
    limit: int = Query(50, ge=1, le=500, description="Max results to return (1-500)"),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """
    Get trusted DecisionResult rows with optional DecisionExplanation data.

    P17 Decision Engine results (lineup/waiver optimization outputs) with
    optional P19 explainability traces. Default as_of_date returns the latest
    available decisions. Returns empty list when no rows exist for the filter.

    For waiver decisions, performs live category-aware optimization using
    current matchup category_deficits. This ensures waiver recommendations
    are contextual to this week's matchup needs.

    Auth: verify_api_key required.
    """
    # For waiver decisions, perform live category-aware optimization
    # This is wrapped in extensive try/except to prevent cascading failures
    _original_as_of_date = as_of_date  # preserve for DB fallback if live path fails
    if decision_type == "waiver" or decision_type is None:
        try:
            from backend.services.decision_engine import (
                optimize_waivers,
                PlayerDecisionInput,
            )
            from backend.fantasy_baseball.category_aware_scorer import CategoryNeedVector

            logger.info("decisions endpoint: starting live category-aware waiver optimization")

            # Fetch live matchup scoreboard to build category_deficits
            try:
                client = get_yahoo_client()
                logger.info("decisions endpoint: Yahoo client obtained successfully")
            except Exception as client_err:
                logger.error("decisions endpoint: failed to get Yahoo client: %s", client_err)
                raise  # Re-raise to fall through to DB query

            my_team_key = os.getenv("YAHOO_TEAM_KEY", "")
            if not my_team_key:
                try:
                    my_team_key = client.get_my_team_key()
                    logger.info("decisions endpoint: obtained team_key=%s", my_team_key)
                except Exception as team_key_err:
                    logger.warning("decisions endpoint: failed to get team_key: %s", team_key_err)
                    pass  # Continue without team_key

            need_vector = None
            if my_team_key:
                try:
                    logger.info("decisions endpoint: fetching scoreboard for team=%s", my_team_key)
                    matchups_scoreboard = client.get_scoreboard()
                    logger.info("decisions endpoint: scoreboard fetched, %d matchups", len(matchups_scoreboard) if matchups_scoreboard else 0)

                    _category_deficits = []

                    for matchup_teams in _iter_scoreboard_matchup_teams(matchups_scoreboard):
                        my_tuple = None
                        opp_tuple = None
                        for t in matchup_teams:
                            if t[0] == my_team_key:
                                my_tuple = t
                            elif len(matchup_teams) == 2:
                                opp_tuple = t

                        if my_tuple and opp_tuple:
                            sid_map: dict = dict(_YAHOO_STAT_FALLBACK)
                            try:
                                _settings_dec = client.get_league_settings()
                                _stat_cats_dec = (
                                    _settings_dec
                                    .get("settings", [{}])[0]
                                    .get("stat_categories", {})
                                    .get("stats", [])
                                )
                                _dec_stat_entries: list = []
                                for _entry_d in _stat_cats_dec:
                                    if isinstance(_entry_d, dict):
                                        _s_d = _entry_d.get("stat", {})
                                        _sid_d = str(_s_d.get("stat_id", ""))
                                        _abbr_d = (
                                            _s_d.get("display_name")
                                            or _s_d.get("abbreviation")
                                            or _s_d.get("name")
                                            or _sid_d
                                        )
                                        _pos_d = _s_d.get("position_type", "")
                                        if _sid_d:
                                            _dec_stat_entries.append((_sid_d, _abbr_d, _pos_d))
                                _dec_abbr_pos: dict = {}
                                for _sid_d, _abbr_d, _pos_d in _dec_stat_entries:
                                    _dec_abbr_pos.setdefault(_abbr_d, set()).add(_pos_d)
                                _P_RENAME = {"HR": "HRA", "K": "K(P)"}
                                _B_RENAME = {"K": "K(B)", "HR": "HR"}
                                for _sid_d, _abbr_d, _pos_d in _dec_stat_entries:
                                    _final = _abbr_d
                                    if len(_dec_abbr_pos.get(_abbr_d, set())) > 1:
                                        if _pos_d == "P" and _abbr_d in _P_RENAME:
                                            _final = _P_RENAME[_abbr_d]
                                        elif _pos_d == "B" and _abbr_d in _B_RENAME:
                                            _final = _B_RENAME[_abbr_d]
                                    sid_map[_sid_d] = _final
                            except Exception as _e_sid:
                                logger.warning(
                                    "decisions endpoint: get_league_settings failed in sid_map build (using fallback): %s",
                                    _e_sid,
                                )

                            def _stats_dict_from_raw(raw_stats_list: list) -> dict:
                                out: dict = {}
                                for st in raw_stats_list:
                                    if not isinstance(st, dict):
                                        continue
                                    stobj = st.get("stat", {})
                                    if not isinstance(stobj, dict):
                                        continue
                                    sid_k = str(stobj.get("stat_id", ""))
                                    if not sid_k:
                                        continue
                                    key2 = sid_map.get(sid_k, sid_k)
                                    if isinstance(key2, str) and key2.isdigit():
                                        continue
                                    try:
                                        out[key2] = float(stobj.get("value", 0) or 0)
                                    except (TypeError, ValueError):
                                        out[key2] = 0.0
                                return out

                            my_stats_dict = _stats_dict_from_raw(my_tuple[2])
                            opp_stats_dict = _stats_dict_from_raw(opp_tuple[2])
                            for cat, my_val_f in my_stats_dict.items():
                                opp_val_f = opp_stats_dict.get(cat, 0.0)
                                result = compare_category(cat, my_val_f, opp_val_f)
                                _category_deficits.append((cat, result.gap))

                    if _category_deficits:
                        _CANONICAL_TO_BOARD = {
                            "R": "r", "H": "h", "HR": "hr", "RBI": "rbi", "SB": "nsb",
                            "AVG": "avg", "OPS": "ops",
                            "W": "w", "K": "k_pit", "SV": "nsv",
                            "ERA": "era", "WHIP": "whip", "QS": "qs", "K9": "k9",
                        }

                        needs = {
                            _CANONICAL_TO_BOARD.get(cat, cat.lower()): deficit
                            for cat, deficit in _category_deficits
                        }
                        need_vector = CategoryNeedVector(needs=needs)
                        logger.info("decisions endpoint: built CategoryNeedVector with %d categories", len(needs))
                except Exception as _sb_err:
                    logger.warning("decisions endpoint: scoreboard fetch failed (non-fatal): %s", _sb_err, exc_info=True)

            # Perform live waiver optimization with category awareness
            if need_vector and as_of_date is None:
                as_of_date = date.today()

            if need_vector:
                import asyncio
                from backend.fantasy_baseball.projection_sync import get_or_create_projection

                # Fetch current roster and waiver pool
                try:
                    logger.info("decisions endpoint: fetching roster and waiver pool")
                    roster = client.get_roster()
                    logger.info("decisions endpoint: fetched %d roster players", len(roster) if roster else 0)

                    free_agents = client.get_free_agents(count=100)
                    logger.info("decisions endpoint: fetched %d free agents", len(free_agents) if free_agents else 0)

                    if not free_agents:
                        logger.warning("decisions endpoint: free_agents is empty/None, skipping live optimization")
                        raise ValueError("No free agents available from Yahoo API")

                    # Build PlayerDecisionInput lists
                    players = []
                    for p in roster:
                        try:
                            proj = get_or_create_projection(p)
                            if proj and proj.bdl_player_id:
                                players.append(PlayerDecisionInput(
                                    bdl_player_id=proj.bdl_player_id,
                                    position=p.get("position", "UTIL"),
                                    composite_z=proj.get("composite_z", 0.0),
                                    z_hr=proj.get("z_hr", 0.0),
                                    z_rbi=proj.get("z_rbi", 0.0),
                                    z_nsb=proj.get("z_nsb", 0.0),
                                    z_r=proj.get("z_r", 0.0),
                                    z_ops=proj.get("z_ops", 0.0),
                                    z_w=proj.get("z_w", 0.0),
                                    z_k=proj.get("z_k", 0.0),
                                    z_nsv=proj.get("z_nsv", 0.0),
                                    z_era=proj.get("z_era", 0.0),
                                    z_whip=proj.get("z_whip", 0.0),
                                ))
                        except Exception as proj_err:
                            logger.warning("decisions endpoint: failed to build player input: %s", proj_err)
                            continue

                    logger.info("decisions endpoint: built %d roster players with projections", len(players))

                    waiver_pool = []
                    for fa in free_agents:
                        try:
                            proj = get_or_create_projection(fa)
                            if proj and proj.bdl_player_id:
                                waiver_pool.append(PlayerDecisionInput(
                                    bdl_player_id=proj.bdl_player_id,
                                    position=fa.get("position", "UTIL"),
                                    composite_z=proj.get("composite_z", 0.0),
                                    z_hr=proj.get("z_hr", 0.0),
                                    z_rbi=proj.get("z_rbi", 0.0),
                                    z_nsb=proj.get("z_nsb", 0.0),
                                    z_r=proj.get("z_r", 0.0),
                                    z_ops=proj.get("z_ops", 0.0),
                                    z_w=proj.get("z_w", 0.0),
                                    z_k=proj.get("z_k", 0.0),
                                    z_nsv=proj.get("z_nsv", 0.0),
                                    z_era=proj.get("z_era", 0.0),
                                    z_whip=proj.get("z_whip", 0.0),
                                ))
                        except Exception as proj_err:
                            logger.warning("decisions endpoint: failed to build waiver pool input: %s", proj_err)
                            continue

                    logger.info("decisions endpoint: built %d waiver pool candidates with projections", len(waiver_pool))

                    if not waiver_pool:
                        logger.warning("decisions endpoint: waiver_pool is empty after projection resolution, skipping live optimization")
                        raise ValueError("No valid waiver pool candidates")

                    # Run optimization with category awareness
                    logger.info("decisions endpoint: running optimize_waivers with need_vector")
                    _waiver_decision, waiver_results = await asyncio.to_thread(
                        optimize_waivers, players, waiver_pool, as_of_date, need_vector
                    )
                    logger.info("decisions endpoint: optimize_waivers returned %d results", len(waiver_results))

                    # Convert results to DecisionResultOut format
                    results = []
                    for wr in waiver_results[:limit]:
                        try:
                            decision_out = DecisionResultOut(
                                bdl_player_id=wr.bdl_player_id,
                                player_name=None,  # Will be resolved below
                                as_of_date=wr.as_of_date,
                                decision_type="waiver",
                                target_slot=wr.target_slot,
                                drop_player_id=wr.drop_player_id,
                                drop_player_name=None,
                                lineup_score=wr.lineup_score,
                                value_gain=wr.value_gain,
                                confidence=wr.confidence,
                                reasoning=wr.reasoning,
                            )

                            # Resolve player names
                            pname = db.query(PlayerIDMapping.full_name).filter(
                                PlayerIDMapping.bdl_id == wr.bdl_player_id
                            ).scalar()
                            decision_out.player_name = pname

                            if wr.drop_player_id:
                                dname = db.query(PlayerIDMapping.full_name).filter(
                                    PlayerIDMapping.bdl_id == wr.drop_player_id
                                ).scalar()
                                decision_out.drop_player_name = dname

                            results.append(DecisionWithExplanation(
                                decision=decision_out,
                                explanation=None,
                            ))

                            try:
                                _league_key = os.getenv("YAHOO_LEAGUE_ID", "default")
                                _target_date = wr.as_of_date or date.today()
                                _now_utc = datetime.now(ZoneInfo("UTC"))
                                _report_blob = {
                                    "decision_type": "waiver",
                                    "target_slot": wr.target_slot,
                                    "drop_player_id": wr.drop_player_id,
                                    "lineup_score": wr.lineup_score,
                                    "value_gain": wr.value_gain,
                                    "confidence": wr.confidence,
                                    "reasoning": wr.reasoning,
                                }
                                _existing_cache = db.query(PlayerValuationCache).filter_by(
                                    player_id=str(wr.bdl_player_id),
                                    target_date=_target_date,
                                    league_key=_league_key,
                                ).first()
                                if _existing_cache:
                                    _existing_cache.report = _report_blob
                                    _existing_cache.computed_at = _now_utc
                                    _existing_cache.data_as_of = _now_utc
                                    _existing_cache.invalidated_at = None
                                    if pname:
                                        _existing_cache.player_name = pname
                                else:
                                    db.add(PlayerValuationCache(
                                        id=str(uuid.uuid4()),
                                        player_id=str(wr.bdl_player_id),
                                        player_name=pname or f"BDL#{wr.bdl_player_id}",
                                        target_date=_target_date,
                                        league_key=_league_key,
                                        report=_report_blob,
                                        computed_at=_now_utc,
                                        data_as_of=_now_utc,
                                    ))
                            except Exception as _cache_err:
                                logger.error(
                                    "decisions endpoint: valuation cache write FAILED for bdl_id=%s: %s",
                                    wr.bdl_player_id, _cache_err,
                                    exc_info=True,
                                )
                                db.rollback()
                        except Exception as res_build_err:
                            logger.warning("decisions endpoint: failed to build result: %s", res_build_err)
                            continue

                    try:
                        db.commit()
                    except Exception as _commit_err:
                        db.rollback()
                        logger.warning("decisions endpoint: valuation cache commit failed: %s", _commit_err)

                    if results:
                        logger.info("decisions endpoint: returning %d live waiver results", len(results))
                        return DecisionsResponse(
                            decisions=results,
                            count=len(results),
                            as_of_date=as_of_date or date.today(),
                            decision_type="waiver",
                        )
                    else:
                        logger.warning("decisions endpoint: no results built from optimization output, falling back to DB")
                except Exception as _live_err:
                    logger.warning("decisions endpoint: live optimization failed (falling back to DB): %s", _live_err, exc_info=True)
        except Exception as _init_err:
            as_of_date = _original_as_of_date  # restore so DB fallback uses correct date
            logger.warning("decisions endpoint: category-aware setup failed (using DB fallback): %s", _init_err, exc_info=True)

    # Build base query with player name and drop player name joins
    # Use aliased PlayerIDMapping for drop player to avoid ambiguity
    DropMapping = aliased(PlayerIDMapping)
    query = (
        db.query(
            DecisionResult,
            PlayerIDMapping.full_name.label("player_name"),
            DropMapping.full_name.label("drop_player_name"),
        )
        .outerjoin(
            PlayerIDMapping,
            DecisionResult.bdl_player_id == PlayerIDMapping.bdl_id,
        )
        .outerjoin(
            DropMapping,
            DecisionResult.drop_player_id == DropMapping.bdl_id,
        )
    )

    # Apply decision_type filter if provided
    if decision_type:
        query = query.filter(DecisionResult.decision_type == decision_type)

    # Resolve as_of_date: default to latest available date in decision_results
    # Use a separate base query for date resolution to avoid filter contamination
    if as_of_date is None:
        date_query = db.query(DecisionResult)
        if decision_type:
            date_query = date_query.filter(DecisionResult.decision_type == decision_type)
        latest_date_row = date_query.order_by(DecisionResult.as_of_date.desc()).first()
        if latest_date_row:
            as_of_date = latest_date_row.as_of_date
        else:
            # No decisions at all - return empty response
            return DecisionsResponse(
                decisions=[],
                count=0,
                as_of_date=date.today(),
                decision_type=decision_type,
            )
    else:
        # Validate that requested date has any data (using separate query)
        date_query = db.query(DecisionResult)
        if decision_type:
            date_query = date_query.filter(DecisionResult.decision_type == decision_type)
        date_exists = date_query.filter(DecisionResult.as_of_date == as_of_date).first()
        if not date_exists:
            # No data for requested date - return empty response
            return DecisionsResponse(
                decisions=[],
                count=0,
                as_of_date=as_of_date,
                decision_type=decision_type,
            )

    # Apply date filter to main query
    query = query.filter(DecisionResult.as_of_date == as_of_date)

    # Filter low-value waiver recommendations BEFORE applying limit
    # This ensures we return up to `limit` high-value results, not fewer after filtering
    WAIVER_VALUE_GAIN_THRESHOLD = 0.10
    if decision_type == "waiver":
        query = query.filter(
            DecisionResult.value_gain.isnot(None),
            DecisionResult.value_gain > WAIVER_VALUE_GAIN_THRESHOLD,
        )
    elif decision_type is None:
        # When no type filter, exclude low-value waivers from mixed results
        query = query.filter(
            or_(
                DecisionResult.decision_type != "waiver",
                and_(
                    DecisionResult.value_gain.isnot(None),
                    DecisionResult.value_gain > WAIVER_VALUE_GAIN_THRESHOLD,
                ),
            )
        )

    # Order by confidence desc, then value_gain desc (most confident/valuable first)
    query = query.order_by(
        DecisionResult.confidence.desc(),
        DecisionResult.value_gain.desc(),
    )

    # Apply limit AFTER filtering
    query = query.limit(limit)

    # Fetch results
    rows = query.all()
    decision_rows = [row[0] for row in rows]  # Extract DecisionResult from tuples
    player_names = {row[0].id: row[1] for row in rows}  # Map decision_id -> player_name
    drop_player_names = {row[0].id: row[2] for row in rows}  # Map decision_id -> drop_player_name

    # Fetch all explanations for these decisions in one query
    decision_ids = [d.id for d in decision_rows]
    explanations = (
        db.query(DecisionExplanation)
        .filter(DecisionExplanation.decision_id.in_(decision_ids))
        .all()
    )
    explanation_map = {e.decision_id: e for e in explanations}

    # Build response
    results = []
    for dr in decision_rows:
        decision_out = DecisionResultOut(
            bdl_player_id=dr.bdl_player_id,
            player_name=player_names.get(dr.id),
            as_of_date=dr.as_of_date,
            decision_type=dr.decision_type,
            target_slot=dr.target_slot,
            drop_player_id=dr.drop_player_id,
            drop_player_name=drop_player_names.get(dr.id),
            lineup_score=dr.lineup_score,
            value_gain=dr.value_gain,
            confidence=dr.confidence,
            reasoning=dr.reasoning,
        )

        explanation_out = None
        if dr.id in explanation_map:
            expl = explanation_map[dr.id]
            factors = [
                FactorDetail(
                    name=f.get("name", ""),
                    value=str(f["value"]) if f.get("value") is not None else None,
                    label=f.get("label"),
                    weight=f.get("weight"),
                    narrative=f.get("narrative"),
                )
                for f in expl.factors_json
            ] if expl.factors_json else []

            explanation_out = DecisionExplanationOut(
                summary=expl.summary,
                factors=factors,
                confidence_narrative=expl.confidence_narrative,
                risk_narrative=expl.risk_narrative,
                track_record_narrative=expl.track_record_narrative,
            )

        results.append(
            DecisionWithExplanation(
                decision=decision_out,
                explanation=explanation_out,
            )
        )

    return DecisionsResponse(
        decisions=results,
        count=len(results),
        as_of_date=as_of_date,
        decision_type=decision_type,
    )


@router.get("/api/fantasy/decisions/status", response_model=DecisionPipelineStatus)
async def get_decisions_status(
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """
    Decision pipeline freshness and coverage observability.

    Returns the latest as_of_date for decision results, row counts by type,
    and a verdict indicating whether the pipeline is healthy, stale, partial,
    or missing data. Use this to show status indicators on the decisions page.

    Auth: verify_api_key required.
    """
    now_et = today_et()

    def _normalize_sql_date(raw_value):
        if isinstance(raw_value, datetime):
            return raw_value.date()
        if isinstance(raw_value, str):
            return date.fromisoformat(raw_value)
        return raw_value

    # Latest as_of_date for decision_results
    dr_latest = db.execute(
        text("SELECT MAX(as_of_date) AS latest_date FROM decision_results")
    ).scalar()
    dr_latest = _normalize_sql_date(dr_latest)

    # Row counts by decision_type for latest date
    dr_counts = {"lineup": None, "waiver": None}
    if dr_latest:
        for dtype in ("lineup", "waiver"):
            cnt = db.execute(
                text("""
                    SELECT COUNT(*) AS n FROM decision_results
                    WHERE as_of_date = :d AND decision_type = :dt
                """),
                {"d": dr_latest, "dt": dtype}
            ).scalar()
            dr_counts[dtype] = int(cnt or 0)

    dr_total = sum(v for v in dr_counts.values() if v is not None) if dr_latest else None

    # Compute freshness verdict (data stale if 2+ days old)
    # today_et() returns a date object directly (not datetime), so .date() is not valid
    today_et_date = now_et
    stale_threshold = today_et_date - timedelta(days=2)
    dr_is_fresh = dr_latest and dr_latest >= stale_threshold

    # Determine verdict
    if not dr_latest:
        verdict = "missing"
        message = "No decision data available yet."
    elif not dr_is_fresh:
        verdict = "stale"
        message = f"Decision data is stale (latest: {dr_latest})."
    elif dr_total == 0:
        verdict = "partial"
        message = f"Decision data exists for {dr_latest} but no results found."
    else:
        verdict = "healthy"
        message = f"Decision pipeline is healthy (latest: {dr_latest})."

    return DecisionPipelineStatus(
        verdict=verdict,
        message=message,
        checked_at=now_et.isoformat(),
        decision_results={
            "latest_as_of_date": dr_latest.isoformat() if dr_latest else None,
            "total_row_count": dr_total,
            "breakdown_by_type": dr_counts,
        },
    )


@router.get("/api/fantasy/decisions/accuracy", response_model=DecisionAccuracyResponse)
async def get_decisions_accuracy():
    """
    14-day override accuracy summary from the file-based decision tracker.

    Returns per-day accuracy trend and aggregated override comparison stats
    (how often the user's manual overrides beat the system's recommendations).

    No auth required — read-only, no sensitive data.
    """
    from backend.fantasy_baseball.decision_tracker import get_decision_tracker

    tracker = get_decision_tracker()

    end = datetime.now()
    trend_points: list[DecisionAccuracyTrendPoint] = []

    # Aggregate override stats across the full 14-day window
    total_decisions = 0
    override_better = 0
    override_worse = 0
    latest_date = end.strftime("%Y-%m-%d")

    for offset in range(13, -1, -1):  # 13 days ago … today (oldest first)
        day = (end - timedelta(days=offset)).strftime("%Y-%m-%d")
        acc = tracker.get_daily_accuracy(day)

        if acc is None:
            trend_points.append(DecisionAccuracyTrendPoint(date=day, accuracy_pct=-1.0))
            continue

        resolved_count = acc.correct_predictions + acc.incorrect_predictions
        day_accuracy = (
            round(acc.correct_predictions / resolved_count, 4)
            if resolved_count > 0
            else -1.0
        )
        trend_points.append(DecisionAccuracyTrendPoint(date=day, accuracy_pct=day_accuracy))

        # Accumulate from today (offset==0) for the top-level summary
        if offset == 0:
            total_decisions = acc.total_decisions
            override_better = acc.override_better_count
            override_worse = acc.override_worse_count

    override_total = override_better + override_worse
    override_accuracy_pct = (
        round(override_better / override_total, 4) if override_total > 0 else 0.0
    )

    return DecisionAccuracyResponse(
        date=latest_date,
        total_overrides=total_decisions,
        better_count=override_better,
        worse_count=override_worse,
        override_accuracy_pct=override_accuracy_pct,
        daily_trend=trend_points,
    )


# ============================================================================
# Phase 4: Matchup Scoreboard (P1 Page)
# ============================================================================

@router.get("/api/fantasy/scoreboard")
async def get_matchup_scoreboard(
    week: Optional[int] = Query(None, description="Matchup week number (1-25), defaults to current week"),
    opponent_name: Optional[str] = Query(None, description="Opponent team name, fetched from Yahoo if not provided"),
    db: Session = Depends(get_db),
) -> Dict:
    """
    GET /api/fantasy/scoreboard

    Returns complete Matchup Scoreboard with 18 category rows.

    Gate Criteria:
    - All 18 scoring categories present
    - Current stats from Yahoo
    - ROW projections (L3)
    - Category math (L1)
    - Monte Carlo win probabilities (L4)
    - Constraint budget state
    - Freshness metadata

    Phase 4.5a Priority 1: Wired to live Yahoo data.
    """
    from backend.services.scoreboard_orchestrator import assemble_matchup_scoreboard

    try:
        client = get_yahoo_client()
    except YahooAuthError:
        logger.error("scoreboard: Yahoo not configured")
        raise HTTPException(status_code=503, detail="Yahoo not configured")

    # Default to current week if not provided
    if week is None:
        week = _compute_mlb_current_week(datetime.now(ZoneInfo("America/New_York")).date())

    # Resolve my_team_key once (same env-var + API fallback as /api/fantasy/matchup).
    _my_team_key = os.getenv("YAHOO_TEAM_KEY", "")
    if not _my_team_key:
        try:
            _my_team_key = client.get_my_team_key()
        except Exception:
            _my_team_key = ""

    # Fetch live matchup stats using the proven get_scoreboard() path that
    # /api/fantasy/matchup already uses successfully.  The previous approach
    # (get_matchup_stats()) had fragile nested-struct team-key matching that
    # silently returned {} on shape variations, producing 0W-0L-18T on the
    # roster page.  Replacing it with _iter_scoreboard_matchup_teams() fixes
    # both pages to read from the same Yahoo data source.
    my_current_stats: Dict[str, float] = {}
    opp_current_stats: Dict[str, float] = {}
    safe_opponent_name = opponent_name or "Opponent"

    yahoo_fetch_time = datetime.now(ZoneInfo("America/New_York"))
    try:
        raw_matchups = client.get_scoreboard(week=week)
        yahoo_fetch_time = datetime.now(ZoneInfo("America/New_York"))
        logger.info("scoreboard: fetched %d raw matchups for week %d", len(raw_matchups or []), week)
    except YahooAuthError as auth_err:
        logger.error("scoreboard: Yahoo auth failed for week %d: %s", week, auth_err, exc_info=False)
        raise HTTPException(status_code=401, detail="Yahoo authentication expired")
    except YahooAPIError as api_err:
        logger.error(
            "scoreboard: Yahoo API error for week %d — HTTP %s — full_body=%r",
            week,
            api_err.status_code,
            str(api_err),
            exc_info=True,
        )
        if api_err.status_code == 400:
            raise HTTPException(
                status_code=400,
                detail=f"Yahoo rejected request (bad parameter or expired token) — {str(api_err)[:200]}",
            )
        if api_err.status_code in (401, 403):
            raise HTTPException(status_code=401, detail="Yahoo authentication expired — re-auth required")
        if api_err.status_code == 503:
            raise HTTPException(status_code=503, detail="Yahoo service unavailable")
        raise HTTPException(status_code=502, detail=f"Yahoo API error: {str(api_err)[:100]}")
    except Exception as yahoo_err:
        logger.error(
            "scoreboard: unexpected error fetching scoreboard for week %d: %s",
            week, yahoo_err, exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Scoreboard fetch failed: {type(yahoo_err).__name__}")

    # Convert raw Yahoo stat dicts [{stat: {stat_id, value}}] → {canonical_code: float}.
    # _flatten_scoreboard_team_entry already extracts the stats list; this function
    # translates stat_ids to canonical codes and coerces values to float.
    _stat_id_map: dict = dict(_YAHOO_STAT_FALLBACK)

    def _parse_stats_to_float(stats_raw: list) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for s in stats_raw:
            if not isinstance(s, dict):
                continue
            stat = s.get("stat", {})
            if not isinstance(stat, dict):
                continue
            sid = str(stat.get("stat_id", ""))
            code = _stat_id_map.get(sid, "")
            if not code:
                continue
            val_raw = stat.get("value", "")
            if isinstance(val_raw, str) and "/" in val_raw:
                try:
                    val_raw = val_raw.split("/")[0]
                except (ValueError, IndexError):
                    val_raw = "0"
            try:
                out[code] = float(val_raw)
            except (TypeError, ValueError):
                pass
        return out

    for _matchup_teams in _iter_scoreboard_matchup_teams(raw_matchups or []):
        _my_tuple = None
        for _t in _matchup_teams:
            _t_key = _t[0]
            if not _t_key or not _my_team_key:
                continue
            # CRITICAL: Use exact match ONLY. Substring matching causes false positives
            # when team IDs are numeric (e.g., "t.7" matches "t.71", "t.72", etc.)
            if _t_key == _my_team_key:
                _my_tuple = _t
                break
        if _my_tuple is None:
            continue
        _opp_tuple = next((_t for _t in _matchup_teams if _t[0] != _my_tuple[0]), None)
        my_current_stats = _parse_stats_to_float(_my_tuple[2])
        opp_current_stats = _parse_stats_to_float(_opp_tuple[2]) if _opp_tuple else {}
        if _opp_tuple and _opp_tuple[1]:
            safe_opponent_name = _opp_tuple[1]
        logger.info(
            "scoreboard: parsed my_stats=%d cats, opp_stats=%d cats, opponent=%r (my_key=%r, my_name=%r)",
            len(my_current_stats), len(opp_current_stats), safe_opponent_name,
            _my_tuple[0], _my_tuple[1],
        )
        break
    else:
        logger.error(
            "scoreboard: CRITICAL — could not find my team (key=%r) in %d matchups. Returning empty stats.",
            _my_team_key, len(raw_matchups or []),
        )
        # Log all available team keys for debugging
        all_team_keys = []
        for mt in _iter_scoreboard_matchup_teams(raw_matchups or []):
            for t in mt:
                all_team_keys.append(t[0])
        logger.error("scoreboard: available team_keys in matchups: %r", all_team_keys)

    # Derive real constraint values from the parsed stats and current date.
    # ip_accumulated and days_remaining are computable without extra API calls.
    # acquisitions_used and il_used require a separate Yahoo call — left for a
    # dedicated budget-sync task (the budget endpoint already handles those).
    _now_et = datetime.now(ZoneInfo("America/New_York"))
    _ip_accumulated = float(my_current_stats.get("IP", 0.0))
    _ip_minimum = 18.0  # Yahoo H2H weekly IP floor (matches budget endpoint)
    _days_remaining = max(0, 6 - _now_et.weekday())  # Mon=0 → 6 days left; Sun=6 → 0

    # Mock player scores (empty for now)
    my_player_scores = []

    # Assemble scoreboard with defensive error handling
    try:
        result = assemble_matchup_scoreboard(
            week=week,
            opponent_name=safe_opponent_name,
            my_current_stats=my_current_stats,
            opp_current_stats=opp_current_stats,
            my_player_scores=my_player_scores,
            opp_player_scores=None,
            ip_accumulated=_ip_accumulated,
            ip_minimum=_ip_minimum,
            games_remaining=_days_remaining,
            days_remaining=_days_remaining,
            acquisitions_used=5,
            il_used=1,
            n_monte_carlo_sims=1000,
            force_stale=False,
            fetched_at=yahoo_fetch_time,
        )
        logger.debug("scoreboard: assembled scoreboard for week %d", week)
    except ValueError as val_err:
        logger.error("scoreboard: invalid data for week %d: %s", week, val_err, exc_info=False)
        raise HTTPException(status_code=400, detail=f"Invalid scoreboard data: {str(val_err)[:100]}")
    except TypeError as type_err:
        logger.error("scoreboard: type error assembling scoreboard for week %d: %s", week, type_err, exc_info=True)
        raise HTTPException(status_code=500, detail="Scoreboard assembly failed: data structure mismatch")
    except Exception as orch_err:
        logger.error("scoreboard: unexpected error in orchestrator for week %d: %s", week, orch_err, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Scoreboard assembly failed: {type(orch_err).__name__}")

    # Serialize result with defensive handling
    try:
        return {
            "week": result.week,
            "opponent_name": result.opponent_name,
            "categories_won": result.categories_won,
            "categories_lost": result.categories_lost,
            "categories_tied": result.categories_tied,
            "projected_won": result.projected_won,
            "projected_lost": result.projected_lost,
            "projected_tied": result.projected_tied,
            "overall_win_probability": result.overall_win_probability,
            "rows": [
                {
                    "category": r.category,
                    "category_label": r.category_label,
                    "is_lower_better": r.is_lower_better,
                    "is_batting": r.is_batting,
                    "my_current": r.my_current,
                    "opp_current": r.opp_current,
                    "current_margin": r.current_margin,
                    "my_projected_final": r.my_projected_final,
                    "opp_projected_final": r.opp_projected_final,
                    "projected_margin": r.projected_margin,
                    "status": r.status.value if r.status else None,
                    "flip_probability": r.flip_probability,
                    "delta_to_flip": r.delta_to_flip,
                    "games_remaining": r.games_remaining,
                    "ip_context": r.ip_context,
                }
                for r in result.rows
            ],
            "budget": {
                "acquisitions_used": result.budget.acquisitions_used,
                "acquisitions_remaining": result.budget.acquisitions_remaining,
                "acquisition_limit": result.budget.acquisition_limit,
                "acquisition_warning": result.budget.acquisition_warning,
                "il_used": result.budget.il_used,
                "il_total": result.budget.il_total,
                "ip_accumulated": result.budget.ip_accumulated,
                "ip_minimum": result.budget.ip_minimum,
                "ip_pace": result.budget.ip_pace.value,
                "as_of": result.budget.as_of.isoformat(),
            },
            "freshness": {
                "primary_source": result.freshness.primary_source,
                "fetched_at": result.freshness.fetched_at.isoformat() if result.freshness.fetched_at else None,
                "computed_at": result.freshness.computed_at.isoformat(),
                "staleness_threshold_minutes": result.freshness.staleness_threshold_minutes,
                "is_stale": result.freshness.is_stale,
            },
        }
    except AttributeError as attr_err:
        logger.error("scoreboard: attribute error serializing result for week %d: %s", week, attr_err, exc_info=True)
        raise HTTPException(status_code=500, detail="Scoreboard serialization failed: missing field")


@router.get("/api/fantasy/budget")
async def get_constraint_budget(
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> Dict:
    """
    GET /api/fantasy/budget

    Returns current constraint budget state for the global header.

    Gate Criteria:
    - Acquisitions used/remaining (with warning at 6+)
    - IL slots used/total
    - IP accumulated vs minimum
    - IP pace flag (BEHIND/ON_TRACK/COMPLETE)
    - Freshness metadata

    Phase 4.5a Priority 1: Wired to live Yahoo data.
    """
    from backend.services.scoreboard_orchestrator import compute_budget_state
    from backend.services.constraint_helpers import count_weekly_acquisitions
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    try:
        client = get_yahoo_client()
    except YahooAuthError:
        raise HTTPException(status_code=503, detail="Yahoo not configured")

    now_et = datetime.now(ZoneInfo("America/New_York"))
    team_key = os.getenv("YAHOO_TEAM_KEY", "469.l.72586.t.7")

    # Fetch live data from Yahoo
    acquisitions_used = 0
    il_used = 0
    il_total = 3  # Yahoo standard IL slots
    acquisition_limit = 8  # Yahoo standard adds

    # 1. Count IL players from roster
    try:
        roster = client.get_roster(team_key=team_key)
        il_count = sum(1 for p in roster if p.get("selected_position") in ["IL", "IL60"])
        il_used = il_count
    except (YahooAuthError, YahooAPIError):
        pass  # Fall back to 0

    # 2. Count acquisitions since Monday 00:00 ET (Yahoo matchup week start)
    transactions: list = []
    days_since_monday = now_et.weekday()  # Monday=0
    week_start = now_et.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_since_monday)
    week_end = now_et
    try:
        transactions = client.get_transactions(t_type="add")
        logger.info("budget: fetched %d transactions from Yahoo", len(transactions))
        if transactions:
            logger.debug("budget: sample txn[0] keys=%s type=%s ts=%s",
                         list(transactions[0].keys()),
                         transactions[0].get("type"),
                         transactions[0].get("timestamp"))
        acquisitions_used = count_weekly_acquisitions(
            transactions, team_key, week_start, week_end
        )
        logger.info("budget: yahoo acquisitions_used=%d (week %s–%s)", acquisitions_used, week_start.date(), week_end.date())
    except Exception as _acq_err:
        logger.warning("budget: acquisitions count failed: %s", _acq_err, exc_info=True)

    # Read local RosterAcquisition rows and take max to eliminate Yahoo API lag
    try:
        from backend.models import RosterAcquisition
        local_count = db.query(RosterAcquisition).filter(
            RosterAcquisition.team_key == team_key,
            RosterAcquisition.week_start >= week_start,
        ).count()
        if local_count > acquisitions_used:
            logger.info("budget: local_count=%d > yahoo=%d — using local", local_count, acquisitions_used)
            acquisitions_used = local_count
    except Exception as _local_err:
        logger.warning("budget: local acquisition query failed: %s", _local_err)

    # 3. Season calendar — week number, pace metadata
    current_week = _compute_mlb_current_week(now_et.date())
    league_meta: dict = {}
    # Yahoo sync guard: if our epoch drifts from Yahoo's authoritative value, trust Yahoo
    try:
        league_meta = client.get_league()
        yahoo_week = int(league_meta.get("current_week") or 0)
        if yahoo_week > 0 and yahoo_week != current_week:
            logger.warning(
                "budget: computed week=%d differs from Yahoo current_week=%d — using Yahoo value",
                current_week,
                yahoo_week,
            )
            current_week = yahoo_week
    except Exception as _week_sync_err:
        logger.debug("budget: Yahoo week sync skipped: %s", _week_sync_err)
    season_days_elapsed = max(0, (now_et.date() - _MLB_FIRST_MATCHUP_MONDAY).days)
    weeks_remaining = max(0, _FANTASY_TOTAL_WEEKS - current_week)
    # Days left in the current Yahoo matchup week (weeks run Mon–Sun)
    days_in_week_remaining = max(0, 7 - now_et.weekday())  # Monday=0 → 7 remaining

    # 4. IP tracking - primary: get_matchup_stats; fallback: get_scoreboard
    ip_accumulated = 0.0
    ip_data_available = False
    ip_as_of: Optional[str] = None
    try:
        matchup_stats = client.get_matchup_stats(week=current_week, my_team_key=team_key)
        if matchup_stats:
            my_stats = matchup_stats.get("my_stats", {})
            if "IP" in my_stats:
                ip_accumulated = float(my_stats["IP"])
                ip_data_available = True
                _h = now_et.hour % 12 or 12
                ip_as_of = f"{_h}:{now_et.strftime('%M')} {'AM' if now_et.hour < 12 else 'PM'} ET"
            elif my_stats:
                # Stats returned but no IP key — Yahoo may not have committed pitching yet
                ip_data_available = True
    except (YahooAuthError, YahooAPIError, Exception) as exc:
        logger.warning("budget: IP primary (get_matchup_stats) failed, trying scoreboard: %s", exc)

    if not ip_data_available:
        try:
            _sb_matchups = client.get_scoreboard(week=current_week)
            for _matchup_teams in _iter_scoreboard_matchup_teams(_sb_matchups or []):
                for _t_key, _t_name, _t_stats in _matchup_teams:
                    if _t_key and team_key and (_t_key == team_key or _t_key in team_key or team_key in _t_key):
                        for _s in _t_stats:
                            _stat = _s.get("stat", {})
                            if str(_stat.get("stat_id", "")) == "50":  # stat_id 50 = IP
                                try:
                                    ip_accumulated = float(_stat.get("value", 0.0))
                                    ip_data_available = True
                                    _h = now_et.hour % 12 or 12
                                    ip_as_of = f"{_h}:{now_et.strftime('%M')} {'AM' if now_et.hour < 12 else 'PM'} ET"
                                except (TypeError, ValueError):
                                    pass
                        break
        except Exception as _sb_err:
            logger.debug("budget: scoreboard IP fallback failed: %s", _sb_err)

    ip_minimum = 18.0  # Yahoo H2H standard (innings pitched per week) - matches scoreboard_orchestrator.py

    # Count season-total acquisitions from the already-fetched transaction list
    # Use the same team filtering as weekly acquisitions to avoid counting league-wide transactions
    acquisitions_this_season = 0
    try:
        season_start = _MLB_FIRST_MATCHUP_MONDAY
        season_start_ts = datetime.combine(season_start, datetime.min.time(), tzinfo=ZoneInfo("America/New_York")).timestamp()
        now_ts = now_et.timestamp()

        for txn in transactions:
            # Filter to add transactions only (same logic as count_weekly_acquisitions)
            if txn.get("type") not in ("add", "add/drop"):
                continue

            # Filter by date range (season_start to now)
            raw_ts = txn.get("timestamp")
            if raw_ts is None:
                continue
            try:
                ts_float = float(raw_ts)
            except (TypeError, ValueError):
                continue

            if not (season_start_ts <= ts_float <= now_ts):
                continue

            # Filter by team (same logic as count_weekly_acquisitions)
            dest_team = (
                txn.get("destination_team_key")
                or txn.get("destination_team", {}).get("team_key")
            )

            # Fallback: walk transaction_data or players for player-level destination
            if not dest_team:
                txn_data = txn.get("transaction_data") or txn.get("players") or []
                if isinstance(txn_data, dict):
                    items = [v for k, v in txn_data.items() if k != "count" and isinstance(v, dict)]
                elif isinstance(txn_data, list):
                    items = txn_data
                else:
                    items = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    dest_team = (
                        item.get("destination_team_key")
                        or item.get("destination_team", {}).get("team_key")
                    )
                    if dest_team:
                        break
                    for _v in item.values():
                        if isinstance(_v, dict):
                            dest_team = (
                                _v.get("destination_team_key")
                                or _v.get("destination_team", {}).get("team_key")
                            )
                            if dest_team:
                                break
                        elif isinstance(_v, list):
                            for _sub in _v:
                                if isinstance(_sub, dict):
                                    dest_team = (
                                        _sub.get("destination_team_key")
                                        or _sub.get("destination_team", {}).get("team_key")
                                    )
                                    if dest_team:
                                        break
                            if dest_team:
                                break
                    if dest_team:
                        break

            if dest_team == team_key:
                acquisitions_this_season += 1
    except Exception:
        pass  # non-critical; leave as 0

    # 5. Waiver priority — my team's rolling waiver rank among league teams
    waiver_priority_out: Optional[dict] = None
    try:
        _wpr = client.get_team_waiver_priorities()
        _num_teams = int(league_meta.get("num_teams", 0))
        if not _num_teams:
            _num_teams = len(_wpr)
        _my_priority = _wpr.get(team_key)
        if _my_priority is not None and _num_teams > 0:
            _waiver_type = int(league_meta.get("waiver_type", 0))
            if _waiver_type == 0:  # rolling waivers
                if _my_priority <= 3:
                    _rec = "High priority — use claims aggressively before your rank resets"
                elif _my_priority <= _num_teams // 2:
                    _rec = "Moderate priority — be selective with claims"
                else:
                    _rec = "Low priority — prioritize free-agent pickups over waiver claims"
                waiver_priority_out = {
                    "priority": _my_priority,
                    "total": _num_teams,
                    "waiver_type": "rolling",
                    "recommendation": _rec,
                }
    except Exception as _wp_err:
        logger.debug("budget: waiver_priority fetch non-fatal: %s", _wp_err)

    budget = compute_budget_state(
        acquisitions_used=acquisitions_used,
        acquisition_limit=acquisition_limit,
        il_used=il_used,
        il_total=il_total,
        ip_accumulated=ip_accumulated,
        ip_minimum=ip_minimum,
        days_remaining=days_in_week_remaining,
        season_days_elapsed=season_days_elapsed,
    )

    return {
        "budget": {
            "acquisitions_used": budget.acquisitions_used,
            "acquisitions_remaining": budget.acquisitions_remaining,
            "acquisition_limit": budget.acquisition_limit,
            "acquisition_warning": budget.acquisition_warning,
            "il_used": budget.il_used,
            "il_total": budget.il_total,
            "ip_accumulated": budget.ip_accumulated,
            "ip_minimum": budget.ip_minimum,
            "ip_pace": budget.ip_pace.value,
            "ip_data_available": ip_data_available,
            "ip_as_of": ip_as_of,
            "as_of": budget.as_of.isoformat(),
            "week_label": f"Week {current_week}",
            "weeks_remaining": weeks_remaining,
            "days_in_week_remaining": days_in_week_remaining,
            "acquisitions_this_season": acquisitions_this_season,
        },
        "waiver_priority": waiver_priority_out,
        "freshness": {
            "primary_source": "yahoo",
            "fetched_at": now_et.isoformat(),
            "computed_at": now_et.isoformat(),
            "staleness_threshold_minutes": 60,
            "is_stale": False,
        },
    }


@router.get("/api/fantasy/coverage")
async def get_player_coverage(
    top_n: int = 50,
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Coverage audit: what fraction of top-N free agents have real cat_scores vs draft board fallback."""
    from backend.models import PlayerProjection, CanonicalProjection, PlayerIdentity
    from backend.fantasy_baseball.id_resolution_service import _normalize_name

    proj_names = {
        _normalize_name(r.player_name or "")
        for r in db.query(PlayerProjection).filter(
            PlayerProjection.cat_scores.isnot(None)
        ).all()
    }
    identity_names = {
        r.normalized_name for r in db.query(PlayerIdentity).all()
    }

    try:
        client = get_yahoo_client()
        players = []
        for start in range(0, top_n, 25):
            batch = client.get_free_agents(start=start, count=min(25, top_n - start))
            players.extend(batch)
            if len(players) >= top_n:
                break
        players = players[:top_n]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Yahoo fetch failed: {e}")

    results = []
    for p in players:
        name = p.get("name", "")
        norm = _normalize_name(name)
        has_fg = norm in proj_names
        has_id = norm in identity_names
        results.append({
            "name": name,
            "percent_owned": p.get("percent_owned", 0.0),
            "has_fangraphs_cat_scores": has_fg,
            "has_player_identity": has_id,
            "data_tier": "FANGRAPHS_ROS" if has_fg else "DRAFT_BOARD_FALLBACK",
        })

    results.sort(key=lambda x: x["percent_owned"] or 0, reverse=True)
    total = len(results)
    return {
        "total_checked": total,
        "fangraphs_coverage_pct": round(100 * sum(1 for r in results if r["has_fangraphs_cat_scores"]) / max(total, 1), 1),
        "identity_coverage_pct": round(100 * sum(1 for r in results if r["has_player_identity"]) / max(total, 1), 1),
        "missing_fangraphs": [r for r in results if not r["has_fangraphs_cat_scores"]][:20],
        "players": results,
    }


@router.get("/api/fantasy/freshness", tags=["freshness"])
def get_freshness_report(db: Session = Depends(get_db)):
    """
    Return normalized freshness state for all data sources.

    Severity levels:
    - fresh: age < 60 minutes
    - warning: 60 <= age < 120 minutes
    - critical: age >= 120 minutes
    - unknown: no timestamp available
    """
    from backend.contracts import FreshnessReport, FreshnessSeverity, FreshnessState
    from backend.fantasy_baseball.daily_lineup_optimizer import get_lineup_freshness
    from backend.services.dashboard_service import DashboardService
    from backend.services.waiver_edge_detector import get_waiver_freshness

    sources_raw = []

    try:
        sources_raw.append(get_waiver_freshness())
    except Exception as e:
        logger.warning(f"freshness: waiver_edge_detector error: {e}")

    try:
        sources_raw.append(get_lineup_freshness())
    except Exception as e:
        logger.warning(f"freshness: daily_lineup_optimizer error: {e}")

    try:
        svc = DashboardService()
        sources_raw.extend(svc.get_freshness_states())
    except Exception as e:
        logger.warning(f"freshness: dashboard_service error: {e}")

    sources = [FreshnessState(**s) if isinstance(s, dict) else s for s in sources_raw]

    severities = [s.severity for s in sources]
    severity_order = {
        FreshnessSeverity.CRITICAL: 3,
        FreshnessSeverity.WARNING: 2,
        FreshnessSeverity.UNKNOWN: 1,
        FreshnessSeverity.FRESH: 0,
    }
    overall = max(severities, key=lambda sv: severity_order.get(sv, 0)) if severities else FreshnessSeverity.UNKNOWN

    report = FreshnessReport(
        sources=sources,
        overall_severity=overall,
        stale_count=sum(1 for s in sources if s.is_stale),
        fresh_count=sum(1 for s in sources if not s.is_stale),
    )
    return report.model_dump()
