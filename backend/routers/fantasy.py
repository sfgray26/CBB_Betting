"""
Fantasy router — all /api/fantasy/*, /api/dashboard/*, /api/user/preferences* routes.

Strangler-fig extraction from backend/main.py.
Do NOT import from other backend.routers modules here.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import text, func
from typing import List, Optional
import logging
import os
import json as _json
import asyncio
from zoneinfo import ZoneInfo
from datetime import date, datetime, timedelta
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
)
from backend.utils.fantasy_stat_contract import YAHOO_STAT_ID_FALLBACK, LEAGUE_SCORING_CATEGORIES
from backend.utils.time_utils import today_et
from backend.fantasy_baseball.yahoo_client_resilient import (
    YahooAuthError,
    YahooAPIError,
    ResilientYahooClient,
    get_yahoo_client,
    get_resilient_yahoo_client,
)
from backend.fantasy_baseball.daily_lineup_optimizer import get_lineup_optimizer
from backend.services.job_queue_service import submit_job as jq_submit, get_job_status as jq_status

logger = logging.getLogger(__name__)
router = APIRouter()

# Module-level MLB probable-starts cache (shared state — same pattern as main.py)
_STARTS_CACHE: dict = {}

# Shared fallback for Yahoo numeric stat category IDs.
_YAHOO_STAT_FALLBACK: dict = dict(YAHOO_STAT_ID_FALLBACK)


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

    _injury_lookup: dict = {
        p.get("name", "").lower(): (p.get("status") if isinstance(p.get("status"), str) else None)
        for p in _lineup_roster
        if p.get("name")
    }

    _name_to_player_key: dict = {
        p.get("name", "").strip().lower(): p.get("player_key", "")
        for p in _lineup_roster
        if p.get("name") and p.get("player_key")
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
                _pname = a["player_name"]
                batters.append(LineupPlayerOut(
                    player_id=a["player_id"] or _pname,
                    player_key=_name_to_player_key.get(_pname.lower().strip(), "") or None,
                    name=_pname,
                    team=team,
                    position="?",
                    implied_runs=round(a.get("implied_runs", opp_impl), 2),
                    park_factor=round(a.get("park_factor", 1.0), 3),
                    lineup_score=round(a.get("smart_score", 0), 3),
                    start_time=start,
                    opponent=opp,
                    status="START" if a["slot"] != "BN" else "BENCH",
                    assigned_slot=a["slot"],
                    has_game=a.get("has_game", False),
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
            )
            batters = []
            for s in solved_slots:
                opp, start, opp_impl = _get_game_context(s.player_team)
                batters.append(LineupPlayerOut(
                    player_id=s.player_name,
                    player_key=_name_to_player_key.get(s.player_name.lower().strip(), "") or None,
                    name=s.player_name,
                    team=s.player_team,
                    position=s.positions[0] if s.positions else "?",
                    implied_runs=round(opp_impl, 2),
                    park_factor=round(s.park_factor, 3),
                    lineup_score=round(s.lineup_score, 3),
                    start_time=start,
                    opponent=opp,
                    status="START" if s.slot != "BN" else "BENCH",
                    assigned_slot=s.slot,
                    has_game=s.has_game,
                    injury_status=_injury_lookup.get(s.player_name.lower()),
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
            _b_name = b.get("name", "")
            batters.append(LineupPlayerOut(
                player_id=str(b.get("player_id", _b_name)),
                player_key=_name_to_player_key.get(_b_name.lower().strip(), "") or None,
                name=_b_name,
                team=team,
                position=(b.get("positions") or ["OF"])[0],
                implied_runs=round(opp_impl, 2),
                park_factor=float(b.get("park_factor", 1.0)),
                lineup_score=float(b.get("score", 0)),
                start_time=start,
                opponent=opp,
                status="START" if i < 9 else "BENCH",
                assigned_slot=None,
                has_game=b.get("has_game", True),
                injury_status=_injury_lookup.get(_b_name.lower()),
            ))

    for _b in batters:
        if _b.status == "START" and not _b.opponent:
            _b.status = "BENCH"
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
                opp_factor = max(0, 5.0 - opp_implied)
                park_factor_score = (2.0 - park_factor) * 5
                sp_score = min(10, opp_factor + park_factor_score)

            status = "START" if has_start else "NO_START"
            if not is_sp:
                status = "RP"

            pitcher_type = "SP" if is_sp else "RP"

            pitchers.append(StartingPitcherOut(
                player_id=p.get("player_key") or p.get("name", ""),
                player_key=p.get("player_key") or _name_to_player_key.get(p.get("name", "").lower().strip(), "") or None,
                name=p.get("name", ""),
                team=team,
                pitcher_type=pitcher_type,
                opponent=opponent,
                opponent_implied_runs=round(opp_implied, 2),
                park_factor=round(park_factor, 3),
                sp_score=round(sp_score, 3),
                start_time=start_time,
                status=status,
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
        from backend.fantasy_baseball.daily_briefing import get_briefing_generator
        generator = get_briefing_generator(record_decisions=record_decisions)
        briefing = generator.generate(
            roster=roster,
            projections=projections,
            game_date=briefing_date,
        )

        return {
            "date": briefing_date,
            "generated_at": briefing.generated_at.isoformat(),
            "strategy": briefing.strategy,
            "risk_profile": briefing.risk_profile,
            "overall_confidence": briefing.overall_confidence,
            "summary": {
                "total_decisions": briefing.total_decisions,
                "easy_decisions": briefing.easy_decisions,
                "tough_decisions": briefing.tough_decisions,
                "monitor_count": briefing.monitor_count,
            },
            "categories": [
                {
                    "category": c.category,
                    "current": c.current,
                    "opponent": c.opponent,
                    "status": c.status,
                    "urgency": c.urgency,
                }
                for c in briefing.categories
            ],
            "starters": [p.to_card() for p in briefing.start_recommendations],
            "bench": [p.to_card() for p in briefing.bench_recommendations[:5]],
            "monitor": [p.to_card() for p in briefing.monitor_list],
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

    _YAHOO_CAT_TO_BOARD = {
        "R": "r", "H": "h", "HR": "hr", "RBI": "rbi", "TB": "tb",
        "SB": "nsb", "AVG": "avg", "OPS": "ops",
        "W": "w", "L": "l", "K": "k_pit", "SO": "k_pit",
        "SV": "nsv", "ERA": "era", "WHIP": "whip",
        "QS": "qs", "K9": "k9", "K/9": "k9",
    }

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
        free_agents = client.get_free_agents(
            position=position or "", start=_fa_start, count=per_page
        )

        try:
            matchups = client.get_scoreboard()
            for m in matchups:
                if isinstance(m, dict):
                    teams = m.get("teams", {})
                    team_keys_in_matchup = []
                    team_names = {}
                    if isinstance(teams, list):
                        raw_entries = [item.get("team", []) for item in teams if isinstance(item, dict)]
                    elif isinstance(teams, dict):
                        count_t = int(teams.get("count", 0))
                        raw_entries = [teams.get(str(ti), {}).get("team", []) for ti in range(count_t)]
                    else:
                        raw_entries = []
                    for t_entry in raw_entries:
                        t_meta = {}
                        if isinstance(t_entry, list):
                            for sub in t_entry:
                                if isinstance(sub, list):
                                    for item in sub:
                                        if isinstance(item, dict):
                                            t_meta.update(item)
                                elif isinstance(sub, dict):
                                    t_meta.update(sub)
                        tk = t_meta.get("team_key", "")
                        tn = t_meta.get("name", "")
                        team_keys_in_matchup.append(tk)
                        team_names[tk] = tn
                    if my_team_key in team_keys_in_matchup:
                        for tk in team_keys_in_matchup:
                            if tk != my_team_key:
                                matchup_opponent = team_names.get(tk, "TBD")
                        break
        except Exception:
            pass

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
            if matchup_opponent != "TBD":
                matchups2 = client.get_scoreboard()
                lower_better = {"ERA", "WHIP", "L", "K(B)", "HRA"}
                for m2 in matchups2:
                    if not isinstance(m2, dict):
                        continue
                    teams2 = m2.get("teams", {})
                    team_stats_map: dict = {}
                    if isinstance(teams2, list):
                        team_entries2 = [item["team"] for item in teams2 if isinstance(item, dict) and "team" in item]
                    elif isinstance(teams2, dict):
                        count2 = int(teams2.get("count", 0))
                        team_entries2 = [teams2.get(str(ti2), {}).get("team", []) for ti2 in range(count2)]
                    else:
                        continue
                    for entry2 in team_entries2:
                        t_meta2: dict = {}
                        stats_raw2: list = []
                        items2 = entry2 if isinstance(entry2, list) else [entry2]
                        for sub2 in items2:
                            if isinstance(sub2, list):
                                for it2 in sub2:
                                    if isinstance(it2, dict):
                                        t_meta2.update(it2)
                            elif isinstance(sub2, dict):
                                if "team_stats" in sub2:
                                    inner2 = sub2["team_stats"].get("stats", [])
                                    if isinstance(inner2, list):
                                        stats_raw2 = inner2
                                else:
                                    t_meta2.update(sub2)
                        tk2 = t_meta2.get("team_key", "")
                        sd2: dict = {}
                        for st2 in stats_raw2:
                            if isinstance(st2, dict):
                                stobj = st2.get("stat", {})
                                if isinstance(stobj, dict):
                                    sid_k = str(stobj.get("stat_id", ""))
                                    key2 = sid_map.get(sid_k, sid_k)
                                    try:
                                        sd2[key2] = float(stobj.get("value", 0) or 0)
                                    except (TypeError, ValueError):
                                        sd2[key2] = 0.0
                        team_stats_map[tk2] = sd2
                    if my_team_key not in team_stats_map:
                        continue
                    my_stats = team_stats_map[my_team_key]
                    opp_key = next((k for k in team_stats_map if k != my_team_key), None)
                    if not opp_key:
                        continue
                    opp_stats = team_stats_map[opp_key]
                    for cat, my_val in my_stats.items():
                        opp_val = opp_stats.get(cat, 0.0)
                        if lower_better.issuperset({cat}):
                            deficit = my_val - opp_val
                            winning = my_val < opp_val
                        else:
                            deficit = opp_val - my_val
                            winning = my_val > opp_val
                        category_deficits.append(
                            CategoryDeficitOut(
                                category=cat,
                                my_total=my_val,
                                opponent_total=opp_val,
                                deficit=deficit,
                                winning=winning,
                            )
                        )
                    break
        except Exception:
            category_deficits = []

        from backend.fantasy_baseball.player_board import get_or_create_projection as _get_proj

        def _hot_cold_flag(cat_contributions: dict) -> Optional[str]:
            scores = list(cat_contributions.values())
            if not scores:
                return None
            avg = sum(scores) / len(scores)
            if avg > 0.4:
                return "HOT"
            if avg < -0.3:
                return "COLD"
            return None

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
                _translated_stats[_translated_key] = v

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

            need_score = 0.0
            contributions: dict = {}

            if category_deficits:
                cat_scores = board_player.get("cat_scores", {})
                for cd in category_deficits:
                    if cd.winning or cd.deficit <= 0:
                        continue
                    board_key = _YAHOO_CAT_TO_BOARD.get(cd.category)
                    if not board_key or board_key not in cat_scores:
                        continue
                    player_z = cat_scores[board_key]
                    if player_z <= 0:
                        continue
                    opp_total = abs(cd.opponent_total) or 1.0
                    deficit_weight = cd.deficit / opp_total
                    contribution = deficit_weight * player_z
                    contributions[cd.category] = round(contribution, 3)
                    need_score += contribution
            else:
                need_score = board_player.get("z_score", 0.0)

            _hc: Optional[str] = None
            try:
                _hc = _hot_cold_flag(contributions) if contributions else _hot_cold_flag(
                    {k: v for k, v in (board_player.get("cat_scores") or {}).items()}
                )
            except Exception:
                pass

            _status = p.get("status") or None
            _injury_note = p.get("injury_note") or None

            return WaiverPlayerOut(
                player_id=p.get("player_key") or "",
                name=name,
                team=p.get("team") or "",
                position=positions[0] if positions else "?",
                need_score=round(need_score, 3),
                category_contributions=contributions,
                owned_pct=p.get("percent_owned", 0.0),
                starts_this_week=p.get("starts_this_week", 0),
                projected_saves=_raw_nsv,
                hot_cold=_hc,
                status=_status,
                injury_note=_injury_note,
                stats=_translated_stats,
            )

        top_available = [_to_waiver_player(p) for p in free_agents]
        if min_z_score is not None:
            top_available = [p for p in top_available if p.need_score >= min_z_score]
        top_available = [p for p in top_available if p.owned_pct <= max_percent_owned]
        if sort == "percent_owned":
            top_available.sort(key=lambda x: x.owned_pct, reverse=True)
        else:
            top_available.sort(key=lambda x: x.need_score, reverse=True)

        import difflib as _difflib_starts
        from datetime import date as _dt, timedelta as _td
        _today = date_type.today()
        _week_end_ts = _today + _td(days=6)

        def _fetch_mlb_probable_starts(start_date: str, end_date: str) -> dict:
            """Return {pitcher_full_name_lower: starts_count} via public MLB Stats API (6h cached)."""
            import httpx as _httpx
            _now = datetime.utcnow()
            if _STARTS_CACHE.get("data") and _STARTS_CACHE.get("fetched_at"):
                age_h = (_now - _STARTS_CACHE["fetched_at"]).total_seconds() / 3600
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

        starts_map = _fetch_mlb_probable_starts(
            _today.strftime("%Y-%m-%d"), _week_end_ts.strftime("%Y-%m-%d")
        )

        sp_fas = [p for p in free_agents if "SP" in (p.get("positions") or [])]
        two_start_pitchers_raw = []
        for _sp in sp_fas[:50]:
            _sp_name = (_sp.get("name") or "").strip().lower()
            _starts = starts_map.get(_sp_name, 0)
            if _starts == 0 and starts_map:
                _best = max(
                    starts_map.keys(),
                    key=lambda k: _difflib_starts.SequenceMatcher(None, _sp_name, k).ratio(),
                    default=None,
                )
                if _best and _difflib_starts.SequenceMatcher(None, _sp_name, _best).ratio() >= 0.90:
                    _starts = starts_map[_best]
            if _starts >= 2:
                _sp["starts_this_week"] = _starts
                two_start_pitchers_raw.append(_to_waiver_player(_sp))
        two_start_pitchers = sorted(
            two_start_pitchers_raw, key=lambda x: x.need_score, reverse=True
        )[:5]

        _closer_fas = [f for f in top_available if f.category_contributions.get("nsv", 0) > 0.5]
        _closer_alert = None
        if len(_closer_fas) == 0:
            _closer_alert = "NO_CLOSERS"
        elif len(_closer_fas) < 2:
            _closer_alert = "LOW_CLOSERS"

        from backend.services.waiver_edge_detector import il_capacity_info as _il_cap
        _il_info = _il_cap(my_roster) if my_roster else {"used": 0, "total": 2, "available": 0}

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
    )


@router.post("/api/fantasy/waiver/add")
async def add_fantasy_waiver_player(
    add_player_key: str = Query(..., description="Yahoo player key to add (mlb.p.XXXXX)"),
    drop_player_key: Optional[str] = Query(None, description="Yahoo player key to drop (mlb.p.XXXXX)"),
    user: str = Depends(verify_api_key),
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

    return {
        "success": bool(ok),
        "added": add_key,
        "dropped": drop_key,
        "team_key": team_key,
    }


@router.get("/api/fantasy/waiver/recommendations")
async def get_waiver_recommendations(
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Actionable ADD/DROP/ADD_DROP recommendations."""
    from datetime import date as date_type, timedelta
    from backend.schemas import (
        RosterMoveRecommendation, WaiverRecommendationsResponse,
        WaiverPlayerOut, CategoryDeficitOut,
    )
    from backend.fantasy_baseball.player_board import get_or_create_projection as _get_proj
    from backend.fantasy_baseball.statcast_loader import (
        build_statcast_signals, statcast_need_score_boost,
    )

    today = date_type.today()
    week_end = today + timedelta(days=(6 - today.weekday()))

    matchup_opponent = "TBD"
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
            from backend.schemas import CategoryDeficitOut as _CDOut
            matchups = client.get_scoreboard()
            for m in matchups:
                if not isinstance(m, dict):
                    continue
                teams = m.get("teams", {})
                raw_entries = []
                if isinstance(teams, list):
                    raw_entries = [item.get("team", []) for item in teams if isinstance(item, dict)]
                elif isinstance(teams, dict):
                    count_t = int(teams.get("count", 0))
                    raw_entries = [teams.get(str(ti), {}).get("team", []) for ti in range(count_t)]
                team_keys_in_matchup = []
                team_stats: dict = {}
                team_names: dict = {}
                for t_entry in raw_entries:
                    t_meta: dict = {}
                    t_stat_cats: dict = {}
                    if isinstance(t_entry, list):
                        for sub in t_entry:
                            if isinstance(sub, list):
                                for item in sub:
                                    if isinstance(item, dict):
                                        t_meta.update(item)
                            elif isinstance(sub, dict):
                                t_meta.update(sub)
                                if "team_stats" in sub:
                                    stats_block = sub["team_stats"].get("stats", [])
                                    for s_entry in stats_block:
                                        if isinstance(s_entry, dict):
                                            s = s_entry.get("stat", {})
                                            t_stat_cats[s.get("stat_id")] = s.get("value")
                    tk = t_meta.get("team_key", "")
                    tn = t_meta.get("name", "")
                    team_keys_in_matchup.append(tk)
                    team_stats[tk] = t_stat_cats
                    team_names[tk] = tn
                if my_team_key in team_keys_in_matchup:
                    for tk in team_keys_in_matchup:
                        if tk != my_team_key:
                            matchup_opponent = team_names.get(tk, "TBD")
                    break
        except Exception:
            pass

        free_agents = client.get_free_agents(count=40)

        def _score_fa(p: dict) -> WaiverPlayerOut:
            positions = p.get("positions") or []
            name = (p.get("name") or "").strip()
            bp = _get_proj(p)
            need_score = bp.get("z_score", 0.0)
            return WaiverPlayerOut(
                player_id=p.get("player_key") or "",
                name=name,
                team=p.get("team") or "",
                position=positions[0] if positions else "?",
                need_score=round(need_score, 3),
                category_contributions=bp.get("cat_scores", {}) if bp else {},
                owned_pct=p.get("percent_owned", 0.0),
                starts_this_week=0,
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

        my_roster_scored: list = []
        for rp in my_roster:
            bp = _get_proj(rp)
            my_roster_scored.append({
                "name": (rp.get("name") or "").strip(),
                "player_key": rp.get("player_key") or "",
                "positions": rp.get("positions") or [],
                "z_score": bp.get("z_score", 0.0),
                "is_proxy": bp.get("is_proxy", False),
                "cat_scores": bp.get("cat_scores") or {},
                "starts_this_week": int(rp.get("starts_this_week", 1)),
                "status": rp.get("status"),
                "injury_note": rp.get("injury_note"),
                "is_undroppable": bool(rp.get("is_undroppable", 0)),
            })

        _IL_STATUSES = {"IL", "IL10", "IL60", "NA", "OUT"}

        def _weakest_safe_to_drop(target_positions: list) -> Optional[dict]:
            candidates = [
                rp for rp in my_roster_scored
                if not rp.get("is_undroppable", False)
                and any(pos in rp["positions"] for pos in target_positions)
            ]
            if not candidates:
                return None
            active = [p for p in candidates if p.get("status") not in _IL_STATUSES]
            if len(active) == 1:
                return None
            if len(active) == 0:
                return min(candidates, key=lambda x: x.get("z_score") or 0.0)
            return min(active, key=lambda x: x.get("z_score") or 0.0)

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

        for fa in scored_fas[:15]:
            if len(recommendations) >= 5:
                break

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
                        category_targets=[],
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
            drop_score_adj = drop_candidate["z_score"] + statcast_need_score_boost(drop_signals)

            if drop_score_adj >= adjusted_need:
                continue

            gain = adjusted_need - drop_candidate["z_score"]
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
                f"Net gain: {gain:+.1f} ({drop_candidate['z_score']:+.1f} -> {adjusted_need:+.1f}){signal_text}{drop_signal_text}."
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
                _mcmc = _sim_move(
                    my_roster=my_roster_scored,
                    opponent_roster=[],
                    add_player=_add_for_mcmc,
                    drop_player_name=drop_candidate["name"],
                    n_sims=1000,
                )
                if _mcmc.get("mcmc_enabled") and abs(_mcmc["win_prob_gain"]) >= 0.005:
                    wp_before_pct = round(_mcmc["win_prob_before"] * 100)
                    wp_after_pct = round(_mcmc["win_prob_after"] * 100)
                    wp_gain_pct = round(_mcmc["win_prob_gain"] * 100)
                    rationale += (
                        f" Win prob: {wp_before_pct}% -> {wp_after_pct}%"
                        f" ({wp_gain_pct:+d}%)."
                    )
            except Exception:
                pass

            recommendations.append(RosterMoveRecommendation(
                action="ADD_DROP",
                add_player=fa,
                drop_player_name=drop_candidate["name"],
                drop_player_position=drop_candidate["positions"][0] if drop_candidate["positions"] else "?",
                rationale=rationale,
                category_targets=[],
                need_score=round(gain, 3),
                confidence=confidence,
                statcast_signals=fa_signals,
                regression_delta=fa_reg_delta,
                win_prob_before=_mcmc.get("win_prob_before", 0.0),
                win_prob_after=_mcmc.get("win_prob_after", 0.0),
                win_prob_gain=_mcmc.get("win_prob_gain", 0.0),
                category_win_probs=_mcmc.get("category_win_probs_after", {}),
                mcmc_enabled=_mcmc.get("mcmc_enabled", False),
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

    return WaiverRecommendationsResponse(
        week_end=week_end,
        matchup_opponent=matchup_opponent,
        recommendations=sorted(recommendations, key=lambda r: r.need_score, reverse=True),
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


@router.get("/api/fantasy/roster", response_model=RosterResponse)
async def get_fantasy_roster(user: str = Depends(verify_api_key)):
    """Return the authenticated user's current Yahoo roster enriched with z-scores."""
    from backend.fantasy_baseball.player_board import get_or_create_projection

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

    players_map: dict = {}
    for p in raw_players:
        player_key = p.get("player_key") or ""
        if not player_key:
            continue
        name = p.get("name") or ""
        proj = get_or_create_projection(p) if name else {}
        raw_status = p.get("status")
        if isinstance(raw_status, bool):
            status_str = "Active" if raw_status else "Inactive"
        else:
            status_str = raw_status if raw_status else None

        injury_note = p.get("injury_note")
        if isinstance(injury_note, bool):
            injury_note = None

        players_map[player_key] = RosterPlayerOut(
            player_key=player_key,
            name=name,
            team=p.get("team"),
            positions=p.get("positions") or [],
            status=status_str,
            injury_note=injury_note if injury_note else None,
            injury_status=status_str,
            z_score=proj.get("z_score"),
            is_undroppable=bool(p.get("is_undroppable", 0)),
            is_proxy=bool(proj.get("is_proxy", False)),
            cat_scores=proj.get("cat_scores") or {},
            selected_position=p.get("selected_position"),
        )
    players_out = list(players_map.values())

    return RosterResponse(
        team_key=team_key,
        players=players_out,
        count=len(players_out),
    )


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
    """Return current week's matchup: opponent name + category-by-category breakdown."""
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

        _PITCHER_RENAME = {"HR": "HRA", "K": "K(P)"}
        _BATTER_RENAME = {"K": "K(B)", "HR": "HR"}

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
    except Exception as _e:
        logger.warning("get_league_settings failed, using fallback stat_id_map: %s", _e)

    if not active_stat_abbrs and LEAGUE_SCORING_CATEGORIES:
        active_stat_abbrs = set(LEAGUE_SCORING_CATEGORIES)

    try:
        matchups = client.get_scoreboard()
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
            if t[0] == my_team_key:
                my_entry = t
                break
            if t[0] and my_team_key and (t[0] in my_team_key or my_team_key in t[0]):
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

        return MatchupResponse(
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

    return MatchupResponse(week=week, my_team=_stub_my, opponent=_stub_opp, message="Your team was not found in the current week's matchup.")


# ============================================================================
# DASHBOARD
# ============================================================================

from backend.services.dashboard_service import get_dashboard_service


@router.get("/api/dashboard")
async def get_dashboard(user: str = Depends(verify_api_key)):
    """Phase B: Enhanced Dashboard"""
    service = get_dashboard_service()
    dashboard = await service.get_dashboard(user_id=user)
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

@router.post("/api/fantasy/matchup/simulate")
async def simulate_matchup(
    payload: MatchupSimulateRequest,
    user: str = Depends(verify_api_key),
):
    """Monte Carlo simulation of a weekly H2H matchup."""
    from backend.fantasy_baseball.mcmc_simulator import simulate_weekly_matchup
    n = min(max(100, payload.n_sims), 5000)
    result = simulate_weekly_matchup(
        my_roster=payload.my_roster,
        opponent_roster=payload.opponent_roster,
        n_sims=n,
    )
    return result


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

    # Build query
    query = db.query(PlayerScore).filter(
        PlayerScore.bdl_player_id == bdl_player_id,
        PlayerScore.window_days == window_days,
    )

    # Resolve as_of_date
    if as_of_date is None:
        # Default to latest available score for this player/window
        latest = query.order_by(PlayerScore.as_of_date.desc()).first()
        if not latest:
            raise HTTPException(
                status_code=404,
                detail=f"No player_scores found for bdl_player_id={bdl_player_id} window_days={window_days}",
            )
        as_of_date = latest.as_of_date
    else:
        # Query for specific date
        latest = query.filter(PlayerScore.as_of_date == as_of_date).first()
        if not latest:
            raise HTTPException(
                status_code=404,
                detail=f"No player_scores found for bdl_player_id={bdl_player_id} window_days={window_days} as_of_date={as_of_date}",
            )

    # Build category_scores
    category_scores = PlayerScoreCategoryBreakdown(
        z_hr=latest.z_hr,
        z_rbi=latest.z_rbi,
        z_nsb=latest.z_nsb,
        z_avg=latest.z_avg,
        z_obp=latest.z_obp,
        z_era=latest.z_era,
        z_whip=latest.z_whip,
        z_k_per_9=latest.z_k_per_9,
    )

    # Build score output
    score_out = PlayerScoreOut(
        bdl_player_id=latest.bdl_player_id,
        as_of_date=latest.as_of_date,
        window_days=latest.window_days,
        player_type=latest.player_type,
        games_in_window=latest.games_in_window,
        composite_z=latest.composite_z,
        score_0_100=latest.score_0_100,
        confidence=latest.confidence,
        category_scores=category_scores,
    )

    return PlayerScoresResponse(
        bdl_player_id=bdl_player_id,
        requested_window_days=window_days,
        as_of_date=as_of_date,
        score=score_out,
    )
