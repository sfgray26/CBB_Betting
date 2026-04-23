"""
Fetch Rest-of-Season (RoS) projections from FanGraphs and hydrate PlayerProjection table.

Uses pybaseball batting_stats() and pitching_stats() with season="rostest"
for Steamer RoS projections. Maps player names to PlayerIDMapping for
internal player_id resolution, and enriches team from statcast_performance.

Pipeline:
  1. fetch_ros_projections() — Get RoS data from FanGraphs
  2. resolve_player_ids() — Map names → player_id via PlayerIDMapping
  3. write_projections_to_db() — Upsert to PlayerProjection table
  4. run_ros_backfill() — Full pipeline with status reporting

Idempotent: Can be run multiple times without side effects.
Graceful degradation: Returns empty dict on any failure.
"""

import dataclasses
import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# User-Agent patch import (ensures pybaseball uses browser headers)
# ---------------------------------------------------------------------------

def _ensure_user_agent() -> None:
    """Ensure pybaseball session has browser User-Agent to avoid 403s."""
    try:
        from backend.fantasy_baseball.pybaseball_loader import _patch_pybaseball_user_agent
        _patch_pybaseball_user_agent()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Data class for RoS projection
# ---------------------------------------------------------------------------

@dataclass
class ROSProjection:
    """Single player's Rest-of-Season projection."""
    player_id: str
    player_name: str
    team: str
    positions: list[str]
    hr: float
    r: float
    rbi: float
    sb: float
    w: float
    k_pit: float
    qs: float
    # Additional stats for completeness
    avg: float = 0.250
    obp: float = 0.320
    slg: float = 0.400
    ops: float = 0.720
    era: float = 4.00
    whip: float = 1.30


# ---------------------------------------------------------------------------
# Fetch RoS projections from FanGraphs
# ---------------------------------------------------------------------------

def fetch_ros_projections(db: Session, year: int = 2026) -> dict[str, ROSProjection]:
    """
    Fetch FanGraphs RoS projections and return dict[player_name, ROSProjection].

    Args:
        db: Database session (for context, not used in fetch)
        year: Season year (default 2026)

    Returns:
        Dict mapping player name to ROSProjection. Empty dict on failure.

    Note:
        Uses season="rostest" which returns Steamer RoS projections.
        Falls back to empty dict on any failure (graceful degradation).
    """
    _ensure_user_agent()

    try:
        import pybaseball
    except ImportError:
        logger.warning("pybaseball not installed -- skipping RoS projection fetch")
        return {}

    try:
        pybaseball.cache.enable()
    except Exception:
        pass

    projections: dict[str, ROSProjection] = {}

    # Fetch RoS batting stats
    try:
        df_batting = pybaseball.batting_stats(year, season="rostest", qual=10)
        logger.info("pybaseball: fetched %d RoS batting lines", len(df_batting))

        for _, row in df_batting.iterrows():
            name = str(row.get("Name") or "").strip()
            if not name:
                continue

            projections[name] = ROSProjection(
                player_id="",  # Will resolve via PlayerIDMapping
                player_name=name,
                team=str(row.get("Team") or "FA").replace("-", ""),  # Handle "-TM" for free agents
                positions=[],
                hr=_float(row.get("HR", 15)),
                r=_float(row.get("R", 65)),
                rbi=_float(row.get("RBI", 65)),
                sb=_float(row.get("SB", 10)),
                w=0, k_pit=0, qs=0,  # Batter defaults
                avg=_float(row.get("AVG", 0.250)),
                obp=_float(row.get("OBP", 0.320)),
                slg=_float(row.get("SLG", 0.400)),
                ops=_float(row.get("OPS", 0.720)),
            )
    except Exception as e:
        logger.error("RoS batting fetch failed: %s", e)

    # Fetch RoS pitching stats
    try:
        df_pitching = pybaseball.pitching_stats(year, season="rostest", qual=5)
        logger.info("pybaseball: fetched %d RoS pitching lines", len(df_pitching))

        for _, row in df_pitching.iterrows():
            name = str(row.get("Name") or "").strip()
            if not name:
                continue

            games_started = _float(row.get("GS", 30))

            # Merge if already exists (multi-pos players like Ohtani)
            if name in projections:
                proj = projections[name]
                proj.w = _float(row.get("W", 10))
                proj.k_pit = _float(row.get("SO", 150))
                proj.qs = games_started * 0.6  # Approximate QS from GS
                proj.era = _float(row.get("ERA", 4.00))
                proj.whip = _float(row.get("WHIP", 1.30))
            else:
                projections[name] = ROSProjection(
                    player_id="",
                    player_name=name,
                    team=str(row.get("Team") or "FA").replace("-", ""),
                    positions=[],
                    hr=0, r=0, rbi=0, sb=0,  # Pitcher defaults
                    w=_float(row.get("W", 10)),
                    k_pit=_float(row.get("SO", 150)),
                    qs=games_started * 0.6,
                    avg=0, obp=0, slg=0, ops=0,
                    era=_float(row.get("ERA", 4.00)),
                    whip=_float(row.get("WHIP", 1.30)),
                )
    except Exception as e:
        logger.error("RoS pitching fetch failed: %s", e)

    logger.info("Fetched %d unique RoS projections from FanGraphs", len(projections))
    return projections


def _float(val) -> float:
    """Safe float conversion with default to 0."""
    try:
        return float(val or 0)
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Resolve player IDs via PlayerIDMapping
# ---------------------------------------------------------------------------

def resolve_player_ids(db: Session, projections: dict[str, ROSProjection]) -> dict[str, ROSProjection]:
    """
    Resolve player_id via PlayerIDMapping and enrich team from statcast_performance.

    Args:
        db: Database session
        projections: Dict of player_name -> ROSProjection

    Returns:
        Dict of player_id -> ROSProjection for matches found in PlayerIDMapping.
    """
    from backend.models import PlayerIDMapping, StatcastPerformance

    # Build name → ID lookup from PlayerIDMapping
    name_to_id = {}
    for row in db.execute(select(PlayerIDMapping)).scalars():
        key = row.player_name.lower().strip()
        name_to_id[key] = row.player_id

    # Build team lookup from statcast_performance (more accurate team data)
    team_lookup = {}
    try:
        for row in db.execute(select(StatcastPerformance)).scalars():
            team_lookup[row.player_name.lower()] = row.team
    except Exception:
        pass  # statcast_performance may not exist

    resolved = {}
    for name, proj in projections.items():
        key = name.lower().strip()
        if key in name_to_id:
            proj.player_id = name_to_id[key]
            if key in team_lookup:
                proj.team = team_lookup[key]
            resolved[proj.player_id] = proj

    logger.info("Resolved %d/%d player IDs via PlayerIDMapping", len(resolved), len(projections))
    return resolved


# ---------------------------------------------------------------------------
# Write projections to PlayerProjection table
# ---------------------------------------------------------------------------

def write_projections_to_db(db: Session, projections: dict[str, ROSProjection]) -> int:
    """
    Write ROS projections to PlayerProjection table.

    Performs upsert: updates existing rows, inserts new ones.

    Args:
        db: Database session
        projections: Dict of player_id -> ROSProjection

    Returns:
        Count of rows written (updated + inserted).
    """
    from backend.models import PlayerProjection

    written = 0
    for player_id, proj in projections.items():
        # Check existing row
        existing = db.execute(
            select(PlayerProjection).where(PlayerProjection.player_id == player_id)
        ).scalar_one_or_none()

        if existing:
            # Update counting stats with real projections
            existing.hr = int(proj.hr)
            existing.r = int(proj.r)
            existing.rbi = int(proj.rbi)
            existing.sb = int(proj.sb)
            existing.avg = proj.avg if proj.avg > 0 else existing.avg
            existing.obp = proj.obp if proj.obp > 0 else existing.obp
            existing.slg = proj.slg if proj.slg > 0 else existing.slg
            existing.ops = proj.ops if proj.ops > 0 else existing.ops

            # Update pitcher stats if applicable
            if proj.w > 0:
                existing.w = int(proj.w)
                existing.k_pit = int(proj.k_pit)
                existing.qs = int(proj.qs)
                existing.era = proj.era if proj.era > 0 else existing.era
                existing.whip = proj.whip if proj.whip > 0 else existing.whip

            # Update metadata
            if proj.team and proj.team != "FA":
                existing.team = proj.team
            if proj.positions:
                existing.positions = proj.positions
        else:
            # Insert new row
            new_row = PlayerProjection(
                player_id=player_id,
                player_name=proj.player_name,
                team=proj.team if proj.team != "FA" else None,
                positions=proj.positions or None,
                # Rate stats
                avg=proj.avg if proj.avg > 0 else 0.250,
                obp=proj.obp if proj.obp > 0 else 0.320,
                slg=proj.slg if proj.slg > 0 else 0.400,
                ops=proj.ops if proj.ops > 0 else 0.720,
                # Counting stats
                hr=int(proj.hr),
                r=int(proj.r),
                rbi=int(proj.rbi),
                sb=int(proj.sb),
                # Pitcher stats
                w=int(proj.w) if proj.w > 0 else None,
                k_pit=int(proj.k_pit) if proj.k_pit > 0 else None,
                qs=int(proj.qs) if proj.qs > 0 else None,
                era=proj.era if proj.era > 0 else None,
                whip=proj.whip if proj.whip > 0 else None,
            )
            db.add(new_row)

        written += 1

    try:
        db.commit()
        logger.info("Wrote %d projection rows to PlayerProjection table", written)
    except Exception as e:
        db.rollback()
        logger.error("Failed to commit projections: %s", e)
        raise

    return written


# ---------------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------------

def run_ros_backfill(db: Session, year: int = 2026) -> dict:
    """
    Full RoS projection backfill pipeline.

    Executes: fetch → resolve → write

    Returns:
        Dict with keys:
            status: "success" | "partial" | "failed"
            fetched: int (projections from FanGraphs)
            resolved: int (player IDs matched)
            written: int (rows written to database)
            message: str (human-readable summary)

    Example:
        >>> result = run_ros_backfill(db)
        >>> print(result)
        {'status': 'success', 'fetched': 850, 'resolved': 420, 'written': 420, 'message': '...'}
    """
    projections = fetch_ros_projections(db, year)
    if not projections:
        return {
            "status": "failed",
            "fetched": 0,
            "resolved": 0,
            "written": 0,
            "message": "Failed to fetch projections from FanGraphs"
        }

    resolved = resolve_player_ids(db, projections)
    if not resolved:
        return {
            "status": "partial",
            "fetched": len(projections),
            "resolved": 0,
            "written": 0,
            "message": f"Fetched {len(projections)} projections but resolved 0 player IDs"
        }

    written = write_projections_to_db(db, resolved)

    return {
        "status": "success" if written > 0 else "partial",
        "fetched": len(projections),
        "resolved": len(resolved),
        "written": written,
        "message": f"Backfilled {written} player projections ({len(projections)} fetched, {len(resolved)} resolved)"
    }
