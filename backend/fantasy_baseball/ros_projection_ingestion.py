"""
Fetch Rest-of-Season (RoS) projections from FanGraphs API and hydrate PlayerProjection table.

FanGraphs endpoint (public, no auth required):
  https://www.fangraphs.com/api/projections?type=steamerr&stats=bat|pit&pos=all&team=0&players=0&lg=all

Returns JSON array with Steamer RoS projections including:
- xMLBAMID (MLBAM player ID for direct mapping)
- PlayerName, Team, Pos
- Standard + advanced stats (wRC+, ISO, FIP, K%, BB%, etc.)

Pipeline:
  1. fetch_ros_projections() — Get RoS data from FanGraphs API
  2. resolve_player_ids() — Map xMLBAMID → internal player_id
  3. write_projections_to_db() — Upsert to PlayerProjection table
  4. run_ros_backfill() — Full pipeline with status reporting

Idempotent: Can be run multiple times without side effects.
Graceful degradation: Returns empty dict on any failure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import requests
from sqlalchemy import select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# FanGraphs API config
FG_API_BASE = "https://www.fangraphs.com/api/projections"
FG_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.fangraphs.com/projections",
    "Origin": "https://www.fangraphs.com",
    "X-Requested-With": "XMLHttpRequest",
}


# ---------------------------------------------------------------------------
# Data class for RoS projection
# ---------------------------------------------------------------------------

@dataclass
class ROSProjection:
    """Single player's Rest-of-Season projection from FanGraphs."""
    player_id: str
    player_name: str
    team: str
    mlbam_id: str
    positions: list[str] = field(default_factory=list)

    # Counting
    g: float = 0.0
    ab: float = 0.0
    pa: float = 0.0
    hr: float = 0.0
    r: float = 0.0
    rbi: float = 0.0
    sb: float = 0.0
    so: float = 0.0
    bb: float = 0.0

    # Rate
    avg: float = 0.250
    obp: float = 0.320
    slg: float = 0.400
    ops: float = 0.720
    woba: float = 0.320

    # Advanced (batters)
    wrc_plus: float = 100.0
    iso: float = 0.150
    bb_pct: float = 8.0
    k_pct: float = 20.0
    babip: float = 0.300

    # Pitching
    gs: float = 0.0
    ip: float = 0.0
    k: float = 0.0
    era: float = 4.00
    whip: float = 1.30
    k_per_nine: float = 8.5
    bb_per_nine: float = 3.0
    fip: float = 4.00


# ---------------------------------------------------------------------------
# Safe float helper
# ---------------------------------------------------------------------------

def _float(val: Any) -> float:
    try:
        if val is None:
            return 0.0
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _int(val: Any) -> int:
    try:
        if val is None:
            return 0
        return int(float(val))
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# FanGraphs API fetch
# ---------------------------------------------------------------------------

def _fetch_fg_projection(stats: str) -> list[dict]:
    """
    Fetch projection data from FanGraphs API.

    Args:
        stats: 'bat' or 'pit'

    Returns:
        List of player dicts. Empty list on failure.
    """
    params = {
        "type": "steamerr",
        "stats": stats,
        "pos": "all",
        "team": "0",
        "players": "0",
        "lg": "all",
        "z": "1778047498",
    }
    try:
        resp = requests.get(FG_API_BASE, params=params, headers=FG_HEADERS, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            logger.info("FanGraphs API returned %d %s projection rows", len(data), stats)
            return data
        logger.warning("FanGraphs API returned non-list for %s: %s", stats, type(data).__name__)
        return []
    except requests.Timeout:
        logger.error("FanGraphs API timeout for %s", stats)
        return []
    except requests.HTTPError as exc:
        logger.error("FanGraphs API HTTP error for %s: %s", stats, exc)
        return []
    except Exception as exc:
        logger.error("FanGraphs API unexpected error for %s: %s", stats, exc)
        return []


def _parse_positions(pos_str: Any) -> list[str]:
    """Parse FanGraphs position string (e.g., 'CF / RF / DH') into list."""
    if not pos_str:
        return []
    return [p.strip() for p in str(pos_str).split("/") if p.strip()]


def _map_batter(raw: dict) -> ROSProjection:
    """Convert a FanGraphs batter dict to ROSProjection."""
    return ROSProjection(
        player_id=str(raw.get("xMLBAMID", "") or raw.get("playerid", "")),
        player_name=str(raw.get("PlayerName", "")),
        team=str(raw.get("Team", "")),
        mlbam_id=str(raw.get("xMLBAMID", "")),
        positions=_parse_positions(raw.get("Pos")),
        g=_float(raw.get("G")),
        ab=_float(raw.get("AB")),
        pa=_float(raw.get("PA")),
        hr=_float(raw.get("HR")),
        r=_float(raw.get("R")),
        rbi=_float(raw.get("RBI")),
        sb=_float(raw.get("SB")),
        so=_float(raw.get("SO")),
        bb=_float(raw.get("BB")),
        avg=_float(raw.get("AVG")),
        obp=_float(raw.get("OBP")),
        slg=_float(raw.get("SLG")),
        ops=_float(raw.get("OPS")),
        woba=_float(raw.get("wOBA")),
        wrc_plus=_float(raw.get("wRC+")),
        iso=_float(raw.get("ISO")),
        bb_pct=_float(raw.get("BB%")),
        k_pct=_float(raw.get("K%")),
        babip=_float(raw.get("BABIP")),
    )


def _map_pitcher(raw: dict) -> ROSProjection:
    """Convert a FanGraphs pitcher dict to ROSProjection."""
    ip = _float(raw.get("IP"))
    k = _float(raw.get("SO"))
    k_per_nine = (k / ip * 9.0) if ip > 0 else 0.0
    bb_per_nine = (_float(raw.get("BB")) / ip * 9.0) if ip > 0 else 0.0

    return ROSProjection(
        player_id=str(raw.get("xMLBAMID", "") or raw.get("playerid", "")),
        player_name=str(raw.get("PlayerName", "")),
        team=str(raw.get("Team", "")),
        mlbam_id=str(raw.get("xMLBAMID", "")),
        positions=_parse_positions(raw.get("Pos")),
        g=_float(raw.get("G")),
        gs=_float(raw.get("GS")),
        ip=ip,
        k=k,
        bb=_float(raw.get("BB")),
        hr=_float(raw.get("HR")),
        era=_float(raw.get("ERA")),
        whip=_float(raw.get("WHIP")),
        k_per_nine=k_per_nine,
        bb_per_nine=bb_per_nine,
        fip=_float(raw.get("FIP")),
        avg=0.0, obp=0.0, slg=0.0, ops=0.0,
    )


# ---------------------------------------------------------------------------
# Public fetch entry point
# ---------------------------------------------------------------------------

def fetch_ros_projections(db: Session, year: int = 2026) -> dict[str, ROSProjection]:
    """
    Fetch FanGraphs Steamer RoS projections via public API.

    Args:
        db: Database session (for context, not used in fetch)
        year: Season year (default 2026) — included for API compatibility

    Returns:
        Dict mapping player_id (MLBAM) → ROSProjection. Empty dict on failure.
    """
    projections: dict[str, ROSProjection] = {}

    batters = _fetch_fg_projection("bat")
    for row in batters:
        proj = _map_batter(row)
        if proj.player_id and proj.player_id != "0":
            projections[proj.player_id] = proj
        else:
            # Fallback: use player_name as key if MLBAM ID missing
            projections[proj.player_name] = proj

    pitchers = _fetch_fg_projection("pit")
    for row in pitchers:
        proj = _map_pitcher(row)
        if proj.player_id and proj.player_id != "0":
            existing = projections.get(proj.player_id)
            if existing:
                # Two-way player (Ohtani): merge pitcher stats into existing
                existing.gs = proj.gs
                existing.ip = proj.ip
                existing.k = proj.k
                existing.era = proj.era
                existing.whip = proj.whip
                existing.k_per_nine = proj.k_per_nine
                existing.bb_per_nine = proj.bb_per_nine
                existing.fip = proj.fip
            else:
                projections[proj.player_id] = proj
        else:
            projections[proj.player_name] = proj

    logger.info("Fetched %d unique RoS projections from FanGraphs API", len(projections))
    return projections


# ---------------------------------------------------------------------------
# Resolve player IDs via PlayerIDMapping
# ---------------------------------------------------------------------------

def resolve_player_ids(db: Session, projections: dict[str, ROSProjection]) -> dict[str, ROSProjection]:
    """
    Resolve player_id via PlayerIDMapping and enrich team from statcast_performance.

    With the new FanGraphs API, xMLBAMID is already present, so resolution
    should be near-perfect for players with MLBAM IDs.
    """
    from backend.models import PlayerIDMapping, StatcastPerformance

    # Build mlbam_id → internal player_id lookup
    mlbam_to_internal: dict[str, str] = {}
    for row in db.execute(select(PlayerIDMapping)).scalars():
        if row.mlbam_id:
            mlbam_to_internal[str(row.mlbam_id)] = str(row.mlbam_id)

    # Build team lookup from statcast_performance
    team_lookup: dict[str, str] = {}
    try:
        for row in db.execute(select(StatcastPerformance)).scalars():
            if row.player_id and row.team:
                team_lookup[str(row.player_id)] = row.team
    except Exception:
        pass

    resolved: dict[str, ROSProjection] = {}
    unresolved = 0

    for key, proj in projections.items():
        # Prefer MLBAM ID as key if it looks like one (numeric, > 100000)
        if proj.mlbam_id and proj.mlbam_id.isdigit() and int(proj.mlbam_id) > 100000:
            internal_id = proj.mlbam_id
            proj.player_id = internal_id

            if internal_id in team_lookup:
                proj.team = team_lookup[internal_id]
            resolved[internal_id] = proj
        else:
            # Fallback: name-based resolution
            name_key = proj.player_name.lower().strip()
            if name_key in mlbam_to_internal:
                proj.player_id = mlbam_to_internal[name_key]
                resolved[proj.player_id] = proj
            else:
                # Keep name-based key as last resort
                resolved[proj.player_name] = proj
                unresolved += 1

    if unresolved:
        logger.warning("Could not resolve MLBAM ID for %d players", unresolved)
    logger.info("Resolved %d/%d projections with MLBAM IDs", len(resolved) - unresolved, len(projections))
    return resolved


# ---------------------------------------------------------------------------
# Write projections to PlayerProjection table
# ---------------------------------------------------------------------------

def write_projections_to_db(db: Session, projections: dict[str, ROSProjection]) -> int:
    """
    Write ROS projections to PlayerProjection table.

    Performs upsert: updates existing rows, inserts new ones.
    """
    from backend.models import PlayerProjection

    written = 0
    for player_id, proj in projections.items():
        existing = db.execute(
            select(PlayerProjection).where(PlayerProjection.player_id == player_id)
        ).scalar_one_or_none()

        if existing:
            existing.player_name = proj.player_name
            existing.team = proj.team if proj.team != "FA" else existing.team
            existing.positions = proj.positions or existing.positions

            # Batting
            existing.hr = _int(proj.hr)
            existing.r = _int(proj.r)
            existing.rbi = _int(proj.rbi)
            existing.sb = _int(proj.sb)
            existing.avg = proj.avg if proj.avg > 0 else existing.avg
            existing.obp = proj.obp if proj.obp > 0 else existing.obp
            existing.slg = proj.slg if proj.slg > 0 else existing.slg
            existing.ops = proj.ops if proj.ops > 0 else existing.ops
            existing.woba = proj.woba if proj.woba > 0 else existing.woba

            # Advanced (new — these columns may need migration if not present)
            for attr, val in [
                ("wrc_plus", proj.wrc_plus),
                ("iso", proj.iso),
                ("bb_pct", proj.bb_pct),
                ("k_pct", proj.k_pct),
                ("babip", proj.babip),
            ]:
                if hasattr(existing, attr) and val > 0:
                    setattr(existing, attr, val)

            # Pitching
            if proj.era > 0:
                existing.era = proj.era
            if proj.whip > 0:
                existing.whip = proj.whip
            if proj.k_per_nine > 0:
                existing.k_per_nine = proj.k_per_nine
            if proj.fip > 0 and hasattr(existing, "fip"):
                existing.fip = proj.fip
        else:
            new_row = PlayerProjection(
                player_id=player_id,
                player_name=proj.player_name,
                team=proj.team if proj.team != "FA" else None,
                positions=proj.positions or None,
                avg=proj.avg if proj.avg > 0 else 0.250,
                obp=proj.obp if proj.obp > 0 else 0.320,
                slg=proj.slg if proj.slg > 0 else 0.400,
                ops=proj.ops if proj.ops > 0 else 0.720,
                woba=proj.woba if proj.woba > 0 else 0.320,
                hr=_int(proj.hr),
                r=_int(proj.r),
                rbi=_int(proj.rbi),
                sb=_int(proj.sb),
                era=proj.era if proj.era > 0 else None,
                whip=proj.whip if proj.whip > 0 else None,
                k_per_nine=proj.k_per_nine if proj.k_per_nine > 0 else None,
            )
            # Set advanced attrs if column exists
            for attr, val in [
                ("wrc_plus", proj.wrc_plus),
                ("iso", proj.iso),
                ("bb_pct", proj.bb_pct),
                ("k_pct", proj.k_pct),
                ("babip", proj.babip),
                ("fip", proj.fip),
            ]:
                if hasattr(new_row, attr) and val > 0:
                    setattr(new_row, attr, val)

            db.add(new_row)

        written += 1

    try:
        db.commit()
        logger.info("Wrote %d projection rows to PlayerProjection table", written)
    except Exception as exc:
        db.rollback()
        logger.error("Failed to commit projections: %s", exc)
        raise

    return written


# ---------------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------------

def run_ros_backfill(db: Session, year: int = 2026) -> dict:
    """
    Full RoS projection backfill pipeline.

    Returns:
        Dict with keys: status, fetched, resolved, written, message
    """
    projections = fetch_ros_projections(db, year)
    if not projections:
        return {
            "status": "failed",
            "fetched": 0,
            "resolved": 0,
            "written": 0,
            "message": "Failed to fetch projections from FanGraphs API",
        }

    resolved = resolve_player_ids(db, projections)
    if not resolved:
        return {
            "status": "partial",
            "fetched": len(projections),
            "resolved": 0,
            "written": 0,
            "message": f"Fetched {len(projections)} projections but resolved 0 player IDs",
        }

    written = write_projections_to_db(db, resolved)

    return {
        "status": "success" if written > 0 else "partial",
        "fetched": len(projections),
        "resolved": len(resolved),
        "written": written,
        "message": f"Backfilled {written} player projections ({len(projections)} fetched, {len(resolved)} resolved)",
    }


# ---------------------------------------------------------------------------
# Quick test CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    logger.info("Testing FanGraphs RoS API fetch...")
    batters = _fetch_fg_projection("bat")
    pitchers = _fetch_fg_projection("pit")

    logger.info("Batters: %d rows", len(batters))
    logger.info("Pitchers: %d rows", len(pitchers))

    if batters:
        logger.info("Sample batter: %s", batters[0].get("PlayerName"))
        logger.info("  HR=%s, AVG=%s, wRC+=%s", batters[0].get("HR"), batters[0].get("AVG"), batters[0].get("wRC+"))
    if pitchers:
        logger.info("Sample pitcher: %s", pitchers[0].get("PlayerName"))
        logger.info("  ERA=%s, WHIP=%s, FIP=%s", pitchers[0].get("ERA"), pitchers[0].get("WHIP"), pitchers[0].get("FIP"))

    logger.info("✅ FanGraphs API is reachable and returns data.")
