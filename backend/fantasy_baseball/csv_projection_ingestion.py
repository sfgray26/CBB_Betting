"""
CSV-based projection ingestion as fallback for FanGraphs 403 issues.

When FanGraphs blocks automated access (403 Forbidden), we can still hydrate
PlayerProjection with real data by:

1. Manually downloading projection CSVs from FanGraphs/Steamer
2. Saving them to data/projections/
3. Running this ingestion module

This provides a resilient fallback path while the 403 issue is investigated.

Usage:
    python -m backend.fantasy_baseball.csv_projection_ingestion

Expected CSV format (FanGraphs export):
    Name,Team,PA,HR,R,RBI,SB,AVG,OBP,SLG,OPS,W,SO,ERA,WHIP,GS
    (Standard Fangraphs batting/pitching export columns)
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class CSVProjection:
    """Single player's projection from CSV."""
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
    avg: float = 0.250
    obp: float = 0.320
    slg: float = 0.400
    ops: float = 0.720
    era: float = 4.00
    whip: float = 1.30


def load_projections_from_csv(csv_path: Path) -> dict[str, CSVProjection]:
    """
    Load projections from a CSV file (FanGraphs export format).

    Args:
        csv_path: Path to CSV file with FanGraphs projection data

    Returns:
        Dict mapping player_name -> CSVProjection
    """
    projections = {}

    if not csv_path.exists():
        logger.warning(f"CSV file not found: {csv_path}")
        return {}

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                name = str(row.get('Name') or '').strip()
                if not name:
                    continue

                # Parse counting stats
                hr = _float(row.get('HR', 15))
                r = _float(row.get('R', 65))
                rbi = _float(row.get('RBI', 65))
                sb = _float(row.get('SB', 10))

                # Parse rate stats
                avg = _float(row.get('AVG', 0.250))
                obp = _float(row.get('OBP', 0.320))
                slg = _float(row.get('SLG', 0.400))
                ops = _float(row.get('OPS', 0.720))

                # Pitcher stats (may be empty for batters)
                w = _float(row.get('W', 0))
                k_pit = _float(row.get('SO', 0))
                era = _float(row.get('ERA', 4.00))
                whip = _float(row.get('WHIP', 1.30))
                games_started = _float(row.get('GS', 0))
                qs = games_started * 0.6  # Approximate

                team = str(row.get('Team', 'FA')).replace('-', '').strip()

                projections[name] = CSVProjection(
                    player_id="",  # Will resolve
                    player_name=name,
                    team=team,
                    positions=[],
                    hr=hr, r=r, rbi=rbi, sb=sb,
                    w=w, k_pit=k_pit, qs=qs,
                    avg=avg, obp=obp, slg=slg, ops=ops,
                    era=era if era > 0 else 4.00,
                    whip=whip if whip > 0 else 1.30,
                )

        logger.info(f"Loaded {len(projections)} projections from {csv_path.name}")
        return projections

    except Exception as e:
        logger.error(f"Failed to load CSV {csv_path}: {e}")
        return {}


def _float(val) -> float:
    """Safe float conversion."""
    try:
        return float(val or 0)
    except (ValueError, TypeError):
        return 0.0


def resolve_player_ids(db: Session, projections: dict[str, CSVProjection]) -> dict[str, CSVProjection]:
    """Resolve player_id via PlayerIDMapping."""
    from backend.models import PlayerIDMapping, StatcastPerformance

    # Build name -> ID lookup
    name_to_id = {}
    for row in db.execute(select(PlayerIDMapping)).scalars():
        if not row.mlbam_id:
            continue
        key = row.full_name.lower().strip()
        name_to_id[key] = str(row.mlbam_id)

    # Build team lookup from statcast_performance
    team_lookup = {}
    try:
        for row in db.execute(select(StatcastPerformance)).scalars():
            team_lookup[row.player_name.lower()] = row.team
    except Exception:
        pass

    resolved = {}
    for name, proj in projections.items():
        key = name.lower().strip()
        if key in name_to_id:
            proj.player_id = name_to_id[key]
            if key in team_lookup:
                proj.team = team_lookup[key]
            resolved[proj.player_id] = proj

    logger.info(f"Resolved {len(resolved)}/{len(projections)} player IDs")
    return resolved


def write_projections_to_db(db: Session, projections: dict[str, CSVProjection]) -> int:
    """Write CSV projections to PlayerProjection table."""
    from backend.models import PlayerProjection

    written = 0
    for player_id, proj in projections.items():
        existing = db.execute(
            select(PlayerProjection).where(PlayerProjection.player_id == player_id)
        ).scalar_one_or_none()

        if existing:
            existing.hr = int(proj.hr)
            existing.r = int(proj.r)
            existing.rbi = int(proj.rbi)
            existing.sb = int(proj.sb)
            existing.avg = proj.avg if proj.avg > 0 else existing.avg
            existing.obp = proj.obp if proj.obp > 0 else existing.obp
            existing.slg = proj.slg if proj.slg > 0 else existing.slg
            existing.ops = proj.ops if proj.ops > 0 else existing.ops
            if proj.w > 0:
                existing.w = int(proj.w)
                existing.k_pit = int(proj.k_pit)
                existing.qs = int(proj.qs)
                existing.era = proj.era if proj.era > 0 else existing.era
                existing.whip = proj.whip if proj.whip > 0 else existing.whip
            if proj.team and proj.team != "FA":
                existing.team = proj.team
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
                hr=int(proj.hr),
                r=int(proj.r),
                rbi=int(proj.rbi),
                sb=int(proj.sb),
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
        logger.info(f"Wrote {written} projection rows to database")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to commit projections: {e}")
        raise

    return written


def run_csv_backfill(db: Session, csv_path: Optional[Path] = None) -> dict:
    """
    Full CSV projection backfill pipeline.

    Args:
        db: Database session
        csv_path: Path to CSV file. If None, looks for data/projections/fangraphs_ros.csv

    Returns:
        Dict with status and counts
    """
    if csv_path is None:
        # Default projection file location
        csv_path = Path(__file__).resolve().parents[2] / "data" / "projections" / "fangraphs_ros.csv"

    projections = load_projections_from_csv(csv_path)
    if not projections:
        return {
            "status": "failed",
            "fetched": 0,
            "resolved": 0,
            "written": 0,
            "message": f"Failed to load projections from {csv_path}"
        }

    resolved = resolve_player_ids(db, projections)
    if not resolved:
        return {
            "status": "partial",
            "fetched": len(projections),
            "resolved": 0,
            "written": 0,
            "message": f"Loaded {len(projections)} projections but resolved 0 player IDs"
        }

    written = write_projections_to_db(db, resolved)

    return {
        "status": "success" if written > 0 else "partial",
        "fetched": len(projections),
        "resolved": len(resolved),
        "written": written,
        "message": f"Backfilled {written} player projections from CSV"
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from backend.models import SessionLocal

    logging.basicConfig(level=logging.INFO)

    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None

    db = SessionLocal()
    result = run_csv_backfill(db, Path(csv_arg) if csv_arg else None)

    print(f"\nResult: {result['message']}")
    print(f"  Status: {result['status']}")
    print(f"  Fetched: {result['fetched']}")
    print(f"  Resolved: {result['resolved']}")
    print(f"  Written: {result['written']}")
