"""
Steamer CSV Projection Ingestion

Reads Steamer 2026 batting and pitching projections from CSV files and
ingests them into the PlayerProjection table. This provides real projection
variance for the waiver/draft engines.

Usage:
    python -m backend.fantasy_baseball.csv_projection_ingestion
    or POST to /api/admin/data-quality/ingest-csv-projections

Steamer Batting Columns (mapped):
    Name, Team, POS, G, PA, AB, H, 2B, 3B, HR, R, RBI, BB, SO, SB, CS, AVG, OBP, SLG, OPS

Steamer Pitching Columns (mapped):
    Name, Team, POS, W, L, ERA, G, GS, IP, H, HR, BB, SO, SV, BS, WHIP

Derived Stats (calculated):
    - k_per_nine = SO / (IP / 9) for pitchers
    - bb_per_nine = BB / (IP / 9) for pitchers
    - qs = round(GS * 0.5) estimated quality starts (50% of starts)
    - nsv = SV - BS (net saves)
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sqlalchemy import text, select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# Default paths relative to project root
BATTING_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "projections" / "steamer_batting_2026.csv"
PITCHING_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "projections" / "steamer_pitching_2026.csv"


@dataclass
class BatterProjection:
    """Batter projection from Steamer CSV."""
    player_id: str = ""
    player_name: str = ""
    team: str = ""
    positions: list[str] = None

    # Rate stats
    avg: float = 0.250
    obp: float = 0.320
    slg: float = 0.400
    ops: float = 0.720
    woba: float = 0.320
    xwoba: float = 0.320

    # Counting stats
    hr: int = 15
    r: int = 65
    rbi: int = 65
    sb: int = 5

    def __post_init__(self):
        if self.positions is None:
            self.positions = []


@dataclass
class PitcherProjection:
    """Pitcher projection from Steamer CSV."""
    player_id: str = ""
    player_name: str = ""
    team: str = ""
    positions: list[str] = None

    # Rate stats
    era: float = 4.00
    whip: float = 1.30
    k_per_nine: float = 8.5
    bb_per_nine: float = 3.0

    # Counting stats
    w: int = 0
    l: int = 0
    hr_pit: int = 0  # HR allowed
    k_pit: int = 0  # Strikeouts
    qs: int = 0  # Quality starts
    nsv: int = 0  # Net saves (SV - BS)

    def __post_init__(self):
        if self.positions is None:
            self.positions = []


def _parse_decimal(value: str, default: float = 0.0) -> float:
    """Parse decimal string like '199.2' to float 199.666... for IP."""
    if not value or value.strip() == "":
        return default
    try:
        value = value.strip()
        if "." in value:
            parts = value.split(".")
            if len(parts) == 2:
                whole = float(parts[0])
                # In baseball, .1 = 1/3 inning, .2 = 2/3 inning
                partial = float(parts[1])
                if partial == 1:
                    return whole + 1/3
                elif partial == 2:
                    return whole + 2/3
        return float(value)
    except (ValueError, TypeError):
        return default


def _float(value, default: float = 0.0) -> float:
    """Safe float conversion."""
    try:
        return float(value) if value not in (None, "", "NA", "-") else default
    except (ValueError, TypeError):
        return default


def _int(value, default: int = 0) -> int:
    """Safe int conversion."""
    try:
        return int(float(value)) if value not in (None, "", "NA", "-") else default
    except (ValueError, TypeError):
        return default


def ensure_pitcher_columns(db: Session) -> bool:
    """
    Ensure pitcher counting stat columns exist in player_projections table.

    Returns True if columns were added, False if they already existed.
    """
    required_columns = {
        "w": "INTEGER DEFAULT 0",
        "l": "INTEGER DEFAULT 0",
        "hr_pit": "INTEGER DEFAULT 0",
        "k_pit": "INTEGER DEFAULT 0",
        "qs": "INTEGER DEFAULT 0",
        "nsv": "INTEGER DEFAULT 0",
    }

    try:
        # Check existing columns
        result = db.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'player_projections'
        """))
        existing_columns = {row[0] for row in result}

        added = []
        for col_name, col_def in required_columns.items():
            if col_name not in existing_columns:
                db.execute(text(f"""
                    ALTER TABLE player_projections
                    ADD COLUMN {col_name} {col_def}
                """))
                added.append(col_name)
                logger.info(f"Added column: player_projections.{col_name}")

        if added:
            db.commit()
            logger.info(f"Schema evolution: added {len(added)} pitcher counting stat columns")
        else:
            logger.info("All pitcher counting stat columns already exist")

        return len(added) > 0

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to ensure pitcher columns: {e}")
        raise


def load_batting_projections(csv_path: Path) -> dict[str, BatterProjection]:
    """Load batter projections from Steamer batting CSV."""
    projections = {}

    if not csv_path.exists():
        logger.warning(f"Batting CSV not found: {csv_path}")
        return {}

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                name = str(row.get('Name') or '').strip()
                if not name:
                    continue

                # Parse position string (may be like "OF", "SS", or "1B/OF")
                pos_str = str(row.get('POS') or '').strip()
                positions = [p.strip() for p in pos_str.split('/') if p.strip()] if pos_str else []

                # Rate stats
                avg = _float(row.get('AVG'), 0.250)
                obp = _float(row.get('OBP'), 0.320)
                slg = _float(row.get('SLG'), 0.400)
                ops = _float(row.get('OPS'), 0.720)

                # Counting stats
                hr = _int(row.get('HR'), 15)
                r = _int(row.get('R'), 65)
                rbi = _int(row.get('RBI'), 65)
                sb = _int(row.get('SB'), 5)

                team = str(row.get('Team', 'FA')).replace('-', '').strip()

                projections[name] = BatterProjection(
                    player_name=name,
                    team=team,
                    positions=positions,
                    avg=avg if avg > 0 else 0.250,
                    obp=obp if obp > 0 else 0.320,
                    slg=slg if slg > 0 else 0.400,
                    ops=ops if ops > 0 else 0.720,
                    hr=hr,
                    r=r,
                    rbi=rbi,
                    sb=sb,
                )

        logger.info(f"Loaded {len(projections)} batter projections from {csv_path.name}")
        return projections

    except Exception as e:
        logger.error(f"Failed to load batting CSV {csv_path}: {e}")
        return {}


def load_pitching_projections(csv_path: Path) -> dict[str, PitcherProjection]:
    """Load pitcher projections from Steamer pitching CSV."""
    projections = {}

    if not csv_path.exists():
        logger.warning(f"Pitching CSV not found: {csv_path}")
        return {}

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                name = str(row.get('Name') or '').strip()
                if not name:
                    continue

                # Parse position string
                pos_str = str(row.get('POS') or '').strip()
                positions = [p.strip() for p in pos_str.split('/') if p.strip()] if pos_str else []

                # Counting stats
                w = _int(row.get('W'), 0)
                l = _int(row.get('L'), 0)
                hr = _int(row.get('HR'), 0)  # Home runs allowed
                so = _int(row.get('SO'), 0)  # Strikeouts
                sv = _int(row.get('SV'), 0)
                bs = _int(row.get('BS'), 0)
                gs = _int(row.get('GS'), 0)

                # IP may be decimal like "199.2" meaning 199 + 2/3
                ip = _parse_decimal(row.get('IP'), 0)
                bb = _float(row.get('BB'), 0)

                # Rate stats
                era = _float(row.get('ERA'), 4.00)
                whip = _float(row.get('WHIP'), 1.30)

                # Derived stats
                # k_per_nine = SO / (IP / 9)
                k_per_nine = (so / (ip / 9)) if ip > 0 else 8.5

                # bb_per_nine = BB / (IP / 9)
                bb_per_nine = (bb / (ip / 9)) if ip > 0 else 3.0

                # qs (quality starts) ~50% of GS
                qs = round(gs * 0.5) if gs > 0 else 0

                # nsv (net saves) = SV - BS
                nsv = sv - bs

                team = str(row.get('Team', 'FA')).replace('-', '').strip()

                projections[name] = PitcherProjection(
                    player_name=name,
                    team=team,
                    positions=positions,
                    era=era if era > 0 else 4.00,
                    whip=whip if whip > 0 else 1.30,
                    k_per_nine=k_per_nine,
                    bb_per_nine=bb_per_nine,
                    w=w,
                    l=l,
                    hr_pit=hr,
                    k_pit=so,
                    qs=qs,
                    nsv=nsv if nsv > 0 else 0,
                )

        logger.info(f"Loaded {len(projections)} pitcher projections from {csv_path.name}")
        return projections

    except Exception as e:
        logger.error(f"Failed to load pitching CSV {csv_path}: {e}")
        return {}


def resolve_player_ids(db: Session, batters: dict[str, BatterProjection],
                       pitchers: dict[str, PitcherProjection]) -> tuple[dict, dict]:
    """Resolve player_id via PlayerIDMapping for batters and pitchers."""
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

    resolved_batters = {}
    for name, proj in batters.items():
        key = name.lower().strip()
        if key in name_to_id:
            proj.player_id = name_to_id[key]
            if key in team_lookup:
                proj.team = team_lookup[key]
            resolved_batters[proj.player_id] = proj

    resolved_pitchers = {}
    for name, proj in pitchers.items():
        key = name.lower().strip()
        if key in name_to_id:
            proj.player_id = name_to_id[key]
            if key in team_lookup:
                proj.team = team_lookup[key]
            resolved_pitchers[proj.player_id] = proj

    logger.info(f"Resolved {len(resolved_batters)}/{len(batters)} batters, "
                f"{len(resolved_pitchers)}/{len(pitchers)} pitchers")
    return resolved_batters, resolved_pitchers


def write_projections_to_db(db: Session, batters: dict, pitchers: dict) -> int:
    """Write Steamer projections to PlayerProjection table."""
    from backend.models import PlayerProjection

    written = 0

    # Merge batters and pitchers into single projection dict
    # A player could appear in both (e.g., Ohtani) - pitcher stats take precedence for those fields
    all_projections = {}
    all_projections.update({pid: ("batter", proj) for pid, proj in batters.items()})
    all_projections.update({pid: ("pitcher", proj) for pid, proj in pitchers.items()})

    for player_id, (ptype, proj) in all_projections.items():
        existing = db.execute(
            select(PlayerProjection).where(PlayerProjection.player_id == player_id)
        ).scalar_one_or_none()

        if existing:
            # Update existing projection
            if isinstance(proj, BatterProjection):
                existing.hr = proj.hr
                existing.r = proj.r
                existing.rbi = proj.rbi
                existing.sb = proj.sb
                existing.avg = proj.avg if proj.avg > 0 else existing.avg
                existing.obp = proj.obp if proj.obp > 0 else existing.obp
                existing.slg = proj.slg if proj.slg > 0 else existing.slg
                existing.ops = proj.ops if proj.ops > 0 else existing.ops
                if proj.positions:
                    existing.positions = proj.positions
            elif isinstance(proj, PitcherProjection):
                existing.era = proj.era if proj.era > 0 else existing.era
                existing.whip = proj.whip if proj.whip > 0 else existing.whip
                existing.k_per_nine = proj.k_per_nine if proj.k_per_nine > 0 else existing.k_per_nine
                existing.bb_per_nine = proj.bb_per_nine if proj.bb_per_nine > 0 else existing.bb_per_nine
                # Pitcher counting stats (may not exist in old schema)
                if hasattr(existing, 'w'):
                    existing.w = proj.w
                if hasattr(existing, 'l'):
                    existing.l = proj.l
                if hasattr(existing, 'hr_pit'):
                    existing.hr_pit = proj.hr_pit
                if hasattr(existing, 'k_pit'):
                    existing.k_pit = proj.k_pit
                if hasattr(existing, 'qs'):
                    existing.qs = proj.qs
                if hasattr(existing, 'nsv'):
                    existing.nsv = proj.nsv
                if proj.positions:
                    existing.positions = proj.positions

            # Update team if available
            if proj.team and proj.team != "FA":
                existing.team = proj.team

            existing.prior_source = "steamer"
            existing.update_method = "csv"

        else:
            # Insert new projection
            if isinstance(proj, BatterProjection):
                new_row = PlayerProjection(
                    player_id=player_id,
                    player_name=proj.player_name,
                    team=proj.team if proj.team != "FA" else None,
                    positions=proj.positions or None,
                    avg=proj.avg,
                    obp=proj.obp,
                    slg=proj.slg,
                    ops=proj.ops,
                    woba=proj.woba,
                    xwoba=proj.xwoba,
                    hr=proj.hr,
                    r=proj.r,
                    rbi=proj.rbi,
                    sb=proj.sb,
                    prior_source="steamer",
                    update_method="csv",
                )
            elif isinstance(proj, PitcherProjection):
                new_row = PlayerProjection(
                    player_id=player_id,
                    player_name=proj.player_name,
                    team=proj.team if proj.team != "FA" else None,
                    positions=proj.positions or None,
                    era=proj.era if proj.era > 0 else None,
                    whip=proj.whip if proj.whip > 0 else None,
                    k_per_nine=proj.k_per_nine if proj.k_per_nine > 0 else None,
                    bb_per_nine=proj.bb_per_nine if proj.bb_per_nine > 0 else None,
                    prior_source="steamer",
                    update_method="csv",
                )
                # Set pitcher counting stats if columns exist
                if hasattr(new_row, 'w'):
                    new_row.w = proj.w
                if hasattr(new_row, 'l'):
                    new_row.l = proj.l
                if hasattr(new_row, 'hr_pit'):
                    new_row.hr_pit = proj.hr_pit
                if hasattr(new_row, 'k_pit'):
                    new_row.k_pit = proj.k_pit
                if hasattr(new_row, 'qs'):
                    new_row.qs = proj.qs
                if hasattr(new_row, 'nsv'):
                    new_row.nsv = proj.nsv

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


def run_steamer_ingestion(db: Session,
                          batting_path: Optional[Path] = None,
                          pitching_path: Optional[Path] = None) -> dict:
    """
    Full Steamer CSV projection ingestion pipeline.

    Args:
        db: Database session
        batting_path: Path to batting CSV. If None, uses default steamer_batting_2026.csv
        pitching_path: Path to pitching CSV. If None, uses default steamer_pitching_2026.csv

    Returns:
        Dict with status and counts
    """
    # Use default paths if not provided
    if batting_path is None:
        batting_path = BATTING_CSV_PATH
    if pitching_path is None:
        pitching_path = PITCHING_CSV_PATH

    # Step 1: Ensure schema has pitcher counting stats
    ensure_pitcher_columns(db)

    # Step 2: Load projections from CSVs
    batters = load_batting_projections(batting_path)
    pitchers = load_pitching_projections(pitching_path)

    if not batters and not pitchers:
        return {
            "status": "failed",
            "batters_fetched": 0,
            "pitchers_fetched": 0,
            "resolved": 0,
            "written": 0,
            "message": "Failed to load projections from CSV files"
        }

    # Step 3: Resolve player IDs
    resolved_batters, resolved_pitchers = resolve_player_ids(db, batters, pitchers)

    total_resolved = len(resolved_batters) + len(resolved_pitchers)
    if total_resolved == 0:
        return {
            "status": "partial",
            "batters_fetched": len(batters),
            "pitchers_fetched": len(pitchers),
            "resolved": 0,
            "written": 0,
            "message": f"Loaded {len(batters)} batters and {len(pitchers)} pitchers but resolved 0 player IDs"
        }

    # Step 4: Write to database
    written = write_projections_to_db(db, resolved_batters, resolved_pitchers)

    return {
        "status": "success" if written > 0 else "partial",
        "batters_fetched": len(batters),
        "pitchers_fetched": len(pitchers),
        "resolved": total_resolved,
        "written": written,
        "message": f"Ingested {written} Steamer projections ({len(resolved_batters)} batters, {len(resolved_pitchers)} pitchers)"
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from backend.models import SessionLocal

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    batting_arg = sys.argv[1] if len(sys.argv) > 1 else None
    pitching_arg = sys.argv[2] if len(sys.argv) > 2 else None

    db = SessionLocal()
    try:
        result = run_steamer_ingestion(
            db,
            Path(batting_arg) if batting_arg else None,
            Path(pitching_arg) if pitching_arg else None
        )

        print(f"\n{'='*60}")
        print("Steamer Projection Ingestion Result")
        print(f"{'='*60}")
        print(f"Status:    {result['status']}")
        print(f"Batters:   {result.get('batters_fetched', 0)} fetched, "
              f"{len([p for p in [result.get('resolved_batters')] if p])} resolved")
        print(f"Pitchers:  {result.get('pitchers_fetched', 0)} fetched, "
              f"{len([p for p in [result.get('resolved_pitchers')] if p])} resolved")
        print(f"Written:   {result['written']} rows")
        print(f"Message:   {result['message']}")
        print(f"{'='*60}\n")

        sys.exit(0 if result['status'] == 'success' else 1)
    finally:
        db.close()
