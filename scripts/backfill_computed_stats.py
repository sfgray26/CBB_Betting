"""
Backfill computed stats (ops, whip, caught_stealing) for existing mlb_player_stats rows.
"""
import os
import sys
from sqlalchemy import update

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from backend.models import SessionLocal, MLBPlayerStats


def _parse_innings_pitched(ip):
    """Convert BDL innings pitched format to decimal."""
    if ip is None:
        return None
    if isinstance(ip, (int, float)):
        return float(ip)
    if isinstance(ip, str):
        parts = ip.split(".")
        try:
            innings = int(parts[0])
            outs = int(parts[1]) if len(parts) > 1 else 0
            return innings + (outs / 3.0)
        except (ValueError, IndexError):
            return None
    return None


def backfill_computed_stats():
    """Backfill ops, whip, caught_stealing for existing rows."""
    db = SessionLocal()

    try:
        # Backfill OPS
        ops_updated = db.execute(
            update(MLBPlayerStats)
            .where(MLBPlayerStats.ops.is_(None))
            .where(MLBPlayerStats.obp.is_not(None))
            .where(MLBPlayerStats.slg.is_not(None))
            .values(ops=MLBPlayerStats.obp + MLBPlayerStats.slg)
        ).rowcount

        # Backfill WHIP (more complex, need to parse IP)
        stats_to_update = db.query(MLBPlayerStats).filter(
            MLBPlayerStats.whip.is_(None),
            MLBPlayerStats.walks_allowed.is_not(None),
            MLBPlayerStats.hits_allowed.is_not(None),
            MLBPlayerStats.innings_pitched.is_not(None)
        ).all()

        whip_updated = 0
        for stat in stats_to_update:
            ip_decimal = _parse_innings_pitched(stat.innings_pitched)
            if ip_decimal is not None and ip_decimal > 0:
                stat.whip = (stat.walks_allowed + stat.hits_allowed) / ip_decimal
                whip_updated += 1

        # Backfill caught_stealing (default to 0)
        cs_updated = db.execute(
            update(MLBPlayerStats)
            .where(MLBPlayerStats.caught_stealing.is_(None))
            .values(caught_stealing=0)
        ).rowcount

        db.commit()

        print(f"Backfill complete:")
        print(f"  OPS updated: {ops_updated}")
        print(f"  WHIP updated: {whip_updated}")
        print(f"  Caught stealing updated: {cs_updated}")

        # Verification
        ops_count = db.query(MLBPlayerStats).filter(
            MLBPlayerStats.ops.is_not(None)
        ).count()
        whip_count = db.query(MLBPlayerStats).filter(
            MLBPlayerStats.whip.is_not(None)
        ).count()
        cs_count = db.query(MLBPlayerStats).filter(
            MLBPlayerStats.caught_stealing.is_not(None)
        ).count()

        print(f"\nVerification:")
        print(f"  OPS NOT NULL: {ops_count}")
        print(f"  WHIP NOT NULL: {whip_count}")
        print(f"  Caught stealing NOT NULL: {cs_count}")

    finally:
        db.close()


if __name__ == "__main__":
    backfill_computed_stats()
