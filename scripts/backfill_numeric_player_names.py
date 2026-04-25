#!/usr/bin/env python3
"""
Backfill player names in player_projections table.

Issue: 353 rows have numeric MLBAM IDs in player_name column instead of actual names.
Fix: Join with player_id_mapping to backfill correct names.

Run via: railway run python scripts/backfill_numeric_player_names.py
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from backend.models import SessionLocal, PlayerIDMapping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def backfill_player_names() -> dict:
    """Backfill numeric player names from player_id_mapping table."""
    db = SessionLocal()
    try:
        # Step 1: Count affected rows
        count_result = db.execute(
            text("SELECT COUNT(*) FROM player_projections WHERE player_name ~ '^[0-9]+$'")
        )
        affected_count = count_result.scalar()
        logger.info("Found %d rows with numeric player_name", affected_count)

        if affected_count == 0:
            return {"status": "success", "rows_updated": 0, "message": "No rows to fix"}

        # Step 2: Backfill using UPDATE with JOIN
        # PostgreSQL doesn't support JOIN in UPDATE directly, use a subquery
        update_result = db.execute(
            text("""
                UPDATE player_projections pp
                SET player_name = pid.full_name
                FROM player_id_mapping pid
                WHERE pp.player_name ~ '^[0-9]+$'
                  AND pid.mlbam_id::text = pp.player_name
            """)
        )

        db.commit()
        logger.info("Updated %d rows", update_result.rowcount)

        # Step 3: Verify - count remaining numeric names
        remaining_result = db.execute(
            text("SELECT COUNT(*) FROM player_projections WHERE player_name ~ '^[0-9]+$'")
        )
        remaining_count = remaining_result.scalar()

        return {
            "status": "success",
            "rows_updated": update_result.rowcount,
            "remaining_numeric_names": remaining_count,
            "message": f"Backfilled {update_result.rowcount} player names, {remaining_count} remaining (unmatched)",
        }

    except Exception as exc:
        db.rollback()
        logger.error("Backfill failed: %s", exc)
        return {"status": "error", "error": str(exc)}
    finally:
        db.close()


if __name__ == "__main__":
    result = backfill_player_names()
    print(result)

    if result.get("status") == "error":
        sys.exit(1)
    elif result.get("remaining_numeric_names", 0) > 0:
        logger.warning("Some player names could not be matched - check player_id_mapping coverage")
        sys.exit(0)
    else:
        logger.info("All numeric player names backfilled successfully")
        sys.exit(0)
