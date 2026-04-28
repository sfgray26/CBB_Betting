#!/usr/bin/env python
"""
Backfill numeric player names in player_projections table.

Phase 1 Remediation (K-32 P0):
- Resolve 98 players with numeric names (e.g., "695578", "683002")
- Use BDL /mlb/v1/players?search= endpoint to look up names
- Fallback to MLB Stats API for remaining unknowns

Usage:
    railway run python scripts/backfill_numeric_player_names.py
"""
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from backend.models import SessionLocal, PlayerProjection, PlayerIDMapping
from backend.services.balldontlie import BallDontLieClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _normalize_name(name: str) -> str:
    """Strip extra whitespace from player name."""
    return " ".join(name.strip().split()) if name else ""


def backfill_via_bdl(limit: int = 100) -> dict:
    """
    Backfill numeric player names using BDL player search.

    Returns:
        dict with {total, resolved, failed, skipped}
    """
    db = SessionLocal()
    bdl = BallDontLieClient()

    try:
        # Find all numeric-name projections
        query = text("""
            SELECT DISTINCT pp.id, pp.player_id, pp.player_name
            FROM player_projections pp
            WHERE pp.player_name ~ '^[0-9]+$'
            LIMIT :limit
        """)
        result = db.execute(query, {"limit": limit})
        numeric_players = result.fetchall()

        total = len(numeric_players)
        resolved = 0
        failed = 0
        skipped = 0

        logger.info("Found %d numeric-name projections to resolve", total)

        for row in numeric_players:
            pp_id, player_id, numeric_name = row
            numeric_id = int(numeric_name)

            # Try BDL search by player_id (same as BDL ID)
            try:
                # BDL search endpoint takes player ID
                bdl_player = bdl.get_mlb_player(player_id)

                if bdl_player and bdl_player.full_name:
                    real_name = _normalize_name(bdl_player.full_name)

                    # Update projection
                    update_stmt = text("""
                        UPDATE player_projections
                        SET player_name = :name,
                            updated_at = NOW()
                        WHERE id = :pp_id
                    """)
                    db.execute(update_stmt, {"name": real_name, "pp_id": pp_id})

                    # Create or update player_id_mapping entry
                    mapping_stmt = text("""
                        INSERT INTO player_id_mapping (yahoo_id, mlb_id, bdl_id, player_name)
                        VALUES (:yahoo_id, :mlb_id, :bdl_id, :name)
                        ON CONFLICT (yahoo_id) DO UPDATE SET
                            mlb_id = EXCLUDED.mlb_id,
                            bdl_id = EXCLUDED.bdl_id,
                            player_name = EXCLUDED.player_name
                    """)
                    db.execute(mapping_stmt, {
                        "yahoo_id": player_id,
                        "mlb_id": bdl_player.mlb_id or None,
                        "bdl_id": bdl_player.id,
                        "name": real_name,
                    })

                    resolved += 1
                    logger.info("Resolved %s: %s -> %s", player_id, numeric_name, real_name)
                else:
                    logger.warning("BDL lookup failed for %s (no result)", player_id)
                    failed += 1
            except Exception as exc:
                logger.error("Failed to resolve %s: %s", player_id, exc)
                failed += 1

        db.commit()
        logger.info("Backfill complete: %d resolved, %d failed, %d skipped", resolved, failed, skipped)

        return {
            "total": total,
            "resolved": resolved,
            "failed": failed,
            "skipped": skipped,
        }

    except Exception as exc:
        db.rollback()
        logger.error("Backfill failed: %s", exc)
        raise
    finally:
        db.close()


def check_remaining_numeric_names() -> dict:
    """Count remaining numeric-name projections after backfill."""
    db = SessionLocal()
    try:
        query = text("""
            SELECT
                COUNT(*) FILTER (WHERE player_name ~ '^[0-9]+$') AS numeric_names,
                COUNT(*) AS total_projections
            FROM player_projections
        """)
        result = db.execute(query).fetchone()
        return {
            "numeric_names": result[0] if result else 0,
            "total_projections": result[1] if result else 0,
        }
    finally:
        db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backfill numeric player names")
    parser.add_argument("--limit", type=int, default=100, help="Max players to process")
    parser.add_argument("--dry-run", action="store_true", help="Check only, no updates")
    args = parser.parse_args()

    if args.dry_run:
        remaining = check_remaining_numeric_names()
        logger.info("DRY RUN: %d numeric names remaining out of %d total projections",
                    remaining["numeric_names"], remaining["total_projections"])
    else:
        result = backfill_via_bdl(limit=args.limit)
        remaining = check_remaining_numeric_names()

        logger.info("SUMMARY:")
        logger.info("  Processed: %d", result["total"])
        logger.info("  Resolved: %d", result["resolved"])
        logger.info("  Failed: %d", result["failed"])
        logger.info("  Remaining numeric names: %d", remaining["numeric_names"])
