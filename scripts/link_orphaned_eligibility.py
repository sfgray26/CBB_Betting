"""
Link Orphaned Position Eligibility Records via Fuzzy Name Matching

Task 21: Link 477 orphaned position_eligibility records to player_id_mapping
via difflib.SequenceMatcher with 85% similarity threshold.

Usage:
    python scripts/link_orphaned_eligibility.py [--dry-run] [--verbose]

Expected Output:
    - Linked count: ~400-450 records
    - Remaining orphans: <50 records
    - Success rate: 85-95%

Author: Task 21 Implementation
Date: 2026-04-10
"""

import logging
import sys
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from difflib import SequenceMatcher
from typing import List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy import create_engine, text, func, or_
from sqlalchemy.orm import Session

from backend.models import SessionLocal, PlayerIDMapping, PositionEligibility

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SIMILARITY_THRESHOLD = 0.85  # 85% similarity required for fuzzy match
MAX_CANDIDATES = 10  # Only check top 10 candidates per orphan


def normalize_name_for_matching(name: str) -> str:
    """
    Normalize player name for fuzzy matching.

    Steps:
    1. Convert to lowercase
    2. Remove accents/Unicode
    3. Remove extra whitespace
    4. Remove common suffixes (Jr., Sr., II, III, IV)
    """
    import unicodedata
    import re

    if not name:
        return ""

    # Lowercase
    name = name.lower()

    # Remove accents
    name = unicodedata.normalize('NFKD', name)
    name = ''.join([c for c in name if not unicodedata.combining(c)])

    # Remove common suffixes
    name = re.sub(r'\s+jr\.?\s*$', '', name)
    name = re.sub(r'\s+sr\.?\s*$', '', name)
    name = re.sub(r'\s+ii\s*$', '', name)
    name = re.sub(r'\s+iii\s*$', '', name)
    name = re.sub(r'\s+iv\s*$', '', name)

    # Remove extra whitespace
    name = ' '.join(name.split())

    return name


def calculate_similarity(name1: str, name2: str) -> float:
    """
    Calculate similarity between two names using SequenceMatcher.

    Returns:
        float: Similarity score between 0.0 and 1.0
    """
    norm1 = normalize_name_for_matching(name1)
    norm2 = normalize_name_for_matching(name2)

    if not norm1 or not norm2:
        return 0.0

    return SequenceMatcher(None, norm1, norm2).ratio()


def find_best_match(
    orphan: PositionEligibility,
    mapping_candidates: List[PlayerIDMapping],
    verbose: bool = False
) -> Optional[PlayerIDMapping]:
    """
    Find best matching player_id_mapping record for an orphaned position_eligibility.

    Args:
        orphan: Orphaned PositionEligibility record
        mapping_candidates: List of PlayerIDMapping candidates
        verbose: Enable verbose logging

    Returns:
        Best matching PlayerIDMapping or None if no match above threshold
    """
    best_match = None
    best_score = 0.0

    orphan_name = orphan.player_name or ""
    orphan_last = orphan.last_name or ""

    for candidate in mapping_candidates:
        # Calculate similarity scores
        full_name_score = calculate_similarity(orphan_name, candidate.full_name)

        # Also try matching just last names
        last_name_score = 0.0
        if orphan_last:
            candidate_last = candidate.full_name.split()[-1] if candidate.full_name else ""
            last_name_score = calculate_similarity(orphan_last, candidate_last)

        # Use the higher score
        score = max(full_name_score, last_name_score)

        if score > best_score:
            best_score = score
            best_match = candidate

        if verbose and score > 0.7:
            logger.info(
                f"  Similarity: {orphan_name} vs {candidate.full_name} = "
                f"{score:.3f} (full: {full_name_score:.3f}, last: {last_name_score:.3f})"
            )

    if best_score >= SIMILARITY_THRESHOLD:
        if verbose:
            logger.info(
                f"  MATCH FOUND: {orphan_name} -> {best_match.full_name} "
                f"(score: {best_score:.3f})"
            )
        return best_match

    return None


def link_orphaned_records(
    dry_run: bool = True,
    verbose: bool = False
) -> dict:
    """
    Link orphaned position_eligibility records to player_id_mapping.

    Args:
        dry_run: If True, don't commit changes
        verbose: Enable verbose logging

    Returns:
        dict with status, linked_count, remaining_count, success_rate, elapsed_ms
    """
    t0 = datetime.now(ZoneInfo("America/New_York"))
    logger.info("=" * 60)
    logger.info("Starting orphaned position_eligibility linking")
    logger.info("=" * 60)

    try:
        db = SessionLocal()

        # Count orphaned records
        total_elig = db.query(func.count(PositionEligibility.id)).scalar()
        linked = db.query(func.count(PositionEligibility.id)).join(
            PlayerIDMapping, PositionEligibility.bdl_player_id == PlayerIDMapping.bdl_id
        ).scalar()
        orphan_count = total_elig - linked

        logger.info(f"Total position_eligibility records: {total_elig}")
        logger.info(f"Linked to player_id_mapping: {linked}")
        logger.info(f"Orphaned records: {orphan_count}")

        if orphan_count == 0:
            logger.info("No orphaned records found - exiting")
            return {
                "status": "success",
                "linked_count": 0,
                "remaining_count": 0,
                "success_rate": 100.0,
                "elapsed_ms": (datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds() * 1000
            }

        # Fetch all candidates from player_id_mapping
        logger.info("Fetching player_id_mapping candidates...")
        all_candidates = db.query(PlayerIDMapping).all()
        logger.info(f"Loaded {len(all_candidates)} candidates")

        # Fetch orphaned records
        logger.info("Fetching orphaned position_eligibility records...")
        orphans = db.query(PositionEligibility).outerjoin(
            PlayerIDMapping, PositionEligibility.bdl_player_id == PlayerIDMapping.bdl_id
        ).filter(PlayerIDMapping.bdl_id.is_(None)).all()

        logger.info(f"Processing {len(orphans)} orphaned records...")

        # Link orphans to candidates
        linked_count = 0
        failed_count = 0

        for i, orphan in enumerate(orphans, 1):
            if verbose:
                logger.info(f"\n[{i}/{len(orphans)}] Processing: {orphan.player_name}")

            best_match = find_best_match(orphan, all_candidates, verbose)

            if best_match:
                # Update the orphan record
                orphan.bdl_player_id = best_match.bdl_id
                linked_count += 1

                if verbose:
                    logger.info(
                        f"  LINKED: {orphan.player_name} -> "
                        f"{best_match.full_name} (bdl_id: {best_match.bdl_id})"
                    )
            else:
                failed_count += 1
                if verbose:
                    logger.warning(f"  NO MATCH: {orphan.player_name}")

            # Progress update every 50 records
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(orphans)} processed ({linked_count} linked)")

        # Commit changes
        if not dry_run:
            logger.info("\nCommitting changes to database...")
            db.commit()
            logger.info("Changes committed successfully")
        else:
            logger.info("\nDRY RUN - rolling back changes")
            db.rollback()

        # Verify results
        remaining_orphans = orphan_count - linked_count
        success_rate = (linked_count / orphan_count * 100) if orphan_count > 0 else 0.0

        logger.info("\n" + "=" * 60)
        logger.info("RESULTS:")
        logger.info(f"  Linked: {linked_count}")
        logger.info(f"  Remaining orphans: {remaining_orphans}")
        logger.info(f"  Success rate: {success_rate:.1f}%")
        logger.info("=" * 60)

        elapsed = (datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds() * 1000

        return {
            "status": "success",
            "linked_count": linked_count,
            "remaining_count": remaining_orphans,
            "success_rate": success_rate,
            "elapsed_ms": elapsed
        }

    except Exception as e:
        logger.error(f"Error during linking: {e}")
        if 'db' in locals():
            db.rollback()
        raise
    finally:
        if 'db' in locals():
            db.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Link orphaned position_eligibility records via fuzzy matching"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate linking without committing changes"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be committed\n")

    try:
        result = link_orphaned_records(
            dry_run=args.dry_run,
            verbose=args.verbose
        )

        logger.info("\nOperation completed successfully!")
        logger.info(f"Status: {result['status']}")
        logger.info(f"Linked: {result['linked_count']}")
        logger.info(f"Remaining: {result['remaining_count']}")
        logger.info(f"Success rate: {result['success_rate']:.1f}%")
        logger.info(f"Elapsed: {result['elapsed_ms']:.0f}ms")

        # Exit with error if too many orphans remain
        if result['remaining_count'] > 50:
            logger.warning(
                f"\nWARNING: {result['remaining_count']} orphans remaining "
                f"(target: <50)"
            )
            sys.exit(1)

        sys.exit(0)

    except Exception as e:
        logger.error(f"\nOperation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
