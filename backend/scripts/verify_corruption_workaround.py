"""
Verify the PlayerIDMapping corruption workaround in the optimizer.

Tests if players with corrupted BDL IDs can still find scores via the workaround.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sqlalchemy import func
from backend.models import SessionLocal, PlayerScore, PlayerIDMapping


def test_workaround(player_name: str, yahoo_key: str, corrupted_bdl_id: int):
    """
    Test if the workaround would find scores for a player with corrupted BDL ID.

    Simulates what the optimizer does:
    1. Look up scores using corrupted bdl_id
    2. If not found, look for alternative bdl_ids with same mlbam_id
    3. Check if any alternative has scores
    """
    db = SessionLocal()

    print(f"\n{'='*60}")
    print(f"Testing: {player_name}")
    print(f"{'='*60}")
    print(f"Yahoo Key: {yahoo_key}")
    print(f"Corrupted BDL ID: {corrupted_bdl_id}")

    # Step 1: Check if scores exist under corrupted BDL ID
    score_corrupted = db.query(PlayerScore).filter(
        PlayerScore.bdl_player_id == corrupted_bdl_id
    ).order_by(PlayerScore.as_of_date.desc()).first()

    print(f"\nStep 1: Scores under corrupted BDL ID {corrupted_bdl_id}?")
    if score_corrupted:
        print(f"  YES - score={score_corrupted.score_0_100}, date={score_corrupted.as_of_date}")
        print(f"  -> Workaround not needed")
        db.close()
        return True
    else:
        print(f"  NO - trigger workaround")

    # Step 2: Find mlbam_id for this player
    mapping = db.query(PlayerIDMapping).filter(
        PlayerIDMapping.yahoo_key == yahoo_key
    ).first()

    if not mapping:
        print(f"  ERROR: No mapping found for yahoo_key={yahoo_key}")
        db.close()
        return False

    mlbam_id = mapping.mlbam_id
    print(f"\nStep 2: Found mlbam_id={mlbam_id}")

    # Step 3: Find alternative BDL IDs for this mlbam_id
    alt_mappings = db.query(PlayerIDMapping.bdl_id).filter(
        PlayerIDMapping.mlbam_id == mlbam_id,
        PlayerIDMapping.bdl_id != corrupted_bdl_id,
        PlayerIDMapping.bdl_id.isnot(None)
    ).all()

    print(f"\nStep 3: Found {len(alt_mappings)} alternative BDL IDs:")
    for alt_row in alt_mappings:
        print(f"  - bdl_id={alt_row.bdl_id}")

        # Check if this alternative has scores
        alt_score = db.query(PlayerScore).filter(
            PlayerScore.bdl_player_id == alt_row.bdl_id
        ).order_by(PlayerScore.as_of_date.desc()).first()

        if alt_score:
            print(f"    -> HAS scores! score={alt_score.score_0_100}, date={alt_score.as_of_date}")
            print(f"  *** WORKAROUND WOULD WORK ***")
            db.close()
            return True
        else:
            print(f"    -> No scores")

    print(f"  *** WORKAROUND FAILED - No scores found for any alternative BDL ID ***")
    db.close()
    return False


def main():
    print("="*60)
    print("CORRUPTION WORKAROUND VERIFICATION")
    print("="*60)

    # Test the corrupted players
    test_cases = [
        ("Sam Antonacci", "469.p.64533", 803011),
        ("Juan Soto", "469.p.10626", 665742),
        ("Carson Benge", "469.p.64329", 701807),
        ("Munetaka Murakami", "469.p.66369", 808959),
    ]

    results = []
    for name, yahoo_key, corrupted_bdl_id in test_cases:
        works = test_workaround(name, yahoo_key, corrupted_bdl_id)
        results.append((name, works))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, works in results:
        status = "WORKS" if works else "FAILS"
        print(f"  {name}: {status}")

    working = sum(1 for _, w in results if w)
    print(f"\nTotal: {working}/{len(results)} workarounds successful")

    if working == len(results):
        print(f"\n*** ALL WORKAROUNDS SUCCESSFUL ***")


if __name__ == "__main__":
    main()
