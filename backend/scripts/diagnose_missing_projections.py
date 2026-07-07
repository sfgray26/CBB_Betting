"""
Diagnostic script to check why players have no projections.

For each missing player, checks:
1. PlayerIDMapping table (Yahoo key -> BDL ID)
2. player_scores table (BDL ID -> score)
3. player_rolling_stats table (BDL ID -> rolling stats)
4. Identifies root cause
"""

import os
import sys
from datetime import date, timedelta

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sqlalchemy import text
from backend.models import SessionLocal, PlayerIDMapping, PlayerScore, PlayerRollingStats


def check_player(db, player_name: str, yahoo_key: str = None) -> dict:
    """
    Check why a player has no projection.

    Returns dict with diagnostic info.
    """
    result = {
        "name": player_name,
        "yahoo_key": yahoo_key,
        "in_mapping": False,
        "bdl_id": None,
        "mlbam_id": None,
        "in_scores": False,
        "in_rolling_stats": False,
        "latest_score_date": None,
        "latest_score_value": None,
        "rolling_stats_count": 0,
        "root_cause": "UNKNOWN"
    }

    print(f"\n{'='*60}")
    print(f"Checking: {player_name}")
    print(f"Yahoo Key: {yahoo_key}")
    print(f"{'='*60}")

    # Step 1: Check PlayerIDMapping
    if yahoo_key:
        mapping = db.query(PlayerIDMapping).filter(
            PlayerIDMapping.yahoo_key == yahoo_key
        ).first()

        if mapping:
            result["in_mapping"] = True
            result["bdl_id"] = mapping.bdl_id
            result["mlbam_id"] = mapping.mlbam_id
            print("[OK] Found in PlayerIDMapping")
            print(f"  BDL ID: {mapping.bdl_id}")
            print(f"  MLBAM ID: {mapping.mlbam_id}")
        else:
            print("[MISS] NOT in PlayerIDMapping table")
            result["root_cause"] = "MISSING_MAPPING"
            return result
    else:
        # Try to find by name
        mapping = db.query(PlayerIDMapping).filter(
            PlayerIDMapping.full_name.ilike(f"%{player_name}%")
        ).first()

        if mapping:
            result["in_mapping"] = True
            result["bdl_id"] = mapping.bdl_id
            result["mlbam_id"] = mapping.mlbam_id
            print("[OK] Found in PlayerIDMapping (by name)")
            print(f"  Yahoo Key: {mapping.yahoo_key}")
            print(f"  BDL ID: {mapping.bdl_id}")
            print(f"  MLBAM ID: {mapping.mlbam_id}")
        else:
            print("[MISS] NOT in PlayerIDMapping table (no yahoo_key provided, name search failed)")
            result["root_cause"] = "MISSING_MAPPING"
            return result

    # Step 2: Check player_scores
    if result["bdl_id"]:
        scores = db.query(PlayerScore).filter(
            PlayerScore.bdl_player_id == result["bdl_id"]
        ).order_by(PlayerScore.as_of_date.desc()).limit(1).first()

        if scores:
            result["in_scores"] = True
            result["latest_score_date"] = str(scores.as_of_date)
            result["latest_score_value"] = scores.score_0_100
            print("[OK] Found in player_scores")
            print(f"  Latest date: {scores.as_of_date}")
            print(f"  Score: {scores.score_0_100}")
        else:
            print("[MISS] NOT in player_scores table")
            result["root_cause"] = "MISSING_SCORE"

    # Step 3: Check player_rolling_stats
    if result["bdl_id"]:
        rolling = db.query(PlayerRollingStats).filter(
            PlayerRollingStats.bdl_player_id == result["bdl_id"]
        ).count()

        result["rolling_stats_count"] = rolling
        if rolling > 0:
            result["in_rolling_stats"] = True
            print(f"[OK] Found in player_rolling_stats ({rolling} rows)")
        else:
            print("[MISS] NOT in player_rolling_stats table")
            if result["root_cause"] == "UNKNOWN":
                result["root_cause"] = "MISSING_ROLLING_STATS"

    # Determine root cause if still unknown
    if result["root_cause"] == "UNKNOWN":
        if result["in_mapping"] and result["in_rolling_stats"] and not result["in_scores"]:
            result["root_cause"] = "INGESTION_GAP"  # Has rolling stats but no score
        elif result["in_mapping"] and not result["in_rolling_stats"]:
            result["root_cause"] = "NO_GAME_DATA"  # Has mapping but no stats
        elif result["in_mapping"] and result["in_scores"]:
            result["root_cause"] = "STALE_DATA"  # Has data but might be old

    print(f"\nRoot Cause: {result['root_cause']}")
    return result


def main():
    db = SessionLocal()

    try:
        # Missing players from the optimizer output
        missing_players = [
            ("Dillon Dingler", None),  # C
            ("Sam Antonacci", None),   # 2B
            ("Tyler Soderstrom", None),  # 1B
            ("Jordan Walker", None),   # OF
            ("Carson Benge", None),    # OF
            ("Munetaka Murakami", None),  # 3B
            ("Edwin Díaz", None),      # Util (P)
            ("Cristopher Sánchez", None),  # P
            ("Juan Soto", None),       # OF (missing entirely)
        ]

        print("="*60)
        print("MISSING PLAYER PROJECTION DIAGNOSTIC")
        print("="*60)

        results = []
        for name, yahoo_key in missing_players:
            result = check_player(db, name, yahoo_key)
            results.append(result)

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        root_cause_counts = {}
        for r in results:
            cause = r["root_cause"]
            root_cause_counts[cause] = root_cause_counts.get(cause, 0) + 1

        for cause, count in root_cause_counts.items():
            print(f"  {cause}: {count}")

        print(f"\nTotal missing: {len(results)}")

        # Print detailed results
        print("\n" + "="*60)
        print("DETAILED RESULTS")
        print("="*60)
        for r in results:
            print(f"\n{r['name']}:")
            print(f"  Root Cause: {r['root_cause']}")
            print(f"  BDL ID: {r['bdl_id']}")
            print(f"  In Mapping: {r['in_mapping']}")
            print(f"  In Scores: {r['in_scores']}")
            print(f"  In Rolling Stats: {r['in_rolling_stats']}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
