#!/usr/bin/env python3
"""
Diagnostic script for Jarren Duran and Chandler Simpson COLD misclassification.

Run in Railway container to query production database.

Usage:
    cd /app && python scripts/diagnostic_momentum_cold.py

Output:
    - Momentum data for both players
    - Cohort distribution (mean, std, percentiles)
    - Root cause analysis
"""

import os
import sys
from datetime import datetime, timezone

# Add backend to path
sys.path.insert(0, '/app/backend')

from sqlalchemy import text
from backend.models import SessionLocal

# Player names to diagnose
TARGET_PLAYERS = [
    {"name": "jarren duran", "player_type": "hitter"},
    {"name": "chandler simpson", "player_type": "hitter"},
]

def main():
    db = SessionLocal()
    today = datetime.now(timezone.utc).date()

    print("=" * 80)
    print("MOMENTUM DIAGNOSTIC: COLD MISCLASSIFICATION")
    print("=" * 80)
    print(f"As of: {today.isoformat()}")
    print()

    # Step 1: Get BDL player IDs for target players
    print("Step 1: Finding BDL player IDs...")
    print("-" * 80)

    player_ids = {}
    for target in TARGET_PLAYERS:
        name = target["name"]
        result = db.execute(text("""
            SELECT DISTINCT bdl_player_id, full_name
            FROM player_id_mapping
            WHERE LOWER(full_name) LIKE :name_pattern
            LIMIT 5
        """), {"name_pattern": f"%{name}%"})

        rows = result.fetchall()
        if not rows:
            print(f"⚠️  NO MATCH: {name}")
            continue

        for row in rows:
            bdl_id, full_name = row
            print(f"Found: {full_name} -> BDL ID: {bdl_id}")
            player_ids[bdl_id] = {"name": full_name, "player_type": target["player_type"]}

        print()

    if not player_ids:
        print("❌ No players found. Aborting.")
        return

    # Step 2: Get most recent momentum records
    print("Step 2: Momentum records for target players...")
    print("-" * 80)

    for bdl_id, info in player_ids.items():
        result = db.execute(text("""
            SELECT
                bdl_player_id,
                as_of_date,
                player_type,
                delta_z,
                signal,
                composite_z_14d,
                composite_z_30d,
                score_14d,
                score_30d,
                confidence_14d,
                confidence_30d,
                confidence,
                computed_at
            FROM player_momentum
            WHERE bdl_player_id = :bdl_id
            ORDER BY as_of_date DESC
            LIMIT 5
        """), {"bdl_id": bdl_id})

        rows = result.fetchall()
        if not rows:
            print(f"⚠️  No momentum records for {info['name']} (BDL ID: {bdl_id})")
            print()
            continue

        print(f"\n📊 {info['name']} (BDL ID: {bdl_id})")
        print("-" * 80)

        for row in rows:
            (
                bdl_id, as_of_date, player_type, delta_z, signal,
                composite_z_14d, composite_z_30d, score_14d, score_30d,
                confidence_14d, confidence_30d, confidence, computed_at
            ) = row

            print(f"  Date:      {as_of_date}")
            print(f"  Signal:    {signal}")
            print(f"  Delta Z:   {delta_z:.3f}")
            print(f"  14d Z:     {composite_z_14d:.3f} (score: {score_14d:.1f}, conf: {confidence_14d:.2f})")
            print(f"  30d Z:     {composite_z_30d:.3f} (score: {score_30d:.1f}, conf: {confidence_30d:.2f})")
            print(f"  Confidence:{confidence:.2f}")
            print(f"  Computed:  {computed_at}")
            print()

    # Step 3: Get cohort distribution for hitters on most recent date
    print("Step 3: Cohort distribution analysis (hitters)...")
    print("-" * 80)

    # Get most recent momentum date
    result = db.execute(text("""
        SELECT as_of_date
        FROM player_momentum
        WHERE player_type = 'hitter'
        ORDER BY as_of_date DESC
        LIMIT 1
    """))
    most_recent_date = result.scalar()
    print(f"Most recent momentum date: {most_recent_date}")

    if most_recent_date:
        # Get cohort statistics
        result = db.execute(text("""
            SELECT
                COUNT(*) as cohort_size,
                AVG(delta_z) as mean_delta,
                STDDEV(delta_z) as std_delta,
                MIN(delta_z) as min_delta,
                MAX(delta_z) as max_delta,
                PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY delta_z) as p90_delta,
                PERCENTILE_CONT(0.70) WITHIN GROUP (ORDER BY delta_z) as p70_delta,
                PERCENTILE_CONT(0.30) WITHIN GROUP (ORDER BY delta_z) as p30_delta,
                PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY delta_z) as p10_delta
            FROM player_momentum
            WHERE as_of_date = :date
              AND player_type = 'hitter'
        """), {"date": most_recent_date})

        row = result.fetchone()
        if row:
            (
                cohort_size, mean_delta, std_delta, min_delta, max_delta,
                p90_delta, p70_delta, p30_delta, p10_delta
            ) = row

            print()
            print(f"Cohort Size:      {cohort_size}")
            print(f"Delta Z Mean:     {mean_delta:.4f}")
            print(f"Delta Z StdDev:   {std_delta:.4f}")
            print(f"Delta Z Range:    [{min_delta:.4f}, {max_delta:.4f}]")
            print()
            print(f"Percentiles (delta_z):")
            print(f"  90th percentile (SURGING threshold):  {p90_delta:.4f}")
            print(f"  70th percentile (HOT threshold):      {p70_delta:.4f}")
            print(f"  30th percentile (COLD threshold):     {p30_delta:.4f}")
            print(f"  10th percentile (COLLAPSING threshold): {p10_delta:.4f}")
            print()

        # Compute z_score_delta for each target player
        print("Step 4: Z-score delta analysis for target players...")
        print("-" * 80)

        for bdl_id, info in player_ids.items():
            result = db.execute(text("""
                SELECT delta_z
                FROM player_momentum
                WHERE bdl_player_id = :bdl_id
                  AND as_of_date = :date
                  AND player_type = 'hitter'
                LIMIT 1
            """), {"bdl_id": bdl_id, "date": most_recent_date})

            row = result.fetchone()
            if not row:
                continue

            delta_z = row[0]
            z_score_delta = (delta_z - mean_delta) / std_delta if std_delta > 0 else 0.0

            print(f"\n{info['name']}:")
            print(f"  Delta Z:         {delta_z:.4f}")
            print(f"  Z-score delta:   {z_score_delta:.4f} (standard deviations from mean)")
            print()

            # Compare to percentiles
            if z_score_delta >= (p90_delta - mean_delta) / std_delta if std_delta > 0 else 0.0:
                print(f"  → Z-score >= 90th percentile → Should be SURGING ✅")
            elif z_score_delta >= (p70_delta - mean_delta) / std_delta if std_delta > 0 else 0.0:
                print(f"  → Z-score >= 70th percentile → Should be HOT ✅")
            elif z_score_delta >= (p30_delta - mean_delta) / std_delta if std_delta > 0 else 0.0:
                print(f"  → Z-score >= 30th percentile → Should be STABLE ✅")
            elif z_score_delta >= (p10_delta - mean_delta) / std_delta if std_delta > 0 else 0.0:
                print(f"  → Z-score >= 10th percentile → Should be COLD ✅")
            else:
                print(f"  → Z-score < 10th percentile → Should be COLLAPSING ✅")
            print()

    # Step 5: Get player_scores for 14d and 30d windows
    print("Step 5: Player scores (14d vs 30d) for target players...")
    print("-" * 80)

    for bdl_id, info in player_ids.items():
        result = db.execute(text("""
            SELECT
                bdl_player_id,
                as_of_date,
                window_days,
                composite_z,
                score_0_100,
                confidence,
                computed_at
            FROM player_scores
            WHERE bdl_player_id = :bdl_id
              AND as_of_date = :date
              AND window_days IN (14, 30)
            ORDER BY window_days
        """), {"bdl_id": bdl_id, "date": most_recent_date})

        rows = result.fetchall()
        if not rows:
            print(f"⚠️  No player_scores for {info['name']}")
            print()
            continue

        print(f"\n{info['name']}:")
        print("-" * 80)

        for row in rows:
            bdl_id, as_of_date, window_days, composite_z, score_0_100, confidence, computed_at = row
            print(f"  Window:     {window_days} days")
            print(f"  Composite Z: {composite_z:.3f}")
            print(f"  Score:      {score_0_100:.1f}")
            print(f"  Confidence: {confidence:.2f}")
            print(f"  Computed:   {computed_at}")
            print()

        # Compute delta_z manually
        if len(rows) == 2:
            score_14d = next(r for r in rows if r[3] == 14)
            score_30d = next(r for r in rows if r[3] == 30)

            delta_z_computed = score_14d[4] - score_30d[4]
            print(f"  Computed Delta Z: {delta_z_computed:.4f}")
            print()

    # Step 6: Check computation timestamps (stale data detection)
    print("Step 6: Data freshness check...")
    print("-" * 80)

    result = db.execute(text("""
        SELECT
            MAX(computed_at) as last_momentum_compute,
            NOW() as now
        FROM player_momentum
    """))
    row = result.fetchone()

    if row:
        last_compute, now = row
        if last_compute:
            lag_hours = (now - last_compute).total_seconds() / 3600 if now else None
            print(f"Last momentum computation: {last_compute}")
            print(f"Current time: {now}")
            print(f"Lag: {lag_hours:.1f} hours")

            if lag_hours and lag_hours > 12:
                print("⚠️  WARNING: Data is stale (>12 hours old)")
                print("   Recent surges may not be captured.")
            elif lag_hours and lag_hours > 6:
                print("ℹ️  INFO: Data is moderately fresh (6-12 hours)")
                print("   Recent surges may be partially captured.")
            else:
                print("✅ Data is fresh (<6 hours)")
        print()

    db.close()

    print("=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()