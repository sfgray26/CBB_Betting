#!/usr/bin/env python3
"""
Chandler Simpson Investigation — Why is he marked COLD when delta_z = +0.083 > HOT threshold?

Expected: HOT (delta_z +0.083 > HOT threshold +0.0568)
Actual: COLD

Hypotheses:
1. Stale persisted signal (signal computed earlier, delta_z changed but signal not updated)
2. Confidence gate (confidence too low, forced to COLD)
3. Level gate (score_0_100 < 25th percentile, forced downgrade)
4. Dirty BDL mapping (using wrong BDL ID)

Run in Railway container.

Usage:
    cd /app && python scripts/investigate_chandler_simpson.py
"""

import sys
from datetime import datetime, timezone

sys.path.insert(0, '/app/backend')

from sqlalchemy import text
from backend.models import SessionLocal

# Chandler Simpson's BDL IDs
CHANDLER_BDL_IDS = [653217, 802415]

def main():
    db = SessionLocal()
    today = datetime.now(timezone.utc).date()

    print("=" * 80)
    print("CHANDLER SIMPSON INVESTIGATION")
    print("=" * 80)
    print(f"As of: {today.isoformat()}")
    print()

    # Step 1: Check all three BDL ID mappings
    print("Step 1: BDL ID mappings for Chandler Simpson...")
    print("-" * 80)

    result = db.execute(text("""
        SELECT
            bdl_id,
            full_name,
            mlbam_id,
            is_primary
        FROM player_id_mapping
        WHERE LOWER(full_name) LIKE '%chandler%simpson%'
        ORDER BY bdl_id NULLS LAST
    """))

    rows = result.fetchall()
    if rows:
        for row in rows:
            bdl_id, full_name, mlbam_id, is_primary = row
            print(f"  BDL ID: {bdl_id or 'None'}, Name: {full_name}, MLBAM: {mlbam_id}, Primary: {is_primary}")
    else:
        print("  ⚠️  No mappings found for Chandler Simpson")
    print()

    # Step 2: Get momentum data for both BDL IDs
    print("Step 2: Momentum data for both BDL IDs...")
    print("-" * 80)

    for bdl_id in CHANDLER_BDL_IDS:
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
            LIMIT 10
        """), {"bdl_id": bdl_id})

        rows = result.fetchall()
        if not rows:
            print(f"\n⚠️  No momentum records for BDL ID: {bdl_id}")
            print()
            continue

        print(f"\n📊 BDL ID: {bdl_id}")
        print("-" * 80)

        for row in rows:
            (
                bdl_id, as_of_date, player_type, delta_z, signal,
                composite_z_14d, composite_z_30d, score_14d, score_30d,
                confidence_14d, confidence_30d, confidence, computed_at
            ) = row

            print(f"  Date:        {as_of_date}")
            print(f"  Signal:      {signal}")
            print(f"  Delta Z:     {delta_z:.4f}")
            print(f"  14d Z:       {composite_z_14d:.4f} (score: {score_14d:.1f}, conf: {confidence_14d:.2f})")
            print(f"  30d Z:       {composite_z_30d:.4f} (score: {score_30d:.1f}, conf: {confidence_30d:.2f})")
            print(f"  Confidence:  {confidence:.2f}")
            print(f"  Computed:    {computed_at}")
            print()

    # Step 3: Get cohort 25th percentile for score_14d (level gate)
    print("Step 3: Level gate check (score_14d < 25th percentile?)...")
    print("-" * 80)

    result = db.execute(text("""
        SELECT
            COUNT(*) as cohort_size,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY score_14d) as p25_score_14d,
            AVG(score_14d) as mean_score_14d,
            STDDEV(score_14d) as std_score_14d
        FROM player_momentum
        WHERE as_of_date = (
            SELECT as_of_date
            FROM player_momentum
            WHERE bdl_player_id = 653217
            ORDER BY as_of_date DESC
            LIMIT 1
        )
          AND player_type = 'hitter'
    """))

    row = result.fetchone()
    if row:
        cohort_size, p25_score_14d, mean_score_14d, std_score_14d = row

        print(f"Cohort Size:              {cohort_size}")
        print(f"25th Percentile (score_14d): {p25_score_14d:.1f}")
        print(f"Mean (score_14d):          {mean_score_14d:.1f}")
        print(f"StdDev (score_14d):        {std_score_14d:.1f}")
        print()

        # Check if Chandler's score_14d is below 25th percentile
        result = db.execute(text("""
            SELECT score_14d, confidence, signal
            FROM player_momentum
            WHERE bdl_player_id = 653217
            ORDER BY as_of_date DESC
            LIMIT 1
        """))

        row = result.fetchone()
        if row:
            score_14d, confidence, signal = row

            print(f"Chandler Simpson (BDL 653217):")
            print(f"  score_14d:    {score_14d:.1f}")
            print(f"  25th pct:     {p25_score_14d:.1f}")
            print(f"  confidence:   {confidence:.2f}")
            print(f"  signal:       {signal}")
            print()

            if score_14d < p25_score_14d:
                print(f"❌ LEVEL GATE TRIGGERED: score_14d ({score_14d:.1f}) < 25th percentile ({p25_score_14d:.1f})")
                print(f"   This forces downgrade to COLD regardless of delta_z.")
            else:
                print(f"✅ Level gate NOT triggered: score_14d ({score_14d:.1f}) >= 25th percentile ({p25_score_14d:.1f})")

            if confidence < 0.60:
                print(f"❌ CONFIDENCE GATE TRIGGERED: confidence ({confidence:.2f}) < 0.60")
                print(f"   This may force downgrade to COLD.")
            else:
                print(f"✅ Confidence gate NOT triggered: confidence ({confidence:.2f}) >= 0.60")
            print()
    print()

    # Step 4: Manual signal classification (simulate classify_signal)
    print("Step 4: Manual signal classification simulation...")
    print("-" * 80)

    # Get cohort delta_z distribution
    result = db.execute(text("""
        SELECT
            AVG(delta_z) as mean_delta,
            STDDEV(delta_z) as std_delta,
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY delta_z) as p90_delta,
            PERCENTILE_CONT(0.70) WITHIN GROUP (ORDER BY delta_z) as p70_delta,
            PERCENTILE_CONT(0.30) WITHIN GROUP (ORDER BY delta_z) as p30_delta,
            PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY delta_z) as p10_delta
        FROM player_momentum
        WHERE as_of_date = (
            SELECT as_of_date
            FROM player_momentum
            WHERE bdl_player_id = 653217
            ORDER BY as_of_date DESC
            LIMIT 1
        )
          AND player_type = 'hitter'
    """))

    row = result.fetchone()
    if row:
        mean_delta, std_delta, p90_delta, p70_delta, p30_delta, p10_delta = row

        print(f"Cohort Delta Z Thresholds:")
        print(f"  90th percentile (SURGING):  {p90_delta:.4f}")
        print(f"  70th percentile (HOT):      {p70_delta:.4f}")
        print(f"  30th percentile (COLD):     {p30_delta:.4f}")
        print(f"  10th percentile (COLLAPSING): {p10_delta:.4f}")
        print(f"  Mean: {mean_delta:.4f}, StdDev: {std_delta:.4f}")
        print()

        # Compute z-score delta
        result = db.execute(text("""
            SELECT delta_z, score_14d, confidence, signal
            FROM player_momentum
            WHERE bdl_player_id = 653217
            ORDER BY as_of_date DESC
            LIMIT 1
        """))

        row = result.fetchone()
        if row:
            delta_z, score_14d, confidence, signal = row

            z_score_delta = (delta_z - mean_delta) / std_delta if std_delta > 0 else 0.0

            print(f"Chandler Simpson (BDL 653217):")
            print(f"  delta_z:       {delta_z:.4f}")
            print(f"  z_score_delta: {z_score_delta:.4f}")
            print(f"  score_14d:     {score_14d:.1f}")
            print(f"  confidence:    {confidence:.2f}")
            print(f"  Actual signal: {signal}")
            print()

            # Determine expected signal based on delta_z
            if delta_z >= p90_delta:
                expected_signal = "SURGING"
            elif delta_z >= p70_delta:
                expected_signal = "HOT"
            elif delta_z >= p30_delta:
                expected_signal = "STABLE"
            elif delta_z >= p10_delta:
                expected_signal = "COLD"
            else:
                expected_signal = "COLLAPSING"

            print(f"Expected signal (delta_z only): {expected_signal}")
            print()

            if signal != expected_signal:
                print(f"❌ SIGNAL MISMATCH: Actual {signal} != Expected {expected_signal}")
                print(f"   Root cause: Level gate or confidence gate overriding delta_z classification")
            else:
                print(f"✅ Signal matches delta_z classification")
            print()
    print()

    # Step 5: Check if signal is stale (computed earlier than delta_z update)
    print("Step 5: Stale signal check...")
    print("-" * 80)

    result = db.execute(text("""
        SELECT
            as_of_date,
            delta_z,
            signal,
            computed_at
        FROM player_momentum
        WHERE bdl_player_id = 653217
        ORDER BY as_of_date DESC
        LIMIT 5
    """))

    rows = result.fetchall()
    if rows:
        print(f"Last 5 records for Chandler Simpson (BDL 653217):")
        for row in rows:
            as_of_date, delta_z, signal, computed_at = row
            print(f"  {as_of_date}: delta_z={delta_z:.4f}, signal={signal}, computed={computed_at}")

        # Check if signal stayed COLD while delta_z changed
        if len(rows) >= 2:
            latest_signal = rows[0][2]
            prev_delta_z = rows[1][1]
            latest_delta_z = rows[0][1]

            if latest_signal == prev_signal == "COLD":
                if abs(latest_delta_z - prev_delta_z) > 0.01:
                    print()
                    print(f"❌ POTENTIAL STALE SIGNAL: Signal stayed COLD while delta_z changed")
                    print(f"   Previous delta_z: {prev_delta_z:.4f}")
                    print(f"   Latest delta_z:    {latest_delta_z:.4f}")
                else:
                    print()
                    print(f"✅ Signal consistent with delta_z")
    print()

    # Step 6: Check player_scores for 14d vs 30d windows
    print("Step 6: Player scores (14d vs 30d) for Chandler Simpson...")
    print("-" * 80)

    for bdl_id in CHANDLER_BDL_IDS:
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
              AND as_of_date = (
                  SELECT as_of_date
                  FROM player_scores
                  WHERE bdl_player_id = :bdl_id
                  ORDER BY as_of_date DESC
                  LIMIT 1
              )
              AND window_days IN (14, 30)
            ORDER BY window_days
        """), {"bdl_id": bdl_id})

        rows = result.fetchall()
        if not rows:
            print(f"\n⚠️  No player_scores for BDL ID: {bdl_id}")
            print()
            continue

        print(f"\n📊 BDL ID: {bdl_id}")
        print("-" * 80)

        for row in rows:
            bdl_id, as_of_date, window_days, composite_z, score_0_100, confidence, computed_at = row
            print(f"  Window:     {window_days} days")
            print(f"  Composite Z: {composite_z:.4f}")
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

    db.close()

    print("=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()