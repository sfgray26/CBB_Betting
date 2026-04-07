#!/usr/bin/env python
"""
UAT Pipeline Health Check
=========================
Run this locally against the Railway public DB URL to validate pipeline outputs.

Usage:
    DATABASE_URL=<public_url> python scripts/uat_pipeline_check.py

Get the public URL from Railway dashboard:
  Database service -> Connect -> Public URL (uses roundhouse.proxy.rlwy.net or similar)

Or pass via env:
  export DATABASE_URL="postgresql://postgres:<password>@roundhouse.proxy.rlwy.net:<port>/railway"
  python scripts/uat_pipeline_check.py
"""

import os
import sys
from datetime import date, timedelta

from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import sessionmaker

db_url = os.environ.get("DATABASE_URL")
if not db_url:
    print("ERROR: DATABASE_URL not set.")
    print("Get the public URL from Railway dashboard > DB service > Connect tab.")
    sys.exit(1)

engine = create_engine(db_url)
Session = sessionmaker(bind=engine)
db = Session()

yesterday = date.today() - timedelta(days=1)
SEPARATOR = "-" * 60


def section(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


# ── 1. Table health ────────────────────────────────────────────
section("1. TABLE HEALTH")

tables = [
    ("mlb_game_log",        "game_date"),
    ("mlb_player_stats",    "game_date"),
    ("player_rolling_stats","as_of_date"),
    ("player_scores",       "as_of_date"),
    ("player_momentum",     "as_of_date"),
    ("simulation_results",  "as_of_date"),
    ("player_id_mapping",   "created_at"),
]

for table, date_col in tables:
    try:
        row = db.execute(text(
            f"SELECT COUNT(*), MAX({date_col}) FROM {table}"
        )).fetchone()
        n, latest = row
        status = "OK" if n > 0 else "EMPTY"
        print(f"  [{status}] {table}: {n} rows, latest={latest}")
    except Exception as e:
        print(f"  [ERROR] {table}: {str(e)[:80]}")


# ── 2. Score distribution ──────────────────────────────────────
section("2. PLAYER SCORES (14d window, yesterday)")

try:
    rows = db.execute(text("""
        SELECT bdl_player_id, player_type, composite_z, score_0_100,
               z_hr, z_rbi, z_avg, z_era, z_whip, confidence
        FROM player_scores
        WHERE as_of_date = :d AND window_days = 14
        ORDER BY score_0_100 DESC
    """), {"d": yesterday}).fetchall()

    if not rows:
        print("  NO DATA — scoring job may not have run yet")
        print(f"  (Looked for as_of_date={yesterday})")
    else:
        hitters  = [r for r in rows if r.player_type == "hitter"]
        pitchers = [r for r in rows if r.player_type == "pitcher"]
        scores   = [r.score_0_100 for r in rows]
        mean_s   = sum(scores) / len(scores)
        print(f"  Total: {len(rows)} players ({len(hitters)} hitters, {len(pitchers)} pitchers)")
        print(f"  Score range: {min(scores):.1f} - {max(scores):.1f}  (expect 0-100)")
        print(f"  Mean score:  {mean_s:.1f}  (expect ~50)")
        print(f"  Confidence range: {min(r.confidence for r in rows):.2f} - {max(r.confidence for r in rows):.2f}")

        print("\n  TOP 10 PLAYERS:")
        for r in rows[:10]:
            hr  = f"{r.z_hr:.2f}"   if r.z_hr   is not None else "  --"
            avg = f"{r.z_avg:.2f}"  if r.z_avg  is not None else "  --"
            era = f"{r.z_era:.2f}"  if r.z_era  is not None else "  --"
            print(f"    id={r.bdl_player_id:6d} [{r.player_type:7s}]"
                  f"  score={r.score_0_100:5.1f}  z={r.composite_z:+.3f}"
                  f"  z_hr={hr}  z_avg={avg}  z_era={era}")

        print("\n  BOTTOM 5 PLAYERS:")
        for r in rows[-5:]:
            print(f"    id={r.bdl_player_id:6d} [{r.player_type:7s}]"
                  f"  score={r.score_0_100:5.1f}  z={r.composite_z:+.3f}")
except Exception as e:
    print(f"  ERROR: {e}")


# ── 3. Momentum distribution ───────────────────────────────────
section("3. MOMENTUM SIGNALS (yesterday)")

try:
    rows = db.execute(text("""
        SELECT signal, COUNT(*), AVG(delta_z), AVG(confidence)
        FROM player_momentum
        WHERE as_of_date = :d
        GROUP BY signal
        ORDER BY COUNT(*) DESC
    """), {"d": yesterday}).fetchall()

    if not rows:
        print("  NO DATA")
    else:
        total = sum(r[1] for r in rows)
        print(f"  Total players with momentum: {total}")
        for signal, count, avg_dz, avg_conf in rows:
            print(f"    {signal:12s}: {count:4d} ({count/total*100:4.0f}%)"
                  f"  avg_dz={avg_dz:+.3f}  avg_conf={avg_conf:.2f}")

        # Sanity check: STABLE should be largest bucket
        signals_dict = {r[0]: r[1] for r in rows}
        if signals_dict.get("STABLE", 0) < total * 0.3:
            print("  WARN: STABLE bucket < 30% — may indicate data issue")
        if signals_dict.get("SURGING", 0) > total * 0.4:
            print("  WARN: SURGING > 40% — may indicate Z-score normalization issue")

        print("\n  TOP 5 SURGING:")
        surging = db.execute(text("""
            SELECT bdl_player_id, delta_z, composite_z_14d, composite_z_30d, confidence
            FROM player_momentum
            WHERE as_of_date = :d AND signal = 'SURGING'
            ORDER BY delta_z DESC LIMIT 5
        """), {"d": yesterday}).fetchall()
        for r in surging:
            print(f"    id={r.bdl_player_id:6d}  dz={r.delta_z:+.3f}"
                  f"  z14={r.composite_z_14d:+.3f}  z30={r.composite_z_30d:+.3f}"
                  f"  conf={r.confidence:.2f}")
except Exception as e:
    print(f"  ERROR: {e}")


# ── 4. Monte Carlo sanity ──────────────────────────────────────
section("4. SIMULATION RESULTS (yesterday)")

try:
    rows = db.execute(text("""
        SELECT bdl_player_id, player_type,
               proj_hr_p25, proj_hr_p50, proj_hr_p75,
               proj_k_p50, proj_era_p50, proj_whip_p50
        FROM simulation_results
        WHERE as_of_date = :d
        ORDER BY proj_hr_p50 DESC NULLS LAST
        LIMIT 10
    """), {"d": yesterday}).fetchall()

    total_sim = db.execute(text(
        "SELECT COUNT(*) FROM simulation_results WHERE as_of_date = :d"
    ), {"d": yesterday}).scalar()

    if not total_sim:
        print("  NO DATA — simulation job may not have run yet")
    else:
        print(f"  Total simulated: {total_sim} players")
        print("\n  TOP HITTERS BY P50 HR PROJECTION:")
        for r in rows:
            if r.player_type == "hitter" and r.proj_hr_p50:
                print(f"    id={r.bdl_player_id:6d}  HR: P25={r.proj_hr_p25:.0f}"
                      f"  P50={r.proj_hr_p50:.0f}  P75={r.proj_hr_p75:.0f}")

        # ERA sanity — should be between 2.0 and 8.0 for P50
        bad_era = db.execute(text("""
            SELECT COUNT(*) FROM simulation_results
            WHERE as_of_date = :d AND player_type = 'pitcher'
            AND (proj_era_p50 < 0 OR proj_era_p50 > 20)
        """), {"d": yesterday}).scalar()
        if bad_era:
            print(f"\n  WARN: {bad_era} pitchers have ERA P50 outside [0, 20]")
        else:
            print("\n  ERA sanity: OK (all P50 ERA in reasonable range)")
except Exception as e:
    print(f"  ERROR: {e}")


# ── 5. Player identity mapping ─────────────────────────────────
section("5. PLAYER ID MAPPING")

try:
    total_map = db.execute(text("SELECT COUNT(*) FROM player_id_mapping")).scalar()
    resolved  = db.execute(text(
        "SELECT COUNT(*) FROM player_id_mapping WHERE mlbam_id IS NOT NULL"
    )).scalar()
    manual    = db.execute(text(
        "SELECT COUNT(*) FROM player_id_mapping WHERE source = 'manual'"
    )).scalar()
    pybaseball = db.execute(text(
        "SELECT COUNT(*) FROM player_id_mapping WHERE source = 'pybaseball'"
    )).scalar()
    print(f"  Total mappings: {total_map}")
    print(f"  With mlbam_id:  {resolved} ({resolved/max(total_map,1)*100:.0f}%)")
    print(f"  Source manual:  {manual}")
    print(f"  Source pybaseball: {pybaseball}")

    # Sample a few names
    samples = db.execute(text(
        "SELECT full_name, bdl_id, mlbam_id, source FROM player_id_mapping LIMIT 5"
    )).fetchall()
    print("\n  Sample entries:")
    for r in samples:
        print(f"    {r.full_name:25s}  bdl={r.bdl_id}  mlbam={r.mlbam_id}  src={r.source}")
except Exception as e:
    print(f"  ERROR: {e}")


db.close()
print(f"\n{SEPARATOR}")
print("  UAT COMPLETE")
print(SEPARATOR)
