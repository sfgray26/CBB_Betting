"""
Root cause investigation script for ops/whip NULL issues.

Run this on Railway to diagnose why computed fields are NULL.
"""

import sys
sys.path.insert(0, ".")

from backend.models import SessionLocal, MLBPlayerStats
from sqlalchemy import text

db = SessionLocal()

print("=" * 80)
print("ROOT CAUSE INVESTIGATION: ops/whip NULL issues")
print("=" * 80)
print()

# Investigation 1: Check if obp/slg data exists
print("INVESTIGATION 1: Does source data (obp/slg) exist?")
result = db.execute(text("""
    SELECT
        COUNT(*) as total_rows,
        COUNT(obp) as has_obp,
        COUNT(slg) as has_slg,
        COUNT(ops) as has_ops,
        COUNT(*) FILTER (WHERE obp IS NOT NULL AND slg IS NOT NULL) as has_both
    FROM mlb_player_stats
""")).fetchone()

print(f"Total rows: {result.total_rows}")
print(f"Has obp: {result.has_obp}")
print(f"Has slg: {result.has_slg}")
print(f"Has ops: {result.has_ops}")
print(f"Has BOTH obp and slg: {result.has_both}")
print()

# Investigation 2: Sample rows with obp/slg to see actual values
print("INVESTIGATION 2: Sample rows with OBP/SLG data")
result = db.execute(text("""
    SELECT bdl_player_id, obp, slg, ops, game_date
    FROM mlb_player_stats
    WHERE obp IS NOT NULL OR slg IS NOT NULL
    ORDER BY game_date DESC
    LIMIT 5
""")).fetchall()

print(f"Sample rows:")
for i, row in enumerate(result, 1):
    print(f"  {i}. Player {row.bdl_player_id}: obp={row.obp}, slg={row.slg}, ops={row.ops}, date={row.game_date}")
print()

# Investigation 3: Check if raw_payload has the data
print("INVESTIGATION 3: Does raw_payload contain obp/slg?")
result = db.execute(text("""
    SELECT
        COUNT(*) FILTER (WHERE raw_payload::text LIKE '%obp%') as has_obp_payload,
        COUNT(*) FILTER (WHERE raw_payload::text LIKE '%slg%') as has_slg_payload,
        COUNT(*) FILTER (WHERE raw_payload::text LIKE '%ops%') as has_ops_payload
    FROM mlb_player_stats
""")).fetchone()

print(f"Rows with 'obp' in raw_payload: {result.has_obp_payload}")
print(f"Rows with 'slg' in raw_payload: {result.has_slg_payload}")
print(f"Rows with 'ops' in raw_payload: {result.has_ops_payload}")
print()

# Investigation 4: Check whip components
print("INVESTIGATION 4: Does WHIP source data exist?")
result = db.execute(text("""
    SELECT
        COUNT(*) as total_rows,
        COUNT(walks_allowed) as has_bb,
        COUNT(hits_allowed) as has_h,
        COUNT(whip) as has_whip,
        COUNT(*) FILTER (WHERE walks_allowed IS NOT NULL AND hits_allowed IS NOT NULL) as has_components
    FROM mlb_player_stats
""")).fetchone()

print(f"Total rows: {result.total_rows}")
print(f"Has walks_allowed: {result.has_bb}")
print(f"Has hits_allowed: {result.has_h}")
print(f"Has whip: {result.has_whip}")
print(f"Has BOTH components: {result.has_components}")
print()

# Investigation 5: Check ERA anomaly
print("INVESTIGATION 5: ERA > 100 issue")
result = db.execute(text("""
    SELECT bdl_player_id, era, earned_runs, innings_pitched, game_date, opponent_team
    FROM mlb_player_stats
    WHERE era > 100
    ORDER BY era DESC
    LIMIT 1
""")).fetchone()

if result:
    print(f"Found ERA > 100:")
    print(f"  Player: {result.bdl_player_id}")
    print(f"  ERA: {result.era}")
    print(f"  Earned Runs: {result.earned_runs}")
    print(f"  Innings Pitched: {result.innings_pitched}")
    print(f"  Game Date: {result.game_date}")
    print(f"  Opponent: {result.opponent_team}")
else:
    print("No ERA > 100 found (already fixed?)")
print()

# Investigation 6: Orphaned position_eligibility
print("INVESTIGATION 6: Orphaned position_eligibility records")
result = db.execute(text("""
    SELECT COUNT(*) as orphaned
    FROM position_eligibility pe
    LEFT JOIN player_id_mapping pim ON pe.yahoo_player_key = pim.yahoo_key
    WHERE pe.yahoo_player_key IS NOT NULL AND pim.yahoo_key IS NULL
""")).fetchone()

print(f"Orphaned position_eligibility records: {result.orphaned}")

db.close()

print("=" * 80)
print("ROOT CAUSE ANALYSIS COMPLETE")
print("=" * 80)
