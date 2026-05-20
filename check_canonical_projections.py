#!/usr/bin/env python3
"""
Check CanonicalProjection table specifically
"""

import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

os.chdir('/mnt/c/Users/sfgra/repos/Fixed/cbb-edge')
sys.path.insert(0, '/mnt/c/Users/sfgra/repos/Fixed/cbb-edge')

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text, inspect

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

def run_query(query, params=None):
    with engine.connect() as conn:
        result = conn.execute(text(query), params or {})
        return [dict(row._mapping) for row in result]

print("=" * 80)
print("CANONICAL PROJECTIONS ANALYSIS")
print("=" * 80)

# Check schema
try:
    inspector = inspect(engine)
    columns = inspector.get_columns('canonical_projections')
    print("\nTable Schema:")
    for col in columns:
        print(f"  - {col['name']}: {col['type']}")
except Exception as e:
    print(f"Error getting schema: {e}")

# Check data
try:
    result = run_query("SELECT COUNT(*) as cnt FROM canonical_projections")
    count = result[0]['cnt']
    print(f"\nTotal rows: {count}")
except Exception as e:
    print(f"Error counting: {e}")

# Check for recent data
try:
    result = run_query("""
        SELECT MAX(created_at) as latest, MIN(created_at) as earliest
        FROM canonical_projections
    """)
    if result and result[0]['latest']:
        print(f"Latest record: {result[0]['latest']}")
        print(f"Earliest record: {result[0]['earliest']}")
except Exception as e:
    print(f"Error getting dates: {e}")

# Check for nulls in critical columns
try:
    critical_cols = ['player_id', 'source', 'projected_value', 'fpts_projection']
    print("\nNull check in critical fields:")
    for col in critical_cols:
        try:
            result = run_query(f"""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) as nulls
                FROM canonical_projections
            """)
            if result:
                total = result[0]['total']
                nulls = result[0]['nulls'] or 0
                if total > 0:
                    pct = (nulls / total) * 100
                    print(f"  {col}: {nulls}/{total} null ({pct:.1f}%)")
        except Exception as e:
            print(f"  {col}: Error - {e}")
except Exception as e:
    print(f"Error checking nulls: {e}")

# Check by source
try:
    result = run_query("""
        SELECT source, COUNT(*) as cnt, MAX(created_at) as latest
        FROM canonical_projections
        GROUP BY source
    """)
    print("\nBy source:")
    for row in result:
        print(f"  {row['source']}: {row['cnt']} rows (latest: {row['latest']})")
except Exception as e:
    print(f"Error grouping by source: {e}")

print("\n" + "=" * 80)

# Also check player_projections in detail
print("\nPLAYER PROJECTIONS ANALYSIS")
print("=" * 80)

try:
    inspector = inspect(engine)
    columns = inspector.get_columns('player_projections')
    print("\nTable Schema:")
    for col in columns:
        print(f"  - {col['name']}: {col['type']}")
except Exception as e:
    print(f"Error getting schema: {e}")

try:
    result = run_query("SELECT COUNT(*) as cnt FROM player_projections")
    count = result[0]['cnt']
    print(f"\nTotal rows: {count}")
except Exception as e:
    print(f"Error counting: {e}")

try:
    result = run_query("""
        SELECT MAX(created_at) as latest, MIN(created_at) as earliest
        FROM player_projections
    """)
    if result and result[0]['latest']:
        print(f"Latest record: {result[0]['latest']}")
        print(f"Earliest record: {result[0]['earliest']}")
except Exception as e:
    print(f"Error getting dates: {e}")

# Sample data
try:
    result = run_query("""
        SELECT * FROM player_projections
        LIMIT 3
    """)
    if result:
        print("\nSample data (first row):")
        for key, val in result[0].items():
            print(f"  {key}: {val}")
except Exception as e:
    print(f"Error getting sample: {e}")

print("\n" + "=" * 80)
