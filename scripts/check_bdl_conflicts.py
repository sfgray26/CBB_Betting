#!/usr/bin/env python
"""Check for bdl_id and yahoo_key conflicts in player_id_mapping."""
import os
import sys
from sqlalchemy import create_engine, text

db_url = os.environ.get("DATABASE_URL", "")
if not db_url:
    print("ERROR: DATABASE_URL not set")
    sys.exit(1)

engine = create_engine(db_url)

with engine.connect() as conn:
    # Check for duplicate bdl_ids
    result = conn.execute(text("""
        SELECT bdl_id, COUNT(*) as cnt
        FROM player_id_mapping
        WHERE bdl_id IS NOT NULL
        GROUP BY bdl_id
        HAVING COUNT(*) > 1
        ORDER BY cnt DESC
        LIMIT 10
    """))
    dup_bdl = result.fetchall()
    print(f"Found {len(dup_bdl)} bdl_id conflicts:")
    for row in dup_bdl:
        print(f"  bdl_id={row[0]} appears {row[1]} times")

    # Show sample of conflict for bdl_id=1607
    if dup_bdl:
        print("\nSample conflict rows (bdl_id=1607):")
        result2 = conn.execute(text("""
            SELECT id, yahoo_key, yahoo_id, bdl_id, full_name, source
            FROM player_id_mapping
            WHERE bdl_id = 1607
            ORDER BY id
        """))
        for r in result2:
            print(f"  id={r[0]} | yahoo_key={r[1]} | yahoo_id={r[2]} | bdl_id={r[3]} | name={r[4]} | source={r[5]}")

    # Check for duplicate yahoo_keys
    result3 = conn.execute(text("""
        SELECT yahoo_key, COUNT(*) as cnt
        FROM player_id_mapping
        WHERE yahoo_key IS NOT NULL
        GROUP BY yahoo_key
        HAVING COUNT(*) > 1
        ORDER BY cnt DESC
        LIMIT 10
    """))
    dup_yahoo = result3.fetchall()
    print(f"\nFound {len(dup_yahoo)} yahoo_key conflicts:")
    for row in dup_yahoo:
        print(f"  yahoo_key={row[0]} appears {row[1]} times")

    # Overall stats
    result4 = conn.execute(text("""
        SELECT
            COUNT(*) as total,
            COUNT(yahoo_key) as with_yahoo_key,
            COUNT(bdl_id) as with_bdl_id,
            COUNT(yahoo_id) as with_yahoo_id
        FROM player_id_mapping
    """))
    row = result4.fetchone()
    print(f"\nTotal: {row[0]} | yahoo_key: {row[1]} | bdl_id: {row[2]} | yahoo_id: {row[3]}")
