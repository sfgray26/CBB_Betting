#!/usr/bin/env python
"""Check Yahoo ID sync status in production database."""
import os
import sys
from sqlalchemy import create_engine, text

db_url = os.environ.get("DATABASE_URL", "")
if not db_url:
    print("ERROR: DATABASE_URL not set")
    sys.exit(1)

engine = create_engine(db_url)

with engine.connect() as conn:
    # Overall counts
    result = conn.execute(text("""
        SELECT
            COUNT(*) as total,
            COUNT(yahoo_key) as with_yahoo_key,
            COUNT(bdl_id) as with_bdl_id,
            COUNT(yahoo_id) as with_yahoo_id
        FROM player_id_mapping
    """))
    row = result.fetchone()
    print(f"Total: {row[0]} | yahoo_key: {row[1]} | bdl_id: {row[2]} | yahoo_id: {row[3]}")

    # Sample rows with yahoo_key
    result2 = conn.execute(text("""
        SELECT full_name, yahoo_key, yahoo_id, bdl_id
        FROM player_id_mapping
        WHERE yahoo_key IS NOT NULL
        LIMIT 5
    """))
    print("Sample rows with yahoo_key:")
    for r in result2:
        print(f"  {r[0]} | yahoo_key: {r[1]} | yahoo_id: {r[2]} | bdl_id: {r[3]}")
