import os
from sqlalchemy import create_engine, text

e = create_engine(os.environ["DATABASE_URL"])
with e.connect() as c:
    for table in ["projection_snapshots", "projection_cache_entries", "player_id_mapping"]:
        print(f"\n=== {table} ===")
        # Get column names
        r = c.execute(text(
            f"SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_name='{table}' ORDER BY ordinal_position"
        ))
        cols = r.fetchall()
        for col in cols:
            print(f"  {col[0]}: {col[1]}")
        # Sample row
        ct = c.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
        print(f"  => {ct} rows")
        if ct > 0:
            r2 = c.execute(text(f"SELECT * FROM {table} LIMIT 1"))
            row = r2.fetchone()
            if row:
                print(f"  Sample: {dict(row._mapping)}")

    # Also check player_projections columns even though empty
    print("\n=== player_projections columns ===")
    r = c.execute(text(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name='player_projections' ORDER BY ordinal_position"
    ))
    for col in r:
        print(f"  {col[0]}: {col[1]}")
