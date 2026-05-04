import os
from sqlalchemy import create_engine, text

e = create_engine(os.environ["DATABASE_URL"])
with e.connect() as c:
    # Sample positions values for pitchers and hitters
    r = c.execute(text(
        "SELECT player_name, player_type, positions "
        "FROM player_projections "
        "LIMIT 20"
    ))
    rows = r.fetchall()
    print(f"{'name':<25} {'type':<10} positions")
    print("-" * 70)
    for row in rows:
        print(f"{str(row[0]):<25} {str(row[1]):<10} {row[2]}")

    # Count NULLs
    null_ct = c.execute(text(
        "SELECT COUNT(*) FROM player_projections WHERE player_type IS NULL"
    )).scalar()
    total = c.execute(text("SELECT COUNT(*) FROM player_projections")).scalar()
    print(f"\nNULL player_type: {null_ct}/{total}")

    # Sample NULL rows to see positions format
    print("\nSample NULL rows:")
    r2 = c.execute(text(
        "SELECT player_name, positions FROM player_projections "
        "WHERE player_type IS NULL LIMIT 10"
    ))
    for row in r2:
        print(f"  {row[0]}: {row[1]}")
