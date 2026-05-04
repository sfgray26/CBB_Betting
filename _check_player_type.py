from sqlalchemy import create_engine, text
import os

engine = create_engine(os.environ["DATABASE_URL"])
with engine.connect() as conn:
    r = conn.execute(text(
        "SELECT player_type, COUNT(*) as cnt FROM player_projections GROUP BY player_type ORDER BY player_type NULLS FIRST"
    ))
    rows = r.fetchall()
    total = sum(row[1] for row in rows)
    print(f"{'player_type':<12} {'count':>8}  {'pct':>6}")
    print("-" * 32)
    for row in rows:
        ptype = str(row[0]) if row[0] is not None else "NULL"
        pct = row[1] / total * 100 if total else 0
        print(f"{ptype:<12} {row[1]:>8}  {pct:>5.1f}%")
    print(f"{'TOTAL':<12} {total:>8}")
