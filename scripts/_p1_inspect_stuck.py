"""P-1 Step 4: Inspect the 8 stuck rows + 1639 unbackfillable sample."""
from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()
try:
    # The 8 rows that my diagnostic says are backfillable but endpoint ignores
    print("=== STUCK WHIP ROWS (backfillable per diagnostic) ===")
    rows = db.execute(text("""
        SELECT id, bdl_player_id, walks_allowed, hits_allowed, innings_pitched,
               whip, game_date
        FROM mlb_player_stats
        WHERE whip IS NULL
          AND walks_allowed IS NOT NULL AND hits_allowed IS NOT NULL
          AND innings_pitched IS NOT NULL AND innings_pitched != ''
        LIMIT 10
    """)).fetchall()
    for r in rows:
        print(dict(r._mapping))

    # Sample unbackfillable ops
    print("\n=== UNBACKFILLABLE OPS SAMPLE ===")
    rows = db.execute(text("""
        SELECT id, bdl_player_id, at_bats, hits, obp, slg, ops,
               innings_pitched, position
        FROM mlb_player_stats
        WHERE ops IS NULL AND (obp IS NULL OR slg IS NULL)
        LIMIT 10
    """)).fetchall()
    for r in rows:
        print(dict(r._mapping))
finally:
    db.close()
