"""P-1 Step 1 + Step 3: Diagnose NULL ops/whip residuals. Temporary."""
from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()
try:
    q = text("""
      SELECT
        COUNT(*) FILTER (WHERE ops IS NULL) AS null_ops,
        COUNT(*) FILTER (WHERE ops IS NULL AND obp IS NOT NULL AND slg IS NOT NULL) AS null_ops_backfillable,
        COUNT(*) FILTER (WHERE ops IS NULL AND (obp IS NULL OR slg IS NULL)) AS null_ops_unbackfillable,
        COUNT(*) FILTER (WHERE whip IS NULL) AS null_whip,
        COUNT(*) FILTER (WHERE whip IS NULL AND walks_allowed IS NOT NULL AND hits_allowed IS NOT NULL AND innings_pitched IS NOT NULL AND innings_pitched != '') AS null_whip_backfillable,
        COUNT(*) FILTER (WHERE whip IS NULL AND (walks_allowed IS NULL OR hits_allowed IS NULL OR innings_pitched IS NULL OR innings_pitched = '')) AS null_whip_unbackfillable,
        COUNT(*) AS total_rows
      FROM mlb_player_stats
    """)
    row = db.execute(q).fetchone()
    d = dict(row._mapping)
    for k, v in d.items():
        print(f"{k}: {v}")
finally:
    db.close()
