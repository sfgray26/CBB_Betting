"""P-3 Step 6: Export unmatchable orphans to CSV."""
import csv
from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()
try:
    rows = db.execute(text("""
        SELECT id, yahoo_player_key, player_name, primary_position, player_type
        FROM position_eligibility
        WHERE bdl_player_id IS NULL
        ORDER BY player_name, yahoo_player_key
    """)).fetchall()
    out_path = "reports/2026-04-11-unmatchable-orphans.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "yahoo_player_key", "player_name", "primary_position", "player_type"])
        for r in rows:
            w.writerow([r.id, r.yahoo_player_key, r.player_name, r.primary_position, r.player_type])
    print(f"Exported {len(rows)} rows to {out_path}")

    # Quick breakdown by player_type
    batters = sum(1 for r in rows if r.player_type == "batter")
    pitchers = sum(1 for r in rows if r.player_type == "pitcher")
    print(f"  batters:  {batters}")
    print(f"  pitchers: {pitchers}")
finally:
    db.close()
