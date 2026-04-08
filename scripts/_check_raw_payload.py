"""Check raw_payload from BDL to see what fields are returned."""
import os, json
from sqlalchemy import create_engine, text

e = create_engine(os.environ["DATABASE_URL"])
with e.connect() as c:
    # Sample hitter (has home_runs but no ab)
    r = c.execute(text(
        "SELECT bdl_player_id, game_date, ab, hits, home_runs, "
        "innings_pitched, raw_payload "
        "FROM mlb_player_stats WHERE home_runs IS NOT NULL "
        "AND innings_pitched IS NULL LIMIT 1"
    )).fetchone()
    if r:
        print(f"=== HITTER (BDL#{r.bdl_player_id}, {r.game_date}) ===")
        print(f"  ab={r.ab}, hits={r.hits}, hr={r.home_runs}, ip={r.innings_pitched}")
        if r.raw_payload:
            payload = r.raw_payload if isinstance(r.raw_payload, dict) else json.loads(r.raw_payload)
            print(f"\n  RAW PAYLOAD KEYS: {sorted(payload.keys())}")
            print(f"\n  FULL PAYLOAD:")
            for k, v in sorted(payload.items()):
                print(f"    {k}: {v}")
    else:
        print("No hitter with HR but no IP found")

    # Sample pitcher
    print()
    r2 = c.execute(text(
        "SELECT bdl_player_id, game_date, ab, hits, home_runs, "
        "innings_pitched, earned_runs, raw_payload "
        "FROM mlb_player_stats WHERE innings_pitched IS NOT NULL LIMIT 1"
    )).fetchone()
    if r2:
        print(f"=== PITCHER (BDL#{r2.bdl_player_id}, {r2.game_date}) ===")
        print(f"  ab={r2.ab}, hits={r2.hits}, hr={r2.home_runs}, ip={r2.innings_pitched}")
        if r2.raw_payload:
            payload2 = r2.raw_payload if isinstance(r2.raw_payload, dict) else json.loads(r2.raw_payload)
            print(f"\n  RAW PAYLOAD KEYS: {sorted(payload2.keys())}")
            print(f"\n  FULL PAYLOAD:")
            for k, v in sorted(payload2.items()):
                print(f"    {k}: {v}")
    
    # Count rows by type
    print(f"\n=== ROW BREAKDOWN ===")
    counts = c.execute(text("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN innings_pitched IS NOT NULL THEN 1 ELSE 0 END) as pitchers,
            SUM(CASE WHEN innings_pitched IS NULL AND home_runs IS NOT NULL THEN 1 ELSE 0 END) as hitters,
            SUM(CASE WHEN innings_pitched IS NULL AND home_runs IS NULL THEN 1 ELSE 0 END) as neither
        FROM mlb_player_stats
    """)).fetchone()
    print(f"  Total: {counts.total}")
    print(f"  Pitchers (has IP): {counts.pitchers}")
    print(f"  Hitters (has HR, no IP): {counts.hitters}")
    print(f"  Neither: {counts.neither}")
