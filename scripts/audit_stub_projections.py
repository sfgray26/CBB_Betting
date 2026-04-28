"""
Audit player_projections for stub/default contamination.

Identifies:
  - Batter stubs:   hr=15, r=65, rbi=65, sb=5, avg=0.250, ops=0.720
  - Pitcher stubs:  era=4.0, whip=1.3, w=0, qs=0, k_pit=0
  - Zombie rows:    match BOTH batter AND pitcher stub patterns simultaneously
  - Linked rows:    joined to player_id_mapping where yahoo_key is NOT NULL

Hard stop: prints findings + sample rows. Does NOT delete anything.
Decision rule for safe deletion: Zombie rows + prior_source IN ('stub','default').
Steamer rows are preserved regardless.
"""

import os
import sys

# Add repo root so backend imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set")
    sys.exit(1)

engine = create_engine(DATABASE_URL)

BATTER_STUB_CONDITIONS = """
    hr = 15
    AND r = 65
    AND rbi = 65
    AND sb = 5
    AND ABS(avg - 0.250) < 0.001
    AND ABS(ops - 0.720) < 0.001
"""

PITCHER_STUB_CONDITIONS = """
    ABS(era - 4.0) < 0.001
    AND ABS(whip - 1.3) < 0.001
    AND w = 0
    AND qs = 0
    AND k_pit = 0
"""

def run_audit():
    with engine.connect() as conn:
        # --- Total rows ---
        total = conn.execute(text("SELECT COUNT(*) FROM player_projections")).scalar()
        print(f"\n=== player_projections audit ===")
        print(f"Total rows: {total}")

        # --- Batter stubs ---
        batter_count = conn.execute(text(f"""
            SELECT COUNT(*) FROM player_projections
            WHERE {BATTER_STUB_CONDITIONS}
        """)).scalar()
        print(f"\nBatter stubs (hr=15,r=65,rbi=65,sb=5,avg=0.25,ops=0.72): {batter_count}")

        # --- Pitcher stubs ---
        pitcher_count = conn.execute(text(f"""
            SELECT COUNT(*) FROM player_projections
            WHERE {PITCHER_STUB_CONDITIONS}
        """)).scalar()
        print(f"Pitcher stubs (era=4.0,whip=1.3,w=0,qs=0,k_pit=0):        {pitcher_count}")

        # --- Zombie rows (match BOTH patterns) ---
        zombie_count = conn.execute(text(f"""
            SELECT COUNT(*) FROM player_projections
            WHERE ({BATTER_STUB_CONDITIONS})
            AND ({PITCHER_STUB_CONDITIONS})
        """)).scalar()
        print(f"Zombie rows (match both batter + pitcher defaults):         {zombie_count}")

        # --- Prior source breakdown ---
        print("\nPrior source breakdown:")
        rows = conn.execute(text("""
            SELECT prior_source, COUNT(*) AS cnt
            FROM player_projections
            GROUP BY prior_source
            ORDER BY cnt DESC
        """)).fetchall()
        for r in rows:
            print(f"  {r[0]:<20} {r[1]}")

        # --- Rows where prior_source = 'stub' or 'default' ---
        stub_source_count = conn.execute(text("""
            SELECT COUNT(*) FROM player_projections
            WHERE prior_source IN ('stub', 'default')
        """)).scalar()
        print(f"\nRows with prior_source IN ('stub','default'): {stub_source_count}")

        # --- Linked rows (projections with a yahoo_key match) ---
        linked_count = conn.execute(text("""
            SELECT COUNT(DISTINCT pp.id)
            FROM player_projections pp
            JOIN player_id_mapping pim
              ON pp.player_id = pim.yahoo_id
              OR pp.player_id = pim.yahoo_key
            WHERE pim.yahoo_key IS NOT NULL
        """)).scalar()
        print(f"Linked rows (joined to player_id_mapping with yahoo_key):  {linked_count}")

        # --- Sample zombie rows ---
        if zombie_count > 0:
            print(f"\nSample zombie rows (up to 5):")
            sample = conn.execute(text(f"""
                SELECT id, player_id, player_name, team, prior_source, update_method,
                       hr, r, rbi, sb, avg, ops, era, whip, w, qs, k_pit
                FROM player_projections
                WHERE ({BATTER_STUB_CONDITIONS})
                AND ({PITCHER_STUB_CONDITIONS})
                LIMIT 5
            """)).fetchall()
            cols = ["id","player_id","player_name","team","prior_source","update_method",
                    "hr","r","rbi","sb","avg","ops","era","whip","w","qs","k_pit"]
            print("  " + " | ".join(f"{c:<14}" for c in cols))
            for row in sample:
                print("  " + " | ".join(f"{str(v):<14}" for v in row))

        # --- Sample stub-source rows ---
        if stub_source_count > 0:
            print(f"\nSample prior_source='stub'/'default' rows (up to 5):")
            sample = conn.execute(text("""
                SELECT id, player_id, player_name, team, prior_source, update_method,
                       hr, r, rbi, avg, era, whip
                FROM player_projections
                WHERE prior_source IN ('stub', 'default')
                LIMIT 5
            """)).fetchall()
            cols = ["id","player_id","player_name","team","prior_source","update_method",
                    "hr","r","rbi","avg","era","whip"]
            print("  " + " | ".join(f"{c:<14}" for c in cols))
            for row in sample:
                print("  " + " | ".join(f"{str(v):<14}" for v in row))

        # --- Safe-delete candidates (Zombie OR stub source, NOT Steamer) ---
        safe_delete_count = conn.execute(text(f"""
            SELECT COUNT(*) FROM player_projections
            WHERE (
                ({BATTER_STUB_CONDITIONS})
                AND ({PITCHER_STUB_CONDITIONS})
            )
            OR prior_source IN ('stub', 'default')
        """)).scalar()
        print(f"\n--- SUMMARY ---")
        print(f"Safe-delete candidates (Zombie OR stub source): {safe_delete_count}")
        print(f"Steamer rows preserved:                         {total - safe_delete_count}")
        print("\nHard stop: DO NOT delete without architect approval.")
        print("Report this output and await instruction.")

if __name__ == "__main__":
    run_audit()
