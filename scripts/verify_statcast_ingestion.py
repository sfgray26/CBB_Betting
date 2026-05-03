#!/usr/bin/env python
"""Quick verification that Statcast ingestion worked."""

import os
import sys
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        # Try Railway connection string
        db_url = "postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway"

    engine = create_engine(db_url)

    with engine.begin() as conn:
        print("=== Statcast Ingestion Verification ===\n")

        # Check statcast_performances
        result = conn.execute(text('SELECT COUNT(*) FROM statcast_performances;'))
        count = result.fetchone()[0]
        print(f"statcast_performances: {count} rows")

        if count > 0:
            result = conn.execute(text('''
                SELECT MAX(game_date), COUNT(DISTINCT player_id) FROM statcast_performances;
            '''))
            max_date, players = result.fetchone()
            print(f"  - max game_date: {max_date}")
            print(f"  - unique players: {players}")
        else:
            print("  ❌ No data found!")

if __name__ == "__main__":
    main()
