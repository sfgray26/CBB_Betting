#!/usr/bin/env python
"""Quick audit of fantasy baseball database state."""

import os
import sys
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)

    engine = create_engine(db_url)

    with engine.begin() as conn:
        print("=== Fantasy Baseball Database Audit (May 2, 2026) ===\n")

        # Check player_projections
        try:
            result = conn.execute(text('SELECT COUNT(*) FROM player_projections;'))
            proj_count = result.fetchone()[0]
            print(f"player_projections: {proj_count} rows")

            if proj_count > 0:
                result = conn.execute(text('''
                    SELECT
                        COUNT(*) FILTER (WHERE player_type = 'hitter') AS hitters,
                        COUNT(*) FILTER (WHERE player_type = 'pitcher') AS pitchers,
                        COUNT(*) FILTER (WHERE player_type IS NULL) AS null_type
                    FROM player_projections
                '''))
                row = result.fetchone()
                print(f"  - hitters: {row[0]}")
                print(f"  - pitchers: {row[1]}")
                print(f"  - null type: {row[2]}")
        except Exception as e:
            print(f"player_projections: ERROR - {e}")

        # Check statcast_performances
        try:
            result = conn.execute(text('SELECT COUNT(*) FROM statcast_performances;'))
            stat_count = result.fetchone()[0]
            print(f"\nstatcast_performances: {stat_count} rows")

            if stat_count > 0:
                result = conn.execute(text('SELECT MAX(game_date), COUNT(DISTINCT player_id) FROM statcast_performances;'))
                max_date, players = result.fetchone()
                print(f"  - max game_date: {max_date}")
                print(f"  - unique players: {players}")
        except Exception as e:
            print(f"statcast_performances: ERROR - {e}")

        # Check mlb_game_log
        try:
            result = conn.execute(text('SELECT COUNT(*) FROM mlb_game_log;'))
            game_count = result.fetchone()[0]
            print(f"\nmlb_game_log: {game_count} rows")
        except Exception as e:
            print(f"mlb_game_log: ERROR - {e}")

        # Check mlb_odds_snapshot
        try:
            result = conn.execute(text('SELECT COUNT(*) FROM mlb_odds_snapshot;'))
            odds_count = result.fetchone()[0]
            print(f"mlb_odds_snapshot: {odds_count} rows")
        except Exception as e:
            print(f"mlb_odds_snapshot: ERROR - {e}")

        # Check cat_scores presence
        try:
            result = conn.execute(text('''
                SELECT
                    COUNT(*) FILTER (WHERE cat_scores IS NOT NULL) AS has_cat_scores,
                    COUNT(*) FILTER (WHERE cat_scores IS NULL) AS no_cat_scores
                FROM player_projections
            '''))
            row = result.fetchone()
            print(f"\nplayer_projections cat_scores:")
            print(f"  - has cat_scores: {row[0]}")
            print(f"  - no cat_scores: {row[1]}")
        except Exception as e:
            print(f"cat_scores check: ERROR - {e}")

if __name__ == "__main__":
    main()
