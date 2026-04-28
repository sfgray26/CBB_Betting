#!/usr/bin/env python
"""Check probable_pitchers table status."""
import os
import sys
from datetime import date, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import SessionLocal
from sqlalchemy import text

def main():
    db = SessionLocal()
    try:
        # Check total rows
        total = db.execute(text('SELECT COUNT(*) FROM probable_pitchers')).scalar()
        print(f'Total rows: {total}')

        # Check rows for today/tomorrow
        today = date.today()
        tomorrow = today + timedelta(days=1)

        today_count = db.execute(
            text('SELECT COUNT(*) FROM probable_pitchers WHERE game_date = :today'),
            {'today': today}
        ).scalar()
        tomorrow_count = db.execute(
            text('SELECT COUNT(*) FROM probable_pitchers WHERE game_date = :tomorrow'),
            {'tomorrow': tomorrow}
        ).scalar()

        print(f'Today ({today}): {today_count} rows')
        print(f'Tomorrow ({tomorrow}): {tomorrow_count} rows')

        # Check is_confirmed counts
        confirmed_today = db.execute(
            text('SELECT COUNT(*) FROM probable_pitchers WHERE game_date = :today AND is_confirmed = true'),
            {'today': today}
        ).scalar()
        confirmed_tomorrow = db.execute(
            text('SELECT COUNT(*) FROM probable_pitchers WHERE game_date = :tomorrow AND is_confirmed = true'),
            {'tomorrow': tomorrow}
        ).scalar()

        print(f'Confirmed today: {confirmed_today}')
        print(f'Confirmed tomorrow: {confirmed_tomorrow}')

        # Show recent rows if any
        if total > 0:
            recent = db.execute(text('''
                SELECT game_date, team, pitcher_name, is_confirmed, opponent
                FROM probable_pitchers
                WHERE game_date >= :today
                ORDER BY game_date, team
                LIMIT 10
            '''), {'today': today}).fetchall()
            print('\nRecent rows:')
            for r in recent:
                print(f'  {r[0]} | {r[1]} vs {r[4]} | {r[2]} | confirmed={r[3]}')
    finally:
        db.close()

if __name__ == "__main__":
    main()
