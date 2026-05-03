#!/usr/bin/env python
"""Production signal validation script."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['DATABASE_URL'] = "postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway"

from sqlalchemy import text
from backend.models import SessionLocal

db = SessionLocal()
try:
    # Check player_projections table for cat_scores (JSONB)
    rows = db.execute(text('''
      SELECT player_name, player_type, cat_scores
      FROM player_projections
      WHERE updated_at > NOW() - INTERVAL '48 hours'
      LIMIT 20
    ''')).fetchall()

    print(f'\n[OK] Total rows (last 48h): {len(rows)}')
    print('\nTop 20 players by recency:')
    print('-' * 110)

    cat_nonzero = 0
    hitters = 0
    pitchers = 0

    for r in rows:
        player_type = r[1] if r[1] else 'unknown'
        cat_scores = r[2] if r[2] else {}

        if player_type == 'hitter':
            hitters += 1
        elif player_type == 'pitcher':
            pitchers += 1

        # Check if cat_scores has any non-zero values
        has_nonzero = any(v != 0.0 for v in cat_scores.values()) if cat_scores else False
        if has_nonzero:
            cat_nonzero += 1

        # Format cat_scores for display (first 5 categories)
        cats_str = str(dict(list(cat_scores.items())[:5])) if cat_scores else '{}'

        print(f'{r[0]:30} | type: {player_type:8} | cats: {cats_str}')

    print('-' * 110)
    print(f'\nSignal validation:')
    print(f'  Hitters:   {hitters}/{len(rows)}')
    print(f'  Pitchers: {pitchers}/{len(rows)}')
    print(f'  cat_score non-zero: {cat_nonzero}/{len(rows)} ({cat_nonzero/len(rows)*100:.0f}%)')

    # Check statcast_performances for xwoba_diff signals
    statcast_rows = db.execute(text('''
      SELECT player_name, xwoba, woba
      FROM statcast_performances
      WHERE game_date > CURRENT_DATE - INTERVAL '7 days'
        AND xwoba IS NOT NULL
        AND woba IS NOT NULL
      ORDER BY game_date DESC
      LIMIT 10
    ''')).fetchall()

    print(f'\n[OK] Statcast xwOBA signals (last 7 days): {len(statcast_rows)} rows')
    if statcast_rows:
        for r in statcast_rows[:5]:  # Show first 5
            xwoba_diff = r[1] - r[2] if r[1] and r[2] else 0.0
            print(f'  {r[0]:30} | xwOBA: {r[1]:.3f} | wOBA: {r[2]:.3f} | diff: {xwoba_diff:+.3f}')

    if cat_nonzero >= len(rows) * 0.8:
        print('\n[PASS] cat_score distribution healthy')
    else:
        print(f'\n[FAIL] cat_score only {cat_nonzero/len(rows)*100:.0f}% non-zero')

    if statcast_rows:
        print('[PASS] Statcast xwOBA signals present')
    else:
        print('[WARN] No recent Statcast xwOBA data')

finally:
    db.close()
