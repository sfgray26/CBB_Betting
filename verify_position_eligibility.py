#!/usr/bin/env python3
"""Quick verification script for position_eligibility table data."""
import os
import sys
from sqlalchemy import create_engine, text

def main():
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        print("ERROR: DATABASE_URL not found")
        sys.exit(1)

    engine = create_engine(db_url)

    try:
        with engine.connect() as conn:
            # Check row count
            count = conn.execute(text('SELECT COUNT(*) FROM position_eligibility')).scalar()
            print(f"position_eligibility ROWS: {count}")

            if count > 0:
                # Get sample data
                sample = conn.execute(text('''
                    SELECT player_name, bdl_player_id,
                           can_play_c, can_play_1b, can_play_2b, can_play_3b, can_play_ss,
                           can_play_lf, can_play_cf, can_play_rf, can_play_of, can_play_dh
                    FROM position_eligibility
                    LIMIT 5
                ''')).fetchall()

                print("\nSample records:")
                for row in sample:
                    positions = []
                    if row.can_play_c: positions.append('C')
                    if row.can_play_1b: positions.append('1B')
                    if row.can_play_2b: positions.append('2B')
                    if row.can_play_3b: positions.append('3B')
                    if row.can_play_ss: positions.append('SS')
                    if row.can_play_lf: positions.append('LF')
                    if row.can_play_cf: positions.append('CF')
                    if row.can_play_rf: positions.append('RF')
                    if row.can_play_of: positions.append('OF')
                    if row.can_play_dh: positions.append('DH')

                    pos_str = ', '.join(positions) if positions else 'None'
                    print(f"  {row.player_name} (BDL: {row.bdl_player_id}): {pos_str}")

                print("\n✅ SUCCESS - Data exists in position_eligibility table")
                return 0
            else:
                print("\n❌ ZERO ROWS - position_eligibility table is empty")
                return 1

    except Exception as exc:
        print(f"ERROR: {exc}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
