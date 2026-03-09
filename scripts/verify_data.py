#!/usr/bin/env python3
"""Verify the generated CSV files."""

import csv

def verify():
    print("=" * 60)
    print("VERIFYING FANTASY BASEBALL 2026 PROJECTION FILES")
    print("=" * 60)
    
    # Verify batting
    with open('data/projections/steamer_batting_2026.csv', 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f"\nBatting Projections:")
        print(f"  Count: {len(rows)} players (target: 300+)")
        if rows:
            print(f"  Columns: {list(rows[0].keys())}")
            print(f"  First player: {rows[0]['Name']}")
            print(f"  Last player: {rows[-1]['Name']}")
        print(f"  Status: {'PASS' if len(rows) >= 300 else 'FAIL'}")

    # Verify pitching
    with open('data/projections/steamer_pitching_2026.csv', 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f"\nPitching Projections:")
        print(f"  Count: {len(rows)} players (target: 200+)")
        if rows:
            print(f"  Columns: {list(rows[0].keys())}")
            print(f"  First player: {rows[0]['Name']}")
            print(f"  Last player: {rows[-1]['Name']}")
        print(f"  Status: {'PASS' if len(rows) >= 200 else 'FAIL'}")

    # Verify ADP
    with open('data/projections/adp_yahoo_2026.csv', 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f"\nYahoo ADP:")
        print(f"  Count: {len(rows)} players (target: 300+)")
        if rows:
            print(f"  Columns: {list(rows[0].keys())}")
            print(f"  First player: {rows[0]['PLAYER NAME']}")
            print(f"  Last player: {rows[-1]['PLAYER NAME']}")
        print(f"  Status: {'PASS' if len(rows) >= 300 else 'FAIL'}")

    # Verify closers
    with open('data/projections/closer_situations_2026.csv', 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f"\nCloser Situations:")
        print(f"  Count: {len(rows)} teams (target: 30)")
        if rows:
            print(f"  Columns: {list(rows[0].keys())}")
        print(f"  Status: {'PASS' if len(rows) >= 30 else 'FAIL'}")

    # Verify injuries
    with open('data/projections/injury_flags_2026.csv', 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f"\nInjury Flags:")
        print(f"  Count: {len(rows)} players")
        if rows:
            print(f"  Columns: {list(rows[0].keys())}")

    # Verify positions
    with open('data/projections/position_eligibility_2026.csv', 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f"\nPosition Eligibility:")
        print(f"  Count: {len(rows)} players")
        if rows:
            print(f"  Columns: {list(rows[0].keys())}")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    verify()
