#!/usr/bin/env python3
"""Add missing _pim_yahoo_key_uc constraint to player_id_mapping table."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models import engine
from sqlalchemy import text

def main():
    with engine.connect() as conn:
        # Check if constraint exists
        result = conn.execute(text(
            "SELECT COUNT(*) FROM pg_constraint WHERE conname = '_pim_yahoo_key_uc'"
        ))
        exists = result.fetchone()[0]
        print(f"Constraint exists: {exists}")

        if not exists:
            print("Adding constraint...")
            conn.execute(text(
                "ALTER TABLE player_id_mapping ADD CONSTRAINT _pim_yahoo_key_uc UNIQUE (yahoo_key)"
            ))
            conn.commit()
            print("Done: Added _pim_yahoo_key_uc constraint")
        else:
            print("Constraint already exists, skipping")

if __name__ == "__main__":
    main()
