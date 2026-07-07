"""
Check why a player with valid data is being treated as fallback.

For Sam Antonacci specifically - check Yahoo key matching.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sqlalchemy import text
from backend.models import SessionLocal, PlayerIDMapping, PlayerScore


def check_yahoo_key_resolution(player_name: str):
    """Check Yahoo key resolution for a specific player."""
    db = SessionLocal()

    print(f"Checking Yahoo key resolution for: {player_name}")
    print("="*60)

    # Find all mappings with similar names
    mappings = db.query(PlayerIDMapping).filter(
        PlayerIDMapping.full_name.ilike(f"%{player_name}%")
    ).all()

    print(f"\nFound {len(mappings)} mappings with similar name:")
    for m in mappings:
        print(f"\n  Mapping:")
        print(f"    full_name: {m.full_name}")
        print(f"    normalized_name: {m.normalized_name}")
        print(f"    yahoo_key: {m.yahoo_key}")
        print(f"    yahoo_id: {m.yahoo_id}")
        print(f"    bdl_id: {m.bdl_id}")
        print(f"    mlbam_id: {m.mlbam_id}")

        # Check if this player has scores
        scores = db.query(PlayerScore).filter(
            PlayerScore.bdl_player_id == m.bdl_id
        ).order_by(PlayerScore.as_of_date.desc()).limit(3).all()

        print(f"    Scores (last 3):")
        for s in scores:
            print(f"      {s.as_of_date}: score_0_100={s.score_0_100}, composite_z={s.composite_z:.2f}")

    # Check what yahoo_key variants the optimizer would look for
    # (This simulates _yahoo_key_variants function)
    print(f"\n" + "="*60)
    print("Yahoo key variants that would be searched:")
    yahoo_keys = [m.yahoo_key for m in mappings if m.yahoo_key]
    for yk in yahoo_keys:
        variants = set()
        if yk:
            # Normalize similar to _normalize_yahoo_key
            normalized = yk.lower().strip()
            variants.add(normalized)
            if normalized.startswith("469.p."):
                short_id = normalized.split(".p.", 1)[-1]
                variants.add(short_id)
            if ".p." in normalized:
                parts = normalized.split(".p.", 1)
                if len(parts) == 2:
                    variants.add(f"469.p.{parts[-1]}")
        print(f"\n  Original: {yk}")
        print(f"  Variants: {variants}")

    db.close()


if __name__ == "__main__":
    check_yahoo_key_resolution("Sam Antonacci")
