"""
Re-normalize player_identities.normalized_name to use accent-stripped NFKD form.

Background:
  The original backfill used SQL lower(trim(full_name)) which preserves accents.
  e.g. 'José Ramírez' → 'josé ramírez' (stored)
  The board (FanGraphs/Steamer) sends ASCII names like 'Jose Ramirez'.
  _normalize_name('Jose Ramirez') was returning 'jose ramirez' (no accent) → no match.

  This script updates all normalized_name values to use NFKD accent-stripping so both
  sides produce the same key: 'jose ramirez'.

Usage:
    $env:DATABASE_URL = "postgresql://..."
    .\\venv\\Scripts\\python scripts/renormalize_player_identities.py
    .\\venv\\Scripts\\python scripts/renormalize_player_identities.py --dry-run
"""
import argparse
import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text


def _normalize_name(name: str) -> str:
    """NFKD accent-stripped + lower + strip. Matches updated id_resolution_service."""
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def main():
    parser = argparse.ArgumentParser(description="Re-normalize player_identities.normalized_name")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)

    engine = create_engine(db_url, pool_pre_ping=True)

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, full_name, normalized_name FROM player_identities ORDER BY id"
        )).fetchall()

    print(f"Loaded {len(rows)} player_identities rows")

    updated = []
    skipped_conflict = []

    # Build set of new normalized names to detect conflicts before writing
    new_norm_map: dict[int, str] = {}
    new_norm_values: dict[str, int] = {}  # new_norm → first row_id that claims it

    for row_id, full_name, current_norm in rows:
        new_norm = _normalize_name(full_name)
        if new_norm != current_norm:
            if new_norm in new_norm_values:
                # Conflict: another row already claimed this normalized name
                skipped_conflict.append((row_id, full_name, new_norm, new_norm_values[new_norm]))
            else:
                new_norm_map[row_id] = new_norm
                new_norm_values[new_norm] = row_id
        else:
            # Already correct — register it so conflicts are detected
            new_norm_values[new_norm] = row_id

    print(f"Rows needing update:  {len(new_norm_map)}")
    print(f"Conflicts (skipped):  {len(skipped_conflict)}")

    if skipped_conflict:
        print("\nConflicts (two identities would get same normalized_name):")
        for row_id, full_name, new_norm, winner_id in skipped_conflict[:10]:
            print(f"  id={row_id} {full_name!r} → {new_norm!r} (conflicts with id={winner_id})")

    if args.dry_run:
        print("\n[DRY RUN] Sample of changes:")
        sample = [(rid, fn, nn) for rid, fn, nn in
                  ((r[0], r[1], new_norm_map[r[0]]) for r in rows if r[0] in new_norm_map)][:20]
        for row_id, full_name, new_norm in sample:
            # find current norm
            current = next(r[2] for r in rows if r[0] == row_id)
            print(f"  id={row_id}: {current!r} → {new_norm!r}  (full_name={full_name!r})")
        print("\n[DRY RUN] No changes written.")
        return

    if not new_norm_map:
        print("Nothing to update — all normalized_name values already correct.")
        return

    # Apply updates in batches
    BATCH = 200
    ids = list(new_norm_map.keys())
    written = 0

    with engine.begin() as conn:
        for i in range(0, len(ids), BATCH):
            batch_ids = ids[i:i + BATCH]
            for row_id in batch_ids:
                conn.execute(text(
                    "UPDATE player_identities SET normalized_name = :nn WHERE id = :id"
                ), {"nn": new_norm_map[row_id], "id": row_id})
                written += 1
            print(f"  Updated {min(i + BATCH, len(ids))}/{len(ids)}...")

    print(f"\nDone. {written} rows updated, {len(skipped_conflict)} skipped (conflict).")
    print("Next: re-run shadow validation to verify coverage improvement.")
    print("  .\\venv\\Scripts\\python scripts/validate_canonical_projections.py --top-misses 20")


if __name__ == "__main__":
    main()
