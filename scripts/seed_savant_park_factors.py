"""
Seed park_factors from the versioned Baseball Savant Statcast snapshot.

Run after scripts/migration_savant_park_factors.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.fantasy_baseball.savant_park_factors import load_savant_park_factor_snapshot
from backend.models import ParkFactor, SessionLocal


UPSERT_FIELDS = [
    "venue_name",
    "team",
    "venue_id",
    "rolling_years",
    "bat_side",
    "condition",
    "year_range",
    "source_url",
    "hr_factor",
    "run_factor",
    "hits_factor",
    "era_factor",
    "whip_factor",
    "woba_factor",
    "wobacon_factor",
    "xwobacon_factor",
    "obp_factor",
    "bb_factor",
    "so_factor",
    "bacon_factor",
    "singles_factor",
    "doubles_factor",
    "triples_factor",
    "hardhit_factor",
    "n_pa",
    "data_source",
    "season",
]


def main() -> None:
    rows = load_savant_park_factor_snapshot()
    db = SessionLocal()
    inserted = 0
    updated = 0

    try:
        for row in rows:
            team = row["team"]
            park_factor = db.query(ParkFactor).filter_by(park_name=team).first()

            if park_factor is None:
                park_factor = ParkFactor(park_name=team)
                db.add(park_factor)
                inserted += 1
            else:
                updated += 1

            # Preserve the legacy lookup key in park_name and store venue separately.
            park_factor.venue_name = row["park_name"]
            for field in UPSERT_FIELDS:
                setattr(park_factor, field, row[field])

        # The app historically accepted CWS as a White Sox alias. Keep it synced.
        white_sox = next((row for row in rows if row["team"] == "CHW"), None)
        if white_sox:
            alias = db.query(ParkFactor).filter_by(park_name="CWS").first()
            if alias is None:
                alias = ParkFactor(park_name="CWS")
                db.add(alias)
                inserted += 1
            else:
                updated += 1

            alias.venue_name = white_sox["park_name"]
            alias.team = "CWS"
            for field in UPSERT_FIELDS:
                if field == "team":
                    continue
                setattr(alias, field, white_sox[field])

        db.commit()
        print(f"Savant park factors seeded: inserted={inserted} updated={updated}")
        print(f"Snapshot rows loaded: {len(rows)}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
