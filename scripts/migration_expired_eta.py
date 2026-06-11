from __future__ import annotations

import os

from sqlalchemy import create_engine, text


def main() -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")

    engine = create_engine(database_url)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                ALTER TABLE ingested_injuries
                ADD COLUMN IF NOT EXISTS expired_eta BOOLEAN NOT NULL DEFAULT FALSE;
                """
            )
        )

    print("Migration complete: expired_eta column added to ingested_injuries")


if __name__ == "__main__":
    main()
