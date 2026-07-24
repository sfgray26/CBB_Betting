"""Create durable Yahoo OAuth token storage.

Idempotent and safe to run while the app is live. This creates a single-row
token store used by backend/fantasy_baseball/yahoo_client_resilient.py so
Yahoo refresh-token rotation survives Railway redeploys.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine, text

YAHOO_TOKEN_PROVIDER = "yahoo_fantasy"
_ET = ZoneInfo("America/New_York")


DDL = """
CREATE TABLE IF NOT EXISTS yahoo_oauth_tokens (
    id SERIAL PRIMARY KEY,
    provider VARCHAR(50) NOT NULL DEFAULT 'yahoo_fantasy',
    access_token TEXT NOT NULL,
    refresh_token TEXT NOT NULL,
    token_type VARCHAR(32),
    expires_at TIMESTAMP WITH TIME ZONE,
    last_refresh_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_yahoo_oauth_tokens_provider UNIQUE (provider)
);

CREATE INDEX IF NOT EXISTS idx_yahoo_oauth_tokens_provider
    ON yahoo_oauth_tokens (provider);
"""

UPSERT_FROM_ENV = """
INSERT INTO yahoo_oauth_tokens (
    provider,
    access_token,
    refresh_token,
    token_type,
    expires_at,
    last_refresh_at,
    created_at,
    updated_at
)
VALUES (
    :provider,
    :access_token,
    :refresh_token,
    :token_type,
    :expires_at,
    :now,
    :now,
    :now
)
ON CONFLICT (provider) DO UPDATE SET
    access_token = EXCLUDED.access_token,
    refresh_token = EXCLUDED.refresh_token,
    token_type = EXCLUDED.token_type,
    expires_at = EXCLUDED.expires_at,
    last_refresh_at = EXCLUDED.last_refresh_at,
    updated_at = EXCLUDED.updated_at;
"""


def _seed_payload_from_env() -> dict | None:
    access_token = os.getenv("YAHOO_ACCESS_TOKEN")
    refresh_token = os.getenv("YAHOO_REFRESH_TOKEN")
    if not access_token or not refresh_token:
        return None

    now = datetime.now(_ET)
    return {
        "provider": YAHOO_TOKEN_PROVIDER,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        # Conservative: seed existing env access token as short-lived so the
        # client can start immediately but refreshes soon into durable storage.
        "expires_at": now + timedelta(minutes=30),
        "now": now,
    }


def main() -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")

    engine = create_engine(database_url)
    with engine.begin() as conn:
        conn.execute(text(DDL))
        seed_payload = _seed_payload_from_env()
        if seed_payload:
            conn.execute(text(UPSERT_FROM_ENV), seed_payload)
            print("Migration complete: yahoo_oauth_tokens table ready; env token row upserted")
        else:
            print("Migration complete: yahoo_oauth_tokens table ready; env token seed skipped")


if __name__ == "__main__":
    main()
