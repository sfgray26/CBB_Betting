"""Durable storage for Yahoo Fantasy OAuth token rotation."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

YAHOO_TOKEN_PROVIDER = "yahoo_fantasy"
_ET = ZoneInfo("America/New_York")


def load_yahoo_tokens() -> Optional[dict]:
    """Load the latest persisted Yahoo token pair, if the table is available."""
    try:
        from backend.models import SessionLocal, YahooOAuthToken

        db: Session = SessionLocal()
        try:
            row = (
                db.query(YahooOAuthToken)
                .filter(YahooOAuthToken.provider == YAHOO_TOKEN_PROVIDER)
                .first()
            )
            if not row or not row.refresh_token:
                return None
            return {
                "access_token": row.access_token,
                "refresh_token": row.refresh_token,
                "expires_at": row.expires_at,
                "token_type": row.token_type,
            }
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001 - persistence must not block client init
        logger.warning("Yahoo token store unavailable; falling back to environment tokens: %s", exc)
        return None


def persist_yahoo_tokens(
    *,
    access_token: str,
    refresh_token: str,
    expires_at: Optional[datetime],
    token_type: Optional[str],
) -> bool:
    """Upsert the rotated Yahoo token pair into the database."""
    try:
        from backend.models import SessionLocal, YahooOAuthToken

        db: Session = SessionLocal()
        try:
            row = (
                db.query(YahooOAuthToken)
                .filter(YahooOAuthToken.provider == YAHOO_TOKEN_PROVIDER)
                .first()
            )
            now = datetime.now(_ET)
            if row is None:
                row = YahooOAuthToken(
                    provider=YAHOO_TOKEN_PROVIDER,
                    access_token=access_token,
                    refresh_token=refresh_token,
                    token_type=token_type,
                    expires_at=expires_at,
                    last_refresh_at=now,
                )
                db.add(row)
            else:
                row.access_token = access_token
                row.refresh_token = refresh_token
                row.token_type = token_type
                row.expires_at = expires_at
                row.last_refresh_at = now
                row.updated_at = now
            db.commit()
            logger.info("Yahoo OAuth tokens persisted to database")
            return True
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001
        logger.error("Yahoo OAuth token DB persistence failed: %s", exc)
        return False
