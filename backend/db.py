"""
backend/db.py -- Shared database engine factory and declarative base.

Both models_edge.py and models_fantasy.py import Base and make_session_local
from here. Neither imports from the other. Each service passes its own
DATABASE_URL at startup via make_engine().
"""
from __future__ import annotations

import logging
import os
import time
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

logger = logging.getLogger(__name__)

# Declarative base shared by all models in both services.
Base = declarative_base()


def make_engine(database_url: str):
    """Create a SQLAlchemy engine for the given URL.

    Uses the same pool settings as the original models.py engine.
    Pass 'sqlite:///:memory:' in tests.
    """
    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        return create_engine(
            database_url,
            connect_args=connect_args,
            echo=False,
        )
    return create_engine(
        database_url,
        pool_size=20,
        max_overflow=40,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False,
    )


def make_session_local(engine) -> sessionmaker:
    """Return a session factory bound to the given engine."""
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def make_get_db(session_local: sessionmaker):
    """Return a FastAPI dependency (generator) for the given session factory.

    Retries up to 3 times on transient connection failures with exponential
    backoff -- same behaviour as the original get_db() in models.py.
    """
    def _get_db() -> Generator[Session, None, None]:
        db = None
        for attempt in range(3):
            try:
                db = session_local()
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                error_str = str(exc).lower()
                if any(k in error_str for k in ("connection", "timeout", "ssl")):
                    time.sleep(0.1 * (2 ** attempt))
                else:
                    raise
        try:
            yield db
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    return _get_db


class NamespacedKey:
    """Utility for Redis key namespacing. Used by redis_client.py.

    Ensures that edge: and fantasy: keys never collide.
    """

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    def key(self, k: str) -> str:
        return f"{self._prefix}:{k}"
