"""
Shared pytest fixtures for the test suite.
"""
import os
import pytest
import httpx
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

# httpx 0.28.1 / Starlette 0.35.1 compatibility shim
_original_httpx_client_init = httpx.Client.__init__

def _httpx_client_init_compat(self, *args, app=None, **kwargs):
    _original_httpx_client_init(self, *args, **kwargs)

httpx.Client.__init__ = _httpx_client_init_compat


@pytest.fixture(scope="function")
def db_session():
    """
    Provide a transactional SQLAlchemy session backed by the Railway PostgreSQL
    database.  Each test runs inside a SAVEPOINT that is restarted automatically
    after each commit(), so all writes are rolled back on teardown.

    Requires DATABASE_URL to be set.  If it is not set the test is skipped so
    the suite stays green in environments without a live database.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        pytest.skip("DATABASE_URL not set — skipping live-db test")

    engine = create_engine(db_url)
    connection = engine.connect()
    transaction = connection.begin()

    Session = sessionmaker(bind=connection, expire_on_commit=False)
    session = Session()

    # Open the initial savepoint.
    session.begin_nested()

    # After each SAVEPOINT ends (commit/rollback), open a fresh one so
    # subsequent queries in the same test stay within the outer rollback.
    @event.listens_for(session, "after_transaction_end")
    def _restart_savepoint(sess, trans):
        if trans.nested and not trans._parent.nested:
            sess.begin_nested()

    yield session

    session.close()
    transaction.rollback()
    connection.close()
    engine.dispose()
