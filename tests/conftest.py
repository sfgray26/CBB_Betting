"""
Shared pytest fixtures for the test suite.
"""
import os
import pytest
import httpx
from unittest.mock import Mock
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

# httpx 0.28.1 / Starlette 0.35.1 compatibility shim
_original_httpx_client_init = httpx.Client.__init__

def _httpx_client_init_compat(self, *args, app=None, **kwargs):
    _original_httpx_client_init(self, *args, **kwargs)

httpx.Client.__init__ = _httpx_client_init_compat


@pytest.fixture
def mock_db_session():
    """Return a mock SQLAlchemy session for unit tests that don't need a live DB."""
    mock_session = Mock()
    mock_session.execute = Mock()
    mock_session.query = Mock()
    mock_session.add = Mock()
    mock_session.commit = Mock()
    mock_session.rollback = Mock()
    mock_session.close = Mock()
    return mock_session


@pytest.fixture
def mock_db_row():
    """Factory for creating mock database row results."""
    def _create_row(**kwargs):
        row = Mock()
        row.__iter__ = Mock(return_value=iter(kwargs.values()))
        row._asdict = Mock(return_value=kwargs)
        row._mapping = kwargs
        for key, value in kwargs.items():
            setattr(row, key, value)
        return row
    return _create_row


@pytest.fixture
def sample_yahoo_player():
    """Return a sample Yahoo player dict for tests that need one."""
    return {
        "player_key": "mlb.p.12345",
        "name": "Test Player",
        "full_name": "Test Player",
        "team": "LAD",
        "positions": ["1B", "OF"],
        "status": "Active",
        "injury_status": None,
        "percent_owned": 85.5,
        "ownership_pct": 85.5,
        "stats": {
            "7": 25.0,
            "8": 45.0,
            "12": 8.0,
            "13": 32.0,
        },
        "selected_position": "1B",
    }


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
