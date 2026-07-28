"""Regression: portfolio.load_from_db must NOT count stale archived-season
unsettled bets as current open positions.

Bug (UAT 2026-07-28): load_from_db counted every BetLog with outcome IS NULL with
no date scope, so 2 never-graded archived-season CBB bets inflated
"Open Positions: 2 pending" while Bet History (date/season-scoped) showed 0.
Fix scopes the pending query to a recent lookback window — no Kelly/risk-math change.
"""
from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.models import BetLog
from backend.services.portfolio import PortfolioManager


def _db():
    engine = create_engine("sqlite:///:memory:")
    BetLog.__table__.create(engine)
    return sessionmaker(bind=engine)()


def _bet(db, bet_id, ts, outcome=None, units=1.0):
    db.add(BetLog(
        id=bet_id,
        game_id=1000 + bet_id,
        pick=f"Pick {bet_id}",
        odds_taken=-110.0,
        timestamp=ts,
        outcome=outcome,
        bet_size_units=units,
        kelly_fractional=0.02,
        conservative_edge=0.03,
        bankroll_at_bet=1000.0,
        profit_loss_dollars=0.0,
    ))


def test_archived_unsettled_bets_do_not_inflate_open_positions():
    db = _db()
    now = datetime.utcnow()
    # 1 live unsettled bet (today) — should count.
    _bet(db, 1, now)
    # 2 archived-season unsettled orphans (~120 days ago) — must NOT count.
    _bet(db, 2, now - timedelta(days=120))
    _bet(db, 3, now - timedelta(days=121))
    db.commit()

    pm = PortfolioManager()
    pm.load_from_db(db)

    positions = pm.get_state().positions
    assert len(positions) == 1, [p.game_id for p in positions]
    assert positions[0].game_id == 1001  # only the recent bet


def test_recent_unsettled_bets_all_count():
    db = _db()
    now = datetime.utcnow()
    _bet(db, 1, now)
    _bet(db, 2, now - timedelta(days=2))
    _bet(db, 3, now - timedelta(days=5))
    db.commit()

    pm = PortfolioManager()
    pm.load_from_db(db)

    assert len(pm.get_state().positions) == 3


def test_settled_bets_are_never_pending():
    db = _db()
    now = datetime.utcnow()
    _bet(db, 1, now, outcome=None)          # pending, recent
    _bet(db, 2, now, outcome=1)             # settled win, recent
    db.commit()

    pm = PortfolioManager()
    pm.load_from_db(db)

    positions = pm.get_state().positions
    assert len(positions) == 1
    assert positions[0].game_id == 1001
