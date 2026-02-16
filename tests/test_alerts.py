"""Tests for alerts.py alert-trigger logic."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from backend.services.alerts import check_performance_alerts, Alert


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_bet(outcome, pl_dollars=0.0, clv=None, bet_dollars=10.0):
    b = MagicMock()
    b.outcome = outcome
    b.profit_loss_dollars = pl_dollars
    b.clv_prob = clv
    b.bet_size_dollars = bet_dollars
    b.timestamp = datetime.utcnow()
    return b


def _run_alerts(bets):
    db = MagicMock()
    q = MagicMock()
    q.filter.return_value = q
    q.order_by.return_value = q
    q.all.return_value = bets
    db.query.return_value = q
    return check_performance_alerts(db)


# ---------------------------------------------------------------------------
# No bets → no alerts
# ---------------------------------------------------------------------------

def test_no_bets():
    alerts = _run_alerts([])
    assert alerts == []


# ---------------------------------------------------------------------------
# CLV alerts
# ---------------------------------------------------------------------------

def test_clv_critical_alert():
    # 15 bets, all with CLV = -1% → mean = -1% → CRITICAL
    bets = [_fake_bet(0, -10.0, clv=-0.01) for _ in range(15)]
    alerts = _run_alerts(bets)
    types = [a.alert_type for a in alerts]
    assert "CLV_NEGATIVE" in types
    critical = [a for a in alerts if a.alert_type == "CLV_NEGATIVE"]
    assert critical[0].severity == "CRITICAL"


def test_clv_warning_alert():
    # CLV between -0.5% and 0% → WARNING
    bets = [_fake_bet(0, -10.0, clv=-0.002) for _ in range(15)]
    alerts = _run_alerts(bets)
    types = [a.alert_type for a in alerts]
    assert "CLV_DECLINING" in types
    warning = [a for a in alerts if a.alert_type == "CLV_DECLINING"]
    assert warning[0].severity == "WARNING"


def test_healthy_clv_no_alert():
    bets = [_fake_bet(1, 9.09, clv=0.01) for _ in range(15)]
    alerts = _run_alerts(bets)
    clv_alerts = [a for a in alerts if "CLV" in a.alert_type]
    assert len(clv_alerts) == 0


# ---------------------------------------------------------------------------
# Drawdown alerts
# ---------------------------------------------------------------------------

def test_drawdown_warning():
    # Build up profit, then lose 20%
    up_bets = [_fake_bet(1, 10.0) for _ in range(10)]   # +100
    down_bets = [_fake_bet(0, -10.0) for _ in range(20)] # -200 → drawdown > 15%
    bets = up_bets + down_bets
    alerts = _run_alerts(bets)
    types = [a.alert_type for a in alerts]
    assert "DRAWDOWN_HIGH" in types
    dd = [a for a in alerts if a.alert_type == "DRAWDOWN_HIGH"]
    assert dd[0].severity == "WARNING"


def test_no_drawdown_when_profitable():
    bets = [_fake_bet(1, 10.0) for _ in range(10)]
    alerts = _run_alerts(bets)
    dd_alerts = [a for a in alerts if "DRAWDOWN" in a.alert_type]
    assert len(dd_alerts) == 0


# ---------------------------------------------------------------------------
# Consecutive losing streak
# ---------------------------------------------------------------------------

def test_losing_streak_warning():
    # 5 wins then 10 losses
    bets = [_fake_bet(1, 9.09)] * 5 + [_fake_bet(0, -10.0)] * 10
    alerts = _run_alerts(bets)
    streak_alerts = [a for a in alerts if a.alert_type == "LOSING_STREAK"]
    assert len(streak_alerts) == 1
    assert streak_alerts[0].severity == "WARNING"
    assert streak_alerts[0].current_value == 10.0


def test_losing_streak_info():
    bets = [_fake_bet(1, 9.09)] * 5 + [_fake_bet(0, -10.0)] * 8
    alerts = _run_alerts(bets)
    streak_alerts = [a for a in alerts if a.alert_type == "LOSING_STREAK"]
    assert len(streak_alerts) == 1
    assert streak_alerts[0].severity == "INFO"


def test_no_streak_alert_after_win():
    # Losses broken by a win at the end
    bets = [_fake_bet(0, -10.0)] * 12 + [_fake_bet(1, 9.09)]
    alerts = _run_alerts(bets)
    streak_alerts = [a for a in alerts if a.alert_type == "LOSING_STREAK"]
    assert len(streak_alerts) == 0


# ---------------------------------------------------------------------------
# Win rate alert
# ---------------------------------------------------------------------------

def test_win_rate_low_alert():
    # 40% win rate over 30 bets → INFO alert
    bets = (
        [_fake_bet(1, 9.09)] * 12 +
        [_fake_bet(0, -10.0)] * 18
    )
    alerts = _run_alerts(bets)
    wr_alerts = [a for a in alerts if a.alert_type == "WIN_RATE_LOW"]
    assert len(wr_alerts) == 1
    assert wr_alerts[0].severity == "INFO"


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------

def test_alert_to_dict():
    a = Alert(
        alert_type="CLV_NEGATIVE",
        severity="CRITICAL",
        message="Test",
        threshold=-0.005,
        current_value=-0.01,
        recommendation="Stop betting",
    )
    d = a.to_dict()
    assert d["alert_type"] == "CLV_NEGATIVE"
    assert d["severity"] == "CRITICAL"
    assert d["recommendation"] == "Stop betting"
