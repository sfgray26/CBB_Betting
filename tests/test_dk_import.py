"""Tests for backend/services/dk_import.py

Covers: parse_dk_csv, amount tolerance, payout detection, and apply_import
without requiring a live database.

Run: pytest tests/test_dk_import.py -v
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from backend.services.dk_import import (
    DKTransaction,
    _amount_close,
    _find_payout_for_wager,
    _parse_amount,
    apply_import,
    parse_dk_csv,
)

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

_PAST = datetime(2026, 1, 15, 10, 0, 0)           # Far past → LOSS if no payout
_PAST_PAYOUT = datetime(2026, 1, 15, 18, 0, 0)    # 8 h after wager → within 48 h window


def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.0000000+00:00")


def _make_csv(*rows: str) -> str:
    header = "TransactionID,Date,Amount,Balance,Description"
    return "\n".join([header] + list(rows))


def _wager_row(tx_id="TX001", amount="25.00", dt=None, desc="Sportsbook Wager"):
    return f"{tx_id},{_fmt_dt(dt or _PAST)},{amount},975.00,{desc}"


def _payout_row(tx_id="TX002", amount="60.00", dt=None, desc="Win Payout"):
    return f"{tx_id},{_fmt_dt(dt or _PAST_PAYOUT)},{amount},1035.00,{desc}"


# ---------------------------------------------------------------------------
# Helpers for DKTransaction objects
# ---------------------------------------------------------------------------

def _make_wager(dk_id="W1", amount=25.0, dt=None) -> DKTransaction:
    return DKTransaction(
        dk_id=dk_id,
        amount=amount,
        timestamp_utc=dt or _PAST,
        description="Sportsbook Wager",
        tx_type="wager",
    )


def _make_payout(dk_id="P1", amount=60.0, dt=None) -> DKTransaction:
    return DKTransaction(
        dk_id=dk_id,
        amount=amount,
        timestamp_utc=dt or _PAST_PAYOUT,
        description="Win Payout",
        tx_type="payout",
    )


# ===========================================================================
# TestParseDkCsv
# ===========================================================================

class TestParseDkCsv:
    def test_single_wager_parsed(self):
        csv = _make_csv(_wager_row())
        data = parse_dk_csv(csv)
        assert len(data.wagers) == 1
        assert len(data.payouts) == 0

    def test_single_payout_parsed(self):
        csv = _make_csv(_payout_row())
        data = parse_dk_csv(csv)
        assert len(data.payouts) == 1
        assert len(data.wagers) == 0

    def test_wager_fields_correct(self):
        csv = _make_csv(_wager_row(tx_id="TXW1", amount="50.00"))
        data = parse_dk_csv(csv)
        w = data.wagers[0]
        assert w.dk_id == "TXW1"
        assert w.amount == pytest.approx(50.0)
        assert w.tx_type == "wager"
        assert isinstance(w.timestamp_utc, datetime)

    def test_non_transaction_rows_skipped(self):
        deposit_row = f"TX999,{_fmt_dt(_PAST)},100.00,1100.00,Deposit"
        csv = _make_csv(deposit_row)
        data = parse_dk_csv(csv)
        assert len(data.wagers) == 0
        assert len(data.payouts) == 0
        assert data.skipped_rows == 1

    def test_empty_csv_returns_empty(self):
        csv = "TransactionID,Date,Amount,Balance,Description\n"
        data = parse_dk_csv(csv)
        assert len(data.wagers) == 0
        assert len(data.payouts) == 0
        assert data.skipped_rows == 0

    def test_bom_stripped(self):
        csv = "\ufeff" + _make_csv(_wager_row())
        data = parse_dk_csv(csv)
        assert len(data.wagers) == 1

    def test_multiple_wagers_parsed(self):
        csv = _make_csv(
            _wager_row("TX1", "10.00", dt=datetime(2026, 1, 15, 10, 0)),
            _wager_row("TX2", "20.00", dt=datetime(2026, 1, 15, 11, 0)),
        )
        data = parse_dk_csv(csv)
        assert len(data.wagers) == 2


# ===========================================================================
# TestParseAmount
# ===========================================================================

class TestParseAmount:
    def test_dollar_sign_stripped(self):
        assert _parse_amount("$25.00") == pytest.approx(25.0)

    def test_negative_absolute_value(self):
        assert _parse_amount("-$10.00") == pytest.approx(10.0)

    def test_comma_stripped(self):
        assert _parse_amount("$1,000.00") == pytest.approx(1000.0)


# ===========================================================================
# TestAmountTolerance
# ===========================================================================

class TestAmountTolerance:
    def test_exact_match_accepted(self):
        assert _amount_close(25.0, 25.0) is True

    def test_within_ten_percent_accepted(self):
        assert _amount_close(24.0, 25.0) is True    # 4% off
        assert _amount_close(26.0, 25.0) is True    # 4% off

    def test_outside_ten_percent_rejected(self):
        assert _amount_close(22.0, 25.0) is False   # 12% off

    def test_zero_reference_rejected(self):
        assert _amount_close(5.0, 0.0) is False


# ===========================================================================
# TestPayoutDetection  (tests _find_payout_for_wager directly)
# ===========================================================================

class TestPayoutDetection:
    def test_eligible_payout_matched(self):
        wager = _make_wager()
        payout = _make_payout()
        result = _find_payout_for_wager(wager, [payout], set())
        assert result is not None
        assert result.dk_id == "P1"

    def test_payout_before_wager_not_matched(self):
        wager = _make_wager()
        early_payout = _make_payout(dt=_PAST - timedelta(hours=1))
        result = _find_payout_for_wager(wager, [early_payout], set())
        assert result is None

    def test_payout_outside_48h_window_rejected(self):
        wager = _make_wager()
        late_payout = _make_payout(dt=_PAST + timedelta(hours=49))
        result = _find_payout_for_wager(wager, [late_payout], set())
        assert result is None

    def test_payout_below_98pct_of_wager_rejected(self):
        # Payout is only 50% of wager — not a real win payout
        wager = _make_wager(amount=100.0)
        small_payout = _make_payout(amount=49.0)
        result = _find_payout_for_wager(wager, [small_payout], set())
        assert result is None

    def test_already_used_payout_skipped(self):
        wager = _make_wager()
        payout = _make_payout()
        used = {"P1"}
        result = _find_payout_for_wager(wager, [payout], used)
        assert result is None


# ===========================================================================
# TestApplyImport  (mock DB — no live database required)
# ===========================================================================

class TestApplyImport:
    def _mock_db(self, bet_log=None):
        db = MagicMock()
        bl = bet_log or MagicMock(notes="")
        db.query.return_value.filter.return_value.first.return_value = bl
        return db, bl

    def test_win_increments_wins(self):
        db, _ = self._mock_db()
        summary = apply_import(db, [{
            "bet_log_id": 1, "dk_wager_id": "TX1",
            "dk_wager_amount": 25.0, "outcome": 1,
            "profit_dollars": 35.0, "payout_amount": 60.0,
        }])
        assert summary.wins == 1
        assert summary.losses == 0
        assert summary.applied == 1
        assert summary.total_profit == pytest.approx(35.0)

    def test_loss_increments_losses(self):
        db, _ = self._mock_db()
        summary = apply_import(db, [{
            "bet_log_id": 1, "dk_wager_id": "TX1",
            "dk_wager_amount": 25.0, "outcome": 0,
            "profit_dollars": -25.0, "payout_amount": 0.0,
        }])
        assert summary.losses == 1
        assert summary.wins == 0
        assert summary.total_profit == pytest.approx(-25.0)

    def test_pending_outcome_skipped(self):
        db, _ = self._mock_db()
        summary = apply_import(db, [{
            "bet_log_id": 1, "dk_wager_id": "TX1",
            "dk_wager_amount": 25.0, "outcome": None,
            "profit_dollars": 0.0, "payout_amount": 0.0,
        }])
        assert summary.applied == 0
        assert summary.pending == 1

    def test_missing_bet_log_adds_error(self):
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        summary = apply_import(db, [{
            "bet_log_id": 999, "dk_wager_id": "TX1",
            "dk_wager_amount": 25.0, "outcome": 1,
            "profit_dollars": 35.0, "payout_amount": 60.0,
        }])
        assert summary.applied == 0
        assert len(summary.errors) == 1

    def test_total_profit_sums_across_multiple_bets(self):
        db = MagicMock()
        bl1 = MagicMock(notes="")
        bl2 = MagicMock(notes="")
        db.query.return_value.filter.return_value.first.side_effect = [bl1, bl2]
        summary = apply_import(db, [
            {"bet_log_id": 1, "dk_wager_id": "W1", "dk_wager_amount": 25.0,
             "outcome": 1, "profit_dollars": 35.0, "payout_amount": 60.0},
            {"bet_log_id": 2, "dk_wager_id": "W2", "dk_wager_amount": 10.0,
             "outcome": 0, "profit_dollars": -10.0, "payout_amount": 0.0},
        ])
        assert summary.total_profit == pytest.approx(25.0)  # 35 - 10
        assert summary.wins == 1
        assert summary.losses == 1
        assert summary.applied == 2
