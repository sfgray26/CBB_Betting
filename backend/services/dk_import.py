"""
DraftKings CSV import service.

Parses DraftKings transaction history CSV and matches wagers / win-payouts
to BetLog paper-trade rows so the dashboard reflects real-money results.

Two import modes
----------------
1. Paper-trade matching (preview_import / apply_import):
     Matches DK wagers to existing model paper-trade BetLogs by proximity:
       - bet_size_dollars within 10% of wager amount
       - timestamp within ±24 hours of the wager

2. Direct import (preview_direct_import / apply_direct_import):
     Creates brand-new real-money BetLog entries for each wager without
     requiring existing paper trades.  Useful when the model's nightly job
     didn't fire, or when the user bet on games the model passed on.

For both modes, win-payouts are detected by:
  - Payout happens AFTER the wager timestamp
  - amount >= wager_amount * 0.98  (allow tiny rounding)
  - Happened within 48 h of the wager

  If a payout is found  → outcome=WIN, profit = payout − stake
  If no payout and >12 h since wager  → outcome=LOSS, profit = −stake
  If no payout and <12 h since wager  → outcome=PENDING (skip for now)

Public API
----------
  parse_dk_csv(content)                    → DKImportData
  preview_import(db, data)                 → List[ImportMatch]
  apply_import(db, confirmed)              → ImportSummary
  preview_direct_import(db, data)          → List[DirectImportPreview]
  apply_direct_import(db, items)           → ImportSummary
"""

import csv
import io
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models import BetLog, Game

logger = logging.getLogger(__name__)

_MATCH_WINDOW_HOURS = 24      # timestamp window for wager → BetLog paper trade (widened to 24h)
_PAYOUT_WINDOW_HOURS = 48     # how long after a wager to look for payout
_AMOUNT_TOLERANCE = 0.10      # 10 % tolerance on dollar amount
_SETTLED_THRESHOLD_HOURS = 12 # wager older than this with no payout → LOSS


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DKTransaction:
    dk_id: str
    amount: float           # absolute value
    timestamp_utc: datetime # naive UTC
    description: str
    tx_type: str            # "wager" | "payout"
    balance_after: float = 0.0


@dataclass
class DKImportData:
    wagers: List[DKTransaction] = field(default_factory=list)
    payouts: List[DKTransaction] = field(default_factory=list)
    skipped_rows: int = 0


@dataclass
class ImportMatch:
    bet_log_id: int
    pick: str
    bet_log_timestamp: datetime
    bet_log_dollars: float
    dk_wager_id: str
    dk_wager_amount: float
    dk_wager_timestamp: datetime
    outcome: Optional[int]   # 1=win, 0=loss, None=pending
    profit_dollars: float
    payout_amount: float     # 0 if loss/pending
    confidence: str          # "HIGH" | "MEDIUM"


@dataclass
class ImportSummary:
    applied: int
    wins: int
    losses: int
    pending: int
    total_profit: float
    errors: List[str] = field(default_factory=list)


@dataclass
class DirectImportPreview:
    """One DK wager candidate for direct BetLog creation (no paper trade required)."""
    dk_wager_id: str
    dk_amount: float
    dk_timestamp: datetime      # naive UTC
    outcome: Optional[int]      # 1=win, 0=loss, None=pending
    profit_dollars: float
    payout_amount: float
    candidate_games: List[Dict] # [{id, matchup}, ...]
    suggested_game_id: Optional[int]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_amount(raw: str) -> float:
    """'$25.00' / '-$10.00' / '25' / '-25' → float (absolute value)."""
    clean = re.sub(r"[,$\s]", "", raw or "0").replace("(", "").replace(")", "")
    try:
        return abs(float(clean))
    except ValueError:
        return 0.0


def _parse_dt(raw: str) -> Optional[datetime]:
    """Parse ISO 8601 DateRaw → naive UTC datetime."""
    if not raw:
        return None
    try:
        # "2026-02-04T02:26:09.0000000+00:00"
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except (ValueError, TypeError):
        return None


def parse_dk_csv(content: str) -> DKImportData:
    """
    Parse raw CSV text from DraftKings transaction history.

    Returns a DKImportData with separate wager and payout lists.
    """
    data = DKImportData()

    # Handle Windows BOM
    if content.startswith("\ufeff"):
        content = content[1:]

    try:
        reader = csv.DictReader(io.StringIO(content))
    except Exception as exc:
        logger.error("CSV parse error: %s", exc)
        return data

    for row in reader:
        # DK occasionally prepends BOM to the first header key
        desc = row.get("Description", "")
        tx_id = row.get("TransactionID", "")
        amount_raw = row.get("AmountRaw", row.get("Amount", "0"))
        balance_raw = row.get("BalanceRaw", row.get("Balance", "0"))
        date_raw = row.get("DateRaw", row.get("Date", ""))

        amount = _parse_amount(amount_raw)
        dt = _parse_dt(date_raw)

        if dt is None or amount == 0 or not tx_id:
            data.skipped_rows += 1
            continue

        desc_lower = desc.lower()
        if "sportsbook wager" in desc_lower:
            data.wagers.append(DKTransaction(
                dk_id=tx_id,
                amount=amount,
                timestamp_utc=dt,
                description=desc,
                tx_type="wager",
                balance_after=_parse_amount(balance_raw),
            ))
        elif "win payout" in desc_lower or "winning" in desc_lower:
            data.payouts.append(DKTransaction(
                dk_id=tx_id,
                amount=amount,
                timestamp_utc=dt,
                description=desc,
                tx_type="payout",
                balance_after=_parse_amount(balance_raw),
            ))
        else:
            data.skipped_rows += 1

    logger.info(
        "DK CSV parsed: %d wagers, %d payouts, %d skipped",
        len(data.wagers), len(data.payouts), data.skipped_rows,
    )
    return data


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _amount_close(a: float, b: float) -> bool:
    if b == 0:
        return False
    return abs(a - b) / b <= _AMOUNT_TOLERANCE


def preview_import(db: Session, data: DKImportData) -> List[ImportMatch]:
    """
    Compute proposed BetLog ↔ DK wager matches without writing anything.

    Returns a list of ImportMatch objects that the caller can display and
    confirm before calling ``apply_import``.
    """
    # Fetch all unlinked paper-trade BetLogs
    bet_logs = (
        db.query(BetLog)
        .filter(
            BetLog.is_paper_trade == True,
            BetLog.executed == False,
        )
        .order_by(BetLog.timestamp.asc())
        .all()
    )

    matches: List[ImportMatch] = []
    used_bet_log_ids: set = set()
    used_wager_ids: set = set()
    now_utc = datetime.utcnow()

    # Sort payouts once for fast lookup
    payouts_sorted = sorted(data.payouts, key=lambda p: p.timestamp_utc)

    for wager in sorted(data.wagers, key=lambda w: w.timestamp_utc):
        if wager.dk_id in used_wager_ids:
            continue

        best_log: Optional[BetLog] = None
        best_confidence = ""
        best_delta = timedelta(days=999)

        for bl in bet_logs:
            if bl.id in used_bet_log_ids:
                continue

            delta = abs(bl.timestamp - wager.timestamp_utc)

            # Window + amount checks
            within_window = delta <= timedelta(hours=_MATCH_WINDOW_HOURS)
            amount_ok = _amount_close(wager.amount, bl.bet_size_dollars or 0)

            if not within_window:
                continue

            confidence = "HIGH" if (amount_ok and delta <= timedelta(minutes=30)) else "MEDIUM"

            if confidence == "HIGH" and best_confidence != "HIGH":
                best_log = bl
                best_confidence = confidence
                best_delta = delta
            elif confidence == best_confidence and delta < best_delta:
                best_log = bl
                best_confidence = confidence
                best_delta = delta

        if best_log is None:
            continue

        # Find a matching win payout for this wager
        matched_payout: Optional[DKTransaction] = None
        for payout in payouts_sorted:
            if payout.dk_id in {m.dk_wager_id for m in matches}:
                continue
            if payout.timestamp_utc <= wager.timestamp_utc:
                continue
            if payout.timestamp_utc > wager.timestamp_utc + timedelta(hours=_PAYOUT_WINDOW_HOURS):
                continue
            # Payout must return at least the stake (minus small rounding)
            if payout.amount >= wager.amount * 0.98:
                matched_payout = payout
                break

        # Determine outcome
        hours_since_wager = (now_utc - wager.timestamp_utc).total_seconds() / 3600
        if matched_payout is not None:
            outcome = 1
            profit = matched_payout.amount - wager.amount
            payout_amt = matched_payout.amount
        elif hours_since_wager > _SETTLED_THRESHOLD_HOURS:
            outcome = 0
            profit = -wager.amount
            payout_amt = 0.0
        else:
            outcome = None   # still pending
            profit = 0.0
            payout_amt = 0.0

        matches.append(ImportMatch(
            bet_log_id=best_log.id,
            pick=best_log.pick,
            bet_log_timestamp=best_log.timestamp,
            bet_log_dollars=best_log.bet_size_dollars or 0.0,
            dk_wager_id=wager.dk_id,
            dk_wager_amount=wager.amount,
            dk_wager_timestamp=wager.timestamp_utc,
            outcome=outcome,
            profit_dollars=round(profit, 2),
            payout_amount=payout_amt,
            confidence=best_confidence,
        ))

        used_bet_log_ids.add(best_log.id)
        used_wager_ids.add(wager.dk_id)

    logger.info(
        "DK import preview: %d proposed matches out of %d wagers / %d bet_logs",
        len(matches), len(data.wagers), len(bet_logs),
    )
    return matches


def apply_import(db: Session, confirmed: List[Dict]) -> ImportSummary:
    """
    Apply confirmed matches to the database.

    ``confirmed`` is a list of dicts, each with:
      - bet_log_id: int
      - dk_wager_id: str
      - dk_wager_amount: float
      - outcome: int | None  (1=win, 0=loss, None=pending — skip)
      - profit_dollars: float
      - payout_amount: float
    """
    summary = ImportSummary(applied=0, wins=0, losses=0, pending=0, total_profit=0.0)

    for m in confirmed:
        if m.get("outcome") is None:
            summary.pending += 1
            continue

        bl = db.query(BetLog).filter(BetLog.id == m["bet_log_id"]).first()
        if bl is None:
            summary.errors.append(f"BetLog {m['bet_log_id']} not found")
            continue

        bl.executed = True
        bl.bet_size_dollars = m["dk_wager_amount"]
        bl.outcome = m["outcome"]
        bl.profit_loss_dollars = m["profit_dollars"]
        # Store payout for audit
        existing_note = bl.notes or ""
        bl.notes = f"DK_ID:{m['dk_wager_id']}" + (f" | {existing_note}" if existing_note else "")

        summary.applied += 1
        summary.total_profit += m["profit_dollars"]
        if m["outcome"] == 1:
            summary.wins += 1
        else:
            summary.losses += 1

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        summary.errors.append(f"DB commit failed: {exc}")
        logger.error("DK import commit failed: %s", exc)

    return summary


# ---------------------------------------------------------------------------
# Direct import (no paper trade required)
# ---------------------------------------------------------------------------

def _find_payout_for_wager(
    wager: DKTransaction,
    payouts_sorted: List[DKTransaction],
    used_payout_ids: set,
) -> Optional[DKTransaction]:
    """Return the first eligible payout for a wager, or None."""
    for payout in payouts_sorted:
        if payout.dk_id in used_payout_ids:
            continue
        if payout.timestamp_utc <= wager.timestamp_utc:
            continue
        if payout.timestamp_utc > wager.timestamp_utc + timedelta(hours=_PAYOUT_WINDOW_HOURS):
            continue
        if payout.amount >= wager.amount * 0.98:
            return payout
    return None


def _pick_best_game(
    wager_ts: datetime,
    candidates: List,
    assigned_game_ids: set,
) -> Optional[object]:
    """
    Pick the best Game object for a wager using temporal proximity.

    Strategy (in priority order):
      1. Unassigned game starting AFTER the wager (pre-game bet — most common)
      2. Unassigned game with smallest absolute time delta
      3. Already-assigned game starting after the wager (spread + total on same game)
      4. Already-assigned game with smallest absolute time delta

    ``assigned_game_ids`` tracks games already suggested to previous wagers so
    we avoid pointing all same-day wagers at game[0].
    """
    def delta_secs(g: object) -> float:
        return abs((g.game_date - wager_ts).total_seconds())

    unassigned = [g for g in candidates if g.id not in assigned_game_ids]
    assigned   = [g for g in candidates if g.id in assigned_game_ids]

    # 1. Unassigned, future
    future_unassigned = [g for g in unassigned if g.game_date >= wager_ts]
    if future_unassigned:
        return min(future_unassigned, key=delta_secs)

    # 2. Unassigned, any direction
    if unassigned:
        return min(unassigned, key=delta_secs)

    # 3. Already-assigned, future (e.g. spread + total on same game)
    future_assigned = [g for g in assigned if g.game_date >= wager_ts]
    if future_assigned:
        return min(future_assigned, key=delta_secs)

    # 4. Already-assigned, any direction
    if assigned:
        return min(assigned, key=delta_secs)

    return None


def preview_direct_import(db: Session, data: DKImportData) -> List[DirectImportPreview]:
    """
    Preview DK wagers for direct creation as real BetLog entries.

    Does NOT require existing paper trades.  For each wager, uses temporal
    proximity to pick the best-fitting game from the database (preferring
    games that start shortly after the wager timestamp, and preferring games
    not already assigned to a prior wager to avoid stacking all same-day
    bets on one game).

    Call ``apply_direct_import`` to commit.
    Returns a list of DirectImportPreview objects.
    """
    items: List[DirectImportPreview] = []
    used_payout_ids: set = set()
    # Tracks game IDs already suggested to a wager (prefer fresh games)
    assigned_game_ids: set = set()
    now_utc = datetime.utcnow()
    payouts_sorted = sorted(data.payouts, key=lambda p: p.timestamp_utc)

    for wager in sorted(data.wagers, key=lambda w: w.timestamp_utc):
        wager_date = wager.timestamp_utc.date()

        # Fetch candidates: same UTC date, with ±1-day fallback
        candidates = (
            db.query(Game)
            .filter(func.date(Game.game_date) == wager_date)
            .order_by(Game.game_date.asc())
            .all()
        )
        if not candidates:
            window_start = datetime(wager_date.year, wager_date.month, wager_date.day) - timedelta(days=1)
            window_end   = window_start + timedelta(days=3)
            candidates = (
                db.query(Game)
                .filter(Game.game_date >= window_start, Game.game_date < window_end)
                .order_by(Game.game_date.asc())
                .all()
            )

        candidate_games = [
            {"id": g.id, "matchup": f"{g.away_team} @ {g.home_team}"}
            for g in candidates
        ]

        # Pick the best game using temporal proximity
        best_game = _pick_best_game(wager.timestamp_utc, candidates, assigned_game_ids)
        if best_game is not None:
            assigned_game_ids.add(best_game.id)
            suggested_id    = best_game.id
            suggested_label = f"{best_game.away_team} @ {best_game.home_team}"
        else:
            suggested_id    = None
            suggested_label = ""

        # Find matching payout
        matched_payout = _find_payout_for_wager(wager, payouts_sorted, used_payout_ids)

        # Determine outcome
        hours_since = (now_utc - wager.timestamp_utc).total_seconds() / 3600
        if matched_payout is not None:
            outcome    = 1
            profit     = matched_payout.amount - wager.amount
            payout_amt = matched_payout.amount
            used_payout_ids.add(matched_payout.dk_id)
        elif hours_since > _SETTLED_THRESHOLD_HOURS:
            outcome    = 0
            profit     = -wager.amount
            payout_amt = 0.0
        else:
            outcome    = None
            profit     = 0.0
            payout_amt = 0.0

        items.append(DirectImportPreview(
            dk_wager_id      = wager.dk_id,
            dk_amount        = wager.amount,
            dk_timestamp     = wager.timestamp_utc,
            outcome          = outcome,
            profit_dollars   = round(profit, 2),
            payout_amount    = payout_amt,
            candidate_games  = candidate_games,
            suggested_game_id= suggested_id,
        ))
        # Attach the label as a plain attribute for the endpoint to forward
        items[-1]._suggested_game_label = suggested_label  # type: ignore[attr-defined]

    logger.info(
        "DK direct import preview: %d wagers, %d with matched games",
        len(items), sum(1 for i in items if i.suggested_game_id),
    )
    return items


def apply_direct_import(db: Session, items: List[Dict]) -> ImportSummary:
    """
    Create new real-money BetLog entries from confirmed direct-import items.

    Each item dict must have:
      - dk_wager_id: str
      - dk_amount: float
      - dk_timestamp: str (ISO 8601)
      - game_id: int
      - outcome: int | None
      - profit_dollars: float
      - pick: str (optional — auto-generated from game if absent)

    Skips any wager whose dk_wager_id is already recorded in BetLog.notes
    to prevent double-imports.
    """
    # Build set of already-imported DK wager IDs from existing notes
    existing_notes = db.query(BetLog.notes).filter(BetLog.notes.like("DK_ID:%")).all()
    already_imported: set = set()
    for (note,) in existing_notes:
        if note:
            # Notes start with "DK_ID:<id>" or "DK_ID:<id> | ..."
            dk_part = note.split("|")[0].strip()
            if dk_part.startswith("DK_ID:"):
                already_imported.add(dk_part[6:].strip())

    summary = ImportSummary(applied=0, wins=0, losses=0, pending=0, total_profit=0.0)

    for item in items:
        dk_wager_id = item.get("dk_wager_id", "?")

        # Skip already-imported wagers
        if dk_wager_id in already_imported:
            logger.debug("Skipping already-imported wager %s", dk_wager_id)
            continue

        game_id = item.get("game_id")
        if not game_id:
            summary.errors.append(
                f"Wager {dk_wager_id}: no game_id selected — skipped"
            )
            continue

        outcome = item.get("outcome")
        dk_amount = float(item.get("dk_amount", 0) or 0)
        profit = float(item.get("profit_dollars", 0) or 0)

        # Parse timestamp
        ts_raw = item.get("dk_timestamp", "")
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
            if ts.tzinfo is not None:
                ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        except (ValueError, TypeError):
            ts = datetime.utcnow()

        # Auto-generate pick from game if not provided
        pick = item.get("pick") or ""
        if not pick:
            game = db.query(Game).filter(Game.id == int(game_id)).first()
            if game:
                pick = f"{game.away_team} @ {game.home_team}"
            else:
                pick = f"DK ${dk_amount:.2f}"

        bet = BetLog(
            game_id=int(game_id),
            pick=pick,
            bet_type="spread",
            odds_taken=-110.0,       # DK CSV does not include odds
            bet_size_dollars=dk_amount,
            bet_size_units=round(dk_amount / 10.0, 2),
            timestamp=ts,
            is_paper_trade=False,
            executed=True,
            outcome=outcome,
            profit_loss_dollars=round(profit, 2) if outcome is not None else None,
            notes=f"DK_ID:{dk_wager_id} | Direct Import",
        )
        db.add(bet)

        if outcome is None:
            summary.pending += 1
        else:
            summary.applied += 1
            summary.total_profit += profit
            if outcome == 1:
                summary.wins += 1
            else:
                summary.losses += 1

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        summary.errors.append(f"DB commit failed: {exc}")
        logger.error("DK direct import commit failed: %s", exc)

    return summary
