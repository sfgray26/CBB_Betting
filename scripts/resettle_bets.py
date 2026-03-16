"""
Re-settle audit script for historical BetLog entries.

Usage:
    python scripts/resettle_bets.py            # dry-run: prints discrepancy report only
    python scripts/resettle_bets.py --apply    # commits corrected outcomes to DB

Connects to the database via DATABASE_URL from the environment (or .env file).
Handles missing DATABASE_URL gracefully — exits with a clear error message.

Exit codes:
    0  no discrepancies found (or --apply completed successfully)
    1  discrepancies found in dry-run mode
    2  fatal error (DB connection, import failure, etc.)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, NamedTuple

# Allow running from the repo root without installing the package.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Load .env before importing anything that reads DATABASE_URL.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_repo_root, ".env"))
except ImportError:
    pass  # python-dotenv not installed — rely on real env vars


def _check_db_url() -> None:
    """Exit early with a clear message if DATABASE_URL is absent."""
    if not os.getenv("DATABASE_URL"):
        print(
            "ERROR: DATABASE_URL environment variable is not set.\n"
            "Set it in your shell or in a .env file at the project root."
        )
        sys.exit(2)


_check_db_url()

try:
    from backend.models import BetLog, Game, SessionLocal
    from backend.services.bet_tracker import calculate_bet_outcome
    from backend.utils.env_utils import get_float_env
except ImportError as exc:
    print(f"ERROR: Could not import backend modules: {exc}")
    print("Run this script from the repository root, e.g.:")
    print("    python scripts/resettle_bets.py")
    sys.exit(2)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Discrepancy(NamedTuple):
    bet_id: int
    pick: str
    game_label: str
    old_outcome: int
    new_outcome: int
    old_pnl: float
    new_pnl: float


OUTCOME_LABEL = {1: "WIN", 0: "LOSS", -1: "PUSH"}


def _fmt_outcome(v: int) -> str:
    return OUTCOME_LABEL.get(v, f"?({v})")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def find_discrepancies(starting_bankroll: float) -> List[Discrepancy]:
    """
    Re-run calculate_bet_outcome on every settled BetLog that has a linked game.
    Returns a list of bets where the recalculated outcome or P&L differs from
    what is currently stored.
    """
    db = SessionLocal()
    discrepancies: List[Discrepancy] = []

    try:
        settled_bets = (
            db.query(BetLog)
            .filter(BetLog.outcome.isnot(None), BetLog.game_id.isnot(None))
            .all()
        )

        print(f"Scanning {len(settled_bets)} settled bet(s) with a linked game...")

        for bet in settled_bets:
            game: Game | None = db.query(Game).filter(Game.id == bet.game_id).first()
            if game is None:
                print(f"  WARNING: bet {bet.id} references game_id={bet.game_id} which does not exist — skipping")
                continue

            if game.home_score is None or game.away_score is None:
                # Game result not yet recorded — cannot re-settle.
                continue

            new_result = calculate_bet_outcome(bet, game, starting_bankroll)

            if new_result is None:
                # Resolver could not determine home/away — flag as unresolvable.
                game_label = f"{game.away_team} @ {game.home_team}"
                discrepancies.append(
                    Discrepancy(
                        bet_id=bet.id,
                        pick=bet.pick,
                        game_label=game_label,
                        old_outcome=bet.outcome,
                        new_outcome=-99,  # sentinel: unresolvable
                        old_pnl=bet.profit_loss_dollars or 0.0,
                        new_pnl=0.0,
                    )
                )
                continue

            outcome_changed = new_result.outcome != bet.outcome
            # Use a small epsilon for float comparison on P&L.
            pnl_changed = abs(new_result.profit_loss_dollars - (bet.profit_loss_dollars or 0.0)) > 0.005

            if outcome_changed or pnl_changed:
                game_label = f"{game.away_team} @ {game.home_team}"
                discrepancies.append(
                    Discrepancy(
                        bet_id=bet.id,
                        pick=bet.pick,
                        game_label=game_label,
                        old_outcome=bet.outcome,
                        new_outcome=new_result.outcome,
                        old_pnl=bet.profit_loss_dollars or 0.0,
                        new_pnl=new_result.profit_loss_dollars,
                    )
                )

    finally:
        db.close()

    return discrepancies


def apply_corrections(discrepancies: List[Discrepancy], starting_bankroll: float) -> int:
    """
    Re-run calculate_bet_outcome for each discrepant bet and commit the
    corrected values to the database.

    Skips bets with new_outcome == -99 (unresolvable picks).

    Returns the number of records updated.
    """
    db = SessionLocal()
    updated = 0

    try:
        for disc in discrepancies:
            if disc.new_outcome == -99:
                print(f"  SKIP bet {disc.bet_id} ({disc.pick}): pick still unresolvable — manual review required")
                continue

            bet: BetLog | None = db.query(BetLog).filter(BetLog.id == disc.bet_id).first()
            if bet is None:
                print(f"  SKIP bet {disc.bet_id}: not found in DB")
                continue

            game: Game | None = db.query(Game).filter(Game.id == bet.game_id).first()
            if game is None:
                print(f"  SKIP bet {disc.bet_id}: game {bet.game_id} not found")
                continue

            new_result = calculate_bet_outcome(bet, game, starting_bankroll)
            if new_result is None:
                print(f"  SKIP bet {disc.bet_id}: resolver returned None at apply time")
                continue

            bet.outcome = new_result.outcome
            bet.profit_loss_dollars = new_result.profit_loss_dollars
            bet.profit_loss_units = new_result.profit_loss_units
            updated += 1

        db.commit()
        print(f"\nCommitted {updated} correction(s) to the database.")

    except Exception as exc:
        db.rollback()
        print(f"\nERROR during apply — rolled back all changes: {exc}")
        sys.exit(2)
    finally:
        db.close()

    return updated


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def print_report(discrepancies: List[Discrepancy]) -> None:
    if not discrepancies:
        print("\nNo discrepancies found. All settled bets are consistent with the current model.")
        return

    print(f"\n{'=' * 80}")
    print(f"DISCREPANCY REPORT — {len(discrepancies)} bet(s) differ from stored outcome")
    print(f"{'=' * 80}")
    print(
        f"{'Bet ID':>6}  {'Pick':<30}  {'Game':<40}  "
        f"{'Old':>5}  {'New':>9}  {'Old P&L':>9}  {'New P&L':>9}"
    )
    print("-" * 120)

    pnl_delta_total = 0.0
    for d in discrepancies:
        new_label = "UNRESOLVABLE" if d.new_outcome == -99 else _fmt_outcome(d.new_outcome)
        pnl_delta = d.new_pnl - d.old_pnl if d.new_outcome != -99 else 0.0
        pnl_delta_total += pnl_delta
        print(
            f"{d.bet_id:>6}  {d.pick:<30}  {d.game_label:<40}  "
            f"{_fmt_outcome(d.old_outcome):>5}  {new_label:>9}  "
            f"${d.old_pnl:>8.2f}  ${d.new_pnl:>8.2f}"
        )

    print("-" * 120)
    print(f"{'Total P&L delta from corrections:':>95}  ${pnl_delta_total:>8.2f}")
    print(f"{'=' * 80}")
    unresolvable = sum(1 for d in discrepancies if d.new_outcome == -99)
    if unresolvable:
        print(
            f"\nWARNING: {unresolvable} bet(s) marked UNRESOLVABLE — the pick team name"
            " could not be matched to either side of the game even with fuzzy logic."
            " These require manual review."
        )
    print(
        "\nRun with --apply to commit corrected outcomes.\n"
        "UNRESOLVABLE bets are always skipped even with --apply."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-settle historical BetLog entries and report P&L discrepancies."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Commit corrected outcomes to the database (default: dry-run only).",
    )
    parser.add_argument(
        "--starting-bankroll",
        type=float,
        default=None,
        help="Override STARTING_BANKROLL env var for unit-value calculation.",
    )
    args = parser.parse_args()

    starting_bankroll = args.starting_bankroll
    if starting_bankroll is None:
        starting_bankroll = get_float_env("STARTING_BANKROLL", "1000")

    print(f"CBB Edge — Bet Resettlement Audit")
    print(f"Starting bankroll: ${starting_bankroll:,.2f}")
    print(f"Mode: {'APPLY (writes to DB)' if args.apply else 'DRY RUN (read-only)'}")
    print()

    discrepancies = find_discrepancies(starting_bankroll)
    print_report(discrepancies)

    if args.apply and discrepancies:
        apply_corrections(discrepancies, starting_bankroll)
    elif not discrepancies:
        sys.exit(0)
    else:
        # Dry-run with discrepancies found — signal to CI that action is needed.
        sys.exit(1)


if __name__ == "__main__":
    main()
