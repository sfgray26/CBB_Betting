"""
audit_confidence.py — Historical SNR vs Outcome Correlation

Queries all resolved BET-verdict predictions that have SNR scores, then
calculates cover rate per SNR tier to help calibrate whether the 0.5× Kelly
floor is actually conservative enough.

Usage:
    python scripts/audit_confidence.py [--days 90] [--min-bets 5]

Output:
    - SNR tier table: count, cover rate, avg edge, avg units
    - Kelly floor recommendation based on actual performance
    - Integrity verdict breakdown: CONFIRMED vs CAUTION vs VOLATILE cover rates
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.models import SessionLocal, Prediction


_SNR_TIERS = [
    ("0.90-1.00 (Alpha)",   0.9, 1.01),
    ("0.70-0.90 (Strong)",  0.7, 0.9),
    ("0.50-0.70 (Moderate)",0.5, 0.7),
    ("0.00-0.50 (Weak)",    0.0, 0.5),
]

_INT_GROUPS = ["CONFIRMED", "CAUTION", "VOLATILE", "not run"]


def _covered(pred) -> bool | None:
    """
    Return True if the bet covered, False if not, None if not determinable.

    Logic: for a BET verdict, bet_side is in full_analysis.calculations.
    If bet_side == "home": covered when actual_margin > spread (home needed to
    beat the number from the home side).
    If bet_side == "away": covered when -actual_margin > -spread, i.e.
    actual_margin < spread.
    Push (actual_margin == spread) → None.
    """
    actual = pred.actual_margin
    if actual is None:
        return None

    fa = pred.full_analysis or {}
    calcs = fa.get("calculations", {})
    inputs = fa.get("inputs", {})
    odds   = inputs.get("odds", {})

    spread   = odds.get("spread")     # home team spread (negative = home fav)
    bet_side = calcs.get("bet_side", "home")

    if spread is None:
        return None

    if bet_side == "home":
        # home team needs to win by more than |spread| if favoured
        covered_margin = actual - spread
    else:
        # away team needs to cover: actual_margin < spread
        covered_margin = spread - actual

    if covered_margin == 0:
        return None   # push
    return covered_margin > 0


def _integrity_group(verdict: str | None) -> str:
    if verdict is None:
        return "not run"
    v = verdict.upper()
    if "CONFIRMED" in v:
        return "CONFIRMED"
    if "CAUTION" in v:
        return "CAUTION"
    if "VOLATILE" in v:
        return "VOLATILE"
    if "ABORT" in v or "RED FLAG" in v:
        return "ABORT"
    return "other"


def _pct(n, d):
    return f"{n/d:.1%}" if d else "—"


def _mean(vals):
    return sum(vals) / len(vals) if vals else None


def run_audit(days: int = 90, min_bets: int = 5):
    db = SessionLocal()
    try:
        from datetime import datetime, timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)

        preds = (
            db.query(Prediction)
            .filter(
                Prediction.snr.isnot(None),
                Prediction.actual_margin.isnot(None),
                Prediction.verdict.like("Bet%"),
                Prediction.created_at >= cutoff,
            )
            .all()
        )
    finally:
        db.close()

    if not preds:
        print(f"No resolved BET predictions with SNR data in the last {days} days.")
        print("This is expected on a fresh deployment - run after the season accumulates results.")
        return

    print(f"\n{'='*64}")
    print(f"  CBB Edge - Confidence Calibration Audit  ({len(preds)} bets, {days}d)")
    print(f"{'='*64}\n")

    # ----------------------------------------------------------------
    # Section 1: SNR tier breakdown
    # ----------------------------------------------------------------
    print("SNR TIER BREAKDOWN")
    print(f"{'Tier':<24} {'N':>5}  {'Cover%':>7}  {'Avg Edge':>9}  {'Avg Units':>10}")
    print("-" * 60)

    tier_rows = []
    for label, lo, hi in _SNR_TIERS:
        bucket = [p for p in preds if lo <= (p.snr or 0.0) < hi]
        if not bucket:
            tier_rows.append((label, 0, None, None, None))
            print(f"{label:<24} {'--':>5}")
            continue
        outcomes = [_covered(p) for p in bucket]
        resolved = [o for o in outcomes if o is not None]
        wins     = sum(1 for o in resolved if o)
        edges    = [p.edge_conservative or 0.0 for p in bucket]
        units    = [p.recommended_units or 0.0 for p in bucket]
        cover_r  = wins / len(resolved) if resolved else None
        tier_rows.append((label, len(bucket), cover_r, _mean(edges), _mean(units)))
        print(
            f"{label:<24} {len(bucket):>5}  "
            f"{_pct(wins, len(resolved)):>7}  "
            f"{(_mean(edges) or 0):.2%}  "
            f"{(_mean(units) or 0):>9.2f}u"
        )

    # ----------------------------------------------------------------
    # Section 2: Integrity verdict breakdown
    # ----------------------------------------------------------------
    print(f"\nINTEGRITY VERDICT BREAKDOWN")
    print(f"{'Group':<14} {'N':>5}  {'Cover%':>7}  {'Avg Kelly x':>11}")
    print("-" * 42)

    for group in _INT_GROUPS:
        bucket = [p for p in preds if _integrity_group(p.integrity_verdict) == group]
        if not bucket:
            continue
        outcomes = [_covered(p) for p in bucket]
        resolved = [o for o in outcomes if o is not None]
        wins     = sum(1 for o in resolved if o)
        scalars  = [
            (p.snr_kelly_scalar or 1.0) *
            (p.full_analysis or {}).get("calculations", {}).get("integrity_kelly_scalar", 1.0)
            for p in bucket
        ]
        print(
            f"{group:<14} {len(bucket):>5}  "
            f"{_pct(wins, len(resolved)):>7}  "
            f"{_mean(scalars) or 0:>10.2f}x"
        )

    # ----------------------------------------------------------------
    # Section 3: Kelly floor recommendation
    # ----------------------------------------------------------------
    print(f"\nKELLY FLOOR CALIBRATION")
    alpha = [r for r in tier_rows if r[1] >= min_bets and r[2] is not None]
    if len(alpha) < 2:
        print(f"Insufficient data (need >={min_bets} bets per tier for calibration).")
        print("Current SNR_KELLY_FLOOR=0.5 remains in effect (conservative baseline).")
        return

    # Simple heuristic: if the weakest SNR tier cover rate is within 3 ppts of
    # the strongest, the floor could be raised. If it's >10 ppts below, lower it.
    best_rate  = max(r[2] for r in alpha)
    worst_rate = min(r[2] for r in alpha)
    gap = best_rate - worst_rate

    print(f"Best-tier cover rate:   {best_rate:.1%}")
    print(f"Worst-tier cover rate:  {worst_rate:.1%}")
    print(f"Gap:                    {gap:.1%}")

    if gap < 0.03:
        rec = "RAISE floor to 0.65 — SNR has low predictive power; uniform sizing is fine."
    elif gap < 0.08:
        rec = "MAINTAIN floor at 0.5 — SNR is adding value but gap is within noise."
    else:
        rec = "LOWER floor to 0.35 — SNR strongly predicts outcomes; penalise low-SNR more."

    print(f"\nRecommendation: {rec}")
    print(f"To apply: set env var SNR_KELLY_FLOOR=<value> and update IDENTITY.md.")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit SNR confidence calibration")
    parser.add_argument("--days",     type=int, default=90,
                        help="Lookback window in days (default: 90)")
    parser.add_argument("--min-bets", type=int, default=5,
                        help="Min bets per tier for calibration (default: 5)")
    args = parser.parse_args()
    run_audit(days=args.days, min_bets=args.min_bets)
