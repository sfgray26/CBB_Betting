#!/usr/bin/env python3
"""
Quick script to review today's bet logs.
"""
import os
import sys
from datetime import datetime, date, timedelta

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models import SessionLocal, BetLog, Game

def get_todays_bets():
    """Fetch today's bet logs from database."""
    db = SessionLocal()
    try:
        # Get bets from last 24 hours
        since = datetime.utcnow() - timedelta(hours=24)
        
        bets = (
            db.query(BetLog, Game)
            .join(Game, BetLog.game_id == Game.id)
            .filter(BetLog.timestamp >= since)
            .order_by(BetLog.timestamp.desc())
            .all()
        )
        
        return bets
    finally:
        db.close()

def format_bet(bet, game):
    """Format a single bet for display."""
    outcome_str = {
        1: "✅ WIN",
        0: "❌ LOSS", 
        None: "⏳ PENDING"
    }.get(bet.outcome, "?")
    
    paper_str = "[PAPER] " if bet.is_paper_trade else "[REAL] "
    
    return f"""
{paper_str}{outcome_str} | {game.away_team} @ {game.home_team}
  Pick: {bet.pick}
  Odds: {bet.odds_taken:+.0f} | Edge: {bet.conservative_edge:.1%} | Units: {bet.bet_size_units:.2f}
  Model Prob: {bet.model_prob:.1%} | Lower CI: {bet.lower_ci_prob:.1%}
  P&L: {bet.profit_loss_units:+.2f} units | ${bet.profit_loss_dollars:+.2f}
  Time: {bet.timestamp.strftime('%Y-%m-%d %H:%M UTC')}
  Notes: {bet.notes or 'None'}
"""

def main():
    print("=" * 70)
    print(f"BET LOG REVIEW — Last 24 Hours ({datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')})")
    print("=" * 70)
    
    bets = get_todays_bets()
    
    if not bets:
        print("\nNo bets found in last 24 hours.\n")
        return
    
    # Separate by status
    pending = [(b, g) for b, g in bets if b.outcome is None]
    completed = [(b, g) for b, g in bets if b.outcome is not None]
    
    print(f"\n📊 SUMMARY: {len(bets)} total bets")
    print(f"   ⏳ Pending: {len(pending)}")
    print(f"   ✅❌ Completed: {len(completed)}")
    
    # Calculate totals
    total_units = sum(b.profit_loss_units or 0 for b, _ in bets)
    total_dollars = sum(b.profit_loss_dollars or 0 for b, _ in bets)
    
    if completed:
        wins = sum(1 for b, _ in completed if b.outcome == 1)
        losses = len(completed) - wins
        win_rate = wins / len(completed) * 100 if completed else 0
        print(f"   📈 Record: {wins}-{losses} ({win_rate:.1f}%)")
        print(f"   💰 Total P&L: {total_units:+.2f} units | ${total_dollars:+.2f}")
    
    # Show pending first
    if pending:
        print("\n" + "=" * 70)
        print("⏳ PENDING BETS")
        print("=" * 70)
        for bet, game in pending:
            print(format_bet(bet, game))
    
    # Show completed
    if completed:
        print("\n" + "=" * 70)
        print("✅❌ COMPLETED BETS")
        print("=" * 70)
        for bet, game in completed:
            print(format_bet(bet, game))
    
    # CLV Summary
    clv_bets = [b for b, _ in bets if b.clv_points is not None]
    if clv_bets:
        avg_clv = sum(b.clv_points for b in clv_bets) / len(clv_bets)
        positive_clv = sum(1 for b in clv_bets if b.clv_points > 0)
        print("\n" + "=" * 70)
        print("📊 CLV SUMMARY")
        print("=" * 70)
        print(f"   Bets with CLV data: {len(clv_bets)}")
        print(f"   Avg CLV: {avg_clv:+.2f} points")
        print(f"   Positive CLV: {positive_clv}/{len(clv_bets)} ({positive_clv/len(clv_bets)*100:.1f}%)")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
