import sys
import os
from datetime import datetime, timedelta
import io

# Add the project root to the path
sys.path.append(os.getcwd())

def run():
    output = io.StringIO()
    try:
        from backend.models import SessionLocal, BetLog

        db = SessionLocal()
        
        # Shift to a 24-hour lookback to catch all games for the current 'Day'
        # This aligns better with how the UI handles different timezones (EST/UTC)
        twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
        
        bets = db.query(BetLog).filter(
            BetLog.timestamp >= twenty_four_hours_ago,
            BetLog.is_paper_trade.is_(True),
            BetLog.bet_size_units > 0
        ).all()
        
        if not bets:
            output.write("🎰 NO RECOMMENDED BETS IN THE LAST 24 HOURS.\n")
        else:
            output.write(f"🏀 CURRENT SLATE RECOMMENDED BETS ({len(bets)}):\n")
            output.write("=" * 25 + "\n")
            for bet in bets:
                pick = bet.pick or "Unknown"
                odds = bet.odds_taken if bet.odds_taken is not None else 0
                units = bet.bet_size_units if bet.bet_size_units is not None else 0.0
                
                output.write(f"✅ {pick}\n")
                output.write(f"   {odds:+.0f} | {units:.2f}u\n")
                output.write("-" * 25 + "\n")
        
        db.close()

    except Exception as e:
        output.write(f"Sync Error: {str(e)}\n")

    message = output.getvalue()
    # Fix Windows console encoding for emoji
    safe_print = lambda s: print(s.encode('ascii', 'ignore').decode('ascii'))
    safe_print(message)
    
    # Send to Discord via OpenClaw
    if message.strip():
        safe_msg = message.replace('"', "'").replace("`", "")
        # PAUSED (2026-04-21): OpenClaw Discord notifications disabled.
        # os.system(f'openclaw message send --channel discord --target "1477436117426110615" --message "{safe_msg}"')
        safe_print(f"[PAUSED] Would send: {safe_msg[:80]}...")

if __name__ == "__main__":
    run()
