#!/usr/bin/env python3
"""
Diagnose Discord vs Streamlit UI bet discrepancy.

This script checks:
1. What bets are in the database (what Streamlit sees)
2. What was sent to Discord (if logged)
3. Any data mismatches
"""

import os
import sys
import json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_database_bets():
    """Check what bets are currently in the database."""
    try:
        from backend.models import SessionLocal, BetLog, Game, Prediction
        
        db = SessionLocal()
        try:
            # Get bets from last 24 hours
            since = datetime.utcnow() - timedelta(hours=24)
            
            bets = (
                db.query(BetLog, Game)
                .join(Game, BetLog.game_id == Game.id)
                .filter(BetLog.timestamp >= since)
                .all()
            )
            
            print(f"\n📊 DATABASE BETS (last 24h): {len(bets)} found")
            print("=" * 60)
            
            for bet, game in bets:
                print(f"  • {game.away_team} @ {game.home_team}")
                print(f"    Pick: {bet.pick} | Units: {bet.bet_size_units} | Time: {bet.timestamp.strftime('%H:%M UTC')}")
            
            return len(bets), bets
        finally:
            db.close()
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        return 0, []


def check_json_files():
    """Check temporary JSON files that might have been sent to Discord."""
    json_files = [
        "tmp_bet_details.json",
        "full_today_data.json",
        "current_recommendations.json",
        "tmp_predictions.json",
    ]
    
    print("\n📁 JSON FILE CHECK")
    print("=" * 60)
    
    for filename in json_files:
        filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Try to count bets
                bet_count = 0
                if isinstance(data, list):
                    bet_count = len(data)
                elif isinstance(data, dict):
                    if 'predictions' in data:
                        bet_count = len([p for p in data['predictions'] if p.get('verdict', '').startswith('Bet')])
                    elif 'bets_recommended' in data:
                        bet_count = data['bets_recommended']
                
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                age_hours = (datetime.now() - mtime).total_seconds() / 3600
                
                print(f"  ✅ {filename}")
                print(f"     Bets: {bet_count} | Modified: {age_hours:.1f}h ago")
            except Exception as e:
                print(f"  ⚠️  {filename} - Error reading: {e}")
        else:
            print(f"  ❌ {filename} - Not found")


def check_discord_logs():
    """Check if there are any Discord notification logs."""
    log_dir = os.path.expanduser("~/.openclaw/notifications")
    
    print("\n📱 DISCORD NOTIFICATION LOGS")
    print("=" * 60)
    
    if not os.path.exists(log_dir):
        print(f"  ❌ Log directory not found: {log_dir}")
        return
    
    # List recent log files
    log_files = sorted(
        [f for f in os.listdir(log_dir) if f.endswith('.log')],
        reverse=True
    )[:5]
    
    if not log_files:
        print("  ❌ No log files found")
        return
    
    for log_file in log_files:
        filepath = os.path.join(log_dir, log_file)
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Look for bet-related entries
            bet_lines = [l for l in lines if 'bet' in l.lower() or 'pick' in l.lower()]
            
            print(f"  📄 {log_file}")
            print(f"     Total lines: {len(lines)} | Bet-related: {len(bet_lines)}")
            
            # Show last few bet entries
            for line in bet_lines[-3:]:
                print(f"     > {line.strip()[:80]}")
        except Exception as e:
            print(f"  ⚠️  {log_file} - Error: {e}")


def check_api_predictions():
    """Check what the API would return for today's predictions."""
    print("\n🔌 API PREDICTIONS CHECK (simulated)")
    print("=" * 60)
    
    try:
        from backend.models import SessionLocal, Prediction, Game
        from datetime import date
        
        db = SessionLocal()
        try:
            today = date.today()
            
            predictions = (
                db.query(Prediction)
                .join(Game)
                .filter(Prediction.prediction_date == today)
                .all()
            )
            
            bets = [p for p in predictions if p.verdict.startswith("Bet")]
            
            print(f"  📊 Today ({today}):")
            print(f"     Total predictions: {len(predictions)}")
            print(f"     BET verdicts: {len(bets)}")
            
            for bet in bets[:5]:  # Show first 5
                game = bet.game
                print(f"     • {game.away_team} @ {game.home_team} - {bet.verdict}")
            
            return len(bets)
        finally:
            db.close()
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 0


def main():
    print("=" * 70)
    print("🔍 DISCORD vs STREAMLIT DIAGNOSTIC TOOL")
    print("=" * 70)
    print(f"Current time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    
    # Run all checks
    db_count, db_bets = check_database_bets()
    check_json_files()
    check_discord_logs()
    api_count = check_api_predictions()
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    print(f"  Database bets (last 24h):  {db_count}")
    print(f"  API predictions today:     {api_count}")
    print()
    
    if db_count == 0 and api_count == 0:
        print("⚠️  NO BETS FOUND in database or API")
        print()
        print("Possible causes:")
        print("  1. Nightly analysis hasn't run yet today")
        print("  2. All games resulted in PASS verdict")
        print("  3. Data was cleared or reset")
        print("  4. Discord notification used cached/old data")
    elif db_count > 0 or api_count > 0:
        print("✅ BETS FOUND in system")
        print()
        print("If Discord shows different count:")
        print("  1. Check if Discord message is from a different date")
        print("  2. Check if Discord used cached JSON file data")
        print("  3. UI might be filtering differently (e.g., game time cutoff)")
    
    print()
    print("Next steps:")
    print("  • Run nightly analysis if not done: python scripts/run_nightly.py")
    print("  • Check Discord message timestamp vs current time")
    print("  • Verify DISCORD_CHANNEL_ID env var is correct")


if __name__ == "__main__":
    main()
