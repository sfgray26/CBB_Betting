
import sys
import os
import requests
from datetime import datetime, timedelta

# Add the project root to the path so we can import our models
sys.path.append(os.getcwd())

from backend.models import SessionLocal, BetLog, Game

# DK Authentication from .env
DK_AUTH_TOKEN = os.getenv("DK_AUTH_TOKEN")
DK_COOKIES = os.getenv("DK_COOKIES")

def fetch_dk_bets(bet_status="open"):
    """
    Fetch bets from DraftKings internal API.
    bet_status can be 'open' or 'settled'
    """
    if not DK_AUTH_TOKEN:
        return {"error": "DK_AUTH_TOKEN not found in environment"}

    # New 2026 Path: bettinghistory instead of sportsbook/mybets
    url = f"https://sportsbook-nash.draftkings.com/api/bettinghistory/v1/{bet_status}?format=json"
    headers = {
        "Authorization": f"Bearer {DK_AUTH_TOKEN}",
        "Cookie": DK_COOKIES or "",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def sync():
    db = SessionLocal()
    summary = []
    
    # 1. Sync Open Bets (To mark as 'executed')
    open_dk = fetch_dk_bets("open")
    if "error" in open_dk:
        print(f"Error fetching open bets: {open_dk['error']}")
    else:
        bets = open_dk.get("bets", [])
        for dk_bet in bets:
            # logic to match dk_bet with local BetLog
            # Simplistic matching by pick name for this draft
            pick_name = dk_bet.get("label") # e.g. "Duke -4.5"
            local_bet = db.query(BetLog).filter(BetLog.pick == pick_name, BetLog.executed == False).first()
            if local_bet:
                local_bet.executed = True
                summary.append(f"MATCHED: {pick_name} marked as EXECUTED.")

    # 2. Sync Settled Bets (To mark outcome)
    settled_dk = fetch_dk_bets("settled")
    if "error" in settled_dk:
        print(f"Error fetching settled bets: {settled_dk['error']}")
    else:
        bets = settled_dk.get("bets", [])
        for dk_bet in bets:
            pick_name = dk_bet.get("label")
            payout = dk_bet.get("payout", 0)
            status = dk_bet.get("status") # e.g. "Won", "Lost"
            
            local_bet = db.query(BetLog).filter(BetLog.pick == pick_name, BetLog.outcome == None).first()
            if local_bet:
                local_bet.outcome = 1 if status == "Won" else 0
                local_bet.profit_loss_dollars = payout - local_bet.bet_size_dollars
                summary.append(f"SETTLED: {pick_name} marked as {status.upper()}.")

    db.commit()
    db.close()
    
    if not summary:
        print("No new bet updates found on DraftKings.")
    else:
        print("\n".join(summary))

if __name__ == "__main__":
    sync()
