
import sys
import os
import csv
from datetime import datetime, timedelta
import re
import requests

# Add the project root to the path
sys.path.append(os.getcwd())

from backend.models import SessionLocal, BetLog

# Discord Notification Config
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL") # We can also use openclaw CLI

def parse_amount(amt_str):
    if not amt_str: return 0.0
    clean = amt_str.replace("$", "").replace(",", "").replace(" ", "").replace("(", "").replace(")", "")
    try: return float(clean)
    except: return 0.0

def import_csv(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    db = SessionLocal()
    updates = 0
    matches = 0
    total_profit = 0.0

    print("\n--- STARTING STRICT SYNC ---")
    with open(file_path, mode='r', encoding='utf-8') as f:
        first_line = f.readline()
        f.seek(0)
        delim = '\t' if '\t' in first_line else ','
        reader = csv.reader(f, delimiter=delim)
        
        if "date" in first_line.lower():
            next(reader)

        matched_db_ids = set()

        for row in reader:
            if not row or len(row) < 5: continue
            
            trans_date_str = row[0].strip()
            details = row[2].strip()        
            amount = parse_amount(row[4].strip())
            
            match_id = re.search(r'ID: ([A-F0-9-]+)', details)
            if not match_id: continue
            dk_id = match_id.group(1)

            dt_csv = None
            for fmt in ("%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M", "%m/%d/%Y %I:%M %p"):
                try:
                    dt_csv = datetime.strptime(trans_date_str, fmt)
                    break
                except: continue
            if not dt_csv: continue

            # Eastern to UTC
            dt_csv_utc = dt_csv + timedelta(hours=5)

            if "wager" in details.lower():
                # STRICT MATCHING: Narrow window (2 mins) + Same Calendar Day
                start_range = dt_csv_utc - timedelta(minutes=2)
                end_range = dt_csv_utc + timedelta(minutes=2)
                
                lb = db.query(BetLog).filter(
                    BetLog.timestamp >= start_range,
                    BetLog.timestamp <= end_range,
                    BetLog.executed == False,
                    ~BetLog.id.in_(matched_db_ids)
                ).first()
                
                if lb:
                    lb.executed = True
                    lb.bet_size_dollars = abs(amount)
                    lb.notes = f"DK_ID: {dk_id}"
                    matched_db_ids.add(lb.id)
                    matches += 1
                    print(f"Linked: {lb.pick} (${abs(amount)})")
            
            elif "win payout" in details.lower():
                local_bet = db.query(BetLog).filter(
                    BetLog.notes.contains(dk_id),
                    BetLog.outcome == None
                ).first()
                
                if local_bet:
                    local_bet.outcome = 1
                    profit = amount - (local_bet.bet_size_dollars or 0)
                    local_bet.profit_loss_dollars = profit
                    updates += 1
                    total_profit += profit
                    print(f"Settled WIN: {local_bet.pick} (+${profit:.2f})")

    db.commit()
    db.close()
    
    summary = f"🔄 **DraftKings Sync Complete**\n- Matches: {matches}\n- Wins: {updates}\n- Profit: ${total_profit:.2f}"
    print(f"\n{summary}")
    
    # Notify Discord via OpenClaw CLI
    os.system(f'openclaw message send --channel discord --target "1477436117426110615" --message "{summary}"')

if __name__ == "__main__":
    import_csv("draftkings_history.csv")
