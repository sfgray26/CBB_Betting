
import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.append(os.getcwd())

from backend.main import SessionLocal
from backend.models import BetLog

def cleanup():
    db = SessionLocal()
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # 1. Clean up duplicate Duke bets
    duke_bets = db.query(BetLog).filter(
        BetLog.timestamp >= today, 
        BetLog.pick == 'Duke -3.0'
    ).order_by(BetLog.id.desc()).all()
    
    if len(duke_bets) > 1:
        for b in duke_bets[1:]:
            db.delete(b)
        print(f"CLEANUP: Deleted {len(duke_bets)-1} duplicate Duke bets.")

    # 2. Clean up generic game matchups that aren't specific picks (e.g., 'UNC @ Duke')
    # These often get created by the odds monitor during testing
    generic_bets = db.query(BetLog).filter(
        BetLog.timestamp >= today,
        BetLog.pick.contains(" @ ")
    ).all()
    
    if generic_bets:
        for b in generic_bets:
            db.delete(b)
        print(f"CLEANUP: Deleted {len(generic_bets)} generic matchup entries.")

    db.commit()
    db.close()

if __name__ == "__main__":
    cleanup()
