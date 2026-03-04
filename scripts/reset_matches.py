
import sys
import os
# Add the project root to the path
sys.path.append(os.getcwd())

from backend.main import SessionLocal
from backend.models import BetLog

def reset():
    db = SessionLocal()
    count = db.query(BetLog).filter(
        BetLog.executed == True, 
        BetLog.notes.like('DK_ID:%')
    ).update(
        {
            'executed': False, 
            'notes': None, 
            'outcome': None, 
            'profit_loss_dollars': 0.0
        },
        synchronize_session=False
    )
    db.commit()
    db.close()
    print(f"RESETS COMPLETE: {count} incorrect matches cleared.")

if __name__ == "__main__":
    reset()
