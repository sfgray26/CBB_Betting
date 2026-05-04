import os
import logging
import time
os.environ['DATABASE_URL'] = "postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway"

logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

from backend.fantasy_baseball.daily_lineup_optimizer import DailyLineupOptimizer

print("=== Starting Optimizer Query Count ===")
optimizer = DailyLineupOptimizer()
start = time.time()
# The mission said optimize_lineup, but the code has DailyLineupOptimizer
# Let's check DailyLineupOptimizer.optimize
try:
    # Based on verify_endpoints.py, it seems we use /api/fantasy/roster/optimize
    # which calls optimize_roster in backend/routers/fantasy.py
    from backend.routers.fantasy import optimize_roster
    from backend.models import SessionLocal
    
    class MockRequest:
        def __init__(self, target_date, yahoo_league_id):
            self.target_date = target_date
            self.yahoo_league_id = yahoo_league_id
            
    req = MockRequest("2026-05-02", "72586")
    db = SessionLocal()
    # We need to run this in an async loop if it's async
    import asyncio
    
    async def run_test():
        result = await optimize_roster(req, db)
        print(f"\nTotal starters: {len(result.starters)}")
        
    asyncio.run(run_test())
    db.close()
except Exception as e:
    print(f"FAILED: {e}")

print(f"\nElapsed: {time.time() - start:.2f}s")
