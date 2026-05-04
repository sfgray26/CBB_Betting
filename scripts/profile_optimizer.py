import os
import cProfile
import pstats
import asyncio
from backend.routers.fantasy import optimize_roster
from backend.models import SessionLocal

os.environ['DATABASE_URL'] = "postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway"

class MockRequest:
    def __init__(self, target_date, yahoo_league_id):
        self.target_date = target_date
        self.yahoo_league_id = yahoo_league_id

async def run_profile():
    req = MockRequest("2026-05-02", "72586")
    db = SessionLocal()
    await optimize_roster(req, db)
    db.close()

if __name__ == "__main__":
    prof = cProfile.Profile()
    prof.enable()
    asyncio.run(run_profile())
    prof.disable()
    
    with open("profile_analysis.txt", "w") as f:
        ps = pstats.Stats(prof, stream=f).sort_stats('cumulative')
        ps.print_stats(20)
    
    print("Profile saved to profile_analysis.txt")
