import os
import time
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

os.environ['DATABASE_URL'] = "postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway"

from backend.fantasy_baseball.mcmc_calibration import convert_yahoo_roster_to_mcmc_format
from backend.models import SessionLocal

# Mock roster
mock_roster = [{"name": "Juan Soto", "positions": ["OF"]}, {"name": "Aaron Judge", "positions": ["OF"]}]

print("=== Benchmarking MCMC Calibration ===")
db = SessionLocal()

# First call (cold cache)
start = time.time()
result1 = convert_yahoo_roster_to_mcmc_format(mock_roster, db=db)
elapsed1 = time.time() - start
print(f"Cold call: {elapsed1:.2f}s")

# Second call (warm cache)
start = time.time()
result2 = convert_yahoo_roster_to_mcmc_format(mock_roster, db=db)
elapsed2 = time.time() - start
print(f"Warm call: {elapsed2:.2f}s")

db.close()
