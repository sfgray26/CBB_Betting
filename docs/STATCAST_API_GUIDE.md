# Statcast API Guide: Bulk Acquisition via `pybaseball`

## 1. Overview
The `pybaseball` library is the most robust tool for scraping and processing Statcast (Savant) data in Python. This guide covers bulk acquisition while avoiding the Savant 40,000-row return limit.

## 2. Basic Statcast Retrieval
### **Installation**
```bash
pip install pybaseball
```

### **Single-Day Fetch**
```python
from pybaseball import statcast
data = statcast(start_dt="2026-03-10", end_dt="2026-03-11")
```

## 3. Bulk Data Retrieval (The "Chunking" Strategy)
Statcast will time out or truncate requests larger than ~40,000 rows. For a full season (~700,000 rows), use a weekly loop.

```python
import pandas as pd
from pybaseball import statcast
from datetime import datetime, timedelta

def get_bulk_statcast(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    all_data = []
    current_date = start
    
    while current_date < end:
        next_date = current_date + timedelta(days=7)
        if next_date > end:
            next_date = end
        
        print(f"Fetching: {current_date.strftime('%Y-%m-%d')} to {next_date.strftime('%Y-%m-%d')}")
        chunk = statcast(start_dt=current_date.strftime('%Y-%m-%d'), 
                         end_dt=next_date.strftime('%Y-%m-%d'))
        all_data.append(chunk)
        current_date = next_date
        
    return pd.concat(all_data)

# Usage
# df = get_bulk_statcast("2025-04-01", "2025-10-01")
```

## 4. Key Metrics for 2026
*   **Hitter:** `launch_speed`, `launch_angle`, `bat_speed`, `squared_up_pct`.
*   **Pitcher:** `release_speed`, `pfx_x` (horizontal break), `pfx_z` (vertical break), `spin_rate`.

## 5. Savant Scraping Alternatives
If the `statcast()` function is throttled, use `pybaseball.statcast_batter()` or `pybaseball.statcast_pitcher()` for player-specific deep dives.
