# Steamer 2026 Projections: Data Acquisition Guide

## 1. Overview
Steamer remains the industry standard for preseason and rest-of-season (ROS) projections. For the 2026 season, full datasets covering ~750 batters and ~450 pitchers are available via FanGraphs.

## 2. Direct Download (CSV)
FanGraphs provides a manual CSV export for members.
*   **Batters:** [Steamer 2026 Hitter Projections](https://www.fangraphs.com/projections?pos=all&stats=bat&type=steamer)
*   **Pitchers:** [Steamer 2026 Pitcher Projections](https://www.fangraphs.com/projections?pos=all&stats=pit&type=steamer)

## 3. Programmatic Acquisition (Python)
The `pybaseball` library is the recommended tool for bulk data retrieval.

### **Installation**
```bash
pip install pybaseball
```

### **Usage Example**
```python
from pybaseball import batting_stats, pitching_stats, amateur_draft

# Fetch Steamer projections (Note: verify latest pybaseball support for 2026)
# Typically: stats = pybaseball.get_projections(type='steamer')
```

## 4. Key Benchmarks (Steamer 2026)
*   **Batters:** Look for `K%` and `BB%` stability in the first 100 PAs.
*   **Pitchers:** Focus on `FIP` vs `ERA` discrepancies in the early season.
