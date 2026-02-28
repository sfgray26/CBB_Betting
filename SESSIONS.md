# CBB Edge — Session History

## 2026-02-28 (Saturday)

### 13:00 — Core Portfolio & Mapping Updates (P0)

#### Changes:
1.  **Implemented Global Proportional Scaler (P0)**:
    - Modified `backend/services/analysis.py` to move the paper-bet allocation logic outside of the main game-analysis loop.
    - **Removed Sequential Scaling**: The system no longer enforces the daily cap game-by-game, which previously exhausted capacity for later-slate games regardless of EV.
    - Added a global scaling calculation that sums all requested dollars for the day's slate and applies a proportional `scaling_factor` if the total exceeds the remaining daily dollars (typically a $50-100 limit based on bankroll).
    - Updated `_create_paper_bet` to accept an optional `scaled_bet_dollars` parameter, allowing the global scaler to dictate exact sizing while preserving EV Displacement logic.

2.  **Fixed Team Mapping Bug (SWAC/Summit League)**:
    - Added explicit mappings in `backend/services/team_mapping.py` for:
        - `North Dakota Fighting Hawks` -> `North Dakota`
        - `North Dakota St Bison` -> `North Dakota St.`
        - `Alabama A&M Bulldogs` -> `Alabama A&M`
        - `Alabama St Hornets` -> `Alabama St.`
    - Verified against KenPom naming conventions to ensure these teams correctly resolve to their respective rating profiles during nightly analysis runs.

#### Verification:
- Created and ran `tests/test_global_scaling.py` (since deleted) to confirm that:
    - Multiple high-unit bets are scaled down proportionally to fit exactly within the daily cap.
    - EV Displacement still functions correctly to free capacity for late-breaking high-edge opportunities.
    - `scaled_bet_dollars` successfully overrides unit-based calculations in `BetLog` generation.
