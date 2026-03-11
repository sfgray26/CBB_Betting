# CBB Edge: Betting History & Codebase Audit Report
**Date:** March 11, 2026  
**Auditor:** Lead Research Analyst (Gemini CLI)  
**Scope:** Review of `docs/response_1773233302014.json` and `backend/services/bet_tracker.py`

---

## 1. Executive Summary
A comprehensive review of the betting history and underlying codebase reveals a **critical systemic flaw** in the automated bet-grading logic. While the predictive model (v9.1) employs advanced analytical techniques, the function used to settle bets (`calculate_bet_outcome`) contains a naming-mismatch bug that invalidates a significant portion of the reported performance data. This has led to both "False Wins" and "False Losses" in the historical records.

## 2. Key Findings: Betting History Analysis
A cross-reference of the JSON betting log against real-world game reports from March 7–10, 2026, confirms multiple instances of incorrect grading.

### **Major Discrepancies Identified**
| Bet ID | Date | Pick | Reported Outcome | Real-World Result | Actual Coverage | Audit Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **112** | Mar 10 | Eastern Washington -3.5 | **WIN (1)** | Idaho 81, EWU 68 | **LOSS (-13)** | **MISGRADED** |
| **2** | Mar 07 | UNC Greensboro -7.5 | **WIN (1)** | UNCG 75, Wofford 72 | **LOSS (+3)** | **MISGRADED** |
| **41** | Mar 07 | Samford Bulldogs -1.5 | **LOSS (0)** | Samford 82, Furman 75 | **WIN (+7)** | **MISGRADED** |

### **Verified Correct (Control Group)**
*   **Bet #11 (Villanova -12.5)**: Won 91-78 (+13). Marked as **Win**. Correct.
*   **Bet #5 (Virginia -9.5)**: Won 76-72 (+4). Marked as **Loss**. Correct.
*   **Bet #110 (Gonzaga -19.5)**: Won 79-68 (+11). Marked as **Loss**. Correct.

---

## 3. Root Cause: Technical Debt in `bet_tracker.py`
The source of these errors is a "Phantom Away Team" bug in the `calculate_bet_outcome` function.

### **Code Analysis**
The logic for determining the picked team's perspective is as follows:
```python
# From backend/services/bet_tracker.py
team_is_home = team.lower() == game.home_team.lower()

if team_is_home:
    margin = home_score - away_score
else:
    # BUG: If the name doesn't match home_team exactly, it defaults to Away
    margin = away_score - home_score
```

### **The Failure Mechanism**
When a `pick` includes a mascot (e.g., "Samford Bulldogs") but the `Game` record uses a raw name (e.g., "Samford"), the exact string match fails. 
1.  **Perspective Swap**: The code defaults to `team_is_home = False`.
2.  **Margin Inversion**: The margin is calculated from the *Away* perspective (`away_score - home_score`).
3.  **Erroneous Grading**: If the actual Home team won (e.g., Samford), the code calculates a negative margin and records a loss. If the actual Home team lost (e.g., Eastern Washington), the code calculates a positive margin and records a win.

---

## 4. Codebase Architectural Review

### **Strengths**
*   **Analytical Depth**: The `CBBEdgeModel` (v9.1) is robust, featuring 2-layer Monte Carlo simulations, Signal-to-Noise (SNR) confidence engine, and fatigue/injury adjustments.
*   **Market Blending**: Excellent implementation of time-decayed weighting that defers to sharp market consensus (Pinnacle/Circa) as tipoff approaches.
*   **Safety Guards**: Effective Z-score divergence checks prevent betting on anomalous data.

### **Weaknesses**
*   **Validation Bottleneck**: High-quality predictive signals are being compromised by low-quality downstream grading logic.
*   **Mapping Inconsistency**: While `team_mapping.py` contains 300+ normalization rules, they are not utilized during the bet-settling phase.
*   **Performance Metrics**: The current profit/loss (P&L) and win-rate metrics in the system are statistically unreliable due to the grading bug.

---

## 5. Strategic Recommendations

### **Priority 1: Immediate Bug Fix**
Refactor `calculate_bet_outcome` to use fuzzy matching or a confirmed mapping from `team_mapping.py`. The logic must explicitly verify that the picked team matches *either* the home or away team before proceeding with margin calculations.

### **Priority 2: Historical Re-Grading**
Execute a "re-settlement" script that:
1.  Iterates through all completed `BetLog` entries.
2.  Correctly maps the `pick` to the `Game` record using normalized names.
3.  Recalculates outcome and P&L based on verified results.

### **Priority 3: Schema Stabilization**
Migrate the `Game` and `BetLog` tables to use a consistent `team_id` or `canonical_name` based on KenPom/D1-Averages to eliminate reliance on inconsistent raw string names from external APIs.

---
**Report Status:** FINAL  
**Action Required:** High-Priority Bug Fix in `backend/services/bet_tracker.py`
