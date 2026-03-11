# Using Sportsbook Data for Fantasy Baseball & DFS Capability Roadmap
**Date:** March 11, 2026  
**Subject:** Translating Vegas Odds to Fantasy Baseball Value & DFS Integration

---

## 1. The Sportsbook Advantage in Fantasy Baseball
Sportsbook lines—moneylines, run totals, and player props—represent the most accurate, financially-backed, and continuously updated projections available. By converting these odds into fantasy expectations, we bypass traditional projection models and align our fantasy decisions with "sharp" money.

### **A. Pitcher Evaluation via Props & Moneylines**
*   **The Strikeout (K) Prop:** The most predictive DFS stat. If a pitcher’s line is set at 7.5 and the "Over" is heavily juiced (e.g., -140), it signals a high-ceiling, high-floor DFS play.
*   **Moneyline (Win Probability):** Platforms like FanDuel heavily reward starting pitcher Wins. Targeting heavy favorites (-150 or better) secures this bonus EV.
*   **Outs Recorded & Earned Runs Allowed:** An "Outs Recorded" prop over 18.5 indicates an expectation to pitch deep into the 6th/7th inning (securing Quality Starts). Low "Earned Runs Allowed" (Under 1.5/2.5) props identify high-floor cash game options.

### **B. Hitter Evaluation & "Stacking" via Implied Totals**
*   **Team Implied Totals (IT):** Derived via `(Game Total / 2) + (Spread / 2)`. Target offenses with an implied total of **5.0 or higher**. This is the primary indicator of runs, RBIs, and at-bats for a lineup.
*   **Home Run Odds:** "To Hit a Home Run" props with odds shorter than +300 identify elite GPP (tournament) upside. 
*   **Total Bases (TB):** A line of 1.5 TB with the "Over" juiced signals a high likelihood of multiple hits or extra-base hits.
*   **Line Movement:** If a game total opens at 8.5 and surges to 9.5, or a moneyline moves from -120 to -160, sharp money is dictating game script. Tracking this late movement allows DFS players to pivot to higher-value stacks before lock.

---

## 2. Roadmap: Building a Daily Fantasy Sports (DFS) Capability
To leverage our existing backend infrastructure (Odds API integration, data scraping) into a dedicated DFS Optimizer for DraftKings/FanDuel, we will execute a multi-phase roadmap.

### **Phase 1: Data Ingestion & Transformation (Weeks 1-3)**
*   **Objective:** Ingest player props and DFS salaries.
*   **Tasks:**
    *   Expand `backend/services/odds.py` to pull Player Props (Strikeouts, Total Bases, HRs, Outs Recorded) from The Odds API.
    *   Build a salary scraper for DraftKings and FanDuel CSV exports.
    *   Create a translation layer: Convert American Odds to implied probabilities, and calculate projected Fantasy Points (e.g., `Proj_FP = (Implied_Ks * K_Value) + (Implied_Win% * Win_Bonus)`).

### **Phase 2: The Core DFS Engine (Weeks 4-6)**
*   **Objective:** Develop the algorithmic core for generating optimal lineups.
*   **Tasks:**
    *   **Value Calculation:** Compute `Proj_FP / ($Salary / 1000)` to find the most efficient plays.
    *   **Linear Programming Optimizer:** Implement a solver (e.g., PuLP or SciPy) to maximize `Proj_FP` subject to DFS constraints (salary cap, positional requirements, max players per team).
    *   **Stacking Logic:** Introduce constraint rules to force correlation (e.g., forcing 4-man or 5-man stacks from teams with Implied Totals > 5.0).

### **Phase 3: Advanced Optimization & Variance (Weeks 7-9)**
*   **Objective:** Build tournament-winning capabilities (GPPs).
*   **Tasks:**
    *   **Ownership Projections:** Integrate or model projected ownership to identify leverage plays. 
    *   **Monte Carlo Lineup Generation:** Rather than a single optimal lineup, run 10,000 simulations using the `adjusted_sd` of player props to generate 150 unique, high-ceiling lineups.
    *   **Weather & Lineup Confirmation:** Automate checks for rainouts and confirmed starting lineups 30 minutes before lock.

### **Phase 4: Dashboard & Orchestration (Weeks 10-12)**
*   **Objective:** User interface and agentic automation.
*   **Tasks:**
    *   Build a DFS tab in the existing Streamlit/Dashboard.
    *   Allow manual toggling of exposures (e.g., "Max 30% exposure to Aaron Judge").
    *   Integrate with OpenClaw to automatically generate a DFS brief and optimal lineups 1 hour before the main slate locks.

---

## 3. Strategic Summary
By marrying our current advanced analytics (Bat Speed, PLV, Stuff+) with live Sportsbook Data (Player Props, Implied Totals), we can construct a hybrid projection model. The Sportsbook data provides the highly accurate *baseline expectation*, while our advanced metrics identify the *variance and breakout candidates* that the market hasn't fully priced in. This dual-pronged approach is the key to a profitable DFS operation.
