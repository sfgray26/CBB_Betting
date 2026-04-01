# Gemini UI/UX Re-Audit (Session 4) — 2026-04-01

## 1. G-1 Verification Results (Deployment & Stability)

*   **Deployment:** \ailway up\ successful.
*   **Player Names:** ✅ CLEAN. No " (Quadriceps)" or similar artifacts in player names. Injury notes correctly separated into their own field.
*   **Matchup Stats:** ✅ NO NEGATIVES. "GS" and other counting stats no longer showing -1.
*   **Dashboard Timestamp:** ✅ ET ANCHORED. Showing EDT (-04:00) which is correct for Apr 1.
*   **Async-Optimize Job:** ❌ CRITICAL FAILURE. 
    *   Job status remains in \processing\ indefinitely.
    *   **Root Cause:** SQL Syntax error in \ackend/services/job_queue_service.py\. 
    *   The use of \:result::jsonb\ in the \UPDATE\ statement causes a \psycopg2.errors.SyntaxError: syntax error at or near ":"\.
    *   Subsequent attempts to mark the job as \ailed\ also fail because the transaction is aborted (\InFailedSqlTransaction\).
    *   **Recommendation:** Change \:result::jsonb\ to \CAST(:result AS jsonb)\ in \job_queue_service.py\. (Note: Ops Lead cannot edit .py files).

## 2. Fantasy UI Read-Only Audit (G-3)

### Lineup Page (/fantasy/lineup)
*   **Scanability:** The table is dense. "Implied Runs" and "Park Factor" consume significant horizontal space.
*   **Confusing Columns:** "Score" (Lineup Score) and "Proj" (Valuation Projection) are styled almost identically. Elite users might confuse "Score" (matchup-specific) with "Proj" (raw value).
*   **UX Friction:** The "Apply to Yahoo" button is only at the bottom. For teams with many pitchers/bench players, this requires significant scrolling.
*   **Missing Feature:** No "Lock" mechanism to protect specific players from being moved by the \Optimize Lineup\ tool.
*   **Empty States:** Both Batters and Pitchers tables show "No games scheduled for this date" when empty. It would be better to show which teams ARE playing or if it's a league-wide off-day.

### Roster Page (/fantasy/roster)
*   **Improvement:** Status column in 2nd position is much better for scanability.
*   **Missing Context:** No recent performance indicators (last 7/14 days). Managers have to navigate away to see who's actually hitting.
*   **Abstract Metrics:** "Z-Score" is technically sound but lacks a tooltip or legend for non-statistical users (e.g. what is a 'good' score?).
*   **Missing Stats:** No counting stats (HR, RBI, etc.) on the roster page, only Z-Score.

### Waiver Page (/fantasy/waiver)
*   **Empty Trackers:** Category/2-start pitchers show dead zones when empty. 
*   **Add Button Inconsistency:** "Add" button in \TwoStartTable\ is \disabled\ with a tooltip "Waiver add is visual only", while \WaiverTable\ links to Yahoo.
*   **Need Score Enigma:** No explanation of what a "good" Need Score is or how it's calculated.

### Dashboard (/dashboard)
*   **Actionability:** "Injury Flags" tell the user to "Move to IL immediately" but provide no direct link or button to perform the action.
*   **Integration:** "Lineup Gaps" identify missing positions but don't link to the Waiver Wire page with appropriate filters applied.

## 3. Yahoo API Response Capture (G-2)

*   Captured full roster raw response, free agents, and league settings.
*   Saved to \eports/GEMINI_YAHOO_RESPONSES_2026-04-01.md\.
