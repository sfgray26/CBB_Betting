# Stuff+/Location+ Data Acquisition Alternatives Research

**Date:** 2026-05-19
**Subject:** Alternatives to blocked FanGraphs automated scraping
**Focus:** Stuff+/Location+ (FanGraphs) vs. Savant Pitch Quality (In-house)

## Executive Summary
The automated ingestion of Stuff+ and Location+ data from FanGraphs is currently blocked by Cloudflare on Railway IP ranges. Since FanGraphs does not offer an official developer API, the platform must transition to either a manual snapshot workflow or activate the in-house Savant Pitch Quality proxy.

## 1. Manual CSV Snapshot Workflow
FanGraphs allows members ($15/month or $80/year) to perform "One-Click Data Exports" of their leaderboards.

### Proposed Workflow
1.  **Subscription:** One project maintainer maintains a FanGraphs Membership.
2.  **Export:** Weekly (or daily during peak season), the maintainer downloads the Pitching Leaderboard containing `Stuff+`, `Location+`, and `Pitching+` columns.
3.  **Deployment:** The CSV is renamed to `stuff_plus_2026.csv` and placed in `data/projections/`.
4.  **Ingestion:** A new loader in `backend/fantasy_baseball/projections_loader.py` will detect this file and upsert values into the database.

### Pros/Cons
*   **Pros:** Highest data quality; uses the industry-standard Stuff+ model.
*   **Cons:** Requires manual intervention; $80/year cost; data staleness between snapshots.

## 2. FanGraphs API Subscription Feasibility
**Feasibility: Low / None**
Web research confirms that FanGraphs **does not offer a public API**. Their data is licensed from third parties (SIS, MLB) under terms that prohibit sub-licensing via API. Automated scraping is explicitly forbidden in their Terms of Service and enforced via Cloudflare's bot protection.

## 3. Savant Pitch Quality Proxy Activation Plan
The platform already has an in-house "Savant Pitch Quality" metric implemented in `backend/fantasy_baseball/savant_pitch_quality.py`. This metric uses raw Statcast data (velocity, spin, movement, whiffs, xwOBA) to calculate a 100-centered score.

### Current Status
*   **Logic:** Feature-complete.
*   **Coverage:** 554 pitchers scored.
*   **Blocker:** Average confidence is currently `0.191`, which is below the target threshold of `0.3` (roughly 12-15 innings of data required for stability).

### Activation Plan
1.  **Refresh Data:** Run the backfill script on Railway to capture latest May performance:
    ```powershell
    railway run python scripts/backfill_savant_pitch_quality.py
    ```
2.  **Confidence Check:** Query the database to verify stability:
    ```sql
    SELECT AVG(sample_confidence) FROM savant_pitch_quality_scores WHERE season = 2026;
    ```
3.  **Enable Flags:** If confidence >= 0.3, enable the feature flags in the `feature_flags` table:
    *   `savant_pitch_quality_enabled`: Enables the 100-centered score in the UI.
    *   `savant_pitch_quality_waiver_signals_enabled`: Enables `BREAKOUT_ARM` and `STREAMER_UPSIDE` badges on the waiver surface.
4.  **Verification:** Confirm that Garrett Crochet and Paul Skenes (high Stuff+ proxies) surface at the top of the Savant rankings.

## Recommendation
1.  **Immediate:** Proceed with the **Savant Pitch Quality Activation Plan** as the primary automated solution. The confidence threshold is expected to be met by late May.
2.  **Secondary:** Implement the **Manual CSV Snapshot** loader as a high-fidelity fallback for "Expert Mode" analysis.
