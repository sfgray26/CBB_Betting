# GEMINI UI AUDIT REPORT - 2026-04-01

## 1. Executive Summary
The frontend is functional but has not yet been updated to utilize the **ARCH-001 API-Worker pattern** deployed in the backend. Several components lack robust loading skeletons and error boundaries, and critical configuration data (like draft dates and stat labels) is hardcoded.

## 2. ARCH-001 Integration Gaps
*   **`lineup/page.tsx`**: The "Optimize Lineup" button currently triggers a simple `refetch()` on the daily lineup endpoint. It **MUST** be updated to:
    1. Call `POST /api/fantasy/lineup/async-optimize`.
    2. Enter a polling state using `GET /api/fantasy/jobs/{job_id}`.
    3. Show a progress indicator while the job is `queued` or `running`.
*   **`waiver/page.tsx`**: The "Load Recommendations" action is also a candidate for the job queue, as it performs complex H2H deficit analysis.

## 3. Loading States & UX
*   **`matchup/page.tsx`**: Lacks a `loading.tsx` file for App Router streaming. The inline skeleton is too basic and doesn't match the final table layout.
*   **`fantasy/page.tsx`**: `DraftBoardTab` displays "Loading draft board..." text. This should be replaced with a `TableSkeleton` similar to the lineup and waiver pages.
*   **`waiver/page.tsx`**: The recommendations section uses a simple pulse div. It should use card-shaped skeletons to prevent layout shift.

## 4. Error Handling
*   **`matchup/page.tsx`**: Missing a directory-level `error.tsx` file. Unhandled runtime errors in this route will bubble up to the root, providing a poor user experience.
*   **Inconsistent Error Messaging**: Many error states use generic "Failed to load" messages. They should provide more specific guidance (e.g., "Yahoo Token Expired - Please Re-authenticate").

## 5. Hardcoded Strings & Technical Debt
*   **Draft Schedule**: `fantasy/page.tsx` has "Draft: March 23 @ 7:30am" hardcoded.
*   **Stat Mapping**: `STAT_LABELS` in `matchup/page.tsx` is hardcoded. This map should ideally be centralized in `@/lib/constants.ts` or fetched from a backend config endpoint to ensure consistency with the backend's `_YAHOO_STAT_FALLBACK`.
*   **Date Handling**: `lineup/page.tsx` uses `new Date()` for local "today". This can lead to "off-by-one" date issues for West Coast users during the 12am-3am ET window.

## 6. Recommended Next Steps
1.  **Phase 1.1 (Frontend Fixes)**: Update `lineup/page.tsx` to use the `async-optimize` endpoint and implement the job polling UI.
2.  **Phase 1.2 (Hardening)**: Add missing `loading.tsx` and `error.tsx` to the `matchup/` route.
3.  **Phase 1.3 (Refactoring)**: Centralize `STAT_LABELS` and move hardcoded draft dates to environment variables or feature flags.
