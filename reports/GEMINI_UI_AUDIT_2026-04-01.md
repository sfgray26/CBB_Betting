# GEMINI UI AUDIT REPORT - 2026-04-01

## 1. Executive Summary
The ARCH-001 Phase 3 implementation successfully introduced the contract-driven API-Worker pattern to the frontend. However, a critical mismatch was identified in the `asyncOptimizeLineup` API call, and several pages still lack robust loading/error handling.

## 2. High Severity: API Contract Mismatch
*   **File:** `frontend/lib/api.ts`
*   **Issue:** `asyncOptimizeLineup` sends `target_date` and `risk_tolerance` in the JSON body of a POST request. The FastAPI backend endpoint (`/api/fantasy/lineup/async-optimize`) expects these as **query parameters**.
*   **Impact:** The "Optimize Lineup" button will fail with a 422 Unprocessable Entity error (missing field: target_date) in production.
*   **Suggested fix:** Update `api.ts` to append these values to the URL as query strings instead of sending them in the body.

## 3. Medium Severity: Missing Error/Loading Infrastructure
*   **File:** `frontend/app/(dashboard)/fantasy/matchup/page.tsx`
*   **Issue:** Missing `loading.tsx` and `error.tsx` in this directory. 
*   **Impact:** Users will see a blank screen or a full-page crash if the matchup data fails to load or is slow.
*   **Suggested fix:** Create `loading.tsx` (using skeletons) and `error.tsx` for the matchup route.

*   **File:** `frontend/app/(dashboard)/fantasy/lineup/page.tsx`
*   **Issue:** `todayStr()` uses local browser time.
*   **Impact:** Users on the West Coast checking the site between 9 PM and midnight PT will see "today's" date as tomorrow's date relative to the MLB/Yahoo ET-based schedule.
*   **Suggested fix:** Use a library or helper that anchors "today" to US Eastern Time (matching the backend's `_now_et()`).

## 4. Low Severity: Hardcoded Strings & UX Polish
*   **File:** `frontend/app/(dashboard)/fantasy/page.tsx`
*   **Issue:** Hardcoded draft date: "Draft: March 23 @ 7:30am".
*   **Impact:** Displays incorrect, stale information to the user.
*   **Suggested fix:** Move to an environment variable or a config endpoint.

*   **File:** `frontend/app/(dashboard)/fantasy/matchup/page.tsx`
*   **Issue:** `STAT_LABELS` is hardcoded with dozens of entries.
*   **Impact:** High maintenance burden and potential for mismatch with backend stat mapping.
*   **Suggested fix:** Centralize stat labels in `frontend/lib/constants.ts`.

*   **File:** `frontend/app/(dashboard)/fantasy/waiver/page.tsx`
*   **Issue:** "Add" button is permanently disabled with a "visual only" title.
*   **Impact:** Confusing UX; the button should either be hidden or implemented as a deep link to Yahoo.

## 5. Component Structure & Performance
*   **File:** `frontend/app/(dashboard)/fantasy/page.tsx`
*   **Issue:** `DraftBoardTab` uses a text-based loading message instead of the shared `TableSkeleton`.
*   **Impact:** Inconsistent UX across the app.
*   **Suggested fix:** Swap the text loader for `TableSkeleton`.

*   **File:** `frontend/app/(dashboard)/fantasy/waiver/page.tsx`
*   **Issue:** Recommendations loading state uses generic pulse divs.
*   **Impact:** Layout shift when the actual recommendation cards load.
*   **Suggested fix:** Use card-shaped skeletons that match `RecCard`.
