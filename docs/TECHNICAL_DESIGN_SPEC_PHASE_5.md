# Technical Design Specification: Phase 5 - Production-Ready Fantasy Intelligence

**Date:** 2026-05-04
**Lead Architect:** Gemini CLI (DevOps Lead)
**Status:** DRAFT
**Goal:** Transition the fantasy platform from "Late MVP" to "Production-Ready" by unifying the projection system, fixing rate-stat valuation logic, and decoupling monolithic services.

---

## 1. Architectural Strategy: The "Unified Projection" Service

The current system has a hybrid state where `ProjectionAssemblyService` populates a "Single Source of Truth" (`CanonicalProjection`), but the API and `WaiverEdgeDetector` still rely on `player_board.py` for runtime Bayesian fusion.

### Data Contract: `CanonicalProjection` Enhancement
We will promote `CanonicalProjection` (V35) to the sole, mandatory source of truth.

**Required Schema Updates (backend/models.py):**
*   **`proj_h` (Float):** Projected hits (numerator for AVG).
*   **`proj_er` (Float):** Projected earned runs (numerator for ERA).
*   **`proj_bb_allowed` (Float):** Projected walks allowed (part of WHIP numerator).
*   **`proj_h_allowed` (Float):** Projected hits allowed (part of WHIP numerator).

### Source of Truth Logic Flow
1.  **Entry Point:** `UnifiedProjectionService.get_projection(player_id)` replaces `player_board.py` logic.
2.  **Implementation:**
    *   Query `CanonicalProjection` + `CategoryImpact` children.
    *   If missing, trigger `ProjectionAssemblyService` to compute and persist a "JIT" (Just-In-Time) V35 row.
    *   Return a strongly-typed `CanonicalProjection` object.

---

## 2. Data Model Overhaul: Denominator Support

The audit flagged "Denominator Blindness." Category value must shift from Z-Score sums to **True Marginal H2H Impact**.

### Schema Migration Plan
1.  **Harden `CategoryImpact`:** Ensure it stores both `projected_numerator` and `projected_denominator` for ALL categories.
2.  **ID Mapping Lock-In:** 
    *   `IdentityResolutionService` must be the gatekeeper for all ingestion.
    *   `IdentityQuarantine` review becomes a required administrative task before data promotes to production.

---

## 3. Algorithmic Logic: Sabermetric Valuation (Marginal Impact)

### The "Marginal Impact" Algorithm
We will refactor `category_aware_scorer.py` to use a win-probability-based approach.

**Logic Flow:**
1.  **Fetch Baseline:** Get current team totals (e.g., `team_hits`, `team_ab`).
2.  **Calculate Delta:** 
    *   `Old_AVG = team_hits / team_ab`
    *   `New_AVG = (team_hits + player_hits) / (team_ab + player_ab)`
    *   `Impact = (New_AVG - Old_AVG) * sensitivity_factor`
3.  **Normalization:** Map the impact to a -3.0 to +3.0 scale for cross-category compatibility.

### Sabermetric Weighting (The "Gap" Fix)
Promote Statcast flags from "Metadata" to "Confidence Multipliers" in `fusion_engine.py`:
*   **xERA vs ERA:** If `|xERA - ERA| > 0.75`, the posterior projection weight shifts an additional 15% toward the observed Statcast metric.

---

## 4. Refactoring Roadmap: Service Decoupling

### Module Breakdown
*   **`projection_service.py`:** Absorbs `CanonicalProjection` retrieval and JIT assembly.
*   **`valuation_service.py`:** Absorbs marginal impact math and `CategoryAwareScorer`.
*   **`identity_service.py`:** Canonical location for all ID translation.

### Dependency Flow
`Request` -> `Router` -> `ValuationService` -> `ProjectionService` -> `IdentityService`.

---

## 5. Implementation Phasing

### Sprint 1: Foundation & Schema (Current)
*   [ ] Add `proj_h`, `proj_er` etc. to `CanonicalProjection`.
*   [ ] Refactor `ProjectionAssemblyService` to populate these new columns.
*   [ ] Implement `IdentityResolutionService.resolve()` as the single entry point.

### Sprint 2: Logic Engine (Next)
*   [ ] Implement "Marginal Impact" math in `category_aware_scorer.py`.
*   [ ] Wire `WaiverEdgeDetector` to pass numerators/denominators from V35 table to the scorer.
*   [ ] Activate Sabermetric confidence weighting in `fusion_engine.py`.

### Sprint 3: Interface & Cleanup
*   [ ] Deprecate `player_board.py` (Move remaining display logic to a simple view).
*   [ ] Normalize FastAPI response schemas to use the `CanonicalProjection` contract.
*   [ ] Implement the `DailySnapshot` regression watchdog.
