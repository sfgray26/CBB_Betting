# Architect Review Queue

Open decisions requiring architectural judgment (not implementation). Extracted from HANDOFF.md to reduce operational noise.

---

## Code / Scope Decisions

### Decision #1: NSB Composite Math

**Context:** Pre-existing test failure `test_composite_z_excludes_z_sb_when_both_populated`.

**Question:** Should composite_z aggregation exclude `z_sb` when both SB and CS are populated?

**Options:**
1. Fix composite_z to exclude z_sb when both SB/CS present
2. Update test expectation to accept current behavior  
3. Remove NSB from composite calculation entirely

**Recommendation:** TBD

**Status:** Open

**Impact:** Phase 2 completion claim

---

### Decision #2: Unknown Yahoo stat_ids

**Context:** Waiver output silently drops unknown stat_ids (e.g., "38"). No visibility when Yahoo API introduces new stat types.

**Question:** Should we log warnings when unknown stat_ids are encountered?

**Options:**
1. Add INFO-level log when stat_id not in YAHOO_ID_INDEX
2. Add WARN-level log (more visible)
3. Keep silent (current behavior)
4. Surface unknown IDs in /admin endpoint for manual review

**Recommendation:** Option 4 (admin visibility)

**Status:** Open

**Impact:** Data enrichment coverage visibility; YAHOO_ID_INDEX maintenance

---

### Decision #3: Schedule Fallback Mode Flag

**Context:** Probable-pitcher fallback generates synthetic implied_runs indistinguishable from sportsbook-derived values.

**Question:** Should API expose "schedule fallback mode" flag?

**Options:**
1. Add `game_context.source` field ("sportsbook" vs "fallback")
2. Add separate `has_sportsbook_odds` boolean
3. Keep invisible (current)

**Recommendation:** Option 1 (source field)

**Status:** Open

**Impact:** UI transparency — users can distinguish real odds from neutral estimates

---

### Decision #4: Player-ID-Mapping Job Model

**Context:** `/admin/backfill/player-id-mapping` is synchronous long-running endpoint. Postman/UAT depends on request timeouts. Reported "no response" in prod (not reproduced locally).

**Question:** Should this remain synchronous or move to job queue?

**Options:**
1. Keep synchronous (current)
2. Move to existing job queue (async with status polling)
3. Hybrid: quick jobs sync, slow jobs queued

**Recommendation:** Option 2 (job queue)

**Status:** Open

**Impact:** UAT reliability; admin endpoint predictability

---

### Decision #5: Projection Extrapolation Caps

**Context:** Decisions API surfaces impossible ROS figures (0.00 ERA, 91.2 HR, 204.4 RBI). Rate extrapolation is uncapped.

**Question:** What are plausible ROS caps per category?

**Options:**
1. Per-category caps based on historical extremes (e.g., ERA >= 0.50, HR <= 73, RBI <= 191)
2. Per-player caps based on career high × 1.2
3. No caps (current)

**Recommendation:** Option 1 (historical extremes)

**Status:** Open; requires Claude implementation once policy decided

**Impact:** Data quality; UI credibility

---

### Decision #6: Proxy Player Pipeline

**Context:** 6 of 23 roster players (Ballesteros, Kim, Murakami, Smith, Arrighetti, De Los Santos) carry `is_proxy: true` with hardcoded `z=-0.8`, empty `cat_scores`, genuinely absent from Steamer/ZiPS.

**Question:** How should proxy players be scored?

**Options:**
1. Synthesize proxy projections from rolling stats
2. Route through Yahoo season stats
3. Accept placeholder `z=-0.8` for non-top-tier assets (current)
4. Build separate proxy projection model

**Recommendation:** Option 3 for now (acceptable for non-core assets); revisit if proxies expand to starting lineup

**Status:** Open

**Impact:** Roster/waiver/decision scoring completeness

---

### Decision #7: Statcast x-stats Integration

**Context:** Phase 4.5a Priority 4 — deferred. `statcast_performances` table populated but only consumed by `data_reliability_engine.py`. xwOBA, barrel%, exit_velocity not wired into player scoring.

**Question:** Should x-stats feed into scoring_engine or decision_engine?

**Options:**

3. Both (full luck-adjusted scoring)


**Recommendation:** Option 2 (decision_engine only) — avoid disrupting established Z-score baselines

**Status:** Open; requires data quality review before implementation

**Impact:** Waiver edge detection; move recommendation confidence

---

## UI Contract Open Questions

From `reports/2026-04-17-ui-specification-contract-audit.md`.

### Q1: Yahoo API Rate Limits

**Question:** What are the rate limits for scoreboard/transactions/roster calls?

**Status:** Open

**Impact:** Caching strategy; determines on-demand vs pre-compute approach

---

### Q2-Q3: Greenfield Category Availability

**Question:** Are W, L, SV, HLD, QS available in Yahoo player season stats?

**Status:** Open

**Impact:** Rolling stats source for 4 greenfield categories; determines whether Yahoo API expansion feasible

---

### Q4: FAAB vs Priority Waivers

**Question:** Does the league use FAAB or priority-based waivers?

**Status:** Open

**Impact:** `ConstraintBudget` contract; UI waiver page design

---

### Q5: Opponent ROW Projections

**Question:** For opponent ROW projections, should we model per-player or pace-based?

**Status:** Open

**Impact:** P2-5 scope; simulation complexity

---

### Q7: Matchup Week Boundary

**Question:** What defines the matchup week boundary?

**Status:** Open

**Impact:** Acquisition counting; games-remaining windows

---

### Q8: Scoreboard Response Time

**Question:** What is acceptable scoreboard response time?

**Status:** Open

**Impact:** Determines on-demand vs pre-compute strategy

---

### Q9: Trade Context Handling

**Question:** How should canonical player row handle trade context (same player as sending / receiving)?

**Status:** Open

**Impact:** Trade page design; roster continuity display

---

## Resolved Questions

- **Q6:** Timezone confirmed as America/New_York
- **Q10:** HLD is supporting, not scoring category
- **Q11:** K_B is `lower_is_better=False` (confirmed)

---

*Last updated: 2026-04-21 — Extracted from HANDOFF.md as part of Wave 1 documentation restructure.*
