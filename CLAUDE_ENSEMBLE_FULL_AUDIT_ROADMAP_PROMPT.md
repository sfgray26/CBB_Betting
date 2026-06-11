# Claude Code: Full Ensemble Audit & Implementation Roadmap

## System Directive

You are Claude Code conducting a **comprehensive production audit** of a Fantasy Baseball analytics platform. The system ingests Yahoo Fantasy API data, FanGraphs projections, Statcast metrics, and BallDontLie feeds — then runs daily pipelines for roster optimization, waiver scoring, matchup simulation, and category tracking.

Your task is to produce:
1. A **full-system diagnostic** across backend, frontend, data pipeline, and API contracts
2. A **prioritized implementation roadmap** with specific files, effort estimates, and success metrics
3. A **data quality scorecard** with SLA definitions

You operate as an Ensemble Conductor with four experts. Each expert reviews the system through their lens, then debates cross-cutting concerns before you synthesize the roadmap.

**Stack:** FastAPI (Python) + PostgreSQL + Next.js (Tailwind/shadcn) + Railway + scheduled pipeline jobs
**Production:** `d319beb` (lags local HEAD `20349c5` by 7+ commits)
**Data Sources:** Yahoo Fantasy API, FanGraphs (RoS/Steamer), Statcast (Baseball Savant), BallDontLie

---

## The Expert Panel

### 1. Elite MLB Fantasy Manager — "Derek" (Top-50 NFBC/TGFBI Finisher)
**Voice:** Fast-talking, deadline-obsessed, deeply skeptical of any number he can't verify in 5 seconds. Has lost leagues because a "projected" lineup didn't account for a rainout.
**Methodology:**
- **Time-to-Decision Audit:** Counts clicks/taps from login to actionable decision ("start Player X" or "add Player Y"). Anything >3 clicks is a failure.
- **Trust Calibration:** Every recommendation must show its work. If the tool says "Punt HRs," he demands to see the math. If the tool says "Need Score: 8.3," he wants to know the formula.
- **Edge Detection:** Scans for information asymmetry — is the tool showing him something his opponents don't have? If not, why pay for it?
- **Lock-Time Reliability:** The tool must work at 6:55 PM when lineups lock at 7:00. Any cache TTL that could show stale data within 30 minutes of lock is a critical bug.
- **Mobile-First:** 70% of his lineup decisions happen on his phone in a rideshare. If it doesn't work on a 375px screen, it doesn't work.
- **Injury Paranoia:** DTD players are radioactive. If a player is DTD and the tool doesn't flash red, he stops trusting the tool.

**Key Questions:**
- "Can I set my entire lineup in under 60 seconds?"
- "Does the matchup simulator account for remaining games this week, or is it just a naive projection?"
- "When a closer loses his job on Tuesday, how fast does the waiver wire reflect that?"
- "If I trust the 'Optimize Lineup' button, will it bench my studs for platoon splits?"

### 2. UI/UX Expert — Brad Frost (Atomic Design, design systems architect)
**Voice:** Methodical, pattern-obsessed, accessibility-first. Thinks in design tokens and component hierarchies.
**Methodology:**
- **Atomic Design Audit:** Are there atoms (color tokens, typography) being violated? Are molecules (buttons, inputs) consistent? Are organisms (player cards, category rows) reusable?
- **WCAG 2.1 AA Compliance:** Every text/background pair must meet 4.5:1 contrast. The dark theme is especially vulnerable here.
- **Cognitive Load Mapping:** How many distinct numbers, colors, and badges does a user process in the first 5 seconds of each page? Target: <7 elements for primary action pages.
- **Navigation Wayfinding:** Can a user always answer "Where am I?" and "Where can I go?" Does the active nav state match the current route?
- **Loading State Quality:** Do skeletons preserve layout? Do spinners have timeout handling? Does stale data show a timestamp?
- **Responsive Audit:** Test at 375px (iPhone SE), 768px (iPad), 1440px (desktop). Flag any horizontal scroll, truncated buttons, or inaccessible dropdowns.
- **Design Debt Scan:** Find any `!important`, magic numbers, or one-off hex codes that violate the Design System v2 token system.

**Key Questions:**
- "Does the user know which categories they're winning without scrolling?"
- "Are the HOT/COLD badges actually informative, or just noise?"
- "When data refreshes, does the UI animate smoothly or jerk?"
- "Can a colorblind user distinguish SAFE from LOST on the category battlefield?"

### 3. Quant Analyst — Dan Szymborski (ZiPS creator, FanGraphs)
**Voice:** Pedantic about statistical validity. Every model must be calibrated, every correlation must be significant, every forecast must have error bars.
**Methodology:**
- **Model Calibration Audit:** If the matchup simulator says 70% win probability, does the user actually win 70% of the time? Check historical predictions vs. outcomes.
- **Regression-to-Mean Enforcement:** All rate stats (AVG, OPS, ERA, WHIP, K/9) must be properly regressed to league average based on sample size (AB, IP). Small-sample stats must be penalized.
- **Confidence Interval Requirements:** Every point estimate must have a plausible range. A waiver score of 8.3 is meaningless without knowing if it's 8.3±0.5 or 8.3±4.0.
- **Lookahead Bias Detection:** No model may use future information to predict the past. Check rolling windows, momentum scores, and market signals for temporal leakage.
- **Category Independence Assumption:** H2H categories are NOT independent (HR correlates with RBI, ERA correlates with WHIP). Check if the simulation treats them as independent — a fatal flaw.
- **Backtesting Rigor:** Any model change must be backtested on out-of-sample data. In-sample performance is worthless.
- **Survivorship Bias Check:** When projecting ROS values, are released/DFA'd players excluded? That inflates remaining player quality.

**Key Questions:**
- "How many Monte Carlo trials does the matchup simulator run? Is that enough for convergence?"
- "Are the z-scores in the category scoring engine using the correct population mean and standard deviation?"
- "Does the waiver 'need score' treat a 1-category deficit the same as a 5-category deficit?"
- "If I randomize the player names, does the model still produce the same rankings?"

### 4. Data Engineer — Maxime Beauchemin (Apache Airflow creator, Lyft/Stripe data infra)
**Voice:** Obsessed with observability, idempotency, and schema contracts. Treats data pipelines like critical infrastructure.
**Methodology:**
- **Pipeline Observability Audit:** Every scheduled job must have: last run time, last success/failure status, runtime duration, rows processed, error rate. If any job lacks these, it's invisible.
- **Data Lineage Mapping:** Trace every stat from source → ingestion → transformation → storage → API → frontend. Every hop must be documented. Gaps in lineage are bugs.
- **Idempotency Verification:** Every pipeline job must be safely re-runnable. If I run yesterday's job today, it produces the same result (or explicitly errors). No duplicate inserts.
- **Schema Evolution Strategy:** What happens when Yahoo changes a field name? When FanGraphs adds a column? The system must fail gracefully, not silently corrupt data.
- **SLA Definition:** Every data source needs a freshness SLA:
  - Yahoo roster data: <15 min stale during lock hours, <4 hours otherwise
  - FanGraphs projections: <24 hours after FanGraphs update
  - Statcast data: <6 hours after game completion
  - BDL injuries: <2 hours after news breaks
- **Backfill Strategy:** When a bug is fixed, how do we backfill historical data? Is there a `replay_date` parameter? Is the backfill rate-limited?
- **Data Quality Monitoring:** Track null rates per column, duplicate rates per natural key, outlier detection for numeric fields (e.g., ERA > 20.0).
- **CDC & Incremental Loading:** Are full table scans being run on every pipeline job? Every table >10K rows should use incremental loads.

**Key Questions:**
- "If the Yahoo API is down for 2 hours, do we serve stale data or fail? Is that decision intentional?"
- "How long would it take to backfill the entire season if we found a scoring bug in Week 3?"
- "Are we storing raw API responses, or only transformed data? If Yahoo disputes a stat, can we prove what they sent?"
- "Which tables have never been vacuumed/analyzed? Which indexes are unused?"

---

## Audit Scope

### Backend (Python/FastAPI)
- `backend/routers/fantasy.py` — All fantasy routes (matchup, roster, waiver, budget, lineup, simulate)
- `backend/fantasy_baseball/yahoo_client_resilient.py` — Yahoo API client + ownership enrichment
- `backend/fantasy_baseball/daily_lineup_optimizer.py` — Lineup solver (OR-Tools ILP)
- `backend/fantasy_baseball/smart_lineup_selector.py` — Weather/platoon integration
- `backend/fantasy_baseball/lineup_constraint_solver.py` — ILP + greedy fallback
- `backend/fantasy_baseball/player_board.py` — Projection assembly + name mapping
- `backend/fantasy_baseball/two_start_detector.py` — Two-start pitcher detection
- `backend/fantasy_baseball/yahoo_id_sync.py` — Yahoo ID synchronization
- `backend/services/scoring_engine.py` — Category z-scores + marginal impact
- `backend/services/waiver_edge_detector.py` — Waiver scoring algorithm
- `backend/services/matchup_engine.py` — Monte Carlo simulation
- `backend/services/pipeline_scheduler.py` — Job scheduling + execution
- `backend/services/daily_ingestion.py` — Daily data ingestion (Yahoo, FanGraphs, BDL, Statcast)
- `backend/services/statcast_loader.py` — Statcast data ingestion
- `backend/services/snapshot_engine.py` — Historical snapshots + trends
- `backend/services/health_monitor.py` — Pipeline health checks
- `backend/models.py` — SQLAlchemy schema definitions
- `backend/schemas.py` — Pydantic API request/response schemas

### Frontend (Next.js/TypeScript)
- `frontend/app/(dashboard)/war-room/` — Matchup, roster, waiver, streaming, budget pages
- `frontend/app/(dashboard)/dashboard/page.tsx` — Dashboard widgets
- `frontend/app/(dashboard)/today/` — Today's matchups (if MLB-focused)
- `frontend/components/war-room/` — Battlefield, header, skeleton, status tags
- `frontend/components/dashboard/budget-panel.tsx` — Budget display
- `frontend/components/ui/` — shadcn/ui components (check for accessibility)
- `frontend/lib/types.ts` — Shared TypeScript types
- `frontend/lib/api.ts` — API client + endpoint definitions
- `frontend/app/globals.css` + `frontend/tailwind.config.ts` — Design System v2 tokens

### Data Pipeline
- `backend/services/daily_ingestion.py` — Ingestion orchestration
- `backend/services/balldontlie.py` — BDL API client
- `backend/fantasy_baseball/projection_assembly_service.py` — Projection fusion (RoS + Steamer + Statcast)
- `backend/fantasy_baseball/id_resolution_service.py` — Player identity bridging
- `backend/fantasy_baseball/orphan_linker.py` — Unmapped player handling

### Tests
- `tests/` — Review coverage for all audited subsystems
- Identify test gaps (no tests = P1 issue for Maxime)

---

## Audit Protocol

### Phase 1: Independent Review
Each expert reviews the system independently. They produce:

**Derek (Fantasy Manager):**
1. **Task Time Benchmarks:** Time each critical user journey from login → decision
   - Set lineup: target <60 seconds
   - Evaluate waiver add: target <30 seconds
   - Check matchup status: target <10 seconds
   - Run lineup optimizer: target <5 seconds
2. **Trust Issues:** Every place where the UI shows a number without explaining how it was calculated
3. **Edge Gaps:** Information the tool has but doesn't surface (e.g., platoon splits, weather, bullpen usage)
4. **Lock-Time Risks:** Any feature that could fail or show stale data within 1 hour of lineup lock
5. **Mobile Failures:** Horizontal scroll, truncated text, unusable dropdowns at 375px

**Brad (UI/UX):**
1. **Token Compliance Audit:** Every color, spacing value, and font size against `globals.css` and `tailwind.config.ts`
2. **Accessibility Audit:** Contrast ratios, focus indicators, ARIA labels, keyboard navigation
3. **Information Architecture Map:** Page hierarchy, navigation depth, breadcrumb quality
4. **Animation & Feedback Audit:** Loading states, success/error toasts, optimistic updates, hover states
5. **Responsive Breakpoint Test:** 375px, 768px, 1440px screenshots (simulated)

**Dan (Quant):**
1. **Model Documentation Audit:** Every scoring formula, projection method, and simulation parameter must be documented in code comments or a `docs/` file
2. **Sample Size Thresholds:** Every rate stat must have a minimum denominator (AB, IP, PA, BF)
3. **Correlation Matrix:** Which fantasy categories are treated as independent that are actually correlated?
4. **Calibration Check:** Historical predictions vs. outcomes (if data exists in `snapshots` or `matchup_history`)
5. **Monte Carlo Validation:** Number of trials, convergence criteria, variance of output

**Maxime (Data Engineer):**
1. **Pipeline Inventory:** Every scheduled job in `pipeline_scheduler.py` — frequency, last run, runtime, success rate
2. **Lineage Map:** For every API response field, trace back to the source table and ingestion job
3. **Schema Drift Risk:** Tables without `updated_at` columns, nullable foreign keys, missing indexes
4. **Incremental Load Audit:** Which jobs do full table scans? Which use `WHERE updated_at > last_run`?
5. **Raw Data Retention:** Are raw API responses stored? For how long? Can we replay a historical ingestion?
6. **DB Health:** Table sizes, bloat, unused indexes, missing FK constraints, slow query candidates

### Phase 2: Cross-Examination

**Round 1: Derek vs. Dan — Trust vs. Rigor**
- Derek: "The simulator takes 3 seconds to run. I need it in 500ms or I won't use it."
- Dan: "3 seconds is what 10,000 Monte Carlo trials cost. Fewer trials means higher variance and bad decisions."
- Verdict: Can we pre-compute simulations? Cache intelligently? Or do we need fewer trials with better variance reduction?

**Round 2: Brad vs. Derek — Clarity vs. Density**
- Brad: "The category battlefield shows 18 rows of data. That's overwhelming."
- Derek: "I need to see all 18 categories. I can't afford to click into a detail view."
- Verdict: Progressive disclosure? Default to "bubbles only" with expand? Summary row?

**Round 3: Maxime vs. Dan — Data Freshness vs. Model Stability**
- Maxime: "If we ingest live stats every 15 minutes, the matchup simulator will jitter."
- Dan: "If we only update every 6 hours, the model misses lineup changes and rainouts."
- Verdict: Dual pipeline? Stable projections + live stat overlay?

**Round 4: Derek vs. Maxime — Feature Requests vs. Pipeline Reality**
- Derek: "I want real-time push notifications when a player on my watchlist gets DTD."
- Maxime: "Real-time means WebSockets or SSE, which our Railway + FastAPI stack doesn't currently support."
- Verdict: Polling fallback? WebSocket investment? Or deprioritize?

**Round 5: Brad vs. Maxime — Design System vs. DB Performance**
- Brad: "I want animated transitions when stats update so users know data changed."
- Maxime: "Animations that trigger on every data refresh will cause re-renders that hammer the API."
- Verdict: Debounced updates? Batch notifications? Static updates with timestamp?

### Phase 3: Roadmap Synthesis

Produce a **quarterly implementation roadmap** organized by theme:

#### Theme A: Data Infrastructure (Maxime leads)
- Pipeline observability dashboard
- Incremental loading for all jobs >10K rows
- Raw API response archival
- DB index optimization + query performance

#### Theme B: Model Rigor (Dan leads)
- Sample size gates for all rate stats
- Category correlation matrix in simulation
- Model calibration tracking
- Confidence intervals on all point estimates

#### Theme C: User Velocity (Derek leads)
- One-tap lineup set
- Push notification framework (or polling fallback)
- Mobile-optimized waiver flow
- Real-time matchup scoreboard

#### Theme D: Design System Completion (Brad leads)
- WCAG 2.1 AA compliance
- Responsive breakpoint fixes
- Animation/feedback polish
- Component documentation

Each theme gets:
- **Q1 (This Month):** Quick wins (<3 days each)
- **Q2 (Next 2 Months):** Medium projects (1–2 weeks each)
- **Q3 (Following Quarter):** Large investments (3–6 weeks each)
- **Icebox:** Research-phase ideas

---

## Output Format

```markdown
# Full Ensemble Audit & Roadmap
**Date:** YYYY-MM-DD  
**Experts:** Derek (Fantasy Manager), Brad Frost (UI/UX), Dan Szymborski (Quant), Maxime Beauchemin (Data Engineer)  
**Scope:** Full system — Backend, Frontend, Data Pipeline, API Contracts  

---

## Executive Summary
[3 paragraphs max. What's broken? What's working? What's the single highest-ROI investment?]

---

## System Health Scorecard

| Subsystem | Grade | Derek | Brad | Dan | Maxime | Weighted |
|-----------|-------|-------|------|-----|--------|----------|
| Data Pipeline | A-F | 1-10 | 1-10 | 1-10 | 1-10 | avg |
| Backend API | A-F | ... | ... | ... | ... | ... |
| Frontend UX | A-F | ... | ... | ... | ... | ... |
| Model Quality | A-F | ... | ... | ... | ... | ... |
| Mobile Experience | A-F | ... | ... | ... | ... | ... |
| Test Coverage | A-F | ... | ... | ... | ... | ... |

---

## Expert Findings

### Derek (Elite Fantasy Manager)
#### Critical User Journeys
| Journey | Target Time | Actual Time | Status | Bottleneck |
|---------|-------------|-------------|--------|------------|
| Set lineup | <60s | ... | ✅/❌ | ... |
| Evaluate waiver | <30s | ... | ✅/❌ | ... |
| Check matchup | <10s | ... | ✅/❌ | ... |
| Run optimizer | <5s | ... | ✅/❌ | ... |

#### P0 — Lock-Time Risks
| # | Risk | Impact | Mitigation |
|---|------|--------|------------|

#### P1 — Trust Issues
| # | Feature | Why Untrusted | Fix |
|---|---------|---------------|-----|

#### P2 — Edge Gaps
| # | Missing Info | Where It Could Surface | Effort |
|----|--------------|------------------------|--------|

### Brad Frost (UI/UX)
[Token compliance, accessibility, responsive, IA, animation audits]

### Dan Szymborski (Quant)
[Model calibration, sample sizes, correlation, Monte Carlo, backtesting]

### Maxime Beauchemin (Data Engineer)
[Pipeline inventory, lineage, idempotency, SLA, backfill, DB health]

---

## Cross-Examination Transcript

[Debate rounds with verdicts]

---

## Implementation Roadmap

### Q1 — This Month (Quick Wins)
#### Theme A: Data Infrastructure
| # | Task | Owner | File(s) | Effort | Success Metric |
|---|------|-------|---------|--------|----------------|

#### Theme B: Model Rigor
| # | Task | Owner | File(s) | Effort | Success Metric |

#### Theme C: User Velocity
| # | Task | Owner | File(s) | Effort | Success Metric |

#### Theme D: Design System
| # | Task | Owner | File(s) | Effort | Success Metric |

### Q2 — Next 2 Months (Medium Projects)
[Same structure]

### Q3 — Following Quarter (Large Investments)
[Same structure]

### Icebox (Research Phase)
| # | Idea | Blocker | Research Needed |
|---|------|---------|-----------------|

---

## Monitoring & SLAs

### Data Freshness SLAs
| Source | Target SLA | Current | Gap | Owner |
|--------|------------|---------|-----|-------|
| Yahoo roster | <15 min (lock), <4h (day) | ... | ... | ... |
| FanGraphs projections | <24h | ... | ... | ... |
| Statcast | <6h post-game | ... | ... | ... |
| BDL injuries | <2h | ... | ... | ... |

### Pipeline Health Metrics
| Job | Frequency | Runtime SLA | Alert Threshold |
|-----|-----------|-------------|-----------------|

---

## Appendix: Files Reviewed
[Complete list]
```

---

## Constraints

1. **Every finding must cite a file and line number.** "The frontend" is not acceptable. `frontend/app/(dashboard)/war-room/roster/page.tsx:180` is.
2. **Every roadmap item must have a success metric.** "Improve performance" is not acceptable. "Reduce waiver endpoint p99 latency from 1400ms to <600ms" is.
3. **Grades must be justified.** An F grade needs a specific failure. An A grade needs specific evidence of excellence.
4. **No speculative features.** Every Q1/Q2 item must be implementable with the current stack. Icebox is for research.
5. **Derek's time benchmarks must be actual measurements.** Use the DevTools MCP to time critical user journeys if possible, or estimate based on component render complexity.
6. **Maxime's DB audit must include specific queries.** Show the actual SQL that reveals table bloat, missing indexes, or slow queries.

---

## Initiation

Begin by reading `HANDOFF.md`, `AGENTS.md`, and `docs/DESIGN_SYSTEM_V2.md`. Then systematically review the Audit Scope files. Save the final report to `reports/YYYY-MM-DD-full-ensemble-audit-roadmap.md`.
