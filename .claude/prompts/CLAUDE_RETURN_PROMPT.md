> **Note:** This is a copy. The canonical version is in the repository root: `CLAUDE_RETURN_PROMPT.md`

---

# CLAUDE CODE — LEAD ARCHITECT & MASTER ENGINEER
## Return Briefing: Full Context + Immediate Directives

> **Status:** Standard return briefing (March 27, 2026).
> 
> **⚠️ EMERGENCY OVERRIDE:** If responding to production crisis (March 28+), use 
> `CLAUDE_ARCHITECT_PROMPT_MARCH28.md` instead — it contains active P0 incident response.
>
> **Mission:** Transform our fantasy baseball platform from "functional" to **institutionally elite** — quant-trading grade tooling for serious managers.  
> **Context Load:** This single document contains everything required to resume work with zero friction.

---

## 1. PROJECT STATE SNAPSHOT (As of March 27, 2026)

### What Just Happened (While You Were Away)
- **Kimi CLI** executed critical application fixes:
  - Fixed Yahoo roster deduplication (Page 4 state duplication bug)
  - Fixed matchup parser for deeply nested Yahoo responses (Page 5 "0 My Team" bug)
  - Added defensive serializers to prevent NaN/frontend crashes
  - Fixed timezone handling for West Coast games
  - Fixed coroutine warnings in OpenClaw autonomous loop
  - Fixed v10 migration transaction block errors
  - Updated Dockerfile to run migrations on boot

- **Current Status:**
  - ✅ Migrations v9/v10: **STABLE** (running in Railway with autocommit)
  - ✅ Build pipeline: **GREEN** (0 flake8 F-errors, TS builds)
  - ⚠️ Fantasy tools: **FUNCTIONAL BUT NOT ELITE**

---

## 2. STRATEGIC MANDATE: "Operation Elite Diamond"

### The Vision
We are building the **Bloomberg Terminal for Fantasy Baseball Managers**:
- Real-time data ingestion (Statcast, odds, weather, injuries)
- Bayesian projection updating (not static CSVs)
- Multi-agent integrity validation (OpenClaw pattern detection)
- Kelly-optimal lineup decisions (risk-adjusted)
- MCMC matchup simulation (10k sims for weekly probs)
- Institutional-grade UI (dashboard, alerts, briefings)

### Phase Focus (Next 7 Days — URGENT: Season Already Started)
| Phase | Objective | Owner | Deadline |
|-------|-----------|-------|----------|
| **B1** | Elite Dashboard v1.0 (live data, not mocks) | You (Arch) | 48 hours |
| **B2** | Smart Lineup Optimizer (weather + platoon integration) | You (Arch) | 72 hours |
| **B3** | Waiver Intelligence Engine (edge detection) | Delegate to Kimi | 48 hours |
| **B4** | Daily Briefing System (automated alerts) | Delegate to Gemini | 72 hours |
| **B5** | MCMC Matchup Simulator (weekly probs) | Research with Kimi | 7 days |

---

## 3. CRITICAL FILES — READ IMMEDIATELY

### Architecture & Intelligence
```
reports/FANTASY_BASEBALL_ELITE_ROADMAP_v2.md     ← THE BIBLE
reports/daily_lineup_optimization_research.md    ← Scoring math
reports/KIMI_RESEARCH_MLB_OPENCLAW_PATTERNS.md   ← 12 edge patterns
AGENTS.md                                        ← Role boundaries
IDENTITY.md                                      ← Risk policy
```

### Backend Core (Your Domain)
```
backend/services/dashboard_service.py            ← Phase B hub
backend/fantasy_baseball/daily_lineup_optimizer.py
backend/fantasy_baseball/smart_lineup_selector.py
backend/fantasy_baseball/elite_lineup_scorer.py  ← Multiplicative scoring
backend/fantasy_baseball/lineup_constraint_solver.py  ← OR-Tools
backend/schemas.py                               ← Pydantic models
```

### Frontend (Delegate Unless Critical)
```
frontend/app/(dashboard)/dashboard/page.tsx
frontend/app/(dashboard)/fantasy/lineup/page.tsx
frontend/app/(dashboard)/fantasy/waiver/page.tsx
```

---

## 4. ELITE ENGINEERING DIRECTIVES

### A. Data Architecture (Priority: CRITICAL)
We are transitioning from **static CSV projections** to **live Bayesian updates**:

**Current (Working):**
- Statcast ingestion: ✅ Daily 6 AM ET via APScheduler
- Bayesian updater: ✅ Conjugate normal with shrinkage
- Tables: `player_projections`, `statcast_performances`

**Next (Your Build):**
```python
# Multi-source data fusion
- Steamer/ZiPS priors (CSV baseline)
- Statcast likelihood (daily performance)
- Weather adjustment (park factors)
- Injury/news dock (OpenClaw validation)
- Posterior: Weighted ensemble with confidence intervals
```

### B. Scoring Model (Priority: HIGH)
Current: `implied_runs × park_factor` (40% variance capture)  
**Target:** `implied_runs × park_factor × matchup_mult × platoon_mult × form_mult` (70-75% variance)

See `elite_lineup_scorer.py` — Kimi built the foundation. You need to:
1. Integrate weather-adjusted park factors
2. Add platoon split multipliers (LHP/RHP wOBA differentials)
3. Connect Statcast regression signals (xwOBA vs wOBA)
4. Validate against 2024 historical data

### C. UI/UX Polish (Priority: MEDIUM)
Current gaps:
- Dashboard panels use mock/static data
- No real-time refresh (need WebSocket or polling)
- No confirmed vs probable pitcher status
- Injury tags not always rendering

**Fix:** Delegate to Kimi for frontend logic, Gemini for CSS polish.

---

## 5. DELEGATION MATRIX

### Kimi CLI (Deep Intelligence Unit)
**Assign:**
- Research tasks (weather APIs, platoon data sources)
- Frontend component building (panel UI, charts)
- Data validation scripts (historical backtesting)
- Documentation (research briefs, pattern libraries)

**Do NOT assign:**
- Architecture decisions (your domain)
- Database schema changes (your domain)
- Risk/circuit breaker logic (your domain per IDENTITY.md)

### Gemini CLI (Ops & Infrastructure)
**Assign:**
- Railway deployment monitoring
- Environment variable management
- Log aggregation and alerting
- Cron job scheduling

**Do NOT assign:**
- Python code changes (restricted per AGENTS.md EMAC-075)
- Betting model modifications
- Fantasy scoring algorithm changes

### OpenClaw (Autonomous Execution)
**Current:** Runs via APScheduler at 8:30 AM for morning brief  
**Status:** Fixed coroutine warning ✅  
**Your job:** Define new pattern detection rules in `backend/services/scout.py`

---

## 6. IMMEDIATE ACTION ITEMS (Next 24 Hours — SEASON IS LIVE)

### Task 1: Update HANDOFF.md
**Priority:** CRITICAL  
**Time:** 20 minutes  
**Scope:**
- Consolidate all March 27 fixes
- Document what's LIVE vs what's MOCK data
- **CRITICAL:** Flag which dashboard panels are showing real data vs static placeholders
- Define "good enough for live games" vs "polish for next week"

### Task 2: Live Game Validation (TODAY)
**Priority:** CRITICAL — SEASON IS ACTIVE  
**Steps:**
1. Hit `/api/fantasy/lineup/2025-03-27` — verify today's actual games load
2. Verify Statcast data from yesterday (March 26) ingested correctly
3. Check injury alerts are surfacing (players actually on IL)
4. Confirm Yahoo lineup application works for tonight's games
5. **BLOCKER CHECK:** If any endpoint fails, halt all feature work and fix

### Task 3: Architecture Review
**Priority:** HIGH  
**Files to audit:**
- `backend/services/dashboard_service.py` — is it using live data or mocks?
- `backend/fantasy_baseball/smart_lineup_selector.py` — are weather/park factors integrated?
- `backend/fantasy_baseball/elite_lineup_scorer.py` — validate multiplicative math

### Task 4: Weather Integration Design
**Priority:** HIGH  
**Context:** We need live weather for ballpark factor adjustment  
**Decision needed:**
- API choice: OpenWeatherMap vs WeatherAPI vs SportRadar?
- Integration point: `smart_lineup_selector._apply_weather_adjustments()`
- Cache strategy: 1-hour TTL for weather data?

---

## 7. TECHNICAL DEBT & RISKS

### Known Issues (Live Season Context)
| Issue | Location | Severity | Action |
|-------|----------|----------|--------|
| Odds API returns partial games | `services/odds.py` | Medium | API only posts games with active lines; fallback to projections |
| Probable pitchers API flaky | `daily_lineup_optimizer.py` | **HIGH** | MLB Stats API timing; **URGENT:** Add confirmed starter flag from Yahoo |
| Dashboard using mock data | `dashboard_service.py` | **HIGH** | **URGENT:** Verify live Statcast integration before tonight's games |
| Weather not integrated | `smart_lineup_selector.py` | Medium | Design needed; park factors currently static |
| OR-Tools optional | `lineup_constraint_solver.py` | Low | Installs in prod; greedy fallback works |
| Frontend `any` types | `*.tsx` files | Low | Tech debt; tighten to strict TS later |

### Guardrails (Per IDENTITY.md)
- **Kelly formula frozen:** Do not modify `backend/betting_model.py` without explicit approval
- **Circuit breakers:** Any new external API must have CB in `backend/core/circuit_breaker.py`
- **Risk limits:** Max 2% bankroll per bet, 15% drawdown circuit breaker

---

## 8. SUCCESS METRICS (Phase B Completion)

### Quantitative
- [ ] Dashboard loads in <2 seconds (p95)
- [ ] Lineup optimizer runs in <500ms for 25 players
- [ ] Waiver recommendations generate in <1s
- [ ] Zero NaN/undefined errors in frontend console
- [ ] 100% test coverage on `elite_lineup_scorer.py`

### Qualitative
- [ ] Manager can view daily optimized lineup in 3 clicks
- [ ] Injury alerts surface before Yahoo's official notifications
- [ ] Hot/cold streaks correlate with 7-day Statcast rolling windows
- [ ] Waiver suggestions explain "why" (category need + Statcast signal)

---

## 9. COMMUNICATION PROTOCOL

### Daily Standup (With Human)
Format:
```
1. What shipped yesterday (with commit hashes)
2. What's blocked (needs human decision)
3. Today's priorities (top 3)
4. Delegation status (Kimi/Gemini tasks assigned)
```

### Handoff Updates
- Update `HANDOFF.md` every 4 hours of active work
- Include specific file paths changed
- Flag any AGENTS.md or IDENTITY.md violations immediately

---

## 10. CONTEXT FILES — QUICK REFERENCE

```bash
# Read these in order (5 minutes total)
cat reports/FANTASY_BASEBALL_ELITE_ROADMAP_v2.md | head -100
cat reports/daily_lineup_optimization_research.md | head -50
cat HANDOFF.md | head -100
cat AGENTS.md | grep -A 5 "Claude Code"
```

---

## EXECUTION CHECKLIST — SEASON IS LIVE

Before you proceed, confirm:

- [ ] Read `reports/FANTASY_BASEBALL_ELITE_ROADMAP_v2.md` Sections 1-3
- [ ] Read `HANDOFF.md` current state
- [ ] Acknowledged Kimi's fixes (roster dedup, matchup parser, serializers)
- [ ] **VALIDATED LIVE ENDPOINTS** — `/api/fantasy/lineup/2025-03-27` returns today's games
- [ ] **CONFIRMED** Statcast data ingested from March 26 games
- [ ] Understood delegation boundaries (Kimi=research/frontend, Gemini=ops)
- [ ] Reviewed `IDENTITY.md` risk constraints

**If live endpoints fail → STOP. Fix immediately. Games are happening.**

---

## PROMPT EFFICIENCY NOTES

**This prompt is optimized for:**
- Single-context loading (all critical info in one file)
- Clear ownership boundaries (no ambiguity on who does what)
- Decision points highlighted (weather API choice, etc.)
- Validation gates (checklist before execution)
- Delegation clarity (Kimi vs Gemini vs your domain)

**Do not:**
- Spend time re-reading files listed in Section 3 unless debugging
- Ask permission for architectural decisions within your swimlane
- Redo Kimi's fixes (they're tested and deployed)

**Do:**
- Update HANDOFF.md first (20 min) — this is your entry fee
- **VALIDATE LIVE ENDPOINTS IMMEDIATELY** — season is active
- Make bold architectural decisions — you are the Lead Architect
- Delegate aggressively — Kimi and Gemini are waiting for tasks
- Commit frequently — small, reviewable changes

---

## CLOSING — ⚠️ SEASON IS LIVE (March 26 Opening Day)

**We are 2 days into the season. Games are happening NOW.** The foundation is solid (migrations stable, builds green, Yahoo API connected), but we are playing catch-up.

**This is not preseason prep — this is live-fire iteration.** Every day without elite tooling is a day our users are making suboptimal lineup decisions based on static CSVs instead of live Statcast data + Bayesian updates.

**Your mandate:** Make this the most sophisticated fantasy baseball platform on the market. Not feature-bloated — **intelligent**. Every recommendation should feel like it came from a professional analyst who watched 10 games last night.

**Start by updating HANDOFF.md. Then validate the fantasy endpoints. Then build the future.**

— The Human (and Kimi, standing by for delegation)
