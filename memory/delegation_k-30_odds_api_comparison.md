# Delegation Bundle — K-30: Odds API vs BDL MLB Odds Comparison

**Agent:** Kimi CLI (Deep Intelligence Unit)
**Date:** April 8, 2026
**Priority:** HIGH — Blocks MLB betting model odds architecture decision

---

## Mission

Compare **OddsAPI Basic** (20k calls/month) and **BallDontLie GOAT MLB** odds capabilities to identify unique differentiators and optimal usage patterns for the MLB Platform.

**Context:** We currently route all MLB odds through BDL (CLAUDE.md guidance), but we have both APIs available. Need to understand:
1. What does OddsAPI offer that BDL doesn't?
2. What does BDL offer that OddsAPI doesn't?
3. Where should we use each API for maximum value?

---

## Research Requirements

### 1. OddsAPI Basic Analysis

**Documentation:** https://oddsapi.com/docs/ (or existing codebase usage)

**Key Questions:**
- What MLB odds endpoints exist? (moneyline, runline, totals, props?)
- What bookmakers are covered? (How many? Which ones?)
- What's the data latency? (How fast do odds updates appear?)
- What's the historical odds coverage? (Can we get closing lines from 2023-2025?)
- What are the rate limits for 20k/month plan?
- Does OddsAPI offer **market movement** data? (opening → current → closing)
- Does OddsAPI offer **consensus** data? (percentage of bets on each side)

**Codebase Context:**
- Current usage: `backend/services/odds.py` (CBB archival closing lines only)
- Feature set: `fetch_closing_lines()`, `capture_lines_job()` (CBB-specific)

### 2. BallDontLie GOAT MLB Analysis

**Documentation:** https://api.balldontlie.io/v1/doc (or existing codebase)

**Key Questions:**
- What MLB odds endpoints exist? (Check `/mlb/v1/` endpoints)
- What bookmakers are covered? (Compare to OddsAPI)
- What's the data latency? (Real-time? Delayed?)
- What's the historical odds coverage? (2023-2025?)
- Are there rate limits? (GOAT plan should be generous)
- Does BDL offer **market movement** data?
- Does BDL offer **consensus** data?
- Does BDL offer **pre-game** vs **live** odds?

**Codebase Context:**
- Current usage: `backend/services/balldontlie.py` (needs MLB expansion)
- Feature set: Currently NCAAB (expired subscription); MLB stubs exist

### 3. Feature Comparison Matrix

Create a table comparing:

| Feature | OddsAPI Basic | BDL GOAT MLB | Winner |
|---------|--------------|--------------|--------|
| Bookmaker coverage | ? | ? | ? |
| Historical odds (2023-2025) | ? | ? | ? |
| Market movement data | ? | ? | ? |
| Consensus data | ? | ? | ? |
| Latency | ? | ? | ? |
| Rate limits | 20k/month | ? | ? |
| Props betting | ? | ? | ? |
| First-half / 5-inning lines | ? | ? | ? |
| Futures odds | ? | ? | ? |
| API reliability | ? | ? | ? |

### 4. Use Case Recommendations

For each MLB Platform feature, recommend which API to use:

**Fantasy Baseball Features:**
- Player prop odds (for DFS/Fanduel overlap)?
- No relevant use case

**MLB Betting Model Features:**
- Opening odds capture (morning odds)
- Closing odds capture (for CLV analysis)
- Market movement tracking (steam moves, reverse line movement)
- Consensus data (sharp vs. public money)
- Live odds (in-game betting)

**Decision Criteria:**
- Use OddsAPI if: [unique differentiator]
- Use BDL if: [unique differentiator]
- Use both if: [complementary strengths]

---

## Constraints

**Rate Limit Budget:**
- OddsAPI: 20k calls/month = ~667 calls/day
- BDL GOAT: Unknown (research required)
- Current usage: OddsAPI for CBB archival only (minimal)

**Codebase Architecture:**
- ADR-004: `betting_model.py` must NOT import CBB model (boundary still exists)
- MLB analysis: `backend/services/mlb_analysis.py` (stub-level)
- Daily ingestion: `backend/services/daily_ingestion.py` (has `_poll_mlb_odds` stub)

**Non-Negotiable:**
- Do NOT use OddsAPI for real-time features if BDL is better (20k limit is tight)
- Do NOT use BDL for CBB (subscription cancelled, will 401)
- Keep CLAUDE.md guidance updated (this decision affects architecture docs)

---

## Deliverables

1. **Feature Comparison Matrix** (table format)
2. **API Endpoint Inventory** (what endpoints exist for each API)
3. **Recommendation Report** (3-5 pages) covering:
   - Unique strengths of each API
   - Weaknesses/gaps of each API
   - Recommended usage patterns per feature
   - Rate limit budget allocation
   - Code migration plan (if needed)

4. **HANDOFF.md update** — summary of findings for S30

---

## Escalation

If you discover:
- **OddsAPI has a killer feature BDL lacks** → Flag as "OddsAPI Unique Value: [feature]"
- **BDL is strictly better** → Recommend sunsetting OddsAPI for MLB
- **Both APIs are mediocre** → Recommend third option (if exists)
- **Rate limits will be a problem** → Flag as "Budget Constraint: Upgrade required"

---

## Timebox

Research: 1-2 hours (max)
Report generation: 1 hour
Total: 3 hours max

**Deadline:** Before S29 ends (architectural decision blocks MLB betting odds work)

---

**Assigned to:** Kimi CLI (Deep Intelligence Unit)
**Review required:** Claude Code (Principal Architect)
**Sign-off:** User (final decision on API usage strategy)
