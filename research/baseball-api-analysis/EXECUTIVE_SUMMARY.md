# Baseball API Research — Executive Summary for Claude Code

## Research Context
**Project:** Dual-purpose application (Fantasy Baseball + MLB Betting)  
**Current State:** CBB Betting V9.2 architecture, College Basketball season ending  
**Priority:** 1) Fix Fantasy Baseball app → 2) Expand betting model to MLB  

---

## Key Findings from Research Documents

### Document 1: OddsAPI Champion vs BallDontLie

| Criteria | OddsAPI Champion (Current) | BallDontLie GOAT |
|----------|---------------------------|------------------|
| **Cost** | $49/mo ($588/yr) | $39.99/mo (~$480/yr) |
| **Rate Limit** | 90,000 calls/month | 600 req/min (~864K/day) |
| **Bookmakers** | 40+ | ~15-20 |
| **MLB Coverage** | Moneyline, spreads, totals, props | Full odds + props |
| **Webhooks** | ❌ Polling only | ✅ Real-time webhooks |
| **MCP Server** | ❌ No | ✅ Official MCP integration |
| **Data Integration** | Odds-only | Stats + Odds unified |

**Verdict:** BallDontLie offers superior rate limits, webhooks, and unified data at lower cost. Savings: ~$108/year.

---

### Document 2: BallDontLie API Deep Dive

**Core Capabilities:**
- **MLB Coverage:** Teams, players, games, box scores, play-by-play, season stats, injuries
- **Webhooks:** 125+ event types, HMAC-SHA256 signed payloads
- **MCP Server:** 250+ endpoints for AI agent integration
- **Google Sheets Integration:** 150+ functions for non-technical access
- **Pricing Tiers:**
  - Free: 5 req/min (no MLB betting)
  - ALL-STAR ($9.99/mo): 60 req/min, basic odds
  - GOAT ($39.99/mo): 600 req/min, full odds + props

**Strategic Position:** Between free APIs (MLB Stats) and enterprise (Sportradar). AI-native with OpenAPI specs.

---

### Document 3: Baseball APIs Technical Research

**Recommended Multi-Tier Architecture:**
1. **Yahoo Fantasy API** → League-specific operations (OAuth 2.0)
2. **MLB Stats API** → Official player statistics (FREE, no key)
3. **Sportradar/BallDontLie** → Real-time game data (if budget allows)
4. **Caching Layer** → 5-15 min TTL for rate limit management

**Key APIs Analyzed:**

| API | Auth | Cost | Best For |
|-----|------|------|----------|
| MLB Stats API (statsapi.mlb.com) | None | Free | Official stats, play-by-play |
| Yahoo Fantasy API | OAuth 2.0 | Free | League/team/roster management |
| ESPN API | None | Free | Basic scores, schedules |
| BallDontLie | API Key | $9.99-$40/mo | Unified stats + odds |
| Sportradar | Contract | $500+/mo | Enterprise real-time |

---

## Integration with OpenClaw Skills

### Previously Investigated Skills (from Gemini recommendations)

| Skill | Status | Relevance to MLB |
|-------|--------|------------------|
| `khaney64/baseball` | ✅ Real | MLB Stats API wrapper — FREE |
| `robbyczgw-cla/sports-ticker` | ✅ Real | ESPN API multi-sport — includes MLB |
| `optionns/openclaw-skills-sports` | ❌ 404 | Hallucinated by Gemini |
| `autogame-17/code-stats` | ❌ 404 | Hallucinated by Gemini |
| `firecrawl/firecrawl-cli` | ⚠️ Real | Standalone CLI, not OpenClaw skill |

**Key Point:** The OpenClaw skills use FREE APIs (MLB Stats, ESPN). BallDontLie would be a NEW premium integration not covered by existing skills.

---

## Architectural Recommendations

### Option A: Free-First Architecture (MVP)
```
Fantasy App → MLB Stats API (statsapi.mlb.com)
          → Yahoo Fantasy API (OAuth)
          → ESPN API (schedules/scores)
```
- **Pros:** Zero cost, official data, proven OpenClaw skills exist
- **Cons:** No real-time webhooks, no unified odds+stats
- **Best for:** Initial fantasy baseball launch, budget constraints

### Option B: Hybrid Architecture (Recommended)
```
Fantasy Core  → MLB Stats API (free, official stats)
              → Yahoo Fantasy API (league operations)
              
Betting Layer → BallDontLie GOAT ($40/mo)
              → Webhooks for live game events
              → MCP server for AI agent integration
```
- **Pros:** Best of both worlds, webhooks for live betting, AI-native
- **Cons:** $40/month cost, additional integration work
- **Best for:** Full fantasy + betting platform

### Option C: Enterprise Architecture
```
All Data → Sportradar/Stats Perform (contract required)
```
- **Pros:** Official partnerships, highest reliability
- **Cons:** $500+/mo, enterprise sales cycle
- **Best for:** Production betting product at scale

---

## Decision Required from Claude Code

### Question 1: Data Strategy
Should we proceed with **Option A (Free)** or **Option B (Hybrid)** for the initial MLB pivot?

### Question 2: BallDontLie Integration
If Hybrid: Should I (Kimi) draft the BallDontLie service wrapper and webhook handlers?

### Question 3: OpenClaw Skill Usage
Should we leverage the existing `khaney64/baseball` and `sports-ticker` skills, or build direct API integrations?

### Question 4: CBB-to-MLB Migration Priority
Sequence confirmation:
1. Fix Fantasy Baseball app (MLB Stats API + Yahoo)
2. Expand betting model to MLB (add BallDontLie)
3. Archive CBB models post-tournament

---

## Files in This Directory

| File | Description |
|------|-------------|
| `01_OddsAPI_Champion_vs_BallDontLie.docx` | Cost/benefit analysis of switching data providers |
| `02_BallDontLie_API_Deep_Dive.docx` | Technical deep dive on BallDontLie platform |
| `03_Baseball_APIs_Technical_Research.docx` | Comprehensive API landscape analysis |
| `EXECUTIVE_SUMMARY.md` | This file |

---

## Next Steps

1. **Await architectural decision** from Claude Code on data strategy
2. **Implement fantasy baseball core** using MLB Stats API + Yahoo
3. **Design betting integration** if BallDontLie selected
4. **Migrate CBB models** to MLB equivalents post-tournament

---

*Report generated: March 31, 2026*  
*Research by: Kimi Deep Intelligence Unit*  
*Skills audit: 2 of 5 claimed skills verified (40% accuracy)*
