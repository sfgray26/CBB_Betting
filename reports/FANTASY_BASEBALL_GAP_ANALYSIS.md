# Elite Fantasy Baseball Platform — Gap Analysis & Architecture Roadmap

**Date:** March 25, 2026  
**Analyst:** Kimi CLI (Deep Intelligence Unit)  
**Scope:** Map current CBB Edge capabilities to elite fantasy baseball spec, identify gaps, create phased roadmap

---

## Executive Summary

### Current State Assessment
| Phase | Spec Features | Current Implementation | Coverage |
|-------|--------------|------------------------|----------|
| Phase 1: Foundation | 2 core features | Partial (Yahoo sync, basic roster) | ~40% |
| Phase 2: Analytics | 2 core features | Partial (Statcast, basic projections) | ~30% |
| Phase 3: Optimizers | 3 core features | Partial (MCMC simulator exists) | ~25% |
| Phase 4: Intelligence | 2 core features | Minimal (Discord alerts only) | ~15% |
| Phase 5: Elite Tools | 3 core features | None | 0% |

**Overall Maturity:** ~25-30% of elite spec implemented

### Critical Insight
**You're NOT building from scratch** — significant infrastructure exists:
- ✅ Yahoo OAuth & API integration
- ✅ Statcast/pybaseball data pipeline
- ✅ MCMC simulation engine
- ✅ OpenClaw autonomous monitoring
- ✅ Next.js 15 frontend

**The gap is FEATURE DEPTH and UX POLISH, not infrastructure.**

---

## Detailed Feature Mapping

### Phase 1: Foundation (Seamless Yahoo Sync & Daily Operations)

#### 1.1 Real-Time Yahoo Bi-Directional Sync

**Spec Requirements:**
- Auto-pull league (rosters, lineups, standings, transactions)
- One-tap lineup edits
- Add/drop, trade proposals
- Mirror Yahoo with superior UI

**Current Implementation:**
```
✅ Yahoo OAuth sync exists (backend/fantasy_baseball/yahoo_api.py)
✅ Roster fetching works
✅ Lineup reading works
⚠️ Lineup SETTING has 422 errors (UAT Issue #6 - BEING FIXED)
⚠️ Add/drop partially works (UAT Issue #7 - BEING FIXED)
❌ Trade proposals not implemented
❌ Real-time sync (currently poll-based)
```

**Gap Analysis:**
| Feature | Status | Gap | Effort |
|---------|--------|-----|--------|
| Roster fetch | ✅ Working | None | — |
| Lineup read | ✅ Working | None | — |
| Lineup write | ⚠️ Buggy | 422 errors | 2-3 days |
| Add/drop | ⚠️ Buggy | API endpoint issues | 2-3 days |
| Trade proposals | ❌ Missing | Full feature needed | 1 week |
| Real-time sync | ❌ Missing | Webhooks/polling | 3-5 days |
| "Superior UI" | ⚠️ Basic | Next.js UI exists but minimal | 2-3 weeks |

**Technical Debt:**
- Yahoo API error handling needs retry logic
- Rate limiting not implemented
- No fallback for API failures

---

#### 1.2 Customizable Daily Dashboard + Push Notifications

**Spec Requirements:**
- Today's lineup gaps
- Hot/cold streaks
- Waiver targets
- Injury flags
- Configurable alerts

**Current Implementation:**
```
✅ Daily Lineup page exists (Next.js)
✅ Waiver Wire page exists
✅ Injury data partially available (pybaseball)
⚠️ Dashboard is basic, not "customizable"
⚠️ Push notifications via Discord only (not mobile push)
⚠️ No "hot/cold streak" visualization
❌ No "lineup gaps" detection
```

**Gap Analysis:**
| Feature | Status | Gap | Effort |
|---------|--------|-----|--------|
| Dashboard exists | ✅ Yes | Basic implementation | — |
| Lineup gaps | ❌ Missing | Detect unfilled positions | 2-3 days |
| Hot/cold streaks | ❌ Missing | Calculate 7/14/30 trends | 3-5 days |
| Waiver targets | ⚠️ Partial | Has page but no ranking | 1 week |
| Injury flags | ⚠️ Partial | Data exists, not integrated | 2-3 days |
| Configurable alerts | ❌ Missing | User preference system | 1 week |
| Mobile push | ❌ Missing | PWA or native push | 1-2 weeks |

**Technical Debt:**
- No user preference storage
- Alert system is Discord-only (not user-facing)
- No mobile app or PWA

---

### Phase 2: Analytical Core (Projections & Insights)

#### 2.1 Multi-Source Projection Aggregator

**Spec Requirements:**
- Fangraphs (ZiPS, Steamer, Depth Charts, ATC, THE BAT/X)
- Auto-blend into custom ROS projections
- Adjust for playing time, park factors, platoon splits
- Export to CSV

**Current Implementation:**
```
✅ pybaseball integration (FanGraphs data)
✅ Basic projection aggregation (team wRC+)
⚠️ Only using xERA for pitchers, not full projection systems
⚠️ No ROS (rest-of-season) blending
⚠️ Park factors hardcoded, not dynamic
⚠️ No platoon split adjustments
❌ No export to CSV
❌ Only 1-2 sources, not 5+
```

**Gap Analysis:**
| Feature | Status | Gap | Effort |
|---------|--------|-----|--------|
| Fangraphs data | ✅ Partial | pybaseball exists | — |
| ZiPS/Steamer | ❌ Missing | Need to fetch specific systems | 1 week |
| ATC/THE BAT | ❌ Missing | May require API/scraping | 1-2 weeks |
| ROS blending | ❌ Missing | Time-weighted projections | 1 week |
| Custom weights | ❌ Missing | User-configurable blending | 3-5 days |
| Park factors | ⚠️ Hardcoded | Dynamic park factor adjustment | 3-5 days |
| Platoon splits | ❌ Missing | LHP/RHP projections | 3-5 days |
| CSV export | ❌ Missing | Data export feature | 2-3 days |

**Technical Debt:**
- pybaseball cache TTL is 24hr (may be stale)
- No versioning for projection updates
- Missing data validation

---

#### 2.2 Advanced Player Scout & Comparison Tool

**Spec Requirements:**
- Side-by-side Statcast (barrel%, xERA, whiff rates)
- vs. LHP/RHP splits
- Recent 7/14/30-day trends
- Replacement-level baselines
- "Streamers with >20% projected roster% increase"

**Current Implementation:**
```
✅ Statcast data in pybaseball (StatcastBatter, StatcastPitcher)
✅ Some advanced metrics (xwOBA, barrel%, xERA)
⚠️ No side-by-side comparison UI
⚠️ No vs LHP/RHP splits in data model
⚠️ No trend visualization (7/14/30 day)
⚠️ No replacement-level baseline
⚠️ No "streamer detector"
```

**Gap Analysis:**
| Feature | Status | Gap | Effort |
|---------|--------|-----|--------|
| Statcast data | ✅ Available | In pybaseball | — |
| Advanced metrics | ✅ Partial | xwOBA, barrel% present | — |
| Side-by-side UI | ❌ Missing | Comparison interface | 1 week |
| LHP/RHP splits | ❌ Missing | Data not fetched | 3-5 days |
| Trend visualization | ❌ Missing | Charts/graphs | 1 week |
| Replacement baseline | ❌ Missing | Calculate replacement level | 3-5 days |
| Streamer detector | ❌ Missing | Algorithm for pickups | 1 week |

---

### Phase 3: Decision Accelerators (Optimizers & Analyzers)

#### 3.1 AI Lineup Optimizer

**Spec Requirements:**
- Max projected cats/points
- Factors matchups, rest days, weather, probable pitchers
- "Quick Set" button pushes to Yahoo

**Current Implementation:**
```
✅ MCMC Simulator exists (MCMCWeeklySimulator)
✅ Basic lineup optimization possible
⚠️ No "one-click" Yahoo push (API errors being fixed)
⚠️ No weather integration
⚠️ No matchup factoring
⚠️ No rest day tracking
⚠️ Probable pitchers partially tracked
```

**Gap Analysis:**
| Feature | Status | Gap | Effort |
|---------|--------|-----|--------|
| Optimization engine | ✅ Exists | MCMC simulator | — |
| Yahoo push | ⚠️ Buggy | 422 errors being fixed | — |
| Weather | ❌ Missing | MLB weather API | 3-5 days |
| Matchups | ❌ Missing | Opp pitcher analysis | 3-5 days |
| Rest days | ❌ Missing | Track player rest | 2-3 days |
| Probable pitchers | ⚠️ Partial | Needs better integration | 2-3 days |
| "Quick Set" | ❌ Missing | One-button optimize + push | 1 week |

---

#### 3.2 Dynamic Trade Analyzer

**Spec Requirements:**
- Projected ROS impact on standings
- "Win probability" shift
- Scans public trade offers
- Suggests counteroffers

**Current Implementation:**
```
❌ NOT IMPLEMENTED
```

**Gap Analysis:**
| Feature | Status | Gap | Effort |
|---------|--------|-----|--------|
| Trade analysis | ❌ Missing | Full feature needed | 2 weeks |
| ROS impact calc | ❌ Missing | Standings projection | 1 week |
| Win probability | ❌ Missing | Category probability | 1 week |
| Trade scanning | ❌ Missing | Yahoo API integration | 1 week |
| Counteroffer AI | ❌ Missing | Suggestion engine | 2 weeks |

---

#### 3.3 Waiver/FAAB & Streaming Optimizer

**Spec Requirements:**
- Ranked add/drop list
- Projected % owned next week
- FAAB bid suggestions
- "Daily streamer" tab

**Current Implementation:**
```
✅ Waiver Wire page exists
✅ WaiverEdgeDetector exists
✅ MCMC simulator for projections
⚠️ No FAAB bid suggestions
⚠️ No "projected % owned"
⚠️ No streaming-specific UI
⚠️ No matchup heatmaps
```

**Gap Analysis:**
| Feature | Status | Gap | Effort |
|---------|--------|-----|--------|
| Waiver wire | ✅ Exists | Basic implementation | — |
| Edge detection | ✅ Exists | WaiverEdgeDetector | — |
| Ranked list | ⚠️ Partial | Needs better sorting | 3-5 days |
| FAAB suggestions | ❌ Missing | Bid optimization | 1 week |
| Projected ownership | ❌ Missing | Ownership prediction | 1-2 weeks |
| Streamer tab | ❌ Missing | Dedicated UI | 1 week |
| Matchup heatmaps | ❌ Missing | Visual matchup tool | 1 week |

---

### Phase 4: Proactive Intelligence

#### 4.1 Injury Risk & News Aggregator

**Spec Requirements:**
- Rotowire/MLB news
- Proprietary severity models
- Simulates roster impact
- Suggests immediate adds

**Current Implementation:**
```
⚠️ OpenClaw monitors but not injury-specific
⚠️ Discord alerts exist but manual
❌ No Rotowire integration
❌ No severity models
❌ No automatic roster impact sim
```

**Gap Analysis:**
| Feature | Status | Gap | Effort |
|---------|--------|-----|--------|
| News aggregation | ❌ Missing | Rotowire/MLB API | 1 week |
| Injury models | ❌ Missing | Severity prediction ML | 2-3 weeks |
| Roster impact sim | ❌ Missing | Automated simulation | 1 week |
| Auto-add suggestions | ❌ Missing | Recommendation engine | 1 week |

---

#### 4.2 Monte Carlo Standings Projector

**Spec Requirements:**
- 10,000+ sims of remaining schedule
- Category clinch probability
- Trade deadline urgency

**Current Implementation:**
```
✅ MCMC Simulator exists (can be adapted)
⚠️ No standings projection
⚠️ No category clinch probability
⚠️ No "trade deadline urgency" metric
```

**Gap Analysis:**
| Feature | Status | Gap | Effort |
|---------|--------|-----|--------|
| MCMC engine | ✅ Exists | Adapt for standings | 1 week |
| Standings projection | ❌ Missing | Full league sim | 1 week |
| Clinch probability | ❌ Missing | Category math | 3-5 days |
| Trade deadline urgency | ❌ Missing | Decision metric | 3-5 days |

---

### Phase 5: Elite Differentiators

#### 5.1 Backtesting Engine + Strategy Simulator

**Spec Requirements:**
- Historical "what-if" moves
- AI suggests custom strategies
- Anonymized league benchmarking

**Current Implementation:**
```
❌ NOT IMPLEMENTED
```

**Gap Analysis:**
| Feature | Status | Gap | Effort |
|---------|--------|-----|--------|
| Backtesting | ❌ Missing | Historical simulation | 2 weeks |
| Strategy AI | ❌ Missing | ML recommendations | 3-4 weeks |
| Benchmarking | ❌ Missing | League comparison | 2 weeks |

---

## Consolidated Gap Summary

### By Effort Level

| Effort | Features | Priority |
|--------|----------|----------|
| **2-3 days** | Fix Yahoo API errors, lineup gaps, injury flags, park factors | 🔴 Critical |
| **3-5 days** | Hot/cold streaks, platoon splits, rest days, weather, replacement baseline | 🟠 High |
| **1 week** | Custom dashboard, trade analysis, FAAB optimizer, standings projection | 🟡 Medium |
| **1-2 weeks** | Mobile push, CSV export, LHP/RHP splits, comparison UI | 🟢 Lower |
| **2+ weeks** | Multi-source projections, backtesting, strategy AI | 🔵 Future |

### By Impact

| Impact | Features | Current Pain |
|--------|----------|--------------|
| **Critical** | Yahoo API fixes, lineup optimizer | UAT blocking issues |
| **High** | Better projections, trade analyzer | Manual workarounds needed |
| **Medium** | Dashboard, notifications | UX friction |
| **Differentiating** | AI strategy, backtesting | Competitive edge |

---

## Recommended Roadmap

### Phase A: Critical Fixes (Next 2 Weeks)
**Goal:** Get existing features working reliably

1. **Fix Yahoo API errors** (UAT Phase 2)
   - Lineup 422 errors
   - Waiver wire 503 errors
   - Add retry logic

2. **Stabilize Core Sync**
   - Add rate limiting
   - Better error handling
   - API fallback modes

3. **Complete UAT Fixes**
   - Calibration (brier score)
   - Projection defaults
   - Odds Monitor

### Phase B: Foundation Complete (Weeks 3-4)
**Goal:** Match basic Yahoo functionality + some improvements

4. **Enhanced Dashboard**
   - Lineup gaps detection
   - Hot/cold streaks
   - Injury flags

5. **Better Projections**
   - ROS blending
   - Park factor adjustments
   - Platoon splits

6. **Notifications**
   - Beyond Discord (email/web push)
   - User-configurable

### Phase C: Analytical Core (Weeks 5-8)
**Goal:** Surpass Yahoo with analytics

7. **Trade Analyzer**
   - Impact calculation
   - ROS projections
   - Win probability shift

8. **Advanced Scout**
   - Side-by-side comparison
   - Trend visualization
   - Streamer detection

9. **Waiver Optimizer**
   - FAAB suggestions
   - Ownership prediction
   - Streaming tab

### Phase D: Intelligence Layer (Weeks 9-12)
**Goal:** Proactive insights

10. **Standings Projection**
    - Monte Carlo simulation
    - Category clinch prob
    - Trade deadline urgency

11. **Injury Intelligence**
    - News aggregation
    - Severity models
    - Auto-suggestions

12. **Mobile Experience**
    - PWA or native
    - Push notifications
    - Offline mode

### Phase E: Elite Differentiators (Months 4-6)
**Goal:** Top 1% tools

13. **Backtesting Engine**
14. **Strategy AI**
15. **Community Benchmarking**

---

## Technical Architecture Recommendations

### Immediate Needs

1. **API Reliability Layer**
```python
class YahooAPIClient:
    def __init__(self):
        self.retry_policy = RetryPolicy(
            max_retries=3,
            backoff_factor=2,
            status_forcelist=[422, 429, 500, 502, 503]
        )
    
    async def set_lineup(self, ...):
        # Add validation before call
        # Add retry logic
        # Add fallback to manual
```

2. **Projection Service**
```python
class ProjectionAggregator:
    SOURCES = ['zips', 'steamer', 'depth_charts', 'atc', 'the_bat']
    
    def blend_projections(self, weights=None):
        # Weighted average of sources
        # ROS time decay
        # Park/Platoon adjustments
```

3. **Notification Service**
```python
class NotificationManager:
    CHANNELS = ['email', 'push', 'discord', 'sms']
    
    def send(self, user, message, priority):
        # User preference-based routing
        # Batch non-urgent
        # Rate limiting
```

### Data Model Enhancements

```sql
-- User preferences
CREATE TABLE user_preferences (
    user_id UUID,
    notification_settings JSONB,
    projection_weights JSONB,
    dashboard_layout JSONB
);

-- Player projections by source
CREATE TABLE player_projections (
    player_id VARCHAR,
    source VARCHAR, -- 'zips', 'steamer', etc.
    ros_war FLOAT,
    updated_at TIMESTAMP
);

-- Trade analysis cache
CREATE TABLE trade_analysis (
    trade_hash VARCHAR,
    projected_impact JSONB,
    win_prob_shift FLOAT
);
```

---

## Success Metrics

| Metric | Current | Phase A | Phase B | Phase C | Elite |
|--------|---------|---------|---------|---------|-------|
| Yahoo API reliability | ~70% | 95% | 98% | 99% | 99.9% |
| Lineup optimization | Manual | Basic | Semi-auto | Auto-push | AI-optimized |
| Projection sources | 1-2 | 2-3 | 4-5 | 5+ | 6+ custom blend |
| Daily active users | N/A | N/A | You | You + friends | Public beta |
| Feature coverage | 25% | 35% | 50% | 75% | 100%+ |

---

## Conclusion

**You're 25-30% to the elite spec, but with solid foundations.**

The critical path is:
1. **Fix UAT issues** (stability)
2. **Complete Phase 1** (match Yahoo)
3. **Build Phase 2-3** (surpass Yahoo)
4. **Add Phase 4-5** (top 1% differentiation)

**Biggest architectural needs:**
- Reliable Yahoo API layer with retries/fallbacks
- Projection aggregation service
- User preference system
- Mobile-ready notifications

**Files Created:**
- `reports/FANTASY_BASEBALL_GAP_ANALYSIS.md` — This analysis
- `CLAUDE_FANTASY_ROADMAP_PROMPT.md` — Implementation prompt (to be created)

---

**Document Version:** KIMI-FANTASY-2026-0325
**Next Step:** Review and approve roadmap, then create detailed implementation prompts
