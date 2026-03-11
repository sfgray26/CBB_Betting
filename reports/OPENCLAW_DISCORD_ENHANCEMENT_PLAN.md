# OpenClaw Discord Enhancement Plan

**Date:** March 11, 2026  
**Author:** Kimi CLI (Deep Intelligence Unit)  
**Status:** Ready for Implementation  
**Document:** OPCL-001

---

## Executive Summary

With Discord's new 16-channel architecture in place, OpenClaw is significantly underutilized. This plan outlines how to increase OpenClaw's impact by **5-10x** through automated intelligence delivery, real-time monitoring, and intelligent escalation workflows.

**Current State:** OpenClaw runs integrity checks during nightly analysis and occasional health checks.  
**Target State:** OpenClaw becomes the primary intelligence hub, delivering automated briefs, monitoring live events, and orchestrating multi-channel alerts.

---

## 1. Current OpenClaw Capabilities

### Core Functions (Already Implemented)
| Feature | Status | Latency | Notes |
|---------|--------|---------|-------|
| Heuristic integrity checks | ✅ LIVE | <1ms | Keyword-based, no LLM needed |
| Async batch processing | ✅ LIVE | <30s/8 games | 8 concurrent workers |
| Escalation queue | ✅ LIVE | N/A | File-based, manual review |
| Telemetry/metrics | ✅ LIVE | Real-time | Performance tracking |
| Discord notifications | ⚠️ LIMITED | Via coordinator | Only high-stakes escalations |

### Discord Channels Available
```
🎯 OPENCLAW INTEL (Currently Underutilized)
├── openclaw-briefs       ← Morning intelligence summaries (NEW USE)
├── openclaw-escalations  ← High-stakes alerts (CURRENT USE)
└── openclaw-health       ← System telemetry (NEW USE)
```

---

## 2. Enhancement Overview

### 2.1 The 5 Pillars of Enhanced OpenClaw

```
┌─────────────────────────────────────────────────────────────────┐
│                    OPENCLAW INTELLIGENCE HUB                   │
├─────────────────────────────────────────────────────────────────┤
│  PILLAR 1          PILLAR 2          PILLAR 3                   │
│  Morning Brief     Live Monitor      Fantasy Watch              │
│  ─────────────     ───────────       ───────────                │
│  Daily 7 AM ET     Game-time alerts  Closer/lineup tracking     │
│  → #briefs         → #escalations    → #fantasy-news            │
│                                                                 │
│  PILLAR 4          PILLAR 5                                       │
│  System Telemetry  Tournament Mode                                │
│  ────────────────  ─────────────                                  │
│  Health dashboards March Madness special ops                      │
│  → #health         → #tournament                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Impact Multipliers

| Enhancement | Current Frequency | Target Frequency | Impact |
|-------------|-------------------|------------------|--------|
| Morning Brief | Manual/never | Daily 7 AM ET | 7x/week intelligence |
| Live Game Monitor | Never | Every 2h on game days | 10-15x/day during March |
| Fantasy Alerts | Never | Daily 7 AM + ad-hoc | 7+ alerts/week |
| System Telemetry | Manual checks | Continuous | Real-time visibility |
| Tournament Mode | Basic | Elite Eight+ special | 8-15 games with extra ops |

---

## 3. Detailed Enhancement Specifications

### 3.1 PILLAR 1: Morning Intelligence Brief (OC-MB)

**Channel:** `#openclaw-briefs`  
**Frequency:** Daily at 7:00 AM ET  
**Trigger:** Cron job + HEARTBEAT

**Content:**
```markdown
🌅 OPENCLAW MORNING BRIEF — March 18, 2026

📊 TODAY'S SLATE
• 4 CBB games analyzed
• 2 BET-tier recommendations  
• 1 CONSIDER (monitor for line moves)

🔍 INTEGRITY SUMMARY
• All games: CONFIRMED ✓
• No injuries or lineup concerns
• Weather: Clear (all indoor venues)

⚡ SHARP MONEY ALERTS
• Gonzaga -3.5 → -4.5 (steam detected, 45 min)
• Recommendation: Wait for line to stabilize

🏀 TOURNAMENT WATCH
• First Four: 2 games tonight
• Upset Probability: 28% (based on 3/6 seed matchups)

📋 ESCALATION QUEUE
• 0 high-stakes games pending review

🎯 FANTASY BASEBALL
• Today's optimal lineup posted in #fantasy-lineups
• Closer watch: Diaz (NYM) — 2 saves in 3 days
```

**Implementation:**
```python
# backend/services/openclaw_briefs.py (NEW)
async def generate_morning_brief() -> DiscordEmbed:
    """Compile overnight intelligence into morning brief."""
    
    # 1. Query today's slate
    slate = await get_today_games()
    
    # 2. Check integrity cache
    integrity_summary = await check_integrity_cache(slate)
    
    # 3. Sharp money overnight
    sharp_signals = await get_overnight_sharp_moves()
    
    # 4. Escalation queue status
    pending = escalation_queue.get_pending()
    
    # 5. Fantasy updates (if in season)
    fantasy_notes = await get_fantasy_overnight_news()
    
    return compile_brief_embed(slate, integrity_summary, 
                               sharp_signals, pending, fantasy_notes)
```

---

### 3.2 PILLAR 2: Live Game Monitor (OC-LM)

**Channel:** `#openclaw-escalations` (for urgent), `#cbb-alerts` (for info)  
**Frequency:** Every 2 hours on game days  
**Trigger:** Cron during active hours (12 PM - 11 PM ET)

**Monitoring Scope:**
- **T-4h:** Starting lineup confirmation
- **T-2h:** Injury/availability updates
- **T-30m:** Final integrity check
- **Live:** Upset alerts (if we add live win probability)

**Content Example:**
```markdown
⚠️ GAME-TIME ALERT — 2:15 PM ET

Game: Gonzaga @ Saint Mary's (9:00 PM ET)
Alert Level: CAUTION 🟡

Issue: Starting PG questionable (ankle)
Source: Multiple beat reporters
Confidence: 75%

Action: Model edge reduced 0.5%
Bet recommendation: DOWNGRADE to CONSIDER

Next check: 6:00 PM ET
```

**Implementation:**
```python
# backend/services/openclaw_live_monitor.py (NEW)
class LiveGameMonitor:
    """Monitor games approaching tip-off for late-breaking news."""
    
    CHECK_INTERVAL_MINUTES = 120  # Every 2 hours
    
    async def check_approaching_games(self):
        """Check all games within 4 hours of tip-off."""
        games = await get_games_within_hours(4)
        
        for game in games:
            # Fresh DDGS search
            search_text = await search_game_news(game)
            
            # Quick integrity check
            result = await openclaw.check_integrity(
                search_text=search_text,
                home_team=game.home_team,
                away_team=game.away_team,
                game_key=game.key,
                recommended_units=game.recommended_units
            )
            
            # Alert if changed or concerning
            if result.verdict != "CONFIRMED":
                await send_live_alert(game, result)
```

---

### 3.3 PILLAR 3: Fantasy Baseball Intelligence (OC-FB)

**Channels:** `#fantasy-news`, `#fantasy-waivers`  
**Frequency:** Daily 7 AM ET + ad-hoc  
**Trigger:** Cron + webhook events

**Daily 7 AM Report:**
```markdown
⚾ FANTASY BASEBALL INTELLIGENCE — March 18

🔥 CLOSER SITUATIONS
• Edwin Diaz (NYM): Locked in, 3 SV in last 4 days
• Ryan Helsley (STL): Shaky, 2 BS this week — monitor
• Kenley Jansen (ATL): Day-to-day (back) — A. Minter next

📋 LINEUP CONFIRMATIONS (7:05 PM ET games)
✅ Mookie Betts (LAD) — Batting 1st, 2B
✅ Shohei Ohtani (LAD) — Batting 2nd, DH
⚠️ Ronald Acuña Jr. (ATL) — Not in lineup (rest)

💰 WAIVER WIRE SLEEPERS
• Cade Marlowe (SEA, OF): Hitting .340, 2 HR last week
• Joey Ortiz (MIL, 2B/SS): Starting 5 straight, eligibility

📰 NEWS SNAPSHOT
• Justin Verlander (HOU): Return pushed to April 15
• Wander Franco (TB): Cleared to resume baseball activities
```

**Closer Monitor (Special Feature):**
```python
# backend/services/openclaw_closer_monitor.py (NEW)
async def monitor_closer_situations():
    """Track closer changes and alert fantasy managers."""
    
    for team in mlb_teams:
        closer_status = await get_closer_status(team)
        
        # Alert on changes
        if closer_status.changed_since_yesterday:
            await send_closer_alert(
                channel="fantasy-news",
                team=team,
                old_closer=closer_status.previous,
                new_closer=closer_status.current,
                reason=closer_status.change_reason
            )
```

---

### 3.4 PILLAR 4: System Telemetry Dashboard (OC-TM)

**Channel:** `#openclaw-health`  
**Frequency:** Real-time (every 5 min)  
**Trigger:** Background daemon

**Dashboard Elements:**
```markdown
📊 OPENCLAW TELEMETRY — 14:35:00 UTC

🔧 INTEGRITY CHECKS (Last 24h)
• Total: 47 checks
• Avg latency: 0.8ms
• Distribution: CONFIRMED 42 | CAUTION 4 | VOLATILE 1

⚡ SHARP MONEY
• Signals detected: 12
• High confidence: 3
• Model edge adjusted: +0.5% to -0.8%

🎯 PREDICTIONS
• Today: 4 games
• BET tier: 2
• CONSIDER: 1
• PASS: 1

🚨 SYSTEM HEALTH
• Data sources: 2/2 ✅ (KenPom, BartTorvik)
• Odds monitor: ✅ (last poll 2 min ago)
• Discord: ✅ (connected)
• Database: ✅ (response <10ms)

📋 ESCALATION QUEUE
• Pending: 0
• Resolved today: 2
```

**Implementation:**
```python
# backend/services/openclaw_telemetry.py (NEW)
class TelemetryDashboard:
    """Real-time system health and performance tracking."""
    
    async def post_telemetry_update(self):
        """Post updated telemetry to Discord."""
        telemetry = openclaw.telemetry.to_dict()
        
        embed = DiscordEmbed(
            title="OpenClaw Telemetry",
            description=self.format_telemetry(telemetry),
            color=0x00FF00 if self.is_healthy() else 0xFF0000
        )
        
        await send_to_channel("openclaw-health", embed)
```

---

### 3.5 PILLAR 5: Tournament Special Operations (OC-TO)

**Channel:** `#cbb-tournament`  
**Duration:** March 18 - April 7  
**Trigger:** Tournament-specific events

**Enhanced Monitoring for Tournament:**

| Round | Special Features | Channels |
|-------|-----------------|----------|
| First Four | Standard monitoring | #cbb-tournament |
| Round of 64 | Upset probability updates | #cbb-tournament |
| Round of 32 | Cinderella tracking | #cbb-tournament, #cbb-alerts |
| Sweet 16 | 2x monitoring frequency | All CBB channels |
| **Elite Eight+** | **Kimi escalation, manual review** | **#openclaw-escalations** |
| Final Four | Real-time updates | All channels |
| Championship | Maximum alert frequency | All channels |

**Tournament Features:**
```python
# Tournament-specific enhancements
tournament_config = {
    "first_four": {
        "monitoring_interval": 120,  # 2 hours
        "upset_alerts": True,
        "cinderella_tracking": False
    },
    "elite_eight": {
        "monitoring_interval": 60,   # 1 hour
        "upset_alerts": True,
        "cinderella_tracking": True,
        "force_kimi_review": True,   # All games
        "discord_threads": True      # Per-game threads
    }
}
```

**Cinderella Tracker:**
```markdown
👠 CINDERELLA WATCH — Sweet 16

🟢 ACTIVE CINDERELLAS (Seed 11+)
• 12-seed McNeese State — Upset 5-seed Clemson
  → Next: vs 4-seed Illinois (22% upset probability)
  
• 11-seed Drake — Upset 6-seed Michigan  
  → Next: vs 3-seed Kentucky (18% upset probability)

📊 HISTORICAL CONTEXT
• 12-seeds in Sweet 16: Win 32% vs 4-seeds
• Both teams trending up (3-0 ATS in tournament)
```

---

## 4. Implementation Phases

### Phase 1: Foundation (Week 1 — March 12-18)
**Goal:** Enable Morning Brief and basic telemetry

| Task | File | Effort | Owner |
|------|------|--------|-------|
| Create `openclaw_briefs.py` | NEW | 4h | Claude |
| Create telemetry dashboard | NEW | 3h | Claude |
| Add Discord webhook helpers | `discord_notifier.py` | 2h | Claude |
| Test morning brief cron | — | 2h | Kimi |
| **Deliverable:** Daily 7 AM briefs active | | | |

### Phase 2: Live Operations (Week 2 — March 19-25)
**Goal:** Live monitoring during First Four and Round of 64

| Task | File | Effort | Owner |
|------|------|--------|-------|
| Create `openclaw_live_monitor.py` | NEW | 6h | Claude |
| Implement game-time alerts | `discord_notifier.py` | 3h | Claude |
| Tournament mode config | NEW | 4h | Claude |
| Test escalation workflows | — | 3h | Kimi |
| **Deliverable:** Live monitoring active for tournament | | | |

### Phase 3: Fantasy Integration (Week 3-4 — March 26-April 7)
**Goal:** Fantasy Baseball intelligence (parallel with tournament)

| Task | File | Effort | Owner |
|------|------|--------|-------|
| Create `openclaw_closer_monitor.py` | NEW | 5h | Claude |
| Lineup confirmation alerts | NEW | 4h | Claude |
| Waiver wire sleeper detection | NEW | 4h | Claude |
| Integrate with fantasy channels | `discord_notifier.py` | 2h | Claude |
| **Deliverable:** Fantasy alerts active by Opening Day | | | |

### Phase 4: Polish & Scale (Post-Tournament — April 8+)
**Goal:** Full automation, historical analysis, 2027 prep

| Task | File | Effort | Owner |
|------|------|--------|-------|
| Historical brief archive | NEW | 4h | Claude |
| Performance analytics | NEW | 6h | Kimi |
| Self-tuning thresholds | `openclaw_lite.py` | 6h | Kimi |
| Documentation | `docs/OPENCLAW_OPERATIONS.md` | 4h | Gemini |
| **Deliverable:** Fully autonomous OpenClaw v4.0 | | | |

---

## 5. HEARTBEAT Integration

### New HEARTBEAT Entries

```yaml
# HEARTBEAT: Morning Brief Generation
Trigger: 7:00 AM ET daily
Owner: OpenClaw
Action:
  1. Generate overnight summary
  2. Query today's slate
  3. Check integrity cache
  4. Compile sharp money moves
  5. Post to #openclaw-briefs
Target: <30s generation time

---

# HEARTBEAT: Live Game Monitor
Trigger: Every 2 hours (12 PM - 11 PM ET on game days)
Owner: OpenClaw
Action:
  1. Query games within 4 hours
  2. Fresh DDGS search for each
  3. Quick integrity check
  4. Alert if verdict changed
Target: <60s for full slate check

---

# HEARTBEAT: Fantasy Baseball Monitor
Trigger: 7:00 AM ET daily + ad-hoc webhooks
Owner: OpenClaw
Action:
  1. Check closer situations
  2. Pull lineup confirmations
  3. Identify waiver sleepers
  4. Post to #fantasy-news
Target: <45s generation time

---

# HEARTBEAT: Telemetry Dashboard
Trigger: Every 5 minutes (background)
Owner: OpenClaw
Action:
  1. Collect telemetry metrics
  2. Check system health
  3. Post to #openclaw-health
  4. Alert on anomalies
Target: <5s collection + post
```

---

## 6. Discord Channel Utilization Matrix

### Current vs Target Usage

| Channel | Current | Target | Change |
|---------|---------|--------|--------|
| `#openclaw-briefs` | 0 msgs/day | 1 msg/day | NEW |
| `#openclaw-escalations` | 0-1 msgs/week | 2-5 msgs/day | +10x |
| `#openclaw-health` | 0 msgs/day | 288 msgs/day* | NEW |
| `#cbb-alerts` | Manual only | 5-10 msgs/day | +5x |
| `#fantasy-news` | 0 msgs/day | 1-2 msgs/day | NEW |
| `#fantasy-waivers` | 0 msgs/day | 2-3 msgs/week | NEW |

*Telemetry posts every 5 min = 288/day; consider hourly summary instead

### Recommended: Hourly Telemetry Summary
Instead of 288 individual posts/day, use:
```python
# On the hour, every hour
telemetry_summary = aggregate_last_hour_telemetry()
if telemetry_summary.has_anomalies():
    await send_alert("openclaw-health", telemetry_summary)
else:
    await send_status("openclaw-health", telemetry_summary)  # Quiet update
```

---

## 7. Success Metrics

### KPIs

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Intelligence messages/day | 0.1 | 15-20 | Discord message count |
| Time to alert (game-time news) | Manual | <15 min | Alert timestamp - news timestamp |
| Escalation queue resolution | 24h | <4h | Queue entry → resolution time |
| Fantasy alert accuracy | N/A | >80% | Correct predictions / total |
| System uptime visibility | Manual | 99.9% | Telemetry coverage |
| Tournament coverage | Basic | Full | Games with monitoring / total games |

### User Impact

| Stakeholder | Current Pain | Solution | Benefit |
|-------------|--------------|----------|---------|
| **Bettor** | Miss late news | Live monitor | Better informed decisions |
| **Fantasy Manager** | Manual lineup checks | 7 AM briefs | Time saved, better rosters |
| **Operator** | Manual system checks | Telemetry dashboard | Proactive issue detection |
| **Analyst** | No overnight summary | Morning brief | Faster morning workflow |
| **Risk Manager** | Delayed escalations | Real-time queue | Faster manual review |

---

## 8. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| DDGS rate limiting | Medium | Alerts delayed | Exponential backoff, reduce frequency |
| Discord API limits | Low | Messages dropped | Batch alerts, use embeds |
| False positive alerts | Medium | Alert fatigue | Tune thresholds, confidence gating |
| Information overload | Medium | Channels noisy | Tiered alerts (info vs warning vs critical) |
| System overload | Low | Performance degraded | Async architecture, caching, semaphore limits |

---

## 9. Files to Create/Modify

### New Files
```
backend/services/
├── openclaw_briefs.py          # Morning brief generation
├── openclaw_live_monitor.py    # Game-time monitoring
├── openclaw_closer_monitor.py  # Fantasy closer tracking
├── openclaw_telemetry.py       # Dashboard & health
└── openclaw_tournament.py      # Tournament special ops

dashboard/pages/
└── 13_OpenClaw_Control.py      # Manual trigger UI (optional)
```

### Modified Files
```
backend/services/
├── discord_notifier.py         # Add new channel helpers
├── openclaw_lite.py            # Integrate new modules
└── analysis.py                 # Hook live monitoring

HEARTBEAT.md                    # Add new operational loops
HANDOFF.md                      # Add mission assignments
```

---

## 10. Immediate Next Steps

### This Week (March 12-18)
1. **Claude Code:** Implement `openclaw_briefs.py` — Morning Brief (Phase 1)
2. **Claude Code:** Implement basic telemetry dashboard (Phase 1)
3. **Kimi CLI:** Test morning brief generation, tune content
4. **Gemini CLI:** Document operational procedures

### Next Week (March 19-25) — Tournament Begins
1. **Claude Code:** Implement live monitoring (Phase 2)
2. **Claude Code:** Tournament mode activation
3. **Kimi CLI:** Validate escalation workflows under load
4. **OpenClaw:** Execute autonomous monitoring

### Post-Tournament (April 8+)
1. **Claude Code:** Fantasy Baseball integration (Phase 3)
2. **Kimi CLI:** Performance analysis and threshold tuning
3. **Gemini CLI:** Full documentation of v4.0 capabilities

---

**Document Version:** OPCL-001  
**Status:** Ready for Implementation  
**Estimated Effort:** 40-50 hours over 4 weeks  
**Expected Impact:** 10x increase in OpenClaw value delivery
