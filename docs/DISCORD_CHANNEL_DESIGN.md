# Discord Channel Architecture Design

**Date:** March 11, 2026  
**Purpose:** Redesign Discord structure for CBB Edge + Fantasy Baseball platform  
**Status:** Design Phase — Ready for Implementation

---

## Executive Summary

**Current State:** Single `bets` channel handling all communications  
**Proposed State:** 10-12 channels across 5 categories with clear separation of concerns

**Design Principles:**
1. **Separation by Domain** — CBB betting vs Fantasy Baseball vs System
2. **Frequency-Based Routing** — High-frequency alerts in dedicated channels
3. **Actionability** — Channels designed around user actions (bet, set lineup, review)
4. **Signal vs Noise** — Critical alerts separated from routine updates

---

## Channel Architecture

### Category 1: 🏀 CBB EDGE (College Basketball)

| Channel | Purpose | Frequency | Example Messages |
|---------|---------|-----------|------------------|
| **cbb-bets** | Official bet recommendations | 0-5/day | "BET: Gonzaga -3.5 (1.5u) @ -110" |
| **cbb-morning-brief** | Daily slate summary | 1/day @ 9 AM | "Today's Slate: 3 bets, 2 marginal" |
| **cbb-alerts** | Line movement, sharp signals | 2-10/day | "🚨 Sharp money on Duke: -3 → -4.5" |
| **cbb-tournament** | March Madness specific | Mar 18-Apr 7 | "Elite Eight Alert: UNC vs Duke" |

**Migration:** Rename existing `bets` → `cbb-bets`

---

### Category 2: ⚾ FANTASY BASEBALL

| Channel | Purpose | Frequency | Example Messages |
|---------|---------|-----------|------------------|
| **fantasy-lineups** | Daily lineup recommendations | 1/day @ 7 AM | "Today's Optimal Lineup: C Contreras, 1B Olson..." |
| **fantasy-waivers** | Waiver wire pickups | 2-3/week | "🎯 Waiver Add: Colton Cowser (12% rostered)" |
| **fantasy-news** | Injury updates, closer changes | As needed | "⚠️ Closer Alert: Edwin Diaz (NYM) unavailable" |
| **fantasy-draft** | Draft assistant (Mar 20-23) | During draft | "Round 8 Pick: Jordan Walker (ADP 95, Tier 6)" |

**Note:** `fantasy-draft` can be archived after March 23 draft

---

### Category 3: 🎯 OPENCLAW INTELLIGENCE

| Channel | Purpose | Frequency | Example Messages |
|---------|---------|-----------|------------------|
| **openclaw-briefs** | Research summaries | 1-2/day | "Tournament Intel: Houston Cougars profile" |
| **openclaw-escalations** | High-stakes alerts | 0-3/day | "🚨 HIGH-STAKES: UNC @ Duke (2.0u) — Manual review required" |
| **openclaw-health** | System health checks | 1/day @ 6 AM | "✅ All systems operational (MAE: 8.2)" |

---

### Category 4: ⚙️ SYSTEM OPERATIONS

| Channel | Purpose | Frequency | Example Messages |
|---------|---------|-----------|------------------|
| **system-errors** | Critical failures | As needed | "❌ Error: Odds API timeout — using cached lines" |
| **system-logs** | Routine operations | 5-20/day | "✓ Settled 3 bets (2W/1L), P&L: +$12.50" |
| **data-alerts** | Data source issues | As needed | "⚠️ KenPom API degraded — BartTorvik only" |

---

### Category 5: 💬 GENERAL

| Channel | Purpose | Who Can Post |
|---------|---------|--------------|
| **general-chat** | Human discussion | Everyone |
| **admin-commands** | Bot commands (!status, !run-analysis) | Admins only |

---

## Channel Descriptions (for Discord Setup)

```
🏀 CBB EDGE
├── cbb-bets — Official bet recommendations from the model (0-5/day)
├── cbb-morning-brief — Daily 9 AM slate summary and model insights
├── cbb-alerts — Line movement, sharp money signals, injury impacts
└── cbb-tournament — March Madness specific updates (Mar 18-Apr 7)

⚾ FANTASY BASEBALL
├── fantasy-lineups — Daily optimal lineup recommendations (7 AM ET)
├── fantasy-waivers — Waiver wire pickups and drops (Tue/Fri)
├── fantasy-news — Injury updates, closer situations, weather
└── fantasy-draft — Live draft assistant (Mar 20-23 only)

🎯 OPENCLAW INTELLIGENCE
├── openclaw-briefs — Research reports and tournament intelligence
├── openclaw-escalations — High-stakes bets requiring manual review (≥1.5u)
└── openclaw-health — Daily system health and performance metrics

⚙️ SYSTEM OPERATIONS
├── system-errors — Critical system failures requiring attention
├── system-logs — Routine bet settlement and data fetch logs
└── data-alerts — Data source degradation or outages

💬 GENERAL
├── general-chat — Discussion and questions
└── admin-commands — Bot admin commands (restricted)
```

---

## Message Formatting Standards

### cbb-bets
```
🏀 BET RECOMMENDATION

**Gonzaga Bulldogs -3.5** @ -110
Bet Size: 1.5 units ($37.50)
Confidence: 72% | Edge: 4.2%

Model Notes:
• KenPom margin: +5.2
• Fatigue adj: +0.3 (rest advantage)
• Sharp signal: CONFIRMED ✓

Game: Gonzaga @ Saint Mary's — 9:00 PM ET
```

### cbb-morning-brief
```
📋 MORNING BRIEF — March 12, 2026

**Today's Slate:** 12 games analyzed
**Official Bets:** 3 recommended
**Marginal:** 2 consider

Top Opportunities:
1. Gonzaga -3.5 (1.5u) — 72% conf
2. Houston -6.5 (1.0u) — 68% conf
3. Duke +2.0 (0.75u) — 65% conf

Market Notes:
• Sharp money detected on Duke (line moved +1.5)
• 2 injury concerns flagged (see #cbb-alerts)

System Status: ✅ Healthy (MAE: 8.2)
```

### openclaw-escalations
```
🚨 HIGH-STAKES ESCALATION

Game: UNC @ Duke (Elite Eight)
Recommended: 2.0 units
Verdict: VOLATILE

Escalation Reason:
• High stakes (≥1.5u threshold)
• Tournament game (Elite Eight+)
• Integrity check: VOLATILE — late injury news

Action Required: Manual review before tipoff
Queue ID: 20260312_090000_UNC_Duke
```

### fantasy-lineups
```
⚾ TODAY'S OPTIMAL LINEUP — March 12

**Hitters:**
C: Willson Contreras (STL) vs LHP — 8.2 proj
1B: Matt Olson (ATL) vs RHP — 7.8 proj
2B: Marcus Semien (TEX) vs RHP — 7.5 proj
...

**Pitchers:**
SP: Spencer Strider (ATL) vs WSH — 22.4 proj
SP: Kevin Gausman (TOR) vs BAL — 19.8 proj
RP: Edwin Diaz (NYM) — 4.2 proj (CLOSER)

**Benched Today:**
• Ohtani (LAA) — pitching only
• Judge (NYY) — off day

Total Projected: 142.3 points
League Avg: 128.5 points
Edge: +10.7%
```

---

## Notification Routing Logic

### Current (Single Channel)
```python
# Everything goes to one channel
send_discord_message("all_bets_channel", message)
```

### Proposed (Channel Routing)
```python
# Route by message type
def route_notification(message_type, content, severity="normal"):
    routes = {
        # CBB Betting
        "bet_recommendation": "cbb-bets",
        "morning_brief": "cbb-morning-brief",
        "line_movement": "cbb-alerts",
        "sharp_signal": "cbb-alerts",
        "tournament_update": "cbb-tournament",
        
        # Fantasy Baseball
        "lineup_recommendation": "fantasy-lineups",
        "waiver_suggestion": "fantasy-waivers",
        "injury_alert": "fantasy-news",
        "draft_pick": "fantasy-draft",
        
        # OpenClaw
        "research_brief": "openclaw-briefs",
        "high_stakes_escalation": "openclaw-escalations",
        "system_health": "openclaw-health",
        
        # System
        "critical_error": "system-errors",
        "routine_log": "system-logs",
        "data_degradation": "data-alerts",
    }
    
    channel = routes.get(message_type, "general-chat")
    
    # Critical errors also ping @admin
    if severity == "critical":
        content = "@admin " + content
    
    return send_discord_message(channel, content)
```

---

## Implementation Phases

### Phase 1: Foundation (Immediate)
**Channels to Create:**
1. `cbb-bets` (migrate existing)
2. `fantasy-lineups`
3. `system-errors`
4. `general-chat`

**Code Changes:**
- Update `discord_notifier.py` with channel routing
- Add channel IDs to environment variables
- Test with 3-5 messages per channel

### Phase 2: CBB Enhancement (Before March 18)
**Channels:**
5. `cbb-morning-brief`
6. `cbb-alerts`
7. `openclaw-escalations`

**Code Changes:**
- Morning briefing automation
- Sharp money alerts
- High-stakes escalation queue integration

### Phase 3: Tournament Mode (March 18)
**Channels:**
8. `cbb-tournament`
9. `openclaw-briefs`

**Code Changes:**
- Tournament-specific formatting
- Research report delivery

### Phase 4: Fantasy Full Launch (March 20-23)
**Channels:**
10. `fantasy-waivers`
11. `fantasy-news`
12. `fantasy-draft`

**Code Changes:**
- Waiver wire analyzer
- Closer situation monitor
- Draft board integration

### Phase 5: Monitoring (Ongoing)
**Channels:**
13. `openclaw-health`
14. `system-logs`
15. `data-alerts`

---

## Environment Variable Structure

```bash
# Discord Configuration
DISCORD_BOT_TOKEN=MTQ3NzQwOTg4NTA0OTMyMzgxNA.GnEBcJ...

# Channel IDs (fill in after creation)
DISCORD_CHANNEL_CBB_BETS=1477436117426110615          # Existing
DISCORD_CHANNEL_CBB_BRIEF=0000000000000000000
DISCORD_CHANNEL_CBB_ALERTS=0000000000000000000
DISCORD_CHANNEL_CBB_TOURNAMENT=0000000000000000000

DISCORD_CHANNEL_FANTASY_LINEUPS=0000000000000000000
DISCORD_CHANNEL_FANTASY_WAIVERS=0000000000000000000
DISCORD_CHANNEL_FANTASY_NEWS=0000000000000000000
DISCORD_CHANNEL_FANTASY_DRAFT=0000000000000000000

DISCORD_CHANNEL_OPENCLAW_BRIEFS=0000000000000000000
DISCORD_CHANNEL_OPENCLAW_ESCALATIONS=0000000000000000000
DISCORD_CHANNEL_OPENCLAW_HEALTH=0000000000000000000

DISCORD_CHANNEL_SYSTEM_ERRORS=0000000000000000000
DISCORD_CHANNEL_SYSTEM_LOGS=0000000000000000000
DISCORD_CHANNEL_DATA_ALERTS=0000000000000000000

DISCORD_CHANNEL_GENERAL=0000000000000000000
```

---

## Code Updates Required

### 1. Update `discord_notifier.py`

Add channel routing:
```python
import os

CHANNEL_IDS = {
    "cbb-bets": os.getenv("DISCORD_CHANNEL_CBB_BETS"),
    "cbb-morning-brief": os.getenv("DISCORD_CHANNEL_CBB_BRIEF"),
    "cbb-alerts": os.getenv("DISCORD_CHANNEL_CBB_ALERTS"),
    # ... etc
}

def send_to_channel(channel_name: str, message: str, mention_admin: bool = False):
    """Send message to specific channel by name."""
    channel_id = CHANNEL_IDS.get(channel_name)
    if not channel_id:
        logger.warning(f"Channel {channel_name} not configured")
        return False
    
    if mention_admin:
        message = "@admin " + message
    
    return _post_to_discord(channel_id, message)
```

### 2. Update `analysis.py`

Route bet notifications:
```python
# Instead of:
send_bet_notification(bet_details)

# Use:
from backend.services.discord_notifier import route_notification

route_notification(
    message_type="bet_recommendation",
    content=format_bet_message(bet_details),
    severity="normal"
)
```

### 3. Update `openclaw_lite.py`

Route escalations:
```python
# In escalation queue
route_notification(
    message_type="high_stakes_escalation",
    content=format_escalation_message(escalation_data),
    severity="high"
)
```

---

## Permission Recommendations

| Channel | @everyone | @admin | Bot |
|---------|-----------|--------|-----|
| cbb-bets | Read | Read/Write | Write |
| fantasy-lineups | Read | Read/Write | Write |
| openclaw-escalations | Read | Read/Write | Write |
| system-errors | Read | Read/Write | Write |
| system-logs | None | Read | Write |
| admin-commands | None | Write | Read |
| general-chat | Write | Write | Read |

---

## Migration Checklist

**Preparation:**
- [ ] Create all channels listed above
- [ ] Set channel descriptions
- [ ] Configure permissions
- [ ] Copy channel IDs to environment

**Code Updates:**
- [ ] Update `discord_notifier.py` with routing
- [ ] Update `analysis.py` bet notifications
- [ ] Update `sentinel.py` health checks
- [ ] Update `openclaw_lite.py` escalations
- [ ] Update fantasy modules for lineup/waiver

**Testing:**
- [ ] Test each channel with sample message
- [ ] Verify @admin mentions work
- [ ] Check permission boundaries
- [ ] Confirm fallback to general-chat works

**Go-Live:**
- [ ] Announce new structure in general-chat
- [ ] Pin channel descriptions
- [ ] Monitor for 48 hours

---

## Success Metrics

**Week 1:**
- All channels receiving appropriate traffic
- No messages in wrong channels
- Zero permission errors

**Week 2:**
- Users able to find relevant info quickly
- Reduced noise in primary channels
- High engagement with morning briefs

**Tournament (Mar 18):**
- Tournament channel actively used
- Escalations properly flagged
- System health transparent

---

**Next Step:** Create Discord channels and provide channel IDs for configuration.
