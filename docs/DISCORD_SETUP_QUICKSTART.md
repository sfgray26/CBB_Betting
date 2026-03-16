# Discord Channel Setup — Quickstart Guide

**Purpose:** Step-by-step instructions for setting up the new Discord channel structure

---

## Step 1: Create Categories

In your Discord server, create these **Categories** (right-click server name → Create Category):

1. `🏀 CBB EDGE`
2. `⚾ FANTASY BASEBALL`
3. `🎯 OPENCLAW INTEL`
4. `⚙️ SYSTEM OPS`
5. `💬 GENERAL`

---

## Step 2: Create Channels

Under each category, create channels with these **exact names**:

### 🏀 CBB EDGE
- `cbb-bets` (Text Channel)
- `cbb-morning-brief` (Text Channel)
- `cbb-alerts` (Text Channel)
- `cbb-tournament` (Text Channel)

### ⚾ FANTASY BASEBALL
- `fantasy-lineups` (Text Channel)
- `fantasy-waivers` (Text Channel)
- `fantasy-news` (Text Channel)
- `fantasy-draft` (Text Channel) — *temporary, archive after March 23*

### 🎯 OPENCLAW INTEL
- `openclaw-briefs` (Text Channel)
- `openclaw-escalations` (Text Channel)
- `openclaw-health` (Text Channel)

### ⚙️ SYSTEM OPS
- `system-errors` (Text Channel)
- `system-logs` (Text Channel)
- `data-alerts` (Text Channel)

### 💬 GENERAL
- `general-chat` (Text Channel)
- `admin-commands` (Text Channel) — *restrict to admin role*

---

## Step 3: Set Channel Descriptions

Click gear icon (⚙️) next to each channel → Channel Topic:

| Channel | Description |
|---------|-------------|
| cbb-bets | Official bet recommendations from the CBB Edge model (0-5/day) |
| cbb-morning-brief | Daily 9 AM slate summary and model insights |
| cbb-alerts | Line movement, sharp money signals, injury impacts |
| cbb-tournament | March Madness specific updates (Mar 18-Apr 7) |
| fantasy-lineups | Daily optimal lineup recommendations (7 AM ET) |
| fantasy-waivers | Waiver wire pickups and drops (Tue/Fri) |
| fantasy-news | Injury updates, closer situations, weather |
| fantasy-draft | Live draft assistant (Mar 20-23 only) |
| openclaw-briefs | Research reports and tournament intelligence |
| openclaw-escalations | High-stakes bets requiring manual review (≥1.5u) |
| openclaw-health | Daily system health and performance metrics |
| system-errors | Critical system failures requiring attention |
| system-logs | Routine bet settlement and data fetch logs |
| data-alerts | Data source degradation or outages |
| general-chat | Discussion and questions |
| admin-commands | Bot admin commands (admin only) |

---

## Step 4: Configure Permissions

### For `admin-commands` channel:
1. Go to channel settings → Permissions
2. Click `+` to add role
3. Select `@everyone` → Set "Send Messages" to ❌ DENY
4. Click `+` again
5. Select your admin role → Set "Send Messages" to ✅ ALLOW

### For `system-logs` channel (optional — reduce noise):
1. Permissions → `@everyone`
2. Set "View Channel" to ❌ DENY
3. Add admin role → "View Channel" ✅ ALLOW
4. Bot needs "View Channel" ✅ ALLOW

---

## Step 5: Get Channel IDs

### Method 1: Discord Developer Mode (Recommended)

1. User Settings (gear icon) → Advanced → Enable **Developer Mode**
2. Right-click each channel → "Copy Channel ID"
3. Paste into the table below

### Method 2: From Discord URL

Discord web URLs look like:
```
https://discord.com/channels/123456789/1477436117426110615
                                      └───────────────────┘
                                           Channel ID
```

---

## Step 6: Fill in Channel IDs

Copy this table and fill in your channel IDs:

```
DISCORD_CHANNEL_CBB_BETS=1477436117426110615          <- Your existing channel
DISCORD_CHANNEL_CBB_BRIEF=____________________
DISCORD_CHANNEL_CBB_ALERTS=____________________
DISCORD_CHANNEL_CBB_TOURNAMENT=____________________

DISCORD_CHANNEL_FANTASY_LINEUPS=____________________
DISCORD_CHANNEL_FANTASY_WAIVERS=____________________
DISCORD_CHANNEL_FANTASY_NEWS=____________________
DISCORD_CHANNEL_FANTASY_DRAFT=____________________

DISCORD_CHANNEL_OPENCLAW_BRIEFS=____________________
DISCORD_CHANNEL_OPENCLAW_ESCALATIONS=____________________
DISCORD_CHANNEL_OPENCLAW_HEALTH=____________________

DISCORD_CHANNEL_SYSTEM_ERRORS=____________________
DISCORD_CHANNEL_SYSTEM_LOGS=____________________
DISCORD_CHANNEL_DATA_ALERTS=____________________

DISCORD_CHANNEL_GENERAL=____________________
```

---

## Step 7: Update Environment

Add the filled-in values to your `.env` file and Railway environment variables.

---

## Step 8: Notify Me

Once complete, send me:
1. The filled-in channel ID list
2. Your Discord server ID (right-click server name → Copy Server ID)

I'll update the code and deploy the new routing!

---

## Quick Reference Card

```
🏀 CBB EDGE                    ⚾ FANTASY BASEBALL
├── cbb-bets                   ├── fantasy-lineups
├── cbb-morning-brief          ├── fantasy-waivers
├── cbb-alerts                 ├── fantasy-news
└── cbb-tournament             └── fantasy-draft

🎯 OPENCLAW INTEL              ⚙️ SYSTEM OPS
├── openclaw-briefs            ├── system-errors
├── openclaw-escalations       ├── system-logs
└── openclaw-health            └── data-alerts

💬 GENERAL
├── general-chat
└── admin-commands [admin only]
```

---

**Questions?** Ask in #general-chat or check full design doc: `docs/DISCORD_CHANNEL_DESIGN.md`
