# Discord Notification Setup (Simplified)

## Overview

The Discord notification system has been **simplified** to 3 essential channels:

1. **cbb-bets** — Morning briefs, live bet alerts, tournament updates
2. **cbb-results** — End-of-day results, weekly summaries, P&L tracking  
3. **cbb-alerts** — System issues, urgent notifications

Fantasy baseball channels have been removed (on pause until April 7).

---

## Required Environment Variables

### Essential
```bash
DISCORD_BOT_TOKEN=your_bot_token_here
DISCORD_CHANNEL_CBB_BETS=1234567890123456789
DISCORD_CHANNEL_CBB_RESULTS=1234567890123456789
DISCORD_CHANNEL_CBB_ALERTS=1234567890123456789
```

### Optional
```bash
DASHBOARD_URL=https://your-app.railway.app  # For links in Discord messages
```

---

## How to Get Channel IDs

1. Enable Developer Mode in Discord:
   - User Settings → Advanced → Developer Mode: ON

2. Right-click your channel → "Copy Channel ID"

---

## What Gets Sent to Each Channel

### cbb-bets
- 🌅 **Morning Brief** (7 AM ET daily) — Today's recommended bets
- ⚡ **Live Bet Alerts** — When lines move favorably
- 🏀 **Tournament Updates** — Upsets, Cinderella teams, bracket progress

### cbb-results
- 📊 **End-of-Day Results** (11 PM ET daily) — Win/loss record, P&L
- 📈 **Weekly Summary** (Sunday) — Week recap, trends, best/worst bets
- 💰 **Season P&L** — Running total

### cbb-alerts
- ⚠️ **System Warnings** — Data delays, API issues
- 🚨 **Errors** — Critical failures requiring attention
- ℹ️ **Info** — Scheduled maintenance, model updates

---

## Testing

### Via API
```bash
# Test the simplified Discord system
curl -X POST https://your-app.railway.app/admin/discord/test-simple \
  -H "X-API-Key: your_admin_key"
```

### Via Discord
Check each channel for the 🧪 TEST message.

---

## Troubleshooting

### No messages received
1. Check `DISCORD_BOT_TOKEN` is set correctly
2. Verify bot is a member of your server
3. Check bot has "Send Messages" permission in each channel
4. Check Railway logs: `railway logs`

### Messages not formatted correctly
- Ensure you're using the simplified system (`discord_simple.py`)
- Legacy `discord_notifier.py` channels may still exist but are deprecated

---

## Migration from Old System

If you were using the old 14-channel system:

1. **Pick 3 channels** you actually want to use
2. **Update env vars** to the 3 new ones above
3. **Remove old channel IDs** (optional — they'll just be ignored)
4. **Test** with `/admin/discord/test-simple`

---

## Scheduled Jobs

| Job | Time | Channel | Description |
|-----|------|---------|-------------|
| Morning Brief | 7:00 AM ET | cbb-bets | Today's picks |
| End of Day | 11:00 PM ET | cbb-results | Daily results |
| Weekly Summary | Sunday 11:30 PM ET | cbb-results | Week recap |
| System Alerts | As needed | cbb-alerts | Issues/errors |

---

## Manual Triggers

```bash
# Send morning brief now
curl -X POST https://your-app.railway.app/admin/discord/send-todays-bets \
  -H "X-API-Key: your_admin_key"

# Test Discord connection
curl -X POST https://your-app.railway.app/admin/discord/test-simple \
  -H "X-API-Key: your_admin_key"
```
