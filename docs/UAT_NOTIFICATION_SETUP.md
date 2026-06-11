# UAT Notification Setup - Discord & WhatsApp

This guide configures UAT notifications to be sent via Discord or WhatsApp when automated tests complete.

## Overview

The UAT system supports three notification methods:
1. **Discord Webhook** (Easiest - recommended)
2. **Discord Bot** (More control, allows replies)
3. **WhatsApp** (Mobile notifications via Twilio)

You can configure multiple channels - notifications will be sent to all configured platforms.

---

## Option 1: Discord Webhook (Recommended)

### Why Webhook?
- Easiest to set up (no bot registration)
- Works immediately
- Supports rich embeds with colors
- Perfect for automated notifications

### Setup Steps

#### Step 1: Create Discord Webhook

1. Open your Discord server
2. Go to **Server Settings** → **Integrations** → **Webhooks**
3. Click **New Webhook**
4. Name it "CBB Edge UAT"
5. Select the channel (e.g., #system-alerts)
6. Click **Copy Webhook URL**

Example webhook URL:
```
https://discord.com/api/webhooks/123456789/abcdefgh123456789
```

#### Step 2: Configure Environment

Add to your `.env` file:

```bash
# Discord Webhook (Primary method)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/123456789/abcdefgh123456789
```

Or set as environment variable:
```bash
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/123456789/abcdefgh123456789"
```

#### Step 3: Test

```bash
cd /mnt/c/Users/sfgra/repos/Fixed/cbb-edge
./scripts/run_uat.sh
```

You should see a notification in Discord:

```
🎽 Elite FM Score: 8.2/10 | 📊 Quant Score: 7.4/10
[View Full Report] [View Report]
```

---

## Option 2: Discord Bot

### Why Bot?
- Can respond to commands
- More interactive features
- Better for multi-server setups
- Can DM users

### Setup Steps

#### Step 1: Create Discord Bot

1. Go to https://discord.com/developers/applications
2. Click **New Application**
3. Name it "CBB Edge UAT Bot"
4. Go to **Bot** section
5. Click **Add Bot**
6. Enable these **Privileged Gateway Intents**:
   - Message Content Intent
7. Click **Reset Token** and copy the token

#### Step 2: Invite Bot to Server

1. Go to **OAuth2** → **URL Generator**
2. Select scopes:
   - `bot`
   - `applications.commands`
3. Select bot permissions:
   - Send Messages
   - Embed Links
   - Attach Files
   - Read Message History
   - Use External Emojis
4. Copy the generated URL
5. Open URL in browser and invite to your server
6. Select the channel (e.g., #system-alerts)

#### Step 3: Get Channel ID

1. Enable Developer Mode in Discord:
   - User Settings → Advanced → Developer Mode (ON)
2. Right-click your notification channel
3. Click **Copy Channel ID**

#### Step 4: Configure Environment

Add to `.env`:

```bash
# Discord Bot (Alternative to webhook)
DISCORD_BOT_TOKEN=<your-discord-bot-token>
DISCORD_CHANNEL_ID=1477436117426110615
```

#### Step 5: Test

```bash
./scripts/run_uat.sh
```

---

## Option 3: WhatsApp (Twilio)

### Why WhatsApp?
- Mobile notifications
- Immediate alerts
- Good for on-call scenarios

### Prerequisites
- Twilio account
- WhatsApp Business API access or Sandbox

### Setup Steps

#### Step 1: Twilio Setup

1. Sign up at https://www.twilio.com/
2. Get a Twilio phone number
3. Activate WhatsApp Sandbox:
   - Console → Messaging → Try it out → Send a WhatsApp message
   - Or apply for WhatsApp Business API for production

#### Step 2: Get Credentials

From Twilio Console:
- Account SID (starts with AC...)
- Auth Token
- WhatsApp-enabled phone number

#### Step 3: Configure Environment

Add to `.env`:

```bash
# WhatsApp via Twilio
WHATSAPP_WEBHOOK_URL=https://api.twilio.com/2010-04-01/Accounts/YOUR_ACCOUNT_SID/Messages.json
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token
WHATSAPP_FROM_NUMBER=whatsapp:+14155238886  # Twilio sandbox number
WHATSAPP_TO_NUMBER=whatsapp:+1234567890      # Your phone
```

#### Step 4: Join Sandbox

Send message to Twilio sandbox number:
```
join <your-sandbox-code>
```

#### Step 5: Test

```bash
./scripts/run_uat.sh
```

---

## Configuration Reference

### Environment Variables

| Variable | Platform | Required For | Example |
|----------|----------|--------------|---------|
| `DISCORD_WEBHOOK_URL` | Discord | Webhook | `https://discord.com/api/webhooks/...` |
| `DISCORD_BOT_TOKEN` | Discord | Bot | `<your-discord-bot-token>` |
| `DISCORD_CHANNEL_ID` | Discord | Bot | `1477436117426110615` |
| `WHATSAPP_WEBHOOK_URL` | WhatsApp | Twilio | `https://api.twilio.com/...` |
| `TWILIO_ACCOUNT_SID` | WhatsApp | Twilio | `ACxxxxxxxx...` |
| `TWILIO_AUTH_TOKEN` | WhatsApp | Twilio | `your_token` |
| `WHATSAPP_FROM_NUMBER` | WhatsApp | Twilio | `whatsapp:+14155238886` |
| `WHATSAPP_TO_NUMBER` | WhatsApp | Twilio | `whatsapp:+1234567890` |

### Example .env File

```bash
# UAT Configuration
UAT_BASE_URL=https://your-app.railway.app
UAT_API_KEY=your_api_key_here

# Discord Webhook (Primary)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/123456789/abcdefgh

# OR Discord Bot (Alternative)
# DISCORD_BOT_TOKEN=<your-discord-bot-token>
# DISCORD_CHANNEL_ID=1477436117426110615

# AND/OR WhatsApp
# WHATSAPP_WEBHOOK_URL=https://api.twilio.com/2010-04-01/Accounts/ACxxx/Messages.json
# TWILIO_ACCOUNT_SID=ACxxxxxxxx
# TWILIO_AUTH_TOKEN=xxxxxxxx
# WHATSAPP_FROM_NUMBER=whatsapp:+14155238886
# WHATSAPP_TO_NUMBER=whatsapp:+1234567890
```

---

## Notification Format

### Discord Notification

**Pass (Green)**:
```
✅ UAT Report - PASS
🎽 Elite FM Score: 8.5/10 | 📊 Quant Score: 8.2/10
[View Full Report]
```

**Warning (Yellow)**:
```
⚠️ UAT Report - ACCEPTABLE
🎽 Elite FM Score: 7.2/10 | 📊 Quant Score: 6.8/10
⚠️ 2 Critical Issues
[View Full Report]
```

**Fail (Red)**:
```
❌ UAT Report - FAIL
🎽 Elite FM Score: 4.5/10 | 📊 Quant Score: 3.8/10
❌ 5 Critical Issues
[View Full Report]
```

### WhatsApp Notification

```
*CBB Edge UAT Report*

Status: PASS ✅

🎽 Elite FM Score: 8.5/10
📊 Quant Score: 8.2/10

View report: https://your-app.railway.app/reports/uat/latest_summary.md
```

---

## Advanced Configuration

### Multiple Discord Channels

You can configure different channels for different notification types:

```bash
# .env file
DISCORD_CHANNEL_SYSTEM_ALERTS=1481293221316395088  # For UAT failures
DISCORD_CHANNEL_SYSTEM_LOGS=1481294557936353521     # For all runs
DISCORD_CHANNEL_GENERAL=1481294687607455764         # Summary only
```

Update `scripts/run_uat.sh` to use specific channels based on severity.

### Conditional Notifications

Edit `scripts/run_uat.sh` to only notify on failures:

```bash
# Only send Discord notification on failure
if [ "$STATUS" == "❌ FAIL" ] || [ "$STATUS" == "⚠️ ACCEPTABLE" ]; then
    # Send notification code
fi
```

### Custom Notification Messages

Modify the message templates in:
- `scripts/uat_automation.py` (Python code)
- `scripts/run_uat.sh` (Bash script)

---

## Troubleshooting

### Discord Webhook Not Working

```bash
# Test webhook manually
curl -X POST YOUR_WEBHOOK_URL \
  -H "Content-Type: application/json" \
  -d '{"content": "Test message"}'
```

**Common Issues:**
- Webhook URL copied incorrectly
- Channel permissions prevent bot messages
- Rate limiting (max 5 messages per 5 seconds)

### Discord Bot Not Working

```bash
# Test bot token
curl -H "Authorization: Bot YOUR_BOT_TOKEN" \
  https://discord.com/api/v10/users/@me
```

**Common Issues:**
- Bot not invited to server
- Missing permissions (Send Messages, Embed Links)
- Bot token regenerated (old token invalid)
- Channel ID incorrect

### WhatsApp Not Working

**Common Issues:**
- Phone number format must include `whatsapp:` prefix
- Sandbox session expired (re-send "join" message)
- Twilio account needs billing setup
- Auth token incorrect

### No Notifications at All

1. Check environment variables are loaded:
   ```bash
   env | grep DISCORD
   env | grep WHATSAPP
   ```

2. Add debug logging to `run_uat.sh`:
   ```bash
   echo "Discord webhook: $DISCORD_WEBHOOK_URL"
   echo "WhatsApp webhook: $WHATSAPP_WEBHOOK_URL"
   ```

3. Run with verbose output:
   ```bash
   ./scripts/run_uat.sh 2>&1 | tee uat_debug.log
   ```

---

## Security Best Practices

### 1. Never Commit Secrets

Add to `.gitignore`:
```
.env
*.env
scripts/.env
```

### 2. Use Environment Variables in Production

```bash
# Railway deployment
railway variables set DISCORD_WEBHOOK_URL="https://..."
```

### 3. Rotate Tokens Regularly

- Discord: Regenerate webhook URL monthly
- Twilio: Rotate auth tokens quarterly

### 4. Limit Bot Permissions

Only grant minimum required permissions:
- Send Messages
- Embed Links
- Read Message History

---

## Testing Notifications

### Quick Test Script

Create `test_notifications.sh`:

```bash
#!/bin/bash

# Load environment
source .env

echo "Testing Discord Webhook..."
if [ -n "$DISCORD_WEBHOOK_URL" ]; then
    curl -s -X POST "$DISCORD_WEBHOOK_URL" \
        -H "Content-Type: application/json" \
        -d '{"content": "📢 Test notification from CBB Edge UAT"}'
    echo -e "\n✅ Discord webhook sent"
else
    echo "❌ DISCORD_WEBHOOK_URL not set"
fi

echo -e "\nTesting WhatsApp..."
if [ -n "$WHATSAPP_WEBHOOK_URL" ]; then
    curl -s -X POST "$WHATSAPP_WEBHOOK_URL" \
        --data-urlencode "To=$WHATSAPP_TO_NUMBER" \
        --data-urlencode "From=$WHATSAPP_FROM_NUMBER" \
        --data-urlencode "Body=Test notification from CBB Edge UAT" \
        -u "$TWILIO_ACCOUNT_SID:$TWILIO_AUTH_TOKEN"
    echo -e "\n✅ WhatsApp sent"
else
    echo "❌ WHATSAPP_WEBHOOK_URL not set"
fi
```

Run:
```bash
chmod +x test_notifications.sh
./test_notifications.sh
```

---

## Migration from Old System

If you were using `DISCORD_WEBHOOK` (old variable name):

```bash
# Old
DISCORD_WEBHOOK=https://...

# New
DISCORD_WEBHOOK_URL=https://...
```

Update your `.env` file and the system will continue working.

---

## Next Steps

1. ✅ Choose notification platform(s)
2. ✅ Complete platform-specific setup
3. ✅ Add credentials to `.env`
4. ✅ Run `./scripts/run_uat.sh` to test
5. ✅ Verify notifications received
6. → Schedule automated runs (already configured)

---

## Support

- **Discord Webhooks**: https://discord.com/developers/docs/resources/webhook
- **Discord Bot API**: https://discord.com/developers/docs/reference
- **Twilio WhatsApp**: https://www.twilio.com/docs/whatsapp/api

---

**Last Updated:** 2026-05-17  
**Version:** 1.0
