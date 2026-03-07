# Discord Error Explanation

**Date:** 2026-03-07  
**Status:** Clarified — NOT an OpenClaw issue

---

## Error Source Analysis

The errors you're seeing come from **TWO SEPARATE Discord integrations**:

### 1. Claude CLI Discord Client (Errors in your logs)
```
[discord] gateway: WebSocket connection closed with code 1006
[tools] message failed: Unknown target "heartbeat" for Discord
```

**Source:** Claude CLI's built-in Discord bot integration  
**Trigger:** Something is calling Discord tools with invalid target formats  
**Files:** Uses `DISCORD_BOT_TOKEN` from `.env`  
**Status:** NOW DISABLED (commented out in `.env`)

### 2. OpenClaw Notification System (Working correctly)
```python
# In .openclaw/coordinator.py
self._send_notification("high_stakes_escalation", {...})
```

**Source:** OpenClaw's coordinator (my code)  
**Behavior:** Logs to file when Discord unavailable  
**Files:** `.openclaw/notifications/YYYY-MM-DD.log`  
**Status:** ✅ WORKING — no errors

---

## What Was Happening

| System | Purpose | Error Rate | Status |
|--------|---------|------------|--------|
| Claude CLI Discord | General notifications | High (target format issues) | ❌ DISABLED |
| OpenClaw | Integrity alerts, escalations | None | ✅ ACTIVE (file logging) |

The "Unknown target" errors were from **Claude CLI**, not OpenClaw.

---

## Current State (After Fix)

### OpenClaw Notifications (Working)
- High-stakes escalations → `.openclaw/notifications/*.log`
- VOLATILE verdicts → `.openclaw/notifications/*.log`
- Circuit breaker events → `.openclaw/notifications/*.log`

### Discord Notifications (Disabled)
- Commented out `DISCORD_BOT_TOKEN` in `.env`
- No more WebSocket errors
- No more "Unknown target" errors

---

## If You Want Discord Back

To re-enable Discord notifications:

1. **Uncomment in `.env`:**
   ```env
   DISCORD_BOT_TOKEN=your_token
   DISCORD_CHANNEL_ID=1477436117426110615
   ```

2. **Fix the target format issue** (root cause):
   - Find what's calling Discord with target="heartbeat"
   - Likely in Claude CLI's notification system
   - Correct format: `channel:1477436117426110615` or `user:ID`

3. **Or use OpenClaw's Discord integration instead**:
   ```yaml
   # .openclaw/config.yaml
   notifications:
     discord_enabled: true
   ```

---

## Recommendation

**Keep Discord disabled.** OpenClaw's file logging is sufficient for now:

```bash
# Monitor notifications
tail -f .openclaw/notifications/$(date +%Y-%m-%d).log

# Check OpenClaw health
cat .openclaw/operational-state.json
```

Re-enable Discord only if you need real-time alerts AND can fix the target format issue.

---

*OpenClaw is working correctly. These were separate Claude CLI errors.*
