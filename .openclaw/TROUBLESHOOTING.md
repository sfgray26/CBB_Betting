# OpenClaw Troubleshooting Guide

**Last Updated:** 2026-03-07  
**Maintainer:** Kimi CLI (OpenClaw Config Owner)

---

## Common Issues

### Issue 1: Discord WebSocket Disconnects (Codes 1005, 1006)

**Symptoms:**
```
[discord] gateway: WebSocket connection closed with code 1005
[discord] gateway: Attempting resume with backoff: 1000ms
```

**Root Cause:**
- Code 1005 = Normal closure (no status code provided)
- Code 1006 = Abnormal closure (network issue, client/server unexpectedly closed)

These are **normal** WebSocket behaviors. Discord.js (and the Discord API) automatically reconnect with exponential backoff.

**Resolution:**
- ✅ **No action required** — auto-reconnect handles this
- If persistent (>5 min disconnected), check:
  1. Discord API status: https://discordstatus.com/
  2. Bot token validity (regenerate if needed)
  3. Network connectivity

**Configuration:**
```yaml
# .openclaw/config.yaml
notifications:
  # Set to false to disable Discord entirely (file logging only)
  discord_enabled: false
  
  # Always keep this true for operational audit trail
  log_fallback: true
```

---

### Issue 2: "Unknown target" Errors

**Symptoms:**
```
[tools] message failed: Unknown target "heartbeat" for Discord
[tools] message failed: Unknown target "discord" for Discord
Hint: <channelId|user:ID|channel:ID>
```

**Root Cause:**
External tools/framework attempting to send Discord messages with invalid target formats. OpenClaw's coordinator now handles this correctly.

**Resolution:**
- ✅ **Fixed in coordinator.py v2.1** — notifications now use proper Discord API format
- Notifications without `DISCORD_BOT_TOKEN` are logged to `.openclaw/notifications/`

**Verify Fix:**
```bash
# Check notification log
cat .openclaw/notifications/$(date +%Y-%m-%d).log

# Should see entries like:
# [2026-03-07T10:30:00Z] high_stakes_escalation: 🧠 High-stakes game flagged: UNC @ Duke (1.5u)
```

---

### Issue 3: Circuit Breaker Opens Frequently

**Symptoms:**
```
Circuit breaker OPEN - falling back to remote
```

**Root Cause:**
qwen2.5:3b failing repeatedly (Ollama not running, model not loaded, timeout).

**Resolution:**
```bash
# 1. Check Ollama status
ollama ps

# Should show:
# NAME            ID              SIZE    PROCESSOR       UNTIL
# qwen2.5:3b      ...             2.0 GB  100% GPU        Forever

# 2. If not running, start it
ollama run qwen2.5:3b &

# 3. Test with a simple query
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:3b",
  "prompt": "Say OK",
  "stream": false
}'

# 4. Reset circuit breaker (restart coordinator)
# Circuit breaker auto-resets after 60s (configurable in config.yaml)
```

**Config Tuning:**
```yaml
# .openclaw/config.yaml
circuit_breaker:
  failure_threshold: 5        # Increase if transient failures
  recovery_timeout_seconds: 60  # Decrease for faster recovery
  half_open_max_calls: 2
```

---

### Issue 4: DDGS Rate Limiting

**Symptoms:**
```
duckduckgo_search.exceptions.RateLimitException: 202
```

**Root Cause:**
Too many DDGS requests in short timeframe.

**Resolution:**
```python
# In analysis.py or openclaw_baseline.py:
# Add delay between searches
time.sleep(1.0)  # 1 second between teams

# Reduce concurrency
semaphore = asyncio.Semaphore(4)  # Was 8, reduce to 4
```

**Config:**
```yaml
# .openclaw/config.yaml
local:
  limits:
    max_concurrent: 4  # Reduce from 8 if rate limited
    timeout_seconds: 10
    retry_attempts: 3  # Increase retries
    retry_delay_seconds: 2  # Increase delay
```

---

### Issue 5: Integrity Sweep Not Producing Verdicts

**Symptoms:**
- No `integrity_verdict` field in predictions
- All return "Sanity check unavailable"

**Diagnosis:**
```python
# Check if Ollama is reachable
python -c "
import requests
resp = requests.post('http://localhost:11434/api/generate', json={
    'model': 'qwen2.5:3b',
    'prompt': 'Say test',
    'stream': False
})
print('Status:', resp.status_code)
print('Response:', resp.json().get('response', 'ERROR'))
"

# Check DDGS
python -c "
from duckduckgo_search import DDGS
with DDGS() as ddgs:
    results = list(ddgs.text('test', max_results=1))
    print('Results:', len(results))
"
```

**Resolution:**
1. **Ollama not running:** Start with `ollama run qwen2.5:3b &`
2. **Model not pulled:** `ollama pull qwen2.5:3b`
3. **DDGS not installed:** `pip install duckduckgo-search>=5.0`
4. **Network issues:** Check firewall/proxy settings

---

## Health Check Commands

```bash
# 1. Full OpenClaw diagnostics
python -c "
from .openclaw.coordinator import get_coordinator, Engine, TaskType, TaskContext
import asyncio

c = get_coordinator()
print('Coordinator Stats:', c.get_stats())
print('Circuit Breaker:', c.circuit_breaker.state)
print('Config Loaded:', bool(c.config))
"

# 2. Test integrity check flow
python -c "
import asyncio
from .openclaw.coordinator import check_integrity

async def test():
    result = await check_integrity(
        home_team='Duke',
        away_team='UNC',
        verdict='Bet 1.0u Duke -4',
        search_results='No injuries reported'
    )
    print('Result:', result)

asyncio.run(test())
"

# 3. Verify notification logging
ls -la .openclaw/notifications/
cat .openclaw/notifications/*.log

# 4. Check token usage
cat .openclaw/token-usage.jsonl | tail -10
```

---

## Log Locations

| Log | Location | Description |
|-----|----------|-------------|
| Token Usage | `.openclaw/token-usage.jsonl` | API calls, latency, costs |
| Notifications | `.openclaw/notifications/YYYY-MM-DD.log` | Discord fallback logs |
| Operational State | `.openclaw/operational-state.json` | Last checks, version |
| Sweep Telemetry | `.openclaw/sweeps/YYYY-MM-DD.jsonl` | Integrity sweep results |
| Active Task | `.openclaw/active-task.md` | Current mission status |

---

## Emergency Procedures

### Procedure 1: Complete OpenClaw Reset

```bash
# 1. Kill Ollama
pkill -f ollama

# 2. Clear circuit breaker state
# (Restart Python process using coordinator)

# 3. Restart Ollama
ollama run qwen2.5:3b &

# 4. Verify
python -c "
from .openclaw.coordinator import get_coordinator
c = get_coordinator()
print('Circuit:', c.circuit_breaker.state)
print('Stats:', c.get_stats())
"

# 5. Clear notification backlog
rm .openclaw/notifications/*.log
```

### Procedure 2: Disable Discord Notifications

```yaml
# Edit .openclaw/config.yaml
notifications:
  discord_enabled: false
  log_fallback: true
```

### Procedure 3: Switch to File-Only Logging

```python
# In coordinator.py, force fallback
notifications:
  discord_enabled: false  # Always false
  log_fallback: true      # Always true
```

---

## Contact

**OpenClaw Issues:** Kimi CLI (config owner)  
**Discord Integration:** Gemini CLI (infrastructure)  
**Core Architecture:** Claude Code (architect)

---

*This guide is updated as new issues are discovered and resolved.*
