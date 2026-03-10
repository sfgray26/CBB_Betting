# Discord + Railway Troubleshooting Guide

> **Issue:** "Sync Error: No module named 'sqlalchemy'"  
> **Date:** March 10, 2026  
> **Status:** Investigation + Fix Deployed

---

## Problem Summary

Discord notifications stopped working with error:
```
Sync Error: No module named 'sqlalchemy'
```

This indicates the Railway deployment is failing to install Python dependencies.

---

## Root Cause Analysis

### Possible Causes

1. **Nixpacks cache issue** — Railway's build system may have cached an incomplete environment
2. **Missing requirements.txt** in build context
3. **Pip install failure** during build (silent failure)
4. **Python version mismatch** — Build using different Python than expected

### Files Involved

| File | Purpose | Status |
|------|---------|--------|
| `requirements.txt` | Python dependencies | ✅ Present |
| `railway.json` | Railway config (legacy) | ✅ Present |
| `railway.toml` | Railway config (new) | ✅ **ADDED** |
| `Procfile` | Process definition | ❌ Missing (optional) |

---

## Fix Applied

### 1. Created `railway.toml` (Explicit Build Config)

```toml
[build]
builder = "nixpacks"

[build.nixpacks]
python_version = "3.11"

[build.nixpacks.phases.setup]
nixPkgs = ["python311", "python311Packages.pip", "gcc"]

[build.nixpacks.phases.install]
cmds = [
  "pip install --upgrade pip setuptools wheel",
  "pip install -r requirements.txt"
]
```

**Why:** Explicit install commands ensure dependencies are installed correctly.

### 2. Created `scripts/preflight_check.py`

Pre-flight checks run before app startup:
- Verifies SQLAlchemy, FastAPI, NumPy, etc. are installed
- Checks environment variables
- Fails fast with clear error messages

### 3. Updated Start Command

```toml
startCommand = "python scripts/preflight_check.py && uvicorn backend.main:app --host 0.0.0.0 --port $PORT"
```

**Why:** If dependencies are missing, the check fails before Uvicorn starts, giving clearer logs.

---

## Deployment Steps

To apply the fix:

```bash
# 1. Push the changes
git add railway.toml scripts/preflight_check.py
git commit -m "fix(railway): Add explicit build config and preflight checks"
git push origin main

# 2. Redeploy on Railway
# Railway will auto-deploy on push, or:
railway up

# 3. Monitor logs
railway logs
```

---

## Verification

After deployment, check Railway logs for:

```
🔍 Checking critical dependencies...
   ✅ SQLAlchemy
   ✅ FastAPI
   ✅ Uvicorn
   ...
✅ All critical dependencies present!
🚀 Pre-flight checks passed. Starting application...
```

If you see this, the fix worked.

---

## Discord-Specific Issues

### Issue: No Discord messages received

**Check 1:** Is `DISCORD_BOT_TOKEN` set in Railway environment?
```bash
railway variables
# Should show: DISCORD_BOT_TOKEN=your_token_here
```

**Check 2:** Is the bot in your server?
- Go to Discord Developer Portal → Applications → Your Bot → OAuth2
- Generate invite URL with `bot` scope and `Send Messages` permission
- Join bot to your server

**Check 3:** Test manually
```bash
python -c "
import os
os.environ['DISCORD_BOT_TOKEN'] = 'your_token'
os.environ['DISCORD_CHANNEL_ID'] = '1477436117426110615'
from backend.services.discord_notifier import send_todays_bets
send_todays_bets(None, {'games_analyzed': 10, 'bets_recommended': 2, 'games_considered': 3})
"
```

### Issue: "Unknown target" errors (OLD - FIXED)

These were from **Claude CLI's** Discord integration, not the app. Fixed by disabling in `.env`.

---

## Monitoring Commands

```bash
# View Railway logs
railway logs

# Check app health
railway status

# Verify environment variables
railway variables

# SSH into running container (debug)
railway connect

# Restart deployment
railway up
```

---

## Rollback Plan

If the fix causes issues:

```bash
# Revert to previous commit
git revert HEAD
git push

# Or manually restore railway.json and remove railway.toml
git checkout HEAD~1 -- railway.json
git rm railway.toml
git commit -m "Revert: Railway config changes"
git push
```

---

## Related Files

- `backend/services/discord_notifier.py` — Discord integration
- `.claude/DISCORD_ERRORS_EXPLAINED.md` — Previous Discord error analysis
- `.github/workflows/deploy.yml` — CI/CD pipeline

---

## Next Steps

1. ✅ Deploy fix to Railway
2. ⏳ Monitor logs for successful dependency install
3. ⏳ Test Discord notification manually
4. ⏳ Verify nightly cron job runs successfully

---

*Document created to resolve March 10, 2026 deployment issue.*
