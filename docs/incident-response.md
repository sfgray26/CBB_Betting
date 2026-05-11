# Incident Response Runbook

**Version:** 1.0
**Last Updated:** 2026-05-11
**Platform:** Railway (fantasy-app-production-5079.up.railway.app)

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `railway status` | Check service status |
| `railway logs --service` | View live logs |
| `python scripts/uptime_check.py` | Run health check locally |
| `GET /health/pipeline` | Pipeline health status |
| `GET /health/db` | Database connection status |

---

## Section 1: Database Connection Issues

### Symptoms
- `/health/db` returns 503
- "Database connection failed" errors in logs
- Timeouts on DB queries

### Diagnosis

```bash
# Check DB connection string
railway variables

# Test connection from production
railway run python -c "from backend.models import SessionLocal; db = SessionLocal(); db.execute('SELECT 1'); print('OK')"

# Check connection pool status
railway logs | grep -i "pool\|connection"
```

### Resolution Steps

1. **Verify DATABASE_URL is correct**
   ```bash
   railway variables get DATABASE_URL
   ```

2. **Check if Railway PostgreSQL service is running**
   - Go to Railway dashboard → PostgreSQL service
   - Verify status is "Running"
   - Check "Suspended" state (auto-suspends after 1 hour of inactivity)

3. **Connection pool exhaustion**
   - Symptoms: "pool exhausted" errors
   - Fix: Increase `pool_size` in backend/models.py or restart service
   ```bash
   railway up --service
   ```

4. **Schema drift**
   - Run pending migrations:
   ```bash
   railway run alembic upgrade head
   ```

### Escalation
- If DB service is down: Railway support ticket
- If schema corruption: Restore from latest snapshot

---

## Section 2: Scheduler Failures

### Symptoms
- `/health` returns `"scheduler": "stopped"`
- Jobs not running (stale data in `/health/pipeline`)
- APScheduler not starting

### Diagnosis

```bash
# Check scheduler state in logs
railway logs | grep -i "scheduler\|apscheduler"

# Verify ENABLE_INGESTION_ORCHESTRATOR
railway variables get ENABLE_INGESTION_ORCHESTRATOR
```

### Resolution Steps

1. **Check startup sequence**
   - Scheduler starts in `@asynccontextmanager` lifespan
   - If crash before scheduler init: check syntax errors in main.py

2. **Verify scheduler configuration**
   ```python
   # Should see in logs:
   # "APScheduler started"
   # "Ingestion Orchestrator enabled"
   ```

3. **Restart service**
   ```bash
   railway up --service
   ```

4. **Check for stuck jobs**
   ```bash
   # Query advisory locks
   railway run python -c "
   from backend.models import SessionLocal
   from sqlalchemy import text
   db = SessionLocal()
   result = db.execute(text('SELECT locktype, classid, objid, pid, granted FROM pg_locks WHERE locktype = \\\"advisory\\\"'))
   for row in result:
       print(row)
   "
   ```

5. **Kill stuck jobs if needed**
   - See `scripts/_kill_locks.py`

### Prevention
- Add alerting for "scheduler stopped" state
- Monitor `/health` every 5 minutes

---

## Section 3: Yahoo API Auth Expiry

### Symptoms
- 401 Unauthorized on Yahoo endpoints
- "invalid_token" errors
- Roster/matchup data not updating

### Diagnosis

```bash
# Check Yahoo token status
railway run python -c "
from backend.fantasy_baseball.yahoo_client_resilient import get_yahoo_client
client = get_yahoo_client()
print(f'Token valid: {client.is_token_valid()}')
print(f'Expires at: {client.token_expires_at}')
"
```

### Resolution Steps

1. **Refresh token manually**
   ```bash
   # Use the admin token refresh endpoint (if available)
   curl -X POST https://fantasy-app-production-5079.up.railway.app/admin/yahoo/refresh-token
   ```

2. **Re-authenticate if refresh fails**
   - Go to Yahoo OAuth flow
   - Update YAHOO_REFRESH_TOKEN in Railway variables

3. **Check token expiry time**
   - Tokens typically expire after 1 hour
   - Auto-refresh should happen before expiry

### Prevention
- Implement proactive token refresh (30 min before expiry)
- Monitor `/api/fantasy/roster` for 401s

---

## Section 4: Pipeline Starvation

### Symptoms
- `/health/pipeline` shows "stale" for critical jobs
- No recent entries in `data_ingestion_logs`
- Jobs stuck in "RUNNING" status

### Diagnosis

```bash
# Check recent logs
railway logs --tail 100

# Query ingestion logs
railway run python -c "
from backend.models import SessionLocal, DataIngestionLog
from datetime import datetime, timedelta
db = SessionLocal()
cutoff = datetime.now() - timedelta(hours=4)
logs = db.query(DataIngestionLog).filter(
    DataIngestionLog.started_at >= cutoff
).order_by(DataIngestionLog.started_at.desc()).limit(20).all()
for log in logs:
    print(f'{log.job_type}: {log.status} at {log.started_at}')
"
```

### Resolution Steps

1. **Check for stuck advisory locks**
   ```bash
   railway run python scripts/_kill_locks.py
   ```

2. **Manually trigger stuck jobs**
   ```bash
   railway run python -c "
   from backend.services.daily_ingestion import DailyIngestionOrchestrator
   orchestrator = DailyIngestionOrchestrator()
   await orchestrator.run_job('mlb_game_log')
   "
   ```

3. **Check scheduler is running**
   - See Section 2

4. **Verify job configuration**
   - Check cron expressions in main.py
   - Verify job IDs are unique

### Prevention
- Implement timeout for long-running jobs
- Alert on "stale" status in critical chain

---

## Section 5: Rollback Procedure

### When to Rollback
- Critical bug in production
- Data corruption from new code
- Performance degradation

### Pre-Rollback Checklist
1. Identify last known good commit
2. Backup current database state (if schema changes)
3. Notify users of planned rollback

### Rollback Steps

```bash
# 1. Checkout last known good commit
git log --oneline -10  # Find the commit
git checkout <commit-sha>

# 2. Deploy to Railway
railway up

# 3. Verify health
python scripts/uptime_check.py --url https://fantasy-app-production-5079.up.railway.app

# 4. Monitor logs
railway logs --tail 50
```

### Post-Rollback
1. Verify `/health` returns healthy
2. Check `/health/pipeline` for stale jobs
3. Monitor for 15 minutes before considering stable
4. Create hotfix branch from rollback commit

### Database Rollback (if needed)
```bash
# Identify migration to rollback
railway run alembic history

# Rollback to specific version
railway run alembic downgrade <revision>
```

---

## Section 6: Emergency Contacts

| Role | Contact |
|------|---------|
| On-Call Engineer | [TO BE DEFINED] |
| Railway Support | https://railway.app/support |
| Yahoo API Support | https://developer.yahoo.com/forum/ |

---

## Section 7: Monitoring Setup

### UptimeRobot Configuration
- **URL:** `https://fantasy-app-production-5079.up.railway.app/health`
- **Check Interval:** 5 minutes
- **Alert Threshold:** 2 consecutive failures
- **Notification:** Email + Discord webhook

### Cron Job (Alternative)
```bash
# Add to crontab: */5 * * * * /path/to/venv/python /path/to/scripts/uptime_check.py
```

### Discord Alerts
Set up webhook to post alerts when:
- `/health/pipeline` returns 503
- `/health/db` returns non-200
- Scheduler status changes to "stopped"

---

## Appendix: Advisory Lock Reference

| Lock ID | Job Name | Threshold |
|---------|----------|-----------|
| 100_001 | mlb_odds | 4h |
| 100_002 | statcast | 4h |
| 100_003 | rolling_z | 4h |
| 100_004 | cbb_ratings | 26h |
| 100_033 | bdl_injuries | 2h |
| 100_032 | savant_ingestion | 2h |

See `CLAUDE.md` for complete list.
