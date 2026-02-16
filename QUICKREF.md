# CBB Edge Analyzer - Quick Reference

**One-page cheat sheet for common operations**

---

## 🚀 Installation (Choose One)

### Automated (Recommended)
```bash
python3 setup.py
```

### Manual
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your keys
python scripts/init_db.py --seed
```

### Railway (Cloud)
```bash
railway init
railway add  # PostgreSQL
railway up
railway run python scripts/init_db.py --seed
```

---

## 🎮 Running Locally

### Start Backend
```bash
source venv/bin/activate
uvicorn backend.main:app --reload
# API: http://localhost:8000
```

### Start Dashboard
```bash
source venv/bin/activate
streamlit run dashboard/app.py
# UI: http://localhost:8501
```

### Start Database (Docker)
```bash
docker run -d --name cbb-postgres \
  -e POSTGRES_PASSWORD=pass \
  -p 5432:5432 postgres:15
```

---

## 🔍 Testing & Debugging

### Run Tests
```bash
pytest tests/ -v
pytest tests/test_betting_model.py -v
```

### Check Health
```bash
curl http://localhost:8000/health
```

### View Logs
```bash
# Local
tail -f logs/app.log

# Railway
railway logs --tail 100
```

---

## 🔑 API Usage

### Set API Key
```bash
export API_KEY=your_api_key_from_env
```

### Get Today's Predictions
```bash
curl http://localhost:8000/api/predictions/today \
  -H "X-API-Key: $API_KEY"
```

### Get Recommended Bets
```bash
curl http://localhost:8000/api/predictions/bets?days=7 \
  -H "X-API-Key: $API_KEY"
```

### Trigger Analysis Manually
```bash
curl -X POST http://localhost:8000/admin/run-analysis \
  -H "X-API-Key: $API_KEY"
```

### Check Performance
```bash
curl http://localhost:8000/api/performance/summary \
  -H "X-API-Key: $API_KEY"
```

### Log a Bet
```bash
curl -X POST http://localhost:8000/api/bets/log \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": 123,
    "pick": "Duke -4.5",
    "odds_taken": -110,
    "bet_size_units": 1.0,
    "bet_size_dollars": 10
  }'
```

### Update Bet Outcome
```bash
curl -X PUT http://localhost:8000/api/bets/1/outcome \
  -H "X-API-Key: $API_KEY" \
  -d "outcome=1&closing_line=-5.5"
```

---

## 🗄️ Database

### Connect to DB
```bash
# Local (Docker)
psql postgresql://postgres:pass@localhost:5432/cbb_edge

# Railway
railway connect postgres
```

### Reset Database
```bash
python scripts/init_db.py --drop --seed
# ⚠️ WARNING: Deletes all data!
```

### Backup Database
```bash
# Local
pg_dump $DATABASE_URL > backup.sql

# Restore
psql $DATABASE_URL < backup.sql
```

---

## 🚢 Deployment

### Railway
```bash
# Login
railway login

# Deploy
railway up

# Set env vars (in dashboard)
railway open

# View logs
railway logs

# Restart
railway restart
```

### GitHub
```bash
# Push to GitHub
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/USER/REPO.git
git push -u origin main
```

---

## 📊 Monitoring

### Check Scheduler
```bash
curl http://localhost:8000/admin/scheduler/status \
  -H "X-API-Key: $API_KEY"
```

### Database Stats
```bash
# Connect to postgres
psql $DATABASE_URL

# Run queries
SELECT COUNT(*) FROM games;
SELECT COUNT(*) FROM predictions;
SELECT COUNT(*) FROM bet_logs;
SELECT verdict, COUNT(*) FROM predictions GROUP BY verdict;
```

### Performance Metrics
```bash
# Via API
curl http://localhost:8000/api/performance/summary \
  -H "X-API-Key: $API_KEY" | jq

# Via dashboard
open http://localhost:8501
```

---

## 🐛 Troubleshooting

### Restart Everything
```bash
# Stop all
docker stop cbb-postgres
pkill -f uvicorn
pkill -f streamlit

# Start all
docker start cbb-postgres
uvicorn backend.main:app --reload &
streamlit run dashboard/app.py &
```

### Clear Logs
```bash
rm -rf logs/*.log
```

### Reinstall Dependencies
```bash
pip install -r requirements.txt --upgrade
```

### Check Python Version
```bash
python --version
# Should be 3.11+
```

---

## 📁 File Locations

### Config
- `.env` - All secrets
- `railway.json` - Deploy config
- `requirements.txt` - Dependencies

### Code
- `backend/betting_model.py` - Core model
- `backend/main.py` - API app
- `backend/services/analysis.py` - Nightly job

### Docs
- `README.md` - Overview
- `QUICKSTART.md` - Deploy guide
- `INSTALL.md` - Full install
- `IMPLEMENTATION.md` - Dev roadmap

### Data
- Database (PostgreSQL)
- Logs (`logs/`)
- Cache (in-memory)

---

## 🎯 Common Workflows

### Morning Check (Daily)
```bash
# 1. Check dashboard
open http://localhost:8501

# 2. Review bets
curl http://localhost:8000/api/predictions/bets \
  -H "X-API-Key: $API_KEY"

# 3. Place bets manually at sportsbook

# 4. Log bets in system
```

### After Games Settle
```bash
# Update each bet with outcome
curl -X PUT http://localhost:8000/api/bets/{ID}/outcome \
  -H "X-API-Key: $API_KEY" \
  -d "outcome=1"  # 1=win, 0=loss
```

### Weekly Review
```bash
# Check performance
curl http://localhost:8000/api/performance/summary \
  -H "X-API-Key: $API_KEY" | jq

# Check calibration
curl http://localhost:8000/api/performance/calibration \
  -H "X-API-Key: $API_KEY" | jq

# Review in dashboard
open http://localhost:8501
```

---

## 📞 Help

### Error Messages
1. Check logs first
2. Search INSTALL.md
3. Check GitHub issues
4. Open new issue with logs

### Documentation
- `README.md` - Start here
- `INSTALL.md` - Installation issues
- `IMPLEMENTATION.md` - Development
- `QUICKSTART.md` - Deployment

---

## 🔐 Security

### Generate New API Key
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Update .env
```bash
nano .env
# Change API_KEY_USER1
# Restart app
```

### Rotate Database Password
```bash
# Railway
railway variables set DATABASE_URL=new_url

# Local
# Update .env and restart postgres
```

---

## 💡 Tips

- 🌙 Nightly job runs at 3 AM ET (configurable)
- 📊 Most days: 0-2 bets (85-95% PASS rate is normal)
- 💰 Max bet size: 1.5% of bankroll
- 📈 Check CLV weekly (must be positive)
- 🛑 Stop if CLV < -0.5% over 50 bets

---

**More help?** See full documentation in `INSTALL.md`
