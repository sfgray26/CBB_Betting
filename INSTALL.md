# 🏀 CBB Edge Analyzer - Complete Installation Guide

**Production-Ready College Basketball Betting Framework**  
Version 7.0 - February 2026

---

## 📦 What's Included

This complete package contains everything you need to deploy a systematic, data-driven betting framework:

### Code (3,800+ lines)
- ✅ **Version 7 Betting Model** (fully implemented)
- ✅ **FastAPI Backend** (REST API + scheduler)
- ✅ **PostgreSQL Database** (9 tables)
- ✅ **Streamlit Dashboard** (monitoring UI)
- ✅ **Test Suite** (15 unit tests, 100% pass)
- ✅ **Deployment Configs** (Railway + GitHub Actions)

### Documentation
- ✅ **README.md** - Project overview
- ✅ **QUICKSTART.md** - Deployment in 15 min
- ✅ **IMPLEMENTATION.md** - Week-by-week plan
- ✅ **SUMMARY.md** - Executive summary
- ✅ **This file** - Complete installation guide

### Files Breakdown
```
20 Python files  (~3,800 lines)
6  Markdown docs (~8,500 words)
4  Config files  (Railway, GitHub Actions, env)
1  Setup script  (automated installation)
```

---

## 🚀 Three Ways to Install

### Option 1: Automated Setup (Recommended for Local)

**One command does everything:**

```bash
# Extract the archive
tar -xzf cbb-edge-complete.tar.gz
cd cbb-edge

# Run automated setup
python3 setup.py
```

This will:
1. Create virtual environment
2. Install all dependencies
3. Generate secure API keys
4. Start PostgreSQL (Docker)
5. Initialize database
6. Run test suite

**Time: 5 minutes**

---

### Option 2: Manual Setup (Full Control)

```bash
# 1. Extract and navigate
tar -xzf cbb-edge-complete.tar.gz
cd cbb-edge

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
nano .env  # Add your API keys

# 5. Start database (Docker)
docker run -d --name cbb-postgres \
  -e POSTGRES_PASSWORD=yourpass \
  -e POSTGRES_DB=cbb_edge \
  -p 5432:5432 \
  postgres:15

# 6. Initialize database
python scripts/init_db.py --seed

# 7. Run tests
pytest tests/ -v

# 8. Start backend
uvicorn backend.main:app --reload

# 9. Start dashboard (new terminal)
streamlit run dashboard/app.py
```

**Time: 10 minutes**

---

### Option 3: Deploy to Railway (Production)

**One-click cloud deployment:**

```bash
# 1. Extract and push to GitHub
tar -xzf cbb-edge-complete.tar.gz
cd cbb-edge

git init
git add .
git commit -m "Initial commit: CBB Edge Analyzer v7"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main

# 2. Deploy to Railway
npm install -g @railway/cli
railway login
railway init
railway add  # Select PostgreSQL
railway up

# 3. Set environment variables in Railway dashboard
# THE_ODDS_API_KEY, KENPOM_API_KEY, API_KEY_USER1

# 4. Initialize database (one time)
railway run python scripts/init_db.py --seed

# 5. Monitor logs
railway logs --tail 100
```

**Time: 15 minutes**  
**Cost: $15-25/mo**

---

## 🔑 Required API Keys

### 1. The Odds API (Required)
- **Free tier**: 500 requests/month
- **Paid tier**: $10-50/mo for more requests
- **Sign up**: https://the-odds-api.com
- **Purpose**: Real-time odds data

### 2. KenPom API (Required)
- **Cost**: $95/year (~$8/mo)
- **Sign up**: https://kenpom.com/api
- **Purpose**: Team efficiency ratings (core of model)

### 3. Database (Required)
- **Local**: PostgreSQL via Docker (free)
- **Cloud**: Railway ($15/mo) or Supabase (free tier)

### 4. Optional Services
- **BartTorvik**: Free (web scraping) or paid API if available
- **EvanMiya**: Free tier or paid
- **Twilio**: SMS alerts ($1/mo)
- **SendGrid**: Email alerts (free tier)

**Total minimum cost**: $8/mo (KenPom only, local hosting)  
**Recommended production**: $33-83/mo (Railway + all APIs)

---

## ✅ Verification Steps

After installation, verify everything works:

### 1. Check API Health

```bash
# If running locally
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "database": "connected",
  "scheduler": "running"
}
```

### 2. Run Tests

```bash
pytest tests/ -v

# Expected: 15 passed in ~3s
```

### 3. Check Dashboard

Visit http://localhost:8501

Should show:
- ✅ "Total Bets: 0"
- ✅ Performance section (empty)
- ✅ No errors in UI

### 4. Trigger Manual Analysis

```bash
# Get your API key from .env
export API_KEY=your_api_key_user1_value

# Trigger nightly job manually
curl -X POST http://localhost:8000/admin/run-analysis \
  -H "X-API-Key: $API_KEY"

# Check results
curl http://localhost:8000/api/predictions/today \
  -H "X-API-Key: $API_KEY" | jq
```

Should return:
- `games_analyzed: 10-50` (depending on time of day)
- `bets_recommended: 0-5` (most days 0-1)
- No fatal errors

---

## 🎯 First Week Checklist

### Day 1: Deploy & Verify
- [ ] Install locally or deploy to Railway
- [ ] Verify health endpoint returns "healthy"
- [ ] Run test suite (15 tests pass)
- [ ] Trigger manual analysis
- [ ] Check dashboard populates

### Day 2: Configure Alerts
- [ ] Add Twilio credentials (optional)
- [ ] Test SMS alert on bet recommendation
- [ ] Or: Set reminder to check dashboard daily

### Day 3: Monitor First Real Run
- [ ] Verify cron runs at 3 AM ET
- [ ] Check logs for errors
- [ ] Verify predictions stored in database

### Day 4-7: Paper Trading Setup
- [ ] Create spreadsheet for tracking
- [ ] Log every bet recommendation
- [ ] Mark as "paper trade" in bet log
- [ ] Track closing lines for CLV

---

## 📊 Expected Behavior

### Normal Operation

**Daily at 3 AM ET:**
1. Fetches odds for ~50-150 games
2. Fetches ratings from KenPom/BartTorvik/EvanMiya
3. Analyzes each game
4. Stores predictions
5. Sends alerts for bets (if any)

**Typical results:**
- Games analyzed: 50-150
- Bets recommended: 0-5 (often 0)
- **PASS rate: 85-95%** ✅ This is correct!

### What "Success" Looks Like

**After 100 paper bets (2-3 months):**
- Mean CLV > +0.5%
- Win rate: 52-58%
- ROI: 2-6%
- Calibration error < 7%

**If you see this → model is working, consider going live**

### What "Failure" Looks Like

**Red flags after 50+ bets:**
- Mean CLV < -0.5%
- Win rate < 48%
- ROI < -5%
- System recommending >10 bets/day

**If you see this → STOP, recalibrate, review model**

---

## 🐛 Troubleshooting

### "No bets recommended for days"

✅ **This is normal!** Expected behavior.

Check:
- Are games being analyzed? (Check logs)
- Are ratings fresh? (Check `DataFetch` table)
- Is model too conservative? (Review penalties in DB)

**Action**: None needed if above is OK.

---

### "Database connection failed"

**Local (Docker):**
```bash
# Check if container is running
docker ps | grep cbb-postgres

# Restart if needed
docker start cbb-postgres

# Test connection
psql postgresql://postgres:yourpass@localhost:5432/cbb_edge
```

**Railway:**
```bash
# Get connection string
railway variables

# Test connection
railway run python scripts/init_db.py --check
```

---

### "Odds API quota exceeded"

**Free tier**: 500 requests/month ≈ 16/day

**If exceeded:**
- Reduce fetch frequency (every other day)
- Upgrade to paid tier ($10/mo)
- Cache odds (implement caching in `odds.py`)

---

### "KenPom ratings not found"

**Possible causes:**
1. API key invalid → Check on kenpom.com
2. Team name mismatch → Check team naming
3. API down → Check KenPom status

**Workaround:**
- Manual entry for key games
- Use only BartTorvik temporarily
- Wait for API to recover

---

### "Scheduler not running"

**Check status:**
```bash
curl http://localhost:8000/admin/scheduler/status \
  -H "X-API-Key: $API_KEY"
```

**Restart:**
```bash
# Local
# Just restart uvicorn

# Railway
railway restart
```

---

## 📚 File Reference

### Core Files (Must Understand)

1. **`backend/betting_model.py`** (500 lines)
   - Version 7 framework implementation
   - All the math lives here

2. **`backend/services/analysis.py`** (350 lines)
   - Orchestrates nightly job
   - Ties everything together

3. **`backend/main.py`** (450 lines)
   - FastAPI application
   - API endpoints + scheduler

4. **`backend/models.py`** (400 lines)
   - Database schema
   - SQLAlchemy ORM

### Configuration Files

5. **`.env`** - All secrets and config
6. **`requirements.txt`** - Python dependencies
7. **`railway.json`** - Railway deployment
8. **`.github/workflows/deploy.yml`** - CI/CD

### Scripts

9. **`setup.py`** - Automated installation
10. **`scripts/init_db.py`** - Database setup

### Documentation

11. **`README.md`** - Project overview
12. **`QUICKSTART.md`** - Fast deployment
13. **`IMPLEMENTATION.md`** - Development plan
14. **`SUMMARY.md`** - Executive summary

---

## 🎓 Learning Path

### Week 1: Understand the Framework
- Read `README.md` fully
- Study `backend/betting_model.py`
- Run tests and understand why they pass

### Week 2: Customize & Deploy
- Adjust model parameters if needed
- Deploy to Railway
- Set up monitoring

### Weeks 3-12: Validate
- Paper trade 100 bets
- Track CLV meticulously
- Recalibrate if needed

### Month 4+: Scale (If Successful)
- Go live with 0.25% Kelly
- Gradually increase to 1.5%
- Continue quarterly recalibration

---

## 🆘 Getting Help

### 1. Check Documentation
- `README.md` - General questions
- `IMPLEMENTATION.md` - Development issues
- `QUICKSTART.md` - Deployment problems

### 2. Review Logs
```bash
# Local
tail -f logs/app.log

# Railway
railway logs --tail 100
```

### 3. Debug Mode
```bash
# Set in .env
LOG_LEVEL=DEBUG

# Restart app
```

### 4. GitHub Issues
Open an issue at: https://github.com/YOUR_USERNAME/YOUR_REPO/issues

Include:
- Error message
- Logs (last 50 lines)
- What you've tried

---

## ⚠️ Final Warnings

### Before Betting Real Money

1. ✅ **Paper trade 100 bets minimum**
2. ✅ **Verify CLV > +0.5%**
3. ✅ **Check calibration (predicted vs actual)**
4. ✅ **Understand why model works (or doesn't)**
5. ✅ **Set stop-loss at 15% bankroll**

### Never

- ❌ Skip paper trading
- ❌ Bet more than 1.5% bankroll per bet
- ❌ Chase losses
- ❌ Ignore negative CLV
- ❌ Bet when model says PASS

### Remember

- This tool finds edge 5-10% of the time
- Expected ROI is 2-6%, not 50%+
- Variance is real (20% drawdowns normal)
- Most bettors lose (be systematic, not typical)

---

## 🎉 You're Ready!

Everything you need is in this archive. The code is production-ready, tested, and documented.

**Your next command:**

```bash
python3 setup.py
```

Or follow the manual steps above.

**Good luck!** 🍀

---

*Built with Version 7 framework incorporating peer-reviewed research and real-world betting principles. Not financial advice. Past performance ≠ future results. Gamble responsibly.*

j01F3n2sSzbhi-jNAEULNkgzFqRXgOl2FuIDgKRoyfg