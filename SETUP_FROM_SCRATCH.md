# CBB Edge Analyzer - Fresh Setup Guide for Windows
# Version 1.0 - Simplified, No Enhancements

## Prerequisites

Before starting, ensure you have:

1. **Python 3.11+** installed
   - Download: https://www.python.org/downloads/
   - During install: ✅ Check "Add Python to PATH"
   - Verify: `python --version` (should show 3.11 or higher)

2. **Git** installed (you already have this - Git Bash)
   - Verify: `git --version`

3. **Docker Desktop** (optional, for local database)
   - Download: https://www.docker.com/products/docker-desktop/
   - Or use Railway/Supabase for database

4. **API Keys** (get these first):
   - The Odds API: https://the-odds-api.com (free tier)
   - KenPom API: https://kenpom.com/api ($95/year)

---

## Step-by-Step Setup (15 minutes)

### Step 1: Extract and Navigate

```bash
# Extract the archive
cd ~/repos
tar -xzf cbb-edge-v1-windows.tar.gz
cd cbb-edge

# Verify files
ls
# Should see: backend/ dashboard/ scripts/ tests/ requirements.txt README.md etc.
```

### Step 2: Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate it (Git Bash)
source venv/Scripts/activate

# Verify activation
which python
# Should show: ~/repos/cbb-edge/venv/Scripts/python

# Your prompt should now show (venv) at the start
```

### Step 3: Install Dependencies

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install all requirements (takes 2-3 minutes)
python -m pip install -r requirements.txt

# Verify installation
python -m pip list | grep fastapi
# Should show: fastapi 0.109.0
```

### Step 4: Setup Database

**Option A: Docker (Recommended for Local)**

```bash
# Start PostgreSQL container
docker run -d \
  --name cbb-postgres \
  -e POSTGRES_PASSWORD=mypassword \
  -e POSTGRES_DB=cbb_edge \
  -p 5432:5432 \
  postgres:15

# Verify it's running
docker ps | grep cbb-postgres

# Your DATABASE_URL will be:
# postgresql://postgres:mypassword@localhost:5432/cbb_edge
```

**Option B: Railway (Free Tier)**

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Create project and add PostgreSQL
railway init
railway add
# Select: PostgreSQL

# Get connection string
railway variables
# Copy the DATABASE_URL value
```

**Option C: Supabase (Free Tier)**

1. Go to https://supabase.com
2. Create new project
3. Go to Settings → Database
4. Copy "Connection string" (URI format)

### Step 5: Create .env File

```bash
# Copy template
cp .env.example .env

# Edit with notepad
notepad .env
```

**Add these values** (replace with your actual keys):

```bash
# Database (choose one from Step 4)
DATABASE_URL=postgresql://postgres:mypassword@localhost:5432/cbb_edge

# API Keys (REQUIRED)
THE_ODDS_API_KEY=your_odds_api_key_here
KENPOM_API_KEY=your_kenpom_api_key_here

# Authentication (generate random 32+ char string)
API_KEY_USER1=your_random_secure_api_key_here

# App Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
NIGHTLY_CRON_HOUR=3
NIGHTLY_CRON_TIMEZONE=America/New_York

# Model Parameters (Version 1 - Simple, Proven)
BASE_SD=11.0
WEIGHT_KENPOM=1.0
WEIGHT_BARTTORVIK=0.0
WEIGHT_EVANMIYA=0.0
HOME_ADVANTAGE=3.09

# Betting Parameters
MAX_KELLY_FRACTION=0.20
FRACTIONAL_KELLY_DIVISOR=2
MAX_BANKROLL_PCT_PER_BET=1.5
STARTING_BANKROLL=1000
```

**Generate secure API key:**
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Step 6: Initialize Database

```bash
# Run initialization script
python scripts/init_db.py --seed

# Expected output:
# 🔧 Initializing CBB Edge database...
# ✅ Database tables created successfully
# 📋 Tables: games, predictions, bet_logs, model_parameters, ...
# 🌱 Seeding test data...
# ✅ Test data seeded
# 🎉 Database initialization complete!
```

**If you get connection error:**

```bash
# Docker: Check container is running
docker ps | grep cbb-postgres

# If not running:
docker start cbb-postgres

# Test connection manually
python -c "import psycopg2; conn = psycopg2.connect('YOUR_DATABASE_URL_HERE'); print('✅ Connected!')"
```

### Step 7: Run Tests

```bash
# Run test suite
python -m pytest tests/ -v

# Expected output:
# tests/test_betting_model.py::TestMonteCarloCI::test_returns_three_values PASSED
# tests/test_betting_model.py::TestMonteCarloCI::test_ci_bounds_are_valid PASSED
# ... (15 tests total)
# ==================== 15 passed in 2.3s ====================
```

### Step 8: Start Backend

```bash
# Start FastAPI server
python -m uvicorn backend.main:app --reload --port 8000

# Expected output:
# INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
# INFO:     Started reloader process
# 🚀 Starting CBB Edge Analyzer
# ⏰ Nightly cron scheduled for 3:00 America/New_York
```

**Test it in browser:** http://localhost:8000

Should show:
```json
{
  "app": "CBB Edge Analyzer",
  "version": "7.0",
  "status": "operational",
  "timestamp": "2026-02-11T..."
}
```

**Test health endpoint:** http://localhost:8000/health

```json
{
  "status": "healthy",
  "database": "connected",
  "scheduler": "running"
}
```

### Step 9: Start Dashboard (New Terminal)

**Open a new Git Bash window:**

```bash
# Navigate to project
cd ~/repos/cbb-edge

# Activate venv
source venv/Scripts/activate

# Set environment variables for dashboard
export API_URL=http://localhost:8000
export API_KEY=your_api_key_user1_from_env

# Start Streamlit
python -m streamlit run dashboard/app.py

# Expected output:
#   You can now view your Streamlit app in your browser.
#   Local URL: http://localhost:8501
#   Network URL: http://192.168.x.x:8501
```

**Visit dashboard:** http://localhost:8501

Should show:
- Dashboard header
- "Total Bets: 0"
- Performance section (empty)
- No errors

---

## Verification Checklist

After setup, verify everything works:

### ✅ 1. Backend Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "database": "connected",
  "scheduler": "running",
  "timestamp": "2026-02-11T21:30:00.000Z"
}
```

### ✅ 2. API Authentication

```bash
# Set your API key
export API_KEY=your_api_key_user1_value

# Test authenticated endpoint
curl http://localhost:8000/api/predictions/today \
  -H "X-API-Key: $API_KEY"
```

Expected (first time):
```json
{
  "date": "2026-02-11",
  "total_games": 0,
  "bets_recommended": 0,
  "predictions": []
}
```

### ✅ 3. Database Connection

```bash
# Connect to database
docker exec -it cbb-postgres psql -U postgres -d cbb_edge

# List tables
\dt

# Expected output:
# List of relations
# Schema | Name                   | Type  | Owner
# public | games                  | table | postgres
# public | predictions            | table | postgres
# public | bet_logs               | table | postgres
# ... (9 tables total)

# Query tables
SELECT COUNT(*) FROM games;
# Should return: 0 (no games yet)

# Exit
\q
```

### ✅ 4. Trigger Manual Analysis

```bash
# Set API key
export API_KEY=your_api_key_user1_value

# Trigger nightly job manually
curl -X POST http://localhost:8000/admin/run-analysis \
  -H "X-API-Key: $API_KEY"

# Expected output:
{
  "message": "Analysis job started"
}
```

**Check backend logs** - should show:
```
🌙 Starting nightly analysis
📡 Fetching odds from The Odds API...
✅ Fetched odds for X games
📊 Fetching ratings from all sources...
✅ KenPom: fetched X teams
🎯 Analyzing X games...
✅ Analysis complete in X.Xs
   Games analyzed: X
   Bets recommended: 0-2
   Errors: 0
```

### ✅ 5. Dashboard Access

1. Open http://localhost:8501
2. Click "Today's Bets" tab
3. Should show any recommended bets (likely none first run)
4. Click "Performance" tab
5. Should show "Not enough data yet"

---

## Common Issues & Fixes

### Issue: "ModuleNotFoundError: No module named 'fastapi'"

**Fix:**
```bash
# Ensure venv is activated
source venv/Scripts/activate

# Verify Python is from venv
which python
# Should show: ~/repos/cbb-edge/venv/Scripts/python

# Reinstall dependencies
python -m pip install -r requirements.txt
```

### Issue: "Database connection failed"

**Fix:**
```bash
# Check PostgreSQL is running
docker ps | grep postgres

# If not running:
docker start cbb-postgres

# If doesn't exist:
docker run -d --name cbb-postgres \
  -e POSTGRES_PASSWORD=mypassword \
  -e POSTGRES_DB=cbb_edge \
  -p 5432:5432 \
  postgres:15

# Update DATABASE_URL in .env to match password
```

### Issue: "Port 8000 already in use"

**Fix:**
```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill it (use PID from above)
taskkill /PID <pid> /F

# Or use different port
python -m uvicorn backend.main:app --reload --port 8001

# Update dashboard to use new port:
export API_URL=http://localhost:8001
```

### Issue: "The Odds API: Invalid API key"

**Fix:**
```bash
# Verify API key in .env
cat .env | grep THE_ODDS_API_KEY

# Test API key manually
curl "https://api.the-odds-api.com/v4/sports/?apiKey=YOUR_KEY"

# Should return list of sports, not error
```

### Issue: "Tests failing"

**Fix:**
```bash
# Check which tests fail
python -m pytest tests/ -v

# Common issues:
# 1. Database not running → start Docker
# 2. Modules not installed → pip install -r requirements.txt
# 3. Environment variables not set → check .env exists
```

---

## What's Configured (Version 1 - Simple)

### ✅ What You Have

**Model**: Version 7 framework with **conservative approach**
- Uses KenPom AdjEM only (proven methodology)
- No Four Factors double-counting
- No magic constants
- Clean, maintainable code

**Features**:
- ✅ 2-layer Monte Carlo confidence intervals
- ✅ Safe Kelly criterion calculation
- ✅ Vig removal from American odds
- ✅ Penalty budget with ceiling
- ✅ Conservative decision threshold
- ✅ Nightly cron scheduler (3 AM ET)
- ✅ REST API with 15 endpoints
- ✅ Streamlit dashboard
- ✅ PostgreSQL database (9 tables)
- ✅ API key authentication

**Expected Behavior**:
- PASS rate: 85-95% (most games)
- Bets per day: 0-2 (typical)
- Edge when found: 2-5% post-vig
- Max bet size: 1.5% of bankroll

### ❌ What You Don't Have (Intentionally Removed)

- ❌ Four Factors integration (double-counting risk)
- ❌ Enhanced KenPom client (complexity risk)
- ❌ BartTorvik/EvanMiya scrapers (maintenance nightmare)
- ❌ Magic SD multipliers (unvalidated)
- ❌ FanMatch validation (vanity metric)
- ❌ 10-endpoint API calls (timeout risk)

**This is by design.** Keep it simple until you validate CLV > 0.

---

## Next Steps After Setup

### Day 1: Verify Everything Works

```bash
# 1. Check backend is running
curl http://localhost:8000/health

# 2. Check dashboard loads
# Open: http://localhost:8501

# 3. Trigger manual analysis
curl -X POST http://localhost:8000/admin/run-analysis \
  -H "X-API-Key: $API_KEY"

# 4. Review predictions
curl http://localhost:8000/api/predictions/today \
  -H "X-API-Key: $API_KEY" | jq
```

### Week 1: Paper Trading

1. **Check dashboard every morning** (after 3 AM cron runs)
2. **Log every recommendation** in a spreadsheet
3. **Track closing lines** before games start
4. **Update outcomes** after games complete
5. **Calculate CLV** weekly

**Template spreadsheet columns**:
- Date
- Game
- Pick
- Odds Taken
- Model Prob
- Conservative Edge
- Recommended Units
- Closing Line
- Outcome (W/L)
- CLV (closing - taken)

### Month 1-3: Validate Edge

**Track these metrics**:
- Total bets: (goal: 100+)
- Win rate: (expect 52-58%)
- Mean CLV: (need >+0.5%)
- ROI: (expect 2-6% if valid)

**Decision rules**:
- If CLV > +0.5% after 100 bets → go live
- If CLV < -0.5% after 50 bets → STOP, recalibrate
- If CLV near 0 → need more data

### Month 4+: Live Betting (If Validated)

**Only if:**
- ✅ Mean CLV > +0.5%
- ✅ 100+ paper bets tracked
- ✅ Calibration error < 7%
- ✅ You understand why it works

**Start small:**
- Max 0.5% of bankroll per bet (not 1.5%)
- Total exposure <5% (not 30%)
- Stop loss at 10% drawdown
- Continue tracking CLV

---

## Maintenance

### Daily

- Check nightly cron ran (3 AM logs)
- Review dashboard for bets
- Log any bets placed

### Weekly

- Calculate CLV for week's bets
- Check database size
- Review error logs

### Monthly

- Performance review
- Calibration check
- Model parameter review

### Quarterly

- Recalibration (if >30 bets)
- Weight optimization (if >50 bets)
- System updates

---

## Getting Help

### Documentation

- **README.md** - Full project overview
- **QUICKSTART.md** - Deployment guide
- **IMPLEMENTATION.md** - Development roadmap
- **QUICKREF.md** - Command cheat sheet
- **WINDOWS_SETUP.md** - This file

### Troubleshooting

1. Check logs: `tail -f logs/app.log`
2. Search docs for error message
3. Verify .env configuration
4. Test database connection
5. Check API key validity

### Support

- GitHub Issues: Open an issue with logs
- Documentation: 8 files, 12,000+ words
- Code comments: Every function documented

---

## Important Reminders

### Before Betting Real Money

1. ✅ Paper trade 100 bets minimum
2. ✅ Verify CLV > +0.5%
3. ✅ Check calibration plots
4. ✅ Understand model limitations
5. ✅ Set strict stop-loss rules

### Never

- ❌ Skip paper trading
- ❌ Bet >1.5% bankroll per bet
- ❌ Ignore negative CLV
- ❌ Chase losses
- ❌ Bet when model says PASS

### Remember

- This tool PASSES 90% of games (correct!)
- Expected ROI is 2-6%, not 50%
- Variance is real (20% drawdowns normal)
- Most bettors lose - be systematic

---

## Quick Reference

### Activate Environment

```bash
cd ~/repos/cbb-edge
source venv/Scripts/activate
```

### Start Services

```bash
# Terminal 1: Backend
python -m uvicorn backend.main:app --reload

# Terminal 2: Dashboard
source venv/Scripts/activate
export API_KEY=your_key
python -m streamlit run dashboard/app.py
```

### Common Commands

```bash
# Run tests
python -m pytest tests/ -v

# Check health
curl http://localhost:8000/health

# Trigger analysis
curl -X POST http://localhost:8000/admin/run-analysis \
  -H "X-API-Key: $API_KEY"

# View predictions
curl http://localhost:8000/api/predictions/today \
  -H "X-API-Key: $API_KEY"

# Database backup
docker exec cbb-postgres pg_dump -U postgres cbb_edge > backup.sql
```

---

**Setup complete! You're ready to start paper trading.**

Good luck! 🍀
