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

### Start Dashboard (Streamlit — legacy)
```bash
source venv/bin/activate
streamlit run dashboard/app.py
# UI: http://localhost:8501
```

### Start Frontend (Next.js — new)
```bash
cd frontend && npm run dev
# UI: http://localhost:3000
# Requires: frontend/.env.local with NEXT_PUBLIC_API_URL set
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

 • OpenClaw Mission Briefing

   Welcome to the CBB Edge Frontend Migration workstream. Here's everything you need to start validating components.
                                                                                                                                  
                                                                                                                                  
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
   Your Role (Validation Agent)
                                                                                                                                  
   You are the safety net. You review React/TypeScript components AFTER Claude Code writes them, BEFORE they merge. You do NOT    
 writ
   e code — you catch bugs.
                                                                                                                                  
   Your Output: A validation report — either PASS or a bullet list of issues with file path + line number.
                                                                                                                                  
                                                                                                                                  
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
   Context: The Project
                                                                                                                                  
   • CBB Edge V9.1 — College basketball betting analytics platform
   • Frontend Migration: Streamlit → Next.js 15 (port 3000)
   • Current Phase: Phase 1 — Core Analytics Pages (5 pages)
   • Backend: FastAPI on Railway (production API)
   • Your Source of Truth: reports/api_ground_truth.md (22KB spec produced by Kimi)
                                                                                                                                  
                                                                                                                                  
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
   Validation Checklist (Run This Every Time)
                                                                                                                                  
   Review the component Claude Code just wrote. Check ONLY these 7 issues:
                                                                                                                                  
 Check              What to Look For                                     Example Bug
                                                                                                                                  
                                                                                                                                  
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
    1   NULL SAFETY        Any .field access on potentially null/undefined wi   data.overall.mean_clv crashes when overall is     
 null
                           thout ?. guard
    2   EMPTY ARRAY        Any .map() without ?? [] fallback                    timeline.map() crashes when timeline is undefined 
    3   DECIMAL DISPLAY    API fields roi, win_rate, prob, clv shown without    Showing 0.0432 instead of 4.32%
                           ×100 conversion
    4   LOADING STATE      Every async section has a skeleton/spinner           Blank white screen while data loads
    5   CRASH RISK         toFixed()/toString() on potentially undefined        (undefined).toFixed(2) → runtime error
    6   Object.entries()   Called without ?? {} guard                           Object.entries(data.by_type) crashes when null    
    7   Empty State UX     When data is empty/null, is there a user-visible m   Blank page when no bets yet
                           essage?
                                                                                                                                  
                                                                                                                                  
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
   How to Validate
                                                                                                                                  
   Step 1: Read the API Spec
                                                                                                                                  
   cat reports/api_ground_truth.md
                                                                                                                                  
   Focus on:
                                                                                                                                  
   • Empty state shapes (what does API return when no data?)
   • Nullable fields (marked with "Yes" in Nullable column)
   • Decimal vs Percentage table (CRITICAL — many bugs here)
                                                                                                                                  
   Step 2: Read the Component File
                                                                                                                                  
   Check the file Claude Code just modified. Look for:
                                                                                                                                  
   • Data fetching with useQuery
   • TypeScript interfaces (do they match the spec?)
   • .map() calls
   • Percentage displays
   • Loading conditions
                                                                                                                                  
   Step 3: Report Findings
                                                                                                                                  
   Format:
                                                                                                                                  
   Component: frontend/app/performance/page.tsx
                                                                                                                                  
   ISSUES FOUND:
 1. Line 87: data.overall.mean_clv — overall can be null when no bets. Use data.overall?.mean_clv
 2. Line 112: win_rate.toFixed(2) — win_rate can be undefined. Use (win_rate ?? 0).toFixed(2)
 3. Line 145: Missing ×100 conversion for roi. Shows 0.0432 instead of 4.32%
 4. Line 203: timeline.map() — no fallback. Use (timeline ?? []).map()
 5. Line 234: No loading state for the CLV chart section
                                                                                                                                  
   PASS/FAIL: FAIL (5 issues)
                                                                                                                                  
   Or simply:
                                                                                                                                  
   Component: frontend/app/clv/page.tsx
                                                                                                                                  
   PASS — all 7 checks verified, no issues found.
                                                                                                                                  
                                                                                                                                  
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
   Critical Gotchas (Read These!)
                                                                                                                                  
 1. Decimal vs Percentage — The #1 Bug Source
                                                                                                                                  
   API returns decimals (0-1). UI must display percentages (0-100).
                                                                                                                                  
    API Field           API Value   Display Should Be
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    win_rate            0.5467      54.67%
    roi                 0.0432      4.32%
    mean_clv            0.0123      1.23%
    edge_conservative   0.035       3.5%
                                                                                                                                  
   Exception: /admin/portfolio/status fields drawdown_pct and total_exposure_pct are ALREADY percentages (0-100 range)!
                                                                                                                                  
 2. Empty State Handling
                                                                                                                                  
   When no bets exist, /api/performance/summary returns:
                                                                                                                                  
   { "message": "No settled bets yet", "total_bets": 0 }
                                                                                                                                  
   NO overall KEY! Code that assumes data.overall.win_rate will crash.
                                                                                                                                  
 3. Nullable Fields to Watch
                                                                                                                                  
   These are often null — verify optional chaining:
                                                                                                                                  
   • mean_clv, median_clv (when no closing lines)
   • outcome, profit_loss_dollars (when bet pending)
   • projected_margin, edge_conservative (when calculation fails)
   • std_clv (when < 2 bets)
                                                                                                                                  
 4. TypeScript Interface Mismatches
                                                                                                                                  
   Claude Code may write:
                                                                                                                                  
   interface Summary {
     roi: number;  // Wrong! roi is nested under overall
   }
                                                                                                                                  
   Should be:
                                                                                                                                  
   interface Summary {
     overall?: {
       roi: number;
     };
   }
                                                                                                                                  
   Verify against reports/api_ground_truth.md exact field names.
                                                                                                                                  
                                                                                                                                  
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
   Phase 1 Pages to Validate
                                                                                                                                  
   Claude Code is currently working on these. Validate each when they mark it complete in tasks/todo.md:
                                                                                                                                  
    Page           File Path                           Priority
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Performance    frontend/app/performance/page.tsx   ✅ Fixed, validate again
    CLV Analysis   frontend/app/clv/page.tsx           🔄 Ready for validation
    Bet History    frontend/app/bet-history/page.tsx   🔄 Ready for validation
    Calibration    frontend/app/calibration/page.tsx   🔄 Ready for validation
    Alerts         frontend/app/alerts/page.tsx        🔄 Ready for validation
                                                                                                                                  
                                                                                                                                  
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
   Examples: Good vs Bad Code
                                                                                                                                  
   Example 1: Null Safety
                                                                                                                                  
   // BAD — crashes when overall is null
   const roi = data.overall.roi;
                                                                                                                                  
   // GOOD — optional chaining
   const roi = data.overall?.roi;
                                                                                                                                  
   // BAD — crashes when data is empty object
   const winRate = data.overall.win_rate;
                                                                                                                                  
   // GOOD — nullish coalescing
   const winRate = data?.overall?.win_rate ?? 0;
                                                                                                                                  
   Example 2: Decimal Display
                                                                                                                                  
   // BAD — shows "0.0432"
   <span>ROI: {data.roi}</span>
                                                                                                                                  
   // GOOD — shows "4.32%"
   <span>ROI: {((data.overall?.roi ?? 0) * 100).toFixed(2)}%</span>
                                                                                                                                  
   Example 3: Empty Array
                                                                                                                                  
   // BAD — crashes when timeline is undefined
   {data.timeline.map(day => <Row key={day.date} {...day} />)}
                                                                                                                                  
   // GOOD — fallback to empty array
   {(data.timeline ?? []).map(day => <Row key={day.date} {...day} />)}
                                                                                                                                  
   Example 4: Loading State
                                                                                                                                  
   // BAD — blank screen
   if (isLoading) return null;
                                                                                                                                  
   // GOOD — skeleton UI
   if (isLoading) return <Skeleton className="h-96" />;
                                                                                                                                  
   // GOOD — explicit spinner
   if (isLoading) return <div className="p-8 text-center"><Spinner /> Loading...</div>;
                                                                                                                                  
   Example 5: Object.entries()
                                                                                                                                  
   // BAD — crashes when by_type is null
   {Object.entries(data.by_type).map(([type, stats]) => ...)}
                                                                                                                                  
   // GOOD — nullish coalescing
   {Object.entries(data.by_type ?? {}).map(([type, stats]) => ...)}
                                                                                                                                  
                                                                                                                                  
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
   Your First Task
                                                                                                                                  
   Validate the CLV Analysis page (frontend/app/clv/page.tsx):
                                                                                                                                  
 1. Read reports/api_ground_truth.md — search for "GET /api/performance/clv-analysis"
 2. Read frontend/app/clv/page.tsx
 3. Check against the 7-point checklist
 4. Report findings as a numbered list with line numbers
                                                                                                                                  
   Expected issues to catch:
                                                                                                                                  
   • mean_clv displayed without ×100
   • Missing ?. on nested fields
   • .map() without fallback
   • Empty state not handled
                                                                                                                                  
                                                                                                                                  
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
   Files You Need to Know
                                                                                                                                  
    File                          Purpose
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    reports/api_ground_truth.md   API shapes — your bible
    FRONTEND_MIGRATION.md         Workstream status and context
    frontend/lib/types.ts         TypeScript interfaces (should match ground truth)
    tasks/todo.md                 Phase checklist — see what's in progress
                                                                                                                                  
                                                                                                                                  
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
   Questions?
                                                                                                                                  
   If something is unclear:
                                                                                                                                  
 1. Check FRONTEND_MIGRATION.md — it has design system colors and architectural decisions
 2. Check reports/api_ground_truth.md — it has exact API shapes
 3. Ask Claude Code (they own the architecture)
 4. Do NOT guess — ask for clarification
                                                                                                                                  
                                                                                                                                  
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
   Start your validation now. Report your findings on the CLV Analysis page.
                                                                                                                                  

 A scheduled reminder has been triggered. The reminder content is:
