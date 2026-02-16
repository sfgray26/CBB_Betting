# Quick Start Guide

## Local Development (5 minutes)

### 1. Setup Environment

```bash
# Clone your repo
git clone https://github.com/sfgray26/hello-world.git cbb-edge
cd cbb-edge

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Copy template
cp .env.example .env

# Edit .env and add your keys:
nano .env  # or use your favorite editor
```

Required keys:
- `DATABASE_URL`: PostgreSQL connection string
- `THE_ODDS_API_KEY`: From https://the-odds-api.com
- `KENPOM_API_KEY`: From https://kenpom.com/api
- `API_KEY_USER1`: Generate a random 32+ char string

### 3. Setup Local Database

```bash
# Install PostgreSQL locally, or use Docker:
docker run --name cbb-postgres -e POSTGRES_PASSWORD=yourpass -p 5432:5432 -d postgres:15

# Update DATABASE_URL in .env:
DATABASE_URL=postgresql://postgres:yourpass@localhost:5432/cbb_edge

# Initialize database
python scripts/init_db.py --seed
```

### 4. Run Backend

```bash
# Start FastAPI server
uvicorn backend.main:app --reload --port 8000

# Test in browser: http://localhost:8000
# API docs: http://localhost:8000/docs
```

### 5. Run Dashboard

```bash
# In a new terminal (keep backend running)
source venv/bin/activate

# Set API connection
export API_URL=http://localhost:8000
export API_KEY=your_api_key_user1_value

# Start Streamlit
streamlit run dashboard/app.py
```

Dashboard opens at: http://localhost:8501

---

## Deploy to Railway (15 minutes)

### 1. Create Railway Account

Visit https://railway.app and sign up (free tier available)

### 2. Install Railway CLI

```bash
npm install -g @railway/cli
railway login
```

### 3. Create New Project

```bash
# In your cbb-edge directory
railway init

# When prompted:
# - Project name: cbb-edge
# - Create new project: Yes
```

### 4. Add PostgreSQL Database

```bash
railway add

# Select: PostgreSQL
# This automatically creates a database and sets DATABASE_URL
```

### 5. Set Environment Variables

Go to Railway dashboard → your project → Variables tab

Add:
```
THE_ODDS_API_KEY=your_key
KENPOM_API_KEY=your_key
API_KEY_USER1=random_secure_string_32_chars_minimum
ENVIRONMENT=production
```

### 6. Deploy

```bash
# Push code and deploy
railway up

# Initialize database (one time only)
railway run python scripts/init_db.py --seed

# Check logs
railway logs
```

Your API will be live at: `https://your-project.up.railway.app`

### 7. Deploy Streamlit Dashboard

Option A: Streamlit Cloud (Free)
1. Go to https://share.streamlit.io
2. Connect your GitHub repo
3. Select `dashboard/app.py` as main file
4. Add secrets (in Streamlit Cloud settings):
   ```toml
   API_URL = "https://your-railway-app.up.railway.app"
   API_KEY = "your_api_key_user1"
   ```

Option B: Railway (paid)
- Create second Railway service for Streamlit
- Costs ~$5/mo extra

---

## Verify Deployment

### Test API

```bash
# Replace with your Railway URL
export API_URL=https://your-project.up.railway.app
export API_KEY=your_api_key

# Health check
curl $API_URL/health

# Get predictions (should be empty initially)
curl $API_URL/api/predictions/today \
  -H "X-API-Key: $API_KEY"

# Trigger manual analysis (admin only)
curl -X POST $API_URL/admin/run-analysis \
  -H "X-API-Key: $API_KEY"
```

### Test Dashboard

Visit your Streamlit URL and verify:
- ✅ Dashboard loads
- ✅ Can see "Total Bets: 0"
- ✅ Performance section shows "No data yet"

---

## First Real Run

### 1. Wait for Nightly Job

The scheduler runs at 3 AM ET automatically. Or trigger manually:

```bash
curl -X POST $API_URL/admin/run-analysis \
  -H "X-API-Key: $API_KEY"
```

### 2. Check Results

```bash
# See today's predictions
curl $API_URL/api/predictions/today \
  -H "X-API-Key: $API_KEY" | jq
```

### 3. If Bets Are Recommended

The dashboard will show them under "Today's Bets" with:
- Projected margin
- Edge estimate
- Recommended units

**Place bets manually** with your sportsbook, then log them:

```bash
curl -X POST $API_URL/api/bets/log \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": 123,
    "pick": "Duke -4.5",
    "odds_taken": -110,
    "bet_size_units": 1.0,
    "bet_size_dollars": 10.00,
    "bankroll_at_bet": 1000
  }'
```

### 4. After Game Settles

Update the outcome:

```bash
curl -X PUT $API_URL/api/bets/1/outcome?outcome=1&closing_line=-5.5 \
  -H "X-API-Key: $API_KEY"
```

(outcome: 1=win, 0=loss)

---

## Monitoring

### Check Scheduler Status

```bash
curl $API_URL/admin/scheduler/status \
  -H "X-API-Key: $API_KEY"
```

### View Logs (Railway)

```bash
railway logs --tail 100
```

### Performance Metrics

Check dashboard "Performance" tab weekly for:
- CLV trending positive (>0.5%)
- Win rate 52-58%
- ROI positive
- Calibration error <7%

---

## Troubleshooting

### "No bets recommended for days"

✅ **This is normal!** Expected PASS rate is 85-95%.

Check:
- Are ratings being fetched? (Check logs)
- Are odds current? (Check logs for API errors)
- Is model too conservative? (Check penalties in DB)

### "API Error: 401 Unauthorized"

- Verify `X-API-Key` header is set
- Check `API_KEY_USER1` in environment variables

### "Database connection failed"

Railway:
```bash
railway variables  # Get DATABASE_URL
railway run python scripts/init_db.py --check
```

### "Odds API quota exceeded"

Free tier: 500 requests/month

Calculate: ~16 requests/day (fine for nightly job)

If exceeded:
- Upgrade to paid tier ($10-50/mo)
- Reduce fetching frequency

---

## Next Steps

1. ✅ **Paper trade for 100 bets** (2-3 months)
2. ✅ **Verify CLV > 0**
3. ✅ **Check calibration** (predicted vs actual)
4. ✅ **If successful, scale units gradually**

---

## Support

- GitHub Issues: https://github.com/sfgray26/hello-world/issues
- Framework Docs: See `README.md`

Good luck! 🍀
