# CBB Edge Analyzer

**Production-ready college basketball betting framework with automated analysis, CLV tracking, and performance monitoring.**

Built on Version 7 of the rigorous betting framework - designed to PASS 85-95% of games and only bet when the lower-bound edge is positive after all uncertainty adjustments.

## 🎯 Features

- **Nightly automated analysis** of all major CBB games
- **Real-time odds** via The Odds API
- **Multi-system ratings** (KenPom, BartTorvik, EvanMiya)
- **Monte Carlo uncertainty quantification**
- **Closing Line Value (CLV) tracking**
- **Automatic quarterly recalibration**
- **Performance dashboard** with calibration plots
- **SMS/Email alerts** for betting opportunities

## 📊 Tech Stack

- **Backend**: FastAPI (Python 3.11+) with SQLAlchemy
- **Database**: PostgreSQL (Railway or Supabase)
- **Hosting**: Railway.app ($5-15/mo)
- **Dashboard**: Streamlit (v1) → Next.js (v2)
- **Auth**: API Key (simple, secure)
- **Cron**: APScheduler (in-process)
- **APIs**: The Odds API ($10-50/mo), KenPom API ($8/mo)

**Total cost**: ~$25-80/mo

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL (local or Railway/Supabase)
- The Odds API key (free tier: https://the-odds-api.com)
- KenPom API key ($95/year: https://kenpom.com/api)

### Local Development

```bash
# 1. Clone and setup
git clone https://github.com/sfgray26/hello-world.git cbb-edge
cd cbb-edge
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup environment
cp .env.example .env
# Edit .env with your API keys and database URL

# 4. Initialize database
python scripts/init_db.py

# 5. (Optional) Backfill historical data
python scripts/backfill_historical.py --days 60

# 6. Run backend
uvicorn backend.main:app --reload --port 8000

# 7. Run dashboard (separate terminal)
streamlit run dashboard/app.py
```

### Deploy to Railway

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login and initialize
railway login
railway init

# 3. Add PostgreSQL
railway add

# Select PostgreSQL, then link it

# 4. Set environment variables in Railway dashboard
# THE_ODDS_API_KEY, KENPOM_API_KEY, API_KEY_USER1, etc.

# 5. Deploy
railway up

# 6. Deploy Streamlit dashboard to Streamlit Cloud
# Connect your GitHub repo at share.streamlit.io
```

## 📁 Project Structure

```
cbb-edge/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI app + cron scheduler
│   ├── models.py            # SQLAlchemy database models
│   ├── schemas.py           # Pydantic schemas
│   ├── betting_model.py     # Version 7 framework core
│   ├── auth.py              # API key authentication
│   └── services/
│       ├── odds.py          # The Odds API integration
│       ├── ratings.py       # KenPom/BartTorvik/EvanMiya
│       ├── analysis.py      # Nightly job orchestration
│       └── alerts.py        # Email/SMS notifications
├── dashboard/
│   ├── app.py               # Streamlit dashboard
│   └── components/          # Reusable UI components
├── scripts/
│   ├── init_db.py           # Database initialization
│   ├── backfill_historical.py
│   └── manual_recalibrate.py
├── tests/
│   ├── test_betting_model.py
│   └── test_api.py
├── .env.example
├── requirements.txt
├── railway.json             # Railway config
└── README.md
```

## 🔐 Authentication

The app uses simple API key authentication for personal use:

```python
# Request headers
headers = {"X-API-Key": "your-secret-key-here"}
```

Set `API_KEY_USER1` in your environment variables.

## 📈 Using the System

### Daily Workflow

1. **3 AM ET**: Nightly cron job runs automatically
   - Fetches latest odds and ratings
   - Analyzes all games for the day
   - Stores predictions in database
   - Sends alerts for betting opportunities

2. **Morning**: Check dashboard
   - Review recommended bets
   - Verify data freshness
   - Manually place bets with sportsbook

3. **After games**: Mark outcomes
   - Update bet logs with results
   - System auto-calculates CLV

### Manual Operations

```bash
# Trigger analysis manually
curl -X POST http://localhost:8000/admin/run-analysis \
  -H "X-API-Key: your-key"

# Force recalibration
python scripts/manual_recalibrate.py

# View performance metrics
curl http://localhost:8000/api/performance \
  -H "X-API-Key: your-key"
```

## 🎲 Betting Philosophy

This framework is **conservative by design**:

- **Expected PASS rate**: 85-95% of games
- **Typical edge when found**: 2-5% post-vig
- **Expected long-term ROI**: 2-6% (if valid)
- **Bet frequency**: 5-15 games per week (peak season)

**The system will say NO most of the time. This is correct.**

## 📊 Performance Tracking

### Key Metrics

- **Mean CLV**: Average edge vs closing line (should be >0.5%)
- **ROI**: Actual profit/loss vs total risked
- **Calibration**: Do 60% predictions win 60% of the time?
- **Win Rate**: Raw wins/losses (expect 52-58%)

### Stop-Loss Rules

The system will **pause betting** if:
- Mean CLV < -0.5% over 50 bets
- Drawdown exceeds 15% of bankroll
- Calibration error > 7%

**Manual review required** before resuming.

## 🔧 Maintenance

### Quarterly Recalibration (Automatic)

On first day of Jan/Apr/Jul/Oct:
1. Analyzes last 90 days of bets
2. Recalculates optimal weights for KenPom/Bart/EM
3. Adjusts SD if calibration is off
4. Sends email report

### Scraper Maintenance

BartTorvik/EvanMiya scrapers may break when sites update:
- Check `logs/scraper_failures.log`
- Update CSS selectors in `backend/services/ratings.py`
- Consider paid API access if available

### Monitoring

- **UptimeRobot**: Monitor Railway app (free)
- **Email alerts**: On scraper failures, CLV drift, betting opportunities
- **Weekly reports**: Performance summary every Monday

## 🧪 Testing

```bash
# Run test suite
pytest tests/

# Test betting model in isolation
python -m pytest tests/test_betting_model.py -v

# Test API endpoints
python -m pytest tests/test_api.py -v

# Manual test of nightly job (dry run)
python backend/services/analysis.py --dry-run
```

## 📝 Environment Variables

Required:
```
DATABASE_URL=postgresql://user:pass@host:5432/cbb_edge
THE_ODDS_API_KEY=your_odds_api_key
KENPOM_API_KEY=your_kenpom_key
API_KEY_USER1=random_secure_string_min_32_chars
```

Optional:
```
BARTTORVIK_USERNAME=username  # If using paid tier
EVANMIYA_API_KEY=key          # If available
TWILIO_ACCOUNT_SID=sid        # For SMS alerts
TWILIO_AUTH_TOKEN=token
TWILIO_FROM_NUMBER=+1234567890
TWILIO_TO_NUMBER=+1234567890
SENDGRID_API_KEY=key          # For email alerts
ALERT_EMAIL=your@email.com
```

## 🐛 Troubleshooting

### "No bets recommended for 7 days"

This is **normal** if:
- Markets are efficient (expected 85-95% PASS rate)
- Ratings are closely aligned with market
- Current matchups have high uncertainty

This is a **problem** if:
- Scraper failures (check logs)
- Stale data (check freshness tiers)
- Model parameters need recalibration

### "CLV is negative"

If mean CLV < -0.5% over 50+ bets:
1. **STOP BETTING IMMEDIATELY**
2. Check if you're getting best available lines
3. Verify odds are fetched <10min before kickoff
4. Review injury/ratings data for accuracy
5. Consider if market has become sharper

### Database connection errors

Railway Postgres:
```bash
# Get connection string
railway variables

# Test connection
psql $DATABASE_URL
```

## 📚 Additional Resources

- [Version 7 Framework Documentation](docs/FRAMEWORK.md)
- [API Reference](docs/API.md)
- [Model Parameters](docs/PARAMETERS.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

## ⚠️ Disclaimer

This software is for **educational and personal use only**. 

- Not financial advice
- No guaranteed profitability
- Gambling involves risk of loss
- Check local laws regarding sports betting
- Never bet more than you can afford to lose
- Past performance ≠ future results

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

Built on publicly available research:
- KenPom methodology (Ken Pomeroy)
- Woodland & Woodland (2001) - correlation studies
- Weimar & Wicker (2018) - injury impact
- Boulier et al. (2006) - CBB injury effects

---

**Ready to deploy?** Follow the Quick Start guide above.

**Questions?** Open an issue on GitHub.

**Found a bug?** PRs welcome!
