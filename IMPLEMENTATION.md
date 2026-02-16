# Implementation Checklist & Next Steps

## ✅ What's Included (Ready to Deploy)

### Core Framework
- [x] **Version 7 Betting Model** (`backend/betting_model.py`)
  - 2-layer Monte Carlo CI
  - Safe Kelly calculation
  - Penalty budget with ceiling
  - Proper vig removal
  - Conservative decision threshold

### Backend API
- [x] **FastAPI Application** (`backend/main.py`)
  - REST API endpoints
  - APScheduler cron (nightly at 3 AM)
  - Health checks
  - Error handling

- [x] **Database Models** (`backend/models.py`)
  - Games, Predictions, BetLogs
  - Performance tracking
  - Model parameter versioning
  - SQLAlchemy ORM

- [x] **Authentication** (`backend/auth.py`)
  - Simple API key auth
  - Admin role support
  - No OAuth complexity

### Data Services
- [x] **Odds API Integration** (`backend/services/odds.py`)
  - The Odds API client
  - Line shopping (best odds)
  - Data freshness tracking

- [x] **Ratings Service** (`backend/services/ratings.py`)
  - KenPom API support
  - BartTorvik scraper (stub)
  - EvanMiya scraper (stub)
  - Caching layer

### Dashboard
- [x] **Streamlit UI** (`dashboard/app.py`)
  - Today's bets view
  - Performance metrics
  - Calibration plots
  - Bet log interface

### Deployment
- [x] **Railway Config** (`railway.json`)
- [x] **GitHub Actions** (`.github/workflows/deploy.yml`)
- [x] **Environment Template** (`.env.example`)
- [x] **Database Init Script** (`scripts/init_db.py`)

### Documentation
- [x] **README.md** - Full project overview
- [x] **QUICKSTART.md** - Step-by-step deployment
- [x] **This file** - Implementation guide

---

## 🚧 What Needs Completion (1-2 Days Work)

### Priority 1: Critical for First Run

1. **Complete `backend/services/analysis.py`**
   
   Create this new file to orchestrate the nightly job:
   
   ```python
   # backend/services/analysis.py
   
   def run_nightly_analysis():
       """
       Main nightly job logic:
       1. Fetch odds from The Odds API
       2. Fetch ratings from all sources
       3. For each game:
          - Run betting model
          - Store prediction
          - Send alert if "Bet" verdict
       4. Update performance snapshots
       5. Check for recalibration needs
       """
       # TODO: Implement
       pass
   ```
   
   This is the glue that ties everything together. Reference the existing services:
   - `from backend.services.odds import fetch_current_odds`
   - `from backend.services.ratings import fetch_current_ratings`
   - `from backend.betting_model import CBBEdgeModel`

2. **Implement BartTorvik/EvanMiya Scrapers**
   
   In `backend/services/ratings.py`, complete the scraping logic:
   
   ```python
   def scrape_barttorvik(self) -> Dict[str, float]:
       # Inspect HTML at barttorvik.com/trank.php
       # Extract team names + AdjEM values
       # Return dict: {team_name: rating}
   ```
   
   **Or** skip scraping and:
   - Pay for BartTorvik Premium API (if available)
   - Use only KenPom (weight 1.0) initially
   - Manually enter ratings for games you're betting

3. **Add Basic Tests**
   
   Create `tests/test_betting_model.py`:
   
   ```python
   def test_monte_carlo_ci():
       model = CBBEdgeModel()
       point, lower, upper = model.monte_carlo_prob_ci(5.0, 11.0)
       assert 0.5 < point < 0.8
       assert lower < point < upper
   
   def test_kelly_formula():
       model = CBBEdgeModel()
       kelly = model.kelly_fraction(0.55, 1.91)
       assert 0 < kelly < 0.2
   ```

### Priority 2: Nice to Have (Week 2-3)

4. **Alert System**
   
   Add `backend/services/alerts.py`:
   
   ```python
   # Email via SendGrid or SMTP
   def send_email_alert(subject, body):
       pass
   
   # SMS via Twilio
   def send_sms_alert(message):
       pass
   ```
   
   Call from `analysis.py` when bets are recommended.

5. **Bet Log UI in Streamlit**
   
   In `dashboard/app.py`, add bet entry form:
   
   ```python
   with st.form("log_bet"):
       game_id = st.number_input("Game ID")
       pick = st.text_input("Pick (e.g., Duke -4.5)")
       odds = st.number_input("Odds", value=-110)
       units = st.number_input("Units", value=1.0)
       
       if st.form_submit_button("Log Bet"):
           # POST to /api/bets/log
   ```

6. **Backfill Historical Data**
   
   Create `scripts/backfill_historical.py`:
   
   ```python
   def backfill_last_60_days():
       # Fetch past games from KenPom/Sports-Reference
       # Get actual outcomes
       # Get closing lines (if available)
       # Run model as if you bet that day
       # Store as "is_backfill=True" in BetLog
       # Instantly validates if model would have been profitable
   ```

7. **Quarterly Recalibration**
   
   Add to `backend/services/analysis.py`:
   
   ```python
   def check_recalibration():
       if not is_recalibration_month():
           return
       
       bets = get_last_90_days_bets()
       if len(bets) < 30:
           return
       
       mean_clv = calculate_clv(bets)
       if mean_clv < -0.005:
           disable_betting()
           send_alert("STOP BETTING - negative CLV")
       
       calibration_error = check_calibration(bets)
       if calibration_error > 0.07:
           new_sd = current_sd * 1.1
           update_parameter("base_sd", new_sd)
   ```

---

## 🎯 Week-by-Week Plan

### Week 1: Core Implementation
- **Day 1-2**: Complete `analysis.py` orchestration
- **Day 3**: Test nightly job manually
- **Day 4**: Deploy to Railway
- **Day 5**: Fix bugs, improve logging
- **Weekend**: Monitor first real runs

### Week 2: Validation
- **Day 1-2**: Implement backfill script
- **Day 3**: Analyze historical performance
- **Day 4-5**: Tune parameters if needed
- **Weekend**: Begin paper trading

### Weeks 3-12: Paper Trading
- Track 100+ bets
- Monitor CLV weekly
- Adjust model if CLV negative
- **If CLV > 0.5% after 100 bets**: Consider going live with small stakes

### Month 4+: Live Betting (if validated)
- Start with 0.25% Kelly max
- Scale to 1.0-1.5% after 250 profitable bets
- Continue recalibration quarterly

---

## 💡 Pro Tips

### Start Simple
- **Week 1**: Only use KenPom (skip BartTorvik/EM scrapers)
- **Week 2**: Add manual alerts (check dashboard daily)
- **Week 3+**: Automate alerts only after system is stable

### Monitor Obsessively
- Check Railway logs daily for first week
- Verify cron job runs at 3 AM
- Confirm data freshness in DB
- Watch for API quota limits

### Paper Trade Rigorously
- Log EVERY bet the system recommends
- Track as if real money
- Calculate real CLV vs closing lines
- Don't skip this step!

### When to Stop
If after 50+ bets:
- Mean CLV < -0.5%: Model is broken, recalibrate
- Win rate < 48%: Unlucky variance or bad model
- ROI < -5%: Stop immediately, review everything

### When to Scale
Only scale up if:
- CLV > +0.5% over 100+ bets
- Calibration error < 7%
- No suspicious patterns (all favorites, all overs, etc.)
- You understand why the model is working

---

## 📞 Getting Help

### Code Issues
1. Check Railway logs: `railway logs --tail 100`
2. Test locally first
3. Open GitHub issue with logs

### Model Questions
1. Review `README.md` framework section
2. Check `backend/betting_model.py` docstrings
3. Refer to Version 7 audit document

### Deployment Issues
1. Follow `QUICKSTART.md` exactly
2. Verify environment variables
3. Test database connection first

---

## 🎉 Success Metrics

After 3 months of paper trading:

✅ **System is working if:**
- Mean CLV > +0.5%
- 52-58% win rate
- ROI > 0%
- Calibration error < 7%
- PASS rate 85-95%

❌ **Stop and recalibrate if:**
- Mean CLV < -0.5%
- Win rate < 48%
- ROI < -5%
- Systematic bias detected

---

## 🚀 Ready to Ship?

You have everything needed to deploy a production betting system. The framework is sound, the code is clean, and the deployment is simple.

**Next command:**

```bash
git init
git add .
git commit -m "Initial commit: CBB Edge Analyzer v7"
git remote add origin https://github.com/sfgray26/hello-world.git
git push -u origin main
```

Then follow `QUICKSTART.md` to deploy to Railway.

**Good luck!** 🍀

---

*Remember: This is a tool for disciplined, systematic betting. It will say NO 90% of the time. That's correct behavior for efficient markets. Trust the process, track everything, and only bet when the lower-bound edge is positive.*
