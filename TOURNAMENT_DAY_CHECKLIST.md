# 🏀 TOURNAMENT DAY MASTER CHECKLIST

**Tournament Starts:** March 18, 2026 (First Four)  
**First Round:** March 19-20  
**Championship:** April 7

---

## ✅ PRE-GAME CHECKLIST (Run Before First Tip-off)

### System Health
- [ ] Deploy latest code to Railway: `railway up`
- [ ] Verify dashboard loads: Check Railway URL
- [ ] Run pre-tournament check: `python scripts/pretournament_check.py`
- [ ] Clear browser cache for users

### Data Verification
- [ ] Confirm bracket_2026.json has all 64 teams
- [ ] Verify First Four teams are marked correctly
- [ ] Check composite ratings look reasonable
- [ ] Run fresh 50k simulations: `python scripts/run_bracket_sims.py --sims 50000`

### Discord Notifications
- [ ] Verify DISCORD_WEBHOOK_URL in Railway env vars
- [ ] Test morning briefing job
- [ ] Test end-of-day results job
- [ ] Confirm notification schedule is active

### UI Testing
- [ ] Page 13 (Tournament Bracket): Runs simulations
- [ ] Page 14 (Bracket Visual): Chaos slider works
- [ ] Check upset explanations display correctly
- [ ] Verify mobile responsiveness

---

## 📊 DAILY WORKFLOW (Each Tournament Day)

### Morning (7:00 AM ET)
- [ ] Morning briefing sent to Discord
- [ ] Check for any last-minute injury news
- [ ] Review sharp money alerts
- [ ] Verify model predictions are current

### Throughout Day
- [ ] Monitor Discord for user questions
- [ ] Watch for significant line movements
- [ ] Update First Four results as games complete
- [ ] Track CLV on recommended bets

### Evening (11:00 PM ET)
- [ ] End-of-day results sent to Discord
- [ ] Update simulation results with actual outcomes
- [ ] Recalculate championship probabilities
- [ ] Prepare next day's briefings

---

## 🚨 EMERGENCY PROCEDURES

### If Discord Notifications Stop
1. Check Railway logs: `railway logs`
2. Verify webhook URL is still valid
3. Test manually: `python -c "from backend.services.discord_bet_embeds import *; send_test()"`
4. Restart Railway service if needed

### If Bracket Data Is Wrong
1. DO NOT PANIC
2. Check data/bracket_2026.json for errors
3. Update from DB if needed: `python scripts/build_bracket_from_db.py`
4. Re-run simulations: `python scripts/run_bracket_sims.py --quick`
5. Redeploy: `railway up`

### If Simulations Won't Run
1. Check outputs/tournament_2026/ permissions
2. Verify bracket_2026.json is valid JSON
3. Try with fewer sims: `--sims 1000`
4. Check for missing team data

### If UI Won't Load
1. Check Railway dashboard for errors
2. Verify Dockerfile builds correctly
3. Test locally: `streamlit run dashboard/Home.py`
4. Redeploy with empty commit if needed

---

## 📈 KEY METRICS TO TRACK

### Model Performance
- [ ] Win rate vs closing line
- [ ] CLV on recommended bets
- [ ] Upset prediction accuracy
- [ ] Cinderella team identification

### System Health
- [ ] Discord notification delivery rate
- [ ] Dashboard uptime
- [ ] Simulation run times
- [ ] User engagement

---

## 🎯 SUCCESS CRITERIA

### By End of First Round
- [ ] Model correctly predicted 60%+ of games
- [ ] CLV positive on recommended bets
- [ ] At least 2 Cinderella teams identified
- [ ] No major system outages

### By Sweet 16
- [ ] Final Four probabilities refined
- [ ] At least 1 double-digit seed in S16
- [ ] User feedback incorporated

### By Championship
- [ ] Model correctly predicted champion OR runner-up
- [ ] Positive ROI on tournament betting
- [ ] Full data pipeline validated

---

## 📝 NOTES

### Known Limitations
- First Four winners inherit loser's rating (update manually after games)
- Live odds not integrated (manual entry only)
- No in-game betting support

### Quick Commands
```bash
# Fresh simulations
python scripts/run_bracket_sims.py --sims 50000 --workers 4

# Import user ratings
python scripts/update_bracket_from_csv.py --csv ratings.csv

# Pre-tournament check
python scripts/pretournament_check.py

# Deploy to Railway
railway up
```

---

## 🍀 GOOD LUCK!

May the chaos be ever in your favor.

*Last Updated: March 17, 2026*
