# CBB BETTING PROJECT — COMPLETE CONTEXT FOR CLAUDE CODE

## Executive Summary

This is a **college basketball betting analytics platform** with a sophisticated V9.1 prediction model, tournament bracket simulator, and Discord notification system. The project has been under active development with ~4,000+ lines added in the past week alone.

**Current Status:** All core systems functional. Tournament deadline (March 18) approaching. Bracket simulator now uses advanced chaos mode.

---

## 🏗️ Architecture Overview

### Tech Stack
- **Backend:** Python 3.11, FastAPI, SQLAlchemy, PostgreSQL
- **Data Sources:** KenPom (51%), BartTorvik (49%), EvanMiya dropped (Cloudflare blocking)
- **Frontend:** Streamlit (dashboard pages 1-14)
- **Deployment:** Railway (Docker-based)
- **Notifications:** Discord webhooks with scheduled briefings

### Directory Structure
```
CBB_Betting/
├── backend/
│   ├── main.py                    # FastAPI app, health checks, Discord jobs
│   ├── models.py                  # SQLAlchemy models
│   ├── services/
│   │   ├── analysis.py            # V9.1 betting model core
│   │   ├── conference_hca.py      # Conference-specific home court advantage
│   │   ├── recency_weight.py      # Late-season form weighting
│   │   ├── sharp_money.py         # Steam/RLM detection
│   │   ├── bracket_simulator.py   # Monte Carlo tournament sim
│   │   ├── discord_bet_embeds.py  # Rich Discord notifications
│   │   └── ...
│   └── tournament/
│       ├── matchup_predictor.py   # Single game predictions with all factors
│       ├── bracket_simulator.py   # Parallel Monte Carlo simulation
│       ├── smart_bracket.py       # NEW: Chaos-based upset prediction
│       ├── cinderella_tracker.py  # Double-digit seed analysis
│       └── futures_analyzer.py    # EV calculations vs market odds
├── dashboard/
│   └── pages/
│       ├── 13_Tournament_Bracket.py    # Run simulations, view results
│       └── 14_Bracket_Visual.py        # NEW: Smart bracket visual with chaos slider
├── data/
│   └── bracket_2026.json          # Full tournament field with ratings
├── outputs/tournament_2026/       # Simulation results
│   ├── sim_results.json
│   ├── championship_probs.csv
│   ├── upset_heatmap_r64.csv
│   └── cinderella_rankings.csv
└── scripts/
    ├── run_bracket_sims.py        # CLI for Monte Carlo
    ├── update_bracket_from_csv.py # Import user ratings
    └── openclaw_scheduler_improved.py
```

---

## 🧠 V9.1 Prediction Model

### Composite Rating Formula
```python
composite = 0.51 × KenPom_AdjEM + 0.49 × BartTorvik_AdjEM
```

### Key Enhancements (P0-P3 Completed)

#### P0: Data Pipeline Audit ✅
- Verified 2-source mode (KenPom + BartTorvik)
- 365 teams in BartTorvik public CSV (no auth required)
- EvanMiya intentionally dropped (Cloudflare blocking)

#### P1: Sharp Money Detection ✅
**File:** `backend/services/sharp_money.py` (400 lines)
```python
class SharpMoneyAnalyzer:
    - Steam detection (≥1.5 pts in <30 min)
    - Opener gap detection (≥2.0 pts vs model)
    - Reverse line movement (RLM) detection
    - Edge adjustment: +0.5% when aligned, -0.8% when opposed
```

#### P2: Conference-Specific HCA ✅
**File:** `backend/services/conference_hca.py` (280 lines)
- 25 conferences mapped with custom HCA values
- Big Ten: 3.6, Big 12: 3.4, SEC: 3.2, SWAC: 1.5
- Pace-adjusted scaling for neutral sites

#### P3: Late-Season Recency Weighting ✅
**File:** `backend/services/recency_weight.py` (280 lines)
- 2x weight for days 0-2
- Tournament mode detection (March 15+)
- Margin SE inflation +0.20 for neutral sites

---

## 🏀 Tournament Bracket Simulator

### Monte Carlo Engine
**File:** `backend/tournament/bracket_simulator.py`

**Key Settings (MAXIMUM CHAOS MODE):**
```python
ROUND_HIST_WEIGHT = {
    1: 0.75,   # R64: 75% history (was 55%)
    2: 0.50,   # R32: 50% history
    3: 0.20,   # S16: 20% history
    4: 0.10,   # E8: 10% history
    5: 0.00,   # F4: pure model
}
_ADJM_BASE_DIVISOR = 14.0  # was 10.0 (reduces AdjEM dominance)
_TOURNAMENT_SD_FACTOR = 1.40
```

### 7 Intelligence Upgrades in Matchup Predictor
**File:** `backend/tournament/matchup_predictor.py`

1. **Per-round SD multipliers:** R64: 1.12x → Champ: 1.0x
2. **Seed-matchup historical blend:** 40% history for 1v16, 20% for 8v9
3. **Style-based variance:** Pace mismatch + high 3PT rate = chaos
4. **Tournament experience factor:** ±1.5 pts for returning players
5. **Recent form factor:** March performance ±2 pts
6. **Composite rating:** 55% KenPom + 45% BartTorvik
7. **Runner-up probability tracking**

### Historical Upset Rates Used
```python
SEED_UPSET_RATES = {
    (1, 16): 0.013,  # 1.3%
    (2, 15): 0.067,  # 6.7%
    (3, 14): 0.153,  # 15.3%
    (4, 13): 0.216,  # 21.6%
    (5, 12): 0.352,  # 35.2% ← famous 12-5 upset zone
    (6, 11): 0.389,  # 38.9%
    (7, 10): 0.394,  # 39.4%
    (8, 9):  0.487,  # 48.7%
}
```

---

## 🧠 NEW: Smart Bracket Generator

**File:** `backend/tournament/smart_bracket.py` (just built)

### What It Does
Generates bracket predictions using **ALL available data** with chaos-adjusted upset thresholds.

### Data Sources Combined
| Factor | Weight | Source |
|--------|--------|--------|
| Monte Carlo results | 40% | sim_results.json (10k-50k sims) |
| V9.1 model | 40% | Composite ratings |
| Historical seed rates | 20% | 2000-2024 data |
| Style boosts | Variable | Pace, 3PT, defense |
| Cinderella boosts | Variable | Tourney exp, form, momentum |

### Style-Based Upset Triggers
```python
# Pace mismatch (fast vs slow = chaos)
PACE_MISMATCH_THRESHOLD = 10.0
pace_boost = +8% if underdog is faster

# 3PT variance (live by the three...)
HIGH_3PT_THRESHOLD = 0.40
three_pt_boost = +12% if both teams high 3PT%

# Defensive edge
def_boost = +6% if underdog better def_eFG%
```

### Cinderella Factors
```python
# Tournament experience
tourney_exp_boost = +5% if underdog has >20% more exp

# Recent form (hot team)
recent_form_boost = +10% if underdog.form > 1.0

# Playing above seed rating
momentum_boost = +6% if actual rating diff < expected
```

### Chaos Threshold Formula
```python
# Higher chaos = lower threshold for predicting upsets
chaos_threshold = 0.50 - (chaos_level × 0.30)

# Examples:
# Chaos 0.0: threshold = 50% (favorites unless >50% upset prob)
# Chaos 0.5: threshold = 35% (upsets at 35% prob)
# Chaos 0.8: threshold = 26% (upsets at 26% prob)
# Chaos 1.0: threshold = 20% (upsets at 20% prob)
```

---

## 🎮 Bracket Visual UI (Page 14)

**File:** `dashboard/pages/14_Bracket_Visual.py`

### Features
- **Chaos Level Slider:** 0.0 - 1.0 (just implemented)
- **Mode Indicators:**
  - 🏆 Chalk (0.0)
  - 📊 Model (0.1-0.3)
  - ⚡ Style-Aware (0.4-0.6)
  - 🎭 Cinderella (0.7-0.8)
  - 🔥 Maximum Chaos (0.9-1.0)
- **Upset Analysis Panel:** Shows each predicted upset with explanation
- **Smart Bracket Integration:** Uses new `smart_bracket.py` generator

### Current Issue
The text output works but the **visual bracket HTML** still shows favorites. Need to verify the visual is using the smart generator output correctly.

---

## 📱 Discord Notification System

### Critical Fixes Recently Applied
**Files:** `backend/services/discord_bet_embeds.py`, `openclaw_briefs_improved.py`

#### Jobs Schedule
```python
# Morning Briefing: 7:00 AM ET daily
# End of Day Results: 11:00 PM ET daily
# Tournament Bracket: When bracket releases
```

#### Fixed Issues
- `_morning_briefing_job()` now actually calls Discord (was only logging)
- Added `_end_of_day_results_job()` for daily recap
- Added `_tournament_bracket_job()` for bracket release notifications
- Created rich embeds with bet summaries

---

## 📊 Current Data Flow

### Bracket Simulation Pipeline
1. `data/bracket_2026.json` → Team ratings (composite, style, experience)
2. `scripts/run_bracket_sims.py` → Monte Carlo (10k-50k simulations)
3. `outputs/tournament_2026/sim_results.json` → Raw probabilities
4. `backend/tournament/smart_bracket.py` → Chaos-adjusted predictions
5. `dashboard/pages/14_Bracket_Visual.py` → UI with slider

### Daily Betting Pipeline
1. Data refresh (KenPom + BartTorvik)
2. Sharp money detection (steam, RLM)
3. V9.1 analysis with HCA + recency weights
4. CLV attribution tracking
5. Discord notifications

---

## 🎯 KNOWN ISSUES & TODOs

### Critical (Before March 18)
- [ ] Verify bracket visual HTML shows upsets at chaos 0.8+
- [ ] Ensure Discord jobs are firing on schedule
- [ ] Confirm DB connection on Railway for live ratings

### Enhancement Opportunities
1. **Live Odds Integration:** Currently manual entry, could wire to odds API
2. **Parlay Optimization:** Basic implementation exists, could enhance
3. **In-Game Betting:** No live/in-play support currently
4. **Historical Backtesting:** Limited validation of model accuracy
5. **Player Props:** No individual player analysis

### Technical Debt
- [ ] Some duplicate logic between `bracket_simulator.py` and `smart_bracket.py`
- [ ] Error handling in smart bracket fallback could be cleaner
- [ ] Session state management in Streamlit could use refactoring

---

## 🔮 Potential Enhancements for Claude Code

### 1. **Player-Level Analysis** 🏀
- Parse box scores for player efficiency
- Injury impact modeling
- Matchup-specific player advantages

### 2. **Live Betting Module** ⚡
- WebSocket connection for live odds
- Momentum detection (scoring runs)
- Fouls/trouble analysis for player props

### 3. **Advanced Parlay Builder** 🎯
- Correlation analysis (don't parlay correlated bets)
- Kelly Criterion for parlay sizing
- Round-robin optimization

### 4. **Model Validation Dashboard** 📈
- Backtesting against historical results
- CLV tracking over time
- ROI by bet type, conference, time of season

### 5. **Social Features** 👥
- User bet tracking
- Leaderboards
- Bet sharing to Discord

### 6. **ML Enhancements** 🤖
- Train XGBoost on historical features
- Ensemble methods (combine V9.1 + ML)
- Feature importance analysis

---

## 📋 Key Commands

```bash
# Run fresh bracket simulations
python scripts/run_bracket_sims.py --bracket data/bracket_2026.json --sims 50000

# Import user CSV ratings
python scripts/update_bracket_from_csv.py --csv ratings.csv

# Run tests
pytest tests/test_tournament_data.py -v

# Deploy to Railway
railway up
```

---

## 🗣️ Context for Claude Code

### What You Should Know
1. **This is a working system** — not a prototype. It has real predictions, real money tracking, real Discord notifications.

2. **Tournament deadline is March 18** — First Four tipoff. The bracket simulator needs to be solid by then.

3. **CLV is the north star** — All changes measured against closing line value, not just ROI.

4. **Data before features** — The user prioritizes data pipeline integrity over shiny features.

5. **Fantasy Baseball is paused** — Until after April 7 championship. Focus is CBB tournament.

### User Preferences
- Wants sophisticated, data-driven outputs
- Prefers deterministic predictions when possible ("show me the math")
- Values upset prediction and Cinderella stories in tournament mode
- Uses Discord for notifications
- Deploys via Railway

### What to Look For
- **Inconsistencies** between bracket simulator and smart bracket
- **Missing error handling** in new smart bracket code
- **Performance issues** with 50k simulations
- **UI/UX improvements** for the chaos slider and upset explanations
- **Test coverage** — many new modules lack comprehensive tests

---

## 📞 Questions to Ask

1. Should the smart bracket use the **Monte Carlo results** more directly (currently 40% weight)?

2. Should we add **probability-weighted randomization** option ("surprise me" mode)?

3. Should the **visual bracket** show upset paths differently (colors, arrows, etc.)?

4. What's the **validation plan** for these predictions vs actual tournament results?

5. Should we **wire live odds** for real-time value detection during tournament?

---

## 🎓 Recent Decisions (for context)

- **Dropped EvanMiya:** Cloudflare blocking, 2-source mode (KP+BT) is robust
- **BartTorvik auth:** Username/password env vars are legacy; cloudscraper with headers works
- **Sharp money edge:** +0.5% when aligned, -0.8% when opposed
- **Bracket chalkiness fix:** Increased history weight to 75% for R64, reduced AdjEM divisor to 14.0
- **Cinderella boost:** Lower seeds (12, 11, 13) get +1.0 to +2.0 point rating boost

---

**END OF CONTEXT**

You now have full understanding of the CBB Betting project. The Smart Bracket Generator was just built tonight and needs polish. The tournament starts March 18. What enhancements or fixes do you recommend?
