# 🎯 Draft Day War Room Blueprint 2026
## Treemendous League · Yahoo ID 72586 · 12-Team H2H One Win · 18 Categories

**Draft Date:** March 23, 2026 @ 7:30am EDT  
**Format:** Snake Draft · 90-Second Clock · 23 Rounds (276 picks)  
**Your Edge:** Steamer Projections + Bat Tracking Analytics + Statcast Integration

---

# 1. Macro Draft Strategy & Positional Scarcity

## 1.1 Optimal Roster Construction: "Modified Pocket Aces"

For your 18-category H2H league with 90-second pick windows, I recommend a **Modified Pocket Aces** approach that leverages your Steamer projections:

| Round Range | Strategy | Target Archetype |
|-------------|----------|------------------|
| **R1-R3** | Elite Bats First | 5-tool hitters with 25+/25+ or 40+ HR upside |
| **R4-R6** | Pocket Aces Window | SPs with sub-3.30 ERA, 200+ K upside |
| **R7-R9** | Middle Infield Premium | 2B/SS with speed or power scarcity |
| **R10-R14** | SP Depth + CI Bats | High-K SPs + Corner infield value |
| **R15-R20** | Closer Sprint | Target 2-3 NSV sources (your league counts Net Saves!) |
| **R21-R23** | Upside Flyers | Prospects with 15%+ Blast% or 115+ Stuff+ |

**Why This Works for Your League:**
- NSV (Net Saves) as a category makes closers more valuable — a closer with 30 saves but 8 blown saves is toxic
- 18 categories mean you can't punt anything — balanced rosters win
- Your Steamer data shows SP depth through Round 8 — don't reach early

## 1.2 Positional Scarcity Tiers (Critical Drop-offs)

### 🚨 TIER 1 SCARCITY — Draft Early or Suffer

| Position | The Cliff | When to Reach | Your Steamer Flag |
|----------|-----------|---------------|-------------------|
| **Catcher** | After Cal Raleigh (#14 ADP) | Rounds 3-5 | William Contreras is only other 20+ HR C |
| **Shortstop** | After Elly De La Cruz (#8) | Rounds 6-8 | Massive drop to Zach Neto (#33) tier |
| **Third Base** | After Junior Caminero (#17) | Rounds 7-9 | Jose Ramirez carries 3B — reach if needed |
| **2B Elite** | After Ketel Marte (#27) | Rounds 8-10 | Brice Turang (#46) is last speed+power 2B |

**Your Draft Application Should:**
- Color-code these positions in red when approaching the cliff
- Pop an alert: "2B SCARCITY IN 3 PICKS" when approaching Round 8
- Track roster composition live — if you have no 2B/SS by Round 9, force a reach

### ⚠️ TIER 2 SCARCITY — Manageable but Monitor

| Position | The Drop-off | When to Act |
|----------|--------------|-------------|
| **SP Ace** | After Cristopher Sanchez (#28) | R6-R7 latest for reliable ace |
| **Elite Closer** | After Mason Miller (#44) | R4-R6 for locked roles |
| **OF Power** | After Kyle Schwarber (#20) | R5-R7 for 35+ HR OF |

## 1.3 Category-Specific Strategy

### For NSV (Net Saves) — Your Secret Weapon

Your closer_situations_2026.csv tracks this perfectly. Target these tiers:

| Tier | Closers | Projected NSV | When to Draft |
|------|---------|---------------|---------------|
| **Elite** | Clase, Hader, Miller, Diaz | 28-34 | Rounds 4-6 |
| **Solid** | Williams, Iglesias, Munoz | 24-28 | Rounds 8-11 |
| **Speculative** | Fairbanks, Hoffman, Finnegan | 18-22 | Rounds 13-16 |
| **Committee Avoid** | Pagan, Neris, Estevez | <15 | Only in R20+ |

**Rule:** Never draft a closer without checking your `closer_situations_2026.csv` role column. Only "locked" and "likely" roles are safe.

### For Speed (SB/NSB)

Your Steamer data shows only 12 players with 25+ SB projection. The speed cliff:
- **Elite:** Bobby Witt Jr. (31), Elly De La Cruz (38), Corbin Carroll (32)
- **Good:** Ronald Acuna Jr. (24), Gunnar Henderson (24), Trea Turner (26)
- **Last Call:** Pete Crow-Armstrong (projection?), Zach Neto (27), Maikel Garcia (26)

**Action:** If you don't have 20+ SB pace by Round 10, reach for the last speedsters.

---

# 2. Custom Data vs. Market Consensus (Your Edge)

## 2.1 Model Loves / Market Hates — TARGETS

Your Steamer projections + advanced analytics flag these undervalued players:

### 🔥 SLEEPER TARGETS

| Player | Your Model Rank | Yahoo ADP | Why Your Model Loves Them |
|--------|-----------------|-----------|---------------------------|
| **Bryan Woo** | SP #4 (2.95 ERA, 195 K) | #36 | Elite Stuff+ projection, SEA defense, ADP has him 12+ spots too low |
| **Jackson Merrill** | OF Top 15 | #60+ | 23 HR / 7 SB projection, SDP lineup context ignored by market |
| **Geraldo Perdomo** | SS Top 12 | #70+ | 19 SB, .365 OBP, multi-eligibility — perfect late MI fill |
| **Masyn Winn** | SS Sleeper | #90+ | 16 HR / 12 SB projection at SS — last starter-tier SS |
| **Brice Turang** | 2B Value | #46 | 20+ SB, solid AVG — your model sees 2B scarcity he represents |

**Draft Action:** Target these 1-2 rounds ahead of ADP. Your model sees what Yahoo drafters miss.

### 📊 Advanced Analytics Breakouts

Per your `ADVANCED_ANALYTICS_INTEGRATION.md`, target hitters with:
- **Blast% > 15%** (Gold standard for power)
- **Swing Length < 7.2 ft** (AVG floor, low K%)
- **Stuff+ > 115** for pitchers (breakout indicator)

| Breakout Candidate | Metric | Target Round |
|-------------------|--------|--------------|
| **Hunter Greene** | 100+ mph FB, high Stuff+ | R8-10 (risk discount) |
| **Garrett Crochet** | Your model's SP #3 | R4-6 (reach justified) |
| **Junior Caminero** | 32 HR projection at 3B | R15-18 (market sleeping) |

## 2.2 Model Flags / Market Loves — AVOIDS

Your analytics identify these overvalued players:

### 🚫 BUST CANDIDATES

| Player | Yahoo ADP | Your Model Warning | Why to Avoid |
|--------|-----------|-------------------|--------------|
| **Elly De La Cruz** | #8 | 38 SB but .263 AVG, 137 SO | ADP assumes peak — model sees risk |
| **Pete Alonso** | #25 | Moving to BAL, power decline | Not worth 2nd-3rd round pick |
| **Jazz Chisholm Jr.** | #22 | Injury flag in your data, NY pressure | Avoid flag in injury_flags_2026.csv |
| **Wander Franco** | N/A | Suspension/Restricted list | Check your position_eligibility_2026.csv status |
| **Tyler Glasnow** | SP Rank ~15 | 138 IP projection, injury history | Never healthy — let someone else take the risk |
| **Chris Sale** | #40 | Your model: injury_risk status | 165 IP cap, TJS history |

**Draft Action:** Let these players go past their ADP. If they fall 10+ spots, reconsider.

### ⚠️ REGRESSION RISKS (Per Your Analytics Engine)

Your `draft_analytics.py` flags players with:
- xwOBA_diff > +0.020 (lucky last year)
- xERA_diff < -0.40 (lucky ERA)

| Player Type | Red Flag | Action |
|-------------|----------|--------|
| High AVG hitters | .320+ AVG with .280 xBA | Sell high in trades, avoid at ADP |
| Low ERA pitchers | Sub-2.50 ERA with 3.50+ xERA | Regression coming — don't pay for last year |

---

# 3. External Tool Stack Integration

## 3.1 Tab 1: Fangraphs Roster Resource (Live Depth Charts)

**URL:** `https://www.fangraphs.com/roster-resource/`  
**What to Watch During Draft:**
- **Lineup Position:** Is your target batting 1st-3rd (more PA) or 7th-9th?
- **Platoon Risk:** Lefty hitters vs. LHP — check if they're in a platoon
- **Playing Time:** Ensure 500+ PA projection is realistic

**Draft Moment Usage:**
- Before picking any hitter Rounds 10+: Verify they're not in a platoon
- For closers: Cross-reference Roster Resource depth chart with your `closer_situations_2026.csv`

## 3.2 Tab 2: Baseball Savant (Statcast Leaderboards)

**URL:** `https://baseballsavant.mlb.com/leaderboard`  
**What to Watch:**
- **Exit Velocity / Barrel%:** Confirm your Blast% targets
- **Sprint Speed:** Verify SB projections for speed targets
- **Stuff+:** For pitchers — confirm your draft_analytics.py recommendations

**Draft Moment Usage:**
- When deciding between two similar hitters: Compare their Barrel%
- For SP targets: Check Stuff+ — 115+ is breakout territory

## 3.3 Tab 3: FantasyPros Draft Simulator (Market Pulse)

**URL:** `https://www.fantasypros.com/mlb/mock-draft/`  
**What to Watch:**
- **Live ADP Movement:** Is a player's ADP climbing/falling?
- **Expert Consensus:** How do experts rank vs. your model?
- **Position Runs:** Are drafters grabbing 2B early? Adjust your strategy.

**Draft Moment Usage:**
- During long waits between picks: Check if position runs are happening
- When considering a reach: See how early experts would take them

## 3.4 Tab 4: Rotowire MLB Lineups (Daily Confirmation)

**URL:** `https://www.rotowire.com/baseball/daily-lineups.htm`  
**What to Watch:**
- **Confirmed Starters:** For post-draft waiver streaming
- **Closer Usage Patterns:** Who's getting save chances?
- **Weather Delays:** Game PPDs affect weekly H2H matchups

**Draft Moment Usage:**
- Less critical during draft, essential for in-season management
- Bookmark for your daily lineup optimizer integration

## 3.5 Your Custom Application (Primary Tool)

**Your Dashboard Should Show:**
1. **Real-time pick counter** (Pick X of 276)
2. **Top available by z-score** (your Steamer projections)
3. **Position need tracker** (your roster composition gaps)
4. **Color-coded recommendations:**
   - 🟢 Must-have (top tier value)
   - 🟡 Good value (worth ADP)
   - 🔴 Skip (model flags bust/regression)
5. **Reach calculator:** ADP vs. your rank differential

---

# 4. The 24-Hour Checklist

## 4.1 T-Minus 24 Hours (Today, March 22)

### High Priority (Do These First)

- [ ] **Verify Yahoo OAuth Token**
  ```bash
  python -m backend.fantasy_baseball.yahoo_client
  ```
  - Ensure `YAHOO_REFRESH_TOKEN` is valid and not expired
  - Run test: Can you fetch league data?

- [ ] **Load Final Projections**
  ```bash
  python -m backend.fantasy_baseball.projections_loader
  ```
  - Confirm: "Loaded 461 batters, 251 pitchers"
  - Verify ADP merged: "ADP matched 306/712 players"

- [ ] **Test Draft Engine**
  ```bash
  python -m backend.fantasy_baseball.draft_analytics
  ```
  - Generate cheat sheet
  - Verify targets/avoid lists populate

### Medium Priority

- [ ] **Refresh Closer Situations**
  - Check MLB.com transaction wire for closer changes
  - Update `closer_situations_2026.csv` if needed
  - Cross-reference with Roster Resource

- [ ] **Check Injury Updates**
  - Review `injury_flags_2026.csv`
  - Any new TJS announcements? (Mason Miller trade?)
  - Update avoid flags if status changed

- [ ] **Download Latest ADP**
  - Re-scrape FantasyPros if possible
  - ADP moves in final 24 hours — your edge is freshness

## 4.2 T-Minus 12 Hours (Tonight, March 22)

- [ ] **Final Data Validation**
  - Run full pipeline: `projections_loader` → `draft_analytics`
  - Check for any CSV parsing errors
  - Verify z-scores calculate correctly

- [ ] **Mock Draft Simulation**
  - Run through 2-3 mock scenarios with your app
  - Test pick timing: Can you make decisions in 90 seconds?
  - Identify any UI lag or data loading issues

- [ ] **Set Up External Tool Tabs**
  - Open Fangraphs Roster Resource (bookmark your targets)
  - Open Baseball Savant (bookmark leaderboards)
  - Open FantasyPros (ADP page)
  - Test browser performance — close unnecessary tabs

## 4.3 T-Minus 2 Hours (Draft Morning, March 23)

- [ ] **System Health Check**
  - Restart your draft application fresh
  - Clear browser cache if needed
  - Test internet connection stability
  - Have backup device ready (phone with Yahoo app)

- [ ] **Final Cheat Sheet Print**
  - Generate and print `generate_draft_cheat_sheet()` output
  - Circle your top 5 targets per position
  - Mark your "do not draft" list

- [ ] **Discord/Notification Setup**
  - If using Discord integration, test webhook
  - Set phone notifications for draft reminders
  - Prepare draft channel: `#fantasy-draft-live`

## 4.4 T-Minus 30 Minutes (Pre-Draft)

- [ ] **Login and Verify**
  - Log into Yahoo Fantasy
  - Confirm draft room access
  - Verify keeper locks are correct
  - Check your draft slot/position

- [ ] **Mental Preparation**
  - Review your tiered rankings one last time
  - Set your first 3 picks strategy based on draft position
  - Remember: 90 seconds is plenty — don't panic pick

- [ ] **Environment Setup**
  - Minimize distractions
  - Coffee/water ready
  - Phone on silent except draft alerts
  - Two monitors recommended: Custom app + Yahoo draft room

---

# 5. Draft Day Execution Rules

## 5.1 The Golden Rules

1. **Trust Your Z-Scores** — Your Steamer projections are more accurate than Yahoo's ranks
2. **Position Scarcity > Best Player** — Taking a Tier 2 2B over a Tier 1 OF is often correct
3. **NSV is Sacred** — Net Saves category means blown saves hurt double — prioritize locked closers
4. **Never Reach > 20 Picks** — If your model says R8 value, don't take in R5
5. **Speed is Finite** — Only 12 players project 25+ SB — if you need speed, get it by Round 10

## 5.2 Round-by-Round Quick Reference

| Rounds | Focus | Key Decision Points |
|--------|-------|---------------------|
| 1-3 | Elite Bats | Judge/Soto/Ohtani tier — take best available |
| 4-6 | Pocket Aces | Skubal, Skenes, Crochet — last chance for SP1 |
| 7-9 | MI Premium | 2B/SS scarcity window — reach if needed |
| 10-14 | SP Depth + CI | Target high-K SPs, grab 1B/3B value |
| 15-18 | Closer Sprint | 2-3 NSV sources — check role certainty |
| 19-23 | Flyers | Blast% sleepers, Stuff+ breakouts |

## 5.3 When You're On the Clock

1. **Check your custom app** — Top available by z-score
2. **Verify position need** — Do you have this position filled?
3. **Check ADP delta** — Are you reaching? (OK if scarcity demands it)
4. **Quick tools check** — Fangraphs for lineup spot, Savant for confirmation
5. **Pick with confidence** — 90 seconds is plenty, don't rush

---

# 6. Post-Draft Actions

## 6.1 Immediate (Within 1 Hour)

- [ ] Export final roster from Yahoo
- [ ] Run roster analysis: Any category punts? Balance issues?
- [ ] Identify waiver wire priorities for Week 1

## 6.2 Week 1 Setup

- [ ] Activate daily lineup optimizer
- [ ] Set up closer situation monitoring
- [ ] Configure Discord alerts for your team

---

# Appendix: Quick Reference Tables

## Your Top 10 Steamer Rankings (vs. Yahoo ADP)

| Rank | Player | Your Model | Yahoo ADP | Delta | Action |
|------|--------|------------|-----------|-------|--------|
| 1 | Shohei Ohtani | #1 | #1 | 0 | Must draft if available |
| 2 | Aaron Judge | #2 | #2 | 0 | Anchor OF |
| 3 | Bobby Witt Jr. | #3 | #3 | 0 | 30/30 anchor |
| 4 | Juan Soto | #4 | #4 | 0 | Elite OBP anchor |
| 5 | Tarik Skubal | SP #1 | SP #1 | 0 | First SP off board |
| 6 | Paul Skenes | SP #2 | SP #2 | 0 | Elite stuff |
| 7 | Garrett Crochet | SP #3 | SP #4 | +1 | Model loves him |
| 8 | Cal Raleigh | C #1 | C #1 | 0 | Catcher anchor |
| 9 | Cristopher Sanchez | SP #5 | SP #6 | +1 | Reach if needed |
| 10 | Gunnar Henderson | SS #2 | SS #4 | +2 | Model higher than market |

## Position Scarcity Cliff Alert Thresholds

| Position | Alert Triggers At | Last Tier 1 Player |
|----------|-------------------|-------------------|
| C | 14 picks made | Cal Raleigh |
| 1B | 25 picks made | Vladimir Guerrero Jr. |
| 2B | 30 picks made | Ketel Marte |
| 3B | 20 picks made | Junior Caminero |
| SS | 35 picks made | Elly De La Cruz |
| OF | 45 picks made | Kyle Tucker |
| SP | 40 picks made | Cristopher Sanchez |
| RP | 50 picks made | Mason Miller |

---

**Document Generated:** March 22, 2026  
**For:** Treemendous League Draft (March 23, 2026 @ 7:30am EDT)  
**System:** Steamer 2026 Projections + Statcast Analytics + Yahoo ADP

**Good luck. Trust your model. Win your league.** 🏆
