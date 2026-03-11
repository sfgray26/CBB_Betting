# OPERATIONAL HANDOFF (EMAC-063)

> Ground truth as of March 10, 2026. Operator: Claude Code (Master Architect).
> See `IDENTITY.md` for risk policy · `AGENTS.md` for roles · `HEARTBEAT.md` for loops.
> Full roadmap: `docs/MLB_FANTASY_ROADMAP.md` · CBB plan: `tasks/cbb_enhancement_plan.md`

---

## 0. ARCHITECT DECISIONS (March 10, 2026)

### 0.1 Agent Role Change — Gemini CLI
**Decision:** Gemini CLI is now **Research-Only**. No production code from Gemini.
- **Was:** DevOps + code + research
- **Now:** Research, data sourcing, documentation, API investigation
- **Reason:** Code quality insufficient; Gemini PRs broke more than they fixed
- **All production code:** Claude Code only

Gemini research deliverables go to: `docs/` as markdown reports
Gemini research tasks: See Section 10 HANDOFF PROMPTS

### 0.2 CBB Model — Status
All P0–P4 + K-8/K-9/K-10 complete. 606/609 tests pass. Tournament-ready.
- Model V9.1 active with fatigue + sharp money + conference HCA + recency SD
- March Madness monitoring window: Mar 18 – Apr 7

### 0.3 Fantasy Baseball — Active Priority
- Yahoo OAuth: Fix deployed. User must complete one-time auth locally.
- Draft: March 23 @ 7:30am — draft board assistant needed by March 22
- Daily optimizer: Built (`daily_lineup_optimizer.py`) — wires sportsbook odds
- Full roadmap: `docs/MLB_FANTASY_ROADMAP.md`
- **Yahoo API Research: COMPLETE** — polling-based live draft tracker ready for implementation (see Section 5.4, Section 16)

---

## 1. EXECUTIVE SUMMARY (Updated)

**Status:** ⚠️ **INFRASTRUCTURE HEALTHY BUT MODEL ACCURACY DEGRADED**

Infrastructure is production-ready (Railway, Discord, scheduler all operational).
However, the core predictive model is running on only 1 of 3 rating sources —
this is the primary cause of limited returns. Enhancement plan written and ready
for implementation.

Previous deliverables still valid:
- **V9.1 Fatigue Model** — Captures schedule/travel/altitude edges
- **OpenClaw Lite** — 26,000× faster integrity checks, no Ollama dependency
- **Discord System** — Fully operational with fallback narratives
- **O-8 Baseline** — Ready for March 16 tournament prep

**Next Action:** P0 data audit → P1 sharp money → P2/P3 quick wins. All before Mar 18.

---

## 2. SYSTEM STATUS

### 2.1 Core Infrastructure

| Component | Status | Detail | Last Verified |
|-----------|--------|--------|---------------|
| Railway API | ✅ Healthy | All deps installing correctly, preflight checks pass | 2026-03-11 |
| Database | ✅ Connected | PostgreSQL operational, 365 teams loaded | 2026-03-11 |
| Scheduler | ✅ 10 jobs | Nightly@3AM, outcomes every 2h, lines every 30m | 2026-03-11 |
| Discord | ✅ Ready | Token configured, template fallbacks active | 2026-03-11 |
| Streamlit | ✅ Fixed | Expander key error resolved | 2026-03-11 |
| V9.1 Model | ✅ Active | Fatigue integration live | 2026-03-11 |

### 2.2 Model Components

| Feature | Status | File | Tests |
|---------|--------|------|-------|
| Fatigue Model (K-8) | ✅ LIVE + WIRED | `backend/services/fatigue.py` | 23 pass ✅ |
| OpenClaw Lite (K-9) | ✅ LIVE | `backend/services/openclaw_lite.py` | 18 pass ✅ |
| Sharp Money (P1) | ✅ LIVE + WIRED | `backend/services/sharp_money.py` | 15 pass ✅ |
| Conference HCA (P2) | ✅ LIVE + WIRED | `conference_hca.py` + `team_conference_lookup.py` | 35 pass ✅ |
| Recency Weight (P3) | ✅ LIVE + WIRED | `backend/services/recency_weight.py` | 20 pass ✅ (5% SD bump in March) |
| Seed-Spread Scalars (A-26) | ✅ LIVE | `betting_model.py` | 26 pass |
| Tournament SD Bump | ✅ LIVE | `betting_model.py` (1.15x neutral) | Active |
| Line Movement Monitor | ✅ LIVE | `odds_monitor.py` | Runs 30m |
| Integrity Sweep | ✅ LIVE | `scout.py` + Lite fallback | Async 8-worker |

### 2.3 Upcoming Deadlines

| Date | Event | Status | Owner |
|------|-------|--------|-------|
| **Mar 16 ~9 PM ET** | O-8 Baseline Execution | ⏳ Ready | OpenClaw |
| **Mar 18** | First Four Begins | ⏳ Monitor | All |
| **Mar 20** | Fantasy Keeper Deadline | ⚠️ 9 days | TBD |
| **Mar 23** | Fantasy Draft Day | ⚠️ 12 days | TBD |

---

## 3. COMPLETED WORK (March 10-11)

### 3.0 Claude Code — EMAC-061: Service Wiring + Game 248 Fix

**EMAC-060 bug fixes (carried over):**
- `sharp_money.py`: Missing `Tuple` import + `SharpSignal.none()` bad kwarg + detection guards fixed
- `fatigue.py`: `<= 100` boundary, steeper long-distance formula (2000mi → 0.85, 3000mi → 1.0)
- `conference_hca.py`: Added `'c-usa'` variation
- `recency_weight.py`: Negative `days_ago` clamp to 0
- `openclaw_lite.py`: Reordered decision logic — CAUTION before VOLATILE for multi-risk signals
- Fantasy baseball: 161 duplicate players removed (two-way players)

**EMAC-061 deliverables:**

**1. Game 248 force-delete fix** (`backend/main.py`):
- Added `force: bool = Query(default=False)` to `DELETE /admin/games/{game_id}`
- When `?force=true`: deletes BetLogs first, then predictions, closing_lines, then game
- Without force: same 409 error with improved message ("use ?force=true")
- Response now includes `bet_logs_deleted` count

**2. Service wiring** (`backend/services/analysis.py`):
- **Fatigue**: After matchup engine, queries DB for each team's last completed game, calls `calculate_game_fatigue()`, passes `fatigue_margin_adj` to `analyze_game()` (param already existed, was never passed)
- **Recency SD bump**: After dynamic_base_sd computed from game total, applies `*= 1.05` during March (is_late_season())
- **Sharp money**: After `analyze_game()`, pulls line history from OddsMonitor, calls `detect_sharp_signal()`, calls `apply_sharp_adjustment()` to edge (±0.5–0.8%)
- **model_version**: Updated from hardcoded `"v9.0"` → `"v9.1"` in both update and create paths

**3. Conference HCA — COMPLETE** (Mission K-10):
- `team_conference_lookup.py` built with CSV-backed lookup
- `team_conference_map.csv` covers 365 D1 teams
- Wired into analysis.py after fatigue block (adds delta to matchup_margin_adj)

**Test result: 595/598 pass (3 pre-existing DB-auth failures, unchanged)**

### 3.1 Kimi CLI Deliverables

| Mission | Files | Lines | Tests | Status |
|---------|-------|-------|-------|--------|
| **K-10: Conf HCA Wiring** | `team_conference_lookup.py`, `team_conference_map.csv` | 350 | 35 | ✅ Complete |
| **K-8: Fatigue Model** | `fatigue.py`, `docs/FATIGUE_MODEL.md` | 530 | 23 | ✅ Merged |
| **K-9: OpenClaw Lite** | `openclaw_lite.py`, migration | 200 | 18+12 | ✅ Merged |
| **O-8: Baseline Script** | `openclaw_baseline.py`, tests | 500 | 5 | ✅ Ready |
| **D-1: Discord Cleanup** | `scout.py` fallbacks, fixes | 180 | 5 | ✅ Merged |
| **P0: Data Audit** | Verified 2-source (KP+BT) working | — | — | ✅ Complete |
| **P1: Sharp Money** | `sharp_money.py`, `test_sharp_money.py` | 400 | 15+ | ✅ Merged |
| **P2: Conf HCA** | `conference_hca.py`, `test_conference_hca.py` | 280 | 18+ | ✅ Merged |
| **P3: Recency** | `recency_weight.py`, `test_recency_weight.py` | 350 | 20+ | ✅ Merged |
| **P4: Recal Audit** | `recalibration_audit.py`, endpoint | 200 | — | ✅ Merged |
| **Railway Fix** | `railway.toml`, `preflight_check.py` | 150 | — | ✅ Merged |
| **Build Fix** | Fixed `cloudscraper>=2.3.7` → `>=1.2.0` | 1 | — | ✅ Merged |

**Total:** ~3,400 lines added, 110+ tests, 19 commits pushed.

### 3.2 P1: Sharp Money Detection (NEW)

**Files:** `backend/services/sharp_money.py` (400 lines)
**Tests:** `tests/test_sharp_money.py` (280 lines, 15 tests)

**Features Implemented:**
1. **Steam Detection** — Rapid ≥1.5 pt moves in <30 minutes
2. **Opener Gap Detection** — Large divergence from opening line (≥2.0 pts)
3. **Reverse Line Movement** — Line moves against public betting %
4. **Edge Adjustment** — Auto-adjust model edge based on signal alignment

**Configuration (env vars):**
```
STEAM_THRESHOLD_PTS=1.5
STEAM_WINDOW_MINUTES=30
OPENER_GAP_THRESHOLD=2.0
RLM_PUBLIC_THRESHOLD=60
```

**Integration:**
```python
from backend.services.sharp_money import detect_sharp_signal, apply_sharp_adjustment

# Get signal
signal = detect_sharp_signal(game_key, line_history, current_spread)

# Apply to edge
adjusted_edge, details = apply_sharp_adjustment(base_edge, signal, model_side)
```

**Discord Alerts:** Added `send_ratings_source_alert()` to coordinator.py

### 3.3 P2: Conference-Specific HCA (NEW)

**Files:** `backend/services/conference_hca.py` (220 lines)
**Tests:** `tests/test_conference_hca.py` (200 lines, 18 tests)

**Conference HCA Table:**
| Conference | HCA | Difficulty |
|------------|-----|------------|
| Big Ten | 3.6 | EXTREME |
| Big 12 | 3.4 | EXTREME |
| SEC | 3.2 | HIGH |
| ACC | 3.0 | HIGH |
| Big East | 2.9 | MODERATE |
| WCC | 2.7 | MODERATE |
| AAC | 2.6 | MODERATE |
| A-10 | 2.5 | MODERATE |
| SWAC | 1.5 | LOW |
| MEAC | 1.5 | LOW |
| Neutral | 0.0 | N/A |

**Usage:**
```python
from backend.services.conference_hca import get_conference_hca, apply_conference_hca

# Get HCA for conference
hca = get_conference_hca("Big Ten", is_neutral=False)  # Returns 3.6

# Apply with pace adjustment
adjusted_hca, meta = apply_conference_hca("SEC", pace_ratio=1.05)
```

**Features:**
- Name normalization (handles "Big Ten", "B1G", "big ten", etc.)
- Pace-adjusted HCA scaling
- Neutral site override (0.0)
- Difficulty ratings for road games

### 3.3 K-10: Conference HCA Wiring — Mission Complete ✅

**Files Created:**
- `data/team_conference_map.csv` — 365 D1 teams with conference assignments
- `backend/services/team_conference_lookup.py` — Lookup service with HCA delta calculation

**Files Modified:**
- `backend/services/analysis.py` — Wired conference HCA into fatigue block
- `tests/test_conference_hca.py` — Added 11 new tests (35 total)

**Conference Map Coverage:**
- All Power conferences (ACC, Big Ten, Big 12, SEC, Big East, Pac-12)
- Major mid-majors (WCC, AAC, A-10, MWC, MVC, WAC)
- All low-majors (SWAC, MEAC, Ivy, etc.)
- **Total: 365 teams mapped**

**Integration:**
```python
from backend.services.team_conference_lookup import get_conference_hca_delta

# After fatigue block in analysis.py
conf_hca_delta = get_conference_hca_delta(home_team, is_neutral)
matchup_margin_adj += conf_hca_delta  # Add delta vs baseline 3.09
```

**HCA Delta Logic:**
- Returns conference HCA - 3.09 baseline
- Big Ten home game: 3.6 - 3.09 = **+0.51** added to home margin
- ACC home game: 3.0 - 3.09 = **-0.09** (slight penalty vs baseline)
- Neutral site: Always 0.0
- Unknown team: Uses DEFAULT_HCA (2.5) - 3.09 = **-0.59**

**New Tests (35 total in test_conference_hca.py):**
- `test_team_lookup_duke_acc` — Duke → ACC
- `test_team_lookup_gonzaga_wcc` — Gonzaga → WCC
- `test_team_lookup_unknown_returns_none` — Unknown → None
- `test_conference_hca_for_team_big_ten` — MSU → 3.6 HCA
- `test_conference_hca_for_team_default_on_unknown` — Unknown → DEFAULT_HCA
- `test_neutral_site_returns_zero` — Neutral site = 0.0
- `test_hca_delta_calculation` — Delta math correct
- `test_hca_delta_neutral_returns_zero` — Neutral delta = 0.0
- `test_hca_delta_unknown_returns_default_delta` — Unknown delta = -0.59
- `test_case_insensitive_team_lookup` — Case-insensitive matching
- `test_multiple_teams_per_conference` — All ACC teams correct

**Test Result: 35/35 pass (606 total passed, 3 pre-existing failures)**

### 3.4 P3: Late-Season Recency Weighting (NEW)

**Files:** `backend/services/recency_weight.py` (280 lines)
**Tests:** `tests/test_recency_weight.py` (220 lines, 20 tests)

**Recency Weight Table (Late Season):**
| Days Ago | Weight | Period |
|----------|--------|--------|
| 0-2 | 2.0x | Very Recent |
| 3-7 | 1.6-1.9x | Last Week |
| 8-14 | 1.2-1.5x | Two Weeks |
| 15-21 | 1.0-1.1x | Three Weeks |
| 22+ | 1.0x | Older |

**Tournament Mode (March 15+):**
- Neutral site override: HCA = 0.0
- Margin SE inflation: +0.20 (higher upset variance)
- Form window: Last 14 days only
- Recency weights: Active

**Usage:**
```python
from backend.services.recency_weight import (
    is_late_season, 
    get_recency_weight,
    get_tournament_adjustments
)

# Check season phase
if is_late_season():
    weight = get_recency_weight(days_ago=5)  # Returns 1.7x

# Get tournament adjustments
adj = get_tournament_adjustments(is_neutral=True)
# Returns: margin_se_inflation=0.20, form_window_days=14
```

### 3.5 P4: Recalibration Audit (NEW)

**Files:** `scripts/recalibration_audit.py` (200 lines)
**Endpoint:** `GET /admin/recalibration/audit`

**Purpose:** Validate recalibration pipeline before tournament

**Checks:**
1. **Data Sufficiency** — Count settled bets with prediction links (need ≥30)
2. **Current Parameters** — home_advantage, sd_multiplier from model_parameters
3. **Drift Detection** — Compare to baselines (HA=3.09, SD=0.85)
4. **Recency** — Days since last recalibration

**CLI Usage:**
```bash
# Run audit
python scripts/recalibration_audit.py

# Output includes:
# - Settled bets count
# - Current parameter values
# - Drift percentages
# - Recommendations
```

**API Response:**
```json
{
  "settled_bets": 45,
  "sufficient_data": true,
  "home_advantage": 3.12,
  "sd_multiplier": 0.82,
  "ha_drift_pct": 1.0,
  "sd_drift_pct": 3.5,
  "drift_alert": false,
  "days_since_recalibration": 2,
  "recommendations": {
    "needs_more_data": false,
    "stale_recalibration": false,
    "parameter_drift": false
  }
}
```

### 3.6 P0: Data Pipeline Audit (COMPLETE)

**Findings:**
- ✅ BartTorvik: Public CSV working (365 teams, no auth needed)
- ✅ EvanMiya: Intentionally dropped (Cloudflare, 2-source mode by design)
- ✅ Model running 2-source composite (KenPom 51% / BartTorvik 49%)
- ✅ `/admin/ratings/status` endpoint exists
- ✅ Discord alerts fire when <2 sources active
- ✅ `BARTTORVIK_USERNAME/PASSWORD` are legacy (only in docs)

### 3.6 Key Technical Decisions

1. **Ollama Dependency Removed** — All LLM functions now have template fallbacks
2. **Graceful Degradation Chain** — Ollama → OpenClaw Lite → Seed-based defaults
3. **Fatigue Model Integration** — Added `fatigue_margin_adj` param to `analyze_game()`
4. **Conference HCA** — Replaces flat 3.09 with conference-specific values
5. **Recency Weighting** — 2x weight for recent games in March (late season)
6. **V9.1 Version Bump** — Model version tracks features (fatigue = v9.1)

### 3.7 Documentation Created

- `docs/FATIGUE_MODEL.md` — Fatigue model specification
- `docs/OPENCLAW_LITE_PLAN.md` — Migration analysis
- `docs/UAT_MARCH_10_2026.md` — Test results (58 tests, 100%)
- `docs/RAILWAY_DISCORD_FIX.md` — Troubleshooting guide
- `docs/SHARP_MONEY.md` — Sharp money detection spec (NEW)
- `docs/CONFERENCE_HCA.md` — Conference HCA guide (NEW)
- `docs/RECENCY_WEIGHTING.md` — Late-season weighting guide (NEW)
- `SYSTEM_STATUS.md` — Full system overview

---

## 4. RECOMMENDED NEXT ENHANCEMENTS

### 4.1 High Priority (Tournament Value)

#### **E-1: Sharp Money / Steam Detection**
**Impact:** High — detects when sharp money moves lines opposite to public
**Effort:** Medium (~200 lines)
**Implementation:**
```python
# New: backend/services/sharp_money.py
def detect_sharp_money(
    open_line: float,
    current_line: float,
    public_betting_pct: float,  # From The Action Network or similar
) -> Dict[str, Any]:
    """
    Returns:
        - sharp_side: "home" | "away" | None
        - confidence: 0.0-1.0
        - pattern: "reverse_line_movement" | "steam" | "none"
    """
```
**Files:** New service + integration into `odds_monitor.py`

#### **E-2: Conference-Specific Home Court Advantage**
**Impact:** Medium — better HCA for Big Ten vs SWAC games
**Effort:** Low (~50 lines)
**Implementation:**
```python
CONFERENCE_HCA = {
    "Big Ten": 3.6, "Big 12": 3.4, "SEC": 3.2,
    "ACC": 3.0, "Pac-12": 2.8, "Mid-major": 2.5
}
```
**Files:** Modify `betting_model.py` HCA calculation

#### **E-3: Late-Season Weight Adjustment**
**Impact:** Medium — weight recent games more heavily in March
**Effort:** Low (~30 lines)
**Implementation:** Add time-decay factor to ratings composite
**Files:** `betting_model.py` margin calculation

### 4.2 Medium Priority (Post-Tournament)

#### **E-4: ML-Based Recalibration**
**Impact:** High — learn optimal parameters from CLV data
**Effort:** High (~500 lines + model training)
**Implementation:** XGBoost model for parameter prediction
**Files:** New `ml_recalibration.py` service

#### **E-5: Live/In-Play Betting Engine**
**Impact:** High — second half lines, live win probability
**Effort:** High (~800 lines)
**Implementation:** Markov state machine for live game simulation
**Files:** New `live_betting.py` module

### 4.3 Lower Priority (Nice to Have)

#### **E-6: Weather/Travel Delay Effects**
**Impact:** Low — rare edge cases
**Effort:** Low (~40 lines)

#### **E-7: Alternative Line Shopping**
**Impact:** Medium — better value on alt spreads
**Effort:** Medium (~150 lines)

---

## 5. FANTASY BASEBALL STATUS

### 5.1 Current State
- 244 players in database
- Draft engine functional
- Keeper engine fixed
- **Needs:** Yahoo OAuth, Steamer CSVs (300+ rows), keeper UI polish

### 5.2 Deadlines
- **Mar 20:** Keeper deadline (9 days)
- **Mar 23:** Draft day (12 days)

### 5.3 Recommendation
Complete Fantasy Baseball Phase 0 before March 18 (tournament starts), OR defer until after tournament (April 7+).

---

## 5.4 YAHOO FANTASY API RESEARCH (Kimi CLI — March 11, 2026)

**Full Report:** `reports/YAHOO_FANTASY_API_RESEARCH.md`

### Key Findings for Draft Day (March 23)

#### 🟡 Live Draft Capabilities — LIMITED

**What IS Possible:**
- ✅ **Polling-based updates** — Call `draft_results()` every 5-10 seconds during live draft
- ✅ Returns picks as they happen (pick number, round, team_key, player_id)
- ✅ Filter available players with `status='A'` parameter
- ✅ Updates during the draft with players drafted thus far

**What's NOT Available:**
- ❌ **No websockets or push notifications** — must poll
- ❌ **No real-time "on the clock" notifications**
- ❌ **No auction nomination tracking**
- ❌ **No draft timer visibility**

#### Core Endpoints

```python
from yahoo_fantasy_api import League

# Get draft results (poll this every 5 seconds during draft)
draft_results = league.draft_results()
# Returns: [{'pick': 1, 'round': 1, 'team_key': '...', 'player_id': 9490}, ...]

# Get available players
available = league.players(status='A', count=1000)

# Get player details
player_details = league.player_details(player_id)
```

#### Recommended Implementation Pattern

```python
async def poll_draft_updates(league, last_pick_count=0):
    """Poll draft_results() every 5-10 seconds during live draft"""
    while True:
        draft_results = league.draft_results()
        current_picks = len(draft_results)
        
        if current_picks > last_pick_count:
            new_picks = draft_results[last_pick_count:]
            for pick in new_picks:
                await notify_discord_new_pick(pick)
            last_pick_count = current_picks
            
        await asyncio.sleep(5)  # 5-second polling interval
```

#### Alternative Data Sources

| Source | Best For | Access |
|--------|----------|--------|
| **Fangraphs** | Projections (ATC, Steamer, ZiPS) | CSV Download |
| **MLB Stats API** | Real stats, player info | Free, no key needed |
| **FantasyPros** | ADP, rankings | Web scraping |
| **Baseball-Reference** | Historical stats | Limited API |

#### Integration Architecture

**Pre-Draft Setup:**
1. Load Fangraphs projections CSV (ATC recommended)
2. Load FantasyPros ADP data
3. Merge with Yahoo player pool via name normalization
4. Build player ID mapping table

**During Live Draft:**
1. Poll `draft_results()` every 5 seconds
2. Send Discord notification to `#fantasy-draft` channel
3. Update available player pool locally
4. Check if it's our pick (infer from pick number + draft order)
5. Send "ON THE CLOCK" alert to Discord with recommendations

#### Environment Variables Needed

```bash
# Already configured
YAHOO_CLIENT_ID=...
YAHOO_CLIENT_SECRET=...
YAHOO_REFRESH_TOKEN=...
YAHOO_LEAGUE_ID=...

# Discord channel (already set up)
DISCORD_CHANNEL_FANTASY_DRAFT=1481294129450319893
```

#### Implementation Checklist for Claude Code

**Before March 22:**
- [ ] Create `backend/fantasy_baseball/draft_tracker.py` with polling loop
- [ ] Wire `draft_results()` polling to Discord `#fantasy-draft` channel
- [ ] Add "ON THE CLOCK" detection (track pick number vs our draft slot)
- [ ] Integrate with existing player board for recommendations
- [ ] Test with mock draft data

**Draft Day (March 23 @ 7:30am ET):**
- [ ] Start polling loop 15 minutes before draft
- [ ] Monitor rate limits (max 1 req/5 sec)
- [ ] Have manual override ready if API fails

---

---

## 5.5 ADVANCED ANALYTICS RESEARCH (Gemini CLI — March 11, 2026)

**Full Report:** `reports/ADVANCED_ANALYTICS_INTEGRATION.md`

### Summary of Findings

Gemini researched cutting-edge 2026 baseball metrics for fantasy evaluation. Here's what's actionable:

#### 🎯 The "Triple Crown" Hitting Metrics (Statcast Bat Tracking)

| Metric | Elite Threshold | What It Means | Data Source |
|--------|-----------------|---------------|-------------|
| **Bat Speed** | 75+ mph | Power potential | Baseball Savant (statcast) |
| **Squared-Up%** | 35%+ | Barrel control/accuracy | Baseball Savant |
| **Blast%** | 15%+ | Fast + Squared Up = Elite power | Derived from Statcast |
| **Swing Length** | < 7.2 ft | Contact ability/Low K% | Baseball Savant |

#### ⚾ Next-Gen Pitching Metrics

| Metric | Elite Threshold | What It Means | Data Source |
|--------|-----------------|---------------|-------------|
| **Stuff+** | 115+ | Physical pitch quality | FanGraphs ($) or Baseball Prospectus |
| **PLV** | 5.5+ | Pitch-level value | PitcherList.com (scrapeable) |
| **SSW** | High | Seam-Shifted Wake ("invisible" movement) | Baseball Savant R&D |

### Feasibility Assessment

| Metric | Accessible? | Method | Priority |
|--------|-------------|--------|----------|
| Bat Speed | ⚠️ Partial | pybaseball library, limited 2026 coverage | Medium |
| Squared-Up% | ⚠️ Partial | Baseball Savant CSV search | Medium |
| Blast% | ⚠️ Calculate | Derive from Statcast data | Medium |
| Swing Length | ✅ Yes | pybaseball.statcast_batter() | High |
| Stuff+ | ❌ No | Behind paywall (FanGraphs $) | Low |
| PLV | ⚠️ Maybe | PitcherList.com charts | Low |

### Immediate Actions for Claude Code

**Phase 1: Statcast Integration (Post-Draft)**
```python
# Use pybaseball to fetch available metrics
from pybaseball import statcast_batter, playerid_lookup

# Get bat speed, swing length for player pool
# Note: 2026 data may be limited until season starts
```

**Phase 2: Player Board Enhancement**
- Add columns: `bat_speed`, `swing_length`, `blast_rate` (if available)
- Use as tiebreaker when z-scores are close
- Flag "hidden gems": Low ADP + elite swing metrics

**Phase 3: Sleeper Identification Filter**
```python
# Example filter from Gemini research
def identify_sleepers(players):
    return [
        p for p in players
        if p['adp'] > 150  # Late round
        and p.get('swing_length', 999) < 7.2  # Short swing = contact
        and p.get('squared_up_pct', 0) > 30  # Good barrel control
    ]
```

### Research Quality Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Metric Definitions** | ⭐⭐⭐⭐⭐ | Clear, accurate descriptions |
| **Thresholds** | ⭐⭐⭐⭐⭐ | Specific, actionable numbers |
| **Data Source Viability** | ⭐⭐⭐☆☆ | Some metrics paywalled or 2026-limited |
| **Implementation Path** | ⭐⭐⭐☆☆ | Suggests file that doesn't exist yet |
| **Overall** | ⭐⭐⭐⭐☆ | Good strategic guidance, execution TBD |

### Recommendation

**Before Draft (March 23):**
- ❌ Do NOT attempt to integrate — too late, data may not be ready

**After Draft (Season Start):**
- ✅ Add pybaseball Statcast integration to daily optimizer
- ✅ Pull 2026 bat tracking data as it becomes available
- ✅ Use for waiver wire sleeper identification

**For 2027 Draft:**
- ✅ Build full Statcast pipeline in offseason
- ✅ Create "Blast%" and "Swing Length" columns in projections

---

## 6. NEXT PRIORITY (March 11-17)

**Status:** P0 ✅ | P1 ✅ | **P2/P3 NEXT**

| Option | Work | Timeline | Tournament Value | Status |
|--------|------|----------|------------------|--------|
| **P0** | Data Pipeline Audit | 0.5 day | Critical | ✅ Complete |
| **P1** | Sharp Money Detection | 1 day | High | ✅ Complete |
| **P2** | Conference-Specific HCA | 0.5 day | Medium | ✅ Complete |
| **P3** | Late-Season Recency Weighting | 0.5 day | Medium | ✅ Complete |
| **P4** | Recalibration Audit | 0.5 day | Medium | ✅ Complete |
| **C** | Fantasy Baseball completion | 3-4 days | Time-sensitive (Mar 20) | ⚠️ Defer |
| **P4** | Recalibration Audit | 0.5 day | Medium | Post-tournament |
| **E** | ML Recalibration (E-4) | 1-2 weeks | High | Post-tournament |

**Kimi recommendation:** P2 (Conf HCA) → P3 (Recency) → defer C until after tournament.

---

## 6.5 OPENCLAW DISCORD ENHANCEMENT PLAN (OPCL-001)

**Full Plan:** `reports/OPENCLAW_DISCORD_ENHANCEMENT_PLAN.md`  
**Status:** Ready for implementation  
**Priority:** HIGH — 10x impact multiplier for OpenClaw  
**Assigned to:** Claude Code (implementation) + OpenClaw (execution)

### Problem Statement

Discord's 16-channel architecture is deployed, but OpenClaw is underutilized:
- Current: Integrity checks during nightly analysis only (~1x/day)
- Target: Continuous intelligence hub (15-20x/day)
- **Impact gap:** 10-20x underperformance vs potential

### The 5 Pillars

| Pillar | Channel | Frequency | Impact |
|--------|---------|-----------|--------|
| **1. Morning Brief** | `#openclaw-briefs` | Daily 7 AM ET | Overnight summary + today's slate |
| **2. Live Monitor** | `#openclaw-escalations` | Every 2h on game days | Game-time news alerts |
| **3. Fantasy Intel** | `#fantasy-news` | Daily 7 AM + ad-hoc | Closer/lineup/waiver alerts |
| **4. Telemetry** | `#openclaw-health` | Hourly | System health dashboard |
| **5. Tournament Ops** | `#cbb-tournament` | Mar 18-Apr 7 | Elite Eight+ special monitoring |

### Implementation Phases

**Phase 1: Foundation (March 12-18)** — PRE-TOURNAMENT
- [ ] `openclaw_briefs.py` — Daily 7 AM morning brief
- [ ] `openclaw_telemetry.py` — System health dashboard
- [ ] Discord webhook helpers
- [ ] **Deliverable:** Daily briefs active by March 18

**Phase 2: Live Operations (March 19-25)** — TOURNAMENT BEGINS
- [ ] `openclaw_live_monitor.py` — Game-time monitoring
- [ ] Tournament mode activation
- [ ] Upset/cinderella tracking
- [ ] **Deliverable:** Live monitoring for First Four & Round of 64

**Phase 3: Fantasy Integration (March 26-April 7)** — TOURNAMENT + FANTASY
- [ ] `openclaw_closer_monitor.py` — Fantasy closer tracking
- [ ] Lineup confirmation alerts
- [ ] Waiver wire sleeper detection
- [ ] **Deliverable:** Fantasy alerts active by MLB Opening Day

**Phase 4: Polish (Post-Tournament)**
- [ ] Historical analysis
- [ ] Self-tuning thresholds
- [ ] Full documentation
- [ ] **Deliverable:** OpenClaw v4.0 autonomous operations

### Morning Brief Example

```markdown
🌅 OPENCLAW MORNING BRIEF — March 18, 2026

📊 TODAY'S SLATE
• 4 CBB games analyzed
• 2 BET-tier recommendations  
• 1 CONSIDER (monitor for line moves)

🔍 INTEGRITY SUMMARY
• All games: CONFIRMED ✓
• No injuries or lineup concerns

⚡ SHARP MONEY ALERTS
• Gonzaga -3.5 → -4.5 (steam detected)
• Recommendation: Wait for line to stabilize

🏀 TOURNAMENT WATCH
• First Four: 2 games tonight
• Upset Probability: 28%

📋 ESCALATION QUEUE
• 0 high-stakes games pending review
```

### Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Intelligence messages/day | 0.1 | 15-20 |
| Time to alert (late news) | Manual | <15 min |
| Escalation resolution | 24h | <4h |
| Tournament coverage | Basic | Full (67 games) |

### Immediate Actions for Claude Code

**This Week (Priority 1 after EMAC-064):**
1. Implement `backend/services/openclaw_briefs.py`
2. Add `send_morning_brief()` to `discord_notifier.py`
3. Create cron trigger in scheduler
4. Test end-to-end by March 18

**Reference:** Full specification in `reports/OPENCLAW_DISCORD_ENHANCEMENT_PLAN.md`

---

## 7. ENVIRONMENT VARIABLES (Railway)

### Required (All Set ✅)
```
DATABASE_URL=postgresql://...
THE_ODDS_API_KEY=...
KENPOM_API_KEY=...
API_KEY_USER1=...
DISCORD_BOT_TOKEN=MTQ3NzQwOTg4NTA0OTMyMzgxNA.GnEBcJ...
DISCORD_CHANNEL_ID=1477436117426110615
```

### Optional (Not Required)
```
BARTTORVIK_USERNAME/PASSWORD (not set)
EVANMIYA_API_KEY (not set)
TWILIO_* (not set)
SENDGRID_API_KEY (not set)
```

---

## 8. ACTIVE MISSIONS

### 8.1 OpenClaw — O-8 Baseline Execution
**When:** March 16, 2026 ~9:00 PM ET
**Command:** `python scripts/openclaw_baseline.py --year 2026`
**Output:** `data/pre_tournament_baseline_2026.json` + `reports/o8_baseline_summary_2026.md`
**Status:** ⏳ Ready for autonomous execution

### 8.1a OpenClaw — OPCL-001 Discord Enhancement (NEW MISSION)
**When:** Phase 1: March 12-18; Phase 2: March 19-25  
**Owner:** Claude Code (implementation) → OpenClaw (execution)  
**Full Plan:** `reports/OPENCLAW_DISCORD_ENHANCEMENT_PLAN.md`

**Phase 1 Deliverables (by March 18):**
- [ ] `openclaw_briefs.py` — Daily 7 AM morning brief to `#openclaw-briefs`
- [ ] `openclaw_telemetry.py` — Hourly system health to `#openclaw-health`
- [ ] Integration with `discord_notifier.py`

**Phase 2 Deliverables (March 19-25):**
- [ ] `openclaw_live_monitor.py` — Game-time monitoring every 2h
- [ ] Tournament mode activation
- [ ] Cinderella/upset tracking for March Madness

**Phase 3 Deliverables (March 26-April 7):**
- [ ] `openclaw_closer_monitor.py` — Fantasy baseball closer alerts
- [ ] Lineup confirmation notifications
- [ ] Waiver wire sleeper detection

**Impact Target:** 10x increase in OpenClaw value delivery (0.1 → 15-20 msgs/day)

### 8.2 Gemini CLI — Research Missions

**COMPLETED RESEARCH:**

#### G-R6: Advanced Baseball Analytics (COMPLETE ✅)
**File:** `reports/ADVANCED_ANALYTICS_INTEGRATION.md`
**Delivered:** March 11, 2026
**Contents:**
- Statcast Bat Tracking metrics (Bat Speed, Squared-Up%, Blast%, Swing Length)
- Next-gen pitching metrics (Stuff+, PLV, Seam-Shifted Wake)
- Plate discipline framework (Heart%, Shadow Zone, Waste%)
- 2026 benchmark thresholds for fantasy evaluation

**Assessment:** Research is sound but data sources are partially paywalled/limited for 2026 season. Recommend post-draft integration via pybaseball.

**PENDING VERIFICATION:**

#### G-16: O-10 Line Monitor Verification
**Task:** Verify O-10 line monitor post-deploy
**Status:** Pending verification

### 8.3 Kimi CLI — Enhancement Development
**Status:** ✅ P0-P3 COMPLETE | Ready for tournament monitoring

**COMPLETED:**
- P0: Data audit — BartTorvik confirmed working, 2-source mode active
- P1: Sharp Money Detection — steam, opener gap, RLM detection live
- P2: Conference HCA — Big Ten 3.6, Big 12 3.4, etc. with pace adjustment
- P3: Late-season recency — 2x weight for last 3 days, tournament mode ready

**READY:**
- Tournament monitoring mode (Mar 18-Apr 7)
- O-8 Baseline execution (Mar 16 ~9 PM ET)
- Fantasy Baseball Phase 0 (deferred to post-tournament)

---

## 9. HIVE WISDOM (Updated)

| Lesson | Source |
|--------|--------|
| Conference HCA: Big Ten 3.6 pts vs SWAC 1.5 pts = significant road differential | P2 |
| Recency weighting: 2x for last 3 days, 1.6x for last week in March | P3 |
| Tournament mode: neutral HCA=0, margin SE +0.20, 14-day form window | P3 |
| Sharp money detection: steam ≥1.5 pts in <30 min = high confidence signal | P1 |
| Opener gap ≥2.0 pts suggests market correction toward sharp opinion | P1 |
| RLM detection requires public betting % data (Action Network integration) | P1 |
| Fatigue model adds 0.5-2.0 point edge in B2B/altitude spots | K-8 |
| OpenClaw Lite: 26,000× faster, 100% match rate vs Ollama | K-9 |
| BartTorvik public CSV needs no auth (cloudscraper only) | P0 |
| EvanMiya intentionally dropped — 2-source mode robust by design | P0 |
| Bat Speed 75+ mph correlates more with HR potential than raw Exit Velocity | G-R6 |
| Blast% (Fast + Squared-Up) is the single best predictor of elite power | G-R6 |
| Swing Length < 7.2 ft identifies high-AVG, low-K% sleepers | G-R6 |
| Stuff+ 115+ identifies high K/9 breakout candidates (paywalled) | G-R6 |
| OpenClaw morning brief: 7 AM daily intelligence summary for operator | OPCL-001 |
| OpenClaw telemetry: Hourly system health visibility prevents surprises | OPCL-001 |
| OpenClaw live monitor: 2-hour checks catch late-breaking news | OPCL-001 |
| Tournament mode: Elite Eight+ special monitoring with Kimi escalation | OPCL-001 |
| Template fallbacks essential when Ollama unavailable | D-1 |
| `key` parameter breaks older Streamlit versions | D-1 |
| Railway needs explicit `railway.toml` for reliable builds | Railway Fix |
| Discord token must be set in Railway Variables, not just `.env` | D-1 |
| cloudscraper 1.x series latest — 2.x doesn't exist (build fix) | Build Fix |

---

## 10. HANDOFF PROMPTS

### CLAUDE CODE (Master Architect) — EMAC-063: Fantasy Baseball Draft Board + Live Tracker
```
MISSION: Build Draft Board assistant + Live Draft Tracker by March 22 (draft is March 23 @ 7:30am)

CONTEXT:
- CBB V9.1 is complete, tournament-ready. All P0-P4 + K-10 wired. 606/609 pass.
- Fantasy Baseball is now active priority (draft in ~12 days)
- Yahoo OAuth fix is deployed. User needs to complete one-time local auth (see Setup tab).
- daily_lineup_optimizer.py is BUILT (sportsbook odds + implied runs + batter/pitcher rank)
- Full roadmap: docs/MLB_FANTASY_ROADMAP.md
- Yahoo API Research COMPLETE: reports/YAHOO_FANTASY_API_RESEARCH.md

PART 1 — Draft Board (dashboard/pages/11_Fantasy_Baseball.py, tab_draft) — MOSTLY COMPLETE:
✅ Filterable player board (551 Steamer players)
✅ Position scarcity snapshot
✅ Tier/strategy framework
⚠️ Remaining: Injury flags column, closer NSV overrides display

PART 2 — Live Draft Tracker (NEW — priority for March 22):
File: dashboard/pages/12_Live_Draft.py (already scaffolded)

REQUIREMENTS:
1. Yahoo API Integration:
   - Use yahoo_fantasy_api.League.draft_results() endpoint
   - Poll every 5 seconds during active draft
   - Detect new picks by comparing pick count between polls

2. Discord Integration (#fantasy-draft channel):
   - Send message on each new pick: "Pick #23: [Team] selects [Player] ([Position])"
   - Send "🚨 YOU'RE ON THE CLOCK!" alert when it's our pick
   - Include top 3 recommendations with each on-the-clock alert

3. Draft State Tracking:
   - Track: Current pick number, our next pick, picks until our turn
   - Display: Draft board grid (1-23 rounds × teams)
   - Mark drafted players as taken (strikethrough/highlight)

4. Recommendations Engine:
   - Query projections_loader.load_full_board()
   - Filter: Remove already-drafted players
   - Sort: By z-score (highest first)
   - Consider: Position scarcity (flag if C/SS/2B getting thin)
   - Display: Top 5 recommendations with reason ("Best available", "Position need", etc.)

5. On-The-Clock Detection:
   - Calculate from: draft_results length + league settings (teams, rounds)
   - Know our draft slot (configurable: 1-12)
   - Alert: 2 picks before our turn (gives time to decide)

IMPLEMENTATION NOTES:
- Yahoo API has NO websockets — must poll (see Section 5.4 for research)
- Rate limit: Max 1 request per 5 seconds
- Draft results return: pick, round, team_key, player_id
- Need to map player_id to our player board via name matching
- Graceful fallback: If API fails, allow manual pick entry

DISCORD CHANNEL: DISCORD_CHANNEL_FANTASY_DRAFT=1481294129450319893

TESTING:
- Test polling loop with mock data before March 22
- Verify Discord messages route correctly
- Test on-the-clock detection logic

FILES TO CREATE/MODIFY:
- dashboard/pages/12_Live_Draft.py — Main draft tracker UI
- backend/fantasy_baseball/draft_tracker.py — Polling logic (new)
- backend/services/discord_notifier.py — Add send_draft_pick() function
- tests/test_draft_tracker.py — Unit tests (new)

GUARDIAN NOTES:
- Do NOT touch betting_model.py, analysis.py, or CBB services during tournament window
- Fantasy code is isolated — safe to iterate without affecting CBB
- Run `python -m pytest tests/ -q` before any commit
- Yahoo OAuth must be working before March 23 (test with yahoo_client.py)
```

---

### CLAUDE CODE — OPCL-001: OpenClaw Discord Enhancement (Phase 1)
```
MISSION: Implement Phase 1 OpenClaw Discord enhancements by March 18

CONTEXT:
- Discord 16-channel architecture is LIVE and tested
- OpenClaw currently underutilized: ~0.1 msgs/day vs potential 15-20 msgs/day
- Full plan: reports/OPENCLAW_DISCORD_ENHANCEMENT_PLAN.md
- This is PRIORITY 1 after EMAC-064 (bet settlement fix)

PHASE 1 DELIVERABLES (Must complete by March 18):

1. Morning Brief Module (backend/services/openclaw_briefs.py)
   - Runs daily at 7:00 AM ET via cron
   - Queries overnight data:
     * Today's slate (game count, BET/CONSIDER/PASS distribution)
     * Integrity cache (all CONFIRMED? any CAUTION/VOLATILE?)
     * Sharp money moves (overnight line changes)
     * Escalation queue status (pending count)
   - Compiles Discord embed with emoji-rich formatting
   - Posts to #openclaw-briefs (channel ID: 1481294197045858395)
   - Include: Tournament countdown (days to First Four, etc.)

2. Telemetry Dashboard (backend/services/openclaw_telemetry.py)
   - Hourly system health summary
   - Metrics to display:
     * Integrity checks (24h count, latency, verdict distribution)
     * Sharp money signals (detected, high confidence, edge adjustments)
     * Predictions (today's count by tier)
     * System health (data sources, odds monitor, Discord, DB)
     * Escalation queue (pending, resolved today)
   - Posts to #openclaw-health (channel ID: 1481294433063276654)
   - Color-code: Green (healthy) / Yellow (warnings) / Red (critical)

3. Discord Integration Updates (backend/services/discord_notifier.py)
   - Add send_morning_brief(embed) function
   - Add send_telemetry_update(embed) function
   - Use existing send_to_channel() routing
   - Include fallback to file logging if Discord fails

4. Scheduler Integration (backend/scheduler.py or cron)
   - 7:00 AM ET: Trigger morning brief generation
   - Every hour: Trigger telemetry update
   - Add to existing heartbeat system

TESTING CHECKLIST:
- [ ] Morning brief generates in <30 seconds
- [ ] All sections populated with real data
- [ ] Discord message posts successfully
- [ ] Telemetry shows accurate metrics
- [ ] Cron triggers fire on schedule
- [ ] Graceful handling when data missing

ACCEPTANCE CRITERIA:
- [ ] Daily brief posted by 7:05 AM ET every day
- [ ] Hourly telemetry visible in #openclaw-health
- [ ] 0 errors in logs for 48-hour test period
- [ ] Content is actually useful (operator reviews)

PRIORITY ORDER:
1. Telemetry dashboard (quickest win — system visibility)
2. Morning brief (highest value — daily intelligence)
3. Scheduler integration (automation)

FILES TO CREATE:
- backend/services/openclaw_briefs.py (NEW ~150 lines)
- backend/services/openclaw_telemetry.py (NEW ~100 lines)

FILES TO MODIFY:
- backend/services/discord_notifier.py (add 2 functions)
- backend/scheduler.py (add cron triggers)
- tests/ (add test_openclaw_briefs.py, test_openclaw_telemetry.py)

GUARDIAN NOTES:
- These are NEW modules — safe to create, won't affect CBB
- Use existing OpenClawLite.telemetry for metrics
- Copy patterns from existing discord_notifier.py functions
- Test with --dry-run flag before enabling live posts
- Document any new environment variables needed
```

---

### GEMINI CLI (Research Only) — New Role: Intelligence Gathering

**IMPORTANT: Gemini is now RESEARCH ONLY. No production code. Deliver findings as markdown reports in docs/.**

#### Mission G-R1: Steamer 2026 Projection Data
```
RESEARCH TASK G-R1: Find best method to download full 2026 Steamer projections

CONTEXT:
- We use Steamer projections for MLB fantasy baseball
- Current data: hardcoded stubs (~80 players)
- Need: Full 2026 Steamer (~750 batters, ~450 pitchers)
- Existing format: data/projections/steamer_batting_2026.csv (columns: Name,Team,G,PA,AB,H,2B,3B,HR,R,RBI,BB,SO,HBP,SF,AVG,OBP,SLG,OPS,wOBA,wRC+,BsR,Off,Def,WAR)
- Existing format: data/projections/steamer_pitching_2026.csv (columns: Name,Team,W,L,ERA,G,GS,IP,H,ER,HR,BB,SO,WHIP,K/9,BB/9,K/BB,H/9,HR/9,AVG,BABIP,LOB%,GB%,HR/FB,FIP,xFIP,WAR)

RESEARCH QUESTIONS:
1. What is the direct download URL for Steamer 2026 batting projections from FanGraphs?
   (Look for CSV export buttons on fangraphs.com/projections.aspx)
2. Is there a public API or bulk download that doesn't require FanGraphs account?
3. What alternative projection systems are freely available (ZiPS, PECOTA, ATC)?
4. Is FantasyPros projections API free? What's the endpoint?
5. Can we scrape Baseball Reference's projections page?

DELIVERABLE: docs/PROJECTION_DATA_SOURCES.md
Format:
- Source name
- URL or API endpoint
- Auth required? (Y/N)
- Data fields available
- Download method (curl command or Python snippet)
- Recommended for our use case (Y/N + why)
```

#### Mission G-R2: MLB Starting Lineup Confirmations
```
RESEARCH TASK G-R2: Daily starting lineup confirmation sources for lineup optimizer

CONTEXT:
- We need to know each player's batting order position + if they're actually starting TODAY
- Our daily_lineup_optimizer.py ranks players by implied runs, but needs confirmation
  that a player is in the actual starting lineup (not resting or day-off)
- We want this data by 7:00 AM ET daily (before setting lineups)

RESEARCH QUESTIONS:
1. Does MLB Stats API (statsapi.mlb.com) provide lineups before games?
   - Check endpoint: https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=YYYY-MM-DD&hydrate=lineups
2. Does ESPN provide lineup data via a public API?
3. What time are lineups typically posted to these sources (1-2h before game? Morning?)
4. Is there a free RotoWire or RotoBaller lineup API?
5. Any baseball-reference.com endpoint for confirmed lineups?
6. Check: https://www.mlb.com/starting-lineups (what format is this page?)

DELIVERABLE: docs/LINEUP_CONFIRMATION_SOURCES.md
Format:
- Source
- Endpoint/URL
- Typical post time (ET)
- Auth required?
- Python snippet to fetch today's lineups
- Fields returned (batter name, batting order, team, position)
```

#### Mission G-R3: Closer Situations Monitor
```
RESEARCH TASK G-R3: Best data sources for tracking MLB closer situations

CONTEXT:
- NSV (Net Saves = SV - BS) is a category in our league
- We need to track: who is the primary closer for each team? Recent demotions?
- This is critical for waiver wire decisions

RESEARCH QUESTIONS:
1. Is there a free Baseball Reference or FanGraphs API for saves/blown saves by team?
2. CloserMonitor.com — is there a parseable format or API?
3. Rotowire closer chart — is it scrapeable? URL?
4. Does The Athletic or Baseball Prospectus have a free-tier API?
5. Twitter/X — what are the best accounts to follow for real-time closer news?
   (e.g., @RotoWire_MLB, @FantasyPros, beat reporters)
6. Check MLB transaction log: https://www.mlb.com/transactions — can we scrape this?

DELIVERABLE: docs/CLOSER_SITUATION_SOURCES.md
Same format as G-R1.
```

#### Mission G-R4: Statcast Data Access
```
RESEARCH TASK G-R4: Statcast bulk data download for advanced metrics

CONTEXT:
- We want xBA, xwOBA, exit velocity, barrel%, sprint speed for player rankings
- File: backend/fantasy_baseball/statcast_scraper.py already scaffolded
- Need to populate with real API calls

RESEARCH QUESTIONS:
1. Baseball Savant CSV search URL — what query params produce full season player data?
   (Check: https://baseballsavant.mlb.com/statcast_search/csv?...)
2. pybaseball library — what functions give player-level statcast data?
   (pip install pybaseball — check pybaseball.statcast_batter(start_dt, end_dt, player_id))
3. What's the standard approach for bulk download (100+ players)?
4. How to map player name to MLBAM ID (needed for most statcast APIs)?
5. Is there a free endpoint for sprint speed leaderboard?

DELIVERABLE: docs/STATCAST_API_GUIDE.md
Include working Python snippet using pybaseball or direct CSV URL.
```

#### Mission G-R5: Yahoo Fantasy API — Lineup & Transaction Details
```
RESEARCH TASK G-R5: Yahoo Fantasy API technical details for lineup management

CONTEXT:
- We have a working Yahoo OAuth client (backend/fantasy_baseball/yahoo_client.py)
- We added set_lineup() using PUT XML — need to verify the exact XML format
- We added add_drop_player() using POST XML — need to verify format

RESEARCH QUESTIONS:
1. What is the exact XML format for Yahoo Fantasy PUT /team/{team_key}/roster?
   - Does it need <coverage_type>date</coverage_type>?
   - What are the valid position codes for MLB? (C, 1B, 2B, 3B, SS, OF, Util, SP, RP, P, BN, DL)
2. What is the exact XML format for Yahoo Fantasy POST /league/{key}/transactions for add/drop?
3. Is there official Yahoo Fantasy API documentation URL with examples?
4. Any known limitations: rate limits, max roster changes per day?
5. Does the Yahoo API support batch lineup changes (multiple players in one call)?

DELIVERABLE: docs/YAHOO_API_REFERENCE.md
Include verified XML examples and any gotchas/limitations.
```

---

### OPENCLAW (Runtime Intelligence) — Mission O-9: Tournament Monitoring Sweep
```
MISSION O-9: Pre-tournament integrity sweep — March 18 First Four

CONTEXT:
- March 18, 2026: NCAA First Four begins (4 play-in games)
- March 20-21: Round of 64 begins (32 games per day)
- OpenClaw Lite handles integrity checks (no Ollama needed — uses heuristics)
- backend/services/openclaw_lite.py is active and wired into the analysis sweep
- Discord alerts are operational (channel_id=1477436117426110615)

YOUR TASKS (run on March 17, 2026 evening ~7 PM ET):

1. Verify the pre-tournament baseline exists:
   ls data/pre_tournament_baseline_2026.json
   If missing, run: python scripts/openclaw_baseline.py --year 2026

2. Run a manual integrity sweep on the First Four matchups:
   - Search for "NCAA First Four 2026 injury lineup" for each game
   - For each game: call check_integrity_heuristic() with the search text
   - If any ABORT or VOLATILE: flag immediately in HANDOFF.md

3. Verify the Discord bot: python scripts/test_discord.py
   Expected: "Discord bot healthy" message

4. Check odds monitor: GET /admin/odds-monitor/status
   Expected: games_tracked > 0, last_poll within 10 minutes

REPORT TO: HANDOFF.md section 3 with "### O-9 Tournament Readiness"
```

---

## 11. QUICK REFERENCE

### Test Commands
```bash
# Run all tests
python -m pytest tests/ -v

# Test specific modules
python -m pytest tests/test_fatigue.py -v
python -m pytest tests/test_openclaw_lite.py -v

# Check system status
python scripts/preflight_check.py
python scripts/test_discord.py
```

### Deploy Commands
```bash
# Railway
railway logs
railway status

# Git
git log --oneline -10
git status
```

### Monitor Commands
```bash
# Discord notifications
tail -f .openclaw/notifications/$(date +%Y-%m-%d).log

# Railway logs
railway logs --follow
```

---

---

## 12. EMAC-063 PROGRESS (March 11, 2026)

### 12.1 Draft Board Tab — COMPLETE

**File:** `dashboard/pages/11_Fantasy_Baseball.py` — `tab_draft` block

**What was done:**
- Replaced static placeholder with real, filterable player board (551 Steamer players)
- Filters: Position, Type (Batter/Pitcher), Tier (1-10), Top-N, Name search
- Displays Rank, Tier (color-coded), Name, Team, Position, ADP, Z-Score, Key Stats
- Position Scarcity Snapshot: shows T1-T2 counts for C, SS, 2B, SP, RP
- Strategy framework collapsed into expander (less clutter)
- Info banner directs users to `12_Live_Draft.py` on draft day
- Graceful fallback if board fails to load

**Data pipeline confirmed working:**
- `projections_loader.load_full_board()` reads Steamer CSVs (461 batting, 251 pitching, 306 ADP rows)
- `player_board.get_board()` prefers CSVs over hardcoded fallback
- 551 players with real z-scores returned

**Validation:**
- `python -c "import ast; ast.parse(open(..., encoding='utf-8').read())"` -- Syntax OK
- Board logic unit tested: all filters, scarcity snapshot, key stats columns confirmed correct

### 12.2 Projections Loader — Supplemental Files Wired

**File:** `backend/fantasy_baseball/projections_loader.py`

Three supplemental CSVs from Kimi Mission (previously unused) are now wired into `load_full_board()`:

| File | Data | Effect |
|------|------|--------|
| `injury_flags_2026.csv` | 24 players | Adds `injury_risk` field (extreme/high/low) + `injury_note` to player dicts |
| `closer_situations_2026.csv` | 30 teams | Overrides NSV projections for confirmed closers (e.g. Edwin Diaz 31 NSV) |
| `position_eligibility_2026.csv` | 37 players | Overrides positions for multi-eligible players (Mookie Betts gets SS, Ohtani gets SP) |

**New functions added (graceful — silent if file missing):**
- `load_injury_flags()` + `_apply_injury_flags()`
- `load_closer_situations()` + `_apply_closer_situations()`
- `load_position_eligibility()` + `_apply_position_eligibility()`

**Runtime result:** 14 injury-flagged, 19 closer NSV overrides, multi-position eligibility for 37 stars.

**Draft impact:** `12_Live_Draft.py` already reads `p.get("injury_risk")` for risk badges. Injury column now shows in `11_Fantasy_Baseball.py` draft board as "Risk" column.

### 12.3 System Validation (March 11)

- Tests: **601/604 pass** (excl. flaky async file — 3 pre-existing Postgres-auth failures, unchanged)
- `test_integrity_sweep.py::TestDdgsAndCheckAsync` passes 6/6 in isolation; Windows event-loop ordering issue in full-suite run (pre-existing, not our code)
- Remote: Repo up to date with origin/main
- CBB V9.1: All systems ready for March 18 tournament window

**GUARDIAN:** Do NOT touch CBB code (betting_model.py, analysis.py, CBB services) during tournament window (Mar 18 - Apr 7).

---

**Document Version:** EMAC-063
**Last Updated:** March 11, 2026
**Status:** Draft Board tab LIVE + supplemental data wired. 601/604 pass (excl. async flake). Tournament window opens Mar 18. Draft day Mar 23.

---

## 13. BETTING HISTORY AUDIT REVIEW (March 11, 2026)

**Auditor:** Kimi CLI (Deep Intelligence Unit)  
**Scope:** Verification of `reports/BETTING_HISTORY_AUDIT_MARCH_2026.md`  
**Status:** ⚠️ **CRITICAL ISSUE CONFIRMED — Action Required**

---

### 13.1 Executive Summary

A thorough review of the Gemini CLI betting history audit reveals **legitimate critical findings** that require immediate attention. The reported "Phantom Away Team" bug in `calculate_bet_outcome()` is **technically accurate and reproducible**.

**However:** The audit's severity assessment is accurate, but the recommended fix approach requires refinement to align with existing codebase patterns.

---

### 13.2 Technical Verification

#### Bug Confirmed: Name Mismatch in Bet Settlement

**Location:** `backend/services/bet_tracker.py:80`

**Current (Buggy) Code:**
```python
team_is_home = team.lower() == game.home_team.lower()

if team_is_home:
    margin = home_score - away_score
else:
    margin = away_score - home_score  # BUG: Defaults to away perspective on mismatch
```

**Verification Script:**
```
Pick: 'Samford Bulldogs' vs Game.home_team: 'Samford'
team_is_home: False (should be True)

Score: Samford 82, Furman 75 (Samford wins by 7)
Spread: -1.5

BUG: margin = 75 - 82 = -7 (away perspective)
     cover_margin = -7 + (-1.5) = -8.5 → LOSS

CORRECT: margin = 82 - 75 = 7 (home perspective)
         cover_margin = 7 + (-1.5) = 5.5 → WIN
```

**Result:** The bug inverts outcomes when pick names include mascots but Game records use short names.

---

### 13.3 Audit Claims Verification

| Claim | Status | Notes |
|-------|--------|-------|
| **Bug exists in `calculate_bet_outcome`** | ✅ **CONFIRMED** | Reproducible with name mismatch scenario |
| **Bet #112 (EWU -3.5) misgraded** | ⚠️ **PLAUSIBLE** | Cannot verify scores without DB access |
| **Bet #2 (UNCG -7.5) misgraded** | ⚠️ **PLAUSIBLE** | Cannot verify scores without DB access |
| **Bet #41 (Samford -1.5) misgraded** | ⚠️ **PLAUSIBLE** | Pattern matches bug behavior |
| **team_mapping.py not utilized** | ✅ **CONFIRMED** | `bet_tracker.py` imports no normalization |
| **"300+ normalization rules"** | ⚠️ **MISLEADING** | Actual count: ~100 ODDS_TO_KENPOM entries |

---

### 13.4 Impact Assessment

**Severity: HIGH** — The audit correctly identifies this as a critical issue.

**Affected Scenarios:**
1. Picks with full names ("Samford Bulldogs") vs Game short names ("Samford")
2. Picks with abbreviated names ("UNC Greensboro") vs Game full names ("North Carolina Greensboro")
3. Any name variation not exactly matching Game record

**NOT Affected:**
1. Picks that exactly match Game.home_team or Game.away_team
2. Moneyline bets (outright win detection still works regardless of home/away perspective)

---

### 13.5 Root Cause Analysis

**Primary Cause:** `parse_pick()` extracts team names from bet picks, but `calculate_bet_outcome()` compares them directly to `Game.home_team` without normalization.

**Contributing Factors:**
1. The Odds API uses full names with mascots ("Samford Bulldogs")
2. Internal Game records may use short names ("Samford")
3. No fuzzy matching or mapping lookup in bet settlement path
4. Silent failure — defaults to away team perspective on mismatch

---

### 13.6 Recommended Fixes (Prioritized)

#### Option A: Quick Fix — Use Normalized Comparison (Recommended for Immediate)

**File:** `backend/services/bet_tracker.py`

```python
from backend.services.team_mapping import normalize_team_name

def calculate_bet_outcome(bet, game, starting_bankroll=1000.0):
    # ... existing code ...
    team, spread = parse_pick(bet.pick)
    
    # Normalize both sides before comparison
    team_normalized = normalize_team_name(team) or team
    home_normalized = normalize_team_name(game.home_team) or game.home_team
    away_normalized = normalize_team_name(game.away_team) or game.away_team
    
    # Determine perspective with normalized names
    if team_normalized.lower() == home_normalized.lower():
        team_is_home = True
    elif team_normalized.lower() == away_normalized.lower():
        team_is_home = False
    else:
        # Log warning for unmatched team
        logger.warning(f"Could not match pick '{team}' to game {game.id}")
        return None  # Don't settle — requires manual review
    
    # ... rest of function unchanged ...
```

**Pros:**
- Uses existing `team_mapping.py` infrastructure
- Minimal code change (~10 lines)
- Fails safely (returns None instead of wrong result)

**Cons:**
- Adds dependency on `team_mapping` module
- May not catch all edge cases

#### Option B: Robust Fix — Store team_id on BetLog

**Schema Change Required:**
```python
# backend/models.py
class BetLog(Base):
    # ... existing fields ...
    team_id = Column(String)  # KenPom canonical name
    is_home_team = Column(Boolean)  # Determined at bet creation time
```

**Modify bet creation in `analysis.py`:**
```python
# When creating BetLog
bet = BetLog(
    # ... existing fields ...
    team_id=normalize_team_name(pick_team),
    is_home_team=(pick_team == game.home_team)
)
```

**Pros:**
- Eliminates runtime name matching entirely
- Deterministic at bet creation time
- No normalization needed at settlement

**Cons:**
- Requires database migration
- Must backfill historical data
- More invasive change

#### Option C: Hybrid — Parse and Match with Fuzzy Fallback

```python
from rapidfuzz import fuzz

def resolve_team_identity(pick_team, home_team, away_team):
    """Return ('home'|'away'|None) with fuzzy fallback."""
    pick_lower = pick_team.lower()
    home_lower = home_team.lower()
    away_lower = away_team.lower()
    
    # Exact match
    if pick_lower == home_lower:
        return 'home'
    if pick_lower == away_lower:
        return 'away'
    
    # Fuzzy match (token sort ratio handles word order)
    home_score = fuzz.token_sort_ratio(pick_lower, home_lower)
    away_score = fuzz.token_sort_ratio(pick_lower, away_lower)
    
    if home_score > 80 and home_score > away_score:
        return 'home'
    if away_score > 80 and away_score > home_score:
        return 'away'
    
    return None  # Ambiguous
```

**Pros:**
- Handles variations without mapping table
- Uses existing `rapidfuzz` dependency
- Graceful fallback

**Cons:**
- Adds complexity
- Potential false matches at threshold boundary

---

### 13.7 Historical Data Re-Settlement Plan

**Step 1: Identify Potentially Affected Bets**
```sql
-- Find settled bets where pick team doesn't match game home/away exactly
SELECT b.id, b.pick, g.home_team, g.away_team, b.outcome
FROM bet_logs b
JOIN games g ON b.game_id = g.id
WHERE b.outcome IS NOT NULL
  AND LOWER(SPLIT_PART(b.pick, ' ', 1)) != LOWER(g.home_team)
  AND LOWER(SPLIT_PART(b.pick, ' ', 1)) != LOWER(g.away_team);
```

**Step 2: Re-Settlement Script**
```python
# scripts/resettle_bets.py
from backend.services.bet_tracker import calculate_bet_outcome
from backend.models import SessionLocal, BetLog, Game

def resettle_bets():
    db = SessionLocal()
    affected = []
    
    for bet in db.query(BetLog).filter(BetLog.outcome.isnot(None)):
        game = db.query(Game).get(bet.game_id)
        if not game or not game.completed:
            continue
            
        new_result = calculate_bet_outcome(bet, game)
        if new_result and new_result.outcome != bet.outcome:
            affected.append({
                'bet_id': bet.id,
                'pick': bet.pick,
                'old_outcome': bet.outcome,
                'new_outcome': new_result.outcome,
                'game': f"{game.away_team} @ {game.home_team}"
            })
            # Apply correction
            bet.outcome = new_result.outcome
            bet.profit_loss_dollars = new_result.profit_loss_dollars
            bet.profit_loss_units = new_result.profit_loss_units
    
    db.commit()
    return affected
```

**Step 3: Validation**
- Run on staging first
- Cross-reference 10-20 random corrected bets with actual scores
- Generate before/after P&L report

---

### 13.8 Immediate Action Items

| Priority | Action | Owner | ETA |
|----------|--------|-------|-----|
| **P0** | Implement Option A fix in `bet_tracker.py` | Claude Code | March 12 |
| **P0** | Write `scripts/resettle_bets.py` | Claude Code | March 12 |
| **P1** | Test fix with 5 known historical cases | Kimi CLI | March 12 |
| **P1** | Run re-settlement on staging DB | Gemini CLI | March 13 |
| **P1** | Validate corrected P&L against manual calculation | Kimi CLI | March 13 |
| **P2** | Consider Option B schema change for next season | Claude Code | Post-tournament |

---

### 13.9 Audit Assessment Summary

| Aspect | Rating | Commentary |
|--------|--------|------------|
| **Technical Accuracy** | ⭐⭐⭐⭐⭐ | Bug description 100% correct, reproducible |
| **Impact Assessment** | ⭐⭐⭐⭐⭐ | Correctly identified as critical |
| **Severity** | ⭐⭐⭐⭐⭐ | High impact on P&L accuracy |
| **Recommended Fix** | ⭐⭐⭐☆☆ | Too vague; specific approach not provided |
| **Documentation** | ⭐⭐⭐⭐☆ | Good structure, but missing verification data |
| **Overall** | ⭐⭐⭐⭐☆ | Legitimate findings requiring action |

---

### 13.10 Risk of NOT Acting

**Tournament Window (March 18 - April 7):**
- Every settled bet during March Madness has potential for misgrading
- With 67 games in the tournament, even 10% error rate = 6-7 incorrect outcomes
- P&L metrics become unreliable for model recalibration
- User trust degradation if discrepancies discovered

**Recommendation:** Fix P0 items BEFORE March 18 First Four.

---

**Review Completed By:** Kimi CLI  
**Review Date:** March 11, 2026  
**Next Review:** After fix implementation (March 12)


---

## 14. NEXT PRIORITY: Discord Channel Redesign

**Status:** Design Complete → Ready for Channel Creation  
**Owner:** User (Discord setup) → Claude Code (code implementation)  
**Timeline:** Before March 18 (tournament start)

---

### 14.1 Overview

The current single `bets` channel is insufficient for the expanded platform (CBB + Fantasy Baseball). A redesigned channel architecture will:

- Separate CBB betting from Fantasy Baseball operations
- Route high-priority alerts to dedicated channels
- Provide clearer signal-to-noise ratio
- Support tournament-specific communications

---

### 14.2 Proposed Structure

**5 Categories, 16 Channels:**

```
🏀 CBB EDGE                    ⚾ FANTASY BASEBALL
├── cbb-bets                   ├── fantasy-lineups
├── cbb-morning-brief          ├── fantasy-waivers
├── cbb-alerts                 ├── fantasy-news
└── cbb-tournament             └── fantasy-draft

🎯 OPENCLAW INTEL              ⚙️ SYSTEM OPS
├── openclaw-briefs            ├── system-errors
├── openclaw-escalations       ├── system-logs
└── openclaw-health            └── data-alerts

💬 GENERAL
├── general-chat
└── admin-commands
```

**Full design:** `docs/DISCORD_CHANNEL_DESIGN.md`  
**Setup guide:** `docs/DISCORD_SETUP_QUICKSTART.md`

---

### 14.3 Key Channels Explained

| Channel | Purpose | Why It Matters |
|---------|---------|----------------|
| **cbb-bets** | Official recommendations | Clean feed of actionable bets |
| **cbb-morning-brief** | Daily 9 AM summary | Start-of-day context |
| **openclaw-escalations** | High-stakes alerts (≥1.5u) | Manual review queue |
| **fantasy-lineups** | Daily optimal lineups | 7 AM ET before games |
| **system-errors** | Critical failures | Immediate attention needed |

---

### 14.4 Implementation Steps

**Step 1: User Action (Discord Setup)**
- [ ] Create 5 categories in Discord
- [ ] Create 16 channels with descriptions
- [ ] Configure permissions (especially admin-commands)
- [ ] Copy channel IDs (enable Developer Mode)

**Step 2: Claude Code Action (Code Update)**
- [ ] Update `discord_notifier.py` with channel routing
- [ ] Add environment variables for all channel IDs
- [ ] Update `analysis.py` to route bets to cbb-bets
- [ ] Update `sentinel.py` for health channel
- [ ] Update `openclaw_lite.py` for escalations
- [ ] Update fantasy modules for lineup/waiver channels

**Step 3: Testing**
- [ ] Send test message to each channel
- [ ] Verify @admin mentions work
- [ ] Confirm fallback behavior

**Step 4: Go-Live**
- [ ] Announce in general-chat
- [ ] Monitor for 48 hours

---

### 14.5 Environment Variables Needed

Add to Railway + `.env`:

```bash
# Existing
DISCORD_CHANNEL_CBB_BETS=1477436117426110615

# New (fill in after channel creation)
DISCORD_CHANNEL_CBB_BRIEF=
DISCORD_CHANNEL_CBB_ALERTS=
DISCORD_CHANNEL_CBB_TOURNAMENT=

DISCORD_CHANNEL_FANTASY_LINEUPS=
DISCORD_CHANNEL_FANTASY_WAIVERS=
DISCORD_CHANNEL_FANTASY_NEWS=
DISCORD_CHANNEL_FANTASY_DRAFT=

DISCORD_CHANNEL_OPENCLAW_BRIEFS=
DISCORD_CHANNEL_OPENCLAW_ESCALATIONS=
DISCORD_CHANNEL_OPENCLAW_HEALTH=

DISCORD_CHANNEL_SYSTEM_ERRORS=
DISCORD_CHANNEL_SYSTEM_LOGS=
DISCORD_CHANNEL_DATA_ALERTS=
```

---

### 14.6 Message Format Examples

**cbb-bets:**
```
🏀 BET RECOMMENDATION

**Gonzaga Bulldogs -3.5** @ -110
Bet Size: 1.5 units ($37.50)
Confidence: 72% | Edge: 4.2%
Game: Gonzaga @ Saint Mary's — 9:00 PM ET
```

**openclaw-escalations:**
```
🚨 HIGH-STAKES ESCALATION

Game: UNC @ Duke (Elite Eight)
Recommended: 2.0 units
Verdict: VOLATILE

Action Required: Manual review before tipoff
Queue ID: 20260318_190000_UNC_Duke
```

**fantasy-lineups:**
```
⚾ TODAY'S OPTIMAL LINEUP — March 18

**Hitters:**
C: Willson Contreras (STL) — 8.2 proj
1B: Matt Olson (ATL) — 7.8 proj
...

**Pitchers:**
SP: Spencer Strider (ATL) — 22.4 proj

Total Projected: 142.3 points
```

---

### 14.7 Timeline

| Date | Milestone |
|------|-----------|
| March 12 | Channels created, IDs provided |
| March 13 | Code updated, tested |
| March 14 | New structure live |
| March 18 | Tournament mode active (cbb-tournament channel) |
| March 20 | Draft mode active (fantasy-draft channel) |
| March 24 | Archive fantasy-draft channel |

---

### 14.8 Files for Reference

- **Full Design:** `docs/DISCORD_CHANNEL_DESIGN.md`
- **Setup Guide:** `docs/DISCORD_SETUP_QUICKSTART.md`
- **Migration Plan:** Section 14.4 above

---

**Ready to proceed?** Follow the quickstart guide and send me the channel IDs when ready!


---

### 14.9 UPDATE: Discord Channels Configured ✅

**Date:** March 11, 2026  
**Status:** Channel IDs received, code updated, ready for testing

---

#### Channels Created

All 16 channels have been configured with the following IDs:

```
🏀 CBB EDGE
  cbb-bets              1477436117426110615 (existing)
  cbb-morning-brief     1481293065405595701
  cbb-alerts            1481293221316395088
  cbb-tournament        1481293294263865455

⚾ FANTASY BASEBALL
  fantasy-lineups       1481293925506617396
  fantasy-waivers       1481294029273698376
  fantasy-news          1481294077755527209
  fantasy-draft         1481294129450319893

🎯 OPENCLAW INTEL
  openclaw-briefs       1481294197045858395
  openclaw-escalations  1481294383092338810
  openclaw-health       1481294433063276654

⚙️ SYSTEM OPS
  system-errors         1481294516567932980
  system-logs           1481294557936353521
  data-alerts           1481294607726940292

💬 GENERAL
  general-chat          1481294687607455764
  admin-commands        1481294886534647848
```

---

#### Code Updates Complete

**File:** `backend/services/discord_notifier.py` (v2.0)

**Changes:**
1. ✅ Added `CHANNEL_MAP` with all 16 channel environment variable mappings
2. ✅ Added `_get_channel_id()` function for channel name → ID resolution
3. ✅ Added `send_to_channel()` function for direct channel sending
4. ✅ Added `route_notification()` function for message type-based routing
5. ✅ Updated all legacy functions to use new channel routing:
   - `send_todays_bets()` → #cbb-bets
   - `send_health_briefing()` → #openclaw-health
   - `send_verdict_flip_alert()` → #cbb-alerts
   - `send_source_health_alert()` → #data-alerts
6. ✅ Added new functions:
   - `send_high_stakes_escalation()` → #openclaw-escalations
   - `send_fantasy_lineup()` → #fantasy-lineups
   - `send_system_error()` → #system-errors
   - `send_routine_log()` → #system-logs

**Environment Variables:**
- Created `.env.discord.channels` with all channel IDs
- All 16 `DISCORD_CHANNEL_*` variables ready for Railway deployment

---

#### Test Script Created

**File:** `scripts/test_discord_channels.py`

**Usage:**
```bash
# Check configuration (no messages sent)
python scripts/test_discord_channels.py --config-only

# Dry run (show what would be sent)
python scripts/test_discord_channels.py --dry-run

# Send actual test messages
python scripts/test_discord_channels.py
```

---

#### Deployment Checklist

**Before Testing:**
- [ ] Add all `DISCORD_CHANNEL_*` variables to Railway environment
- [ ] Verify `DISCORD_BOT_TOKEN` is set
- [ ] Run `python scripts/test_discord_channels.py --config-only`

**Testing:**
- [ ] Run `python scripts/test_discord_channels.py --dry-run`
- [ ] Run `python scripts/test_discord_channels.py` (send real messages)
- [ ] Verify all 16 channels receive test messages
- [ ] Check formatting in each channel

**Go-Live:**
- [ ] Update `.env` file with new variables
- [ ] Commit changes to git
- [ ] Deploy to Railway
- [ ] Monitor for 24 hours

---

#### Usage Examples

**Send bet recommendation:**
```python
from backend.services.discord_notifier import send_todays_bets

send_todays_bets(bet_details, summary)  # Goes to #cbb-bets
```

**Send high-stakes escalation:**
```python
from backend.services.discord_notifier import send_high_stakes_escalation

send_high_stakes_escalation(
    game_key="UNC@Duke",
    home_team="Duke",
    away_team="UNC",
    recommended_units=2.0,
    integrity_verdict="VOLATILE",
    reason="Late injury news + Elite Eight",
    queue_id="20260312_090000_UNC_Duke"
)  # Goes to #openclaw-escalations with @admin mention
```

**Send fantasy lineup:**
```python
from backend.services.discord_notifier import send_fantasy_lineup

send_fantasy_lineup(lineup_data)  # Goes to #fantasy-lineups
```

**Route by message type:**
```python
from backend.services.discord_notifier import route_notification

route_notification(
    message_type="system_health",
    embed=health_embed,
    severity="normal"
)  # Routes to #openclaw-health
```

---

#### Migration from Legacy

**Before (single channel):**
```python
from backend.services.discord_notifier import send_todays_bets
send_todays_bets(bets, summary)  # Always went to one channel
```

**After (multi-channel):**
```python
# Same function call — now routes to #cbb-bets automatically
from backend.services.discord_notifier import send_todays_bets
send_todays_bets(bets, summary)

# New: Direct channel targeting
from backend.services.discord_notifier import send_to_channel
send_to_channel("fantasy-lineups", embed=lineup_embed)
```

**Backward Compatibility:** ✅ All existing code continues to work

---

**Status:** ✅ **COMPLETED** — All 16 channels tested and working

**Completed:**
- ✅ All environment variables added to Railway
- ✅ Discord token refreshed and working
- ✅ Test script passed — messages sent to all channels
- ✅ Improved bet message format (clearer team/spread display)

---

## 15. TASK ASSIGNMENT: Bet Settlement Name-Matching Fix (EMAC-064)

**Assigned to:** Claude Code (Master Architect)  
**Priority:** CRITICAL — Must complete before March 18 (First Four)  
**Estimated Effort:** 4-6 hours  
**Dependencies:** None (isolated fix)

---

### 15.1 Problem Statement

The `calculate_bet_outcome()` function in `backend/services/bet_tracker.py` contains a critical bug that causes **spread bet misgrading** when team names don't exactly match between the `pick` string and `Game` record.

**Root Cause:**
```python
team_is_home = team.lower() == game.home_team.lower()
```

When `pick = "Samford Bulldogs -1.5"` but `game.home_team = "Samford"`, the comparison fails and the code defaults to away-team perspective, inverting the margin calculation.

**Impact:**
- Historical P&L data is unreliable
- Current bets may be settled incorrectly
- Model recalibration uses bad ground truth

**Severity:** HIGH — Every spread bet with a name mismatch is at risk.

---

### 15.2 Required Deliverables

#### Deliverable 1: Fixed Bet Settlement Logic

**File:** `backend/services/bet_tracker.py`

**Requirements:**
1. Import and use `backend.services.team_mapping.normalize_team_name()` for fuzzy matching
2. Before calculating margin, explicitly resolve which team was picked:
   - Normalize pick team name
   - Normalize both Game.home_team and Game.away_team
   - Match pick to one of them
   - If ambiguous/missing, log warning and return `None` (don't settle)
3. Add unit tests in `tests/test_bet_tracker.py`

**Implementation Pattern:**
```python
from backend.services.team_mapping import normalize_team_name

def resolve_team_identity(pick_team: str, home_team: str, away_team: str) -> Tuple[bool, str]:
    """
    Resolve which team was picked.
    
    Returns:
        (is_home: bool, matched_name: str)
        Raises ValueError if cannot resolve unambiguously
    """
    pick_norm = normalize_team_name(pick_team) or pick_team
    home_norm = normalize_team_name(home_team) or home_team
    away_norm = normalize_team_name(away_team) or away_team
    
    # Fuzzy matching with rapidfuzz
    from rapidfuzz import fuzz
    
    home_score = fuzz.token_sort_ratio(pick_norm.lower(), home_norm.lower())
    away_score = fuzz.token_sort_ratio(pick_norm.lower(), away_norm.lower())
    
    if home_score > 80 and home_score > away_score:
        return True, home_team
    elif away_score > 80 and away_score > home_score:
        return False, away_team
    else:
        raise ValueError(f"Cannot match '{pick_team}' to {home_team} or {away_team}")
```

#### Deliverable 2: Historical Re-Settlement Script

**File:** `scripts/resettle_historical_bets.py`

**Requirements:**
1. Query all `BetLog` entries with `outcome IS NOT NULL` and `is_paper_trade = True`
2. For each bet:
   - Re-run `calculate_bet_outcome()` with fixed logic
   - If outcome differs from stored value, record the correction
   - Update the BetLog with corrected outcome and P&L
3. Generate a correction report:
   - Count of corrected bets
   - Total P&L impact (before vs after)
   - List of affected games
4. Log-only mode (`--dry-run`) for safety

**Usage:**
```bash
# Dry run first
python scripts/resettle_historical_bets.py --dry-run

# Actually apply corrections
python scripts/resettle_historical_bets.py --apply
```

#### Deliverable 3: Validation Tests

**File:** `tests/test_bet_settlement_fix.py`

**Test Cases:**
1. **Exact match** — "Duke" vs "Duke" → correct perspective
2. **Mascot mismatch** — "Samford Bulldogs" vs "Samford" → correct perspective
3. **Away team pick** — "UNC +3.5" on game where UNC is away → correct perspective
4. **Ambiguous match** — "Miami" (could be Miami FL or Miami OH) → error/None
5. **Edge cases** — abbreviations, punctuation differences

**Integration Test:**
- Create BetLog + Game in test DB
- Run settlement
- Verify correct outcome

---

### 15.3 Technical Constraints

**Must Not Break:**
- Existing moneyline settlement logic
- Any downstream P&L calculations
- Database schema (no migrations needed)

**Must Use:**
- Existing `team_mapping.py` normalization
- Existing `rapidfuzz` dependency
- Same database session patterns

**Must Handle:**
- Cases where team_mapping has no entry (graceful fallback)
- Concurrent bet settlement (no race conditions)
- Games where pick team no longer exists in mapping (log warning)

---

### 15.4 Acceptance Criteria

- [ ] All new unit tests pass
- [ ] Historical re-settlement script runs without errors
- [ ] Dry-run shows expected corrections for known-misgraded bets
- [ ] Actual re-settlement produces correct P&L reconciliation
- [ ] No existing tests broken
- [ ] Code review approved by Kimi CLI

---

### 15.5 Context for Claude

**Why This Matters:**
The CBB Edge model is tournament-ready (V9.1, fatigue, sharp money, conference HCA all wired). However, the betting history audit (Section 13) revealed this settlement bug invalidates our ground truth. Before March Madness begins, we need:
1. Correct settlement going forward
2. Corrected historical data for model recalibration

**Files to Read First:**
- `backend/services/bet_tracker.py` — Current (buggy) implementation
- `backend/services/team_mapping.py` — Normalization functions
- `reports/BETTING_HISTORY_AUDIT_MARCH_2026.md` — Audit findings
- `reports/AUDIT_VERIFICATION_KIMI_MARCH_2026.md` — Technical verification

**Design Philosophy:**
- Fail safe: When in doubt, don't settle (return None)
- Explicit over implicit: Clear logging of name matches
- Backward compatible: Existing working code paths unchanged

---

### 15.6 Success Metrics

After completion:
- Zero misgraded spread bets due to name mismatch
- Historical P&L reconciled to within 1% of actual
- Settlement latency <10ms per bet (no regression)
- 100% test coverage for new resolution logic

---

**Kimi CLI Review Required:** Before merging, Kimi will verify:
1. Logic handles all edge cases in test suite
2. Re-settlement script produces expected corrections
3. No performance regression in settlement path

---

**Document Version:** EMAC-065
**Last Updated:** March 11, 2026  
**Next Review:** Upon Claude completion

**Summary of Today's Updates:**
- ✅ Yahoo Fantasy API research complete (Section 5.4, Section 16) — polling-based draft tracker ready for Claude implementation
- ✅ Advanced Analytics research reviewed (Section 5.5) — Gemini's Statcast metrics assessed for post-draft integration
- ✅ OpenClaw Discord Enhancement Plan (Section 6.5, OPCL-001) — 5-pillar strategy for 10x impact increase
- ⚠️ EMAC-064 (Bet Settlement Fix) — Critical, assigned to Claude, due March 18
- 📋 OPCL-001 Phase 1 (OpenClaw Morning Brief + Telemetry) — Ready for Claude, target March 18

---

## 16. YAHOO FANTASY API RESEARCH — IMPLEMENTATION READY

**Research Complete:** March 11, 2026  
**Full Report:** `reports/YAHOO_FANTASY_API_RESEARCH.md`  
**Assigned to:** Claude Code (Master Architect)  
**Deadline:** March 22, 2026 (1 day before draft)

### Summary for Claude Code

The Yahoo Fantasy API **does NOT support real-time websockets** — you must implement a **polling-based approach** for live draft tracking.

#### Core Implementation

```python
# Key endpoint: league.draft_results() — poll every 5 seconds
from yahoo_fantasy_api import League

league = League(oauth_session, league_id)

# During draft, poll this:
results = league.draft_results()  # Returns picks made so far
available = league.players(status='A')  # Returns undrafted players
```

#### Key Limitations
- ❌ No websockets/push notifications
- ❌ No "on the clock" notifications  
- ❌ No draft timer visibility
- ✅ Can detect new picks by comparing results between polls

#### Deliverables for March 22
1. `backend/fantasy_baseball/draft_tracker.py` — Polling loop with new pick detection
2. `dashboard/pages/12_Live_Draft.py` — Live draft UI with Discord integration
3. Discord notifications to `#fantasy-draft` channel (ID: 1481294129450319893)
4. "On the clock" detection from pick number + draft slot

#### Acceptance Criteria
- [ ] Polls draft_results() every 5 seconds without rate limiting errors
- [ ] Sends Discord notification within 10 seconds of new pick
- [ ] Detects our turn and sends "ON THE CLOCK" alert
- [ ] Displays top 5 player recommendations at each pick
- [ ] Graceful fallback to manual entry if API fails

**Reference:** Section 5.4 and Section 10 (CLAUDE CODE handoff prompt)

---

## 17. ADVANCED ANALYTICS RESEARCH — POST-SEASON INTEGRATION

**Research Complete:** March 11, 2026  
**Full Report:** `reports/ADVANCED_ANALYTICS_INTEGRATION.md`  
**Researcher:** Gemini CLI  
**Assigned to:** Future iteration (2026 season or 2027 draft)  

### Summary

Gemini researched cutting-edge 2026 baseball metrics. Research is **strategically valuable but not immediately actionable** due to data availability constraints.

### What's Ready to Use Now

| Metric | Use Case | Implementation |
|--------|----------|----------------|
| **Swing Length** | Identify contact hitters | pybaseball library |
| **Exit Velocity** | Power validation | Already in Statcast pipeline |
| **Barrel%** | Power consistency | Calculate from Statcast |

### What's Paywalled/Limited

| Metric | Source | Cost | Recommendation |
|--------|--------|------|----------------|
| **Stuff+** | FanGraphs | $$$ | Defer to 2027 |
| **PLV** | PitcherList | Free (scrape) | Post-season project |
| **Bat Speed** | Baseball Savant | Limited 2026 data | Monitor availability |

### Implementation Path

**2026 Season (Post-Draft):**
```python
# Add to daily_lineup_optimizer.py
from pybaseball import statcast_batter

# Pull swing metrics for waiver wire sleepers
# Flag: ADP > 150 + Swing Length < 7.2 ft = potential sleeper
```

**2027 Draft Prep:**
- Build Statcast pipeline in Nov-Dec 2026
- Create custom "Blast%" metric from bat speed + squared-up data
- Integrate into projections_loader.py

### Claude Code Note

This research is **strategically sound** but **not a priority** for March 23 draft. The metrics require:
1. 2026 season data (not available yet)
2. pybaseball integration (moderate effort)
3. Validation against actual results

**Defer until after CBB tournament (post-April 7).**

**Reference:** Section 5.5, `reports/ADVANCED_ANALYTICS_INTEGRATION.md`

---

## 18. OPENCLAW DISCORD ENHANCEMENT — IMPLEMENTATION ROADMAP

**Plan:** `reports/OPENCLAW_DISCORD_ENHANCEMENT_PLAN.md`  
**Assigned to:** Claude Code (implementation) → OpenClaw (execution)  
**Timeline:** Phase 1 (March 12-18), Phase 2 (March 19-25), Phase 3 (March 26-Apr 7)

### The 5 Pillars

```
┌─────────────────────────────────────────────────────────────┐
│                  OPENCLAW INTELLIGENCE HUB                 │
├─────────────────────────────────────────────────────────────┤
│  MORNING BRIEF    LIVE MONITOR     FANTASY INTEL           │
│  Daily 7 AM ET    Every 2h         Daily 7 AM + ad-hoc     │
│  → #briefs        → #escalations   → #fantasy-news         │
│                                                            │
│  TELEMETRY        TOURNAMENT OPS                             │
│  Hourly           Mar 18-Apr 7                             │
│  → #health        → #tournament                            │
└─────────────────────────────────────────────────────────────┘
```

### Phase 1: Foundation (March 12-18) — CURRENT PRIORITY

**Goal:** Morning Brief + Telemetry Dashboard active by First Four

| Component | Channel | Status |
|-----------|---------|--------|
| Morning Brief | `#openclaw-briefs` | 🔄 Ready for implementation |
| Telemetry | `#openclaw-health` | 🔄 Ready for implementation |

**Files to Create:**
- `backend/services/openclaw_briefs.py` (~150 lines)
- `backend/services/openclaw_telemetry.py` (~100 lines)

**Files to Modify:**
- `backend/services/discord_notifier.py` (add 2 functions)
- `backend/scheduler.py` (add cron triggers)

### Phase 2: Live Operations (March 19-25) — TOURNAMENT

**Goal:** Game-time monitoring for First Four and Round of 64

| Component | Channel | Trigger |
|-----------|---------|---------|
| Live Monitor | `#openclaw-escalations` | Every 2h on game days |
| Tournament Alerts | `#cbb-tournament` | Upset/cinderella events |

### Phase 3: Fantasy Integration (March 26-Apr 7) — MLB OPENING DAY

**Goal:** Fantasy baseball intelligence by Opening Day

| Component | Channel | Frequency |
|-----------|---------|-----------|
| Closer Monitor | `#fantasy-news` | Daily 7 AM ET |
| Lineup Confirmations | `#fantasy-news` | As lineups post |
| Waiver Sleepers | `#fantasy-waivers` | Weekly + hot pickups |

### Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Intelligence msgs/day | 0.1 | 15-20 | Discord count |
| Time to alert | Manual | <15 min | Timestamp diff |
| System visibility | Spotty | 99.9% | Telemetry coverage |

### Claude Code Priority

**Order of implementation:**
1. **EMAC-064** (Bet Settlement Fix) — CRITICAL, due March 18
2. **OPCL-001 Phase 1** (OpenClaw Morning Brief + Telemetry) — HIGH, due March 18
3. **EMAC-063** (Fantasy Draft Tracker) — MEDIUM, due March 22
4. **OPCL-001 Phase 2** (Live Monitor) — MEDIUM, March 19-25
5. **OPCL-001 Phase 3** (Fantasy Intel) — LOW, March 26-Apr 7

**Reference:** Section 6.5, Section 8.1a, and `reports/OPENCLAW_DISCORD_ENHANCEMENT_PLAN.md`

