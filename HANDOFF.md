# OPERATIONAL HANDOFF (EMAC-061)

> Ground truth as of March 10, 2026. Operator: Claude Code (Master Architect).
> See `IDENTITY.md` for risk policy · `AGENTS.md` for roles · `HEARTBEAT.md` for loops.
> Full enhancement plan: `tasks/cbb_enhancement_plan.md`

---

## 0. ARCHITECT DECISION (March 10, 2026)

**User concern:** Returns and model success have been limited lately.

**Root cause finding:** The model is running in **KenPom-only mode**. Both
`BARTTORVIK_USERNAME/PASSWORD` and `EVANMIYA_API_KEY` are unset in Railway.
The 3-source composite is degraded to 1 source, causing:
- Renormalized weight = 100% KenPom (accuracy loss)
- `margin_se` widened from 1.50 → 1.80 (over-cautious CI)
- Fewer BET verdicts, noisier edges when bets are placed

**Priority order (March Madness window):**
1. **P0 — Fix BartTorvik + EvanMiya data** (1 day) → immediate accuracy gain
2. **P1 — Sharp Money Detection** (2–3 days) → before Mar 18 tournament tip
3. **P2/P3 — Conference HCA + March recency weighting** (0.5 day each) → quick wins
4. **DEFER — Fantasy Baseball** until after Apr 7 championship

**See `tasks/cbb_enhancement_plan.md` for full diagnosis and sprint breakdown.**

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
| Conference HCA (P2) | ✅ MODULE CLEAN | `backend/services/conference_hca.py` | 18 pass ✅ (BLOCKED — needs team→conference map) |
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

### 8.2 Gemini CLI — G-16 Verification
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
| Template fallbacks essential when Ollama unavailable | D-1 |
| `key` parameter breaks older Streamlit versions | D-1 |
| Railway needs explicit `railway.toml` for reliable builds | Railway Fix |
| Discord token must be set in Railway Variables, not just `.env` | D-1 |
| cloudscraper 1.x series latest — 2.x doesn't exist (build fix) | Build Fix |

---

## 10. HANDOFF PROMPTS

### KIMI CLI (Deep Intelligence) — Mission K-10: Conference HCA Data Layer
```
MISSION K-10: Build team→conference mapping for Conference HCA wiring

CONTEXT:
- `backend/services/conference_hca.py` is complete and tested (18 tests pass)
- `get_conference_hca(conference, is_neutral=False)` returns pts advantage (Big Ten=3.6, Big 12=3.4, etc.)
- The analysis pipeline (`backend/services/analysis.py`) CANNOT call it because there is no
  team-to-conference data available. `TeamPlayStyle` has no `conference` field.
- We need a lightweight data lookup to go from team name → conference name.

YOUR TASK:
1. Build `data/team_conference_map.csv` with columns: `team_name,conference`
   - Must cover ALL D1 teams used in the system (~365 teams loaded in DB)
   - Team names should match what BartTorvik uses (e.g., "Duke", "UNC", "Gonzaga", "St. Mary's")
   - Conference names should match conference_hca.py keys (Big Ten, Big 12, SEC, ACC, Big East,
     WCC, AAC, A-10, MAC, MWC, SBC, C-USA, CUSA, Summit, OVC, MAAC, Patriot, CAA,
     ASUN, SoCon, MEAC, SWAC, Horizon, NEC, AE, MVFC, WAC, BWC)
   - Include at minimum the 40+ major conference teams from odds API

2. Build `backend/services/team_conference_lookup.py`:
   ```python
   # Simple CSV-backed lookup
   def get_team_conference(team_name: str) -> Optional[str]:
       """Return conference name for a team, or None if unknown."""

   def get_conference_hca_for_team(team_name: str, is_neutral: bool = False) -> float:
       """Return HCA points for a team's home conference, or DEFAULT_HCA=3.0 if unknown."""
   ```

3. Wire it into `backend/services/analysis.py` at the injection point:
   - After the fatigue block (around line 1310 in current code)
   - Before `analyze_game()` call
   - Calls:
     ```python
     from backend.services.team_conference_lookup import get_conference_hca_for_team
     # ...
     conf_hca = get_conference_hca_for_team(home_team, is_neutral=game_data.get("is_neutral", False))
     # Pass conf_hca to analyze_game() as matchup_margin_adj offset OR store in full_analysis
     ```
   - IMPORTANT: analyze_game() already receives `matchup_margin_adj`. Add conf HCA there IF
     conference HCA differs from baseline 3.09 (i.e., delta = conf_hca - 3.09, not full conf_hca)

4. Add 5 tests to `tests/test_conference_hca.py`:
   - test_team_lookup_duke_acc
   - test_team_lookup_gonzaga_wcc
   - test_team_lookup_unknown_returns_default
   - test_conference_hca_for_team_big_ten
   - test_neutral_site_returns_zero

FILES TO CREATE:
- `data/team_conference_map.csv`
- `backend/services/team_conference_lookup.py`

FILES TO MODIFY:
- `backend/services/analysis.py` (add import + 4-line block after fatigue section)
- `tests/test_conference_hca.py` (add 5 tests)

VERIFICATION:
```bash
python -m pytest tests/test_conference_hca.py -v
python -m pytest tests/ -q  # Full suite must still show 595+ pass, 3 fail
```

REPORT TO: HANDOFF.md section 3 with "Mission K-10 Complete" + test results
```

---

### OPENCLAW (Runtime Intelligence) — Mission O-9: Tournament Monitoring Sweep
```
MISSION O-9: Pre-tournament integrity sweep — March 18 First Four

CONTEXT:
- March 18, 2026: NCAA First Four begins (4 play-in games)
- March 20-21: Round of 64 begins (32 games per day)
- OpenClaw Lite handles integrity checks (no Ollama needed — uses heuristics)
- `backend/services/openclaw_lite.py` is active and wired into the analysis sweep
- Discord alerts are operational (channel_id=1477436117426110615)

YOUR TASKS (run on March 17, 2026 evening ~7 PM ET):

1. Verify the pre-tournament baseline exists:
   ```bash
   ls data/pre_tournament_baseline_2026.json
   cat reports/o8_baseline_summary_2026.md | head -40
   ```
   If missing, run: `python scripts/openclaw_baseline.py --year 2026`

2. Run a manual integrity sweep on the First Four matchups:
   - Search for "NCAA First Four 2026 injury lineup" for each game
   - For each game: call check_integrity_heuristic() with the search text
   - Expected: 4 results, each CONFIRMED or CAUTION
   - If any ABORT or VOLATILE: flag immediately in HANDOFF.md

3. Verify the Discord bot can send alerts:
   ```bash
   python scripts/test_discord.py
   ```
   Expected: "Discord bot healthy" message

4. Check the odds monitor is tracking First Four games:
   ```bash
   curl -H "X-API-Key: $API_KEY_USER1" https://<railway-url>/admin/odds-monitor/status
   ```
   Expected: `games_tracked > 0`, `last_poll` within last 10 minutes

5. Report back to HANDOFF.md:
   - Section heading: "### OpenClaw O-9 Tournament Readiness Check"
   - Include: baseline status, integrity sweep results, Discord status, odds monitor status
   - Flag any ABORT verdicts with the game name and reason

COMMAND TO START:
```bash
cd /path/to/cbb-edge
python -c "
from backend.services.openclaw_lite import OpenClawLite
checker = OpenClawLite()
# Test with First Four game: Texas Southern vs Alabama State (example)
result = checker.check_integrity_heuristic(
    search_text='No injury news. Teams healthy and ready.',
    home_team='Texas Southern',
    away_team='Alabama State',
    recommended_units=1.0
)
print(f'Verdict: {result.verdict}, Confidence: {result.confidence:.2f}')
print(f'Reasoning: {result.reasoning}')
"
```

REPORT TO: HANDOFF.md section 3
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

**Document Version:** EMAC-061
**Last Updated:** March 10, 2026
**Status:** Production Ready — Services Wired — Conference HCA Delegated to Kimi (K-10)
