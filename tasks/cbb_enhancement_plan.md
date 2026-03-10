# CBB Edge Analyzer — Enhancement Plan & Task Tracker

> Architect review: March 10, 2026 | Branch: `claude/review-and-plan-enhancements-5ilpW`
> Context: Model returns limited lately. March Madness window live Mar 18+.
> This plan diagnoses root causes and prescribes prioritized fixes.

---

## ROOT CAUSE DIAGNOSIS

### Why Returns Are Limited

After full codebase review, the limiting factors are, in order of impact:

#### 1. CRITICAL — Missing Rating Sources (KenPom-Only Mode)
`BARTTORVIK_USERNAME/PASSWORD` and `EVANMIYA_API_KEY` are **not set** in Railway.
The model is running on KenPom alone despite being designed for a 3-source composite.
Consequences:
- Weight renormalization forces 100% KenPom → less accurate margin estimates
- `EVANMIYA_DOWN_SE_ADDEND=0.30` widens margin_se from 1.50 → 1.80
- Wider CI → fewer BET verdicts (model over-passes)
- Less accurate margins → when it does bet, edges are noisier
- BartTorvik is publicly accessible (no auth needed for CSV) — just not being fetched

#### 2. HIGH — Infrastructure Work Crowded Out Model Work
Last 15+ commits: Discord, Railway, OpenClaw Lite, Fantasy Baseball.
Zero commits improving core predictive accuracy. The system is well-deployed
but predicting poorly.

#### 3. HIGH — No Sharp Money / Line Movement Signal
E-1 enhancement flagged in HANDOFF but never implemented.
Without reverse-line-movement detection, the model bets into already-corrected
market lines. Sharp books (Pinnacle, Circa) absorb information faster than our
nightly scrape cycle.

#### 4. MEDIUM — Flat Home Court Advantage Across All Conferences
Using 3.09 pts universally. Big Ten road games are significantly harder;
SWAC home games are worth less. Conference-specific HCA = quick, free improvement.

#### 5. MEDIUM — March Context Requires Different Defaults
Regular-season model uses HCA as a primary driver. March Madness is neutral-site.
The seed-spread scalars were added in V9.1 but neutral-site HCA override and
recency weighting (last 30 days more informative in March) are missing.

#### 6. MEDIUM — Recalibration Service Under-Fed
`recalibration.py` needs 30+ settled bets with linked predictions.
Unknown if threshold is met. Without it, home_advantage (3.09) and sd_multiplier
(0.85) are static — any drift goes uncorrected.

#### 7. LOW — Possession Simulator Accuracy Unvalidated
`possession_sim.py` (947 lines) is integrated but its accuracy vs. the
ratings-path hasn't been measured against actual outcomes / CLV. Could be
helping or adding noise.

---

## PRIORITY MATRIX

| Priority | Enhancement | Effort | Impact | Deadline |
|----------|-------------|--------|--------|----------|
| **P0** | Fix BartTorvik + EvanMiya data flow | 1 day | Critical | Immediately |
| **P1** | Sharp Money Detection | 2–3 days | High | Before Mar 18 |
| **P2** | Conference-specific HCA | 0.5 day | Medium | Before Mar 18 |
| **P3** | Late-season / March recency weighting | 0.5 day | Medium | Before Mar 18 |
| **P4** | Recalibration audit + seed | 0.5 day | Medium | Before Mar 18 |
| **P5** | Possession sim accuracy validation | 1–2 days | Medium | Post-tournament |
| **P6** | ML recalibration on CLV data | 1–2 weeks | High | Summer |
| **P7** | Live/in-play betting engine | 1+ week | High | Summer |
| **DEFER** | Fantasy Baseball | — | Off-model | Post-Apr 7 |

---

## SPRINT 1 — Critical Data Pipeline Fix (Mar 10–12) 🔴

### P0.1 — BartTorvik Scraper Audit
- [x] Verify `barttorvik.com/2026_team_results.csv` is publicly accessible (no auth required) — ✅ CONFIRMED
- [x] Check if BARTTORVIK_USERNAME/PASSWORD in ratings.py is actually needed or legacy code — ✅ NOT USED (cloudscraper only)
- [x] Run scraper in isolation and confirm > 300 teams returned — ✅ 366 lines (365 teams)
- [x] If auth IS required: obtain credentials and set in Railway — ✅ NOT NEEDED

### P0.2 — EvanMiya Scraper Audit  
- [x] Verify `evanmiya.com/` HTML table is still accessible and scraper parses correctly — ⚠️ BLOCKED by Cloudflare (by design)
- [x] If scraper fails due to HTML structure changes: fix BeautifulSoup selectors — ⚠️ INTENTIONALLY DROPPED
- [x] Fallback option: if EvanMiya is broken and hard to fix, run 2-source (KenPom + BartTorvik) — ✅ ACTIVE
  with appropriate weight renormalization. Do NOT keep running on 1 source. — ✅ 2-SOURCE MODE WORKING

### P0.3 — Live Rating Coverage Endpoint
- [x] Add `GET /admin/ratings/status` endpoint showing team counts per source — ✅ EXISTS at line 1645
- [x] Add logging to nightly analysis: how many teams were loaded per source — ✅ EXISTS in get_all_ratings()
- [x] Alert Discord if any source returns 0 teams — ✅ send_source_health_alert() in discord_notifier.py

### P0.4 — Clean Up Legacy References
- [x] Remove BARTTORVIK_USERNAME/PASSWORD references from docs — ✅ Only in docs, not code

---

## SPRINT 2 — Tournament Edge Improvements (Mar 12–17) 🟠

### P2 — Conference-Specific Home Court Advantage
- [x] Add `CONFERENCE_HCA` dict to `betting_model.py`:
  ```python
  CONFERENCE_HCA = {
      "big_ten": 3.6, "big_12": 3.4, "sec": 3.2,
      "acc": 3.0, "pac_12": 2.8, "wcc": 2.7,
      "aac": 2.6, "mid_major": 2.5,
      "neutral": 0.0,   # Tournament default
  }
  ```
- [x] Create `backend/services/conference_hca.py` service
- [x] Add conference name normalization for various data sources
- [x] Add tests for all conference HCA values
- [ ] Update `analyze_game()` to accept optional `home_conference` param
- [ ] Source conference from team_mapping or tournament_data where available
- [ ] Add tests in `test_betting_model.py::TestConferenceHCA`

### P3 — Late-Season Recency Weighting
- [x] Add `recency_weight_factor` to analysis: weight last 30-day games 2× vs older
- [x] Create `backend/services/recency_weight.py` service
- [x] In `analyze_game()`: if `is_tournament_game=True` or `date >= Mar 15`:
  - Set `neutral_site = True` → HCA = 0.0
  - Increase `margin_se` by +0.20 (tournament = higher upset variance)
- [x] Add late season detection (March 1+)
- [x] Add tournament mode detection (March 15+)
- [x] Add tests for recency weight calculations
- [ ] Tie to existing seed-spread scalar logic in V9.1 model

### P4 — Recalibration Audit & Seed
- [ ] Query: `SELECT COUNT(*) FROM bet_logs WHERE outcome IS NOT NULL AND prediction_id IS NOT NULL`
- [ ] If < 30: paper-trade pipeline has a gap — fix BetLog auto-creation in analysis.py
- [ ] If ≥ 30: run recalibration manually and review home_advantage output
- [ ] Check model_parameters table for existing recalibration entries

### P1 — Sharp Money / Steam Detection
- [ ] Create `backend/services/sharp_money.py`
- [ ] Core signals:
  1. **Reverse Line Movement (RLM)**: line moves against public betting %
  2. **Steam detection**: ≥1.5 pts rapid shift in <30 min across multiple books
  3. **Opener gap**: current line vs. open line divergence magnitude
- [ ] Inputs: `OddsHistory` snapshots already stored by `odds_monitor.py`
  - Use existing `_history` rolling buffer — no new API calls required
- [ ] Output: `SharpSignal(side="home"|"away"|None, confidence=0.0–1.0, pattern=str)`
- [ ] Integration in `analysis.py`:
  - If sharp signal aligns with model → increase `edge_point` by `+0.5%`
  - If sharp signal opposes model → add `PASS` note and widen `margin_se += 0.30`
- [ ] Discord alert on high-confidence RLM events
- [ ] Tests: `tests/test_sharp_money.py`

---

## SPRINT 3 — Validation & Post-Tournament (Mar 17 – Apr 7)

### P5 — Possession Simulator Accuracy Validation
- [ ] Pull last 60+ completed games from DB where possession_sim was used
- [ ] Compare `possession_sim_win_prob` vs `ratings_path_win_prob` vs actual outcome
- [ ] Compute Brier score for each path
- [ ] If possession_sim Brier < ratings path: make it primary
- [ ] If possession_sim Brier > ratings path: demote to secondary signal only
- [ ] Document in `tasks/lessons.md`

### CLV Audit
- [ ] Pull all bets with CLV data from DB
- [ ] Compute CLV by conference, by bet type (spread vs. ML), by ratings source availability
- [ ] Identify which game types are generating +CLV vs. negative CLV
- [ ] Cut any game type with consistently negative CLV

---

## SPRINT 4 — Off-Season Model Overhaul (Apr 7+)

### E-4 — ML-Based Optimal Weight Learning
- [ ] Collect full 2025–26 season CLV + outcome data (requires 100+ settled bets)
- [ ] Features: margin_delta by source, pace, 3PA, HCA, seed, conference, fatigue
- [ ] Target: cover probability (binary outcome)
- [ ] Train XGBoost / logistic regression for each conference cluster
- [ ] Use predicted probabilities as new weight priors for 2026–27

### E-5 — Live/In-Play Betting Engine
- [ ] Markov state machine for real-time game simulation
- [ ] Connect to live score feeds (ESPN API or The Odds API live lines)
- [ ] Compute live win probability with time-remaining decay
- [ ] Gate: only activate when pre-game model direction is confirmed

### E-6 — Alternative Line Shopping
- [ ] Extend `odds.py` to track alt spreads (+0.5, +1.0, +1.5)
- [ ] Surface best alt spread in `GameAnalysis` output
- [ ] Prioritize alt spread when model edge is marginal (within ±0.5% of threshold)

---

## STRATEGIC DIRECTIVES

1. **March Madness Window (Mar 18 – Apr 7): 100% CBB focus.**
   Fantasy Baseball is on ice until after the championship game.

2. **Data Before Features.** Fix the 3-source rating pipeline BEFORE adding more complexity.
   A model with accurate 3-source input beats a complex model running on 1 source.

3. **CLV Is the North Star.** Every model change is evaluated against CLV, not ROI.
   CLV is the leading indicator; ROI lags by weeks and is affected by variance.

4. **Verify Before Building.** Possession simulator and parlay engine exist.
   Measure their contribution before extending them.

5. **Recalibration Must Run.** Without 30+ settled bets, parameters drift.
   Ensure BetLog auto-creation and outcome settlement are working in the pipeline.

---

## REVIEW SECTION

### March 10, 2026 — Architect Review
- **Diagnosis**: KenPom-only mode is the single largest accuracy gap. BartTorvik
  should be freely scrapeable — this is likely a configuration oversight, not a
  technical limitation. Fixing this alone could meaningfully improve edge accuracy.
- **Second priority**: Sharp money detection before Mar 18 to catch tournament
  line movement patterns not visible in ratings.
- **Fantasy Baseball**: Paused. Time-sensitive deadline noted but CBB tournament
  return per-hour is higher between now and Apr 7.
- **Next action**: P0 data pipeline audit is the first commit. No model changes
  until rating sources are verified.

---

_Last updated: 2026-03-10 by Claude (Architect)_
