# OPERATIONAL HANDOFF (EMAC-040)

> Ground truth as of EMAC-040. Operator: Claude Code (Master Architect).
> Read `IDENTITY.md` for risk policy. Read `AGENTS.md` for roles. Read `HEARTBEAT.md` for loops.

---

## 1. SYSTEM STATUS

**Last completed:** EMAC-040 — Full infrastructure triage complete. All env var parsing fixed across 3 files (49 total calls). Railway is live. Model quality under review.

| Component | Status | Detail |
|-----------|--------|--------|
| V9 Model | OK | SNR + Integrity Kelly scalars. `model_version='v9.0'` |
| Railway API | OK | Live and accepting requests. All syntax errors resolved (EMAC-039/040). |
| Env Var Parsing | OK | `get_float_env` applied to **49 calls** across `betting_model.py` (28), `main.py` (7), `analysis.py` (14). Zero plain `float(os.getenv)` remaining. |
| Dashboard API | OK | `API_URL` = Railway production. Data Cleanup endpoint added (`DELETE /admin/games/{id}`). |
| Analysis Pipeline | OK | `bets_recommended=0` on first clean run (12 games). Reason under investigation (K-3). |
| Tournament SD Bump (A-26 T1) | OK | `TOURNAMENT_MODE_SD_BUMP` (1.15x) when `is_neutral=True`. 4 tests. (EMAC-036) |
| Neutral-Site Fix (A-25) | OK | `parse_odds_for_game` extracts `neutral_site` -> `is_neutral`. (EMAC-034) |
| Recalibration | OK | ha=2.419, sd_mult=1.0. Calibrated on 663 settled bets. Oscillation guard active. |
| Railway DB | OK | PostgreSQL connected. 9 tables initialized. Nightly running. |
| Kimi CLI | OK | K-1 complete (tournament intelligence). K-2 complete (seed data research). |
| OpenClaw v2.0 | OK | O-7 coordinator validation: 4/4 tests. Integrity sweep active in analysis pipeline. |
| Seed-Spread Scalars (A-26 T2) | DEFERRED | K-2 complete. Blocked on March 16 bracket. Architecture ready. |
| SNR Re-Audit (A-19) | DEFERRED | Needs 20+ settled V9-era bets. |
| Gemini Trust Level | DEGRADED | 3 multi-file refactor failures in 1 session. Scope-limited until G-12 complete. |

---

## 2. ARCHITECT REVIEW: MODEL QUALITY & TEAM STATE

### Model Quality — Current Concern

The first clean Railway analysis returned `games_analyzed=12, bets_recommended=0`. This may be:

- **A)** Correct conservatism — no genuine edge in today's slate
- **B)** Ratings data absent or degraded in Railway (KenPom/BartTorvik API keys not set, or returning empty)
- **C)** Post-calibration threshold mismatch — `sd_mult=1.0` (up from 0.85 default) widens the distribution, compressing edges below `MIN_BET_EDGE=2.5%`
- **D)** Home advantage `ha=2.419` (vs default 3.09) slightly deflating projected margins

**Kimi K-3 is the right tool** to audit this — she can read all recent predictions + the full analysis pipeline in one pass and pinpoint the root cause.

### Team Assessment

**Claude Code (Architect)** — Role is sound but execution discipline has been broken. Last 3 EMAC sessions were almost entirely spent fixing Gemini's infrastructure incidents. The Architect must be shielded from DevOps firefighting.

**Gemini CLI (DevOps)** — Caused 3 multi-file refactor failures in a single session (EMAC-038 through 040):
1. Dropped 20+ closing parens from `betting_model.py` + `analysis.py` + `test_betting_model.py`
2. Dropped 20+ closing parens from `main.py`
3. Left 14 `float(os.getenv)` calls un-migrated in `analysis.py`
4. Changed math inside sqrt (wrong: `sqrt(total * mult)` vs correct: `sqrt(total) * mult`)

**Gemini is now SCOPE-LIMITED**: No multi-file refactors without explicit Claude approval + mandatory syntax verification before any commit. Gemini's strength is surgical single-file edits and env var management.

**Kimi CLI (Deep Intelligence)** — Performing well within role. K-1 and K-2 both delivered high-quality structured research. K-3 (model quality audit) is the right next use. Context window advantage is being underutilized.

**OpenClaw (Integrity)** — Working correctly in analysis pipeline. O-6 spot-check still pending. This is blocking verification that integrity verdicts are working in production.

---

## 3. K-2 FINDINGS (Seed Data Research)

Full report: `reports/2026-03-06-seed-data-research.md`

| Question | Answer |
|----------|--------|
| Does Odds API include seeds? | NO — team names only |
| Best secondary source | BallDontLie API ($39.99/mo GOAT tier) or ESPN scraper (free) |
| Earliest reliable seed data | ~8 PM ET March 16 (2h after bracket reveal) |
| Code changes needed | New `tournament_data.py` + enrich `game_data` in `analysis.py` |

---

## 4. ACTIVE MISSIONS

---

### KIMI CLI — K-3: Model Quality Audit [PRIORITY]

**Priority:** HIGH — blocking model quality understanding before tournament.

**Context:**
- First clean Railway analysis: 12 games analyzed, 0 bets recommended, 0 errors
- Known calibration state: `ha=2.419`, `sd_mult=1.0` (wider distribution than default 0.85)
- Odds API returned ~66 games; 12 survived the per-game filter (others likely tipped off already)
- Ratings situation unknown in Railway: KenPom/BartTorvik API key status unclear

**Your task — answer these 4 questions in `reports/2026-03-07-model-quality-audit.md`:**

1. **Ratings data health**: Read `backend/services/ratings.py`. Under what conditions do `kenpom`, `barttorvik`, `evanmiya` return empty dicts vs. real data? What happens to the model when all ratings are null?

2. **Edge inflation / compression pipeline**: Read `backend/betting_model.py` (full). Trace the path from `ratings_input` through `projected_margin` -> `edge_point` -> `edge_conservative`. Given `sd_mult=1.0` (widened distribution) + `ha=2.419`, what is the approximate minimum ratings-quality needed to produce `edge_conservative > 0`?

3. **Conservative threshold audit**: Read `backend/services/analysis.py` (full). List every filter that can cause a game to PASS before `edge_conservative` is even computed. Are any of these overly aggressive for regular-season games?

4. **Calibration sanity check**: `sd_mult=1.0` was calibrated on 663 bets from the V8 era. The V9 model has different SNR + integrity scalars that weren't present in those bets. Is `sd_mult=1.0` still appropriate for V9, or should we decouple the calibration dataset?

**Output:** Save to `reports/2026-03-07-model-quality-audit.md`. Add K-3 FINDINGS section to HANDOFF.md. Update title to EMAC-041.

---

### GEMINI CLI — G-11: Railway Env Var Audit [UNBLOCKED]

**Priority:** HIGH — env var formatting bugs have caused 3 production failures. Must verify all are clean.

**Context:**
- Railway stores some env vars as ` =VALUE` (leading space + equals sign) instead of `VALUE`
- `get_float_env` now handles this everywhere in code, but the root cause was never fixed in Railway
- We know `TOURNAMENT_MODE_SD_BUMP` was stored as ` =1.15` — this caused all 54 game errors

**Your task:**
1. Check Railway CBB Edge service Variables tab. List ALL env vars and their current values.
2. For any var stored as `=VALUE` or ` =VALUE` format: correct it to just `VALUE` in Railway UI.
3. Confirm these are set and correctly formatted (no leading spaces or equals signs):
   - `TOURNAMENT_MODE_SD_BUMP=1.15`
   - `MIN_SD_MULT_DELTA=0.03`
   - `SD_MULTIPLIER=0.85` (or confirm it's using default)
   - `BASE_SD=11.0` (or confirm it's using default)
   - `HOME_ADVANTAGE=3.09` (or confirm DB calibration value is being used)
   - `MIN_BET_EDGE=2.5`
4. Verify `GET /health` returns `{"status": "healthy"}` and `GET /api/predictions/today` returns data.

**MANDATORY RULES for this task:**
- Do NOT modify any Python files. Env var management only.
- If you need to update HANDOFF.md, change only the G-11 completion status line.
- Run `python -m ast backend/main.py` before touching any Python file (even for reading).

**Output:** Update the `| Railway API |` status line in Section 1 of HANDOFF.md. Update title to EMAC-041.

---

### GEMINI CLI — G-12: CI Syntax Guard [SCOPED]

**Priority:** MEDIUM — prevent repeat of the dropped-paren incidents.

**Context:**
- Gemini's EMAC-038 commit dropped 20+ closing parens from 4 files
- These failures bypass `pytest` because `main.py` can't be imported without a live DB
- Need a mandatory syntax check that runs BEFORE any commit

**Your task (single-file, no refactoring):**
1. Add to `.github/workflows/deploy.yml` a new step that runs BEFORE pytest:
   ```bash
   python -m py_compile backend/main.py backend/betting_model.py backend/services/analysis.py
   ```
2. This catches syntax errors that pytest misses (no DB required for `py_compile`).
3. Only touch `.github/workflows/deploy.yml`. No Python file changes.

**Output:** Commit the CI change. Update G-12 status in HANDOFF.md. Update title to EMAC-041.

---

### OPENCLAW — O-6: V9 Integrity Spot-Check [UNBLOCKED]

**Priority:** MEDIUM — verifies integrity verdicts are actually working in Railway prod.

**Context:** Railway DB is live. Nightly analysis has run (12 games, 0 bets). V9.0 predictions are in DB.

**Your task:**
1. Call `GET /api/predictions/today` — check if any predictions have `integrity_verdict` populated.
2. If integrity_verdict is null for all predictions: this means no BET-tier games went through the sweep (expected when 0 bets). This is OK — note it.
3. If any predictions exist with `full_analysis.calculations.integrity_verdict`: verify it is NOT "Sanity check unavailable".
4. Report: `O-6 STATUS: [integrity working / not triggered (0 bets) / sanity check unavailable]`
5. Update `HEARTBEAT.md` status tracker with result.

**Output:** Update HEARTBEAT.md. Add O-6 result to HANDOFF.md Section 1 status table.

---

### CLAUDE CODE — A-26 Task 2: Seed-Spread Kelly Scalars [DEFERRED]

**Priority:** MEDIUM — implement after Selection Sunday (March 16) + bracket announced.

**K-2 research complete.** Architecture:
1. Create `backend/services/tournament_data.py` — BallDontLie integration for seed lookup
2. Enrich `game_data` in `analysis.py` `_analyze_games_pass2()` with `seed_home`, `seed_away`
3. Add seed-spread Kelly scalars to `betting_model.py` after the integrity scalar:
   - `#5 seed favored 6+ pts` → 0.75x Kelly
   - `#2 seed favored 17+ pts` → 0.75x Kelly
   - `#8 seed favored <=3 pts` → 0.80x Kelly
4. Env vars: `SEED_DATA_SOURCE=balldontlie`, `BALLDONTLIE_API_KEY=<key>`
5. Tests: add `TestSeedSpreadScalars` to `test_betting_model.py` (4+ tests)
6. Run `pytest tests/ -q` (438+ must pass)

**Earliest start:** 8 PM ET March 16 (bracket data available from BallDontLie).

---

## 5. DEPENDENCY CHAIN

```
G-11 (env var audit) --> Clean production analysis --> K-3 model quality review

K-3 (model quality audit) --> Decide if threshold/calibration adjustment needed

Bracket (Mar 16) + K-2 DONE --> A-26 T2 (seed scalars) --> Tournament Start (Mar 18)

O-6 (integrity spot-check) --> UNBLOCKED — run now

G-12 (CI syntax guard) --> UNBLOCKED — prevents future dropped-paren incidents
```

---

## 6. ARCHITECT REVIEW QUEUE

- **Model calibration review**: After K-3 findings — may need to tune `MIN_BET_EDGE` or re-examine `sd_mult=1.0` in V9 context.
- **Gemini scope policy**: If G-12 installs correctly, expand trust. If further incidents, restrict Gemini to read-only Railway tasks only.
- **A-26 T2 seed scalars**: Architecture ready. Implement March 16+.
- **Tiered integrity (Elite 8+)**: After tournament starts, wire OpenClaw → Kimi escalation path.
- **SNR re-audit (A-19)**: Trigger after 20+ settled V9-era bets.
- **Season-end recalibration**: Run on full V9-era dataset (target N > 500).

---

## 7. HIVE WISDOM

| Lesson | Source |
|--------|--------|
| `pred_id` (Prediction PK) is the correct Streamlit widget key — never `game_id`. | EMAC-019 |
| Always store `base_sd_override` in context. `None` != "same as original". | EMAC-021 |
| `full_analysis.inputs` has no "game" key. Reconstruct from `p.game` DB relationship. | EMAC-023 |
| One-fire sets must be cleared on cache refresh. | EMAC-023 |
| `async def` without `asyncio.to_thread` wrapping sync I/O = ZERO concurrency. | EMAC-027 |
| True calibration entry point is `backend/services/recalibration.py::run_recalibration()`. | EMAC-028 |
| `sys.path` manipulation belongs inside `if __name__ == "__main__":` guards only. | EMAC-029 |
| sd_multiplier oscillates at noise boundary. Min-delta guard (0.03) prevents flip-flopping. | EMAC-031 |
| `parse_odds_for_game` result dict is the single source of truth for game metadata. | EMAC-034 |
| Use Kimi for tasks requiring >50K tokens simultaneously (performance audits, codebase-wide). | EMAC-034 |
| Single-elimination tournament variance is 15-25% higher. Apply SD bump (1.15x) when is_neutral=True. | K-1 |
| Tournament SD bump applies AFTER all other SD penalties. Order matters — bump last. | EMAC-036 |
| Gemini's large-scale refactors drop closing parens. NEVER approve a multi-file Gemini refactor without running `py_compile` on every modified .py file. | EMAC-038 |
| `get_float_env` (backend/utils/env_utils.py) must be used for ALL env var float reads — 49 calls across 3 files. Any new env var float read must use it. | EMAC-037/038/040 |
| `main.py` syntax errors bypass `pytest` (no DB). Always run `python -m py_compile backend/main.py` after any main.py changes. | EMAC-039 |
| When Gemini breaks a file: restore from pre-Gemini commit (`git checkout HASH -- file.py`), apply only legitimate changes via Python regex. Fastest path. | EMAC-039 |
| `sd_mult=1.0` (recalibrated from V8 era) widens distribution, compresses edges. May need V9-specific recalibration after 50+ V9 bets settle. | EMAC-040 |

---

## 8. HANDOFF PROMPTS

### PROMPT FOR CLAUDE CODE
```
MISSION: EMAC-040 — Standby / A-26 T2 post-March 16
You are Claude Code, Master Architect for CBB Edge Analyzer.

CONTEXT:
- 438/438 tests passing. Railway live. All env var parsing fixed (49 calls).
- First clean production analysis: 12 games, 0 bets, 0 errors. Model quality under review.
- K-3 audit (Kimi) will determine if calibration/threshold adjustment needed.
- A-26 T2 (seed scalars) is your next implementation task — earliest March 16.

STANDBY TASKS (if K-3 surfaces actionable findings):
1. If MIN_BET_EDGE needs tuning: update analysis.py ENV default, add regression test.
2. If sd_mult needs V9-specific recalibration trigger: update recalibration.py guard logic.
3. After March 16: implement A-26 T2 per Section 4.

GUARDIAN DUTIES:
- Review any Gemini commit before pulling to main. Run `python -m py_compile` on all .py files.
- If Gemini breaks a file, restore from git history — never try to hand-edit back.
```

### PROMPT FOR GEMINI CLI
```
MISSION: EMAC-040 / G-11 + G-12 — Env Var Audit + CI Syntax Guard
You are the DevOps Strike Lead (Gemini CLI) for CBB Edge Analyzer.

READ HANDOFF.md Section 4 (G-11 and G-12) for exact steps.

CRITICAL RULES — YOU MUST FOLLOW THESE:
1. NO multi-file Python refactoring. Single file only, and only if explicitly required.
2. Before committing ANY Python change: run `python -m py_compile <file>.py` and confirm OK.
3. Before committing ANYTHING: run `pytest tests/ -q --tb=no` and confirm 438 pass.
4. Your task is env var audit (G-11) + CI syntax step (G-12). Do not expand scope.

CONTEXT:
- Railway env vars may have ' =VALUE' formatting bugs. Audit and correct them.
- Past incidents: dropped parens in 4 files from large refactor. This cannot happen again.
- G-12: add py_compile step to .github/workflows/deploy.yml only. No Python file changes.
```

### PROMPT FOR KIMI CLI
```
MISSION: EMAC-040 / K-3 — Model Quality Audit
You are the Deep Intelligence Unit (Kimi CLI) for CBB Edge Analyzer.

READ HANDOFF.md Section 4 (K-3) for exact questions to answer.

CONTEXT:
- Railway production analysis returned 0 bets on 12 games analyzed.
- Calibrated params: home_advantage=2.419, sd_multiplier=1.0 (widened distribution).
- Key files to read: backend/betting_model.py, backend/services/analysis.py, backend/services/ratings.py
- You have a 1M token context — read all 3 files simultaneously for maximum coherence.

TASK: Answer 4 questions defined in Section 4 K-3. Save to:
reports/2026-03-07-model-quality-audit.md

Add K-3 FINDINGS section to HANDOFF.md. Update title to EMAC-041.
```

### PROMPT FOR OPENCLAW
```
MISSION: EMAC-040 / O-6 — V9 Integrity Spot-Check
You are the Integrity Execution Unit (OpenClaw).

CONTEXT:
- Railway analysis returned 0 bets on 12 games. Integrity sweep only runs on BET-tier games.
- If 0 bets → sweep was not triggered → integrity_verdict should be null in all predictions.
- This is not an error — it confirms the pipeline works correctly.

TASK: Follow O-6 steps in Section 4. If sweep was not triggered (0 bets), note it as
"O-6 STATUS: Not triggered (correct behavior — 0 BET-tier games in slate)".
Update HEARTBEAT.md status tracker. Add result to HANDOFF.md Section 1.
```
