# OPERATIONAL HANDOFF (EMAC-039)

> Ground truth as of EMAC-039. Operator: Claude Code (Master Architect).
> Read `IDENTITY.md` for risk policy. Read `AGENTS.md` for roles. Read `HEARTBEAT.md` for loops.

---

## 1. SYSTEM STATUS

**Last completed:** EMAC-039 — Fixed `backend/main.py` SyntaxError (Gemini's 20+ dropped parens). Railway redeploy in progress. 438/438 tests passing.

| Component | Status | Detail |
|-----------|--------|--------|
| V9 Model | OK | SNR + Integrity Kelly scalars. `model_version='v9.0'` |
| Railway API | REDEPLOYING | `main.py` SyntaxError fixed (EMAC-039). Gemini's 20+ dropped parens reverted. Push a20a43a. |
| Env Var Parsing | OK | `backend/utils/env_utils.py::get_float_env` — handles leading spaces/equals signs. 28 calls in `betting_model.py`, 7 in `main.py`. (EMAC-038/039) |
| Dashboard API | OK | `API_URL` = Railway production in `dashboard/utils.py`. (EMAC-037) |
| Neutral-Site Fix (A-25) | OK | `parse_odds_for_game` extracts `neutral_site` -> `is_neutral`. 5 tests. (EMAC-034) |
| Tournament SD Bump (A-26 T1) | OK | `TOURNAMENT_MODE_SD_BUMP` (default 1.15x) when `is_neutral=True`. 4 tests. (EMAC-036) |
| Recalibration | OK | ha=2.419, sd=1.0. 663 settled bets. Oscillation guard active. (EMAC-029/031) |
| Railway DB | OK | PostgreSQL connected. 9 tables initialized. Nightly analysis producing v9.0 predictions. (EMAC-037) |
| Kimi CLI | OK | Onboarded. K-1 complete. K-3 (coordinator validation) complete. (EMAC-034/035) |
| OpenClaw v2.0 | OK | O-7 coordinator validation: 4/4 tests. Local LLM latency ~5.7s. Escalation logic working. |
| Seed-Spread Scalars (A-26 T2) | DEFERRED | Blocked on K-2 + bracket (Selection Sunday March 16). |
| V9 Prod Verification | OK | Nightly analysis triggered; v9.0 predictions confirmed in DB. (EMAC-037) |
| SNR Re-Audit (A-19) | DEFERRED | Needs 20+ settled V9-era bets. |

---

## 2. K-1 FINDINGS (Tournament Intelligence)

Full report: `reports/2026-03-06-tournament-intelligence.md`

| Finding | Priority | Status |
|---------|----------|--------|
| SD multiplier understates tournament variance 15-25% | HIGH | DONE — A-26 T1 implemented (EMAC-036) |
| #5 seeds as 6+pt favorites: 33% ATS | OPPORTUNITY | DEFERRED — A-26 T2 after Mar 16 |
| #2 seeds favored 17+pts: 37% ATS | OPPORTUNITY | DEFERRED — A-26 T2 after Mar 16 |
| #8 seeds as small favorites: 23% ATS | OPPORTUNITY | DEFERRED — A-26 T2 after Mar 16 |
| Conference overvaluation (SEC 46%, MWC 34% ATS) | MEDIUM | Future enhancement |
| ha=2.419 unvalidated on neutral sites | MEDIUM | A-25 zeroes HCA; monitored |

---

## 3. ACTIVE MISSIONS

---

### CLAUDE CODE — A-26 Task 2: Seed-Spread Kelly Scalars

**Priority:** MEDIUM — DEFERRED until Selection Sunday March 16 + K-2 findings.

**What's done:**
- A-26 T1 complete: `TOURNAMENT_MODE_SD_BUMP` (1.15x) in `analyze_game()` when `is_neutral=True`. 4 tests.
- `get_float_env` cleanly applied to all 28 env var reads in `betting_model.py`. No broken parens.

**When unblocked (after March 16 + K-2 from Kimi):**
Implement seed-spread Kelly scalars post-Kelly in `analyze_game()`:
- `#5 seed favored 6+ pts` -> 0.75x Kelly
- `#2 seed favored 17+ pts` -> 0.75x Kelly
- `#8 seed favored <=3 pts` -> 0.80x Kelly
Requires `seed_home` / `seed_away` in `game_data` (K-2 tells us how to source this).
Run `pytest tests/ -q` (must pass 438+). Update HANDOFF.md to EMAC-039.

---

### GEMINI CLI — G-11: Verify Prod Deployment Health

**Priority:** LOW — prod DB is up, verify everything is running cleanly.

1. Confirm `GET /health` returns `{"status": "healthy"}` on Railway
2. Check `GET /api/predictions/today` returns v9.0 predictions
3. Confirm `TOURNAMENT_MODE_SD_BUMP=1.15` is set in Railway CBB Edge service Variables
4. Confirm `MIN_SD_MULT_DELTA=0.03` is set (oscillation guard env var)
5. Update HANDOFF.md with G-11 confirmation

---

### KIMI CLI — K-2: Seed Data Research [COMPLETE]

**Status:** COMPLETE — Full report: `reports/2026-03-06-seed-data-research.md`

**K-2 FINDINGS:**

| Question | Answer | Implementation |
|----------|--------|----------------|
| Does Odds API include seeds? | **NO** — team names only (e.g., "Duke Blue Devils") | Must use secondary source |
| Best secondary source | **BallDontLie API** ($39.99/mo GOAT tier) — official API with seed field | Recommended for production |
| Earliest reliable data | **~8 PM ET March 16** (2h after bracket reveal) | Schedule first run at 8:30 PM ET |
| Code changes needed | New `tournament_data.py`, modify `analysis.py` + `betting_model.py` | See report for code samples |

**Key Code Changes for Claude (A-26 T2):**
1. Create `backend/services/tournament_data.py` with BallDontLie integration
2. Add seed enrichment to `analysis.py` `analyze_daily_games()`
3. Add seed-spread Kelly scalars to `betting_model.py`:
   - #5 seed favored by 6+ pts → 0.75x Kelly
   - #2 seed favored by 17+ pts → 0.75x Kelly
   - #8 seed favored by ≤3 pts → 0.80x Kelly

**Environment Variables Needed:**
```bash
SEED_DATA_SOURCE=balldontlie
BALLDONTLIE_API_KEY=your_key_here
```

**Unblocks:** A-26 Task 2 — Seed-spread Kelly scalars ready for implementation.

---

### OPENCLAW — O-6: V9 Integrity Spot-Check

**Status:** Ready to run (G-10 is complete, nightly analysis confirmed).

1. Query: `SELECT id, verdict, integrity_verdict, full_analysis->>'calculations' FROM predictions WHERE model_version='v9.0' ORDER BY created_at DESC LIMIT 1`
2. Check `full_analysis["calculations"]["integrity_verdict"]` — must not be "Sanity check unavailable"
3. Report `O-6 CONFIRMED: [verdict]` or escalate to Gemini as G-12 if Ollama unreachable in Railway
4. Update HEARTBEAT.md with result

---

## 4. DEPENDENCY CHAIN

```
K-2 (Seed research) --> A-26 T2 (Seed scalars) --> Selection Sunday (Mar 16)

A-26 T1 (SD bump) DONE --> Tournament Start (Mar 18)

O-6 (Integrity spot-check) --> UNBLOCKED — run now
```

---

## 5. ARCHITECT REVIEW QUEUE

- **A-26 T2 seed scalars**: Design done. Deferred until bracket (March 16). Unblocked by K-2.
- **Tiered integrity (post-tournament)**: After season, wire OpenClaw -> Kimi escalation in `analysis.py` for Elite Eight+ games.
- **SNR re-audit (A-19)**: Deferred. Trigger: 20+ settled V9-era bets.
- **Season-end recalibration**: At season end, run on full V9-era dataset (N > 500 target).

---

## 6. HIVE WISDOM

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
| Use Kimi for tasks requiring >50K tokens simultaneously. | EMAC-034 |
| Single-elimination tournament variance is 15-25% higher. Apply SD bump (1.15x) when is_neutral=True. | K-1 |
| Tournament SD bump applies AFTER all other SD penalties. Order matters — bump last. | EMAC-036 |
| Gemini's large-scale refactors drop closing parens in multi-line rewrites. Always run pytest immediately after Gemini's commits. | EMAC-038 |
| Use `get_float_env` (backend/utils/env_utils.py) for ALL env var reads — handles Railway env var formatting quirks (leading spaces, equals signs). | EMAC-037/038 |
| `main.py` syntax errors don't surface via `pytest` (no DB in CI). Always run `python -c "import ast; ast.parse(open('backend/main.py').read())"` after any main.py changes. | EMAC-039 |
| When Gemini breaks `main.py`: restore from clean pre-Gemini commit, then apply only `get_float_env` substitutions via Python regex. Fastest path back. | EMAC-039 |

---

## 7. HANDOFF PROMPTS

### PROMPT FOR CLAUDE CODE
```
MISSION: EMAC-038 / A-26 Task 2 — Seed-Spread Kelly Scalars (DEFERRED)
You are Claude Code, Master Architect for CBB Edge Analyzer.
Read HANDOFF.md Section 3 (Claude task) for full details.

CONTEXT:
- 438/438 tests passing.
- A-26 T1 complete (tournament SD bump). get_float_env applied cleanly.
- Next: seed-spread scalars AFTER Selection Sunday (March 16) + K-2 findings.

WHEN UNBLOCKED (after March 16 + K-2):
Implement seed-spread Kelly scalars per Section 3.
Run pytest (must pass 438+). Update HANDOFF.md to EMAC-039.
```

### PROMPT FOR GEMINI CLI
```
MISSION: EMAC-038 / G-11 — Verify Prod Health
You are the DevOps Strike Lead (Gemini CLI) for CBB Edge Analyzer.
Read HANDOFF.md Section 3 (Gemini task) for exact steps.

CONTEXT:
- Railway DB is live. V9 predictions confirmed in DB.
- Verify health endpoint, check env vars are set, confirm nightly is clean.
- If any env vars are missing, set them now.
```

### PROMPT FOR KIMI CLI
```
MISSION: EMAC-039 / A-26 T2 — Seed-Spread Kelly Scalars
You are Claude Code, Master Architect for CBB Edge Analyzer.

CONTEXT:
- K-2 research complete. Full report: reports/2026-03-06-seed-data-research.md
- Odds API does NOT include seed data — must use BallDontLie API (paid) or ESPN scraping (free)
- Tournament starts March 18; bracket revealed March 16

TASK:
1. Create backend/services/tournament_data.py with BallDontLie integration
2. Modify analysis.py to fetch/attach seed data during tournament
3. Modify betting_model.py to apply seed-spread Kelly scalars:
   - #5 seed favored by 6+ pts → 0.75x Kelly
   - #2 seed favored by 17+ pts → 0.75x Kelly  
   - #8 seed favored by ≤3 pts → 0.80x Kelly
4. Add SEED_DATA_SOURCE and BALLDONTLIE_API_KEY env vars
5. Run pytest (438 must pass)

Update HANDOFF.md to EMAC-040.
```

### PROMPT FOR OPENCLAW
```
MISSION: EMAC-038 / O-6 — V9 Integrity Spot-Check
You are the Integrity Execution Unit (OpenClaw).

G-10 is COMPLETE. Nightly analysis has run. V9.0 predictions are in Railway DB.
You are unblocked. Run O-6 per Section 3.
Report O-6 CONFIRMED or escalate to Gemini as G-12.
Update HEARTBEAT.md with result.
```
