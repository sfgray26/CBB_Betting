# OPERATIONAL HANDOFF (EMAC-042)

> Ground truth as of EMAC-042. Operator: Claude Code (Master Architect).
> Read `IDENTITY.md` for risk policy. Read `AGENTS.md` for roles. Read `HEARTBEAT.md` for loops.

---

## 1. SYSTEM STATUS

**Last completed:** EMAC-042 — All `float(os.getenv)` calls replaced with `get_float_env` across every backend file (commits 80bc0eb, 748462b). G-12 CI syntax guard live in deploy.yml (commit 3281b51). Railway redeploying — analysis errors should be fully resolved.

| Component | Status | Detail |
|-----------|--------|--------|
| V9 Model | OK | SNR + Integrity Kelly scalars. `model_version='v9.0'` |
| Railway API | OK | Live. All syntax errors resolved (EMAC-039/040/042). |
| Env Var Parsing | OK | `get_float_env` applied to ALL float reads: `betting_model.py` (14), `main.py` (7), `analysis.py` (14), `odds.py` (2). Zero plain `float(os.getenv)` remaining. |
| CI Syntax Guard (G-12) | COMPLETE | `py_compile` step added to `deploy.yml` before pytest. Catches dropped-paren errors without DB. |
| Railway Env Var Audit (G-11) | OPEN | Root-cause fix: correct ` =VALUE` vars in Railway UI. Assigned to Gemini. |
| Dashboard API | OK | `API_URL` = Railway production. Data Cleanup endpoint added (`DELETE /admin/games/{id}`). |
| Analysis Pipeline | OK | `bets_recommended=0` is correct conservatism per K-3 audit. V9-specific recalibration needed at 50 bets. |
| Tournament SD Bump (A-26 T1) | OK | `TOURNAMENT_MODE_SD_BUMP` (1.15x) when `is_neutral=True`. 4 tests. |
| Neutral-Site Fix (A-25) | OK | `parse_odds_for_game` extracts `neutral_site` -> `is_neutral`. |
| Recalibration | OK | ha=2.419, sd_mult=1.0. V8-calibrated; V9-specific recalibration after 50 settled V9 bets. |
| Railway DB | OK | PostgreSQL connected. 9 tables initialized. Nightly running. |
| OpenClaw v2.0 | OK | O-7 coordinator validation: 4/4 tests. Integrity sweep wired in analysis pipeline. |
| Integrity Spot-Check (O-6) | OPEN | Verify integrity_verdict fields in production predictions. Assigned to OpenClaw. |
| Seed-Spread Scalars (A-26 T2) | DEFERRED | Blocked on March 16 bracket. K-4 (Kimi) will produce implementation spec. |
| A-26 T2 Architecture Spec (K-4) | OPEN | Kimi to produce exact implementation blueprint for A-26 T2. See Section 4. |
| SNR Re-Audit (A-19) | DEFERRED | Needs 20+ settled V9-era bets. |
| Gemini Trust Level | DEGRADED → RESTORING | G-12 now in CI. G-11 (single-file env audit) is the trust-rebuild task. No multi-file Python work until 2 clean G-11-style tasks complete. |

---

## 2. SYSTEM CONTEXT

### Model Quality — K-3 Verdict (FINAL)

**0 bets on 12 games = correct conservatism. Not a bug.**

Root causes:
1. `sd_mult=1.0` (vs default 0.85) = +17.6% wider SD → ~2.8pp edge compression
2. `ha=2.419` (vs default 3.09) = -21.7% less HCA → compressed home-team margins
3. V9 structural mismatch: 663 V8-era calibration bets had no SNR/integrity scalars. V9 applies 0.25–1.0x combined Kelly scalar — structural under-sizing relative to calibrated thresholds.

**No code changes needed now.** V9-specific recalibration is the long-term fix, triggered after 50 settled V9-era bets.

### Team State

| Agent | Role | Trust | Notes |
|-------|------|-------|-------|
| Claude Code | Master Architect | FULL | Standby until March 16. Guardian of all commits. |
| Gemini CLI | DevOps Strike Lead | DEGRADED → RESTORING | Single-file tasks only. `py_compile` before every commit. |
| Kimi CLI | Deep Intelligence | FULL | K-1/K-2/K-3 all high-quality. K-4 queued. |
| OpenClaw | Integrity Execution | FULL | O-6 queued. Integrity sweep live in pipeline. |

**Gemini scope rule:** No multi-file Python refactors. Each task must be a single file or a non-Python file. Before committing any `.py` file: `python -m py_compile <file>.py`. Before any push: `pytest tests/ -q --tb=no` (438 must pass).

---

## 3. COMPLETED WORK (ARCHIVE)

| Mission | Who | What |
|---------|-----|------|
| K-1 | Kimi | Tournament intelligence: SD bump 1.15x validated |
| K-2 | Kimi | Seed data research: BallDontLie API recommended, ESPN free fallback |
| K-3 | Kimi | Model quality audit: 0 bets = correct conservatism confirmed |
| G-12 | Claude (took over from degraded Gemini) | `py_compile` step in `deploy.yml` |
| A-25 | Claude | Neutral-site fix: `parse_odds_for_game` → `is_neutral` |
| A-26 T1 | Claude | Tournament SD bump (1.15x) when `is_neutral=True` |
| EMAC-037–042 | Claude | `get_float_env` applied to all 37 float env reads; Railway SyntaxErrors fixed |

---

## 4. ACTIVE MISSIONS

---

### GEMINI CLI — G-11: Railway Env Var Audit [OPEN — HIGH PRIORITY]

**What:** Railway stores some env vars as ` =VALUE` (leading space + equals sign). The code now tolerates this via `get_float_env`, but the correct fix is to clean the vars at source in Railway.

**Why it matters:** The ` =1.15` bug caused 54 game analysis failures across 3 production sessions. Even with `get_float_env` as a safety net, malformed vars add risk and confusion.

**Your tasks (Railway UI only — no Python file changes):**

1. Open Railway dashboard → CBB Edge service → Variables tab.
2. List every variable. For any stored as ` =VALUE` or `=VALUE`: correct it to `VALUE`.
3. Specifically verify these 6 variables are clean (no leading space, no `=` prefix):
   - `TOURNAMENT_MODE_SD_BUMP` → should be `1.15`
   - `MIN_SD_MULT_DELTA` → should be `0.03`
   - `SD_MULTIPLIER` → should be `0.85` (if set; default is fine)
   - `BASE_SD` → should be `11.0` (if set; default is fine)
   - `MIN_BET_EDGE` → should be `2.5`
   - `HOME_ADVANTAGE` → should not be set (DB calibration value `2.419` is authoritative)
4. After saving: hit Railway's "Redeploy" button to pick up changes.
5. Verify: `GET https://cbbbetting-production.up.railway.app/health` returns `{"status": "healthy"}`.
6. Trigger a manual analysis from Admin Panel and confirm zero `' =VALUE'` errors in response.

**Output:** Update HANDOFF.md Section 1 `| Railway Env Var Audit (G-11) |` row to `COMPLETE`. Update title to EMAC-043.

**SCOPE RULES:**
- No Python file changes whatsoever.
- No `git` commands.
- If anything is unclear, stop and report back — do not improvise.

---

### OPENCLAW — O-6: V9 Integrity Spot-Check [OPEN — MEDIUM PRIORITY]

**What:** Verify that V9 integrity verdicts (`VOLATILE`/`CAUTION`/`CONFIRMED`) are being populated in production predictions.

**Context:**
- Analysis pipeline: `_integrity_sweep()` runs ONLY on BET-tier games (not PASS/CONSIDER).
- Last slate had 0 BET-tier games. So integrity_verdict is expected to be null for all predictions — this is NOT a bug.
- The spot-check goal is to verify the field exists and the pipeline didn't silently error.

**Your tasks:**

1. Call `GET https://cbbbetting-production.up.railway.app/api/predictions/today` with header `X-API-Key: <your key>`.
2. For each prediction in the response:
   - Check if `integrity_verdict` field exists anywhere in the JSON (top-level or nested under `full_analysis.calculations`).
   - Check if any verdict is `"Sanity check unavailable"` — this would indicate a silent Ollama failure.
3. Expected result: `integrity_verdict` is null or absent for all predictions (correct — 0 BET-tier games means sweep was not triggered).
4. Report one of:
   - `O-6 STATUS: Not triggered — correct (0 BET-tier games in slate, integrity_verdict=null for all)`
   - `O-6 STATUS: Active — N predictions have verdicts: [list them]`
   - `O-6 STATUS: BROKEN — "Sanity check unavailable" on prediction [id]`

**Output:**
- Update `HEARTBEAT.md` with O-6 status and timestamp.
- Update HANDOFF.md Section 1 `| Integrity Spot-Check (O-6) |` row to COMPLETE with verdict.
- Update title to EMAC-043 (or increment from whatever Gemini left it at).

---

### KIMI CLI — K-4: A-26 T2 Architecture Spec [OPEN — MEDIUM PRIORITY]

**What:** Produce a precise implementation blueprint for A-26 T2 (Seed-Spread Kelly Scalars) so Claude can implement it in under 2 hours on March 16.

**Context:**
- K-2 research found: Odds API has no seed data. BallDontLie API ($39.99/mo GOAT tier) provides seeds. ESPN scraper is free fallback.
- Bracket drops ~6 PM ET March 16. BallDontLie has data ~8 PM ET. Tournament games start March 18.
- Claude needs to implement A-26 T2 during the March 16–18 window (< 48 hours).

**Files to read (read all simultaneously):**
- `backend/betting_model.py` — full file, understand the Kelly chain: `kelly_frac` → SNR scalar → integrity scalar. Your scalars go AFTER integrity scalar.
- `backend/services/analysis.py` — find `_analyze_games_pass2()`, understand `game_data` dict structure passed to `analyze_game()`.
- `backend/services/ratings.py` — understand how external API calls are structured (pattern to follow for BallDontLie).
- `reports/2026-03-06-seed-data-research.md` — your K-2 findings.

**Produce a spec covering exactly:**

1. **BallDontLie API contract**: Exact endpoint URL, request params, response shape, auth header. What field contains seed number? What field contains team name (and how does it map to KenPom team name)?

2. **`backend/services/tournament_data.py` blueprint**: Full class/function signatures with docstrings. Should include: `get_team_seed(team_name: str, year: int) -> Optional[int]`, caching strategy (TTL, key), fallback if API unavailable.

3. **`analysis.py` enrichment point**: Exact line/function where `seed_home` and `seed_away` are added to `game_data`. Show the exact `game_data["seed_home"] = ...` lines.

4. **`betting_model.py` scalar logic**: The exact `_seed_spread_kelly_scalar(seed_fav, seed_dog, spread)` function code, handling all edge cases (both seeds None → 1.0x, only one seed → 1.0x, etc.).

5. **Scalar table** (finalize from K-2 findings — these are the current estimates, verify or adjust):
   - `#5 seed favored 6+ pts over #12` → 0.75x Kelly (upset risk)
   - `#2 seed favored 17+ pts over #15` → 0.75x Kelly (inflated spread, public money)
   - `#8 or #9 seed favored <= 3 pts` → 0.80x Kelly (nearly even matchup)
   - All other seeded matchups → 1.0x (no adjustment)

6. **Env vars needed**: List with defaults.

7. **Test cases for `TestSeedSpreadScalars`**: Write out 5+ test method signatures with inputs/expected outputs.

**Output:** Save full spec to `reports/2026-03-16-a26t2-implementation-spec.md`. Update HANDOFF.md Section 1 `| A-26 T2 Architecture Spec (K-4) |` to COMPLETE. Update title to EMAC-043.

---

### CLAUDE CODE — A-26 Task 2: Seed-Spread Kelly Scalars [DEFERRED — March 16]

**Earliest start:** 8 PM ET March 16 (BallDontLie has bracket data ~2h after 6 PM reveal).

**Input:** K-4 spec from Kimi (`reports/2026-03-16-a26t2-implementation-spec.md`).

**Implementation steps:**
1. Create `backend/services/tournament_data.py` — BallDontLie seed lookup with TTL cache + ESPN fallback.
2. Enrich `game_data` in `analysis.py::_analyze_games_pass2()` with `seed_home`, `seed_away`.
3. Add `_seed_spread_kelly_scalar()` to `betting_model.py` after integrity scalar in `analyze_game()`.
4. Add env vars to `.env.example`: `SEED_DATA_SOURCE=balldontlie`, `BALLDONTLIE_API_KEY=`.
5. Add `TestSeedSpreadScalars` to `tests/test_betting_model.py` (5+ tests per K-4 spec).
6. Run `pytest tests/ -q` — all 438+ must pass.
7. `python -m py_compile` on every touched `.py` file before committing.

**Guardian rule:** Do NOT start before K-4 spec is complete. If K-4 spec is missing, Kimi first.

---

## 5. DEPENDENCY CHAIN

```
G-11 (Gemini — Railway env var cleanup)
  --> Confirms root cause fixed at source, not just in code

O-6 (OpenClaw — integrity spot-check)
  --> UNBLOCKED — run now
  --> Output feeds HEARTBEAT.md

K-4 (Kimi — A-26 T2 spec)
  --> UNBLOCKED — run now
  --> Output feeds A-26 T2 implementation

Bracket reveals March 16 @ 6 PM ET
  + K-4 spec complete
  --> A-26 T2 implementation (Claude, March 16-18)
  --> Tournament starts March 18
```

---

## 6. ARCHITECT REVIEW QUEUE

- **March 16 window**: A-26 T2 must be implemented and deployed in < 48h after bracket. Prep Railway deploy pipeline.
- **Gemini trust restoration**: G-11 is the first trust-rebuild task. If G-11 completes cleanly (no Python touches, no scope creep), consider unlocking single-.py-file tasks. Still no multi-file work.
- **V9 recalibration trigger**: At 50 settled V9-era bets, run recalibration. Current bet rate suggests this is weeks away — set a HEARTBEAT.md monitor.
- **Tiered integrity (Elite 8+)**: After March 22, wire OpenClaw → Kimi escalation for high-stakes games.
- **SNR re-audit (A-19)**: After 20+ settled V9-era bets.
- **Season-end recalibration**: Full V9-era dataset (target N > 500). Off-season task.

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
| `get_float_env` (backend/utils/env_utils.py) must be used for ALL env var float reads. Any new float env var MUST use it. | EMAC-037/038/042 |
| `main.py` syntax errors bypass `pytest` (no DB). Always run `python -m py_compile backend/main.py` after any main.py change. | EMAC-039 |
| When Gemini breaks a file: restore from pre-Gemini commit (`git checkout HASH -- file.py`), apply legitimate changes via Python regex. Fastest path. | EMAC-039 |
| `sd_mult=1.0` widens distribution, compresses edges. V9-specific recalibration after 50+ V9 bets settle. | EMAC-040/K-3 |
| Uncommitted local changes are invisible to Railway. Always verify changes are pushed before attributing errors to the code fix. | EMAC-042 |

---

## 8. HANDOFF PROMPTS

### PROMPT FOR CLAUDE CODE
```
MISSION: EMAC-042 — Standby / Guardian until March 16
You are Claude Code, Master Architect for CBB Edge Analyzer.
Working directory: C:\Users\sfgra\repos\Fixed\cbb-edge

SYSTEM STATE (all confirmed):
- 438/438 tests passing. Railway live and healthy.
- get_float_env applied to all 37 float env var reads (no plain float(os.getenv) remaining).
- CI syntax guard (py_compile) installed in .github/workflows/deploy.yml.
- G-12 COMPLETE. G-11 assigned to Gemini. O-6 assigned to OpenClaw. K-4 assigned to Kimi.
- 0 bets on 12 games = correct conservatism (K-3 verdict). No code changes needed.

YOUR IMMEDIATE TASKS:
None — standby mode until March 16.

IF GEMINI SUBMITS A COMMIT: Review it. Run:
  python -m py_compile <every .py file changed>
  pytest tests/ -q --tb=no
If either fails, do NOT merge. Restore file from git history and re-apply only the legitimate delta.

IF K-4 SPEC IS COMPLETE (Kimi): Read reports/2026-03-16-a26t2-implementation-spec.md.
Confirm spec is implementable. Reply with any gaps to fill before March 16.

ON MARCH 16 AFTER 8 PM ET: Implement A-26 T2 per Section 4 of HANDOFF.md.
Input: K-4 spec. Output: tournament_data.py + analysis.py enrichment + betting_model.py scalars.
```

---

### PROMPT FOR GEMINI CLI
```
MISSION: EMAC-042 / G-11 — Railway Env Var Audit
You are the DevOps Strike Lead (Gemini CLI) for CBB Edge Analyzer.

YOUR SINGLE TASK: Audit and fix Railway environment variables.
Read HANDOFF.md Section 4 (G-11) for exact steps.

CRITICAL RULES — NON-NEGOTIABLE:
1. NO Python file changes. This task is Railway UI only.
2. NO git commands. Railway UI changes only.
3. Do not expand scope. G-11 is the entire mission.
4. If you are unsure about any step, stop and report back.

CONTEXT:
- Railway stores some env vars as ' =VALUE' (leading space + equals sign).
- TOURNAMENT_MODE_SD_BUMP was stored as ' =1.15' — caused 54 game analysis failures.
- The code now tolerates this via get_float_env, but fix the root cause in Railway UI.

VARIABLES TO CHECK (in Railway CBB Edge service → Variables):
  TOURNAMENT_MODE_SD_BUMP → must be: 1.15
  MIN_SD_MULT_DELTA       → must be: 0.03
  MIN_BET_EDGE            → must be: 2.5
  SD_MULTIPLIER           → must be: 0.85 (if set; OK to delete and use default)
  BASE_SD                 → must be: 11.0 (if set; OK to delete and use default)
  HOME_ADVANTAGE          → must NOT be set (DB calibration value 2.419 is authoritative)

VERIFICATION:
After saving vars and redeploying: GET https://cbbbetting-production.up.railway.app/health
Expected: {"status": "healthy"}
Then trigger manual analysis from Admin Panel and confirm no ' =VALUE' errors.

OUTPUT: Update HANDOFF.md Section 1 G-11 row to COMPLETE. Update title to EMAC-043.
```

---

### PROMPT FOR KIMI CLI
```
MISSION: EMAC-042 / K-4 — A-26 T2 Architecture Spec
You are the Deep Intelligence Unit (Kimi CLI) for CBB Edge Analyzer.
You have a 1M token context window — use it.

YOUR TASK: Produce a precise implementation blueprint for Seed-Spread Kelly Scalars.
Claude will use this spec to implement A-26 T2 in < 2 hours on March 16.
The spec must be complete enough that Claude needs zero research — just code.

READ THESE FILES SIMULTANEOUSLY:
1. backend/betting_model.py (full file — understand analyze_game() Kelly chain)
2. backend/services/analysis.py (find _analyze_games_pass2(), understand game_data dict)
3. backend/services/ratings.py (understand external API call pattern to replicate for BallDontLie)
4. reports/2026-03-06-seed-data-research.md (your K-2 findings on seed data sources)

PRODUCE A SPEC covering:
1. BallDontLie API contract: endpoint, auth, response shape, which field = seed, team name format
2. tournament_data.py blueprint: full function signatures with types and docstrings
3. analysis.py enrichment: exact lines where seed_home/seed_away added to game_data
4. betting_model.py scalar: exact _seed_spread_kelly_scalar() function code with edge cases
5. Final scalar table (validate or adjust from K-2 estimates)
6. Env vars with defaults
7. 5+ test case specs for TestSeedSpreadScalars

SAVE TO: reports/2026-03-16-a26t2-implementation-spec.md

UPDATE HANDOFF.md:
- Section 1: change K-4 row to COMPLETE
- Title: EMAC-043
```

---

### PROMPT FOR OPENCLAW
```
MISSION: EMAC-042 / O-6 — V9 Integrity Spot-Check
You are the Integrity Execution Unit (OpenClaw) for CBB Edge Analyzer.

YOUR TASK: Verify V9 integrity verdicts in production.
Read HANDOFF.md Section 4 (O-6) for exact steps.

CONTEXT:
- Railway analysis returned 0 BET-tier games. Integrity sweep only runs on BET-tier games.
- Expected result: integrity_verdict is null/absent for all predictions (sweep was not triggered).
- This is correct behavior — the check is confirming the field exists and no silent errors occurred.

API CALL:
GET https://cbbbetting-production.up.railway.app/api/predictions/today
Header: X-API-Key: <your key>

CHECK FOR:
- Does integrity_verdict field appear anywhere in the response JSON?
- Is any verdict equal to "Sanity check unavailable"? (This = Ollama failure — escalate to Claude)
- Are all verdicts null? (This = correct, sweep not triggered)

REPORT ONE OF:
  O-6 STATUS: Not triggered — correct (0 BET-tier games, integrity_verdict=null for all)
  O-6 STATUS: Active — [N] predictions have verdicts: [list]
  O-6 STATUS: BROKEN — "Sanity check unavailable" on prediction [id] — escalate to Claude

OUTPUT:
- Update HEARTBEAT.md with O-6 status + timestamp.
- Update HANDOFF.md Section 1 O-6 row to COMPLETE with status string.
- Update title to EMAC-043 (or increment from whatever Gemini left it at).
```
