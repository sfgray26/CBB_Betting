# OPERATIONAL HANDOFF (EMAC-036)

> Ground truth as of EMAC-036. Operator: Claude Code (Master Architect).
> Read `IDENTITY.md` for risk policy. Read `AGENTS.md` for roles. Read `HEARTBEAT.md` for loops.

---

### 1. SYSTEM STATUS

**Last completed:** A-26 Task 1 — Tournament Mode SD bump (Claude). 438/438 tests passing.
**Prod DB:** Railway PostgreSQL provisioned + `init_db.py` run successfully (9 tables confirmed). CBB Edge service still needs `DATABASE_URL = ${{ Postgres.DATABASE_URL }}` set and a redeploy.

**Technical State:**

| Component | Status | Detail |
|-----------|--------|--------|
| V9 Model | OK | SNR + Integrity Kelly scalars. `model_version='v9.0'` |
| Neutral-Site Fix (A-25) | OK | `parse_odds_for_game` extracts `neutral_site` → `is_neutral`. 5 tests. (EMAC-034) |
| Tournament SD Bump (A-26 T1) | OK | `TOURNAMENT_MODE_SD_BUMP` (default 1.15x) applied when `is_neutral=True`. 4 tests. (EMAC-036) |
| Recalibration | OK | ha=2.419, sd=1.0. 663 settled bets. sd_multiplier oscillation guard added. (EMAC-029/031) |
| Railway Env Vars (G-3) | OK | `SNR_KELLY_FLOOR`, `INTEGRITY_CAUTION_SCALAR`, `INTEGRITY_VOLATILE_SCALAR` set. |
| Railway DB | PARTIAL | PostgreSQL provisioned. `init_db.py` confirmed 9 tables. CBB Edge service needs `DATABASE_URL` variable + redeploy to connect. |
| Kimi CLI | OK | Onboarded as Deep Intelligence Unit. K-1 complete. (EMAC-034) |
| Seed-Spread Scalars (A-26 T2) | DEFERRED | Needs K-2 seed data research + bracket (Selection Sunday March 16). |
| V9 Prod Verification | PENDING | Blocked on Railway redeploy. Once done, trigger nightly and confirm v9.0 predictions. |
| SNR Re-Audit (A-19) | DEFERRED | Needs 20+ settled V9-era bets. Trigger after nightly run. |

---

### 2. K-1 FINDINGS (Kimi Tournament Intelligence — full report: `reports/2026-03-06-tournament-intelligence.md`)

| Finding | Risk | Action |
|---------|------|--------|
| ha=2.419 unvalidated on neutral-site data | MEDIUM | A-25 zeros HCA correctly; no further action needed pre-tournament |
| SD multiplier understates tournament variance by 15-25% | HIGH | Add `TOURNAMENT_MODE_SD_BUMP=1.15` env var; apply when `is_neutral=True` |
| Matchup engine factors degraded in tournament | MEDIUM | Pace/3PA/drop less reliable; eFG pressure remains stable — no code change needed yet |
| #5 seeds as 6+pt favorites: 33% ATS since 2009 | OPPORTUNITY | 0.75x Kelly scalar for these scenarios |
| #2 seeds favored 17+pts: 37% ATS since 2005 | OPPORTUNITY | 0.75x Kelly scalar for large #2 spreads |
| #8 seeds as small favorites (≤3): 23% ATS | OPPORTUNITY | 0.80x Kelly scalar |
| SEC 46% ATS, MWC 34% ATS in tournament over last decade | MEDIUM | Conference-specific Kelly scalars |

**Claude's A-26 task** is to implement the HIGH-priority SD bump and the seed-spread scalars. See Section 4.

---

### 3. RECALIBRATION HISTORY (for context)

| Date | Change |
|------|--------|
| 2026-02-23 | ha 1.59→1.5, sd 0.94→0.97 |
| 2026-02-25 | sd 0.97→1.0 |
| 2026-02-27 | sd 1.0→0.97, ha 1.5→2.0 |
| 2026-03-04 | weight_kenpom 0.45→0.444, weight_barttorvik 0.40→0.409 (auto_daily) |
| 2026-03-05 | ha 2.0→2.419, sd 0.97→1.0 (EMAC-029) |

---

### 4. ACTIVE MISSIONS

---

#### CLAUDE CODE — A-26 Task 2: Seed-Spread Kelly Scalars (DEFERRED — wait for K-2 + bracket)

**Priority:** MEDIUM. Deferred until Selection Sunday March 16 + K-2 findings from Kimi.

**What was done (EMAC-036):**
- A-26 Task 1 complete: `TOURNAMENT_MODE_SD_BUMP` (default 1.15x) applied in `analyze_game()`
  when `is_neutral=True`. 4 tests added. 438/438 passing.
- Implementation: `backend/betting_model.py` lines ~1149-1163 (after heuristic FF block)
- Env var `TOURNAMENT_MODE_SD_BUMP=1.15` should be added to Railway CBB Edge service

**What's next (implement after March 16):**
Once K-2 findings confirm seed data source, implement seed-spread Kelly scalars:
- `#5 seed favored by 6+ pts` → 0.75x Kelly
- `#2 seed favored by 17+ pts` → 0.75x Kelly
- `#8 seed favored by ≤3 pts` → 0.80x Kelly
Implementation point: post-Kelly scaling in `analyze_game()` using `game_data['seed_home']` / `game_data['seed_away']`.
Requires K-2 research on how to inject seed data into game_data.

---

#### GEMINI CLI — G-10: Railway Redeploy + V9 Prod Verification

**Context:** PostgreSQL provisioned. `init_db.py` run locally with Railway's public DATABASE_URL —
all 9 tables confirmed (`model_parameters, performance_snapshots, data_fetches, team_profiles,
alerts, games, predictions, closing_lines, bet_logs`). The CBB Edge app service is still
connecting to the wrong DB — it needs its Variables tab updated and a redeploy.

**Steps:**
1. Open Railway Dashboard → CBB Edge project → CBB Edge service → Variables
2. Set `DATABASE_URL = ${{ Postgres.DATABASE_URL }}` (Railway internal reference)
3. Set `TOURNAMENT_MODE_SD_BUMP = 1.15` (new env var from A-26)
4. Redeploy the CBB Edge service
5. Confirm `GET /health` returns `{"status": "healthy"}`
6. Trigger `POST /admin/run-analysis` (header: `X-API-Key: <API_KEY_USER1>`)
7. Query: `SELECT id, model_version, snr, created_at FROM predictions ORDER BY created_at DESC LIMIT 3`
8. Confirm at least one row with `model_version='v9.0'` and non-null `snr`
9. Update HANDOFF.md status table (Railway DB → OK) and HEARTBEAT.md with confirmation

---

#### KIMI CLI — K-2: Seed Data Research

**Goal:** Determine how to get tournament seed data into the system before the bracket
is announced (Selection Sunday March 16). This unblocks Claude's A-26 Task 2.

**Questions to answer:**
1. Does The Odds API include seed numbers in the team name or event metadata for
   NCAA tournament games? (e.g., does it return "Duke" or "1 Duke"?)
2. If not, what is the best secondary data source for seed → team name mapping?
   Options: ESPN API, BartTorvik tournament page, NCAA.com, manual entry.
3. What is the earliest reliable source for seed data after Selection Sunday (March 16)?
   How quickly does it propagate to available APIs?
4. What code change is needed in `backend/services/odds.py` or `analysis.py` to
   attach seed data to `game_data` before it reaches `betting_model.py`?

**Output:** Save findings to `reports/2026-03-06-seed-data-research.md`.
Add K-2 FINDINGS section to HANDOFF.md and update title to EMAC-036.

---

#### OPENCLAW — O-6: V9 Integrity Verdict Spot-Check

**Prerequisite:** G-10 must be complete first (prod DB must be connected).
Do not run until Gemini confirms G-10 done and a nightly analysis has produced
`model_version='v9.0'` predictions.

Once unblocked:
1. Query: `SELECT id, verdict, integrity_verdict, full_analysis->>'calculations' FROM predictions WHERE model_version='v9.0' ORDER BY created_at DESC LIMIT 1`
2. Check `full_analysis["calculations"]["integrity_verdict"]` — must not be "Sanity check unavailable"
3. Report `O-6 CONFIRMED: [verdict]` or escalate to Gemini as G-11 if Ollama is unreachable in Railway
4. Update HEARTBEAT.md with result

---

### 5. ARCHITECT REVIEW QUEUE

- **A-26 seed scalars**: Design is done. Deferred until bracket (March 16). Unblocked by K-2 seed data research.
- **Tiered integrity (post-tournament)**: After season, implement OpenClaw → Kimi escalation path in `analysis.py` for Elite Eight+ games.
- **SNR re-audit (A-19)**: Deferred. Trigger: 20+ settled V9-era bets after G-10 + nightly run.
- **Season-end recalibration**: At season end, run on full V9-era dataset (N > 500 target).

---

### 6. HIVE WISDOM (Operational Lessons)

| Lesson | Source |
|--------|--------|
| `pred_id` (Prediction PK) is the correct Streamlit widget key — never `game_id`. | EMAC-019 |
| Always store `base_sd_override` in context. `None` != "same as original" — means "use model default". | EMAC-021 |
| `full_analysis.inputs` has no "game" or "game_data" key. Reconstruct from `p.game` DB relationship. | EMAC-023 |
| One-fire sets must be cleared on cache refresh — otherwise new nightly runs cannot trigger new flips. | EMAC-023 |
| Discord embed fields using `:.1%` or `:.2f` on None crash silently. Guard with `or 0.0`. | EMAC-025 |
| `async def` without `asyncio.to_thread` wrapping sync I/O = ZERO concurrency. | EMAC-027 |
| True calibration entry point is `backend/services/recalibration.py::run_recalibration()`. | EMAC-028 |
| `sys.path` manipulation belongs inside `if __name__ == "__main__":` guards, never at module level. | EMAC-029 |
| `ModelParameter` columns: `id, effective_date, parameter_name, parameter_value, parameter_value_json, reason, changed_by, created_at`. No `updated_at` or `version`. | EMAC-029 |
| sd_multiplier oscillates at the noise boundary. Min-delta guard (0.03) added to prevent flip-flopping. | EMAC-031 |
| `parse_odds_for_game` result dict is the single source of truth for game metadata. Extract all needed fields here. | EMAC-034 |
| Calibration data is only as good as its neutrality distribution. If <20 neutral-site bets in training, tournament ha=0 is unvalidated. | EMAC-034 |
| Use Kimi for tasks requiring >50K tokens of simultaneous context. Gemini hits limits; Claude chunks; Kimi reads whole. | EMAC-034 |
| Single-elimination tournament variance is 15-25% higher than regular season. Apply SD bump (1.15x) when is_neutral=True. | K-1 |
| Seed-spread patterns are exploitable via Kelly scalars, not margin adjustments. Scalars preserve model integrity while respecting market inefficiency. | K-1 |
| Tournament SD bump applies AFTER all other SD penalties (degraded-mode, heuristic FF). Order matters — bump last, ceiling check after. | EMAC-036 |

---

### 7. HANDOFF PROMPTS

#### PROMPT FOR CLAUDE CODE
```
MISSION: EMAC-036 / A-26 Task 2 — Seed-Spread Kelly Scalars (DEFERRED)
You are Claude Code, Master Architect for CBB Edge Analyzer.
Read HANDOFF.md Section 4 (Claude task) for full details.

CONTEXT:
- Tournament starts March 18. 438/438 tests passing.
- A-26 Task 1 complete: TOURNAMENT_MODE_SD_BUMP=1.15 applied when is_neutral=True.
- Next: implement seed-spread Kelly scalars AFTER Selection Sunday (March 16) + K-2 findings.

WHEN UNBLOCKED (after March 16 + K-2 from Kimi):
Implement seed-spread Kelly scalars per the lookup table in Section 4.
Requires seed_home / seed_away in game_data (K-2 tells you how to source this).
Run pytest (must pass 438+). Update HANDOFF.md to EMAC-037.
```

#### PROMPT FOR GEMINI CLI
```
MISSION: EMAC-036 / G-10 — Railway CBB Edge Service Variables + Redeploy
You are the DevOps Strike Lead (Gemini CLI) for CBB Edge Analyzer.
Read HANDOFF.md Section 4 (Gemini task) for exact steps.

CONTEXT:
- Railway PostgreSQL is now provisioned and init_db.py has been run (9 tables created).
- The CBB Edge app service still needs DATABASE_URL and TOURNAMENT_MODE_SD_BUMP set.
- 438/438 tests passing locally.

MISSION: Set variables on CBB Edge service, redeploy, confirm health,
trigger nightly analysis, verify V9 predictions in DB.
Update HEARTBEAT.md with confirmation.
```

#### PROMPT FOR KIMI CLI
```
MISSION: EMAC-036 / K-2 — Seed Data Research
You are the Deep Intelligence Unit (Kimi CLI) for CBB Edge Analyzer.
Read HANDOFF.md Section 4 (Kimi task) for full details.

CONTEXT:
- Tournament bracket announced Selection Sunday March 16 (10 days away)
- Claude needs seed data (1-16 per region) attached to game_data to implement
  seed-spread Kelly scalars from K-1 findings
- Your job: research how to get this data into the system

TASK: Answer the 4 questions in Section 4 (Kimi task).
Save findings to reports/2026-03-06-seed-data-research.md.
Add K-2 FINDINGS section to HANDOFF.md. Update title to EMAC-036.
```

#### PROMPT FOR OPENCLAW
```
MISSION: EMAC-036 / O-6 — V9 Integrity Spot-Check
You are the Integrity Execution Unit (OpenClaw).

PREREQUISITE: Do not run until Gemini confirms G-10 (Railway DB) is fixed
and a nightly analysis has completed with model_version='v9.0' in predictions.

Once unblocked, execute per HANDOFF.md Section 4 (OpenClaw task).
Report O-6 CONFIRMED or escalate to Gemini as G-11.
Update HEARTBEAT.md with result.
```


---

### 10. OPENCLAW v2.0 COORDINATION (New)

**Lead Coordinator:** Kimi CLI (Deep Intelligence Unit)  
**Local Engine:** qwen2.5:3b via Ollama  
**Deployed:** 2026-03-06

#### A. Architecture Changes

OpenClaw has been upgraded from simple Ollama wrapper to an intelligent routing system:

**New Components:**
| File | Purpose |
|------|---------|
| `.openclaw/config.yaml` | Routing rules, budgets, circuit breaker settings |
| `.openclaw/coordinator.py` | Task routing logic with Circuit Breaker pattern |
| `.openclaw/README.md` | Usage guide for hive agents |
| `.openclaw/token-usage.jsonl` | Cost tracking (auto-generated) |

**Routing Logic:**
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Task Request  │────▶│  Coordinator    │────▶│  Local (qwen)   │
│  (with context) │     │  (Kimi logic)   │     │  Fast, Free     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼ (if high-stakes)
                        ┌─────────────────┐
                        │  Kimi CLI       │
                        │  Deep analysis  │
                        └─────────────────┘
```

#### B. Routing Rules (v2.0)

| Condition | Route | Rationale |
|-----------|-------|-----------|
| `recommended_units >= 1.5` | **Kimi** | High exposure requires synthesis |
| `tournament_round >= 4` (Elite Eight+) | **Kimi** | Stakes too high for local |
| `integrity_verdict contains VOLATILE` | **Kimi** | Complex risk scenario |
| `sources_disagree AND edge > 5%` | **Kimi** | Contradictory signals |
| Standard integrity checks | Local → Kimi fallback | Pattern-based, fast |
| Scouting reports | Local only | Low-stakes narrative |
| Health narratives | Local only | Templated output |

#### C. Circuit Breaker Protection

If local LLM fails 5+ times in 60 seconds:
1. Circuit **OPENS** → all traffic routes to Kimi
2. After 60s cooldown → **HALF_OPEN** (test calls)
3. If healthy → **CLOSED** → resume local routing

**Monitor:** `get_coordinator().circuit_breaker.state`

#### D. Cost Budgeting

Daily budget: `$5.00 USD` for Kimi escalations  
Alert at: `80%` ($4.00)  
Log file: `.openclaw/token-usage.jsonl`

**Current Status:**
```bash
# Check daily spend
cat .openclaw/token-usage.jsonl | jq -s '[.[] | select(.engine=="kimi") | .cost_usd] | add'
```

#### E. Hive Responsibilities

**Kimi CLI (Coordinator):**
- Monitor routing effectiveness
- Handle escalated high-stakes tasks
- Adjust routing rules in `config.yaml`
- Review circuit breaker logs weekly

**Claude Code (Architect):**
- Review `coordinator.py` architecture
- Approve routing rule changes
- Ensure circuit breaker doesn't mask systemic issues

**Gemini CLI (DevOps):**
- Keep Ollama service running: `ollama run qwen2.5:3b`
- Monitor disk usage for token logs
- Alert if local LLM repeatedly fails (circuit breaker flapping)

**OpenClaw (Local LLM):**
- Handle all low-stakes tasks
- Escalate on uncertainty
- Maintain <10s response time
- Report failures immediately

#### F. Migration Notes

**Backward Compatibility:**
- `scout.py::perform_sanity_check()` unchanged
- New `coordinator.py::check_integrity()` wraps it with routing
- Existing code continues working

**Gradual Migration:**
```python
# Old way (still works)
from backend.services.scout import perform_sanity_check
result = await perform_sanity_check(...)

# New way (with routing)
from .openclaw.coordinator import check_integrity, TaskContext
ctx = TaskContext(recommended_units=1.5)
result = await check_integrity(..., context=ctx)
# → Auto-escalates to Kimi if high-stakes
```

#### G. Escalation Protocol

When OpenClaw routes to Kimi, the return value is `"KIMI_ESCALATION"`:

```python
result = await check_integrity(..., context=high_stakes_ctx)
if result == "KIMI_ESCALATION":
    # Log to HANDOFF.md or notify operator
    # Kimi CLI will handle this task directly
    pass
```

---

**Questions about OpenClaw v2.0?** Coordinate through Kimi CLI as lead.
