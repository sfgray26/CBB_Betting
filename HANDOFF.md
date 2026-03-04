# 🦅 OPERATIONAL HANDOFF (EMAC-021)

> Ground truth as of EMAC-021. Operator: Claude Code (Master Architect).
> Read `IDENTITY.md` for risk policy. Read `AGENTS.md` for roles. Read `HEARTBEAT.md` for loops.

---

### 1. MISSION INTEL (Ground Truth)

**Operator:** Claude Code (Master Architect)
**Mission accomplished:** EMAC-020 — Data Deduplication & UI Integrity (Mission Clean Sweep)

**Technical State (cumulative):**

| Component | Status | Detail |
|-----------|--------|--------|
| A-11: API Deduplication | ✅ | `/api/predictions/today` now deduplicates by `game_id`. Priority: nightly > opener, then best edge. Fixes the 28-bets-on-77-games anomaly. |
| A-12: UI Dedup Guard | ✅ | `dashboard/pages/1_Todays_Bets.py` adds last-resort client-side dedup by `game_id`. Even if API response contains duplicates, UI shows one card per matchup. |
| A-13: Code Sweep | ✅ | All service files audited. Zero production `print()` calls — only in `__main__` script blocks (betting_model, odds, sentinel, models). No misaligned args found. |
| A-14: SNR Floor Audit | ✅ | `_snr_kelly_scalar()` correctly reads `SNR_KELLY_FLOOR` env (default 0.5). Not being bypassed. Audit confirms n=3 resolved BETs — SNR calibration deferred until n >= 20 per tier. |
| Confidence audit (90d) | ✅ | 3 resolved BETs. 0/3 cover rate (statistically meaningless). SNR_KELLY_FLOOR=0.5 maintained. Re-audit when n >= 20. |
| Full test suite | ✅ | **408/408 passing** after clean sweep. |
| G-3 Railway Env Vars | ⚠️ | **USER ACTION REQUIRED.** Set `SNR_KELLY_FLOOR`, `INTEGRITY_CAUTION_SCALAR`, `INTEGRITY_VOLATILE_SCALAR` in Railway Dashboard. |

---

### 2. ROOT CAUSE: The 28-Bet Anomaly

The inflated bet count was caused by **run_tier duplication**, not a model bug:
- `analysis.py` correctly deduplicates per `run_tier` (nightly vs opener in isolation)
- BUT if both `nightly` and `opener_attack` jobs run on the same day, the same game gets **two separate Prediction rows** (one per tier)
- `/api/predictions/today` previously returned ALL predictions → duplicates surfaced in UI

**Fix applied (two-layer defense):**
1. **API layer** (`backend/main.py:434`): `seen{}` dict keyed by `game_id`, priority `nightly > opener > others`, then best edge tiebreaker
2. **UI layer** (`dashboard/pages/1_Todays_Bets.py:24`): `_seen_gids{}` guard before rendering any card

The SNR math and Kelly floor logic were NOT the culprits — confirmed clean.

---

### 3. DELEGATION BUNDLE: Gemini CLI (DevOps Strike Lead)

**Task G-3 — Railway Env Vars (STILL PENDING USER ACTION)**
- Add to Railway Dashboard: `SNR_KELLY_FLOOR=0.5`, `INTEGRITY_CAUTION_SCALAR=0.75`, `INTEGRITY_VOLATILE_SCALAR=0.5`
- Redeploy + verify `GET /health` returns 200.

**Task G-5 — ReanalysisEngine Wiring (Level 5 Prep)**
- `ReanalysisEngine` + `CachedGameContext` are live in `betting_model.py`. Wire them in `analysis.py`:
  1. After the BET/CONSIDER `model.analyze_game()` call, create an engine and store it:
     ```python
     engine = ReanalysisEngine.from_analysis_pass(model, game_data, odds_input, ratings, ...)
     _reanalysis_cache[_game_key] = engine
     ```
  2. Return `_reanalysis_cache` from `run_nightly_analysis()` alongside the summary dict.
  3. Pass the cache to the `OddsMonitor` so its callback can call `engine.reanalyze(new_spread)`.
- This is EMAC-022 scope.

---

### 4. DELEGATION BUNDLE: OpenClaw (Integrity Execution Unit)

**Task O-5 — Async Verification (outstanding since EMAC-017)**
- Call `POST /admin/run-analysis` with `X-API-Key` header.
- Paste the exact log lines showing `"Triggering concurrent integrity sweep for N candidates..."` with timestamps.
- Verbal claim without log evidence will be rejected.

---

### 5. ARCHITECT REVIEW QUEUE (Next EMAC)

- **Level 5 wiring**: After Gemini completes G-5, peer-review `analysis.py` changes + add ReanalysisEngine tests.
- **SNR re-audit**: Trigger when n >= 20 alpha bets in DB. Run `python scripts/audit_confidence.py --days 90 --min-bets 20`.
- **run_tier strategy review**: Consider whether opener_attack predictions should automatically supersede nightly when they run after (currently nightly always wins). This is a product decision.

---

### 6. HIVE WISDOM (Operational Lessons)

| Lesson | Source |
|--------|--------|
| Verified `/admin/scheduler/status` tracks the new Performance Sentinel job. | EMAC-017 |
| CLI env var setting is brittle in restricted shells; Railway Dashboard UI is source of truth for secrets. | EMAC-017 |
| `GameAnalysis.notes` populated for all game runs — full LLM pipeline confirmed. | EMAC-018 |
| OpenClaw verbal claims without log evidence are unacceptable for O-5 class tasks. | EMAC-017/018 |
| n=50 in directive was aspirational. DB had only 3 resolved BETs. Always verify DB count via audit script before acting on sample-size claims. | EMAC-019 |
| `pred_id` (Prediction PK) is the correct Streamlit widget key — never `game_id` which is not unique per-render. | EMAC-019 |
| Inflated bet count (28 on 77-game slate) was run_tier deduplication failure, not model logic. Fix at API + UI layers. | EMAC-020 |

---

### 7. HANDOFF PROMPTS — COPY AND PASTE THESE

#### PROMPT FOR GEMINI CLI
```
MISSION: EMAC-021 — Railway Env Vars + Level 5 ReanalysisEngine Wiring
You are the DevOps Strike Lead (Gemini CLI).
Read HANDOFF.md Sections 3 and 7 for your directives (Tasks G-3, G-5).

Execute in order:
1. Complete G-3: Set SNR_KELLY_FLOOR, INTEGRITY_CAUTION_SCALAR, INTEGRITY_VOLATILE_SCALAR in Railway Dashboard. Redeploy + verify GET /health returns 200.
2. Begin G-5: Wire ReanalysisEngine.from_analysis_pass() into the BET/CONSIDER loop in analysis.py. Store engines in _reanalysis_cache dict keyed by game_key. Return cache from run_nightly_analysis(). Pass to OddsMonitor callback.
3. Update HANDOFF.md to EMAC-022 when complete.
```

#### PROMPT FOR OPENCLAW
```
MISSION: EMAC-021 — O-5 Async Sweep Live Log Evidence (final)
You are the Integrity Execution Unit (OpenClaw).
Read HANDOFF.md Section 4 for your directive.

Execute:
1. Call POST /admin/run-analysis with X-API-Key header.
2. Capture the ACTUAL log output. Paste the exact line containing "integrity sweep" with timestamp.
3. Report "ASYNC CONFIRMED: [exact log line]" or "SYNC -- needs fix: [reason]".
4. Update HEARTBEAT.md integrity_sweep.last_verified with today's date.
```

#### PROMPT FOR CLAUDE CODE
```
MISSION: EMAC-022 — ReanalysisEngine Test Coverage + Level 5 Peer Review
You are Claude Code, Master Architect for CBB Edge Analyzer.
Read HANDOFF.md Section 5 for your review queue.

When Gemini completes G-5 (ReanalysisEngine wiring in analysis.py):
1. Peer-review the analysis.py changes — verify cache is keyed correctly and passed to OddsMonitor.
2. Add 4 tests to tests/test_betting_model.py:
   - reanalyze() with unchanged spread returns same verdict as analyze_game()
   - reanalyze() with spread moved past threshold flips PASS -> BET verdict
   - reanalyze() with new_total updates base_sd_override correctly
   - from_analysis_pass() factory correctly snapshots all context fields
3. Run pytest tests/ -q --tb=short — must pass.
4. Update HANDOFF.md to EMAC-023.
```
