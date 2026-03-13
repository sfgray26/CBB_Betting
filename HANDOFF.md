# OPERATIONAL HANDOFF (EMAC-069)

> Ground truth as of March 13, 2026. Operator: Claude Code (Master Architect).
> See `IDENTITY.md` for risk policy · `AGENTS.md` for roles · `HEARTBEAT.md` for loops.
> Full roadmap: `docs/MLB_FANTASY_ROADMAP.md` · CBB plan: `tasks/cbb_enhancement_plan.md`

---

## 0. STANDING DECISIONS

- **Gemini CLI is Research-Only.** No production code. Deliverables go to `docs/` as markdown.
- **All production code: Claude Code only.**
- **GUARDIAN (Mar 18 - Apr 7):** Do NOT touch `betting_model.py`, `analysis.py`, or CBB services. All pre-tournament fixes are COMPLETE — no further changes before Apr 7.

---

## 1. SYSTEM STATUS

### CBB Model — V9.1 (recalibration queued for Apr 7)

| Component | Status |
|-----------|--------|
| Railway API | ✅ Healthy |
| PostgreSQL | ✅ Connected (365 teams) |
| Scheduler | ✅ 10 jobs running |
| Discord | ✅ 16 channels operational |
| Test suite | ✅ 683/686 pass (3 pre-existing DB-auth failures) |
| V9.1 Model | ⚠️ Over-conservative — MIN_BET_EDGE lowered to 1.8%; ha/sd_mult/SNR queued Apr 7 |
| Haslametrics scraper | ✅ Built — `backend/services/haslametrics.py`, 12 tests. Wire in Apr 7. |
| OPCL-001 Discord | ✅ Morning brief + telemetry live. Cron wiring pending (user action). |

### Fantasy Baseball — DRAFT-READY

| Component | Status | File |
|-----------|--------|------|
| Yahoo OAuth + Draft Board + Live Tracker | ✅ COMPLETE | `11_Fantasy_Baseball.py`, `12_Live_Draft.py` |
| Draft Tracker backend + Discord alerts | ✅ COMPLETE | `draft_tracker.py`, `discord_notifier.py` |
| Bet settlement fix | ✅ COMPLETE | `bet_tracker.py` — `_resolve_home_away()` |

---

## 2. UPCOMING DEADLINES

| Date | Event | Owner | Action Required |
|------|-------|-------|----------------|
| **Mar 17 ~7 PM ET** | O-9 Pre-tournament sweep | OpenClaw | See Section 4 |
| **Mar 18** | First Four begins | All | Tournament monitoring mode — GUARDIAN active |
| **Mar 20** | Fantasy Keeper Deadline | User | Set keepers in Yahoo UI |
| **Mar 23 7:30am ET** | Fantasy Draft Day | User | Run `12_Live_Draft.py` |
| **Apr 7** | Tournament window closes | Claude Code | Guardian lifts — execute Section 5 mission |

---

## 3. PRE-TOURNAMENT WORK LOG

All work below is shipped and test-validated. Full details in `reports/` and git history.

| Mission | What was done | Status |
|---------|---------------|--------|
| EMAC-068 Fix 1 | `EVANMIYA_DOWN_SE_ADDEND` default 0.30→0.00; `margin_se` back to 1.50 | ✅ |
| EMAC-068 Fix 2 | `FORCE_HOME_ADVANTAGE` / `FORCE_SD_MULTIPLIER` env var overrides in analysis.py | ✅ |
| EMAC-068 Phase 1 | `MIN_BET_EDGE` default 2.5%→1.8% in betting_model.py, analysis.py, line_monitor.py | ✅ |
| OPCL-001 | OpenClaw morning brief + telemetry Discord modules (24 tests pass) | ✅ |
| EMAC-069 | Haslametrics scraper `backend/services/haslametrics.py` (12 tests pass) | ✅ |
| K-11 | CLV attribution — positive CLV confirmed (+0.5–1.0 pts). Full report: `reports/K11_CLV_ATTRIBUTION_MARCH_2026.md` | ✅ |
| K-12 | V9.2 recalibration spec — sd_mult→0.80, ha→2.85, SNR floor→0.75. Full report: `reports/K12_RECALIBRATION_SPEC_V92.md` | ✅ |
| K-13 | Possession sim audit — KEEP verdict (push-aware Kelly, 24 tests). K-14 A/B monitoring post-Apr 7. Full report: `reports/K13_POSSESSION_SIM_AUDIT.md` | ✅ |
| G-R7 | Haslametrics recommended as EvanMiya replacement. Scraper already built. Full spec: `docs/THIRD_RATING_SOURCE.md` | ✅ |
| G-R1–R5 | MLB research complete (Steamer, lineups, closers, Statcast, Yahoo API). Docs in `docs/` | ✅ |

**Why the model has been over-conservative:** V9.1 stacks SNR scalar (~0.70) × integrity scalar (~0.85) × fractional Kelly (÷2.0) = effective divisor ~3.37×, applied on top of V8 params that were calibrated at ÷2.0. MIN_BET_EDGE fix (Phase 1) partially addressed this. Full fix is Phase 2 (Apr 7+).

---

## 4. ACTIVE TASKS

### OpenClaw — before March 18

- [ ] `python scripts/openclaw_scheduler.py --morning-brief --test` — verify Discord embeds
- [ ] `python scripts/openclaw_scheduler.py --telemetry-check --test`
- [ ] Add openclaw_scheduler to Railway cron (daily 7 AM + every 30 min)
- [ ] **O-9 sweep (Mar 17 ~7 PM ET):**
  ```bash
  ls data/pre_tournament_baseline_2026.json  # if missing: python scripts/openclaw_baseline.py --year 2026
  python scripts/test_discord.py
  # GET /admin/odds-monitor/status  — expect games_tracked > 0
  ```
  For each First Four matchup: run `check_integrity_heuristic()`. Flag ABORT or VOLATILE.

### Claude Code — April 7+

Execute in order. Run `pytest tests/ -q` before each commit.

1. **V9.2 Phase 2 params** — in `betting_model.py` / `analysis.py`:
   - `sd_mult` 1.0 → 0.80
   - `ha` 2.419 → 2.85
   - `SNR_KELLY_FLOOR` 0.50 → 0.75
   - Reference: `reports/K12_RECALIBRATION_SPEC_V92.md`

2. **Wire Haslametrics** — scraper already built at `backend/services/haslametrics.py`:
   - Add `from backend.services.haslametrics import get_haslametrics_ratings` to `ratings.py`
   - Assign EvanMiya's former 32.5% weight to Haslametrics in `CBBEdgeModel.weights`
   - Reference: `docs/THIRD_RATING_SOURCE.md`

3. **K-14 pricing engine tracking** — in `analysis.py` + DB migration:
   - Add `pricing_engine` column to `Prediction` model (values: `"markov"` / `"gaussian"`)
   - Write field per-prediction in analysis pipeline
   - Reference: `reports/K13_POSSESSION_SIM_AUDIT.md`

4. **Bump version + validate** — set `model_version = 'v9.2'`, run full test suite, confirm BET rate improvement.

---

## 5. NEXT CLAUDE SESSION PROMPT (post-Apr 7)

```
CONTEXT: Guardian window lifted. CBB model work resumes. All intelligence is in.

STATE:
- V9.1 is over-conservative (effective Kelly divisor ~3.37x vs intended ~2.0x)
- MIN_BET_EDGE already lowered to 1.8% (Phase 1, pre-tournament)
- Haslametrics scraper already built: backend/services/haslametrics.py (12 tests pass)
- K-11 confirms genuine positive CLV — recalibration is directionally correct

MISSION EMAC-070: V9.2 Recalibration + Haslametrics
1. betting_model.py / analysis.py: sd_mult 1.0→0.80, ha 2.419→2.85, SNR_KELLY_FLOOR 0.50→0.75
   Read reports/K12_RECALIBRATION_SPEC_V92.md for exact justification
2. ratings.py: wire backend/services/haslametrics.py as 3rd source (32.5% weight, replaces EvanMiya)
   Read docs/THIRD_RATING_SOURCE.md for integration spec
3. analysis.py + models.py: add pricing_engine field to Prediction, write "markov"/"gaussian" per game
   Read reports/K13_POSSESSION_SIM_AUDIT.md for K-14 spec
4. Bump model_version to 'v9.2'. Run pytest tests/ -q. Confirm BET rate increase.

TARGET: BET rate 3% → 8-12%. CLV already positive (K-11) — just need to unblock the bets.
```

---

## 6. QUICK REFERENCE

```bash
python -m pytest tests/ -q
python scripts/preflight_check.py
python scripts/test_discord.py
railway logs --follow
streamlit run dashboard/app.py
```

---

## 7. HIVE WISDOM

| Lesson | Source |
|--------|--------|
| V9.1 effective Kelly divisor ~3.37× — calibrated params were for ÷2.0 | EMAC-067 |
| CLV > 0 = genuine edge. No amount of tuning fixes CLV < 0 | K-11 |
| Haslametrics uses play-by-play garbage-time filter — cleaner than EvanMiya | G-R7 |
| MIN_BET_EDGE 2.5% was too high given wide CI — 1.8% is the right pre-v9.2 value | K-12 |
| possession_sim: push-aware Kelly is worth keeping; add A/B monitoring not removal | K-13 |
| Bet settlement: use `_resolve_home_away()` — never raw string compare | EMAC-064 |
| Yahoo roster pre-draft returns `players:[]` (empty array) — handle gracefully | EMAC-063 |
| Prediction dedup: `run_tier` NULL causes duplicate rows — use `or_()` filter | EMAC-067 |
| Discord token must be in Railway Variables, not just .env | D-1 |
| Avoid non-ASCII chars in output strings (CP-1252 Windows terminal issue) | Python |

---

**Document Version:** EMAC-069
**Last Updated:** March 13, 2026
**Status:** Pre-tournament complete. 683/686 tests pass. GUARDIAN opens Mar 18. All intelligence gathered (K-11/12/13, G-R7). Haslametrics scraper built. Next: OpenClaw O-9 sweep Mar 17; Claude Code V9.2 Apr 7+.
