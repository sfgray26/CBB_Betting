# OPERATIONAL HANDOFF (EMAC-066)

> Ground truth as of March 12, 2026. Operator: Claude Code (Master Architect).
> See `IDENTITY.md` for risk policy · `AGENTS.md` for roles · `HEARTBEAT.md` for loops.
> Full roadmap: `docs/MLB_FANTASY_ROADMAP.md` · CBB plan: `tasks/cbb_enhancement_plan.md`

---

## 0. STANDING DECISIONS

- **Gemini CLI is Research-Only.** No production code. Deliverables go to `docs/` as markdown.
- **All production code: Claude Code only.**
- **GUARDIAN (Mar 18 - Apr 7):** Do NOT touch `betting_model.py`, `analysis.py`, or CBB services during tournament window.

---

## 1. SYSTEM STATUS

### CBB Model — V9.1 TOURNAMENT-READY

| Component | Status |
|-----------|--------|
| Railway API | ✅ Healthy |
| PostgreSQL | ✅ Connected (365 teams) |
| Scheduler | ✅ 10 jobs running |
| Discord | ✅ 16 channels operational |
| V9.1 Model | ✅ Active (fatigue + sharp money + conf HCA + recency SD) |
| Test suite | ✅ 647/650 pass (3 pre-existing DB-auth failures) |

**All CBB enhancements complete:** Fatigue (K-8), OpenClaw Lite (K-9), Conference HCA (K-10/P2), Sharp Money (P1), Recency Weight (P3), Recalibration Audit (P4), Seed-Spread Scalars (A-26), Tournament SD bump, Integrity sweep.

### Fantasy Baseball — DRAFT-READY

| Component | Status | File |
|-----------|--------|------|
| Yahoo OAuth | ✅ Fixed + deployed | `yahoo_client.py` |
| Draft Board | ✅ Live (551 Steamer players) | `11_Fantasy_Baseball.py` |
| Live Draft Tracker | ✅ Built (manual + Yahoo polling) | `12_Live_Draft.py` |
| Draft Tracker backend | ✅ Built (26 tests) | `fantasy_baseball/draft_tracker.py` |
| Discord draft alerts | ✅ `send_draft_pick` + `send_on_the_clock_alert` | `discord_notifier.py` |
| Bet settlement fix | ✅ `_resolve_home_away()` fuzzy match (20 tests) | `bet_tracker.py` |
| Historical re-settlement | ✅ No discrepancies found | `scripts/resettle_bets.py` |
| Daily lineup optimizer | ✅ Built | `daily_lineup_optimizer.py` |

---

## 2. UPCOMING DEADLINES

| Date | Event | Owner | Action Required |
|------|-------|-------|----------------|
| **Mar 16 ~9 PM ET** | O-8 Baseline Execution | OpenClaw | `python scripts/openclaw_baseline.py --year 2026` |
| **Mar 17 ~7 PM ET** | O-9 Pre-tournament sweep | OpenClaw | See Section 4 |
| **Mar 18** | First Four begins | All | CBB monitoring mode |
| **Mar 20** | Fantasy Keeper Deadline | User | Set keepers in Yahoo UI |
| **Mar 23 7:30am ET** | Fantasy Draft Day | User | Run `12_Live_Draft.py`, begin polling |
| **Apr 7** | Tournament window closes | All | CBB guardian lifts |

---

## 3. ACTIVE MISSIONS

### Claude Code — One pending fix

EMAC-063 (Draft Board + Live Tracker) ✅ COMPLETE
EMAC-064 (Bet Settlement Fix) ✅ COMPLETE

**Pre-tournament fix (before Mar 18):**
- **Deduplication bug** — same game creating multiple prediction records (up to 8x). Fix: add app-level dedup in `backend/services/analysis.py` before inserting a Prediction — check for existing `(game_id, prediction_date)` row and update in place instead of creating new. See Section 4 for details.

**Next tasks (post-draft, after Mar 23):**
- Wire `daily_lineup_optimizer.py` into `yahoo_client.set_lineup()` for auto-submit
- Statcast integration via pybaseball for waiver wire rankings (`reports/ADVANCED_ANALYTICS_INTEGRATION.md`)

**Post-tournament (after Apr 7):**
- CBB recalibration with full tournament data
- Model V9.2 planning (EvanMiya re-evaluation)

### Gemini CLI — Research Missions (Pending)

| Mission | Task | Deliverable |
|---------|------|-------------|
| G-R1 | Steamer 2026 full download (750 batters, 450 pitchers) | `docs/PROJECTION_DATA_SOURCES.md` |
| G-R2 | Daily MLB lineup confirmation sources (by 7 AM ET) | `docs/LINEUP_CONFIRMATION_SOURCES.md` |
| G-R3 | Closer situations monitor (NSV category) | `docs/CLOSER_SITUATION_SOURCES.md` |
| G-R4 | Statcast bulk data via pybaseball | `docs/STATCAST_API_GUIDE.md` |
| G-R5 | Yahoo Fantasy API XML format for set_lineup + add/drop | `docs/YAHOO_API_REFERENCE.md` |
| G-16 | Verify O-10 line monitor post-deploy | Report to HANDOFF.md |

### OpenClaw — Mission O-9: Pre-Tournament Sweep

**Run on March 17, 2026 ~7 PM ET:**

```bash
# 1. Verify baseline
ls data/pre_tournament_baseline_2026.json
# If missing: python scripts/openclaw_baseline.py --year 2026

# 2. Verify Discord bot
python scripts/test_discord.py

# 3. Check odds monitor
# GET /admin/odds-monitor/status -- expect games_tracked > 0
```

For each First Four matchup: search "NCAA First Four 2026 injury lineup [team]" and run `check_integrity_heuristic()`. Flag any ABORT or VOLATILE results here under section 3.

---

## 4. KNOWN ISSUES

### 4.1 Deduplication Bug — HIGH PRIORITY (fix before Mar 18)

**Problem:** Same game creating multiple `Prediction` records per analysis run.

**Evidence (from Kimi audit, March 12):**
| Matchup | Duplicate Count |
|---------|----------------|
| Penn State @ Northwestern | 8 entries |
| Kansas St @ BYU | 6 entries |
| Missouri St @ FIU | 6 entries |
| Syracuse @ SMU | 4 entries |

**Root cause:** `get_or_create_game()` dedupes games by `external_id` correctly, but `Prediction` has no unique constraint on `(game_id, prediction_date)`. Each analysis run creates new rows.

**Fix:** In `backend/services/analysis.py` before prediction insert:
```python
existing = db.query(Prediction).filter_by(game_id=game.id, prediction_date=today).first()
if existing:
    # update in place instead of creating new
```

### 4.2 Paper/Real Bet Gap — LOW (workflow issue, not model)

167 paper trades vs 1 real bet. User confirmed betting outside the app. Options: manual entry via Bet Log page, DK direct import (already built), or accept paper tracking as system of record.

---

## 5. HANDOFF PROMPT — NEXT CLAUDE SESSION

```
CONTEXT (March 12, 2026):
- CBB V9.1 tournament-ready. 647/650 tests pass. Guardian window: Mar 18 - Apr 7.
- Fantasy draft is March 23 @ 7:30am. All draft tooling is COMPLETE.
- EMAC-063 (draft board + live tracker) COMPLETE.
- EMAC-064 (bet settlement fuzzy fix) COMPLETE.

ACTIVE TASK:
- Deduplication bug: fix prediction dedup in backend/services/analysis.py before Mar 18.
  Check for existing Prediction row (game_id + prediction_date) before inserting new.
  See HANDOFF Section 4.1 for full details.

POST-DRAFT TASKS (after March 23):
1. Wire daily_lineup_optimizer.py into yahoo_client.set_lineup() for auto-submit
2. Statcast integration via pybaseball (reports/ADVANCED_ANALYTICS_INTEGRATION.md)

POST-TOURNAMENT TASKS (after April 7):
1. CBB recalibration with full tournament data
2. Model V9.2 planning

GUARDIAN NOTES:
- Do NOT touch betting_model.py or CBB services during Mar 18-Apr 7
  Exception: dedup fix in analysis.py is safe (pure DB query guard, no model logic)
- Run `python -m pytest tests/ -q` before any commit
- Fantasy code is isolated -- safe to iterate anytime
```

---

## 6. QUICK REFERENCE

```bash
# Tests
python -m pytest tests/ -q

# Status checks
python scripts/preflight_check.py
python scripts/test_discord.py

# Railway
railway logs --follow
railway status

# Draft day
streamlit run dashboard/app.py  # navigate to 12_Live_Draft
```

---

## 7. HIVE WISDOM

| Lesson | Source |
|--------|--------|
| Conference HCA: Big Ten 3.6 pts vs SWAC 1.5 pts = significant road differential | P2 |
| Tournament mode: neutral HCA=0, margin SE +0.20, 14-day form window | P3 |
| Sharp money: steam >=1.5 pts in <30 min = high confidence signal | P1 |
| Opener gap >=2.0 pts = market correcting toward sharp opinion | P1 |
| Fatigue model adds 0.5-2.0 pt edge in B2B/altitude spots | K-8 |
| OpenClaw Lite: 26,000x faster than Ollama, 100% match rate | K-9 |
| BartTorvik public CSV needs no auth (cloudscraper only) | P0 |
| EvanMiya intentionally dropped -- 2-source mode robust by design | P0 |
| Bet settlement: use _resolve_home_away() -- never raw string compare | EMAC-064 |
| Yahoo roster pre-draft returns players:[] (empty array) -- handle gracefully | EMAC-063 |
| Prediction dedup: always check (game_id, prediction_date) before insert | EMAC-066 |
| Discord token must be in Railway Variables, not just .env | D-1 |
| Railway needs explicit railway.toml for reliable builds | Railway Fix |
| Nested f-strings with escaped quotes fail in Python < 3.12 | Python |
| Avoid non-ASCII chars in output strings (CP-1252 Windows terminal issue) | Python |

---

**Document Version:** EMAC-066
**Last Updated:** March 12, 2026
**Status:** CBB tournament-ready. Fantasy draft-ready. One pre-tournament fix pending (dedup bug). Guardian window opens Mar 18.
