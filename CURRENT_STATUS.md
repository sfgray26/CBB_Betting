# CURRENT STATUS — CBB Edge

> **Single source of truth for all agents.** Updated every session by Claude Code.
> For full mission context: read `HANDOFF.md`. For swimlane rules: read `ORCHESTRATION.md`.
> For risk policy: read `IDENTITY.md`.

**Last updated:** March 19, 2026 — R64 Day 1
**Document:** EMAC-072

---

## 🚦 System Health

| Subsystem | Status | Last Verified |
|-----------|--------|---------------|
| Railway API (backend) | ✅ Healthy | Mar 19 |
| PostgreSQL (365 teams) | ✅ Connected | Mar 19 |
| V9.1 Model | ✅ Active | Mar 19 |
| Discord (morning brief 7AM / EOD 11PM) | ✅ Working | Mar 16 |
| Odds Monitor (every 5 min) | ✅ Running | Mar 19 |
| Next.js Frontend (:3000) | ✅ All 9 pages live | Mar 19 |
| Streamlit Dashboard (:8501) | ✅ 13 pages (legacy) | Mar 16 |
| Test Suite | ✅ 683/686 pass | Mar 13 |
| CI/CD (Railway auto-deploy) | ✅ Push-to-main | Mar 16 |

---

## 🔒 Guardian Lock

**ACTIVE until April 7, 2026 (NCAA Championship Day)**

Do NOT modify:
- `backend/betting_model.py`
- `backend/services/analysis.py`
- Any file in `backend/services/`

Guardian task file: `.agent_tasks/v9_2_recalibration.md`

---

## 📋 Frontend Status (Next.js)

**Branch:** `claude/fix-clv-null-safety-fPcKB`

| Phase | Status | Pages | Validated |
|-------|--------|-------|-----------|
| 0 — Foundation | ✅ DONE | scaffold, auth, layout, design system | — |
| 1 — Core Analytics | ✅ DONE | /performance, /clv, /bet-history, /calibration, /alerts | OpenClaw ✅ |
| 2 — Trading | ✅ DONE | /today, /live-slate, /odds-monitor | Script ✅ |
| 3 — Tournament | ✅ DONE | /bracket (10k MC sims) | Script ✅ |
| 4 — Mobile & PWA | ⏳ OPEN | viewport, sidebar drawer, touch targets, manifest | — |
| 5 — Polish | ⏳ OPEN | error boundaries, TS strict, Streamlit retire | — |

**Validator:** `./scripts/validate_frontend.sh` — 10/10 PASS (5 warnings)

---

## 🤖 Agent Task Pool

See `.agent_tasks/` for full task files with success criteria.

| Task | File | Status | Agent |
|------|------|--------|-------|
| Phase 4 Mobile/PWA | `.agent_tasks/phase_4_mobile_pwa.md` | OPEN | Claude Code (after Kimi spec) |
| Phase 5 Polish | `.agent_tasks/phase_5_polish.md` | OPEN | Claude Code |
| V9.2 Recalibration | `.agent_tasks/v9_2_recalibration.md` | **GUARDIAN** until Apr 7 | Claude Code |
| Oracle Validation | `.agent_tasks/oracle_validation.md` | OPEN (post-tournament) | Kimi + Claude |

**Done tasks:** `.agent_tasks/done/` (Phase 1, 2, 3 complete)

---

## 🛠️ Agent Harness (New — Mar 19)

Implemented from Anthropic C Compiler best practices research (Kimi CLI, Mar 19):

| Tool | Location | Purpose |
|------|----------|---------|
| **Frontend Validator** | `scripts/validate_frontend.sh` | 7-point null-safety + decimal check, exits 1 on blocking issues |
| **Task Pool** | `.agent_tasks/` | Git-based task locking, 1-3 session tasks, claim/done lifecycle |
| **Current Status** | `CURRENT_STATUS.md` (this file) | Session-start orientation for all agents |
| **Done Archive** | `.agent_tasks/done/` | Completed tasks with lessons |
| **CI (upcoming)** | `.github/workflows/deploy.yml` | TypeScript check job on every PR |

**Key principle from research:** "The harness around the agent matters more than the agent itself."

---

## 📁 Key File Locations

```
backend/betting_model.py        # V9.1 model — GUARDIAN LOCKED
backend/services/bracket_simulator.py  # MC bracket sim (5k sims, 521 lines)
backend/tournament/             # Tournament-specific: fetch_odds, matchup_predictor, bracket_sim
data/bracket_2026.json          # 64-team bracket (market_ml: null → needs odds fetch)
frontend/app/(dashboard)/       # All 9 Next.js pages
frontend/lib/types.ts           # All TypeScript interfaces (source of truth)
frontend/lib/api.ts             # All API endpoint wrappers
reports/api_ground_truth.md     # Kimi-produced API shapes (every endpoint documented)
reports/K12_RECALIBRATION_SPEC_V92.md  # V9.2 params justification
scripts/validate_frontend.sh    # Frontend safety validator
.agent_tasks/                   # Task pool
```

---

## 🧠 Quick Lessons (Hive Wisdom)

| Lesson | Source |
|--------|--------|
| All API decimal fields (roi, win_rate, clv, edge) need × 100 for display | K-11 + Phase 1 |
| Empty API response: `{ message: "...", total_bets: 0 }` — no `overall` key | Phase 1 |
| `parseISO(null)` crashes — always guard before date parsing | Phase 1 fix |
| `predictions ?? []` — always fallback on API array fields before `.map()` | Phase 2 |
| `staleTime: 10_min` on expensive MC endpoints (bracket sim) | Phase 3 |
| `?.toFixed()` optional chaining even inside null-checked block — cleaner | Phase 2 fix |
| `validate_frontend.sh` excludes `!= null` on same line — don't rely on that | Mar 19 |
| Oracle validation: compare our spread vs KenPom+BT consensus, flag if >5pt diff | K-15 spec TBD |

---

## ⏰ Upcoming Deadlines

| Date | Event | Action |
|------|-------|--------|
| **Mar 19 (Today)** | R64 Day 1 — first tip-off ~12:15 PM ET | Monitor; model should auto-pick |
| **Mar 20-21** | R64 continues | Monitor CLV; check Discord morning briefs |
| **Mar 23** | Fantasy Draft Day | Run `12_Live_Draft.py` |
| **Apr 7** | Guardian lifts + Championship Day | Execute V9.2 + Haslametrics |
| **Apr 7+** | Oracle validation + Fantasy Baseball | After Guardian |
