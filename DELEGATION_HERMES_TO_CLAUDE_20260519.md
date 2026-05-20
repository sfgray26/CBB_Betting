# Delegation Bundle: Hermes → Claude Code

> **Date:** 2026-05-19
> **From:** Hermes (Session Orchestrator)
> **To:** Claude Code (Principal Architect)
> **Priority:** P1 — Deploy readiness + branch divergence

---

## ⚠️ CRITICAL FINDING: Branch Divergence

**Ground truth discovered during session startup:**

| Branch | Commit | Status |
|--------|--------|--------|
| `main` (local) | `4a6a3c0` | **HEAD** |
| `origin/stable/cbb-prod` | `4a6a3c0` | In sync with `main` |
| `stable/cbb-prod` (local) | `004972e` | **DIVERGED** — `[ahead 13, behind 4]` |

**What this means:**
- Local `stable/cbb-prod` has **13 commits** that are NOT on `origin/stable/cbb-prod` or `main`
- Local `stable/cbb-prod` is **missing 4 commits** that are on `main`/`origin/stable/cbb-prod`
- The deployed production code is likely on an even older commit (`d319beb` per stale HANDOFF.md, or possibly `873c709` per user brief)

** commits on local `stable/cbb-prod` but NOT on `main`:**
```
004972e fix(lint): resolve 4 flake8 F-violations breaking CI
8e331c4 fix(roster): add resilient count validation and logging to get_roster()
d0e7fd2 Merge origin/stable/cbb-prod: resolve test conflict, accept handedness fixture fix
f8341cc fix(frontend): JSX closing tag fix in roster page; roster view cleanup
6807d4a feat: Design System v3 (light theme), P0 scoreboard fix, game context wiring, test cleanup
d24243c fix(ui): convert dark zinc theme to light across all pages
873c709 fix(tests): resolve 4 failing tests post-handedness migration
aea55bd UI: Yahoo-style roster view + waiver swap bar
948cefd UI: Prominent ADD→DROP swap bar in waiver recommendations
ee05542 Fix 6 failed pipeline jobs (2026-05-18)
7da68b4 fix(projections): propagate live SB override to nsb key; fix utcnow cache comparison
0c1d1f4 fix(projections): propagate live SB override to nsb key; fix utcnow cache comparison
8286fdd feat(docs): add security audit and TODO registry documentation
```

**Commits on `main` but NOT on local `stable/cbb-prod`:**
```
4a6a3c0 fix: acquisitions dict-walk bug + MLB_OPENING_DATE_2026 inline define
92beb1c fix: clean suite on main — port stable/cbb-prod fixes + optimizer injury guard
0bea923 Merge remote-tracking branch 'origin/stable/cbb-prod'
5c0e2b9 Sync main with stable/cbb-prod (BDL #2 auto-heal)
```

---

## 📌 Mission: Resolve Divergence + Deploy Readiness

### Phase 1: Branch Hygiene (Do First)

1. **Inspect the 13 local-only commits on `stable/cbb-prod`**
   - Determine which contain production-critical fixes vs. stale/abandoned work
   - Use `git log stable/cbb-prod --not main --oneline` and inspect each
   - Check if any were already cherry-picked or superseded by `main` commits

2. **Decide merge strategy**
   - Option A: Reset local `stable/cbb-prod` to `origin/stable/cbb-prod` (=`main`) and cherry-pick any valuable commits
   - Option B: Merge `main` into `stable/cbb-prod`, resolve conflicts, force-push
   - Option C: Merge `stable/cbb-prod` into `main`, resolve conflicts, reset `stable/cbb-prod` to match
   - **Recommended:** Option A — `origin/stable/cbb-prod` already includes `main`'s 4 commits. The 13 local commits may be stale.

3. **Validate test suite after resolution**
   - `pytest tests/` must pass before any deploy
   - If tests fail, fix them before proceeding to Phase 2

---

### Phase 2: Remaining P1 Code Fixes (Post-Merge)

The following items from K-NEXT-4 UI UAT Audit still need **code changes** (not just deploy):

| # | Issue | File Hints | Owner |
|---|-------|------------|-------|
| 2 | **Dashboard waiver targets show `Need score: 0.00`** while Waiver Wire page shows real scores | `backend/services/dashboard_service.py` vs. `backend/services/waiver_edge_detector.py` | Pipeline divergence — unify scoring |
| 3 | **Roster "Move player" buttons universally disabled** | `frontend/app/(dashboard)/war-room/roster/page.tsx` | Check button `disabled` logic + API permissions |
| 4 | **Team totals show "–" for OPS and K/9** | `backend/services/dashboard_service.py` or aggregation layer | Missing rate-stat aggregation in team totals |
| 5 | **Budget page is extremely sparse** | `frontend/app/(dashboard)/war-room/budget/page.tsx` | Needs actual budget UI, not placeholder |
| 6 | **Garrett Crochet injury status is boolean `true`** | `backend/routers/fantasy.py` or `backend/schemas.py` | API returns bool instead of string `"IL"` |

**Note:** P1 item #1 (Ownership 0%) is already fixed in code (commits `425f9d6`, `27304f8`) — needs deploy only.

---

### Phase 3: Design System v2 Implementation (Optional / Post-P1)

Kimi's K-NEXT-5 spec is complete in `docs/DESIGN_SYSTEM_V2.md`.
- 4-step migration guide included
- NOT blocking for deploy
- Queue after P1s are resolved and system is stable

---

## 🧵 Files You Will Touch

### Phase 1 (Branch)
- Git operations only — no file edits
- Validate with `pytest tests/` after merge

### Phase 2 (P1 Fixes)
```
backend/services/dashboard_service.py          # Unify waiver scoring pipeline
backend/services/waiver_edge_detector.py       # ^^^
frontend/app/(dashboard)/war-room/roster/page.tsx    # Enable move buttons
backend/routers/fantasy.py                     # Injury status type fix
backend/schemas.py                             # ^^^
frontend/app/(dashboard)/war-room/budget/page.tsx    # Build budget UI
backend/services/team_aggregation.py (or equivalent) # OPS + K/9 totals
```

### Phase 3 (Design System)
```
frontend/styles/ (or tailwind.config.js)
frontend/components/ (category chips, battlefield rows, badges)
docs/DESIGN_SYSTEM_V2.md
```

---

## ✅ Validation Checklist

Before declaring Phase 2 complete:
- [ ] `pytest tests/` — all 2821 tests pass (or new baseline established)
- [ ] Dashboard waiver targets show non-zero need scores
- [ ] Roster "Move" buttons are clickable and trigger API call
- [ ] Team totals include OPS and K/9 (not "–")
- [ ] Budget page shows meaningful data (not 3 lines)
- [ ] Injury status returns string `"IL"` / `"ACTIVE"` / `"DTD"` (not boolean)
- [ ] `py_compile` clean on all modified `.py` files
- [ ] `npm run build` clean in `frontend/` (if TSX modified)

---

## 🚨 Risks & Gotchas

1. **Branch divergence is the highest risk.** The 13 local-only commits may contain fixes that production depends on. Do not discard without inspection.
2. **The 4 commits on `main` include an acquisitions dict-walk bugfix.** If `stable/cbb-prod` local was branched before this, production may have the bug.
3. **Injury status boolean vs. string** — changing the API return type may break the frontend that expects boolean. Coordinate with frontend type definitions.
4. **Budget page** — may require backend API expansion (not just frontend). Check what `/api/fantasy/budget` currently returns.
5. **Do NOT deploy until test suite passes.** The `92beb1c` commit mentions "clean suite on main" — verify this claim.

---

## 📤 Escalation

- **If Railway deploy needed after merge** → Route to Gemini CLI with deploy bundle
- **If UI research / spec refinement needed** → Route to Kimi CLI
- **If audit results need logging** → Route to Hermes

---

## Verbatim Prompt (Copy-Paste Ready for Claude Code)

```
You are Claude Code, Principal Architect for cbb-edge.

GROUND TRUTH:
- Local main is at 4a6a3c0. Local stable/cbb-prod is at 004972e and is [ahead 13, behind 4] relative to origin/stable/cbb-prod (which is at 4a6a3c0, same as main).
- There is branch divergence. Local stable/cbb-prod has 13 commits not on main. Main has 4 commits not on local stable/cbb-prod.
- Production deploy status is unclear — last known deploy was d319beb or 873c709.

YOUR MISSION (in order):
1. Inspect the 13 commits on local stable/cbb-prod that are not on main. Determine which are valuable vs. stale. Use git log stable/cbb-prod --not main --oneline.
2. Resolve the divergence. Recommended: reset local stable/cbb-prod to origin/stable/cbb-prod and cherry-pick any valuable commits. Alternative: merge strategy — your call, but justify it.
3. After branch is clean, run pytest tests/. All tests must pass.
4. Fix the 5 remaining P1 code issues from K-NEXT-4 UI UAT:
   a. Dashboard waiver targets show Need score: 0.00 (unify pipeline with waiver wire page)
   b. Roster "Move player" buttons disabled
   c. Team totals missing OPS and K/9
   d. Budget page sparse / placeholder
   e. Injury status returns boolean true instead of string "IL"
5. Update HANDOFF.md with what you did and what's still unresolved.

CONSTRAINTS:
- Do NOT deploy to Railway — that is Gemini CLI's job.
- Do NOT modify betting_model.py — CBB model is frozen.
- Follow AGENTS.md code quality gates: py_compile, pytest, no utcnow(), no bool-as-string leakage.

REPORT BACK TO HERMES when done.
```

---

*Prepared by Hermes, 2026-05-19*
*Source: HANDOFF.md (2026-05-13), HEARTBEAT.md, git branch inspection, user brief*
