# Expert Audit: Dashboard & War Room Data Quality

> **Auditor:** Claude Code (Master Architect)  
> **Date:** 2026-05-12  
> **Scope:** `/api/fantasy/waiver`, `/api/fantasy/budget`, `/api/fantasy/matchup/simulate`, and corresponding UI surfaces  
> **Method:** Backend code review, database inspection (production PostgreSQL), frontend component analysis

---

## Executive Summary

The Dashboard and War Room are **structurally sound but suffering from five critical data bugs** that render the waiver intelligence, budget tracking, and simulation features effectively useless for an elite manager. These are not "tuning" issues — they are key-mismatch bugs, missing API subresources, and wrong dictionary lookups that produce zeros and false negatives at scale.

| Issue | Severity | Root Cause | User Impact |
|-------|----------|------------|-------------|
| Need score = 0.00 for all players | **P0** | Category key mismatch between `team_needs` (canonical) and `fa_impact` (board keys) | No actionable waiver ranking |
| Ownership = 0% for all players | **P0** | `get_free_agents()` never requests `out=ownership` from Yahoo | Elite manager can't gauge FAAB urgency or waiver priority |
| Budget IP = 0 | **P0** | Budget endpoint looks for `"my_team"`; `get_matchup_stats()` returns `"my_stats"` | Pitching pace flag is always wrong |
| Budget acquisitions inaccurate | **P1** | Rolling 7-day window instead of matchup week; fragile timestamp parsing | Manager burns add/drop budget blindly |
| Simulation shows "behind" when leading | **P1** | Empty `cat_scores` for roster players → MCMC samples noise, not skill | False panic / false confidence |
| UI color clash & missing context | **P2** | War Room uses `bg-black` + `#FFC000`; Dashboard uses `bg-zinc-950` + amber; no stale-data indicators | Looks unprofessional; erodes trust |

---

## 1. Waiver Targets — Ownership & Need Score (P0)

### 1.1 Ownership: Always 0%

**Code:** `backend/fantasy_baseball/yahoo_client_resilient.py:776-812`

```python
def get_free_agents(self, position: str = "", start: int = 0, count: int = 25) -> list[dict]:
    params = {"status": "A", "start": start, "count": count, "sort": "AR"}
    ...
    data = self._get(f"league/{self.league_key}/players", params=params)
```

The endpoint **does not request `out=ownership`**. A comment claims this causes Yahoo 400, but the result is that Yahoo returns no ownership block in the response. `_parse_player()` recursively searches for ownership and falls back to `0.0` when absent.

**Fix:** Re-test `out=ownership` on the `league/{league_key}/players` endpoint. If Yahoo still 400s, fetch ownership via a secondary batch call (the `players;player_keys=.../ownership` subresource is valid on the **players** collection, just not on the **league/players** collection).

### 1.2 Need Score: Always 0.00

This is a **key-mismatch bug** in the scoring pipeline.

**Step 1 — Category deficits use canonical codes (uppercase → lowercased):**

`backend/routers/fantasy.py:1788-1805`
```python
sid_map[_sid_w] = _final   # e.g. "12" → "HR_B"
...
category_deficits.append(CategoryDeficitOut(category=cat, ...))  # cat = "HR_B"
```

**Step 2 — `compute_need_score` lowercases the canonical code:**

`backend/fantasy_baseball/category_aware_scorer.py:208`
```python
needs_dict[cd.category.lower()] = float(cd.deficit)   # → "hr_b"
```

**Step 3 — Player cat_scores use BOARD keys (from `player_projections`):**

`backend/routers/fantasy.py:167-173`
```python
_CANONICAL_TO_BOARD: dict = {
    "HR_B":  "hr",       # ← canonical → board
    "HR_P":  "hr_pit",
    "K_9":   "k9",
    "K_P":   "k_pit",
    "K_B":   "k_bat",
}
```

The DB stores cat_scores with board keys (`"hr"`, `"k_bat"`, `"k_pit"`, etc.).

**Step 4 — `score_fa_against_needs` looks up by the needs key:**

`backend/fantasy_baseball/category_aware_scorer.py:99`
```python
player_z = fa_impact.impacts.get(cat, 0.0)   # cat = "hr_b", impacts has "hr" → 0.0
```

**Result:** Every category contribution is `0.0`. The blended score collapses to `0.4 * player_z_score`. For players without a DB projection (see §1.3), `player_z_score = 0.0`, so **need_score = 0.00 for everyone.**

**Fix:** Normalize board keys to canonical keys BEFORE calling `compute_need_score`, OR normalize inside `score_fa_against_needs` using the inverse of `_CANONICAL_TO_BOARD`.

### 1.3 "Always Just Pitchers"

When all need scores are `0.0`, the waiver endpoint's `sort(key=lambda x: x.need_score, reverse=True)` is a no-op. The list stays in Yahoo's `sort=AR` (percent rostered) order. Pitchers dominate free-agent pools in fantasy baseball, so the surface appears to be "all pitchers."

**Database context:** `player_id_mapping` has **only 2,418 Yahoo IDs** out of 10,928 total rows. That means ~78% of Yahoo players cannot be resolved to a DB projection via ID lookup. The name-resolution fallback (`PlayerIdentity` → `PlayerProjection` by name) helps, but Unicode handling and suffix stripping (`Jr.`, `III`) create additional mismatches.

---

## 2. Constraint Budget — IP & Acquisitions (P0/P1)

### 2.1 IP Accumulated = 0 (Wrong Dict Key)

**Code:** `backend/routers/fantasy.py:5517-5520`
```python
matchup_stats = client.get_matchup_stats(week=current_week, my_team_key=team_key)
if matchup_stats:
    my_stats = matchup_stats.get("my_team", {})          # ← WRONG KEY
    ip_accumulated = float(my_stats.get("IP", 0.0))
```

**`get_matchup_stats` returns:**
```python
return {"my_stats": {}, "opp_stats": {}, "opponent_name": "Unknown"}
```

The key is `"my_stats"`, not `"my_team"`. `my_stats` is always `{}`, so `IP` is always `0.0`.

**Fix:** Change `"my_team"` → `"my_stats"`.

### 2.2 Acquisitions Not Reflecting Reality

**Code:** `backend/routers/fantasy.py:5498-5507`
```python
transactions = client.get_transactions(t_type="add")
week_start = now_et - timedelta(days=7)   # ← rolling 7 days, NOT matchup week
week_end = now_et
acquisitions_used = count_weekly_acquisitions(transactions, team_key, week_start, week_end)
```

Three problems:
1. **Rolling 7-day window** instead of the actual Yahoo matchup week (Mon–Sun or custom league setting). A manager who added 2 players on Monday sees them counted; if they added on Saturday of last week, those drop off the count by Wednesday.
2. **Exception handling is too narrow.** `count_weekly_acquisitions` compares `float <= timestamp <= float`. If Yahoo returns `timestamp` as an ISO string (which it does in some response shapes), a `TypeError` is raised. The budget endpoint only catches `YahooAuthError` / `YahooAPIError`, **not** `TypeError`. The entire budget call would 500. If it's working, Yahoo is returning numeric timestamps — but this is fragile.
3. **Yahoo's `get_transactions(t_type="add")` may return adds with `type="add/drop"`** rather than `"add"`. The filter `txn.get("type") != "add"` would skip these.

**Fix:**
- Use Yahoo's league settings to compute the actual matchup week start/end.
- Broaden exception handling in the budget endpoint to `Exception` (or at least add `TypeError`).
- Accept `"add/drop"` as a valid transaction type for counting.

---

## 3. War Room Simulation — "Behind When Leading" (P1)

### 3.1 Empty cat_scores → Noise-Driven Simulation

**Code:** `backend/routers/fantasy.py:4379-4397` (`_fetch_rosters_for_simulate`)

```python
def _player_dict(p: dict) -> dict:
    cat_scores: dict[str, float] = {}
    proj = _proj_by_name.get(_normalize_identity_name(name))
    if proj and proj.cat_scores:
        for src, dest in _PROJ_TO_SIM_KEY.items():
            val = proj.cat_scores.get(src)
            if val is not None:
                cat_scores[dest] = float(val)
    return {
        "name": name,
        "positions": positions,
        "cat_scores": cat_scores,   # ← empty for many players
        "starts_this_week": 1 if is_pitcher else 0,
    }
```

The MCMC simulator (`backend/fantasy_baseball/mcmc_simulator.py`) draws from `Normal(mean=cat_score, std=position_std)` for each player. When `cat_scores` is empty, every player has `mean = 0.0`. The simulation becomes a pure noise contest.

**Why "behind when leading" happens:**
- Current stats (from Yahoo scoreboard) show the real matchup standings.
- The simulation ignores current stats and re-simulates from roster projections.
- If one roster has more empty projections than the other, the noise is asymmetric.
- A category where the user is currently ahead (e.g., by 5 HR) can show `win_prob < 0.35` (BEHIND) because the simulator sees the opponent's roster as having more non-zero projections, or simply because 1,000 noise draws happened to favor the opponent.

**This is not a quirk — it is a data-quality failure.** An elite manager should see simulation results that are anchored to the current scoreboard state plus projected ROS contributions, not a coin-flip between two teams of zeros.

**Fix:**
1. **Anchor simulation to current stats.** The simulator should start from the Yahoo scoreboard totals and add projected weekly contributions, not simulate from scratch.
2. **Improve projection coverage.** The Yahoo ID sync needs to push well above 2,418 mapped IDs. A name-based fuzzy match against `player_projections` (not just `PlayerIdentity`) would close the gap.
3. **Add a data-quality gate.** If >30% of roster players have empty cat_scores, the simulation should return a `data_quality: "degraded"` flag and the UI should show a warning banner.

---

## 4. UI / UX Issues (P2)

### 4.1 Color Scheme Clash

| Surface | Background | Accent | Issue |
|---------|-----------|--------|-------|
| Dashboard | `bg-zinc-950` | amber-400 / zinc-100 | Warm gray family |
| War Room | `bg-black` | `#FFC000` (hex gold) | Harsh contrast, different gold |
| Streaming Station | `bg-black` | `#FFC000` / `#7D7D7D` | Same as War Room, but labels use `#494949` which is nearly invisible |

**Recommendation:** Unify on `bg-zinc-950` everywhere. Use a single gold token (`amber-400` or `#FFC000`, not both). Increase label contrast — `#494949` on `#1A1A1A` fails WCAG.

### 4.2 Missing Context & Stale-Data Indicators

- **0% owned** is displayed as a literal `0%`. It should read `—` or `N/A` with a tooltip explaining that Yahoo ownership data is unavailable.
- **Need score 0.0** is displayed without any indicator that the projection is missing. A `?` icon or "projection missing" badge would signal data quality.
- **Category deficits** on the Streaming Station say `"(Negative = behind league average)"`. This is wrong — the deficit is vs the **opponent**, not the league average. The copy is confusing.
- **Simulation status tags** (SAFE, LEAD, BUBBLE, BEHIND, LOST) have no explanation of what drives them. An elite manager wants to know: "Why does the simulation think I'll lose ERA when I'm currently ahead?" The UI should show the projected final tally (which exists in the API as `my_proj` / `opp_proj`) in a tooltip or expand row.

---

## 5. Prioritized Fix Plan

### Phase 1: Stop the Bleeding (P0 bugs — this session)

| # | File | Change | Expected Result |
|---|------|--------|-----------------|
| 1 | `backend/routers/fantasy.py:5519` | `"my_team"` → `"my_stats"` | Budget IP reads live Yahoo data |
| 2 | `backend/routers/fantasy.py` | Normalize `cat_scores` keys via `_CANONICAL_TO_BOARD` before passing to `compute_need_score` | Need scores become non-zero and matchup-aware |
| 3 | `backend/fantasy_baseball/yahoo_client_resilient.py:776-812` | Add `out=ownership` to `get_free_agents` or fetch ownership in a secondary batch call | Ownership % populates |
| 4 | `backend/routers/fantasy.py:5498-5507` | Use matchup week boundaries (not rolling 7d); catch `Exception` around `count_weekly_acquisitions`; accept `"add/drop"` type | Budget acquisitions reflect reality |

### Phase 2: Simulation Integrity (P1 bugs — next session)

| # | File | Change | Expected Result |
|---|------|--------|-----------------|
| 5 | `backend/routers/fantasy.py:4309-4417` | Fuzzy-match roster players directly against `player_projections.player_name` when identity resolution fails | More roster players get non-zero cat_scores |
| 6 | `backend/fantasy_baseball/mcmc_simulator.py` | Accept `current_stats` anchor and add projected weekly deltas instead of simulating from zero | Simulation respects current scoreboard lead |
| 7 | `backend/fantasy_baseball/mcmc_simulator.py` | Add `data_quality` flag when >30% roster has empty cat_scores | UI can warn user that simulation is degraded |

### Phase 3: UI Polish (P2 — can be delegated to Kimi)

| # | File | Change | Expected Result |
|---|------|--------|-----------------|
| 8 | `frontend/app/(dashboard)/war-room/**` | Unify background to `bg-zinc-950`; single gold token | Visual consistency |
| 9 | `frontend/app/(dashboard)/war-room/streaming/page.tsx` | Fix deficit copy: "behind opponent" not "behind league average" | Clearer context |
| 10 | `frontend/components/war-room/category-battlefield.tsx` | Add tooltips to status tags showing `my_proj` vs `opp_proj` | Elite manager understands WHY simulation says BEHIND |
| 11 | `frontend/app/(dashboard)/war-room/streaming/page.tsx` | Show `—` instead of `0%` when ownership is missing; add `title` tooltip | No false 0% signals |

---

## Appendix: Database Truth

```
player_id_mapping        → 10,928 rows | 2,418 with yahoo_id  (22% coverage)
player_projections       → 9,728 rows  | 9,728 with cat_scores (100% coverage)
player_identities        → 7,009 rows
mlb_player_stats         → 17,638 rows
player_scores            → 101,712 rows
```

The projection pipeline is healthy. The problem is **resolution** (Yahoo player → DB row), not **computation**.

---

*Next step: Implement Phase 1 fixes. Each fix is bounded and testable.*
