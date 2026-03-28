# Dashboard Service Mock / Fallback Audit

**File audited:** `backend/services/dashboard_service.py`
**Audited by:** Claude Code (Master Architect)
**Date:** 2026-03-27
**Scope:** Every location where mock data, hardcoded fallback values, empty lists, or
placeholder returns substitute for real database queries or live service calls.

---

## Summary

| # | Method | Line(s) | Fallback Type | Severity |
|---|--------|---------|---------------|----------|
| 1 | `_get_lineup_gaps` | 260, 312 | Empty list + hardcoded slot count | HIGH |
| 2 | `_get_lineup_gaps` | 264 | Roster silently becomes `[]` on missing `team_key` | HIGH |
| 3 | `_get_lineup_gaps` | 275 | Hardcoded `required_positions` list | MEDIUM |
| 4 | `_get_lineup_gaps` | 303 | `suggested_add=None` unconditionally | MEDIUM |
| 5 | `_get_streaks` | 339-340 | Early-exit empty lists when Yahoo unavailable | HIGH |
| 6 | `_get_streaks` | 396 | `team` field populated with `player_name` (wrong field) | MEDIUM |
| 7 | `_get_waiver_targets` | 429-434 | Full stub — always returns `[]` | CRITICAL |
| 8 | `_get_injury_flags` | 444, 503-504 | Empty list + zero counts on Yahoo failure | HIGH |
| 9 | `_get_injury_flags` | 493 | `estimated_return=None` unconditionally | MEDIUM |
| 10 | `_get_matchup_preview` | 514-517 | Full stub — always returns `None` | CRITICAL |
| 11 | `_get_probable_pitchers` | 526-532 | Full stub — always returns `([], [])` | CRITICAL |

---

## Detailed Entries

---

### 1. `_get_lineup_gaps` — Yahoo unavailable hard fallback
**Lines:** 258-260, 309-312

**Current return (mock shape):**
```python
return [], 0, 9   # line 260: Yahoo client missing
return [], 0, 9   # line 312: exception during roster fetch
```
`9` is a magic number representing the assumed total slot count for a Yahoo H2H league. It is not read from any configuration, user preference, or DB row.

**What it should query:**
`UserPreferences` table — `dashboard_layout` JSON column already stores panel config; league slot count should either be stored in `user_preferences.waiver_preferences` or in a new `league_config` column. Alternatively, `FantasyLineup` rows (`backend/models.py::FantasyLineup`) have the `positions` JSON dict whose key count gives the actual slot total for that user.

**Filter conditions:**
`UserPreferences.user_id == user_id`

**Expected return shape:**
`tuple[List[LineupGap], int, int]`
- `List[LineupGap]` — gaps detected (may still be empty if no gap)
- `int filled` — populated slots
- `int total` — derived from league settings, not hardcoded

---

### 2. `_get_lineup_gaps` — `team_key=None` silently produces empty roster
**Lines:** 264

**Current return (mock shape):**
```python
roster = client.get_roster(team_key) if team_key else []
```
When `team_key` is `None` (the common path from `get_dashboard`'s optional argument), `roster` is immediately set to `[]`. The rest of the gap-detection logic is then a no-op.

**What it should query:**
`YahooFantasyClient.get_roster()` (no argument form) retrieves the authenticated user's default team. The call at line 335 (inside `_get_streaks`) already uses this pattern correctly: `client.get_roster()`. The same pattern should be applied here.

**Filter conditions:**
None — `get_roster()` resolves the team from the active OAuth session.

**Expected return shape:**
Same as entry 1: `tuple[List[LineupGap], int, int]`

---

### 3. `_get_lineup_gaps` — Hardcoded `required_positions`
**Lines:** 275-276

**Current return (mock shape):**
```python
required_positions = ["C", "1B", "2B", "3B", "SS", "OF", "OF", "OF", "Util"]
```
This assumes every user is in a standard 9-slot H2H league with exactly this slot configuration. It ignores the actual league roster settings.

**What it should query:**
`UserPreferences` (`user_preferences` table) — `waiver_preferences` JSONB column could carry a `league_roster_slots` list, or a dedicated column should store the active league's slot definition. As a richer alternative, `YahooFantasyClient` exposes a `get_league_settings()` method that returns the actual slot structure.

**Filter conditions:**
`UserPreferences.user_id == user_id`

**Expected return shape:**
`List[str]` — ordered position slot labels exactly as the league defines them (variable length).

---

### 4. `_get_lineup_gaps` — `suggested_add` always `None`
**Lines:** 302-305

**Current return (mock shape):**
```python
suggested_add=None  # Would need waiver wire analysis
```
Every `LineupGap` object is emitted with no suggestion, making the gap alert incomplete for the UI consumer.

**What it should query:**
`WaiverEdgeDetector.get_top_moves(my_roster, opponent_roster)` already exists and returns a ranked list of add/drop pairs with `add_player.name`. The first move whose `add_player.positions` contains the gap position is the correct `suggested_add` value.

**Filter conditions:**
`move["add_player"]["positions"]` must include the unfilled `req_pos`.

**Expected return shape:**
`str` — player name of the highest-ranked eligible FA for that slot, or `None` only when the waiver wire is also exhausted.

---

### 5. `_get_streaks` — Early exit when Yahoo is unavailable
**Lines:** 331-340

**Current return (mock shape):**
```python
if not roster:
    return [], []
```
When Yahoo auth is down, the method exits immediately. However, streak data lives entirely in `PlayerDailyMetric` (DB), not in Yahoo. The Yahoo dependency here is used only to scope the query to the user's rostered players — but that roster list could be cached or substituted.

**What it should query:**
`PlayerDailyMetric` (`player_daily_metrics` table) filtered to `sport == "mlb"` and `metric_date >= today - 30 days`. When Yahoo is unavailable, the full unfiltered set can still be surfaced (all players with recent metrics), returning streaks league-wide rather than roster-scoped.

**Filter conditions (primary path):**
```
PlayerDailyMetric.sport == "mlb"
PlayerDailyMetric.metric_date >= date.today() - timedelta(days=30)
PlayerDailyMetric.player_id.in_([p["player_id"] for p in roster])
```

**Filter conditions (Yahoo-unavailable fallback):**
Same as above but without the `.in_()` clause — return top-N by `z_score_recent` across all MLB players.

**Expected return shape:**
`tuple[List[StreakPlayer], List[StreakPlayer]]` — hot list sorted descending by `z_score_recent`, cold list sorted ascending.

---

### 6. `_get_streaks` — `team` field populated with `player_name`
**Line:** 396

**Current return (mock shape):**
```python
team=latest.player_name,  # Would need to get from roster
```
The `StreakPlayer.team` field is intentionally set to the player name string, which is incorrect data. The UI consumer expects a team abbreviation (e.g., `"NYY"`, `"LAD"`).

**What it should query:**
`StatcastPerformance` (`statcast_performances` table) — `team` column (String(10)) holds the correct MLB team abbreviation. Alternatively, the Yahoo `roster` list (already in scope at line 352) carries `player.get("team")` per player dict; this should be used at roster-match time rather than ignored.

**Filter conditions:**
Join on `player_name` or `player_id` between `PlayerDailyMetric` and `StatcastPerformance`, or use `player.get("team")` from the already-fetched roster dict.

**Expected return shape:**
`str` — MLB team abbreviation, e.g. `"NYY"`. The `StreakPlayer` dataclass `team` field should carry this value.

---

### 7. `_get_waiver_targets` — Complete stub (CRITICAL)
**Lines:** 421-434

**Current return (mock shape):**
```python
targets = []
# TODO: Integrate with WaiverEdgeDetector
return targets
```
This is a complete no-op. The method always returns an empty list. Panel B1.3 (Waiver Targets) on the dashboard will always be blank.

**What it should call:**
`WaiverEdgeDetector.get_top_moves(my_roster, opponent_roster, n_candidates=10)` — this method already exists and is fully implemented in `backend/services/waiver_edge_detector.py`. It returns a list of dicts with `add_player`, `drop_player_name`, `need_score`, `win_prob_gain`, and `win_prob_before`/`after`.

The result must then be mapped to `List[WaiverTarget]` dataclass instances using:
- `add_player["player_id"]` → `WaiverTarget.player_id`
- `add_player["name"]` → `WaiverTarget.name`
- `add_player["team"]` → `WaiverTarget.team`
- `add_player["positions"]` → `WaiverTarget.positions`
- `add_player.get("percent_owned", 0.0)` → `WaiverTarget.percent_owned`
- `move["need_score"]` → `WaiverTarget.priority_score`
- Tier classification from `need_score` thresholds → `WaiverTarget.tier`
- `move["win_prob_gain"]` description → `WaiverTarget.reason`

User `waiver_preferences` filters (`min_percent_owned`, `max_percent_owned`, `hide_injured`) from `UserPreferences` should gate the final list.

**Filter conditions:**
`prefs.waiver_preferences["min_percent_owned"]` <= `percent_owned` <= `prefs.waiver_preferences["max_percent_owned"]`

**Expected return shape:**
`List[WaiverTarget]` sorted by `priority_score` descending; top 5 consumed by `get_dashboard`.

---

### 8. `_get_injury_flags` — Yahoo unavailable hard fallback
**Lines:** 443-444, 501-504

**Current return (mock shape):**
```python
return [], 0, 0   # line 444: no Yahoo client
return [], 0, 0   # line 503: exception path
```
Zeros for both `healthy` and `injured` counts are incorrect — the dashboard will display "0 healthy, 0 injured" rather than a degraded-mode indicator.

**What it should query:**
`PatternDetectionAlert` (`pattern_detection_alerts` table) — `is_active == True`, `pattern_type` contains injury signals. As a richer fallback, Yahoo injury data can be cached in `PlayerProjection` rows; the `StatcastPerformance` table does not carry injury status.

As a minimum degraded-mode response: query `PlayerProjection` (`player_projections` table) for all players whose `player_id` matches the last-known roster (from any available cache), and report counts based on the most recent `DataIngestionLog` status.

**Filter conditions (degraded mode):**
```
PatternDetectionAlert.is_active == True
PatternDetectionAlert.game_date >= date.today()
```

**Expected return shape:**
`tuple[List[InjuryFlag], int, int]` — flags list, healthy int, injured int. On full Yahoo failure, the list may be empty but counts should reflect "unknown" state, not zeroes.

---

### 9. `_get_injury_flags` — `estimated_return` always `None`
**Line:** 493

**Current return (mock shape):**
```python
estimated_return=None,  # Would need additional data source
```
The `InjuryFlag.estimated_return` field is unconditionally `None` for all injury records.

**What it should query:**
`PatternDetectionAlert` (`pattern_detection_alerts` table) — `description` and `resolution_notes` columns may carry return-timeline text from OpenClaw's pattern detection. Alternatively, Yahoo's `injury_note` field (already accessed at line 491 as `player.get("injury_note")`) sometimes includes return date text and should be parsed.

**Filter conditions:**
`PatternDetectionAlert.player_id == player["player_id"]` and `PatternDetectionAlert.is_active == True`

**Expected return shape:**
`Optional[str]` — ISO date string `"YYYY-MM-DD"` or human-readable text like `"~2 weeks"`. `None` is acceptable only when genuinely unavailable, not as a structural default.

---

### 10. `_get_matchup_preview` — Complete stub (CRITICAL)
**Lines:** 506-517

**Current return (mock shape):**
```python
# TODO: Integrate with Yahoo scoreboard API
# Use MCMC simulator for projections
return None
```
Panel B1.5 (Matchup Preview) will always be blank. `DashboardData.matchup_preview` will always be `None`.

**What it should call:**
`YahooFantasyClient.get_matchup()` (or equivalent scoreboard call) to retrieve the current week's opponent, projected category scores, and win probability. `WaiverEdgeDetector.mcmc` (if injected) provides win probability through `simulate_roster_move`.

**DB tables involved (supplementary):**
- `PlayerProjection` (`player_projections`) — projected category z-scores in `cat_scores` JSONB
- `PlayerDailyMetric` (`player_daily_metrics`) — recent performance for projection blending
- `PerformanceSnapshot` (`performance_snapshots`) — opponent's historical category averages

**Filter conditions:**
```
PlayerProjection.player_id.in_([rostered_player_ids])  # my team
PlayerProjection.player_id.in_([opponent_player_ids])  # opponent team
```

**Expected return shape:**
`Optional[MatchupPreview]` — fully populated dataclass with `week_number`, `opponent_team_name`, `opponent_record`, `my_projected_categories` (dict of cat→float), `opponent_projected_categories`, `win_probability` (float 0–1), `category_advantages` (List[str]), `category_disadvantages` (List[str]).

---

### 11. `_get_probable_pitchers` — Complete stub (CRITICAL)
**Lines:** 519-532

**Current return (mock shape):**
```python
pitchers = []
two_starts = []
# TODO: Use DailyLineupOptimizer.flag_pitcher_starts()
return pitchers, two_starts
```
Panel B1.6 (Probable Pitchers) will always be blank. Both `probable_pitchers` and `two_start_pitchers` in `DashboardData` will always be empty.

**What it should call:**
`DailyLineupOptimizer.flag_pitcher_starts(roster, game_date)` — already fully implemented at line 620 of `backend/fantasy_baseball/daily_lineup_optimizer.py`. It accepts a roster list and returns each pitcher annotated with `has_start: bool`.

The result must be mapped to `List[ProbablePitcherInfo]`:
- `p["name"]` → `ProbablePitcherInfo.name`
- `p["team"]` → `ProbablePitcherInfo.team`
- `p.get("opponent", "")` → `ProbablePitcherInfo.opponent`
- Current date string → `ProbablePitcherInfo.game_date`
- Whether the pitcher appears twice this week → `ProbablePitcherInfo.is_two_start`
- Matchup quality scoring (opponent run environment from `PlayerProjection.era` context) → `ProbablePitcherInfo.matchup_quality`

**DB tables involved:**
- `PlayerProjection` (`player_projections`) — pitcher's `era`, `k_per_nine`, `whip` for stream scoring
- `PatternDetectionAlert` — bullpen overuse / fatigue signals that affect stream score
- `StatcastPerformance` — recent `ip`, `er`, `pitches` for fatigue detection

**Filter conditions:**
```
PlayerProjection.player_id.in_([pitcher_player_ids])
PatternDetectionAlert.pattern_type.in_(["pitcher_fatigue", "bullpen_overuse"])
PatternDetectionAlert.is_active == True
PatternDetectionAlert.game_date >= date.today()
```

**Expected return shape:**
`tuple[List[ProbablePitcherInfo], List[ProbablePitcherInfo]]`
- First list: all rostered SPs/RPs with a confirmed start today
- Second list: subset of SPs who have two scheduled starts in the current scoring week

---

## Implementation Priority Order

| Priority | Entry | Rationale |
|----------|-------|-----------|
| P0 | 7 — `_get_waiver_targets` stub | Service class (`WaiverEdgeDetector`) is fully built; this is a pure wiring gap |
| P0 | 11 — `_get_probable_pitchers` stub | `DailyLineupOptimizer.flag_pitcher_starts()` is fully built; pure wiring gap |
| P1 | 10 — `_get_matchup_preview` stub | Requires Yahoo scoreboard call + category projection logic; most complex |
| P1 | 2 — `team_key=None` roster fetch | One-line fix; high-frequency code path |
| P2 | 6 — `team` field wrong value | Data correctness bug; breaks UI display |
| P2 | 1, 8 — Yahoo failure fallbacks | Degraded-mode policy decision needed |
| P3 | 3 — Hardcoded positions list | Requires league settings persistence design |
| P3 | 4, 9 — `None` sentinel fields | Supplementary data enrichment |

---

## Notes on Out-of-Scope Items

- `_get_or_create_preferences` (lines 227-235) and both preferences CRUD methods are real DB queries with no mock paths. These are correctly implemented.
- `_get_yahoo_client` (lines 159-169) is a lazy-init pattern with proper error recording; not a mock.
- The `reliability_engine.validate_*` calls throughout are live validation calls, not mocks.
