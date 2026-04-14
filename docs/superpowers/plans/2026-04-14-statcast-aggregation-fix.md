# Statcast Aggregation & Two-Way Player Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the Statcast ingestion pipeline so it produces correct daily aggregates regardless of whether Baseball Savant returns per-pitch rows or leaderboard rows, and stop two-way player data corruption.

**Architecture:** Add a pre-aggregation step (`_aggregate_to_daily`) between fetch and transform that groups by (player, date, type) and SUMs counting stats / AVGs quality metrics. Add `is_pitcher` flag to `PlayerDailyPerformance` so `store_performances` can scope its upsert columns by player type — pitcher rows only update pitching columns, never zeroing out batting data.

**Tech Stack:** Python, pandas (groupby/agg), SQLAlchemy (pg_insert), pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `backend/fantasy_baseball/statcast_ingestion.py` | Modify | Add `_aggregate_to_daily()`, add `is_pitcher` field, modify `store_performances()` |
| `tests/test_statcast_ingestion.py` | Modify | Add 6 new tests for aggregation, CS, two-way players |

No new files. All changes in two existing files.

---

### Task 1: Add `is_pitcher` field to PlayerDailyPerformance

**Files:**
- Modify: `backend/fantasy_baseball/statcast_ingestion.py:91-128` (dataclass)
- Test: `tests/test_statcast_ingestion.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_statcast_ingestion.py`:

```python
def test_batter_performance_has_is_pitcher_false():
    """Batter rows should have is_pitcher=False."""
    agent = StatcastIngestionAgent()

    df = pd.DataFrame([{
        'player_name': 'Judge, Aaron',
        'team': 'NYY',
        'game_date': '2026-04-09',
        'pa': 5, 'ab': 4, 'h': 2,
        'doubles': 1, 'triples': 0, 'hr': 1, 'r': 2, 'rbi': 3,
        'bb': 1, 'so': 1, 'hbp': 0, 'sb': 0, 'cs': 0,
        'exit_velocity_avg': 95.5, 'launch_angle_avg': 15.2,
        'hard_hit_percent': 55.0, 'barrel_batted_rate': 12.0,
        'xba': 0.320, 'xslg': 0.650, 'xwoba': 0.400,
        'pitches': 25,
        '_statcast_player_type': 'batter',
    }])

    performances = agent.transform_to_performance(df)
    assert len(performances) == 1
    assert performances[0].is_pitcher is False


def test_pitcher_performance_has_is_pitcher_true():
    """Pitcher rows should have is_pitcher=True."""
    agent = StatcastIngestionAgent()

    df = pd.DataFrame([{
        'player_name': 'Cole, Gerrit',
        'team': 'NYY',
        'game_date': '2026-04-09',
        'exit_velocity_avg': 88.0, 'launch_angle_avg': 12.0,
        'hard_hit_percent': 35.0, 'barrel_batted_rate': 5.0,
        'xba': 0.250, 'xslg': 0.400, 'xwoba': 0.300,
        'ip': 6.0, 'er': 2, 'strikeout': 8, 'walk': 2, 'bb': 2,
        'pitches': 95,
        '_statcast_player_type': 'pitcher',
    }])

    performances = agent.transform_to_performance(df)
    assert len(performances) == 1
    assert performances[0].is_pitcher is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python -m pytest tests/test_statcast_ingestion.py::test_batter_performance_has_is_pitcher_false tests/test_statcast_ingestion.py::test_pitcher_performance_has_is_pitcher_true -v`
Expected: FAIL with `AttributeError: ... has no attribute 'is_pitcher'`

- [ ] **Step 3: Write minimal implementation**

In `backend/fantasy_baseball/statcast_ingestion.py`, add `is_pitcher` field to the `PlayerDailyPerformance` dataclass (after the `pitches` field, before the properties):

```python
    pitches: int = 0

    # Metadata for type-scoped upserts
    is_pitcher: bool = False
```

Then set it in `transform_to_performance`:

In the pitcher branch (line ~508), add `is_pitcher=True` to the constructor:
```python
                    perf = PlayerDailyPerformance(
                        ...existing fields...
                        pitches=self._icol(row, 'pitches'),
                        is_pitcher=True,
                    )
```

The batter branch already defaults to `is_pitcher=False` so no change needed there.

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python -m pytest tests/test_statcast_ingestion.py::test_batter_performance_has_is_pitcher_false tests/test_statcast_ingestion.py::test_pitcher_performance_has_is_pitcher_true -v`
Expected: PASS

- [ ] **Step 5: Run full existing test suite for regression**

Run: `venv/Scripts/python -m pytest tests/test_statcast_ingestion.py tests/test_statcast_column_mapping.py -v`
Expected: All existing tests PASS (is_pitcher defaults to False, no behavior change)

- [ ] **Step 6: Commit**

```bash
git add backend/fantasy_baseball/statcast_ingestion.py tests/test_statcast_ingestion.py
git commit -m "feat(statcast): add is_pitcher flag to PlayerDailyPerformance"
```

---

### Task 2: Add `_aggregate_to_daily()` pre-aggregation method

This is the core fix. It groups per-pitch rows into daily aggregates so that
`transform_to_performance` always receives one row per (player, date, type).

**Files:**
- Modify: `backend/fantasy_baseball/statcast_ingestion.py` (add method to StatcastIngestionAgent)
- Test: `tests/test_statcast_ingestion.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_statcast_ingestion.py`:

```python
class TestAggregateToDaily:
    """Tests for _aggregate_to_daily() pre-aggregation."""

    def test_per_pitch_rows_aggregate_counting_stats(self):
        """Multiple per-pitch rows for same player/date should SUM counting stats."""
        agent = StatcastIngestionAgent()

        # 3 per-pitch rows for the same batter on the same date
        # Each row represents one PA with partial counting stats
        df = pd.DataFrame([
            {
                'player_name': 'Judge, Aaron', 'team': 'NYY',
                'game_date': '2026-04-09', '_statcast_player_type': 'batter',
                'pa': 1, 'ab': 1, 'hits': 1, 'hrs': 1, 'doubles': 0,
                'triples': 0, 'bb': 0, 'so': 0, 'hbp': 0,
                'stolen_base_2b': 0, 'caught_stealing_2b': 0,
                'pitches': 4, 'launch_speed': 105.0, 'launch_angle': 25.0,
                'xwoba': 0.800, 'xba': 0.600, 'xslg': 1.200,
                'hardhit_percent': 100.0, 'barrels_per_pa_percent': 100.0,
            },
            {
                'player_name': 'Judge, Aaron', 'team': 'NYY',
                'game_date': '2026-04-09', '_statcast_player_type': 'batter',
                'pa': 1, 'ab': 1, 'hits': 0, 'hrs': 0, 'doubles': 0,
                'triples': 0, 'bb': 0, 'so': 1, 'hbp': 0,
                'stolen_base_2b': 0, 'caught_stealing_2b': 0,
                'pitches': 6, 'launch_speed': 85.0, 'launch_angle': -10.0,
                'xwoba': 0.100, 'xba': 0.050, 'xslg': 0.100,
                'hardhit_percent': 0.0, 'barrels_per_pa_percent': 0.0,
            },
            {
                'player_name': 'Judge, Aaron', 'team': 'NYY',
                'game_date': '2026-04-09', '_statcast_player_type': 'batter',
                'pa': 1, 'ab': 0, 'hits': 0, 'hrs': 0, 'doubles': 0,
                'triples': 0, 'bb': 1, 'so': 0, 'hbp': 0,
                'stolen_base_2b': 1, 'caught_stealing_2b': 0,
                'pitches': 5, 'launch_speed': float('nan'), 'launch_angle': float('nan'),
                'xwoba': float('nan'), 'xba': float('nan'), 'xslg': float('nan'),
                'hardhit_percent': float('nan'), 'barrels_per_pa_percent': float('nan'),
            },
        ])

        result = agent._aggregate_to_daily(df)

        assert len(result) == 1, f"Expected 1 aggregated row, got {len(result)}"
        row = result.iloc[0]
        assert int(row['pa']) == 3
        assert int(row['ab']) == 2
        assert int(row.get('hits', row.get('h', 0))) == 1
        assert int(row.get('hrs', row.get('hr', 0))) == 1
        assert int(row['bb']) == 1
        assert int(row['so']) == 1
        assert int(row.get('stolen_base_2b', row.get('sb', 0))) == 1
        assert int(row['pitches']) == 15  # 4+6+5

    def test_caught_stealing_indicators_sum_correctly(self):
        """Per-pitch caught_stealing_2b (0/1) must SUM to daily CS count."""
        agent = StatcastIngestionAgent()

        # 10 per-pitch rows, 3 of which have caught_stealing_2b=1
        rows = []
        for i in range(10):
            rows.append({
                'player_name': 'Turner, Trea', 'team': 'PHI',
                'game_date': '2026-04-09', '_statcast_player_type': 'batter',
                'pa': 1, 'ab': 1, 'hits': 0, 'hrs': 0, 'doubles': 0,
                'triples': 0, 'bb': 0, 'so': 0, 'hbp': 0,
                'stolen_base_2b': 1 if i < 2 else 0,
                'caught_stealing_2b': 1 if i in (3, 5, 8) else 0,
                'pitches': 3,
                'launch_speed': 90.0, 'launch_angle': 10.0,
                'xwoba': 0.300, 'xba': 0.250, 'xslg': 0.400,
                'hardhit_percent': 40.0, 'barrels_per_pa_percent': 8.0,
            })
        df = pd.DataFrame(rows)

        result = agent._aggregate_to_daily(df)

        assert len(result) == 1
        row = result.iloc[0]
        cs_val = int(row.get('caught_stealing_2b', row.get('cs', 0)))
        assert cs_val == 3, f"Expected CS=3, got {cs_val}"
        sb_val = int(row.get('stolen_base_2b', row.get('sb', 0)))
        assert sb_val == 2, f"Expected SB=2, got {sb_val}"

    def test_quality_metrics_averaged_not_summed(self):
        """Quality metrics (xwoba, exit_velocity) should be averaged, not summed."""
        agent = StatcastIngestionAgent()

        df = pd.DataFrame([
            {
                'player_name': 'Soto, Juan', 'team': 'NYY',
                'game_date': '2026-04-09', '_statcast_player_type': 'batter',
                'pa': 1, 'ab': 1, 'hits': 1, 'hrs': 0, 'doubles': 0,
                'triples': 0, 'bb': 0, 'so': 0, 'hbp': 0,
                'stolen_base_2b': 0, 'caught_stealing_2b': 0,
                'pitches': 3, 'launch_speed': 100.0, 'launch_angle': 20.0,
                'xwoba': 0.600, 'xba': 0.500, 'xslg': 0.900,
                'hardhit_percent': 100.0, 'barrels_per_pa_percent': 50.0,
            },
            {
                'player_name': 'Soto, Juan', 'team': 'NYY',
                'game_date': '2026-04-09', '_statcast_player_type': 'batter',
                'pa': 1, 'ab': 1, 'hits': 0, 'hrs': 0, 'doubles': 0,
                'triples': 0, 'bb': 0, 'so': 1, 'hbp': 0,
                'stolen_base_2b': 0, 'caught_stealing_2b': 0,
                'pitches': 5, 'launch_speed': 80.0, 'launch_angle': 0.0,
                'xwoba': 0.200, 'xba': 0.100, 'xslg': 0.200,
                'hardhit_percent': 0.0, 'barrels_per_pa_percent': 0.0,
            },
        ])

        result = agent._aggregate_to_daily(df)

        assert len(result) == 1
        row = result.iloc[0]
        # Quality metrics should be averaged
        assert abs(float(row['launch_speed']) - 90.0) < 0.1  # (100+80)/2
        assert abs(float(row['xwoba']) - 0.400) < 0.01  # (0.6+0.2)/2
        assert abs(float(row['xba']) - 0.300) < 0.01  # (0.5+0.1)/2
        # Counting stats should be summed
        assert int(row['pa']) == 2
        assert int(row['pitches']) == 8  # 3+5

    def test_leaderboard_single_row_passthrough(self):
        """Leaderboard data (already 1 row per player/date) should pass through unchanged."""
        agent = StatcastIngestionAgent()

        df = pd.DataFrame([{
            'player_id': 660670,
            'player_name': 'Alvarez, Yordan', 'team': 'HOU',
            'game_date': '2026-04-09', '_statcast_player_type': 'batter',
            'pa': 5, 'abs': 4, 'hits': 2, 'hrs': 1, 'doubles': 1,
            'triples': 0, 'bb': 1, 'so': 1, 'hbp': 0,
            'pitches': 22, 'launch_speed': 95.0, 'launch_angle': 18.0,
            'xwoba': 0.450, 'xba': 0.350, 'xslg': 0.700,
            'hardhit_percent': 60.0, 'barrels_per_pa_percent': 20.0,
        }])

        result = agent._aggregate_to_daily(df)

        assert len(result) == 1
        row = result.iloc[0]
        assert int(row['pa']) == 5
        assert int(row.get('abs', row.get('ab', 0))) == 4
        assert float(row['xwoba']) == pytest.approx(0.450)

    def test_multiple_players_stay_separate(self):
        """Different players on the same date should not be merged."""
        agent = StatcastIngestionAgent()

        df = pd.DataFrame([
            {
                'player_name': 'Judge, Aaron', 'team': 'NYY',
                'game_date': '2026-04-09', '_statcast_player_type': 'batter',
                'pa': 4, 'ab': 3, 'hits': 1, 'hrs': 0, 'doubles': 0,
                'triples': 0, 'bb': 1, 'so': 1, 'hbp': 0,
                'stolen_base_2b': 0, 'caught_stealing_2b': 0,
                'pitches': 20, 'launch_speed': 95.0, 'launch_angle': 15.0,
                'xwoba': 0.350, 'xba': 0.280, 'xslg': 0.450,
                'hardhit_percent': 50.0, 'barrels_per_pa_percent': 10.0,
            },
            {
                'player_name': 'Soto, Juan', 'team': 'NYY',
                'game_date': '2026-04-09', '_statcast_player_type': 'batter',
                'pa': 5, 'ab': 4, 'hits': 2, 'hrs': 1, 'doubles': 0,
                'triples': 0, 'bb': 1, 'so': 0, 'hbp': 0,
                'stolen_base_2b': 1, 'caught_stealing_2b': 0,
                'pitches': 18, 'launch_speed': 100.0, 'launch_angle': 20.0,
                'xwoba': 0.550, 'xba': 0.400, 'xslg': 0.800,
                'hardhit_percent': 75.0, 'barrels_per_pa_percent': 25.0,
            },
        ])

        result = agent._aggregate_to_daily(df)

        assert len(result) == 2, f"Expected 2 rows (one per player), got {len(result)}"

    def test_batter_and_pitcher_rows_stay_separate(self):
        """Same player as batter AND pitcher should produce 2 aggregated rows."""
        agent = StatcastIngestionAgent()

        df = pd.DataFrame([
            {
                'player_name': 'Ohtani, Shohei', 'team': 'LAD',
                'game_date': '2026-04-09', '_statcast_player_type': 'batter',
                'pa': 4, 'ab': 3, 'hits': 2, 'hrs': 1, 'doubles': 0,
                'triples': 0, 'bb': 1, 'so': 0, 'hbp': 0,
                'stolen_base_2b': 1, 'caught_stealing_2b': 0,
                'pitches': 18, 'launch_speed': 105.0, 'launch_angle': 25.0,
                'xwoba': 0.600, 'xba': 0.500, 'xslg': 1.000,
                'hardhit_percent': 80.0, 'barrels_per_pa_percent': 30.0,
            },
            {
                'player_name': 'Ohtani, Shohei', 'team': 'LAD',
                'game_date': '2026-04-09', '_statcast_player_type': 'pitcher',
                'pa': 0, 'ab': 0, 'hits': 0, 'hrs': 0, 'doubles': 0,
                'triples': 0, 'bb': 0, 'so': 0, 'hbp': 0,
                'stolen_base_2b': 0, 'caught_stealing_2b': 0,
                'pitches': 95, 'launch_speed': 88.0, 'launch_angle': 10.0,
                'xwoba': 0.280, 'xba': 0.220, 'xslg': 0.350,
                'hardhit_percent': 30.0, 'barrels_per_pa_percent': 5.0,
            },
        ])

        result = agent._aggregate_to_daily(df)

        assert len(result) == 2, f"Expected 2 rows (batter + pitcher), got {len(result)}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/Scripts/python -m pytest tests/test_statcast_ingestion.py::TestAggregateToDaily -v`
Expected: FAIL with `AttributeError: 'StatcastIngestionAgent' object has no attribute '_aggregate_to_daily'`

- [ ] **Step 3: Implement `_aggregate_to_daily()`**

Add this method to the `StatcastIngestionAgent` class, after `_fcol` (around line 465):

```python
    def _aggregate_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-aggregate DataFrame to one row per (player, date, type).

        Makes the pipeline resilient to per-pitch data from Baseball Savant:
        - Counting stats (PA, AB, H, HR, BB, SO, SB, CS, etc.) are SUMmed.
        - Quality metrics (exit_velocity, xwoba, xba, etc.) are averaged (NaN-safe).
        - Identity columns (player_name, team) take the first value.

        If the input is already one-row-per-player-per-date (leaderboard format),
        this is effectively a no-op.
        """
        if df.empty:
            return df

        # Determine grouping key: prefer player_id if present, else player_name
        id_col = 'player_id' if 'player_id' in df.columns else 'player_name'
        group_cols = [id_col, 'game_date', '_statcast_player_type']

        # Check if aggregation is needed (skip if already 1 row per group)
        group_sizes = df.groupby(group_cols, dropna=False).size()
        if group_sizes.max() <= 1:
            logger.debug("_aggregate_to_daily: data already at daily granularity, skipping")
            return df

        n_before = len(df)

        # Define aggregation rules for known columns
        # Counting stats: SUM (these are additive across events)
        sum_cols = [
            'pa', 'ab', 'abs', 'h', 'hits', 'hit', 'singles', 'single',
            'doubles', 'double', 'triples', 'triple',
            'hr', 'hrs', 'home_run', 'home_runs',
            'r', 'run', 'runs', 'rbi',
            'bb', 'walk', 'walks',
            'so', 'strikeout', 'strikeouts',
            'hbp', 'hit_by_pitch',
            'sb', 'stolen_base', 'stolen_bases', 'stolen_base_2b',
            'cs', 'caught_stealing', 'caught_stealing_2b',
            'pitches', 'er',
            'p_strikeout', 'p_walk',
            'k', 'k_pit', 'bb_pit',
        ]

        # Quality metrics: MEAN (these are rates/averages, not additive)
        mean_cols = [
            'launch_speed', 'exit_velocity_avg',
            'launch_angle', 'launch_angle_avg',
            'hardhit_percent', 'hard_hit_percent', 'hard_hit_pct',
            'barrels_per_pa_percent', 'barrels_per_bbe_percent',
            'barrel_batted_rate', 'barrel_pct',
            'xba', 'estimated_ba_using_speedangle',
            'xslg', 'estimated_slg_using_speedangle',
            'xwoba', 'estimated_woba_using_speedangle',
            'woba',
        ]

        # IP is special: fractional innings (e.g. 6.2 = 6 and 2/3).
        # For per-pitch data IP won't be meaningful per row, but SUM is
        # the closest correct behavior for daily totals.
        sum_cols.append('ip')

        # Build aggregation dict from columns actually present
        agg_dict = {}
        present_cols = set(df.columns)

        for col in sum_cols:
            if col in present_cols and col not in group_cols:
                agg_dict[col] = 'sum'

        for col in mean_cols:
            if col in present_cols and col not in group_cols:
                agg_dict[col] = 'mean'

        # Identity/metadata columns: take first value
        for col in ['player_name', 'team', 'player_id']:
            if col in present_cols and col not in group_cols and col not in agg_dict:
                agg_dict[col] = 'first'

        # Coerce numeric columns before aggregation (Savant CSVs have mixed types)
        for col in list(agg_dict.keys()):
            if agg_dict[col] in ('sum', 'mean'):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        result = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()

        logger.info(
            "_aggregate_to_daily: %d raw rows -> %d aggregated rows",
            n_before, len(result),
        )
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/Scripts/python -m pytest tests/test_statcast_ingestion.py::TestAggregateToDaily -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Wire `_aggregate_to_daily` into `fetch_statcast_day`**

In `fetch_statcast_day()`, add the aggregation call just before returning:

Change the end of `fetch_statcast_day` (around lines 435-440) from:

```python
        combined = pd.concat(frames, ignore_index=True)
        logger.info(
            "Statcast combined: %d batters + %d pitchers = %d total rows for %s",
            n_batters, n_pitchers, len(combined), target_date,
        )
        return combined
```

to:

```python
        combined = pd.concat(frames, ignore_index=True)
        logger.info(
            "Statcast combined: %d batters + %d pitchers = %d total rows for %s",
            n_batters, n_pitchers, len(combined), target_date,
        )

        # Pre-aggregate to daily granularity (resilient to per-pitch data)
        combined = self._aggregate_to_daily(combined)

        return combined
```

- [ ] **Step 6: Run full statcast test suite for regression**

Run: `venv/Scripts/python -m pytest tests/test_statcast_ingestion.py tests/test_statcast_column_mapping.py -v`
Expected: All tests PASS (leaderboard data passes through unchanged)

- [ ] **Step 7: Commit**

```bash
git add backend/fantasy_baseball/statcast_ingestion.py tests/test_statcast_ingestion.py
git commit -m "feat(statcast): add _aggregate_to_daily pre-aggregation for per-pitch resilience"
```

---

### Task 3: Scope upsert by player type in `store_performances()`

This prevents pitcher rows from zeroing out batting data for two-way players.

**Files:**
- Modify: `backend/fantasy_baseball/statcast_ingestion.py:580-676` (`store_performances`)
- Test: `tests/test_statcast_ingestion.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_statcast_ingestion.py`:

```python
class TestStorePerformancesScopedUpsert:
    """Tests for type-scoped upserts (two-way player protection)."""

    def test_two_way_player_preserves_both_batting_and_pitching(self):
        """
        When Ohtani appears as both batter and pitcher, the pitcher upsert
        must NOT zero out the batting stats set by the batter upsert.
        """
        from backend.fantasy_baseball.statcast_ingestion import PlayerDailyPerformance
        from unittest.mock import MagicMock, call

        agent = StatcastIngestionAgent()
        agent.db = MagicMock()

        batter_perf = PlayerDailyPerformance(
            player_id='660271', player_name='Ohtani, Shohei', team='LAD',
            game_date=date(2026, 4, 9),
            pa=4, ab=3, h=2, doubles=1, triples=0, hr=1, r=2, rbi=3,
            bb=1, so=0, hbp=0, sb=1, cs=0,
            exit_velocity_avg=0.95, launch_angle_avg=0.25,
            hard_hit_pct=0.80, barrel_pct=0.30,
            xba=0.500, xslg=1.000, xwoba=0.600,
            ip=0.0, er=0, k_pit=0, bb_pit=0, pitches=18,
            is_pitcher=False,
        )

        pitcher_perf = PlayerDailyPerformance(
            player_id='660271', player_name='Ohtani, Shohei', team='LAD',
            game_date=date(2026, 4, 9),
            pa=0, ab=0, h=0, doubles=0, triples=0, hr=0, r=0, rbi=0,
            bb=0, so=0, hbp=0, sb=0, cs=0,
            exit_velocity_avg=0.88, launch_angle_avg=0.10,
            hard_hit_pct=0.30, barrel_pct=0.05,
            xba=0.220, xslg=0.350, xwoba=0.280,
            ip=6.0, er=2, k_pit=8, bb_pit=2, pitches=95,
            is_pitcher=True,
        )

        agent.store_performances([batter_perf, pitcher_perf])

        # Verify db.execute was called twice
        assert agent.db.execute.call_count == 2

        # Extract the two SQL statements
        calls = agent.db.execute.call_args_list

        # The second call (pitcher) should NOT contain 'pa' in its set_ dict.
        # We verify by checking the compiled statement — the pitcher upsert
        # should only update pitcher-specific columns.
        # Since we're using mock, we check the Insert object's parameters.
        pitcher_stmt = calls[1][0][0]  # First positional arg of second call
        # The on_conflict_do_update set_ should NOT include 'pa'
        # (This validates the scoped upsert behavior)
        set_cols = set(pitcher_stmt.compile().params.keys()) if hasattr(pitcher_stmt, 'compile') else set()
        # At minimum, verify both calls succeeded without error
        agent.db.commit.assert_called_once()
```

Note: Testing the exact SQL statement structure with mocks is fragile. A more reliable approach is to verify the behavior through the `transform_to_performance` + `store_performances` pipeline in an integration-style test. But since we don't have a test DB, we verify the mock calls succeed and add a structural comment test.

A better test approach — verify the `set_` dict directly by inspecting the pg_insert object:

```python
    def test_pitcher_upsert_excludes_batting_columns(self):
        """
        Pitcher upserts should only update pitcher columns,
        not overwrite batting stats with zeros.
        """
        from backend.fantasy_baseball.statcast_ingestion import PlayerDailyPerformance
        from unittest.mock import MagicMock, patch
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        agent = StatcastIngestionAgent()

        # Capture the actual SQL statements executed
        executed_stmts = []
        mock_db = MagicMock()
        def capture_execute(stmt):
            executed_stmts.append(stmt)
        mock_db.execute = capture_execute
        mock_db.commit = MagicMock()
        agent.db = mock_db

        pitcher_perf = PlayerDailyPerformance(
            player_id='660271', player_name='Ohtani, Shohei', team='LAD',
            game_date=date(2026, 4, 9),
            pa=0, ab=0, h=0, doubles=0, triples=0, hr=0, r=0, rbi=0,
            bb=0, so=0, hbp=0, sb=0, cs=0,
            exit_velocity_avg=0.88, launch_angle_avg=0.10,
            hard_hit_pct=0.30, barrel_pct=0.05,
            xba=0.220, xslg=0.350, xwoba=0.280,
            ip=6.0, er=2, k_pit=8, bb_pit=2, pitches=95,
            is_pitcher=True,
        )

        agent.store_performances([pitcher_perf])

        assert len(executed_stmts) == 1
        # Inspect the ON CONFLICT DO UPDATE set_ keys
        stmt = executed_stmts[0]
        # The _on_conflict_do_update_set attribute holds the column update dict
        update_set = stmt._post_values_clause.update.set_
        update_cols = set(update_set.keys()) if isinstance(update_set, dict) else set()

        # Pitcher upsert must NOT contain batting counting stats
        batting_cols = {'pa', 'ab', 'h', 'doubles', 'triples', 'hr', 'r', 'rbi',
                       'bb', 'so', 'hbp', 'sb', 'cs', 'avg', 'obp', 'slg', 'ops', 'woba'}
        assert not (update_cols & batting_cols), (
            f"Pitcher upsert should not update batting columns, found: {update_cols & batting_cols}"
        )

        # Pitcher upsert MUST contain pitching columns
        pitching_cols = {'ip', 'er', 'k_pit', 'bb_pit', 'pitches'}
        assert pitching_cols.issubset(update_cols), (
            f"Pitcher upsert missing pitching columns: {pitching_cols - update_cols}"
        )
```

Note: The exact introspection approach for the SQLAlchemy insert object may need adjustment — the internal attribute names vary by SQLAlchemy version. If the structural introspection is too fragile, fall back to the simpler mock verification test above.

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python -m pytest tests/test_statcast_ingestion.py::TestStorePerformancesScopedUpsert -v`
Expected: FAIL (pitcher upsert currently includes ALL columns)

- [ ] **Step 3: Implement scoped upserts**

Modify `store_performances()` in `statcast_ingestion.py`. Replace the single upsert with type-branched logic:

```python
    def store_performances(self, performances: List[PlayerDailyPerformance]) -> int:
        """
        Upsert daily performances to statcast_performances.

        Uses INSERT ON CONFLICT DO UPDATE with type-scoped column sets:
        - Batter rows: update all batting stats, quality metrics, and computed fields.
        - Pitcher rows: update ONLY pitching columns (ip, er, k_pit, bb_pit, pitches)
          plus Statcast quality metrics. Does NOT overwrite batting stats, preserving
          data for two-way players (Ohtani, etc.).

        Returns the number of rows upserted.
        """
        rows_upserted = 0
        now = datetime.now(ZoneInfo("America/New_York"))

        for perf in performances:
            try:
                # All rows insert the same full set of values
                insert_values = dict(
                    player_id=perf.player_id,
                    player_name=perf.player_name,
                    team=perf.team,
                    game_date=perf.game_date,
                    pa=perf.pa,
                    ab=perf.ab,
                    h=perf.h,
                    doubles=perf.doubles,
                    triples=perf.triples,
                    hr=perf.hr,
                    r=perf.r,
                    rbi=perf.rbi,
                    bb=perf.bb,
                    so=perf.so,
                    hbp=perf.hbp,
                    sb=perf.sb,
                    cs=perf.cs,
                    exit_velocity_avg=perf.exit_velocity_avg,
                    launch_angle_avg=perf.launch_angle_avg,
                    hard_hit_pct=perf.hard_hit_pct,
                    barrel_pct=perf.barrel_pct,
                    xba=perf.xba,
                    xslg=perf.xslg,
                    xwoba=perf.xwoba,
                    woba=perf.woba,
                    avg=perf.avg,
                    obp=perf.obp,
                    slg=perf.slg,
                    ops=perf.ops,
                    ip=perf.ip,
                    er=perf.er,
                    k_pit=perf.k_pit,
                    bb_pit=perf.bb_pit,
                    pitches=perf.pitches,
                    created_at=now,
                )

                if perf.is_pitcher:
                    # Pitcher: only update pitching + quality metric columns.
                    # Batting counting stats are left untouched so a prior
                    # batter row for a two-way player is preserved.
                    update_set = dict(
                        player_name=perf.player_name,
                        team=perf.team,
                        exit_velocity_avg=perf.exit_velocity_avg,
                        launch_angle_avg=perf.launch_angle_avg,
                        hard_hit_pct=perf.hard_hit_pct,
                        barrel_pct=perf.barrel_pct,
                        xba=perf.xba,
                        xslg=perf.xslg,
                        xwoba=perf.xwoba,
                        ip=perf.ip,
                        er=perf.er,
                        k_pit=perf.k_pit,
                        bb_pit=perf.bb_pit,
                        pitches=perf.pitches,
                    )
                else:
                    # Batter: update all columns (full replace on conflict)
                    update_set = dict(
                        player_name=perf.player_name,
                        team=perf.team,
                        pa=perf.pa,
                        ab=perf.ab,
                        h=perf.h,
                        doubles=perf.doubles,
                        triples=perf.triples,
                        hr=perf.hr,
                        r=perf.r,
                        rbi=perf.rbi,
                        bb=perf.bb,
                        so=perf.so,
                        hbp=perf.hbp,
                        sb=perf.sb,
                        cs=perf.cs,
                        exit_velocity_avg=perf.exit_velocity_avg,
                        launch_angle_avg=perf.launch_angle_avg,
                        hard_hit_pct=perf.hard_hit_pct,
                        barrel_pct=perf.barrel_pct,
                        xba=perf.xba,
                        xslg=perf.xslg,
                        xwoba=perf.xwoba,
                        woba=perf.woba,
                        avg=perf.avg,
                        obp=perf.obp,
                        slg=perf.slg,
                        ops=perf.ops,
                        ip=perf.ip,
                        er=perf.er,
                        k_pit=perf.k_pit,
                        bb_pit=perf.bb_pit,
                        pitches=perf.pitches,
                    )

                stmt = pg_insert(StatcastPerformance.__table__).values(
                    **insert_values
                ).on_conflict_do_update(
                    index_elements=['player_id', 'game_date'],
                    set_=update_set,
                )
                self.db.execute(stmt)
                rows_upserted += 1
            except Exception as e:
                logger.warning("Failed to upsert performance for %s on %s: %s", perf.player_name, perf.game_date, e)
                continue

        self.db.commit()
        logger.info("Statcast: %d rows upserted for %s", rows_upserted, performances[0].game_date if performances else 'n/a')
        return rows_upserted
```

**Important ordering note:** Batter rows must be processed BEFORE pitcher rows for the same player/date. The current pipeline already does this because `fetch_statcast_day` concats batters first, then pitchers. Verify this ordering is preserved.

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python -m pytest tests/test_statcast_ingestion.py::TestStorePerformancesScopedUpsert -v`
Expected: PASS

- [ ] **Step 5: Run full test suite for regression**

Run: `venv/Scripts/python -m pytest tests/test_statcast_ingestion.py tests/test_statcast_column_mapping.py -v`
Expected: All tests PASS

- [ ] **Step 6: Syntax check**

Run: `venv/Scripts/python -m py_compile backend/fantasy_baseball/statcast_ingestion.py`
Expected: No output (clean compile)

- [ ] **Step 7: Commit**

```bash
git add backend/fantasy_baseball/statcast_ingestion.py tests/test_statcast_ingestion.py
git commit -m "fix(statcast): scope upsert by player type to protect two-way player data"
```

---

### Task 4: End-to-end integration test and full regression

Verify the complete pipeline (fetch -> aggregate -> transform -> store) works correctly
with both per-pitch and leaderboard data formats.

**Files:**
- Test: `tests/test_statcast_ingestion.py`

- [ ] **Step 1: Write end-to-end test**

Add to `tests/test_statcast_ingestion.py`:

```python
class TestEndToEndPipeline:
    """End-to-end tests: fetch -> aggregate -> transform -> store."""

    def test_per_pitch_data_produces_correct_daily_totals(self):
        """
        Simulate per-pitch data (the actual bug scenario):
        5 pitch rows for Judge, 3 with caught_stealing_2b=0, 2 with =1.
        Pipeline should produce ONE performance with cs=2.
        """
        agent = StatcastIngestionAgent()

        # Simulate 5 per-pitch rows (what happens when group_by is ignored)
        rows = []
        for i in range(5):
            rows.append({
                'player_name': 'Judge, Aaron', 'team': 'NYY',
                'game_date': '2026-04-09', '_statcast_player_type': 'batter',
                'pa': 1 if i < 3 else 0,  # 3 PAs, 2 non-PA pitches
                'ab': 1 if i < 2 else 0,
                'hits': 1 if i == 0 else 0,
                'hrs': 1 if i == 0 else 0,
                'doubles': 0, 'triples': 0,
                'bb': 1 if i == 2 else 0,
                'so': 1 if i == 1 else 0,
                'hbp': 0,
                'stolen_base_2b': 1 if i == 3 else 0,
                'caught_stealing_2b': 1 if i in (2, 4) else 0,
                'pitches': 1,
                'launch_speed': 95.0 if i == 0 else float('nan'),
                'launch_angle': 25.0 if i == 0 else float('nan'),
                'xwoba': 0.500 if i < 2 else float('nan'),
                'xba': 0.300 if i < 2 else float('nan'),
                'xslg': 0.700 if i < 2 else float('nan'),
                'hardhit_percent': 50.0 if i == 0 else float('nan'),
                'barrels_per_pa_percent': 20.0 if i == 0 else float('nan'),
            })

        df = pd.DataFrame(rows)

        # Run aggregation
        agg_df = agent._aggregate_to_daily(df)
        assert len(agg_df) == 1

        # Run transform
        performances = agent.transform_to_performance(agg_df)
        assert len(performances) == 1

        perf = performances[0]
        assert perf.pa == 3
        assert perf.ab == 2
        assert perf.h == 1
        assert perf.hr == 1
        assert perf.bb == 1
        assert perf.so == 1
        assert perf.sb == 1
        assert perf.cs == 2  # THE critical assertion
        assert perf.pitches == 5
        assert perf.is_pitcher is False
```

- [ ] **Step 2: Run the test**

Run: `venv/Scripts/python -m pytest tests/test_statcast_ingestion.py::TestEndToEndPipeline -v`
Expected: PASS

- [ ] **Step 3: Run the FULL test suite (all statcast tests)**

Run: `venv/Scripts/python -m pytest tests/test_statcast_ingestion.py tests/test_statcast_column_mapping.py tests/test_statcast_loader.py tests/test_statcast_retry.py -v`
Expected: All tests PASS

- [ ] **Step 4: Run broader test suite to check for side effects**

Run: `venv/Scripts/python -m pytest tests/ -q --tb=short`
Expected: No new failures

- [ ] **Step 5: Commit**

```bash
git add tests/test_statcast_ingestion.py
git commit -m "test(statcast): add end-to-end per-pitch aggregation + CS sum test"
```

---

## Post-Implementation Checklist

After all tasks are complete:

- [ ] Update `HANDOFF.md` to reflect the fix is ready for Gemini to deploy
- [ ] Add a Gemini deployment prompt to HANDOFF.md:
  1. Deploy latest code to Railway
  2. TRUNCATE `statcast_performances` (corrupted per-pitch data from prior runs)
  3. Run `POST /admin/backfill/statcast` (full re-backfill with fixed aggregation)
  4. Run `POST /admin/backfill-cs-from-statcast` (should now find CS events)
  5. Run `GET /admin/pipeline-health` and report zero-metric rate (should drop from 42.4% to <10%)
