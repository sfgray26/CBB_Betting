# Savant Pitch Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `savant_pitch_quality`, a disabled-by-default Savant-native pitcher skill signal for waiver and breakout detection.

**Architecture:** Add a focused calculator module that converts existing Savant pitcher rows into a 100-centered composite and waiver signals. Persist results in a dedicated table via a migration/backfill script, seed disabled feature flags, and document agent impact in `HANDOFF.md`.

**Tech Stack:** Python, SQLAlchemy ORM, PostgreSQL, pytest, existing `feature_flags`, existing `statcast_pitcher_metrics` source table.

---

### Task 1: Calculator Tests

**Files:**
- Create: `tests/test_savant_pitch_quality.py`
- Create later: `backend/fantasy_baseball/savant_pitch_quality.py`

- [ ] Add tests for high-skill breakout arm, low-confidence watchlist, ratio risk, and population scoring.
- [ ] Run `python -m pytest tests/test_savant_pitch_quality.py -v` and confirm import failure.

### Task 2: Calculator Module

**Files:**
- Create: `backend/fantasy_baseball/savant_pitch_quality.py`

- [ ] Implement `SavantPitcherInput`, `SavantPitchQualityScore`, `calculate_savant_pitch_quality`, and `score_pitcher_population`.
- [ ] Keep score transparent: arsenal 35%, bat-missing 30%, contact suppression 20%, command/stability 15%, bounded trend adjustment, confidence damping.
- [ ] Run `python -m pytest tests/test_savant_pitch_quality.py -v` and confirm pass.

### Task 3: ORM And Migration

**Files:**
- Modify: `backend/models.py`
- Create: `scripts/migration_savant_pitch_quality.py`

- [ ] Add `SavantPitchQualityScore` ORM model mapped to `savant_pitch_quality_scores`.
- [ ] Add idempotent migration script that creates the table and indexes without touching FanGraphs-specific columns.

### Task 4: Backfill And Flags

**Files:**
- Create: `scripts/backfill_savant_pitch_quality.py`
- Create: `scripts/seed_savant_pitch_quality_flags.py`

- [ ] Read current `statcast_pitcher_metrics` rows.
- [ ] Compute score population and upsert into `savant_pitch_quality_scores`.
- [ ] Seed disabled flags: `savant_pitch_quality_enabled`, `savant_pitch_quality_waiver_signals_enabled`, `savant_pitch_quality_projection_adjustments_enabled`.

### Task 5: Documentation And Handoff

**Files:**
- Modify: `HANDOFF.md`
- Modify as needed: agent docs already touched for MCP context.

- [ ] Add operational summary, feature flags, scripts, and agent routing notes.
- [ ] Make clear this is inactive until migration/backfill/validation runs and flags are enabled.

### Task 6: Verification

**Commands:**
- `python -m pytest tests/test_savant_pitch_quality.py -v`
- `python -m py_compile backend/fantasy_baseball/savant_pitch_quality.py scripts/migration_savant_pitch_quality.py scripts/backfill_savant_pitch_quality.py scripts/seed_savant_pitch_quality_flags.py`
- Optional DB dry run when a reachable database is configured.

Expected: focused tests pass and py_compile succeeds.
