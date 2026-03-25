---
name: db-migrate
description: Run database migrations on Railway. Use when the user asks to apply a migration, run a schema change, or verify that a migration was applied. Always dry-run first.
---

# Database Migration Runner

## When to Use This Skill

Activate when the user says:
- "run the migration"
- "apply the schema change"
- "did the migration run"
- "run migrate_v8_post_draft.py"

## Available Migrations

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/migrate_v8_post_draft.py` | player_daily_metrics, projection_snapshots, pricing_engine column | EPIC-1 |
| `scripts/apply_openclaw_migration.py` | OpenClaw monitoring tables (4 tables) | Applied Mar 25 |
| `scripts/migrate_sport_poll_config.py` | sport_poll_config table (EPIC-5, Apr 8+) | NOT YET |

## Workflow — ALWAYS Follow This Order

### Step 1: Dry run first
```bash
railway run python scripts/migrate_v8_post_draft.py --dry-run
```
Read the SQL output. Confirm it matches the expected schema from HANDOFF.md §2.2.

### Step 2: Apply (only after dry-run confirms correct SQL)
```bash
bash .gemini/skills/db-migrate/scripts/run-migration.sh migrate_v8_post_draft
```

### Step 3: Verify
```bash
bash .gemini/skills/db-migrate/scripts/run-migration.sh --verify-only
```

## Rules

- NEVER run `--downgrade` without Claude Code approval
- NEVER run EPIC-5/EPIC-6 migrations before April 7
- If dry-run SQL looks wrong → STOP and escalate to Claude Code
- After success → update HANDOFF.md §2 to mark migration COMPLETE

## Escalation

- Unexpected SQL in dry-run → Claude Code
- Migration fails mid-run → Claude Code immediately (do NOT retry)
- Schema verification fails → Claude Code
