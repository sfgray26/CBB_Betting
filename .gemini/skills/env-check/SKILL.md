---
name: env-check
description: Verify Railway environment variables for CBB Edge. Use when asked to check env vars, confirm a variable is set, compare current vs required config, or diagnose missing configuration.
---

# Environment Variable Checker

## When to Use This Skill

Activate when:
- "check the env vars"
- "is INTEGRITY_SWEEP_ENABLED set"
- "verify Railway config"
- "what env vars are missing"
- "confirm ENABLE_MLB_ANALYSIS is true"

## Required Variables (HANDOFF.md §5.1)

### Critical — App will not start without these
| Variable | Expected | Notes |
|----------|----------|-------|
| `DATABASE_URL` | `postgres://...` | PostgreSQL connection |
| `THE_ODDS_API_KEY` | `<key>` | Odds API key |
| `KENPOM_API_KEY` | `<key>` | KenPom ratings |
| `API_KEY_USER1` | `<key>` | Admin API key |
| `DISCORD_BOT_TOKEN` | `<token>` | Discord alerts |
| `YAHOO_CLIENT_ID` | `<id>` | Fantasy Baseball |
| `YAHOO_CLIENT_SECRET` | `<secret>` | Fantasy Baseball |
| `YAHOO_REFRESH_TOKEN` | `<token>` | Fantasy Baseball |

### Feature Flags — Must be set correctly NOW
| Variable | Expected Value | Current Status |
|----------|---------------|----------------|
| `INTEGRITY_SWEEP_ENABLED` | `false` | CRITICAL — prevents restart loop |
| `ENABLE_MLB_ANALYSIS` | `true` | MLB model active |
| `ENABLE_INGESTION_ORCHESTRATOR` | `true` | Data pipeline |

### Optional Feature Flags
| Variable | When Needed |
|----------|------------|
| `DISCORD_CHANNEL_FANTASY_WAIVERS` | EPIC-3 Discord routing |
| `DISCORD_CHANNEL_FANTASY_LINEUPS` | EPIC-3 Discord routing |
| `DISCORD_CHANNEL_OPENCLAW_BRIEFS` | OpenClaw morning brief |

## Workflow

### Full check
```bash
bash .gemini/skills/env-check/scripts/check-vars.sh
```

### Critical-only check
```bash
bash .gemini/skills/env-check/scripts/check-vars.sh --critical-only
```

### Check a specific variable
```bash
railway variables | grep INTEGRITY_SWEEP_ENABLED
```

### Set a missing variable
```bash
railway variables set VARIABLE_NAME=value
```

## After Checking

If any critical variable is missing → set it immediately via `railway variables set`.
If any feature flag is wrong → set it and watch logs to confirm no restart loop.
Report findings by updating HANDOFF.md §16.4.
