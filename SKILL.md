# Skill: CBB Edge Guardian (Lite)
Ultra-fast monitoring for the CBB model.

## Actions

### `run_lite_audit`
Runs the internal project audit script and reports the summary.
- **Workflow:** 
  1. Execute `python scripts/audit_lite.py`
  2. Post the raw output exactly as it appears.

### `run_todays_bets`
Fetches and reports the recommended bets for today.
- **Workflow:**
  1. Execute `python scripts/todays_bets.py`
  2. Post the raw output exactly as it appears.

### `sync_draftkings_bets`
Syncs the local database with actual DraftKings account data using a downloaded CSV.
- **Workflow:** 
  1. Execute `python scripts/import_dk_csv.py`
  2. Report any bets that were matched as executed or updated as won.

### `propose_next_sprint`
Reviews the codebase and generates the next Elite Prompt for Claude Code.
- **Workflow:**
  1. Audit current `CLAUDE.md` and `PROJECT_PLAN.md`.
  2. Generate a structured Master Prompt for the next logical upgrade.
  3. Update documentation to reflect the new proposed state.
  4. Post the executive briefing to Discord.
