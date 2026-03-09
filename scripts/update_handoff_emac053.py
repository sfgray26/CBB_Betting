"""Temporary script: update HANDOFF.md to EMAC-053 state. Delete after use."""
with open('HANDOFF.md', 'r', encoding='utf-8') as f:
    content = f.read()

replacements = [
    # 1. Ground truth line
    (
        '> Ground truth as of EMAC-051. Operator: Claude Code (Master Architect).',
        '> Ground truth as of EMAC-052. Operator: Claude Code (Master Architect).'
    ),
    # 2. Last completed
    (
        '**Last completed:** EMAC-051 \u2014 O-10 Line Movement Monitor Implemented (G-15). 481/481 tests passing. Scheduled every 30m.',
        '**Last completed:** EMAC-052 \u2014 A-29 dead .openclaw import removed from analysis.py. G-14/G-15 validated. 481/481 tests passing.'
    ),
    # 3. Add A-29 and O-10 Claude rows before G-15
    (
        '| G-15 | Gemini | O-10 Line Movement Monitor implemented. Scheduled every 30m. Discord alerts wired. |',
        '| A-29 | Claude | Remove dead .openclaw relative import from analysis.py. Non-breaking cleanup. 481/481 tests. |\n'
        '| O-10 | Claude | BET_ADVERSE_MOVE detection in odds_monitor.py: event-driven, T-2h golden window, >2pt moves, 4 tests. |\n'
        '| G-15 | Gemini | O-10 Line Movement Monitor implemented. Scheduled every 30m. Discord alerts wired. |'
    ),
    # 4. Claude current focus
    (
        '| Claude Code | Master Architect | FULL | Monitoring \u2014 no active code missions. |',
        '| Claude Code | Master Architect | FULL | A-30: Wire Nightly Health Check APScheduler + morning briefing audit |'
    ),
    # 5. Kimi current focus
    (
        '| Kimi CLI | Deep Intelligence + **OpenClaw Config Owner** | FULL | K-6: O-8 baseline script design |',
        '| Kimi CLI | Deep Intelligence + **OpenClaw Config Owner** | FULL | K-7: Design A-30 Nightly Health Check thresholds |'
    ),
    # 6. Replace A-28 mission block with A-30
    (
        '### CLAUDE CODE \u2014 A-28: MIN_BET_EDGE 2.0% Experiment [COMPLETE]\n\n'
        'Verify `MIN_BET_EDGE` env var is wired in `betting_model.py` and controls the 2% threshold.\n'
        'If not wired, add it using `get_float_env("MIN_BET_EDGE", "2.0")`.\n'
        'Document expected impact: raising from 2% \u2192 2.5% would further reduce BET-tier volume (~15-20% reduction estimated).\n'
        'No Railway changes needed \u2014 document how operator sets it.\n'
        'Update `.env.example` if missing. Add 1-2 tests if the wiring is new.\n'
        'No commit required unless code change needed \u2014 report findings only.\n\n'
        '**STATUS (EMAC-047 findings):** `MIN_BET_EDGE` is ALREADY FULLY WIRED.\n'
        '- Location: `backend/betting_model.py` lines 1774-1775 (D4 block).\n'
        '- Current default: **2.5%** (not 2.0% \u2014 already tuned more conservatively than spec assumed).\n'
        '- Uses `get_float_env("MIN_BET_EDGE", "2.5")` \u2014 fully env-overridable.\n'
        '- Pass reason string: `f"Edge {edge:.1%} below MIN_BET_EDGE {floor:.1%} \u2014 signal too marginal to size"`.\n'
        '- Missing from `.env.example` \u2014 added in EMAC-047.\n'
        '- **No code change required.** Operator raises threshold via Railway env var `MIN_BET_EDGE=3.0` (etc.).',
        '### CLAUDE CODE \u2014 A-30: Wire Nightly Health Check + Morning Briefing Audit [MEDIUM]\n\n'
        'The HEARTBEAT defines a Nightly Health Check (4:30 AM ET daily) but it is not wired as an APScheduler job.\n'
        'Also: scout.py has `generate_morning_briefing()` \u2014 verify if this is called anywhere in main.py.\n\n'
        'Steps:\n'
        '1. Read `backend/main.py` scheduler jobs section to see what is currently scheduled.\n'
        '2. Read `backend/services/scout.py` to find `generate_morning_briefing()` signature.\n'
        '3. If morning briefing is NOT scheduled: add `_morning_briefing_job` as APScheduler cron at 7 AM ET daily.\n'
        '4. Write `_nightly_health_check_job()` in main.py: logs MAE, predictions count, bets, drawdown. Warns if MAE > 3 pts.\n'
        '5. Add health check job to APScheduler at 4:30 AM ET (after daily_snapshot at 4 AM).\n'
        '6. Update HEARTBEAT.md: Nightly Health Check -> LIVE.\n'
        '7. py_compile + 481 tests. Commit.\n'
        '8. Update HANDOFF.md A-30 to COMPLETE. Advance to EMAC-054.\n\n'
        'Constraints: Single file (main.py). No DB schema changes.'
    ),
    # 7. Hive wisdom - add two new entries
    (
        '| `sd_mult=1.0` widens distribution, compresses edges. V9 recal after 50+ V9 bets settle. | K-3 |',
        '| `sd_mult=1.0` widens distribution, compresses edges. V9 recal after 50+ V9 bets settle. | K-3 |\n'
        '| Gemini O-10 (G-15) and Claude O-10 are COMPLEMENTARY: G-15=DB-driven position monitor (30m), Claude=event-driven in-memory golden-window check. Both needed. | EMAC-052 |\n'
        '| Dead imports in try/except blocks are invisible failures. Remove them -- do not paper over with broader except. | EMAC-052 |'
    ),
    # 8. Claude handoff prompt
    (
        'MISSION: EMAC-052 \u2014 Monitoring mode. No active code missions.\n'
        'Working directory: C:\\Users\\sfgra\\repos\\Fixed\\cbb-edge\n\n'
        'STATE: 481/481 tests. Railway live. V9 fully deployed.\n'
        'O-10 Line Movement Monitor LIVE. Runs every 30m.\n'
        'A-27 calibration review COMPLETE \u2014 see memory/calibration.md.\n'
        'Parameters frozen: ha=2.419, sd_mult=1.0. V9 recal pending (need 50 settled V9 bets).\n\n'
        'GUARDIAN: py_compile + 481 tests before approving any Gemini commit.',
        'MISSION: EMAC-053 \u2014 A-30: Wire Nightly Health Check APScheduler + morning briefing audit.\n'
        'Working directory: C:\\Users\\sfgra\\repos\\Fixed\\cbb-edge\n\n'
        'STATE: 481/481 tests. Railway live. V9 fully deployed.\n'
        'A-29 COMPLETE (dead .openclaw import removed from analysis.py).\n'
        'O-10 LIVE (G-15 DB-driven + Claude event-driven golden-window).\n'
        'Parameters frozen: ha=2.419, sd_mult=1.0. V9 recal pending (need 50 settled V9 bets).\n\n'
        'NEXT: Implement A-30 (see Section 4). py_compile + 481 tests before commit.\n'
        'GUARDIAN: py_compile + 481 tests before approving any Gemini commit.'
    ),
    # 9. Gemini handoff prompt
    (
        'MISSION: G-16 \u2014 Post-Deploy Verification of O-10\n'
        '1. Verify line_monitor job is scheduled in /admin/scheduler/status.\n'
        '2. Check Railway logs for "Starting check_line_movements job".\n'
        '3. Verify Discord bot receives line monitor alerts if movement occurs.\n\n'
        'Update HANDOFF.md G-16 to COMPLETE. Advance title to EMAC-053.',
        'MISSION: G-16 \u2014 Post-Deploy Verification of O-10 (still open)\n'
        'Working directory: C:\\Users\\sfgra\\repos\\Fixed\\cbb-edge\n'
        '1. Verify line_monitor job is scheduled in /admin/scheduler/status.\n'
        '2. Check Railway logs for "Starting check_line_movements job".\n'
        '3. Verify Discord bot receives line monitor alerts if movement occurs.\n\n'
        'Update HANDOFF.md G-16 to COMPLETE. Advance title to EMAC-054.'
    ),
    # 10. Kimi handoff prompt
    (
        'MISSION: K-6 \u2014 Design O-8 Pre-Tournament Baseline Script\n'
        'You are Deep Intelligence Unit AND OpenClaw Config Owner for CBB Edge Analyzer.\n'
        'Read: backend/services/scout.py, reports/openclaw-capabilities-assessment.md, HEARTBEAT.md.\n'
        'Design scripts/openclaw_baseline.py for OpenClaw to execute March 16 ~9 PM ET.\n'
        'Output: 68-team JSON map (team -> seed, region, risk_level, summary). Use DDGS + qwen2.5:3b.\n'
        'Save spec to reports/k6-o8-baseline-spec.md.\n'
        'Update HANDOFF.md K-6 to COMPLETE. Advance title to EMAC-053.',
        'MISSION: K-7 \u2014 Design A-30 Nightly Health Check thresholds\n'
        'Working directory: C:\\Users\\sfgra\\repos\\Fixed\\cbb-edge\n'
        'K-6 COMPLETE. scripts/openclaw_baseline.py ready for March 16.\n\n'
        'Review HEARTBEAT.md Nightly Health Check spec.\n'
        'Read backend/services/performance.py for available metrics (MAE, ROI, etc.).\n'
        'Recommend thresholds for _nightly_health_check_job:\n'
        '  - MAE warning threshold (currently proposed 3.0 pts)\n'
        '  - Drawdown warning vs halt levels\n'
        '  - Min predictions per night for a meaningful check\n\n'
        'Output: reports/k7-health-check-thresholds.md\n'
        'Update HANDOFF.md K-7 to COMPLETE. Advance to EMAC-054.'
    ),
    # 11. OpenClaw handoff prompt
    (
        'MISSION: O-6 \u2014 V9 Integrity Spot-Check\n'
        'GET https://cbbbetting-production.up.railway.app/api/predictions/today\n'
        'Header: X-API-Key: <your key>\n'
        'Check: is integrity_verdict populated in any prediction?\n'
        'Expected: all null (0 BET-tier games = sweep not triggered = correct).\n'
        'Report: "O-6: Not triggered \u2014 correct" or "O-6: BROKEN \u2014 escalate to Kimi".\n'
        'Update HEARTBEAT.md status tracker row for O-6.\n'
        'Update HANDOFF.md O-6 row to COMPLETE. Advance title to EMAC-053.',
        'MISSION: O-8 \u2014 Pre-Tournament Baseline Execution (March 16 ~9 PM ET)\n'
        'O-6 is COMPLETE.\n\n'
        'Prerequisites: Ollama running with qwen2.5:3b on this host.\n'
        'Execute: python scripts/openclaw_baseline.py --year 2026\n\n'
        'Output: data/pre_tournament_baseline_2026.json + reports/o8_baseline_summary_2026.md\n'
        'Flag any teams marked HIGH risk.\n'
        'Update HANDOFF.md O-8 to COMPLETE after execution.'
    ),
]

for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        print(f'  OK: replaced {repr(old[:60])}...')
    else:
        print(f'  MISS: could not find {repr(old[:60])}...')

with open('HANDOFF.md', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done.')
