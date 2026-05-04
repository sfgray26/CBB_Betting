# Claude Handoff Prompt — April 21 Data Quality Fixes

Review and fix the remaining live fantasy data-quality defects using the fresh April 21 probe as source of truth.

Start with these files:

- `reports/2026-04-21-production-data-quality-audit-reviewed.md`
- `reports/2026-04-21-production-data-quality-audit-fresh.md`
- `tasks/uat_findings_post_deploy_v5.md`
- `HANDOFF.md`

Treat the April 21 fresh probe files under `postman_collections/responses/*20260421_190158.json` as current truth over the older April 20 captures.

Focus only on still-live defects:

1. Roster enrichment is still badly degraded.
   - `rolling_7d/14d/15d/30d`, `ros_projection`, `row_projection`, and `game_context` are null across the roster.
   - `bdl_player_id` and `mlbam_id` are null for all roster players.
   - `season_stats` exists, but several expected batting fields such as `K_B`, `TB`, and `NSB` are still null.

2. Waiver intelligence is still mostly hollow.
   - `owned_pct = 0.0` for the whole pool.
   - `starts_this_week = 0` for the whole pool.
   - `statcast_signals = []` and `statcast_stats = null` everywhere.
   - `category_contributions` only populate for a small minority of players.

3. Universal-drop bug persists.
   - In the fresh decisions payload, every waiver decision drops `Seiya Suzuki`.

4. Waiver stat/schema quality remains broken.
   - `K_P` still carries values that do not match strikeout counts.
   - Batters still carry pitcher-only stats like `IP` and `W` in waiver output.

5. Secondary cleanup after the above:
   - impossible ROS projection narratives (`0.00 ERA ROS`, `0.00 WHIP ROS`, etc.)
   - legacy briefing categories (`HR`, `SB`, `K`, `SV`)

Working constraints:

- Do not chase older April 20 route crashes unless they still reproduce in the April 21 response set.
- Fix root causes, not placeholders.
- Start from the owning read/enrichment paths.

Likely code surfaces:

- `backend/routers/fantasy.py`
- `backend/services/player_mapper.py`
- `backend/services/waiver_edge_detector.py`
- any helpers used to hydrate Yahoo stats, rolling stats, player ID mappings, and waiver recommendation logic

Validation requirements:

- Add focused regression tests for each repaired live defect.
- Run narrow validation first.
- Update `HANDOFF.md` with post-fix current truth and any residual risks.

Deliverable goal:

- move the live system from “routes up but data thin” to “roster and waiver outputs are actually trustworthy enough for user decisions.”