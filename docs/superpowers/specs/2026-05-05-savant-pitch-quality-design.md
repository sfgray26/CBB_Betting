# Savant Pitch Quality Design

> Date: 2026-05-05  
> Status: Draft for user review  
> Scope: Fantasy Baseball waiver and breakout detection

## Decision

Build `savant_pitch_quality`, a 100-centered in-house pitcher quality score derived from Baseball Savant data. The first version optimizes for waiver and breakout detection, not broad rest-of-season projection accuracy.

This replaces the blocked FanGraphs Stuff+/Location+ path as the near-term pitcher skill signal. It does not attempt to clone FanGraphs proprietary models.

## Problem

The platform needs a live pitcher skill signal for waiver decisions. FanGraphs Stuff+/Location+ is useful, but automated production access is blocked by Cloudflare/IP reputation from Railway. Baseball Savant data is already part of the platform and avoids that blocker.

The user priority is early waiver edge: identify pitchers whose underlying skill has improved before fantasy projections, ownership, and recent ratios fully adjust.

## Goals

- Detect pitcher breakouts earlier than ERA, WHIP, or ownership percentage.
- Produce an explainable 100-centered score where 100 is league average.
- Use Savant-native data that can be ingested reliably without FanGraphs.
- Feed waiver signals such as `BREAKOUT_ARM`, `SKILL_CHANGE`, `WATCHLIST`, and `STREAMER_UPSIDE`.
- Provide a Bayesian likelihood input for pitcher category projection adjustments.
- Keep production runtime independent of MCP servers.

## Non-Goals

- Do not reproduce FanGraphs Stuff+, Location+, or Pitching+.
- Do not make MCP a production dependency.
- Do not rewrite the full projection engine in the first version.
- Do not directly overwrite Steamer/RoS projections from one Statcast score.
- Do not activate user-facing scoring until backtests and sanity checks pass.

## Data Sources

Primary source: Baseball Savant / Statcast.

Candidate endpoint families:

- `pitch_arsenal_stats`
- `expected_statistics`
- `statcast_pitcher`
- percentile rankings
- pitch-level Statcast data for rolling windows where already available

MCP role:

- Use `mlb-mcp` to accelerate research, endpoint inspection, field discovery, and agent validation.
- Agents may query MCP during analysis or debugging.
- Scheduled ingestion should continue through project-owned scripts/services.

## Score Shape

`savant_pitch_quality` is a 100-centered composite:

```text
savant_pitch_quality =
  35% arsenal_quality
  30% bat_missing_skill
  20% contact_suppression
  15% command_stability
  + trend_adjustment
  x sample_confidence
```

Each component should be normalized against the current season pitcher population. The initial implementation should prefer transparent percentile/z-score normalization over opaque model weights.

## Components

### Arsenal Quality

Purpose: measure whether the pitcher has raw pitch traits and arsenal shape that can sustain skill gains.

Candidate inputs:

- fastball velocity
- pitch-level velocity by pitch type
- spin rate
- induced vertical break and horizontal movement
- extension
- pitch mix changes
- pitch-level run value where available

### Bat-Missing Skill

Purpose: identify strikeout upside and early skill changes.

Candidate inputs:

- whiff percentage
- chase percentage
- CSW percentage if available or derivable
- strikeout percentage
- swinging-strike proxy from pitch-level events

### Contact Suppression

Purpose: separate real breakout arms from pitchers allowing dangerous contact.

Candidate inputs:

- xwOBA allowed
- barrel percentage allowed
- hard-hit percentage allowed
- average exit velocity allowed
- expected slugging allowed

### Command And Stability

Purpose: penalize arms whose raw stuff is not usable because walks or poor locations create ratio risk.

Candidate inputs:

- walk percentage
- zone percentage
- edge percentage
- meatball percentage
- xERA minus ERA for regression risk

### Trend Adjustment

Purpose: make the score useful for waiver edge, where recent changes matter.

Candidate inputs over 7, 14, and 30 days:

- velocity gain or loss
- whiff percentage gain or loss
- chase percentage gain or loss
- xwOBA allowed improvement or deterioration
- pitch mix change, especially new or increased secondary pitch usage
- role or workload change where available

Trend adjustment should be bounded so a tiny recent sample cannot dominate the full score.

### Sample Confidence

Purpose: dampen noisy small samples.

Candidate inputs:

- batters faced
- pitches thrown
- innings pitched
- games started or relief appearances
- number of pitch-level observations for arsenal metrics

Low-confidence pitchers should still be eligible for `WATCHLIST`, but not high-conviction `BREAKOUT_ARM`.

## Waiver Signal Rules

Initial signals:

- `BREAKOUT_ARM`: high score, positive trend, and sufficient confidence.
- `SKILL_CHANGE`: meaningful recent improvement, even if total score is not yet elite.
- `STREAMER_UPSIDE`: above-average score plus favorable upcoming context.
- `WATCHLIST`: strong trend or arsenal change with limited sample.
- `RATIO_RISK`: good bat-missing score but poor command/contact suppression.

Ownership and market signals should modify actionability:

- low ownership plus strong skill trend increases waiver priority.
- rising ownership plus strong skill score increases urgency.
- high ownership should not suppress the signal, but it reduces waiver action value.

## Projection Integration

`savant_pitch_quality` should affect projections as a likelihood signal, not as the projection itself.

Likely first integrations:

- Increase projected strikeouts when bat-missing skill and trend are strong.
- Improve ERA/WHIP outlook when contact suppression and command are both positive.
- Improve QS probability only when role/workload confidence supports it.
- Penalize ERA/WHIP when command or contact suppression flags risk.

The projection updater should persist explanations so downstream waiver recommendations can show why a pitcher moved.

## Storage

Prefer adding a dedicated table instead of overloading `statcast_pitcher_metrics.stuff_plus`.

Candidate table:

```sql
savant_pitch_quality_scores
```

Candidate fields:

- `player_id`
- `player_name`
- `season`
- `as_of_date`
- `savant_pitch_quality`
- `arsenal_quality`
- `bat_missing_skill`
- `contact_suppression`
- `command_stability`
- `trend_adjustment`
- `sample_confidence`
- `signals`
- `inputs`
- `created_at`
- `updated_at`

This preserves the existing FanGraphs-specific columns while making the in-house metric explicit.

## Validation

Backtest and sanity checks should precede activation.

Minimum checks:

- Score distribution centers near 100.
- Known elite pitchers score above average.
- Low-sample relievers do not flood the top of the leaderboard.
- Recent breakout cases surface before ownership fully reacts.
- Ratio-risk arms are flagged when whiffs are high but walks/contact are poor.
- Score movement is explainable from component changes.

Outcome validation targets:

- next-start strikeouts
- next-start ERA/WHIP risk
- 14-day K%, BB%, xwOBA allowed
- waiver add value versus replacement-level streamers

## Feature Flags

Initial flags should default to disabled:

- `savant_pitch_quality_enabled`
- `savant_pitch_quality_waiver_signals_enabled`
- `savant_pitch_quality_projection_adjustments_enabled`

This allows ingestion and backtesting before the score changes production recommendations.

## Rollout

1. Research available Savant fields using `mlb-mcp` and existing repo scrapers.
2. Implement read-only extractor and local score calculator.
3. Add storage migration and backfill script.
4. Run historical/current-season backfill.
5. Validate distribution, coverage, and known-player sanity cases.
6. Add waiver signals behind feature flag.
7. Add Bayesian projection adjustments behind separate feature flag.

## Ownership

Claude owns final architecture and backend model changes. Codex can implement bounded scripts, calculators, tests, and DevOps verification. Kimi can audit fields and write research reports. Gemini should not write code.

## Open Decisions

- Exact Savant endpoints and field names after MCP/repo field discovery.
- Initial component thresholds for waiver signals.
- Whether score normalization should use qualified pitchers only or all pitchers with confidence damping.
- Whether relievers and starters should share one score distribution or separate baselines.
