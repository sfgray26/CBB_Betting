# Detailed Data Quality Audit Summary

Generated: 2026-05-16 13:21:22 EDT

## Issues Found: 44

- EMPTY: alerts
- ERROR: backtest_results - 'datetime.date' object has no attribute 'tzinfo'
- STALE: bet_logs (36 days, latest: 2026-04-09 18:43:03.221211)
- ERROR: canonical_projections - 'datetime.date' object has no attribute 'tzinfo'
- NO_TIMESTAMP: category_impacts (38382 rows)
- EMPTY: closing_lines
- ERROR: daily_snapshots - 'datetime.date' object has no attribute 'tzinfo'
- ERROR: data_ingestion_logs - 'datetime.date' object has no attribute 'tzinfo'
- ERROR: decision_explanations - 'datetime.date' object has no attribute 'tzinfo'
- ERROR: decision_results - 'datetime.date' object has no attribute 'tzinfo'
- EMPTY: deployment_version
- EMPTY: divergence_flags
- EMPTY: execution_decisions
- EMPTY: fantasy_draft_picks
- EMPTY: fantasy_draft_sessions
- ERROR: fantasy_lineups - 'datetime.date' object has no attribute 'tzinfo'
- NO_TIMESTAMP: feature_flags (12 rows)
- STALE: games (36 days, latest: 2026-04-09 18:43:03.047875)
- EMPTY: job_queue
- EMPTY: matchup_context
- ERROR: mlb_game_log - 'datetime.date' object has no attribute 'tzinfo'
- ERROR: mlb_player_stats - 'datetime.date' object has no attribute 'tzinfo'
- STALE: mlb_team (40 days, latest: 2026-04-06 05:00:00.449456+00:00)
- EMPTY: model_parameters
- EMPTY: pattern_detection_alerts
- ERROR: player_daily_metrics - 'datetime.date' object has no attribute 'tzinfo'
- ERROR: player_market_signals - 'datetime.date' object has no attribute 'tzinfo'
- ERROR: player_momentum - 'datetime.date' object has no attribute 'tzinfo'
- ERROR: player_opportunity - 'datetime.date' object has no attribute 'tzinfo'
- ERROR: player_rolling_stats - 'datetime.date' object has no attribute 'tzinfo'
- ERROR: player_scores - 'datetime.date' object has no attribute 'tzinfo'
- EMPTY: player_valuation_cache
- ERROR: predictions - 'datetime.date' object has no attribute 'tzinfo'
- ERROR: probable_pitchers - 'datetime.date' object has no attribute 'tzinfo'
- ERROR: projection_snapshots - 'datetime.date' object has no attribute 'tzinfo'
- ERROR: savant_pitch_quality_scores - 'datetime.date' object has no attribute 'tzinfo'
- ERROR: simulation_results - 'datetime.date' object has no attribute 'tzinfo'
- ERROR: statcast_performances - 'datetime.date' object has no attribute 'tzinfo'
- EMPTY: team_profiles
- EMPTY: threshold_audit
- EMPTY: weather_forecasts
- NULLS: games.external_id has 4 nulls (100.0%)
- ORPHAN: 2 games have no predictions
- FETCH_FAILURES: odds_api_scores has 61 failures in 30 days
