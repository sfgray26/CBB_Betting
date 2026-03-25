// Shared TypeScript types for the CBB Edge frontend.
// Fields marked nullable (number | null) match the API ground truth in
// reports/api_ground_truth.md.

// ---------------------------------------------------------------------------
// Bet history
// ---------------------------------------------------------------------------

export interface BetLog {
  id: number
  game_id: number
  matchup: string
  game_date: string
  pick: string
  bet_type: string
  odds_taken: number
  bet_size_units: number
  bet_size_dollars: number
  model_prob: number | null
  outcome: number | null
  profit_loss_dollars: number | null
  profit_loss_units: number | null
  clv_points: number | null
  clv_prob: number | null
  is_paper_trade: boolean
  timestamp: string | null
  notes: string | null
}

// ---------------------------------------------------------------------------
// CLV analysis
// ---------------------------------------------------------------------------

// clv_prob is typed nullable (number | null) for defensive null-safety even
// though the API only includes bets that have CLV data. This matches the
// pattern used for clv_points (which the API marks explicitly nullable).
export interface ClvBetEntry {
  bet_id: number
  pick: string
  game_date: string | null
  clv_prob: number | null
  clv_points: number | null
  outcome: number | null
}

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------

export interface CalibrationBucket {
  bin: string
  predicted_prob: number
  actual_win_rate: number
  count: number
}

// ---------------------------------------------------------------------------
// Alerts
// ---------------------------------------------------------------------------

export interface Alert {
  id: number
  alert_type: string
  severity: string
  message: string
  threshold: number | null
  current_value: number | null
  acknowledged: boolean
  acknowledged_at?: string | null
  created_at: string | null
}

export interface LiveAlert {
  alert_type: string
  severity: string
  message: string
  recommendation?: string | null
}

// ---------------------------------------------------------------------------
// Phase 2 — Trading
// ---------------------------------------------------------------------------

export interface GameData {
  id: number
  game_date: string
  home_team: string
  away_team: string
  is_neutral: boolean
}

export interface PredictionEntry {
  id: number
  game_id: number
  model_version: string
  prediction_date: string
  projected_margin: number | null
  edge_conservative: number | null
  recommended_units: number | null
  verdict: string
  pass_reason: string | null
  full_analysis: Record<string, unknown> | null
  game: GameData
}

export interface TodaysPredictionsResponse {
  date: string
  total_games: number
  bets_recommended: number
  predictions: PredictionEntry[]
}

export interface OddsMonitorStatus {
  active: boolean
  games_tracked: number
  last_poll: string | null
  quota_remaining: number | null
  quota_updated_at: string | null
  quota_is_low: boolean
}

export interface PortfolioStatusFull {
  current_bankroll: number
  starting_bankroll: number
  drawdown_pct: number
  total_exposure_pct: number
  is_halted: boolean
  halt_reason: string | null
  pending_positions: number
}

export interface SchedulerJob {
  id: string
  name: string
  next_run: string | null
}

export interface SchedulerStatus {
  running: boolean
  jobs: SchedulerJob[]
}

export interface RatingSourceStatus {
  teams: number
  status: 'UP' | 'DOWN' | 'DROPPED'
}

export interface RatingsStatus {
  sources: {
    kenpom: RatingSourceStatus
    barttorvik: RatingSourceStatus
    evanmiya: RatingSourceStatus & { status: 'UP' | 'DOWN' | 'DROPPED' }
    kenpom_four_factors: { teams: number }
  }
  active_count: number
  active_sources: string[]
  model_health: 'OK' | 'DEGRADED' | 'CRITICAL'
  cache_age_hours: number
}

// ---------------------------------------------------------------------------
// Phase 3 — Tournament
// ---------------------------------------------------------------------------

export interface UpsetAlert {
  team: string
  seed: number
  region: string
  r64_win_prob: number
}

export interface TeamAdvancement {
  seed: number
  region: string
  r32_pct: number
  s16_pct: number
  e8_pct: number
  f4_pct: number
  runner_up_pct: number
  champion_pct: number
}

export interface BracketProjection {
  n_sims: number
  data_source: string
  projected_champion: string | null
  projected_final_four: string[]
  upset_alerts: UpsetAlert[]
  advancement_probs: Record<string, TeamAdvancement>
  avg_upsets_per_tournament: number
  avg_championship_margin: number
}

// ---------------------------------------------------------------------------
// Fantasy Baseball
// ---------------------------------------------------------------------------

export interface FantasyPlayer {
  id: string
  name: string
  team: string
  positions: string[]
  type: 'batter' | 'pitcher'
  tier: number
  rank: number
  adp: number
  z_score: number
  proj: Record<string, number>
  cat_scores?: Record<string, number> | null
  is_keeper?: boolean
  keeper_round?: number | null
  injury_risk?: string | null
  injury_note?: string | null
  avoid?: boolean
}

export interface FantasyDraftBoardResponse {
  count: number
  players: FantasyPlayer[]
}

export interface DraftPick {
  pick_number: number
  round: number
  drafter_position: number
  is_my_pick: boolean
  is_keeper?: boolean
  player_id: string
  player_name: string
  player_team: string | null
  player_positions: string[] | null
  player_tier: number | null
  player_adp: number | null
}

export interface DraftSession {
  session_key: string
  my_draft_position: number
  num_teams: number
  num_rounds: number
  current_pick: number
  total_picks: number
  my_picks_count: number
  is_active: boolean
  picks: DraftPick[]
  my_picks: DraftPick[]
}

export interface CreateDraftSessionResponse {
  session_key: string
  my_draft_position: number
  num_teams: number
  num_rounds: number
  message: string
}

export interface RecordPickResponse {
  message: string
  pick_number: number
  player_name: string
  is_my_pick: boolean
  next_recommendations: FantasyPlayer[]
}

// ---------------------------------------------------------------------------
// Fantasy Baseball — Season Ops
// ---------------------------------------------------------------------------

export interface LineupPlayer {
  player_id: string
  name: string
  team: string
  position: string
  implied_runs: number
  park_factor: number
  lineup_score: number
  start_time: string
  opponent: string
  status: 'START' | 'BENCH' | 'UNKNOWN'
}

export interface StartingPitcher {
  player_id: string
  name: string
  team: string
  opponent_implied_runs: number
  park_factor: number
  sp_score: number
  start_time: string
  status: 'START' | 'BENCH' | 'UNKNOWN'
}

export interface DailyLineupResponse {
  date: string
  batters: LineupPlayer[]
  pitchers: StartingPitcher[]
  games_count: number
}

export interface WaiverPlayer {
  player_id: string
  name: string
  team: string
  position: string
  need_score: number
  category_contributions: Record<string, number>
  owned_pct: number
  starts_this_week: number
  statcast_signals?: string[]
}

export interface CategoryDeficit {
  category: string
  my_total: number
  opponent_total: number
  deficit: number
  winning: boolean
}

export interface WaiverWireResponse {
  week_end: string
  matchup_opponent: string
  category_deficits: CategoryDeficit[]
  top_available: WaiverPlayer[]
  two_start_pitchers: WaiverPlayer[]
  pagination?: { page: number; per_page: number; has_next: boolean }
  urgent_alert?: { type: string; player: string; position: string; message: string } | null
}

export interface WaiverRecommendation {
  action: string
  add_player: WaiverPlayer | null
  drop_player_name: string | null
  drop_player_position: string | null
  rationale: string
  need_score: number
  confidence: number
  statcast_signals: string[]
  win_prob_before: number
  win_prob_after: number
  win_prob_gain: number
  mcmc_enabled: boolean
}

export interface WaiverRecommendationsResponse {
  week_end: string
  matchup_opponent: string
  recommendations: WaiverRecommendation[]
  category_deficits: CategoryDeficit[]
}

// ---------------------------------------------------------------------------
// Fantasy Baseball — Yahoo Roster / Matchup / Lineup Apply (EMAC-076)
// ---------------------------------------------------------------------------

export interface RosterPlayer {
  player_key: string
  name: string
  team: string | null
  positions: string[]
  status: string | null
  injury_note: string | null
  z_score: number | null
  is_undroppable: boolean
}

export interface RosterResponse {
  team_key: string
  players: RosterPlayer[]
  count: number
}

export interface MatchupTeam {
  team_key: string
  team_name: string
  stats: Record<string, string | number>
}

export interface MatchupResponse {
  week: number | null
  my_team: MatchupTeam
  opponent: MatchupTeam
  is_playoffs: boolean
}

export interface LineupApplyPlayer {
  player_key: string
  position: string
}
