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

export interface AsyncJobStatus {
  job_id: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  result?: unknown
  error?: string
}

// ═════════════════════════════════════════════════════════════════════════════
// Phase B: Enhanced Dashboard Types
// ═════════════════════════════════════════════════════════════════════════════

export interface LineupGap {
  position: string
  severity: "critical" | "warning" | "info"
  message: string
  suggested_add?: string | null
}

export interface StreakPlayer {
  player_id: string
  name: string
  team: string
  positions: string[]
  trend: "hot" | "cold" | "neutral"
  trend_score: number
  last_7_avg: number
  last_14_avg: number
  last_30_avg: number
  reason: string
}

export interface WaiverTarget {
  player_id: string
  name: string
  team: string
  positions: string[]
  percent_owned: number
  priority_score: number
  tier: "must_add" | "strong_add" | "streamer"
  reason: string
}

export interface InjuryFlag {
  player_id: string
  name: string
  status: "IL" | "IL10" | "IL60" | "DTD" | "OUT"
  injury_note?: string | null
  severity: "critical" | "warning" | "info"
  estimated_return?: string | null
  action_needed: string
}

export interface MatchupPreviewData {
  week_number: number
  opponent_team_name: string
  opponent_record: string
  my_projected_categories: Record<string, number>
  opponent_projected_categories: Record<string, number>
  win_probability: number
  category_advantages: string[]
  category_disadvantages: string[]
}

export interface ProbablePitcherInfo {
  name: string
  team: string
  opponent: string
  game_date: string
  is_two_start: boolean
  matchup_quality: "favorable" | "neutral" | "unfavorable"
  stream_score: number
  reason: string
}

export interface DashboardData {
  timestamp: string
  user_id: string
  lineup_gaps: LineupGap[]
  lineup_filled_count: number
  lineup_total_count: number
  hot_streaks: StreakPlayer[]
  cold_streaks: StreakPlayer[]
  waiver_targets: WaiverTarget[]
  injury_flags: InjuryFlag[]
  healthy_count: number
  injured_count: number
  matchup_preview: MatchupPreviewData | null
  probable_pitchers: ProbablePitcherInfo[]
  two_start_pitchers: ProbablePitcherInfo[]
}

export interface DashboardResponse {
  success: boolean
  timestamp: string
  data: DashboardData
}
