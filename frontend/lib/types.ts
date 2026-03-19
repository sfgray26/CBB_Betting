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
