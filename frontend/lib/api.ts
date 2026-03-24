/**
 * API client for the CBB Edge backend.
 *
 * Reads the API key from the `cbb_api_key` cookie (set at login) and
 * attaches it as the X-API-Key header on every request.
 *
 * Base URL is configured via NEXT_PUBLIC_API_URL (see .env.local.example).
 */

import Cookies from 'js-cookie'
import type {
  BetLog,
  ClvBetEntry,
  CalibrationBucket,
  Alert,
  LiveAlert,
  TodaysPredictionsResponse,
  OddsMonitorStatus,
  PortfolioStatusFull,
  BracketProjection,
  FantasyDraftBoardResponse,
  DraftSession,
  CreateDraftSessionResponse,
  RecordPickResponse,
  SchedulerStatus,
  RatingsStatus,
  DailyLineupResponse,
  WaiverWireResponse,
  RosterResponse,
  MatchupResponse,
  LineupApplyPlayer,
} from '@/lib/types'

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

export function getApiKey(): string {
  return Cookies.get('cbb_api_key') ?? ''
}

export function setApiKey(key: string): void {
  Cookies.set('cbb_api_key', key, { expires: 7, sameSite: 'strict' })
}

export function clearApiKey(): void {
  Cookies.remove('cbb_api_key')
}

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': getApiKey(),
      ...options?.headers,
    },
  })
  if (!res.ok) {
    let detail = ''
    try {
      const body = await res.json()
      detail = body?.detail ?? ''
    } catch {}
    throw new Error(`${res.status}${detail ? `: ${detail}` : `: ${path}`}`)
  }
  return res.json() as Promise<T>
}

// ---------------------------------------------------------------------------
// Response shapes (used only by this module)
// ---------------------------------------------------------------------------

interface ConfidenceTier {
  count: number
  mean_clv: number | null
}

interface ClvAnalysisResponse {
  message?: string
  bets_with_clv: number
  mean_clv?: number
  median_clv?: number
  std_clv?: number | null
  positive_clv_rate?: number
  distribution?: {
    strong_negative: number
    negative: number
    neutral: number
    positive: number
    strong_positive: number
  }
  clv_by_confidence?: Record<string, ConfidenceTier>
  top_10_clv?: ClvBetEntry[]
  bottom_10_clv?: ClvBetEntry[]
  scatter_data?: Array<{
    clv_prob: number
    profit_loss_units: number
    outcome: string
    pick: string
  }>
  status?: string
  recommendation?: string
}

interface OverallStats {
  bets: number
  wins: number
  losses: number
  win_rate: number
  roi: number
  total_profit_dollars: number
  mean_clv: number | null
}

interface BetTypeStats {
  bets: number
  win_rate: number
  roi: number
}

interface RollingWindow {
  bets: number
  win_rate: number
  mean_clv: number | null
}

interface PerformanceSummaryResponse {
  message?: string
  overall?: OverallStats
  by_bet_type?: Record<string, BetTypeStats>
  by_edge_bucket?: Record<string, BetTypeStats>
  rolling_windows?: {
    last_10: RollingWindow
    last_50: RollingWindow
    last_100: RollingWindow
  }
}

interface TimelinePoint {
  date: string
  cumulative_units: number
  daily_units: number
  bets: number
}

interface PerformanceTimelineResponse {
  timeline: TimelinePoint[]
}

interface BetsResponse {
  total: number
  bets: BetLog[]
}

interface CalibrationResponse {
  message?: string
  calibration_buckets?: CalibrationBucket[]
  brier_score?: number | null
  log_loss?: number | null
}

interface AlertsResponse {
  alerts: Alert[]
  live_alerts: LiveAlert[]
  status: string
}

interface PortfolioStatus {
  drawdown_pct: number
  total_exposure_pct: number
  is_paused: boolean
  active_positions: number
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

export const endpoints = {
  clvAnalysis: () =>
    apiFetch<ClvAnalysisResponse>('/api/performance/clv-analysis'),

  performanceSummary: () =>
    apiFetch<PerformanceSummaryResponse>('/api/performance/summary'),

  performanceTimeline: (days: number) =>
    apiFetch<PerformanceTimelineResponse>(`/api/performance/timeline?days=${days}`),

  bets: (status: string, days: number) =>
    apiFetch<BetsResponse>(`/api/bets?status=${status}&days=${days}`),

  calibration: (days: number) =>
    apiFetch<CalibrationResponse>(`/api/performance/calibration?days=${days}`),

  alerts: () =>
    apiFetch<AlertsResponse>('/api/performance/alerts?include_acknowledged=true'),

  acknowledgeAlert: (id: number) =>
    apiFetch<{ message: string }>(`/admin/alerts/${id}/acknowledge`, {
      method: 'POST',
    }),

  portfolioStatus: () =>
    apiFetch<PortfolioStatus>('/admin/portfolio/status'),

  // Phase 2 — Trading
  todaysPredictions: () =>
    apiFetch<TodaysPredictionsResponse>('/api/predictions/today'),

  todaysPredictionsAll: () =>
    apiFetch<TodaysPredictionsResponse>('/api/predictions/today/all'),

  oddsMonitorStatus: () =>
    apiFetch<OddsMonitorStatus>('/admin/odds-monitor/status'),

  portfolioStatusFull: () =>
    apiFetch<PortfolioStatusFull>('/admin/portfolio/status'),

  // Phase 3 — Tournament
  bracketProjection: (nSims = 10000) =>
    apiFetch<BracketProjection>(`/api/tournament/bracket-projection?n_sims=${nSims}`),

  // Fantasy Baseball — Draft Board
  fantasyDraftBoard: (params?: { position?: string; player_type?: string; tier_max?: number; limit?: number }) => {
    const qs = new URLSearchParams()
    if (params?.position) qs.set('position', params.position)
    if (params?.player_type) qs.set('player_type', params.player_type)
    if (params?.tier_max !== undefined) qs.set('tier_max', String(params.tier_max))
    if (params?.limit !== undefined) qs.set('limit', String(params.limit))
    const query = qs.toString()
    return apiFetch<FantasyDraftBoardResponse>(`/api/fantasy/draft-board${query ? `?${query}` : ''}`)
  },

  // Fantasy Baseball — Draft Session
  fantasyCreateSession: (params: { my_draft_position: number; num_teams?: number; num_rounds?: number }) => {
    const qs = new URLSearchParams()
    qs.set('my_draft_position', String(params.my_draft_position))
    if (params.num_teams !== undefined) qs.set('num_teams', String(params.num_teams))
    if (params.num_rounds !== undefined) qs.set('num_rounds', String(params.num_rounds))
    return apiFetch<CreateDraftSessionResponse>(`/api/fantasy/draft-session?${qs.toString()}`, { method: 'POST' })
  },

  fantasyRecordPick: (sessionKey: string, params: { player_id: string; drafter_position: number; is_my_pick?: boolean }) => {
    const qs = new URLSearchParams()
    qs.set('player_id', params.player_id)
    qs.set('drafter_position', String(params.drafter_position))
    if (params.is_my_pick !== undefined) qs.set('is_my_pick', String(params.is_my_pick))
    return apiFetch<RecordPickResponse>(`/api/fantasy/draft-session/${sessionKey}/pick?${qs.toString()}`, { method: 'POST' })
  },

  fantasyGetSession: (sessionKey: string) =>
    apiFetch<DraftSession>(`/api/fantasy/draft-session/${sessionKey}`),

  fantasyDeleteSession: (sessionKey: string) =>
    apiFetch<{ message: string }>(`/api/fantasy/draft-session/${sessionKey}`, { method: 'DELETE' }),

  // Fantasy Baseball — Season Ops
  dailyLineup: (date?: string) =>
    apiFetch<DailyLineupResponse>(`/api/fantasy/lineup/${date ?? new Date().toISOString().slice(0, 10)}`),

  waiverWire: () =>
    apiFetch<WaiverWireResponse>('/api/fantasy/waiver'),

  // Fantasy Baseball — Yahoo roster / matchup / lineup apply
  fantasyRoster: (): Promise<RosterResponse> =>
    apiFetch<RosterResponse>('/api/fantasy/roster'),

  fantasyMatchup: (): Promise<MatchupResponse> =>
    apiFetch<MatchupResponse>('/api/fantasy/matchup'),

  fantasyApplyLineup: (date: string, players: LineupApplyPlayer[]): Promise<{ success: boolean; applied: number; date: string }> =>
    apiFetch('/api/fantasy/lineup/apply', {
      method: 'PUT',
      body: JSON.stringify({ date, players }),
    }),

  // Admin
  schedulerStatus: () =>
    apiFetch<SchedulerStatus>('/admin/scheduler/status'),

  ratingsStatus: () =>
    apiFetch<RatingsStatus>('/admin/ratings/status'),

  featureFlags: () =>
    apiFetch<Record<string, boolean>>('/api/feature-flags'),

  setFeatureFlag: (flag: string, enabled: boolean) =>
    apiFetch<{ flag: string; enabled: boolean }>(
      `/admin/feature-flags/${flag}?enabled=${enabled}`,
      { method: 'POST' },
    ),
}
