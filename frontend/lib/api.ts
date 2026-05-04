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
  SchedulerStatus,
  RatingsStatus,
  DashboardResponse,
  AsyncJobStatus,
  StreakPlayer,
  WaiverTarget,
  DecisionsResponse,
  DecisionPipelineStatus,
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
      const rawDetail = body?.detail
      if (typeof rawDetail === 'object' && rawDetail !== null) {
        detail = rawDetail.error || rawDetail.message || JSON.stringify(rawDetail)
      } else {
        detail = rawDetail ?? ''
      }
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

  asyncOptimizeLineup: (targetDate: string, riskTolerance = 'balanced') => {
    const params = new URLSearchParams({ target_date: targetDate, risk_tolerance: riskTolerance })
    return apiFetch<{ job_id: string; status: string; poll_url: string }>(
      `/api/fantasy/lineup/async-optimize?${params}`,
      { method: 'POST' },
    )
  },

  getJobStatus: (jobId: string) =>
    apiFetch<AsyncJobStatus>(`/api/fantasy/jobs/${jobId}`),

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

  // ═══════════════════════════════════════════════════════════════════════════
  // Phase B: Enhanced Dashboard
  // ═══════════════════════════════════════════════════════════════════════════

  /** Get full dashboard data */
  getDashboard: () =>
    apiFetch<DashboardResponse>('/api/dashboard'),

  /** Get hot/cold streaks */
  getDashboardStreaks: () =>
    apiFetch<{ success: boolean; hot_streaks: StreakPlayer[]; cold_streaks: StreakPlayer[] }>(
      '/api/dashboard/streaks'
    ),

  /** Get waiver targets */
  getDashboardWaiverTargets: () =>
    apiFetch<{ success: boolean; targets: WaiverTarget[] }>(
      '/api/dashboard/waiver-targets'
    ),

  // ═══════════════════════════════════════════════════════════════════════════
  // Fantasy Baseball Decisions (Layer 3F)
  // ═══════════════════════════════════════════════════════════════════════════

  /** Get fantasy decisions with optional filters */
  getDecisions: (params?: {
    decision_type?: 'lineup' | 'waiver'
    as_of_date?: string
    limit?: number
  }) => {
    const searchParams = new URLSearchParams()
    if (params?.decision_type) searchParams.set('decision_type', params.decision_type)
    if (params?.as_of_date) searchParams.set('as_of_date', params.as_of_date)
    if (params?.limit !== undefined) searchParams.set('limit', params.limit.toString())
    const query = searchParams.toString()
    return apiFetch<DecisionsResponse>(`/api/fantasy/decisions${query ? `?${query}` : ''}`)
  },

  /** Get decision pipeline status */
  getDecisionsStatus: () =>
    apiFetch<DecisionPipelineStatus>('/api/fantasy/decisions/status'),
}
