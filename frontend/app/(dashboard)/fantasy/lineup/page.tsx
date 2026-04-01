'use client'

import { useState, useEffect, useRef, Component, type ReactNode } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { ListChecks, RefreshCw, Send, Wand2, AlertTriangle, Clock, CheckCircle2, XCircle } from 'lucide-react'
import { endpoints } from '@/lib/api'
import { Card, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'
import type { LineupPlayer, StartingPitcher, ValuationReport } from '@/lib/types'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function todayStr(): string {
  // en-CA locale produces YYYY-MM-DD; America/New_York anchors to ET
  // so West Coast users see the same "today" as the backend
  return new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' })
}

function formatTime(iso: string | null | undefined): string {
  if (!iso) return '—'
  try {
    const d = new Date(iso)
    if (isNaN(d.getTime())) return '—'
    return d.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
      timeZoneName: 'short',
    })
  } catch {
    return '—'
  }
}

function scoreColor(score: number, scores: number[]): string {
  if (scores.length === 0) return 'text-zinc-300'
  const sorted = [...scores].sort((a, b) => a - b)
  const p25 = sorted[Math.floor(sorted.length * 0.25)]
  const p75 = sorted[Math.floor(sorted.length * 0.75)]
  if (score >= p75) return 'text-emerald-400'
  if (score <= p25) return 'text-zinc-500'
  return 'text-zinc-300'
}

function slotBadge(slot: string | null | undefined) {
  if (!slot) return null
  const isBench = slot === 'BN'
  return (
    <span className={cn(
      'px-1.5 py-0.5 rounded text-xs font-mono',
      isBench
        ? 'bg-zinc-700/60 text-zinc-500 border border-zinc-600/40'
        : 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
    )}>
      {slot}
    </span>
  )
}

function statusBadge(status: LineupPlayer['status'] | StartingPitcher['status']) {
  if (status === 'START') {
    return (
      <span className="px-2 py-0.5 rounded text-xs font-semibold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
        START
      </span>
    )
  }
  if (status === 'BENCH') {
    return (
      <span className="px-2 py-0.5 rounded text-xs font-semibold bg-zinc-700/60 text-zinc-500 border border-zinc-600/40">
        BENCH
      </span>
    )
  }
  const FALLBACK_LABELS: Record<string, string> = {
    UNKNOWN: 'NO START',
    NO_START: 'NO START',
    RP: 'RELIEVER',
  }
  const label = FALLBACK_LABELS[status as string] ?? (status as string) ?? 'NO START'
  return (
    <span className="px-2 py-0.5 rounded text-xs font-semibold bg-zinc-800 text-zinc-600 border border-zinc-700">
      {label}
    </span>
  )
}

// ---------------------------------------------------------------------------
// Loading skeleton
// ---------------------------------------------------------------------------

function TableSkeleton({ rows = 8 }: { rows?: number }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-zinc-800 animate-pulse">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-800 bg-zinc-900/60">
            {[140, 60, 60, 80, 90, 80, 80, 70].map((w, i) => (
              <th key={i} className="px-3 py-3">
                <div className="h-3 bg-zinc-800 rounded" style={{ width: w }} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-800/60">
          {Array.from({ length: rows }).map((_, i) => (
            <tr key={i}>
              {[140, 60, 60, 80, 90, 80, 80, 70].map((w, j) => (
                <td key={j} className="px-3 py-3">
                  <div className="h-3 bg-zinc-800/70 rounded" style={{ width: w * 0.8 }} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Valuation cache banner (Phase 3a)
// ---------------------------------------------------------------------------

function ValuationCacheBanner({
  status,
  count,
  targetDate,
}: {
  status: 'fresh' | 'stale' | 'empty' | 'error' | undefined
  count: number
  targetDate: string
}) {
  if (!status || status === 'fresh') return null

  if (status === 'stale') {
    return (
      <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 px-4 py-2 flex items-center gap-2 text-xs text-amber-400/80">
        <Clock className="h-3.5 w-3.5 flex-shrink-0" />
        Projections cached from a previous run ({count} players) &mdash; live update expected at 6 AM ET
      </div>
    )
  }
  if (status === 'empty') {
    return (
      <div className="rounded-lg border border-zinc-700/40 bg-zinc-800/30 px-4 py-2 flex items-center gap-2 text-xs text-zinc-500">
        <Clock className="h-3.5 w-3.5 flex-shrink-0" />
        No projections yet for {targetDate} &mdash; scores are based on real-time data
      </div>
    )
  }
  return null
}

// ---------------------------------------------------------------------------
// Async optimize status bar (Phase 3b)
// ---------------------------------------------------------------------------

type OptimizeState = 'idle' | 'submitting' | 'queued' | 'processing' | 'completed' | 'failed' | 'timeout'

function OptimizeStatusBar({ state, error }: { state: OptimizeState; error?: string }) {
  if (state === 'idle') return null

  const configs: Record<OptimizeState, { icon: ReactNode; text: string; cls: string }> = {
    idle: { icon: null, text: '', cls: '' },
    submitting: {
      icon: <RefreshCw className="h-3.5 w-3.5 animate-spin" />,
      text: 'Submitting optimization job...',
      cls: 'border-blue-500/30 bg-blue-500/10 text-blue-400',
    },
    queued: {
      icon: <Clock className="h-3.5 w-3.5" />,
      text: 'Queued — waiting to process...',
      cls: 'border-blue-500/30 bg-blue-500/10 text-blue-400',
    },
    processing: {
      icon: <RefreshCw className="h-3.5 w-3.5 animate-spin" />,
      text: 'Optimizing lineup...',
      cls: 'border-amber-500/30 bg-amber-500/10 text-amber-400',
    },
    completed: {
      icon: <CheckCircle2 className="h-3.5 w-3.5" />,
      text: 'Optimization complete — lineup updated',
      cls: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400',
    },
    failed: {
      icon: <XCircle className="h-3.5 w-3.5" />,
      text: error ?? 'Optimization failed',
      cls: 'border-rose-500/30 bg-rose-500/10 text-rose-400',
    },
    timeout: {
      icon: <AlertTriangle className="h-3.5 w-3.5" />,
      text: 'Taking longer than expected — check back in a minute',
      cls: 'border-amber-500/30 bg-amber-500/10 text-amber-400',
    },
  }

  const cfg = configs[state]
  if (!cfg.text) return null

  return (
    <div className={cn('rounded-lg border px-4 py-2 flex items-center gap-2 text-xs', cfg.cls)}>
      {cfg.icon}
      {cfg.text}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Error boundary (Phase 3c)
// ---------------------------------------------------------------------------

interface ErrorBoundaryState { hasError: boolean; message: string }

class LineupErrorBoundary extends Component<
  { children: ReactNode; fallbackLabel: string },
  ErrorBoundaryState
> {
  constructor(props: { children: ReactNode; fallbackLabel: string }) {
    super(props)
    this.state = { hasError: false, message: '' }
  }

  static getDerivedStateFromError(err: unknown): ErrorBoundaryState {
    return { hasError: true, message: err instanceof Error ? err.message : 'Unknown error' }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="rounded-lg border border-zinc-700/40 bg-zinc-800/30 p-6 text-center space-y-2">
          <p className="text-zinc-400 text-sm">{this.props.fallbackLabel} unavailable</p>
          <p className="text-zinc-600 text-xs">{this.state.message}</p>
          <button
            onClick={() => this.setState({ hasError: false, message: '' })}
            className="px-3 py-1 text-xs bg-zinc-700 hover:bg-zinc-600 text-zinc-300 rounded transition-colors"
          >
            Reload
          </button>
        </div>
      )
    }
    return this.props.children
  }
}

// ---------------------------------------------------------------------------
// Batters table
// ---------------------------------------------------------------------------

function BattersTable({
  batters,
  valuationsMap,
}: {
  batters: LineupPlayer[]
  valuationsMap: Map<string, ValuationReport>
}) {
  const scores = batters.map((b) => b.lineup_score)

  if (batters.length === 0) {
    return <p className="text-zinc-600 text-sm text-center py-8">No batters scheduled for this date.</p>
  }

  const sorted = [...batters].sort((a, b) => b.lineup_score - a.lineup_score)

  return (
    <div className="overflow-x-auto rounded-lg border border-zinc-800">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-800 bg-zinc-900/60">
            <th className="px-3 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider">Player</th>
            <th className="px-3 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider w-14">Team</th>
            <th className="px-3 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider w-14">Opp</th>
            <th className="px-3 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider w-24">Time</th>
            <th className="px-3 py-3 text-right text-xs font-semibold text-zinc-500 uppercase tracking-wider w-28">Implied Runs</th>
            <th className="px-3 py-3 text-right text-xs font-semibold text-zinc-500 uppercase tracking-wider w-24">Park Factor</th>
            <th className="px-3 py-3 text-right text-xs font-semibold text-zinc-500 uppercase tracking-wider w-20">Score</th>
            <th className="px-3 py-3 text-right text-xs font-semibold text-zinc-500 uppercase tracking-wider w-20">Proj</th>
            <th className="px-3 py-3 text-center text-xs font-semibold text-zinc-500 uppercase tracking-wider w-16">Slot</th>
            <th className="px-3 py-3 text-center text-xs font-semibold text-zinc-500 uppercase tracking-wider w-24">Action</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-800/60">
          {sorted.map((b) => {
            const isIL = !!(b.injury_status && /^IL|^OUT$/i.test(b.injury_status))
            const val = valuationsMap.get(b.player_id)
            const projValue = val?.report?.composite_value?.point_estimate
            const formDelta = val?.report?.recent_form_delta ?? 0
            return (
              <tr key={b.player_id} className={cn('hover:bg-zinc-800/50 transition-colors', isIL && 'opacity-60')}>
                <td className="px-3 py-2.5">
                  <div className="flex items-center gap-2">
                    <span className={cn('font-medium', isIL ? 'text-zinc-400' : 'text-zinc-100')}>{b.name}</span>
                    {isIL && (
                      <span className="px-1.5 py-0.5 rounded text-xs font-semibold bg-rose-500/15 text-rose-400 border border-rose-500/30">
                        {b.injury_status!.toUpperCase()}
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-zinc-500">{b.position}</div>
                </td>
                <td className="px-3 py-2.5 text-zinc-400 font-mono text-xs">{b.team}</td>
                <td className="px-3 py-2.5 text-zinc-500 font-mono text-xs">{b.opponent}</td>
                <td className="px-3 py-2.5 text-zinc-500 text-xs tabular-nums">{formatTime(b.start_time)}</td>
                <td className="px-3 py-2.5 text-right font-mono text-xs text-zinc-300 tabular-nums">{b.implied_runs.toFixed(2)}</td>
                <td className="px-3 py-2.5 text-right font-mono text-xs text-zinc-400 tabular-nums">{b.park_factor.toFixed(3)}</td>
                <td className={cn('px-3 py-2.5 text-right font-mono text-xs font-semibold tabular-nums', scoreColor(b.lineup_score, scores))}>
                  {b.lineup_score.toFixed(3)}
                </td>
                <td className="px-3 py-2.5 text-right tabular-nums">
                  {projValue != null ? (
                    <span className={cn(
                      'font-mono text-xs font-semibold',
                      formDelta > 0.1 ? 'text-emerald-400' : formDelta < -0.1 ? 'text-rose-400' : 'text-zinc-400'
                    )}>
                      {projValue.toFixed(2)}
                    </span>
                  ) : (
                    <span className="text-zinc-700 text-xs">—</span>
                  )}
                </td>
                <td className="px-3 py-2.5 text-center">{slotBadge(b.assigned_slot)}</td>
                <td className="px-3 py-2.5 text-center">{statusBadge(b.status)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Pitchers table
// ---------------------------------------------------------------------------

function PitchersTable({ pitchers }: { pitchers: StartingPitcher[] }) {
  const spOnly = pitchers.filter((p) => p.pitcher_type === 'SP')
  const scores = spOnly.map((p) => p.sp_score)

  if (spOnly.length === 0) {
    return <p className="text-zinc-600 text-sm text-center py-8">No starting pitchers scheduled for this date.</p>
  }

  const sorted = [...spOnly].sort((a, b) => b.sp_score - a.sp_score)

  return (
    <div className="overflow-x-auto rounded-lg border border-zinc-800">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-800 bg-zinc-900/60">
            <th className="px-3 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider">Pitcher</th>
            <th className="px-3 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider w-14">Team</th>
            <th className="px-3 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider w-24">Time</th>
            <th className="px-3 py-3 text-right text-xs font-semibold text-zinc-500 uppercase tracking-wider w-28">Opp Implied</th>
            <th className="px-3 py-3 text-right text-xs font-semibold text-zinc-500 uppercase tracking-wider w-24">Park Factor</th>
            <th className="px-3 py-3 text-right text-xs font-semibold text-zinc-500 uppercase tracking-wider w-24">SP Score</th>
            <th className="px-3 py-3 text-center text-xs font-semibold text-zinc-500 uppercase tracking-wider w-24">Action</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-800/60">
          {sorted.map((p) => (
            <tr key={p.player_id} className="hover:bg-zinc-800/50 transition-colors">
              <td className="px-3 py-2.5">
                <div className="font-medium text-zinc-100">{p.name}</div>
              </td>
              <td className="px-3 py-2.5 text-zinc-400 font-mono text-xs">{p.team}</td>
              <td className="px-3 py-2.5 text-zinc-500 text-xs tabular-nums">{formatTime(p.start_time)}</td>
              <td className="px-3 py-2.5 text-right font-mono text-xs text-zinc-300 tabular-nums">{p.opponent_implied_runs.toFixed(2)}</td>
              <td className="px-3 py-2.5 text-right font-mono text-xs text-zinc-400 tabular-nums">{p.park_factor.toFixed(3)}</td>
              <td className={cn('px-3 py-2.5 text-right font-mono text-xs font-semibold tabular-nums', scoreColor(p.sp_score, scores))}>
                {p.sp_score === 0 ? '—' : p.sp_score.toFixed(3)}
              </td>
              <td className="px-3 py-2.5 text-center">{statusBadge(p.status)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function DailyLineupPage() {
  const [date, setDate] = useState<string>(todayStr())
  const [applyStatus, setApplyStatus] = useState<'idle' | 'success' | 'error'>('idle')
  const [applyMessage, setApplyMessage] = useState<string>('')

  // Phase 3b: async optimize state machine
  const [optimizeState, setOptimizeState] = useState<OptimizeState>('idle')
  const [optimizeError, setOptimizeError] = useState<string>('')
  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pollAttemptsRef = useRef(0)

  // Main lineup query (unchanged — shows current roster + scores)
  const { data, isLoading, isError, refetch, isFetching } = useQuery({
    queryKey: ['fantasy-lineup', date],
    queryFn: () => endpoints.dailyLineup(date),
    refetchInterval: 5 * 60_000,
  })

  // Phase 3a: valuations cache query — parallel, non-blocking
  const { data: valuationsData } = useQuery({
    queryKey: ['player-valuations', date],
    queryFn: () => endpoints.getPlayerValuations(date),
    staleTime: 10 * 60_000, // 10 min — cache changes at most once per 6h worker run
    retry: false,           // never retry on error — degrade silently
  })

  // Build player_id → ValuationReport map for O(1) lookup in table rows
  const valuationsMap = new Map<string, ValuationReport>(
    (valuationsData?.valuations ?? []).map((v) => [v.player_id, v])
  )

  // Cleanup poll timer on unmount / date change
  useEffect(() => {
    return () => {
      if (pollRef.current) clearTimeout(pollRef.current)
    }
  }, [date])

  // Phase 3b: poll job status
  const pollJob = (jobId: string) => {
    if (pollAttemptsRef.current >= 30) {
      setOptimizeState('timeout')
      return
    }
    pollRef.current = setTimeout(async () => {
      try {
        const status = await endpoints.getJobStatus(jobId)
        pollAttemptsRef.current += 1
        if (status.status === 'completed') {
          setOptimizeState('completed')
          refetch() // pull updated lineup from backend cache
        } else if (status.status === 'failed') {
          setOptimizeState('failed')
          setOptimizeError(status.error ?? 'Job failed')
        } else {
          setOptimizeState(status.status === 'processing' ? 'processing' : 'queued')
          pollJob(jobId)
        }
      } catch {
        // transient fetch error — keep polling
        pollJob(jobId)
      }
    }, 2000)
  }

  const handleAsyncOptimize = async () => {
    if (pollRef.current) clearTimeout(pollRef.current)
    pollAttemptsRef.current = 0
    setOptimizeState('submitting')
    setOptimizeError('')

    try {
      const job = await endpoints.asyncOptimizeLineup(date)
      setOptimizeState('queued')
      pollJob(job.job_id)
    } catch (err) {
      setOptimizeState('failed')
      setOptimizeError(err instanceof Error ? err.message : 'Failed to submit job')
    }
  }

  const { mutate: applyLineup, isPending: isApplying } = useMutation({
    mutationFn: () => {
      const starters = [
        ...(data?.batters ?? [])
          .filter((b) => b.status === 'START')
          .map((b) => ({ player_key: b.player_id, position: b.assigned_slot ?? b.position })),
        ...(data?.pitchers ?? [])
          .filter((p) => p.status === 'START')
          .map((p) => ({ player_key: p.player_id, position: 'SP' })),
      ]
      return endpoints.fantasyApplyLineup(date, starters)
    },
    onSuccess: (result) => {
      setApplyStatus('success')
      setApplyMessage(`Applied ${result.applied} players for ${result.date}`)
    },
    onError: (err: unknown) => {
      let message = 'Failed to apply lineup'
      if (err instanceof Error) {
        message = err.message
      } else if (typeof err === 'object' && err !== null) {
        const e = err as Record<string, unknown>
        message = String(e.detail ?? e.error ?? e.message ?? message)
      }
      setApplyStatus('error')
      setApplyMessage(message)
    },
  })

  const starterCount =
    (data?.batters?.filter((b) => b.status === 'START').length ?? 0) +
    (data?.pitchers?.filter((p) => p.status === 'START' && p.pitcher_type === 'SP').length ?? 0)

  const ilStarters = (data?.batters ?? []).filter(
    (b) => b.status === 'START' && b.injury_status && /^IL|^OUT$/i.test(b.injury_status),
  )
  const hasILStarter = ilStarters.length > 0

  const isOptimizing = ['submitting', 'queued', 'processing'].includes(optimizeState)

  return (
    <div className="space-y-6 max-w-6xl">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-xl font-semibold text-zinc-100 flex items-center gap-2">
            <ListChecks className="h-5 w-5 text-amber-400" />
            Daily Lineup
          </h1>
          <p className="text-sm text-zinc-500 mt-0.5">
            Implied-runs x park-factor scoring &mdash; refreshes every 5 min
            {data && ` · ${data.games_count} games`}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <input
            type="date"
            value={date}
            onChange={(e) => {
              setDate(e.target.value)
              setOptimizeState('idle')
            }}
            className="px-3 py-1.5 text-sm bg-zinc-800 border border-zinc-700 rounded-md text-zinc-200 focus:outline-none focus:border-amber-500"
          />
          <button
            onClick={handleAsyncOptimize}
            disabled={isOptimizing || isFetching}
            className="flex items-center gap-1.5 px-3 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-500 rounded-md transition-colors disabled:opacity-50"
          >
            <Wand2 className={cn('h-4 w-4', isOptimizing && 'animate-spin')} />
            {isOptimizing ? 'Optimizing...' : 'Optimize Lineup'}
          </button>
          <button
            onClick={() => refetch()}
            disabled={isFetching}
            className="flex items-center gap-1.5 px-3 py-2 text-sm text-zinc-400 hover:text-zinc-200 bg-zinc-800 hover:bg-zinc-700 rounded-md transition-colors disabled:opacity-50"
          >
            <RefreshCw className={cn('h-4 w-4', isFetching && 'animate-spin')} />
          </button>
        </div>
      </div>

      {/* Phase 3a: valuation cache status banner */}
      <ValuationCacheBanner
        status={valuationsData?.cache_status}
        count={valuationsData?.count ?? 0}
        targetDate={date}
      />

      {/* Phase 3b: async optimize progress */}
      <OptimizeStatusBar state={optimizeState} error={optimizeError} />

      {/* Error state */}
      {isError && (
        <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-5 flex items-center justify-between">
          <div>
            <p className="text-rose-400 font-medium text-sm">Failed to load lineup data</p>
            <p className="text-rose-400/60 text-xs mt-0.5">
              The backend endpoint may not be available yet.
            </p>
          </div>
          <button
            onClick={() => refetch()}
            className="px-3 py-1.5 text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-md transition-colors"
          >
            Retry
          </button>
        </div>
      )}

      {/* Lineup warnings */}
      {data?.lineup_warnings && data.lineup_warnings.length > 0 && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-4 flex items-start gap-3">
          <AlertTriangle className="h-4 w-4 text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            {data.lineup_warnings
              .filter(w => !w.includes('validation error') && !w.includes('Traceback'))
              .map((w, i) => (
                <p key={i} className="text-amber-300 text-sm">{w}</p>
              ))}
          </div>
        </div>
      )}

      {/* Phase 3c: Error boundary wraps each table independently */}
      <LineupErrorBoundary fallbackLabel="Batters">
        <Card className="p-0">
          <CardHeader className="px-5 pt-5 pb-0 mb-4">
            <CardTitle className="text-base">Batters</CardTitle>
          </CardHeader>
          <div className="px-5 pb-5">
            {isLoading ? (
              <TableSkeleton rows={10} />
            ) : isError ? null : data && data.batters.length === 0 ? (
              <p className="text-zinc-600 text-sm text-center py-8">No games scheduled for this date.</p>
            ) : data ? (
              <BattersTable batters={data.batters} valuationsMap={valuationsMap} />
            ) : null}
          </div>
        </Card>
      </LineupErrorBoundary>

      <LineupErrorBoundary fallbackLabel="Starting Pitchers">
        <Card className="p-0">
          <CardHeader className="px-5 pt-5 pb-0 mb-4">
            <CardTitle className="text-base">Starting Pitchers</CardTitle>
          </CardHeader>
          <div className="px-5 pb-5">
            {isLoading ? (
              <TableSkeleton rows={6} />
            ) : isError ? null : data && data.pitchers.length === 0 ? (
              <p className="text-zinc-600 text-sm text-center py-8">No games scheduled for this date.</p>
            ) : data ? (
              <PitchersTable pitchers={data.pitchers} />
            ) : null}
          </div>
        </Card>
      </LineupErrorBoundary>

      {/* Apply to Yahoo section */}
      {data && (
        <div className="flex flex-col gap-3">
          {hasILStarter && (
            <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 flex items-start gap-2">
              <AlertTriangle className="h-4 w-4 text-rose-400 flex-shrink-0 mt-0.5" />
              <p className="text-rose-300 text-sm">
                Cannot apply: {ilStarters.map((b) => `${b.name} (${b.injury_status})`).join(', ')}{' '}
                {ilStarters.length === 1 ? 'is' : 'are'} on IL and cannot start. Remove from active slots before applying.
              </p>
            </div>
          )}
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
            <button
              onClick={() => {
                setApplyStatus('idle')
                applyLineup()
              }}
              disabled={isApplying || starterCount === 0 || hasILStarter}
              title={hasILStarter ? 'IL players in active slots — remove before applying' : starterCount === 0 ? 'No starters to apply' : undefined}
              className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white text-sm rounded-md font-medium min-h-[44px] transition-colors disabled:cursor-not-allowed"
            >
              {isApplying ? (
                <RefreshCw className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
              {isApplying ? 'Applying...' : `Apply to Yahoo (${starterCount} starters)`}
            </button>

            {applyStatus === 'success' && (
              <p className="text-emerald-400 text-sm">{applyMessage}</p>
            )}
            {applyStatus === 'error' && (
              <p className="text-rose-400 text-sm">{applyMessage}</p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
