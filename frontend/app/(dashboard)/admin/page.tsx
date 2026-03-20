'use client'

import { useQuery } from '@tanstack/react-query'
import { formatDistanceToNow, parseISO } from 'date-fns'
import { ShieldAlert, RefreshCw, CheckCircle2, AlertTriangle, XCircle, Clock, Database, Cpu, Radio } from 'lucide-react'
import { endpoints } from '@/lib/api'
import { Card, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function statusDot(ok: boolean, warn?: boolean) {
  return (
    <span className={cn(
      'inline-block w-2 h-2 rounded-full flex-shrink-0',
      warn ? 'bg-amber-400' : ok ? 'bg-emerald-400' : 'bg-rose-500',
    )} />
  )
}

function drawdownColor(pct: number) {
  if (pct >= 15) return 'text-rose-400'
  if (pct >= 10) return 'text-amber-400'
  if (pct >= 5)  return 'text-yellow-400'
  return 'text-emerald-400'
}

function formatNextRun(iso: string | null): string {
  if (!iso) return 'Not scheduled'
  try {
    return formatDistanceToNow(parseISO(iso), { addSuffix: true })
  } catch {
    return iso
  }
}

function healthBadge(health: string) {
  const cfg = {
    OK:       { cls: 'text-emerald-400 bg-emerald-400/10 border-emerald-500/30', icon: <CheckCircle2 className="h-3 w-3" /> },
    DEGRADED: { cls: 'text-amber-400 bg-amber-400/10 border-amber-500/30',   icon: <AlertTriangle className="h-3 w-3" /> },
    CRITICAL: { cls: 'text-rose-400 bg-rose-400/10 border-rose-500/30',      icon: <XCircle className="h-3 w-3" /> },
  }[health] ?? { cls: 'text-zinc-400 bg-zinc-700 border-zinc-600', icon: null }

  return (
    <span className={cn('flex items-center gap-1 px-2 py-0.5 rounded border text-xs font-semibold', cfg.cls)}>
      {cfg.icon}
      {health}
    </span>
  )
}

// ---------------------------------------------------------------------------
// Portfolio panel
// ---------------------------------------------------------------------------

function PortfolioPanel() {
  const { data, isLoading, isError, refetch, isFetching } = useQuery({
    queryKey: ['portfolio-full'],
    queryFn: endpoints.portfolioStatusFull,
    refetchInterval: 30_000,
  })

  const dd = data?.drawdown_pct ?? 0
  const exp = data?.total_exposure_pct ?? 0
  const barW = Math.min(dd, 100)

  return (
    <Card className="p-0">
      <CardHeader className="px-5 pt-5 pb-0 mb-4">
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <Database className="h-4 w-4 text-zinc-500" />
            Portfolio
          </span>
          <button onClick={() => refetch()} disabled={isFetching} className="text-zinc-600 hover:text-zinc-400 disabled:opacity-40">
            <RefreshCw className={cn('h-3.5 w-3.5', isFetching && 'animate-spin')} />
          </button>
        </CardTitle>
      </CardHeader>
      <div className="px-5 pb-5 space-y-4">
        {isLoading ? (
          <div className="space-y-3 animate-pulse">
            {[1, 2, 3].map(i => <div key={i} className="h-8 bg-zinc-800 rounded" />)}
          </div>
        ) : isError ? (
          <p className="text-rose-400 text-sm">Failed to load — admin key required.</p>
        ) : data ? (
          <>
            {/* Halted banner */}
            {data.is_halted && (
              <div className="rounded-md border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-400 flex items-center gap-2">
                <XCircle className="h-4 w-4 flex-shrink-0" />
                HALTED: {data.halt_reason ?? 'Drawdown threshold exceeded'}
              </div>
            )}

            {/* Drawdown gauge */}
            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-zinc-500">Drawdown</span>
                <span className={cn('font-mono font-semibold', drawdownColor(dd))}>
                  {dd.toFixed(1)}%
                </span>
              </div>
              <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className={cn(
                    'h-full rounded-full transition-all duration-500',
                    dd >= 15 ? 'bg-rose-500' : dd >= 10 ? 'bg-amber-400' : dd >= 5 ? 'bg-yellow-400' : 'bg-emerald-400',
                  )}
                  style={{ width: `${barW}%` }}
                />
              </div>
              <div className="flex justify-between text-xs text-zinc-700 mt-0.5">
                <span>0%</span><span>Warn 10%</span><span>Halt 15%</span>
              </div>
            </div>

            {/* Stats grid */}
            <div className="grid grid-cols-2 gap-3 text-sm">
              {[
                { label: 'Bankroll', value: `$${data.current_bankroll.toFixed(0)}` },
                { label: 'Starting', value: `$${data.starting_bankroll.toFixed(0)}` },
                { label: 'Exposure', value: `${exp.toFixed(1)}%` },
                { label: 'Positions', value: String(data.pending_positions) },
              ].map(({ label, value }) => (
                <div key={label} className="rounded-md bg-zinc-800/50 border border-zinc-700/50 px-3 py-2">
                  <div className="text-xs text-zinc-500">{label}</div>
                  <div className="font-mono font-semibold text-zinc-100">{value}</div>
                </div>
              ))}
            </div>
          </>
        ) : null}
      </div>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Ratings panel
// ---------------------------------------------------------------------------

function RatingsPanel() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['ratings-status'],
    queryFn: endpoints.ratingsStatus,
    refetchInterval: 5 * 60_000,
  })

  const sources = [
    { key: 'kenpom',     label: 'KenPom',     weight: '51%' },
    { key: 'barttorvik', label: 'BartTorvik',  weight: '49%' },
    { key: 'evanmiya',   label: 'EvanMiya',    weight: '—' },
  ] as const

  return (
    <Card className="p-0">
      <CardHeader className="px-5 pt-5 pb-0 mb-4">
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <Cpu className="h-4 w-4 text-zinc-500" />
            Rating Sources
          </span>
          {data && healthBadge(data.model_health)}
        </CardTitle>
      </CardHeader>
      <div className="px-5 pb-5 space-y-3">
        {isLoading ? (
          <div className="space-y-2 animate-pulse">
            {[1, 2, 3].map(i => <div key={i} className="h-10 bg-zinc-800 rounded" />)}
          </div>
        ) : isError ? (
          <p className="text-rose-400 text-sm">Failed to load — admin key required.</p>
        ) : data ? (
          <>
            {sources.map(({ key, label, weight }) => {
              const src = data.sources[key]
              const isUp = src.status === 'UP'
              const isDropped = src.status === 'DROPPED'
              return (
                <div key={key} className="flex items-center justify-between py-2 px-3 rounded-md bg-zinc-800/50 border border-zinc-700/50">
                  <div className="flex items-center gap-2.5">
                    {statusDot(isUp, isDropped)}
                    <span className="text-sm text-zinc-200">{label}</span>
                    <span className="text-xs text-zinc-600">{weight}</span>
                  </div>
                  <div className="text-right">
                    <div className={cn(
                      'text-xs font-semibold',
                      isUp ? 'text-emerald-400' : isDropped ? 'text-zinc-500' : 'text-rose-400',
                    )}>
                      {src.status}
                    </div>
                    {src.teams > 0 && (
                      <div className="text-xs text-zinc-600">{src.teams} teams</div>
                    )}
                  </div>
                </div>
              )
            })}
            <div className="text-xs text-zinc-600 text-right pt-1">
              Cache: {data.cache_age_hours.toFixed(1)}h ago
            </div>
          </>
        ) : null}
      </div>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Scheduler panel
// ---------------------------------------------------------------------------

function SchedulerPanel() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['scheduler-status'],
    queryFn: endpoints.schedulerStatus,
    refetchInterval: 60_000,
  })

  return (
    <Card className="p-0">
      <CardHeader className="px-5 pt-5 pb-0 mb-4">
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-zinc-500" />
            Scheduler
          </span>
          {data && (
            <span className={cn(
              'flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded border',
              data.running
                ? 'text-emerald-400 bg-emerald-400/10 border-emerald-500/30'
                : 'text-rose-400 bg-rose-400/10 border-rose-500/30',
            )}>
              {statusDot(data.running)}
              {data.running ? 'Running' : 'Stopped'}
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <div className="px-5 pb-5">
        {isLoading ? (
          <div className="space-y-2 animate-pulse">
            {[1, 2, 3, 4, 5].map(i => <div key={i} className="h-9 bg-zinc-800 rounded" />)}
          </div>
        ) : isError ? (
          <p className="text-rose-400 text-sm">Failed to load — admin key required.</p>
        ) : data ? (
          <div className="space-y-1.5 max-h-72 overflow-y-auto">
            {data.jobs.map((job) => (
              <div key={job.id} className="flex items-center justify-between py-2 px-3 rounded-md bg-zinc-800/40 border border-zinc-800">
                <span className="text-xs text-zinc-300 truncate max-w-[55%]">{job.name || job.id}</span>
                <span className="text-xs text-zinc-500 font-mono text-right">
                  {formatNextRun(job.next_run)}
                </span>
              </div>
            ))}
            {data.jobs.length === 0 && (
              <p className="text-zinc-600 text-sm text-center py-4">No jobs registered.</p>
            )}
          </div>
        ) : null}
      </div>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Odds Monitor panel
// ---------------------------------------------------------------------------

function OddsMonitorPanel() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['odds-monitor-status'],
    queryFn: endpoints.oddsMonitorStatus,
    refetchInterval: 30_000,
  })

  return (
    <Card className="p-0">
      <CardHeader className="px-5 pt-5 pb-0 mb-4">
        <CardTitle className="flex items-center gap-2">
          <Radio className="h-4 w-4 text-zinc-500" />
          Odds Monitor
        </CardTitle>
      </CardHeader>
      <div className="px-5 pb-5 space-y-3">
        {isLoading ? (
          <div className="space-y-2 animate-pulse">
            {[1, 2, 3].map(i => <div key={i} className="h-9 bg-zinc-800 rounded" />)}
          </div>
        ) : isError ? (
          <p className="text-rose-400 text-sm">Failed to load — admin key required.</p>
        ) : data ? (
          <>
            <div className="flex items-center justify-between py-2 px-3 rounded-md bg-zinc-800/50 border border-zinc-700/50">
              <span className="text-sm text-zinc-400">Status</span>
              <span className="flex items-center gap-1.5 text-sm font-semibold">
                {statusDot(data.active)}
                <span className={data.active ? 'text-emerald-400' : 'text-zinc-500'}>
                  {data.active ? 'Active' : 'Inactive'}
                </span>
              </span>
            </div>
            <div className="flex items-center justify-between py-2 px-3 rounded-md bg-zinc-800/50 border border-zinc-700/50">
              <span className="text-sm text-zinc-400">Games tracked</span>
              <span className="font-mono text-zinc-200">{data.games_tracked}</span>
            </div>
            {data.quota_remaining != null && (
              <div className="flex items-center justify-between py-2 px-3 rounded-md bg-zinc-800/50 border border-zinc-700/50">
                <span className="text-sm text-zinc-400">API quota</span>
                <span className={cn('font-mono text-sm', data.quota_is_low ? 'text-amber-400' : 'text-zinc-200')}>
                  {data.quota_remaining.toLocaleString()}
                  {data.quota_is_low && <span className="ml-1.5 text-xs text-amber-500">LOW</span>}
                </span>
              </div>
            )}
            {data.last_poll && (
              <div className="text-xs text-zinc-600 text-right">
                Last poll: {formatNextRun(data.last_poll)}
              </div>
            )}
          </>
        ) : null}
      </div>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function AdminPage() {
  return (
    <div className="space-y-6 max-w-5xl">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-zinc-100 flex items-center gap-2">
          <ShieldAlert className="h-5 w-5 text-amber-400" />
          Risk Dashboard
        </h1>
        <p className="text-sm text-zinc-500 mt-0.5">
          Live system health · Requires admin API key
        </p>
      </div>

      {/* 2×2 grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <PortfolioPanel />
        <RatingsPanel />
        <SchedulerPanel />
        <OddsMonitorPanel />
      </div>
    </div>
  )
}
