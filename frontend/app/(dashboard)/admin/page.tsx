'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { formatDistanceToNow, parseISO } from 'date-fns'
import { ShieldAlert, RefreshCw, CheckCircle2, AlertTriangle, XCircle, Clock, Database, Cpu, Radio, DollarSign, Play } from 'lucide-react'
import { endpoints } from '@/lib/api'
import { getApiKey } from '@/lib/api'
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
                  {data.quota_remaining?.toLocaleString()}
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
// Settlement trigger panel
// ---------------------------------------------------------------------------

function SettlementPanel() {
  const [status, setStatus] = useState<'idle' | 'loading' | 'done' | 'error'>('idle')
  const [result, setResult] = useState<string | null>(null)
  const [daysFrom, setDaysFrom] = useState<number>(2)

  async function triggerSettlement() {
    setStatus('loading')
    setResult(null)
    try {
      const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'
      const res = await fetch(`${BASE_URL}/admin/force-update-outcomes?days_from=${daysFrom}`, {
        method: 'POST',
        headers: { 'X-API-Key': getApiKey(), 'Content-Type': 'application/json' },
      })
      if (!res.ok) throw new Error(`${res.status}`)
      const data = await res.json()
      const updated = data.outcomes_updated ?? data.bets_updated ?? 0
      const games = data.games_checked ?? data.games_completed ?? 0
      setResult(`Updated ${updated} bet(s) across ${games} game(s)`)
      setStatus('done')
    } catch (e) {
      setResult(String(e))
      setStatus('error')
    }
  }

  return (
    <Card className="p-0">
      <CardHeader className="px-5 pt-5 pb-0 mb-4">
        <CardTitle className="flex items-center gap-2">
          <Play className="h-4 w-4 text-zinc-500" />
          Settlement
        </CardTitle>
      </CardHeader>
      <div className="px-5 pb-5 space-y-3">
        <p className="text-xs text-zinc-500">
          Manually trigger outcome settlement. Increase lookback to settle historical bets.
        </p>
        <div className="flex items-center gap-2">
          <label className="text-xs text-zinc-500 shrink-0">Look back</label>
          <input
            type="number"
            min={1}
            max={30}
            value={daysFrom}
            onChange={e => setDaysFrom(Math.max(1, Math.min(30, parseInt(e.target.value) || 2)))}
            className="w-16 bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-sm text-zinc-200 text-center"
          />
          <label className="text-xs text-zinc-500 shrink-0">days</label>
        </div>
        <button
          onClick={triggerSettlement}
          disabled={status === 'loading'}
          className={cn(
            'w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-md text-sm font-semibold transition-colors',
            status === 'loading'
              ? 'bg-zinc-700 text-zinc-500 cursor-not-allowed'
              : 'bg-amber-500/20 border border-amber-500/40 text-amber-300 hover:bg-amber-500/30',
          )}
        >
          <RefreshCw className={cn('h-4 w-4', status === 'loading' && 'animate-spin')} />
          {status === 'loading' ? 'Running...' : 'Trigger Settlement Now'}
        </button>
        {result && (
          <p className={cn('text-xs font-mono', status === 'error' ? 'text-rose-400' : 'text-emerald-400')}>
            {status === 'error' ? '✗ ' : '✓ '}{result}
          </p>
        )}
      </div>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Bankroll update panel
// ---------------------------------------------------------------------------

function BankrollPanel() {
  const [amount, setAmount] = useState('')
  const [status, setStatus] = useState<'idle' | 'loading' | 'done' | 'error'>('idle')
  const [msg, setMsg] = useState<string | null>(null)

  const { data: current, refetch } = useQuery({
    queryKey: ['bankroll-current'],
    queryFn: async () => {
      const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'
      const res = await fetch(`${BASE_URL}/admin/bankroll`, {
        headers: { 'X-API-Key': getApiKey() },
      })
      if (!res.ok) throw new Error('Failed')
      return res.json() as Promise<{ effective_bankroll: number; source: string; last_set: string | null }>
    },
  })

  async function save() {
    const val = parseFloat(amount)
    if (isNaN(val) || val <= 0) { setMsg('Enter a valid dollar amount'); setStatus('error'); return }
    setStatus('loading')
    setMsg(null)
    try {
      const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'
      const res = await fetch(`${BASE_URL}/admin/bankroll?amount=${val}`, {
        method: 'POST',
        headers: { 'X-API-Key': getApiKey(), 'Content-Type': 'application/json' },
      })
      if (!res.ok) throw new Error(`${res.status}`)
      setMsg(`Bankroll set to $${val.toFixed(2)}`)
      setStatus('done')
      setAmount('')
      refetch()
    } catch (e) {
      setMsg(String(e))
      setStatus('error')
    }
  }

  return (
    <Card className="p-0">
      <CardHeader className="px-5 pt-5 pb-0 mb-4">
        <CardTitle className="flex items-center gap-2">
          <DollarSign className="h-4 w-4 text-zinc-500" />
          Bankroll
        </CardTitle>
      </CardHeader>
      <div className="px-5 pb-5 space-y-3">
        {current && (
          <div className="rounded-md bg-zinc-800/50 border border-zinc-700/50 px-3 py-2 flex justify-between items-center">
            <span className="text-xs text-zinc-500">Current</span>
            <div className="text-right">
              <span className="font-mono font-semibold text-zinc-100">${current.effective_bankroll.toFixed(2)}</span>
              <span className="text-xs text-zinc-600 ml-2">({current.source === 'db_override' ? 'manual' : 'default'})</span>
            </div>
          </div>
        )}
        <div className="flex gap-2">
          <div className="relative flex-1">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500 text-sm">$</span>
            <input
              type="number"
              min="1"
              step="0.01"
              placeholder="New amount"
              value={amount}
              onChange={e => { setAmount(e.target.value); setStatus('idle'); setMsg(null) }}
              onKeyDown={e => e.key === 'Enter' && save()}
              className="w-full pl-7 pr-3 py-2 rounded-md bg-zinc-800 border border-zinc-700 text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-amber-500/60"
            />
          </div>
          <button
            onClick={save}
            disabled={status === 'loading' || !amount}
            className={cn(
              'px-4 py-2 rounded-md text-sm font-semibold transition-colors shrink-0',
              status === 'loading' || !amount
                ? 'bg-zinc-700 text-zinc-500 cursor-not-allowed'
                : 'bg-amber-500/20 border border-amber-500/40 text-amber-300 hover:bg-amber-500/30',
            )}
          >
            Save
          </button>
        </div>
        {msg && (
          <p className={cn('text-xs font-mono', status === 'error' ? 'text-rose-400' : 'text-emerald-400')}>
            {status === 'error' ? '✗ ' : '✓ '}{msg}
          </p>
        )}
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

      {/* Actions row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <SettlementPanel />
        <BankrollPanel />
      </div>

      {/* Status grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <PortfolioPanel />
        <RatingsPanel />
        <SchedulerPanel />
        <OddsMonitorPanel />
      </div>
    </div>
  )
}
