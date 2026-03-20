'use client'

import { useQuery } from '@tanstack/react-query'
import { formatDistanceToNow, parseISO } from 'date-fns'
import { Radio, AlertTriangle, CheckCircle, XCircle, RefreshCw, Activity } from 'lucide-react'
import { endpoints } from '@/lib/api'
import { KpiCard } from '@/components/ui/kpi-card'
import { Card, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'

// ---------------------------------------------------------------------------
// Status row helper
// ---------------------------------------------------------------------------

function StatusRow({
  label,
  value,
  sub,
  ok,
}: {
  label: string
  value: string
  sub?: string
  ok?: boolean
}) {
  return (
    <div className="flex items-center justify-between py-3 border-b border-zinc-800 last:border-0">
      <div className="flex items-center gap-2">
        {ok === true && <CheckCircle className="h-3.5 w-3.5 text-emerald-400 shrink-0" />}
        {ok === false && <XCircle className="h-3.5 w-3.5 text-rose-400 shrink-0" />}
        {ok === undefined && <div className="h-3.5 w-3.5 shrink-0" />}
        <span className="text-sm text-zinc-400">{label}</span>
      </div>
      <div className="text-right">
        <div className="text-sm font-mono tabular-nums text-zinc-200">{value}</div>
        {sub && <div className="text-xs text-zinc-600 mt-0.5">{sub}</div>}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function OddsMonitorPage() {
  const {
    data: monitor,
    isLoading: monitorLoading,
    isError: monitorError,
    dataUpdatedAt,
    refetch,
    isFetching,
  } = useQuery({
    queryKey: ['odds-monitor-status'],
    queryFn: endpoints.oddsMonitorStatus,
    refetchInterval: 60_000,
  })

  const {
    data: portfolio,
    isLoading: portfolioLoading,
    isError: portfolioError,
  } = useQuery({
    queryKey: ['portfolio-full'],
    queryFn: endpoints.portfolioStatusFull,
    refetchInterval: 60_000,
  })

  const lastPollAgo = monitor?.last_poll
    ? formatDistanceToNow(parseISO(monitor.last_poll), { addSuffix: true })
    : 'Never'

  const quotaUpdatedAgo = monitor?.quota_updated_at
    ? formatDistanceToNow(parseISO(monitor.quota_updated_at), { addSuffix: true })
    : '—'

  const drawdownColor =
    (portfolio?.drawdown_pct ?? 0) < 5
      ? 'text-emerald-400'
      : (portfolio?.drawdown_pct ?? 0) < 10
        ? 'text-amber-400'
        : 'text-rose-400'

  const bankrollChange = portfolio
    ? portfolio.current_bankroll - portfolio.starting_bankroll
    : null
  const bankrollChangePct = portfolio
    ? ((portfolio.current_bankroll - portfolio.starting_bankroll) / portfolio.starting_bankroll) * 100
    : null

  return (
    <div className="space-y-6 max-w-3xl">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-zinc-100 flex items-center gap-2">
            <Radio className="h-5 w-5 text-zinc-400" />
            Odds Monitor
          </h1>
          <p className="text-sm text-zinc-500 mt-0.5">
            System health · line movement detection · portfolio exposure
          </p>
        </div>
        <button
          onClick={() => refetch()}
          disabled={isFetching}
          className="flex items-center gap-1.5 text-xs text-zinc-500 hover:text-zinc-300 transition-colors disabled:opacity-40"
        >
          <RefreshCw className={cn('h-3.5 w-3.5', isFetching && 'animate-spin')} />
          {dataUpdatedAt
            ? `Updated ${formatDistanceToNow(dataUpdatedAt, { addSuffix: true })}`
            : 'Refresh'}
        </button>
      </div>

      {/* Monitor KPI row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
        {monitorError ? (
          <div className="col-span-3 rounded-lg border border-rose-500/30 bg-rose-500/10 p-4 text-rose-400 text-sm">
            Failed to load odds monitor status. Admin key required.
          </div>
        ) : (
          <>
            <KpiCard
              title="Monitor"
              value={monitorLoading ? '--' : (monitor?.active ? 'Active' : 'Inactive')}
              unit=""
              trend={monitor?.active ? 'up' : 'down'}
              loading={monitorLoading}
            />
            <KpiCard
              title="Games Tracked"
              value={monitorLoading ? '--' : String(monitor?.games_tracked ?? 0)}
              unit="games"
              trend="neutral"
              loading={monitorLoading}
            />
            <KpiCard
              title="API Quota"
              value={monitorLoading ? '--' : (monitor?.quota_remaining != null ? String(monitor.quota_remaining) : '—')}
              unit="calls left"
              trend={monitor?.quota_is_low ? 'down' : 'neutral'}
              loading={monitorLoading}
            />
          </>
        )}
      </div>

      {/* Quota warning */}
      {!monitorLoading && monitor?.quota_is_low && (
        <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-4 flex items-start gap-3">
          <AlertTriangle className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
          <div>
            <div className="text-sm font-medium text-amber-300">API Quota Low</div>
            <div className="text-xs text-amber-500 mt-0.5">
              Odds monitor has paused polling to preserve quota reserve. Will resume automatically when quota refreshes.
            </div>
          </div>
        </div>
      )}

      {/* Monitor detail */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-zinc-400" />
            Monitor Status
          </CardTitle>
        </CardHeader>
        {monitorLoading ? (
          <div className="space-y-3 pt-2">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-10 bg-zinc-800 rounded animate-pulse" />
            ))}
          </div>
        ) : monitorError ? (
          <div className="text-zinc-500 text-sm">Unavailable</div>
        ) : (
          <div className="pt-2">
            <StatusRow
              label="Status"
              value={monitor?.active ? 'Running' : 'Stopped'}
              ok={monitor?.active}
            />
            <StatusRow
              label="Games Tracked"
              value={String(monitor?.games_tracked ?? 0)}
              sub="games with live line history"
            />
            <StatusRow
              label="Last Poll"
              value={lastPollAgo}
              sub={monitor?.last_poll ?? undefined}
              ok={monitor?.last_poll != null}
            />
            <StatusRow
              label="Quota Remaining"
              value={monitor?.quota_remaining != null ? `${monitor.quota_remaining} calls` : '—'}
              sub={`Updated ${quotaUpdatedAgo}`}
              ok={!monitor?.quota_is_low}
            />
          </div>
        )}
      </Card>

      {/* Portfolio status */}
      <Card>
        <CardHeader>
          <CardTitle>Portfolio Status</CardTitle>
        </CardHeader>
        {portfolioLoading ? (
          <div className="space-y-3 pt-2">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="h-10 bg-zinc-800 rounded animate-pulse" />
            ))}
          </div>
        ) : portfolioError ? (
          <div className="text-zinc-500 text-sm">Unavailable. Admin key required.</div>
        ) : portfolio ? (
          <div className="pt-2">
            {portfolio.is_halted && (
              <div className="mb-3 rounded-md border border-rose-500/40 bg-rose-500/10 p-3 flex items-center gap-2">
                <XCircle className="h-4 w-4 text-rose-400 shrink-0" />
                <div className="text-sm text-rose-300">
                  Betting halted
                  {portfolio.halt_reason && (
                    <span className="text-rose-500 ml-1">— {portfolio.halt_reason}</span>
                  )}
                </div>
              </div>
            )}
            <StatusRow
              label="Current Bankroll"
              value={`$${portfolio.current_bankroll.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
              sub={bankrollChangePct != null
                ? `${bankrollChangePct >= 0 ? '+' : ''}${bankrollChangePct.toFixed(1)}% vs starting ($${portfolio.starting_bankroll.toLocaleString()})`
                : undefined}
              ok={bankrollChange != null ? bankrollChange >= 0 : undefined}
            />
            <StatusRow
              label="Drawdown"
              value={`${portfolio.drawdown_pct.toFixed(1)}%`}
              ok={portfolio.drawdown_pct < 10}
            />
            <StatusRow
              label="Total Exposure"
              value={`${portfolio.total_exposure_pct.toFixed(1)}%`}
              sub="% of bankroll deployed"
              ok={portfolio.total_exposure_pct < 15}
            />
            <StatusRow
              label="Open Positions"
              value={String(portfolio.pending_positions)}
              sub="pending bets"
            />
            <StatusRow
              label="Status"
              value={portfolio.is_halted ? 'HALTED' : 'ACTIVE'}
              ok={!portfolio.is_halted}
            />
          </div>
        ) : null}
      </Card>

      {/* Drawdown gauge */}
      {portfolio && (
        <Card>
          <CardHeader>
            <CardTitle>Drawdown Gauge</CardTitle>
          </CardHeader>
          <div className="pt-2 space-y-2">
            <div className="flex justify-between text-xs text-zinc-500">
              <span>0%</span>
              <span className="text-amber-500">Warning 10%</span>
              <span className="text-rose-500">Halt 15%</span>
            </div>
            <div className="relative h-3 bg-zinc-800 rounded-full overflow-hidden">
              {/* Gradient track */}
              <div
                className="absolute inset-y-0 left-0 rounded-full transition-all duration-500"
                style={{
                  width: `${Math.min(portfolio.drawdown_pct / 15 * 100, 100)}%`,
                  background: portfolio.drawdown_pct < 10
                    ? '#34d399'
                    : portfolio.drawdown_pct < 15
                      ? '#fbbf24'
                      : '#f43f5e',
                }}
              />
            </div>
            <div className={cn('text-right text-sm font-mono tabular-nums font-semibold', drawdownColor)}>
              {portfolio.drawdown_pct.toFixed(1)}% drawdown
            </div>
          </div>
        </Card>
      )}
    </div>
  )
}
