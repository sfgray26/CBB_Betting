'use client'

import { useQuery } from '@tanstack/react-query'
import { format, parseISO, formatDistanceToNow } from 'date-fns'
import { Zap, Clock, TrendingUp, BarChart2, RefreshCw } from 'lucide-react'
import { endpoints } from '@/lib/api'
import { KpiCard } from '@/components/ui/kpi-card'
import { Card, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'
import type { PredictionEntry } from '@/lib/types'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getVerdictType(verdict: string): 'BET' | 'CONSIDER' | 'PASS' {
  if (verdict.startsWith('Bet ')) return 'BET'
  if (verdict.toUpperCase().startsWith('CONSIDER')) return 'CONSIDER'
  return 'PASS'
}

function signed(v: number | null, decimals = 1): string {
  if (v == null) return '—'
  return `${v >= 0 ? '+' : ''}${v.toFixed(decimals)}`
}

function pct(v: number | null, decimals = 1): string {
  if (v == null) return '—'
  return `${(v * 100).toFixed(decimals)}%`
}

function formatGameTime(dateStr: string): string {
  try {
    return format(parseISO(dateStr), 'h:mm a')
  } catch {
    return '—'
  }
}

// ---------------------------------------------------------------------------
// BET card — prominent amber
// ---------------------------------------------------------------------------

function BetCard({ p }: { p: PredictionEntry }) {
  const margin = p.projected_margin
  const homeTeam = p.game.home_team
  const awayTeam = p.game.away_team
  const favored = margin != null && margin < 0 ? homeTeam : awayTeam
  const spread = margin != null ? Math.abs(margin).toFixed(1) : '—'

  return (
    <div className="rounded-lg border border-amber-500/40 bg-amber-500/5 p-5 space-y-3">
      {/* Header row */}
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-bold uppercase tracking-wider text-amber-400 bg-amber-400/15 px-2 py-0.5 rounded-full">
              BET
            </span>
            {p.recommended_units != null && (
              <span className="text-xs text-zinc-400 font-mono tabular-nums">
                {p.recommended_units.toFixed(1)}u
              </span>
            )}
          </div>
          <div className="text-base font-semibold text-zinc-100">
            {awayTeam} <span className="text-zinc-500 font-normal">@</span> {homeTeam}
            {p.game.is_neutral && (
              <span className="ml-2 text-xs text-zinc-500">(neutral)</span>
            )}
          </div>
        </div>
        <div className="text-right shrink-0">
          <div className="text-xs text-zinc-500">
            {formatGameTime(p.game.game_date)}
          </div>
        </div>
      </div>

      {/* Stats row */}
      <div className="flex gap-5 text-sm">
        <div>
          <div className="text-xs text-zinc-500 mb-0.5">Model Spread</div>
          <div className="font-mono tabular-nums text-zinc-200">
            {favored} <span className="text-amber-400">{signed(margin)}</span>
          </div>
        </div>
        <div>
          <div className="text-xs text-zinc-500 mb-0.5">Edge</div>
          <div className="font-mono tabular-nums text-emerald-400">
            {pct(p.edge_conservative)}
          </div>
        </div>
      </div>

      {/* Verdict string */}
      <div className="text-xs text-zinc-400 border-t border-zinc-800 pt-2 font-mono leading-relaxed break-all">
        {p.verdict}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// CONSIDER card — sky
// ---------------------------------------------------------------------------

function ConsiderCard({ p }: { p: PredictionEntry }) {
  const margin = p.projected_margin
  const homeTeam = p.game.home_team
  const awayTeam = p.game.away_team

  return (
    <div className="rounded-lg border border-sky-500/30 bg-sky-500/5 p-4 space-y-2">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold uppercase tracking-wider text-sky-400 bg-sky-400/15 px-2 py-0.5 rounded-full">
            CONSIDER
          </span>
        </div>
        <div className="text-xs text-zinc-500 font-mono">{formatGameTime(p.game.game_date)}</div>
      </div>
      <div className="text-sm font-medium text-zinc-200">
        {awayTeam} @ {homeTeam}
        {p.game.is_neutral && <span className="ml-1.5 text-xs text-zinc-500">(neutral)</span>}
      </div>
      <div className="flex gap-4 text-xs font-mono tabular-nums">
        <span className="text-zinc-400">
          Margin: <span className="text-zinc-200">{signed(margin)}</span>
        </span>
        <span className="text-zinc-400">
          Edge: <span className="text-sky-400">{pct(p.edge_conservative)}</span>
        </span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// PASS row — minimal
// ---------------------------------------------------------------------------

function PassRow({ p }: { p: PredictionEntry }) {
  return (
    <div className="flex items-center justify-between px-3 py-2 rounded border border-zinc-800 bg-zinc-900/50 text-sm">
      <span className="text-zinc-500">
        {p.game.away_team} @ {p.game.home_team}
        {p.game.is_neutral && <span className="ml-1.5 text-xs">(N)</span>}
      </span>
      <div className="flex items-center gap-4 text-xs font-mono tabular-nums text-zinc-600">
        <span>{formatGameTime(p.game.game_date)}</span>
        {p.pass_reason && (
          <span className="text-zinc-700 max-w-[160px] truncate">{p.pass_reason}</span>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function TodayPage() {
  const { data, isLoading, isError, dataUpdatedAt, refetch, isFetching } = useQuery({
    queryKey: ['todays-predictions'],
    queryFn: endpoints.todaysPredictions,
    refetchInterval: 5 * 60 * 1000, // auto-refresh every 5 min
  })

  const predictions = data?.predictions ?? []
  const bets = predictions.filter((p) => getVerdictType(p.verdict) === 'BET')
  const considers = predictions.filter((p) => getVerdictType(p.verdict) === 'CONSIDER')
  const passes = predictions.filter((p) => getVerdictType(p.verdict) === 'PASS')
  const passRate = predictions.length > 0
    ? Math.round((passes.length / predictions.length) * 100)
    : null

  return (
    <div className="space-y-6 max-w-4xl">

      {/* Page header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-zinc-100">Today&apos;s Bets</h1>
          <p className="text-sm text-zinc-500 mt-0.5">
            {data?.date ? format(parseISO(data.date), 'EEEE, MMMM d') : 'Loading…'}
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

      {/* KPI row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {isError ? (
          <div className="col-span-4 rounded-lg border border-rose-500/30 bg-rose-500/10 p-4 text-rose-400 text-sm">
            Failed to load today&apos;s predictions.
          </div>
        ) : (
          <>
            <KpiCard
              title="Games Today"
              value={isLoading ? '--' : String(data?.total_games ?? 0)}
              unit="games"
              trend="neutral"
              loading={isLoading}
            />
            <KpiCard
              title="BETs"
              value={isLoading ? '--' : String(bets.length)}
              unit="picks"
              trend={bets.length > 0 ? 'up' : 'neutral'}
              loading={isLoading}
            />
            <KpiCard
              title="CONSIDERs"
              value={isLoading ? '--' : String(considers.length)}
              unit="marginal"
              trend="neutral"
              loading={isLoading}
            />
            <KpiCard
              title="Pass Rate"
              value={isLoading ? '--' : (passRate != null ? String(passRate) : '--')}
              unit="%"
              trend="neutral"
              loading={isLoading}
            />
          </>
        )}
      </div>

      {/* Loading skeletons */}
      {isLoading && (
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-28 bg-zinc-800 rounded-lg animate-pulse" />
          ))}
        </div>
      )}

      {/* Empty state */}
      {!isLoading && !isError && predictions.length === 0 && (
        <div className="rounded-lg border border-zinc-700 bg-zinc-800/50 p-10 text-center">
          <BarChart2 className="h-8 w-8 text-zinc-600 mx-auto mb-3" />
          <p className="text-zinc-400 text-sm">No predictions for today yet.</p>
          <p className="text-zinc-600 text-xs mt-1">Nightly analysis runs at 3 AM ET.</p>
        </div>
      )}

      {/* BET section */}
      {bets.length > 0 && (
        <Card className="p-0">
          <CardHeader className="px-5 pt-5 pb-0 mb-4">
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-amber-400" />
              <span>Recommended Bets</span>
              <span className="ml-1 text-sm font-normal text-zinc-500">({bets.length})</span>
            </CardTitle>
          </CardHeader>
          <div className="px-5 pb-5 space-y-3">
            {bets.map((p) => <BetCard key={p.id} p={p} />)}
          </div>
        </Card>
      )}

      {/* CONSIDER section */}
      {considers.length > 0 && (
        <Card className="p-0">
          <CardHeader className="px-5 pt-5 pb-0 mb-4">
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-sky-400" />
              <span>Consider</span>
              <span className="ml-1 text-sm font-normal text-zinc-500">({considers.length})</span>
            </CardTitle>
          </CardHeader>
          <div className="px-5 pb-5 grid grid-cols-1 sm:grid-cols-2 gap-3">
            {considers.map((p) => <ConsiderCard key={p.id} p={p} />)}
          </div>
        </Card>
      )}

      {/* PASS section */}
      {passes.length > 0 && (
        <Card className="p-0">
          <CardHeader className="px-5 pt-5 pb-0 mb-3">
            <CardTitle className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-zinc-500" />
              <span className="text-zinc-400">Pass</span>
              <span className="ml-1 text-sm font-normal text-zinc-600">({passes.length})</span>
            </CardTitle>
          </CardHeader>
          <div className="px-5 pb-5 space-y-1.5">
            {passes.map((p) => <PassRow key={p.id} p={p} />)}
          </div>
        </Card>
      )}
    </div>
  )
}
