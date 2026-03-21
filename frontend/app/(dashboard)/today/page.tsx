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

// Parse "Bet 1.00u [T3] Ohio State Buckeyes (home -2.5) @ -106"
interface ParsedVerdict {
  team: string
  side: 'home' | 'away'
  spread: string | null   // "-2.5" | "+3.0" | null (moneyline)
  odds: string            // "-106" | "+102"
  tier: string | null     // "T3" | "T4+" | null
  betType: 'spread' | 'moneyline'
}

function parseVerdict(verdict: string): ParsedVerdict | null {
  // Full format: "Bet 1.00u [T3] Ohio State (home -2.5) @ -106"
  // Also handles T1 suffix: "... @ -106 (consider)"
  const m = verdict.match(
    /^Bet [\d.]+u (?:\[([^\]]+)\] )?(.+?) \((home|away)(?: ([+-][\d.]+))?\) @ ([+-]?\d+)/
  )
  if (!m) return null
  const [, tier, team, side, spread, odds] = m
  return {
    team,
    side: side as 'home' | 'away',
    spread: spread ?? null,
    odds,
    tier: tier ?? null,
    betType: spread ? 'spread' : 'moneyline',
  }
}

// Fallback: reconstruct pick from full_analysis when the portfolio scaler
// overwrites the verdict to the stripped "Bet 0.50u @ -101" format.
function parsedFromFullAnalysis(
  p: PredictionEntry
): ParsedVerdict | null {
  const calcs = (p.full_analysis as Record<string, unknown> | null)?.calculations as Record<string, unknown> | undefined
  if (!calcs) return null
  const betSide = calcs.bet_side as string | undefined
  const betOdds = calcs.bet_odds as number | undefined
  if (!betSide || betOdds == null) return null
  const team = betSide === 'home' ? p.game.home_team : p.game.away_team
  // Derive market spread from projected_margin (home perspective)
  // spread for our side = projected_margin if home, -projected_margin if away
  const spread =
    p.projected_margin != null
      ? betSide === 'home'
        ? p.projected_margin
        : -p.projected_margin
      : null
  return {
    team,
    side: betSide as 'home' | 'away',
    spread: spread != null ? (spread >= 0 ? `+${spread.toFixed(1)}` : spread.toFixed(1)) : null,
    odds: betOdds >= 0 ? `+${betOdds}` : String(betOdds),
    tier: null,
    betType: spread != null ? 'spread' : 'moneyline',
  }
}

// Market spread from verdict, home-team perspective (negative = home favored)
function marketSpreadHome(parsed: ParsedVerdict | null): number | null {
  if (!parsed?.spread) return null
  const val = parseFloat(parsed.spread)
  return parsed.side === 'home' ? val : -val
}

// ---------------------------------------------------------------------------
// BET card — prominent amber
// ---------------------------------------------------------------------------

function BetCard({ p }: { p: PredictionEntry }) {
  const homeTeam = p.game.home_team
  const awayTeam = p.game.away_team
  const parsed = parseVerdict(p.verdict) ?? parsedFromFullAnalysis(p)
  const marketHome = marketSpreadHome(parsed)

  // Line delta: how many points the model sees vs the market (home perspective)
  const lineDelta =
    p.projected_margin != null && marketHome != null
      ? p.projected_margin - marketHome
      : null

  return (
    <div className="rounded-lg border border-amber-500/40 bg-amber-500/5 p-5 space-y-4">
      {/* Row 1: badge + units + time */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold uppercase tracking-wider text-amber-400 bg-amber-400/15 px-2 py-0.5 rounded-full">
            BET
          </span>
          {p.recommended_units != null && (
            <span className="text-sm font-semibold text-zinc-100">
              {(p.recommended_units ?? 0).toFixed(1)}u
            </span>
          )}
          {parsed?.tier && (
            <span className="text-xs text-zinc-600 font-mono">[{parsed.tier}]</span>
          )}
        </div>
        <span className="text-xs text-zinc-500 tabular-nums">
          {formatGameTime(p.game.game_date)}
        </span>
      </div>

      {/* Row 2: matchup */}
      <div className="text-sm text-zinc-400">
        {awayTeam}{' '}
        <span className="text-zinc-600">@</span>{' '}
        {homeTeam}
        {p.game.is_neutral && (
          <span className="ml-1.5 text-xs text-zinc-600">(neutral)</span>
        )}
      </div>

      {/* Row 3: ACTION — what to bet */}
      {parsed ? (
        <div className="rounded-md bg-zinc-800/80 border border-zinc-700 px-4 py-3 flex items-center gap-3">
          {/* Bet type badge */}
          <span className="shrink-0 text-[10px] font-bold uppercase tracking-wider text-zinc-400 bg-zinc-700 px-1.5 py-1 rounded">
            {parsed.betType === 'spread' ? 'SPREAD' : 'MONEYLINE'}
          </span>
          {/* Team + line */}
          <span className="flex-1 font-semibold text-zinc-50 leading-snug">
            {parsed.team}
            {parsed.spread && (
              <span className={cn(
                'ml-2 font-mono text-base',
                parsed.spread.startsWith('-') ? 'text-amber-400' : 'text-sky-400'
              )}>
                {parsed.spread}
              </span>
            )}
          </span>
          {/* Odds */}
          <span className={cn(
            'shrink-0 text-sm font-mono font-semibold tabular-nums',
            parsed.odds.startsWith('+') ? 'text-emerald-400' : 'text-zinc-300'
          )}>
            {parsed.odds}
          </span>
        </div>
      ) : (
        // Fallback if verdict doesn't parse
        <div className="text-xs text-zinc-500 font-mono leading-relaxed">{p.verdict}</div>
      )}

      {/* Row 4: supporting stats */}
      <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs pt-1 border-t border-zinc-800">
        <div className="flex items-center gap-1.5">
          <span className="text-zinc-600">Edge</span>
          <span className="font-mono font-semibold text-emerald-400">{pct(p.edge_conservative)}</span>
        </div>
        {p.projected_margin != null && (
          <div className="flex items-center gap-1.5">
            <span className="text-zinc-600">Model line</span>
            <span className="font-mono text-zinc-300">
              {p.projected_margin >= 0 ? homeTeam : awayTeam}{' '}
              {p.projected_margin >= 0 ? '-' : '+'}{Math.abs(p.projected_margin).toFixed(1)}
            </span>
          </div>
        )}
        {lineDelta != null && Math.abs(lineDelta) > 0.05 && (
          <div className="flex items-center gap-1.5">
            <span className="text-zinc-600">vs market</span>
            <span className={cn(
              'font-mono font-semibold',
              lineDelta > 0 ? 'text-emerald-400' : 'text-rose-400'
            )}>
              {lineDelta > 0 ? '+' : ''}{lineDelta.toFixed(1)} pts
            </span>
          </div>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// CONSIDER card — sky
// ---------------------------------------------------------------------------

function ConsiderCard({ p }: { p: PredictionEntry }) {
  const homeTeam = p.game.home_team
  const awayTeam = p.game.away_team
  const parsed = parseVerdict(p.verdict) ?? parsedFromFullAnalysis(p)

  return (
    <div className="rounded-lg border border-sky-500/30 bg-sky-500/5 p-4 space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-bold uppercase tracking-wider text-sky-400 bg-sky-400/15 px-2 py-0.5 rounded-full">
          CONSIDER
        </span>
        <span className="text-xs text-zinc-500 font-mono">{formatGameTime(p.game.game_date)}</span>
      </div>

      {/* Matchup */}
      <div className="text-sm font-medium text-zinc-300">
        {awayTeam} <span className="text-zinc-600 font-normal">@</span> {homeTeam}
        {p.game.is_neutral && <span className="ml-1.5 text-xs text-zinc-500">(neutral)</span>}
      </div>

      {/* Action row (compact) */}
      {parsed ? (
        <div className="flex items-center gap-2 text-sm">
          <span className="text-[10px] font-bold uppercase tracking-wider text-zinc-500 bg-zinc-700 px-1.5 py-0.5 rounded">
            {parsed.betType === 'spread' ? 'SPR' : 'ML'}
          </span>
          <span className="text-zinc-200 font-medium">
            {parsed.team}
            {parsed.spread && (
              <span className="ml-1.5 font-mono text-sky-400">{parsed.spread}</span>
            )}
          </span>
          <span className={cn(
            'ml-auto font-mono text-xs tabular-nums',
            parsed.odds.startsWith('+') ? 'text-emerald-400' : 'text-zinc-400'
          )}>
            {parsed.odds}
          </span>
        </div>
      ) : null}

      {/* Stats */}
      <div className="flex gap-4 text-xs font-mono tabular-nums">
        <span className="text-zinc-500">
          Edge <span className="text-sky-400">{pct(p.edge_conservative)}</span>
        </span>
        {p.projected_margin != null && (
          <span className="text-zinc-500">
            Model{' '}
            <span className="text-zinc-300">
              {p.projected_margin >= 0 ? homeTeam : awayTeam}{' '}
              {p.projected_margin >= 0 ? '-' : '+'}{Math.abs(p.projected_margin).toFixed(1)}
            </span>
          </span>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// PASS row — minimal
// ---------------------------------------------------------------------------

function PassRow({ p }: { p: PredictionEntry }) {
  return (
    <div className="flex items-center justify-between px-3 py-2.5 rounded border border-zinc-800 bg-zinc-900/50 text-sm min-h-[44px]">
      <span className="text-zinc-500">
        {p.game.away_team} @ {p.game.home_team}
        {p.game.is_neutral && <span className="ml-1.5 text-xs">(N)</span>}
      </span>
      <div className="flex items-center gap-4 text-xs font-mono tabular-nums text-zinc-600">
        <span>{formatGameTime(p.game.game_date)}</span>
        {p.pass_reason && (
          <span className="text-zinc-700 max-w-[140px] truncate hidden sm:block">{p.pass_reason}</span>
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
    refetchInterval: 5 * 60 * 1000,
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
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {isError ? (
          <div className="col-span-4 rounded-lg border border-rose-500/30 bg-rose-500/10 p-4 text-rose-400 text-sm">
            Failed to load today&apos;s predictions.
          </div>
        ) : (
          <>
            <KpiCard title="Games Today" value={isLoading ? '--' : String(data?.total_games ?? 0)} unit="games" trend="neutral" loading={isLoading} />
            <KpiCard title="BETs" value={isLoading ? '--' : String(bets.length)} unit="picks" trend={bets.length > 0 ? 'up' : 'neutral'} loading={isLoading} />
            <KpiCard title="CONSIDERs" value={isLoading ? '--' : String(considers.length)} unit="marginal" trend="neutral" loading={isLoading} />
            <KpiCard title="Pass Rate" value={isLoading ? '--' : (passRate != null ? String(passRate) : '--')} unit="%" trend="neutral" loading={isLoading} />
          </>
        )}
      </div>

      {isLoading && (
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-36 bg-zinc-800 rounded-lg animate-pulse" />
          ))}
        </div>
      )}

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
