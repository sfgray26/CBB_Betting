'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { format, parseISO, formatDistanceToNow } from 'date-fns'
import { Activity, RefreshCw } from 'lucide-react'
import { endpoints } from '@/lib/api'
import { Card, CardHeader, CardTitle } from '@/components/ui/card'
import { DataTable, type Column } from '@/components/ui/data-table'
import { cn } from '@/lib/utils'
import type { PredictionEntry } from '@/lib/types'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

type VerdictType = 'BET' | 'CONSIDER' | 'PASS'

function getVerdictType(verdict: string): VerdictType {
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
// Verdict badge
// ---------------------------------------------------------------------------

function VerdictBadge({ verdict }: { verdict: string }) {
  const type = getVerdictType(verdict)
  return (
    <span
      className={cn(
        'text-xs font-bold uppercase tracking-wider px-2 py-0.5 rounded-full',
        type === 'BET' && 'text-amber-400 bg-amber-400/15',
        type === 'CONSIDER' && 'text-sky-400 bg-sky-400/15',
        type === 'PASS' && 'text-zinc-500 bg-zinc-700/50',
      )}
    >
      {type}
    </span>
  )
}

// ---------------------------------------------------------------------------
// Filter tab
// ---------------------------------------------------------------------------

const FILTERS: Array<{ label: string; value: VerdictType | 'ALL' }> = [
  { label: 'All', value: 'ALL' },
  { label: 'BET', value: 'BET' },
  { label: 'Consider', value: 'CONSIDER' },
  { label: 'Pass', value: 'PASS' },
]

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function LiveSlatePage() {
  const [filter, setFilter] = useState<VerdictType | 'ALL'>('ALL')

  // today/all includes games that have already started
  const { data, isLoading, isError, dataUpdatedAt, refetch, isFetching } = useQuery({
    queryKey: ['live-slate'],
    queryFn: endpoints.todaysPredictionsAll,
    refetchInterval: 5 * 60 * 1000,
  })

  const allPredictions = data?.predictions ?? []
  const filtered =
    filter === 'ALL'
      ? allPredictions
      : allPredictions.filter((p) => getVerdictType(p.verdict) === filter)

  const betCount = allPredictions.filter((p) => getVerdictType(p.verdict) === 'BET').length
  const considerCount = allPredictions.filter((p) => getVerdictType(p.verdict) === 'CONSIDER').length

  const columns: Column<PredictionEntry>[] = [
    {
      key: 'time',
      header: 'Time',
      accessor: (r) => (
        <span className="font-mono tabular-nums text-zinc-400 text-xs">
          {formatGameTime(r.game.game_date)}
        </span>
      ),
      sortValue: (r) => r.game.game_date,
    },
    {
      key: 'matchup',
      header: 'Matchup',
      accessor: (r) => (
        <div className="text-sm">
          <span className="text-zinc-300">{r.game.away_team}</span>
          <span className="text-zinc-600 mx-1.5">@</span>
          <span className="text-zinc-300">{r.game.home_team}</span>
          {r.game.is_neutral && (
            <span className="ml-1.5 text-xs text-zinc-600">(N)</span>
          )}
        </div>
      ),
      sortValue: (r) => `${r.game.away_team} ${r.game.home_team}`,
    },
    {
      key: 'verdict',
      header: 'Verdict',
      accessor: (r) => <VerdictBadge verdict={r.verdict} />,
      sortValue: (r) => getVerdictType(r.verdict),
    },
    {
      key: 'margin',
      header: 'Model Spread',
      accessor: (r) => (
        <span className={cn(
          'font-mono tabular-nums text-sm',
          r.projected_margin == null ? 'text-zinc-600' :
            r.projected_margin < 0 ? 'text-sky-300' : 'text-zinc-300',
        )}>
          {r.projected_margin != null
            ? `${r.game.home_team.split(' ').pop()} ${signed(r.projected_margin)}`
            : '—'}
        </span>
      ),
      sortValue: (r) => r.projected_margin ?? 999,
      className: 'text-right',
      headerClassName: 'text-right',
    },
    {
      key: 'edge',
      header: 'Edge',
      accessor: (r) => (
        <span className={cn(
          'font-mono tabular-nums text-sm',
          r.edge_conservative == null ? 'text-zinc-600' :
            r.edge_conservative > 0.04 ? 'text-emerald-400' :
              r.edge_conservative > 0 ? 'text-emerald-600' : 'text-rose-400',
        )}>
          {pct(r.edge_conservative)}
        </span>
      ),
      sortValue: (r) => r.edge_conservative ?? -999,
      className: 'text-right',
      headerClassName: 'text-right',
    },
    {
      key: 'units',
      header: 'Units',
      accessor: (r) => (
        <span className="font-mono tabular-nums text-sm text-zinc-400">
          {r.recommended_units != null ? `${r.recommended_units.toFixed(1)}u` : '—'}
        </span>
      ),
      sortValue: (r) => r.recommended_units ?? 0,
      className: 'text-right',
      headerClassName: 'text-right',
    },
    {
      key: 'pass_reason',
      header: 'Pass Reason',
      accessor: (r) => (
        <span className="text-xs text-zinc-600 truncate max-w-[180px] block">
          {r.pass_reason ?? '—'}
        </span>
      ),
      sortValue: (r) => r.pass_reason ?? '',
    },
  ]

  return (
    <div className="space-y-6 max-w-7xl">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-zinc-100 flex items-center gap-2">
            <Activity className="h-5 w-5 text-zinc-400" />
            Live Slate
          </h1>
          <p className="text-sm text-zinc-500 mt-0.5">
            {data?.date ? format(parseISO(data.date), 'EEEE, MMMM d') : 'Loading…'}
            {' · '}
            {allPredictions.length} games · {betCount} BET · {considerCount} CONSIDER
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

      {/* Error */}
      {isError && (
        <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-4 text-rose-400 text-sm">
          Failed to load live slate.
        </div>
      )}

      {/* Filter tabs */}
      <div className="flex gap-1 p-1 bg-zinc-800/60 rounded-lg w-fit">
        {FILTERS.map((f) => (
          <button
            key={f.value}
            onClick={() => setFilter(f.value)}
            className={cn(
              'px-3 py-1.5 rounded-md text-xs font-medium transition-colors',
              filter === f.value
                ? 'bg-zinc-700 text-zinc-100'
                : 'text-zinc-500 hover:text-zinc-300',
            )}
          >
            {f.label}
            {f.value !== 'ALL' && (
              <span className="ml-1.5 text-zinc-500">
                {f.value === 'BET' ? betCount :
                  f.value === 'CONSIDER' ? considerCount :
                    allPredictions.length - betCount - considerCount}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Table */}
      <Card className="p-0">
        <CardHeader className="px-6 pt-5 pb-0 mb-2">
          <CardTitle>All Games</CardTitle>
        </CardHeader>
        {isLoading ? (
          <div className="p-6 space-y-2">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="h-10 bg-zinc-800 rounded animate-pulse" />
            ))}
          </div>
        ) : (
          <DataTable
            columns={columns}
            data={filtered}
            keyExtractor={(r) => String(r.id)}
            emptyMessage={
              filter === 'ALL'
                ? 'No predictions for today yet. Nightly analysis runs at 3 AM ET.'
                : `No ${filter} picks today.`
            }
          />
        )}
      </Card>
    </div>
  )
}
