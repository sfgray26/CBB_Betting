'use client'

import { useQuery } from '@tanstack/react-query'
import { Users, RefreshCw, AlertTriangle } from 'lucide-react'
import { endpoints } from '@/lib/api'
import { Card, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'
import { StatusBadge } from '@/components/shared/status-badge'
import type { RosterPlayer } from '@/lib/types'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function zScoreColor(z: number | null): string {
  if (z === null) return 'text-zinc-500'
  if (z >= 3.0) return 'text-amber-400'
  if (z >= 1.5) return 'text-emerald-400'
  if (z >= 0) return 'text-zinc-300'
  return 'text-zinc-500'
}

// ---------------------------------------------------------------------------
// Loading skeleton
// ---------------------------------------------------------------------------

function TableSkeleton() {
  return (
    <div className="overflow-x-auto rounded-lg border border-zinc-800 animate-pulse">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-800 bg-zinc-900/60">
            {[160, 70, 60, 80, 80, 90].map((w, i) => (
              <th key={i} className="px-3 py-3">
                <div className="h-3 bg-zinc-800 rounded" style={{ width: w }} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-800/60">
          {Array.from({ length: 10 }).map((_, i) => (
            <tr key={i}>
              {[160, 70, 60, 80, 80, 90].map((w, j) => (
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
// Roster table
// ---------------------------------------------------------------------------

function RosterTable({ players }: { players: RosterPlayer[] }) {
  if (players.length === 0) {
    return (
      <p className="text-zinc-600 text-sm text-center py-8">No players on roster.</p>
    )
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-zinc-800">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-800 bg-zinc-900/60">
            <th className="px-3 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider">
              Player
            </th>
            <th className="px-3 py-3 text-center text-xs font-semibold text-zinc-500 uppercase tracking-wider w-24">
              Status
            </th>
            <th className="px-3 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider w-20">
              Pos
            </th>
            <th className="px-3 py-3 text-center text-xs font-semibold text-zinc-500 uppercase tracking-wider w-14">
              Slot
            </th>
            <th className="px-3 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider w-16">
              Team
            </th>
            <th className="px-3 py-3 text-right text-xs font-semibold text-zinc-500 uppercase tracking-wider w-24">
              Z-Score
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-800/60">
          {players.map((p) => (
            <tr key={p.player_key} className="hover:bg-zinc-800/50 transition-colors">
              <td className="px-3 py-2.5">
                <div className="font-medium text-zinc-100">{p.name}</div>
                {p.injury_note && (
                  <div className="text-xs text-zinc-500 mt-0.5">{p.injury_note}</div>
                )}
              </td>
              <td className="px-3 py-2.5 text-center">
                <StatusBadge status={p.status} />
              </td>
              <td className="px-3 py-2.5">
                <div className="flex flex-wrap gap-1">
                  {p.positions.slice(0, 3).map((pos) => (
                    <span
                      key={pos}
                      className="px-1.5 py-0.5 rounded text-xs bg-zinc-800 text-zinc-400 border border-zinc-700"
                    >
                      {pos}
                    </span>
                  ))}
                </div>
              </td>
              <td className="px-3 py-2.5 text-center">
                {p.selected_position ? (
                  <span className={cn(
                    'px-1.5 py-0.5 rounded text-xs font-mono font-semibold',
                    p.selected_position === 'BN'
                      ? 'bg-zinc-700/60 text-zinc-500 border border-zinc-600/40'
                      : p.selected_position === 'IL' || p.selected_position?.startsWith('IL')
                        ? 'bg-rose-500/15 text-rose-400 border border-rose-500/30'
                        : 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30',
                  )}>
                    {p.selected_position}
                  </span>
                ) : (
                  <span className="text-zinc-600 text-xs">—</span>
                )}
              </td>
              <td className="px-3 py-2.5 text-zinc-400 font-mono text-xs">
                {p.team ?? '-'}
              </td>
              <td className={cn('px-3 py-2.5 text-right font-mono text-xs font-semibold tabular-nums', zScoreColor(p.z_score))}>
                {p.z_score !== null ? (p.z_score >= 0 ? '+' : '') + p.z_score.toFixed(2) : '-'}
              </td>
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

export default function RosterPage() {
  const { data, isLoading, isError, error, refetch, isFetching } = useQuery({
    queryKey: ['fantasy-roster'],
    queryFn: endpoints.fantasyRoster,
    refetchInterval: 5 * 60_000,
    retry: 1,
  })

  const errorMsg = error instanceof Error ? error.message : ''
  const httpStatus = errorMsg.match(/^(\d{3})/)?.[1] ?? null
  const isYahooNotConfigured = isError && httpStatus === '503'

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-xl font-semibold text-zinc-100 flex items-center gap-2">
            <Users className="h-5 w-5 text-amber-400" />
            My Roster
          </h1>
          <p className="text-sm text-zinc-500 mt-0.5">
            Yahoo Fantasy roster enriched with z-scores
            {data && (
              <span className="ml-2 px-2 py-0.5 text-xs bg-zinc-800 text-zinc-400 rounded font-mono border border-zinc-700">
                {data.team_key}
              </span>
            )}
          </p>
        </div>
        <button
          onClick={() => refetch()}
          disabled={isFetching}
          className="flex items-center gap-1.5 px-3 py-2 text-sm text-zinc-400 hover:text-zinc-200 bg-zinc-800 hover:bg-zinc-700 rounded-md transition-colors disabled:opacity-50 min-h-[44px]"
        >
          <RefreshCw className={cn('h-4 w-4', isFetching && 'animate-spin')} />
          Refresh
        </button>
      </div>

      {/* Error states */}
      {isError && isYahooNotConfigured && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-5 flex items-start gap-3">
          <AlertTriangle className="h-5 w-5 text-amber-400 flex-shrink-0 mt-0.5" />
          <div>
            <div className="flex items-center gap-2">
              <p className="text-amber-300 font-medium text-sm">Yahoo not configured</p>
              <span className="px-1.5 py-0.5 rounded text-xs font-mono font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/40">
                HTTP 503
              </span>
            </div>
            <p className="text-amber-300/60 text-xs mt-0.5">
              Set YAHOO_CLIENT_ID, YAHOO_CLIENT_SECRET, and YAHOO_REFRESH_TOKEN in Railway.
            </p>
          </div>
        </div>
      )}

      {isError && !isYahooNotConfigured && (
        <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-5 flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <p className="text-rose-400 font-medium text-sm">Failed to load roster</p>
              {httpStatus && (
                <span className="px-1.5 py-0.5 rounded text-xs font-mono font-semibold bg-rose-500/20 text-rose-400 border border-rose-500/40">
                  HTTP {httpStatus}
                </span>
              )}
            </div>
            <p className="text-rose-400/60 text-xs mt-0.5">
              {errorMsg || 'Unknown error'}
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

      {/* Roster table */}
      {!isError && (
        <Card className="p-0">
          <CardHeader className="px-5 pt-5 pb-0 mb-4">
            <CardTitle className="text-base">
              Players
              {data && (
                <span className="ml-2 text-sm font-normal text-zinc-500">
                  ({data.count})
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <div className="px-5 pb-5">
            {isLoading ? (
              <TableSkeleton />
            ) : data ? (
              <RosterTable players={data.players} />
            ) : null}
          </div>
        </Card>
      )}
    </div>
  )
}
