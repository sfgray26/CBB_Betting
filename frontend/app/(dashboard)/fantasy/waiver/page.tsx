'use client'

import { useQuery } from '@tanstack/react-query'
import { ArrowLeftRight, RefreshCw } from 'lucide-react'
import { endpoints } from '@/lib/api'
import { Card, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'
import type { CategoryDeficit, WaiverPlayer } from '@/lib/types'

// ---------------------------------------------------------------------------
// Category deficit card
// ---------------------------------------------------------------------------

function DeficitCard({ cat }: { cat: CategoryDeficit }) {
  return (
    <div
      className={cn(
        'rounded-lg border p-4 space-y-1.5',
        cat.winning
          ? 'border-emerald-500/30 bg-emerald-500/5'
          : 'border-rose-500/30 bg-rose-500/5',
      )}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
          {cat.category}
        </span>
        {cat.winning ? (
          <span className="px-2 py-0.5 rounded text-xs font-semibold bg-emerald-500/15 text-emerald-400">
            WINNING
          </span>
        ) : (
          <span className="px-2 py-0.5 rounded text-xs font-semibold bg-rose-500/15 text-rose-400">
            LOSING
          </span>
        )}
      </div>
      <div className="flex items-baseline gap-2">
        <span className="font-mono text-lg font-bold text-zinc-100">
          {cat.my_total % 1 === 0 ? cat.my_total : cat.my_total.toFixed(2)}
        </span>
        <span className="text-zinc-600 text-sm">vs</span>
        <span className="font-mono text-sm text-zinc-400">
          {cat.opponent_total % 1 === 0 ? cat.opponent_total : cat.opponent_total.toFixed(2)}
        </span>
      </div>
      {!cat.winning && (
        <div className="text-xs font-mono text-rose-400">
          deficit: {cat.deficit < 0 ? cat.deficit.toFixed(2) : `+${cat.deficit.toFixed(2)}`}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Category tracker skeleton
// ---------------------------------------------------------------------------

function CategorySkeleton() {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 animate-pulse">
      {Array.from({ length: 10 }).map((_, i) => (
        <div key={i} className="h-24 bg-zinc-800 rounded-lg" />
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Waiver player table
// ---------------------------------------------------------------------------

function WaiverTable({ players, label }: { players: WaiverPlayer[]; label: string }) {
  if (players.length === 0) {
    return <p className="text-zinc-600 text-sm text-center py-8">No {label.toLowerCase()} available.</p>
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-zinc-800">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-800 bg-zinc-900/60">
            <th className="px-3 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider">Player</th>
            <th className="px-3 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider w-12">Pos</th>
            <th className="px-3 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider w-14">Team</th>
            <th className="px-3 py-3 text-right text-xs font-semibold text-zinc-500 uppercase tracking-wider w-24">Need Score</th>
            <th className="px-3 py-3 text-right text-xs font-semibold text-zinc-500 uppercase tracking-wider w-20">Owned%</th>
            <th className="px-3 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider">Key Stats</th>
            <th className="px-3 py-3 text-center text-xs font-semibold text-zinc-500 uppercase tracking-wider w-16">Add</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-800/60">
          {players.map((p) => {
            const topStats = Object.entries(p.category_contributions)
              .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
              .slice(0, 3)
            return (
              <tr key={p.player_id} className="hover:bg-zinc-800/50 transition-colors">
                <td className="px-3 py-2.5">
                  <div className="font-medium text-zinc-100">{p.name}</div>
                </td>
                <td className="px-3 py-2.5 text-zinc-400 text-xs">{p.position}</td>
                <td className="px-3 py-2.5 text-zinc-400 font-mono text-xs">{p.team}</td>
                <td className="px-3 py-2.5 text-right font-mono text-xs font-semibold text-amber-400 tabular-nums">
                  {p.need_score.toFixed(2)}
                </td>
                <td className="px-3 py-2.5 text-right font-mono text-xs text-zinc-400 tabular-nums">
                  {p.owned_pct.toFixed(1)}%
                </td>
                <td className="px-3 py-2.5 text-xs text-zinc-500 font-mono">
                  {topStats.map(([cat, val]) => (
                    <span key={cat} className="mr-2">
                      <span className="text-zinc-600">{cat}:</span>
                      <span className={cn('ml-1', val >= 0 ? 'text-emerald-400' : 'text-rose-400')}>
                        {val >= 0 ? '+' : ''}{typeof val === 'number' && val % 1 !== 0 ? val.toFixed(2) : val}
                      </span>
                    </span>
                  ))}
                </td>
                <td className="px-3 py-2.5 text-center">
                  <button
                    className="px-2 py-0.5 rounded text-xs font-semibold bg-sky-500/15 text-sky-400 border border-sky-500/30 hover:bg-sky-500/25 transition-colors"
                    disabled
                    title="Waiver add is visual only"
                  >
                    Add
                  </button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// 2-start pitcher table
// ---------------------------------------------------------------------------

function TwoStartTable({ pitchers }: { pitchers: WaiverPlayer[] }) {
  if (pitchers.length === 0) {
    return <p className="text-zinc-600 text-sm text-center py-8">No 2-start pitchers available this week.</p>
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-zinc-800">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-800 bg-zinc-900/60">
            <th className="px-3 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider">Pitcher</th>
            <th className="px-3 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider w-14">Team</th>
            <th className="px-3 py-3 text-right text-xs font-semibold text-zinc-500 uppercase tracking-wider w-28">Starts This Week</th>
            <th className="px-3 py-3 text-right text-xs font-semibold text-zinc-500 uppercase tracking-wider w-20">Owned%</th>
            <th className="px-3 py-3 text-right text-xs font-semibold text-zinc-500 uppercase tracking-wider w-24">Need Score</th>
            <th className="px-3 py-3 text-center text-xs font-semibold text-zinc-500 uppercase tracking-wider w-16">Add</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-800/60">
          {pitchers.map((p) => (
            <tr key={p.player_id} className="hover:bg-zinc-800/50 transition-colors">
              <td className="px-3 py-2.5">
                <div className="font-medium text-zinc-100">{p.name}</div>
                <div className="text-xs text-zinc-500">{p.position}</div>
              </td>
              <td className="px-3 py-2.5 text-zinc-400 font-mono text-xs">{p.team}</td>
              <td className="px-3 py-2.5 text-right font-mono text-xs font-semibold text-sky-400 tabular-nums">
                {p.starts_this_week}
              </td>
              <td className="px-3 py-2.5 text-right font-mono text-xs text-zinc-400 tabular-nums">
                {p.owned_pct.toFixed(1)}%
              </td>
              <td className="px-3 py-2.5 text-right font-mono text-xs font-semibold text-amber-400 tabular-nums">
                {p.need_score.toFixed(2)}
              </td>
              <td className="px-3 py-2.5 text-center">
                <button
                  className="px-2 py-0.5 rounded text-xs font-semibold bg-sky-500/15 text-sky-400 border border-sky-500/30 hover:bg-sky-500/25 transition-colors"
                  disabled
                  title="Waiver add is visual only"
                >
                  Add
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Table skeleton
// ---------------------------------------------------------------------------

function TableSkeleton({ rows = 6 }: { rows?: number }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-zinc-800 animate-pulse">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-800 bg-zinc-900/60">
            {[160, 50, 60, 80, 70, 120, 50].map((w, i) => (
              <th key={i} className="px-3 py-3">
                <div className="h-3 bg-zinc-800 rounded" style={{ width: w }} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-800/60">
          {Array.from({ length: rows }).map((_, i) => (
            <tr key={i}>
              {[160, 50, 60, 80, 70, 120, 50].map((w, j) => (
                <td key={j} className="px-3 py-3">
                  <div className="h-3 bg-zinc-800/70 rounded" style={{ width: w * 0.75 }} />
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
// Page
// ---------------------------------------------------------------------------

export default function WaiverWirePage() {
  const { data, isLoading, isError, refetch, isFetching } = useQuery({
    queryKey: ['fantasy-waiver'],
    queryFn: endpoints.waiverWire,
    refetchInterval: 10 * 60_000,
  })

  return (
    <div className="space-y-6 max-w-6xl">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-xl font-semibold text-zinc-100 flex items-center gap-2">
            <ArrowLeftRight className="h-5 w-5 text-amber-400" />
            Waiver Wire
            {data && (
              <span className="text-zinc-500 font-normal text-base ml-1">
                &mdash; Week ending {data.week_end}
              </span>
            )}
          </h1>
          {data && (
            <p className="text-sm text-zinc-500 mt-0.5">
              vs <span className="text-zinc-300">{data.matchup_opponent}</span>
              &nbsp;&middot; refreshes every 10 min
            </p>
          )}
          {!data && !isLoading && (
            <p className="text-sm text-zinc-500 mt-0.5">H2H category deficit analysis</p>
          )}
        </div>
        <button
          onClick={() => refetch()}
          disabled={isFetching}
          className="flex items-center gap-1.5 px-3 py-2 text-sm text-zinc-400 hover:text-zinc-200 bg-zinc-800 hover:bg-zinc-700 rounded-md transition-colors disabled:opacity-50"
        >
          <RefreshCw className={cn('h-4 w-4', isFetching && 'animate-spin')} />
        </button>
      </div>

      {/* Error state */}
      {isError && (
        <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-5 flex items-center justify-between">
          <div>
            <p className="text-rose-400 font-medium text-sm">Failed to load waiver data</p>
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

      {/* Category tracker */}
      <Card className="p-0">
        <CardHeader className="px-5 pt-5 pb-0 mb-4">
          <CardTitle className="text-base">Category Tracker</CardTitle>
        </CardHeader>
        <div className="px-5 pb-5">
          {isLoading ? (
            <CategorySkeleton />
          ) : isError ? null : data && data.category_deficits.length === 0 ? (
            <p className="text-zinc-600 text-sm text-center py-8">No category data available.</p>
          ) : data ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
              {data.category_deficits.map((cat: CategoryDeficit) => (
                <DeficitCard key={cat.category} cat={cat} />
              ))}
            </div>
          ) : null}
        </div>
      </Card>

      {/* Top available */}
      <Card className="p-0">
        <CardHeader className="px-5 pt-5 pb-0 mb-4">
          <CardTitle className="text-base">Top Available Players</CardTitle>
        </CardHeader>
        <div className="px-5 pb-5">
          {isLoading ? (
            <TableSkeleton rows={8} />
          ) : isError ? null : data ? (
            <WaiverTable players={data.top_available} label="available players" />
          ) : null}
        </div>
      </Card>

      {/* 2-start pitchers */}
      <Card className="p-0">
        <CardHeader className="px-5 pt-5 pb-0 mb-4">
          <CardTitle className="text-base">2-Start Pitchers This Week</CardTitle>
        </CardHeader>
        <div className="px-5 pb-5">
          {isLoading ? (
            <TableSkeleton rows={4} />
          ) : isError ? null : data ? (
            <TwoStartTable pitchers={data.two_start_pitchers} />
          ) : null}
        </div>
      </Card>
    </div>
  )
}
