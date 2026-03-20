'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ListChecks, RefreshCw } from 'lucide-react'
import { endpoints } from '@/lib/api'
import { Card, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'
import type { LineupPlayer, StartingPitcher } from '@/lib/types'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function todayStr(): string {
  return new Date().toISOString().slice(0, 10)
}

function formatTime(iso: string): string {
  try {
    const d = new Date(iso)
    return d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true })
  } catch {
    return iso
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
  return (
    <span className="px-2 py-0.5 rounded text-xs font-semibold bg-zinc-800 text-zinc-600 border border-zinc-700">
      UNKNOWN
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
                <div className={`h-3 bg-zinc-800 rounded`} style={{ width: w }} />
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
// Batters table
// ---------------------------------------------------------------------------

function BattersTable({ batters }: { batters: LineupPlayer[] }) {
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
            <th className="px-3 py-3 text-center text-xs font-semibold text-zinc-500 uppercase tracking-wider w-24">Action</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-800/60">
          {sorted.map((b) => (
            <tr key={b.player_id} className="hover:bg-zinc-800/50 transition-colors">
              <td className="px-3 py-2.5">
                <div className="font-medium text-zinc-100">{b.name}</div>
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
              <td className="px-3 py-2.5 text-center">{statusBadge(b.status)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Pitchers table
// ---------------------------------------------------------------------------

function PitchersTable({ pitchers }: { pitchers: StartingPitcher[] }) {
  const scores = pitchers.map((p) => p.sp_score)

  if (pitchers.length === 0) {
    return <p className="text-zinc-600 text-sm text-center py-8">No starting pitchers scheduled for this date.</p>
  }

  const sorted = [...pitchers].sort((a, b) => b.sp_score - a.sp_score)

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
                {p.sp_score.toFixed(3)}
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

  const { data, isLoading, isError, refetch, isFetching } = useQuery({
    queryKey: ['fantasy-lineup', date],
    queryFn: () => endpoints.dailyLineup(date),
    refetchInterval: 5 * 60_000,
  })

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
            onChange={(e) => setDate(e.target.value)}
            className="px-3 py-1.5 text-sm bg-zinc-800 border border-zinc-700 rounded-md text-zinc-200 focus:outline-none focus:border-amber-500"
          />
          <button
            onClick={() => refetch()}
            disabled={isFetching}
            className="flex items-center gap-1.5 px-3 py-2 text-sm text-zinc-400 hover:text-zinc-200 bg-zinc-800 hover:bg-zinc-700 rounded-md transition-colors disabled:opacity-50"
          >
            <RefreshCw className={cn('h-4 w-4', isFetching && 'animate-spin')} />
          </button>
        </div>
      </div>

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

      {/* Batters section */}
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
            <BattersTable batters={data.batters} />
          ) : null}
        </div>
      </Card>

      {/* Starting pitchers section */}
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
    </div>
  )
}
