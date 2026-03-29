'use client'

import { useQuery } from '@tanstack/react-query'
import { Swords, RefreshCw, AlertTriangle, Trophy } from 'lucide-react'
import { endpoints } from '@/lib/api'
import { Card, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'
import type { MatchupResponse } from '@/lib/types'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Known stat display names for the 18-category H2H league
// Includes both Yahoo string keys and numeric category IDs
const STAT_LABELS: Record<string, string> = {
  // Batting — string keys
  R: 'Runs',         H: 'Hits',          HR: 'Home Runs',
  RBI: 'RBI',        SB: 'Stolen Bases', OBP: 'On-Base %',
  AVG: 'Batting Avg', OPS: 'OPS',
  // Batting — numeric IDs
  '3': 'Batting Avg', '7': 'Runs',       '8': 'Hits',
  '12': 'Home Runs',  '13': 'RBI',       '16': 'Stolen Bases',
  '55': 'OPS',        '60': 'Hits',      '85': 'On-Base %',
  // Pitching — string keys
  W: 'Wins',   K: 'Strikeouts', SV: 'Saves', HLD: 'Holds',
  ERA: 'ERA',  WHIP: 'WHIP',    QS: 'Quality Starts',
  BB: 'Walks (P)', IP: 'Innings Pitched', K9: 'K/9',
  NSV: 'Net Saves',
  // Pitching — numeric IDs
  '21': 'Innings Pitched', '23': 'Wins',          '26': 'ERA',
  '27': 'WHIP',            '28': 'Strikeouts',    '29': 'Quality Starts',
  '32': 'Saves',           '38': 'K/BB',          '42': 'Strikeouts',
  '50': 'Innings Pitched', '57': 'Walks (P)',     '62': 'Games Started',
  '83': 'Net Saves',
}

// Ratio stats get 3 decimal places; counting stats get integer display
const RATIO_STATS = new Set(['AVG', 'OBP', 'OPS', 'ERA', 'WHIP', 'K9', '3', '26', '27', '85', '55'])

// ERA / WHIP: lower is better
const LOWER_IS_BETTER = new Set(['ERA', 'WHIP', '26', '27'])

function winClass(cat: string, myVal: number, oppVal: number): string {
  if (myVal === oppVal) return 'text-zinc-400'
  const myWins = LOWER_IS_BETTER.has(cat) ? myVal < oppVal : myVal > oppVal
  return myWins ? 'text-emerald-400 font-semibold' : 'text-rose-400'
}

function formatVal(val: string | number | undefined, cat?: string): string {
  if (val === undefined || val === null) return '-'
  const n = parseFloat(String(val))
  if (isNaN(n)) return String(val)
  if (cat && RATIO_STATS.has(cat)) return n.toFixed(3)
  // IP special case: 6.1 (not 6)
  if (cat === 'IP' || cat === '21' || cat === '50') return n.toFixed(1)
  // Integer counting stats
  return Number.isInteger(n) ? n.toLocaleString() : n.toFixed(1)
}

// ---------------------------------------------------------------------------
// Stat comparison table
// ---------------------------------------------------------------------------

function MatchupTable({ data }: { data: MatchupResponse }) {
  const allCats = Object.keys(data.my_team.stats)
  if (allCats.length === 0) {
    return (
      <p className="text-zinc-500 text-sm text-center py-8">
        No stats yet — all zeros at the start of the week.
      </p>
    )
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-zinc-800">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-800 bg-zinc-900/60">
            <th className="px-4 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider w-32">
              Category
            </th>
            <th className="px-4 py-3 text-right text-xs font-semibold text-zinc-500 uppercase tracking-wider">
              {data.my_team.team_name}
            </th>
            <th className="px-4 py-3 text-right text-xs font-semibold text-zinc-500 uppercase tracking-wider">
              {data.opponent.team_name}
            </th>
            <th className="px-4 py-3 text-center text-xs font-semibold text-zinc-500 uppercase tracking-wider w-20">
              Edge
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-800/60">
          {allCats.map((cat) => {
            const myRaw = data.my_team.stats[cat]
            const oppRaw = data.opponent.stats[cat]
            const myVal = parseFloat(String(myRaw ?? 0))
            const oppVal = parseFloat(String(oppRaw ?? 0))
            const myWins = LOWER_IS_BETTER.has(cat) ? myVal < oppVal : myVal > oppVal
            const tied = myVal === oppVal

            return (
              <tr key={cat} className="hover:bg-zinc-800/40 transition-colors">
                <td className="px-4 py-2.5 text-zinc-400 font-medium">
                  {STAT_LABELS[cat] ?? cat}
                </td>
                <td className={cn('px-4 py-2.5 text-right font-mono tabular-nums', winClass(cat, myVal, oppVal))}>
                  {formatVal(myRaw, cat)}
                </td>
                <td className={cn('px-4 py-2.5 text-right font-mono tabular-nums', winClass(cat, oppVal, myVal))}>
                  {formatVal(oppRaw, cat)}
                </td>
                <td className="px-4 py-2.5 text-center text-xs">
                  {tied ? (
                    <span className="text-zinc-600">TIE</span>
                  ) : myWins ? (
                    <span className="text-emerald-400 font-semibold">WIN</span>
                  ) : (
                    <span className="text-rose-400">LOSS</span>
                  )}
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
// Score summary banner
// ---------------------------------------------------------------------------

function ScoreBanner({ data }: { data: MatchupResponse }) {
  const allCats = Object.keys(data.my_team.stats)
  let myWins = 0
  let oppWins = 0
  let ties = 0

  allCats.forEach((cat) => {
    const myVal = parseFloat(String(data.my_team.stats[cat] ?? 0))
    const oppVal = parseFloat(String(data.opponent.stats[cat] ?? 0))
    if (myVal === oppVal) { ties++; return }
    if (LOWER_IS_BETTER.has(cat) ? myVal < oppVal : myVal > oppVal) myWins++
    else oppWins++
  })

  const leading = myWins > oppWins ? 'leading' : myWins < oppWins ? 'trailing' : 'tied'
  const bannerClass =
    leading === 'leading'
      ? 'border-emerald-500/30 bg-emerald-500/10'
      : leading === 'trailing'
      ? 'border-rose-500/30 bg-rose-500/10'
      : 'border-zinc-700 bg-zinc-800/40'

  return (
    <div className={cn('rounded-lg border p-5 flex items-center justify-between gap-4', bannerClass)}>
      <div className="text-center flex-1">
        <p className="text-2xl font-bold text-zinc-100 tabular-nums">{myWins}</p>
        <p className="text-xs text-zinc-500 mt-1 truncate">{data.my_team.team_name}</p>
      </div>
      <div className="text-center px-4">
        <p className="text-xs text-zinc-500 uppercase tracking-widest font-semibold">
          Week {data.week ?? '?'}{data.is_playoffs ? ' (Playoffs)' : ''}
        </p>
        <p className="text-zinc-600 text-sm mt-1">vs</p>
        {ties > 0 && <p className="text-xs text-zinc-600 mt-1">{ties} tied</p>}
      </div>
      <div className="text-center flex-1">
        <p className="text-2xl font-bold text-zinc-100 tabular-nums">{oppWins}</p>
        <p className="text-xs text-zinc-500 mt-1 truncate">{data.opponent.team_name}</p>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function MatchupPage() {
  const { data, isLoading, isError, error, refetch, isFetching } = useQuery({
    queryKey: ['fantasy-matchup'],
    queryFn: endpoints.fantasyMatchup,
    refetchInterval: 10 * 60_000,
    retry: 1,
  })

  const errorMsg = error instanceof Error ? error.message : ''
  const isYahooNotConfigured = isError && errorMsg.startsWith('503')

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-xl font-semibold text-zinc-100 flex items-center gap-2">
            <Swords className="h-5 w-5 text-amber-400" />
            Current Matchup
          </h1>
          <p className="text-sm text-zinc-500 mt-0.5">
            Live H2H category standings from Yahoo Fantasy
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

      {/* Error: Yahoo not configured */}
      {isError && isYahooNotConfigured && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-5 flex items-start gap-3">
          <AlertTriangle className="h-5 w-5 text-amber-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-amber-300 font-medium text-sm">Yahoo not configured</p>
            <p className="text-amber-300/60 text-xs mt-0.5">
              {errorMsg || 'Set YAHOO_CLIENT_ID, YAHOO_CLIENT_SECRET, and YAHOO_REFRESH_TOKEN in Railway.'}
            </p>
          </div>
        </div>
      )}

      {/* Error: other */}
      {isError && !isYahooNotConfigured && (
        <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-5 flex items-center justify-between">
          <div>
            <p className="text-rose-400 font-medium text-sm">Failed to load matchup</p>
            <p className="text-rose-400/60 text-xs mt-0.5">
              {error instanceof Error ? error.message : 'Unknown error'}
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

      {/* Loading skeleton */}
      {isLoading && (
        <div className="space-y-4 animate-pulse">
          <div className="h-24 rounded-lg bg-zinc-800/60" />
          <div className="h-64 rounded-lg bg-zinc-800/40" />
        </div>
      )}

      {/* Pre-season / no matchup message */}
      {data?.message && (
        <div className="bg-sky-500/10 border border-sky-500/30 rounded-lg p-3 text-sky-300 text-sm">
          {data.message}
        </div>
      )}

      {/* Content */}
      {data && (
        <>
          <ScoreBanner data={data} />
          <Card className="p-0">
            <CardHeader className="px-5 pt-5 pb-0 mb-4">
              <CardTitle className="text-base flex items-center gap-2">
                <Trophy className="h-4 w-4 text-amber-400" />
                Category Breakdown
              </CardTitle>
            </CardHeader>
            <div className="px-5 pb-5">
              <MatchupTable data={data} />
            </div>
          </Card>
        </>
      )}
    </div>
  )
}
