'use client'

import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeftRight, RefreshCw, Clock } from 'lucide-react'
import { endpoints } from '@/lib/api'
import { Card, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'
import { StatusBadge } from '@/components/shared/status-badge'
import type { CategoryDeficit, WaiverPlayer, WaiverRecommendation } from '@/lib/types'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const PRIORITY_STATS = ['HR', 'RBI', 'AVG', 'ERA', 'WHIP', 'K', 'NSV']
// Stats dict is now keyed by abbreviation (translated server-side via league settings sid_map).
const RATIO_DISPLAY = new Set(['AVG', 'ERA', 'WHIP'])

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
// Skeletons
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
                  <div className="flex items-center gap-1.5 flex-wrap">
                    <span className="font-medium text-zinc-100">{p.name}</span>
                    {p.hot_cold === 'HOT' && (
                      <span className="text-xs font-mono bg-red-500/20 text-red-400 border border-red-500/30 px-1.5 py-0.5 rounded">
                        HOT
                      </span>
                    )}
                    {p.hot_cold === 'COLD' && (
                      <span className="text-xs font-mono bg-blue-500/20 text-blue-400 border border-blue-500/30 px-1.5 py-0.5 rounded">
                        COLD
                      </span>
                    )}
                    {p.status && (
                      <StatusBadge status={p.status} />
                    )}
                  </div>
                  {p.statcast_signals && p.statcast_signals.length > 0 && (
                    <div className="flex gap-1 mt-0.5">
                      {p.statcast_signals.map((sig) => (
                        <span
                          key={sig}
                          className={cn(
                            'px-1 py-0.5 rounded text-xs font-semibold',
                            sig === 'BUY_LOW'
                              ? 'bg-emerald-500/15 text-emerald-400'
                              : sig === 'BREAKOUT'
                              ? 'bg-sky-500/15 text-sky-400'
                              : 'bg-zinc-700 text-zinc-400',
                          )}
                        >
                          {sig}
                        </span>
                      ))}
                    </div>
                  )}
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
                  {p.stats && Object.keys(p.stats).length > 0 ? (
                    <div className="flex gap-x-3 gap-y-1 flex-wrap">
                      {PRIORITY_STATS.map((abbr) => {
                        // stats dict is keyed by abbreviation (translated server-side)
                        const raw = p.stats![abbr]
                        if (raw === null || raw === undefined || raw === '') return null
                        const display = RATIO_DISPLAY.has(abbr)
                          ? parseFloat(String(raw)).toFixed(3)
                          : raw
                        return (
                          <span key={abbr}>
                            <span className="text-zinc-600">{abbr}:</span>
                            <span className="ml-1 text-zinc-300">{display}</span>
                          </span>
                        )
                      })}
                    </div>
                  ) : (
                    topStats.map(([cat, val]) => (
                      <span key={cat} className="mr-2">
                        <span className="text-zinc-600">{cat}:</span>
                        <span className={cn('ml-1', val >= 0 ? 'text-emerald-400' : 'text-rose-400')}>
                          {val >= 0 ? '+' : ''}{typeof val === 'number' && val % 1 !== 0 ? val.toFixed(2) : val}
                        </span>
                      </span>
                    ))
                  )}
                </td>
                <td className="px-3 py-2.5 text-center">
                  <a
                    href="https://baseball.fantasysports.yahoo.com/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="px-2 py-0.5 rounded text-xs font-semibold bg-sky-500/15 text-sky-400 border border-sky-500/30 hover:bg-sky-500/25 transition-colors"
                    title="Add on Yahoo Fantasy"
                  >
                    Add
                  </a>
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
// Helpers
// ---------------------------------------------------------------------------

const ACTION_LABELS: Record<string, string> = {
  ADD_DROP: 'Add / Drop',
  ADD: 'Add',
  DROP: 'Drop',
  HOLD: 'Hold',
}

const SIGNAL_LABELS: Record<string, string> = {
  BUY_LOW: 'Buy Low',
  BREAKOUT: 'Breakout',
  SELL_HIGH: 'Sell High',
}

// ---------------------------------------------------------------------------
// Recommendation card
// ---------------------------------------------------------------------------

function RecCard({ rec }: { rec: WaiverRecommendation }) {
  // Strip [BUY_LOW ...] tags from rationale
  const rationale = rec.rationale.replace(/\[BUY_LOW[^\]]*\]/gi, '').replace(/\[BREAKOUT[^\]]*\]/gi, '').trim()

  return (
    <div className="rounded-lg border border-zinc-700 bg-zinc-900/60 p-4 space-y-2">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-2">
          <span
            className={cn(
              'px-2 py-0.5 rounded text-xs font-bold',
              rec.action === 'ADD_DROP'
                ? 'bg-amber-500/15 text-amber-400'
                : 'bg-sky-500/15 text-sky-400',
            )}
          >
            {ACTION_LABELS[rec.action] ?? rec.action}
          </span>
          <span className="text-zinc-100 font-medium text-sm">{rec.add_player?.name}</span>
          {rec.drop_player_name && (
            <span className="text-zinc-500 text-xs">
              &rarr; drop {rec.drop_player_name}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs font-mono text-amber-400 tabular-nums">
            edge: +{rec.need_score.toFixed(2)}
          </span>
          {rec.mcmc_enabled && (
            <span className="text-xs font-mono text-emerald-400 tabular-nums">
              WP: {(rec.win_prob_before * 100).toFixed(0)}% &rarr;{' '}
              {(rec.win_prob_after * 100).toFixed(0)}%
            </span>
          )}
          {rec.statcast_signals.map((sig) => (
            <span
              key={sig}
              className={cn(
                'px-1.5 py-0.5 rounded text-xs font-semibold',
                sig === 'BUY_LOW'
                  ? 'bg-emerald-500/15 text-emerald-400'
                  : sig === 'BREAKOUT'
                  ? 'bg-sky-500/15 text-sky-400'
                  : 'bg-zinc-700 text-zinc-400',
              )}
            >
              {SIGNAL_LABELS[sig] ?? sig}
            </span>
          ))}
        </div>
      </div>
      <p className="text-xs text-zinc-500 leading-relaxed">{rationale}</p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

const POSITIONS = ['C', '1B', '2B', '3B', 'SS', 'OF', 'SP', 'RP']

export default function WaiverWirePage() {
  const [position, setPosition] = useState<string>('')
  const [sort, setSort] = useState<'need_score' | 'percent_owned'>('need_score')
  const [maxOwned, setMaxOwned] = useState<number>(100)
  const [page, setPage] = useState<number>(1)
  const [showRecs, setShowRecs] = useState<boolean>(false)
  const [isStuck, setIsStuck] = useState(false)

  const { data, isLoading, isError, error, refetch, isFetching } = useQuery({
    queryKey: ['fantasy-waiver', position, sort, maxOwned, page],
    queryFn: () =>
      endpoints.waiverWire({
        position: position || undefined,
        sort,
        max_percent_owned: maxOwned,
        page,
      }),
    refetchInterval: 10 * 60_000,
  })

  useEffect(() => {
    let timer: NodeJS.Timeout
    if (isLoading) {
      setIsStuck(false)
      timer = setTimeout(() => {
        setIsStuck(true)
      }, 15000)
    } else {
      setIsStuck(false)
    }
    return () => clearTimeout(timer)
  }, [isLoading])

  const {
    data: recData,
    isLoading: recLoading,
    isError: recError,
    refetch: recRefetch,
  } = useQuery({
    queryKey: ['fantasy-waiver-recs'],
    queryFn: endpoints.waiverRecommendations,
    enabled: showRecs,
    retry: 1,
  })

  return (
    <div className="space-y-6 max-w-6xl">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-xl font-semibold text-zinc-100 flex items-center gap-2 flex-wrap">
            <ArrowLeftRight className="h-5 w-5 text-amber-400" />
            Waiver Wire
            {data && (
              <span className="text-zinc-500 font-normal text-base ml-1">
                &mdash; Week ending {data.week_end}
              </span>
            )}
            {data?.faab_balance != null && (
              <span className="text-xs font-mono bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 px-2 py-0.5 rounded">
                FAAB: ${data.faab_balance.toFixed(0)}
              </span>
            )}
          </h1>
          {data && (
            <p className="text-sm text-zinc-500 mt-0.5">
              {data.matchup_opponent && data.matchup_opponent !== 'TBD' && data.matchup_opponent !== '' && (
                <>vs <span className="text-zinc-300">{data.matchup_opponent}</span>&nbsp;&middot; </>
              )}
              refreshes every 10 min
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

      {/* Filter controls */}
      <div className="flex flex-wrap items-center gap-3">
        <select
          value={position}
          onChange={(e) => {
            setPosition(e.target.value)
            setPage(1)
          }}
          className="px-3 py-1.5 text-sm bg-zinc-800 border border-zinc-700 rounded-md text-zinc-200 focus:outline-none focus:border-amber-500"
        >
          <option value="">All Positions</option>
          {POSITIONS.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>

        <select
          value={sort}
          onChange={(e) => setSort(e.target.value as 'need_score' | 'percent_owned')}
          className="px-3 py-1.5 text-sm bg-zinc-800 border border-zinc-700 rounded-md text-zinc-200 focus:outline-none focus:border-amber-500"
        >
          <option value="need_score">Sort: Need Score</option>
          <option value="percent_owned">Sort: % Owned</option>
        </select>

        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-500">Max owned:</span>
          <input
            type="range"
            min={0}
            max={100}
            value={maxOwned}
            onChange={(e) => setMaxOwned(Number(e.target.value))}
            className="w-24 accent-amber-400"
          />
          <span className="text-xs font-mono text-zinc-400 tabular-nums w-10">{maxOwned}%</span>
        </div>
      </div>

      {/* Urgent injury banner */}
      {data?.urgent_alert && (
        <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-4 flex items-start gap-3">
          <span className="text-amber-400 text-lg leading-none mt-0.5">&#9888;</span>
          <div>
            <p className="text-amber-400 font-semibold text-sm">
              {data.urgent_alert.player} ({data.urgent_alert.position}) &mdash; Replacement Needed
            </p>
            <p className="text-amber-400/70 text-xs mt-0.5">{data.urgent_alert.message}</p>
          </div>
        </div>
      )}

      {/* Error state */}
      {isError && (
        <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-5 flex items-center justify-between">
          <div>
            <p className="text-rose-400 font-medium text-sm">Failed to load waiver data</p>
            <p className="text-rose-400/60 text-xs mt-0.5">
              {error instanceof Error
                ? error.message
                : 'Yahoo API error — check credentials in Railway.'}
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
            isStuck ? (
              <div className="py-12 flex flex-col items-center justify-center border border-zinc-800 rounded-lg bg-zinc-900/40 space-y-4">
                <div className="flex items-center gap-2 text-amber-400">
                  <Clock className="h-5 w-5" />
                  <span className="text-sm font-medium">Taking longer than expected</span>
                </div>
                <p className="text-xs text-zinc-500 text-center max-w-xs">
                  Yahoo&apos;s API may be slow. If this persists, try refreshing the page.
                </p>
                <button
                  onClick={() => refetch()}
                  className="flex items-center gap-2 px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-xs rounded-md transition-colors border border-zinc-700"
                >
                  <RefreshCw className="h-3 w-3" />
                  Retry
                </button>
              </div>
            ) : (
              <CategorySkeleton />
            )
          ) : isError ? null : !data || data.category_deficits.length === 0 ? (
            <p className="text-zinc-600 text-sm text-center py-8">No category data available.</p>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
              {data.category_deficits.map((cat: CategoryDeficit) => (
                <DeficitCard key={cat.category} cat={cat} />
              ))}
            </div>
          )}
        </div>
      </Card>

      {/* Top available */}
      <Card className="p-0">
        <CardHeader className="px-5 pt-5 pb-0 mb-4">
          <CardTitle className="text-base">
            Top Available Players
            {position && (
              <span className="text-zinc-500 font-normal ml-2">&mdash; {position}</span>
            )}
          </CardTitle>
        </CardHeader>
        <div className="px-5 pb-5">
          {isLoading ? (
            !isStuck && <TableSkeleton rows={8} />
          ) : isError ? null : data && data.top_available.length === 0 ? (
            <div className="py-12 flex flex-col items-center justify-center border border-dashed border-zinc-800 rounded-lg">
              <p className="text-zinc-400 text-sm font-medium">No waiver targets found</p>
              <p className="text-zinc-600 text-xs mt-1">Projections update daily after 6 AM ET.</p>
            </div>
          ) : data ? (
            <WaiverTable players={data.top_available} label="available players" />
          ) : null}
        </div>
      </Card>

      {/* Pagination */}
      {data?.pagination && (
        <div className="flex items-center justify-between">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page === 1}
            className="px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 disabled:opacity-30 rounded-md transition-colors text-zinc-300 text-xs"
          >
            Previous
          </button>
          <span className="text-xs font-mono text-zinc-500">Page {page}</span>
          <button
            onClick={() => setPage((p) => p + 1)}
            disabled={!data.pagination.has_next}
            className="px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 disabled:opacity-30 rounded-md transition-colors text-zinc-300 text-xs"
          >
            Next
          </button>
        </div>
      )}

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

      {/* ADD/DROP Recommendations */}
      <Card className="p-0">
        <CardHeader className="px-5 pt-5 pb-0 mb-4 flex flex-row items-center justify-between">
          <CardTitle className="text-base">ADD/DROP Recommendations</CardTitle>
          {!showRecs ? (
            <button
              onClick={() => setShowRecs(true)}
              className="px-3 py-1.5 text-xs bg-amber-500/15 hover:bg-amber-500/25 text-amber-400 border border-amber-500/30 rounded-md transition-colors"
            >
              Load Recommendations
            </button>
          ) : (
            <button
              onClick={() => recRefetch()}
              disabled={recLoading}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-zinc-800 hover:bg-zinc-700 disabled:opacity-50 text-zinc-300 rounded-md transition-colors"
            >
              <RefreshCw className={cn('h-3 w-3', recLoading && 'animate-spin')} />
              Refresh
            </button>
          )}
        </CardHeader>
        <div className="px-5 pb-5">
          {!showRecs ? (
            <p className="text-zinc-600 text-sm text-center py-8">
              Click &ldquo;Load Recommendations&rdquo; to run roster analysis vs free agents.
            </p>
          ) : recLoading ? (
            <div className="space-y-3 animate-pulse">
              {Array.from({ length: 3 }).map((_, i) => (
                <div key={i} className="rounded-lg border border-zinc-800 p-4 flex items-start gap-3">
                  <div className="h-8 w-8 bg-zinc-800 rounded-full flex-shrink-0" />
                  <div className="flex-1 space-y-2">
                    <div className="h-3 bg-zinc-800 rounded w-32" />
                    <div className="h-3 bg-zinc-800/70 rounded w-48" />
                    <div className="h-3 bg-zinc-800/50 rounded w-24" />
                  </div>
                  <div className="h-6 w-12 bg-zinc-800 rounded flex-shrink-0" />
                </div>
              ))}
            </div>
          ) : recError ? (
            <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-4">
              <p className="text-rose-400 text-sm font-medium">Failed to load recommendations</p>
              <p className="text-rose-400/60 text-xs mt-0.5">
                Yahoo auth may be required or roster is empty.
              </p>
            </div>
          ) : !recData?.recommendations.length ? (
            <p className="text-zinc-600 text-sm text-center py-8">
              No strong moves found. Roster looks solid!
            </p>
          ) : (
            <div className="space-y-3">
              {recData.recommendations.map((rec, i) => (
                <RecCard key={i} rec={rec} />
              ))}
            </div>
          )}
        </div>
      </Card>
    </div>
  )
}
