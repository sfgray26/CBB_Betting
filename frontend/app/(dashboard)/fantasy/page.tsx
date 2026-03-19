'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Trophy, Filter, RefreshCw } from 'lucide-react'
import { endpoints } from '@/lib/api'
import { cn } from '@/lib/utils'
import type { FantasyPlayer } from '@/lib/types'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const TIER_COLORS: Record<number, string> = {
  1: 'text-amber-400 bg-amber-400/10 border-amber-500/30',
  2: 'text-sky-400 bg-sky-400/10 border-sky-500/30',
  3: 'text-emerald-400 bg-emerald-400/10 border-emerald-500/30',
  4: 'text-violet-400 bg-violet-400/10 border-violet-500/30',
  5: 'text-zinc-300 bg-zinc-700/50 border-zinc-600/30',
}

function tierBadge(tier: number) {
  const cls = TIER_COLORS[tier] ?? 'text-zinc-500 bg-zinc-800 border-zinc-700'
  return (
    <span className={cn('px-1.5 py-0.5 rounded border text-xs font-mono font-semibold', cls)}>
      T{tier}
    </span>
  )
}

function positionBadges(positions: string[]) {
  // Show up to 3 positions; deduplicate
  const unique = [...new Set(positions)].slice(0, 3)
  return (
    <span className="flex gap-1 flex-wrap">
      {unique.map((p) => (
        <span key={p} className="px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 text-xs border border-zinc-700">
          {p}
        </span>
      ))}
    </span>
  )
}

function zScoreColor(z: number) {
  if (z >= 2.5) return 'text-amber-400'
  if (z >= 1.5) return 'text-emerald-400'
  if (z >= 0.5) return 'text-sky-400'
  if (z < 0) return 'text-rose-400'
  return 'text-zinc-400'
}

function signedZ(z: number) {
  return `${z >= 0 ? '+' : ''}${z.toFixed(2)}`
}

// Key projected stats to show in the table
function projSummary(player: FantasyPlayer): string {
  const p = player.proj
  if (player.type === 'batter') {
    const parts = []
    if (p.hr != null) parts.push(`${p.hr}HR`)
    if (p.r != null) parts.push(`${p.r}R`)
    if (p.rbi != null) parts.push(`${p.rbi}RBI`)
    if (p.nsb != null && p.nsb > 0) parts.push(`${p.nsb}SB`)
    if (p.avg != null) parts.push(`.${Math.round(p.avg * 1000)}`)
    return parts.slice(0, 4).join(' · ')
  } else {
    const parts = []
    if (p.w != null) parts.push(`${p.w}W`)
    if (p.era != null) parts.push(`${p.era.toFixed(2)}ERA`)
    if (p.whip != null) parts.push(`${p.whip.toFixed(2)}WHIP`)
    if (p.k9 != null) parts.push(`${p.k9.toFixed(1)}K/9`)
    if (p.nsv != null && p.nsv > 0) parts.push(`${p.nsv}SV`)
    return parts.slice(0, 4).join(' · ')
  }
}

// ---------------------------------------------------------------------------
// Filter bar
// ---------------------------------------------------------------------------

const POSITION_OPTIONS = ['All', 'C', '1B', '2B', '3B', 'SS', 'OF', 'SP', 'RP']
const TYPE_OPTIONS = [
  { label: 'All', value: '' },
  { label: 'Batters', value: 'batter' },
  { label: 'Pitchers', value: 'pitcher' },
]

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function FantasyPage() {
  const [posFilter, setPosFilter] = useState('All')
  const [typeFilter, setTypeFilter] = useState('')
  const [tierMax, setTierMax] = useState<number | undefined>(undefined)
  const [search, setSearch] = useState('')

  const queryParams = {
    position: posFilter !== 'All' ? posFilter : undefined,
    player_type: typeFilter || undefined,
    tier_max: tierMax,
    limit: 250,
  }

  const { data, isLoading, isError, refetch, isFetching } = useQuery({
    queryKey: ['fantasy-draft-board', queryParams],
    queryFn: () => endpoints.fantasyDraftBoard(queryParams),
    staleTime: 5 * 60_000,
  })

  const players = (data?.players ?? []).filter((p) =>
    search === '' || p.name.toLowerCase().includes(search.toLowerCase()) ||
    p.team.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-zinc-100 flex items-center gap-2">
            <Trophy className="h-6 w-6 text-amber-400" />
            Fantasy Draft Board
          </h1>
          <p className="text-sm text-zinc-500 mt-1">
            2026 Steamer/ZiPS consensus · {data?.count ?? '—'} players
          </p>
        </div>
        <button
          onClick={() => refetch()}
          disabled={isFetching}
          className="flex items-center gap-2 px-3 py-2 text-sm text-zinc-400 hover:text-zinc-200 bg-zinc-800 hover:bg-zinc-700 rounded-md transition-colors disabled:opacity-50"
        >
          <RefreshCw className={cn('h-4 w-4', isFetching && 'animate-spin')} />
          Refresh
        </button>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 items-center">
        <Filter className="h-4 w-4 text-zinc-500 flex-shrink-0" />

        {/* Search */}
        <input
          type="text"
          placeholder="Search player or team..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="px-3 py-1.5 text-sm bg-zinc-800 border border-zinc-700 rounded-md text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-amber-500 w-48"
        />

        {/* Type filter */}
        <div className="flex rounded-md overflow-hidden border border-zinc-700">
          {TYPE_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              onClick={() => setTypeFilter(opt.value)}
              className={cn(
                'px-3 py-1.5 text-xs font-medium transition-colors',
                typeFilter === opt.value
                  ? 'bg-amber-500/20 text-amber-400'
                  : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200',
              )}
            >
              {opt.label}
            </button>
          ))}
        </div>

        {/* Position filter */}
        <div className="flex flex-wrap gap-1">
          {POSITION_OPTIONS.map((pos) => (
            <button
              key={pos}
              onClick={() => setPosFilter(pos)}
              className={cn(
                'px-2.5 py-1 text-xs rounded border transition-colors',
                posFilter === pos
                  ? 'bg-amber-500/20 text-amber-400 border-amber-500/40'
                  : 'bg-zinc-800 text-zinc-400 border-zinc-700 hover:border-zinc-500',
              )}
            >
              {pos}
            </button>
          ))}
        </div>

        {/* Tier filter */}
        <select
          value={tierMax ?? ''}
          onChange={(e) => setTierMax(e.target.value ? Number(e.target.value) : undefined)}
          className="px-2.5 py-1.5 text-xs bg-zinc-800 border border-zinc-700 rounded-md text-zinc-400 focus:outline-none focus:border-amber-500"
        >
          <option value="">All Tiers</option>
          {[1, 2, 3, 4, 5, 6, 7].map((t) => (
            <option key={t} value={t}>Tier ≤ {t}</option>
          ))}
        </select>
      </div>

      {/* Table */}
      {isLoading ? (
        <div className="flex items-center justify-center h-48 text-zinc-500">Loading draft board...</div>
      ) : isError ? (
        <div className="flex items-center justify-center h-48 text-rose-400">
          Failed to load draft board. Check API connection.
        </div>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-zinc-800">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-zinc-800 bg-zinc-900/60">
                <th className="px-3 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider w-12">#</th>
                <th className="px-3 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider w-8">T</th>
                <th className="px-3 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider">Player</th>
                <th className="px-3 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider w-12">Team</th>
                <th className="px-3 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider">Positions</th>
                <th className="px-3 py-3 text-right text-xs font-semibold text-zinc-500 uppercase tracking-wider w-16">ADP</th>
                <th className="px-3 py-3 text-right text-xs font-semibold text-zinc-500 uppercase tracking-wider w-20">Z-Score</th>
                <th className="px-3 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider">Projections</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/60">
              {players.map((player) => (
                <tr
                  key={player.id}
                  className="hover:bg-zinc-800/40 transition-colors group"
                >
                  <td className="px-3 py-2.5 text-zinc-500 font-mono text-xs">{player.rank}</td>
                  <td className="px-3 py-2.5">{tierBadge(player.tier)}</td>
                  <td className="px-3 py-2.5">
                    <div className="font-medium text-zinc-100">{player.name}</div>
                    <div className="text-xs text-zinc-500 capitalize">{player.type}</div>
                  </td>
                  <td className="px-3 py-2.5 text-zinc-400 font-mono text-xs">{player.team}</td>
                  <td className="px-3 py-2.5">{positionBadges(player.positions)}</td>
                  <td className="px-3 py-2.5 text-right text-zinc-400 font-mono text-xs">
                    {player.adp.toFixed(0)}
                  </td>
                  <td className={cn('px-3 py-2.5 text-right font-mono text-xs font-semibold', zScoreColor(player.z_score))}>
                    {signedZ(player.z_score)}
                  </td>
                  <td className="px-3 py-2.5 text-zinc-400 text-xs font-mono">
                    {projSummary(player)}
                  </td>
                </tr>
              ))}
              {players.length === 0 && (
                <tr>
                  <td colSpan={8} className="px-3 py-8 text-center text-zinc-500">
                    No players match the current filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
