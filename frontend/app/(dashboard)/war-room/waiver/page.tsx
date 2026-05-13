'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import type { WaiverAvailablePlayer, WaiverResponse } from '@/lib/types'
import {
  ListFilter, Loader2, AlertCircle, TrendingUp, TrendingDown,
  Flame, Snowflake, AlertTriangle, Users, ChevronDown, ChevronUp,
} from 'lucide-react'
import { cn } from '@/lib/utils'

const POSITION_FILTERS = ['All', 'SP', 'RP', 'OF', '1B', '2B', '3B', 'SS', 'C']

// Maps Yahoo-variant category keys → human-readable display labels.
// null = display-only stat (not a scoring category) — filtered out.
const WAIVER_CAT_LABELS: Record<string, string | null> = {
  H_AB: null,       // display-only (Hits/AB string), NOT scored
  IP: null,         // display-only volume stat, NOT scored
  GS: null,         // display-only, NOT scored
  'K(B)': 'K',      // batter strikeouts (lower-is-better for batting)
  'K(P)': 'Ks',     // pitcher strikeouts
  HRA: 'HRA',       // HR allowed (pitcher)
  NSV: 'SV',        // net saves
  // Canonical codes
  R: 'R', H: 'H', HR: 'HR', HR_B: 'HR', HR_P: 'HRA',
  RBI: 'RBI', TB: 'TB', AVG: 'AVG', OPS: 'OPS', NSB: 'NSB', SB: 'SB',
  W: 'W', L: 'L', ERA: 'ERA', WHIP: 'WHIP', K_9: 'K/9', QS: 'QS',
  K_B: 'K', K_P: 'Ks',
}

function waiverCatLabel(key: string): string | null {
  if (key in WAIVER_CAT_LABELS) return WAIVER_CAT_LABELS[key]
  return key  // unknown key: show as-is rather than silently dropping
}

// Rate stats use 2 decimal places; counting stats use whole numbers
const RATE_CATS = new Set(['ERA', 'WHIP', 'AVG', 'OPS', 'K/9', 'K_9', 'K(9)'])
function fmtStat(catKey: string, val: number): string {
  const label = waiverCatLabel(catKey) ?? catKey
  if (RATE_CATS.has(label) || RATE_CATS.has(catKey)) return val.toFixed(2)
  return Number.isInteger(val) ? val.toString() : val.toFixed(1)
}

function NeedBar({ score }: { score: number }) {
  const pct = Math.min(100, Math.max(0, score * 100))
  const color = score >= 0.7 ? 'bg-emerald-400' : score >= 0.4 ? 'bg-amber-400' : 'bg-zinc-600'
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-[#2A2A2A] rounded-full overflow-hidden">
        <div className={cn('h-full rounded-full', color)} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-white tabular-nums w-8 text-right">
        {score.toFixed(2)}
      </span>
    </div>
  )
}

function OwnershipBadge({ pct }: { pct: number | null | undefined }) {
  if (pct === null || pct === undefined) {
    return <span className="text-[10px] text-[#494949]">— owned</span>
  }
  return (
    <span className={cn(
      'text-[10px] tabular-nums',
      pct >= 70 ? 'text-amber-400' : pct >= 30 ? 'text-[#969696]' : 'text-[#494949]',
    )}>
      {pct.toFixed(0)}% owned
    </span>
  )
}

function HotColdBadge({ hotCold }: { hotCold?: string | null }) {
  if (!hotCold) return null
  if (hotCold === 'HOT') {
    return (
      <span className="flex items-center gap-0.5 text-[10px] text-orange-400 font-semibold">
        <Flame className="h-3 w-3" /> HOT
      </span>
    )
  }
  return (
    <span className="flex items-center gap-0.5 text-[10px] text-sky-400 font-semibold">
      <Snowflake className="h-3 w-3" /> COLD
    </span>
  )
}

function PlayerRow({ player }: { player: WaiverAvailablePlayer }) {
  const ownedPct = player.percent_owned ?? player.owned_pct ?? null
  const positions = player.positions ?? (player.position ? [player.position] : [])
  const needMatches = (player.category_need_match ?? []).map((k) => waiverCatLabel(k) ?? k).filter(Boolean)

  return (
    <div className="bg-[#202020] rounded-lg p-4 flex flex-col sm:flex-row sm:items-start gap-3">
      {/* Identity */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <p className="text-sm font-bold text-white truncate">{player.name}</p>
          <HotColdBadge hotCold={player.hot_cold} />
          {player.injury_status && (
            <span className="text-[10px] px-1.5 py-0.5 bg-rose-900/30 text-rose-400 border border-rose-800/40 rounded font-semibold">
              {player.injury_status}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 mt-0.5 flex-wrap">
          <span className="text-xs text-[#7D7D7D]">{player.team}</span>
          <OwnershipBadge pct={ownedPct} />
          {positions.map((pos) => (
            <span key={pos} className="text-[10px] px-1.5 py-0.5 bg-[#2A2A2A] text-[#969696] rounded">
              {pos}
            </span>
          ))}
        </div>
        {player.two_start && (
          <p className="text-[10px] text-emerald-400 mt-1 font-semibold">
            2-START WEEK
            {player.start1_opp ? ` · vs ${player.start1_opp}` : ''}
            {player.start2_opp ? `, ${player.start2_opp}` : ''}
          </p>
        )}
        {/* Category need matches — shows which of your deficits this player addresses */}
        {needMatches.length > 0 && (
          <div className="flex gap-1 mt-1.5 flex-wrap">
            <span className="text-[9px] text-[#494949] uppercase tracking-wider self-center">fills:</span>
            {needMatches.map((label) => (
              <span key={label} className="text-[10px] px-1.5 py-0.5 bg-amber-900/20 text-amber-400 border border-amber-800/30 rounded font-bold">
                {label}
              </span>
            ))}
          </div>
        )}
        {player.statcast_signals && player.statcast_signals.length > 0 && (
          <div className="flex gap-1 mt-1 flex-wrap">
            {player.statcast_signals.map((sig) => (
              <span key={sig} className="text-[10px] px-1.5 py-0.5 bg-[#1A2A1A] text-emerald-500 border border-emerald-900/40 rounded">
                {sig}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Match score — higher = better fits your team's current category deficits */}
      <div className="w-full sm:w-40 flex-shrink-0">
        <p className="text-[9px] text-[#494949] uppercase tracking-wider mb-1">Match Score</p>
        <NeedBar score={player.need_score} />
        <p className="text-[9px] text-[#494949] mt-0.5">fit for your gaps</p>
      </div>
    </div>
  )
}

function CategoryDeficitsBar({ deficits, opponent }: {
  deficits: WaiverResponse['category_deficits']
  opponent?: string
}) {
  const [expanded, setExpanded] = useState(false)

  // Filter out display-only stats (H_AB, IP, GS) and any other non-scoring keys
  const scored = (deficits ?? []).filter((d) => waiverCatLabel(d.category) !== null)
  if (scored.length === 0) return null

  const behind = scored.filter((d) => !d.winning)
  const leading = scored.filter((d) => d.winning)
  // Tied = deficit exactly 0 in both directions
  const tied = scored.filter((d) => d.deficit === 0)
  const netBehind = behind.length - tied.length

  // Show first 4 behind, expand to show all
  const behindVisible = expanded ? behind : behind.slice(0, 4)
  const leadingVisible = expanded ? leading : leading.slice(0, 3)
  const hasMore = behind.length > 4 || leading.length > 3

  return (
    <div className="bg-[#202020] rounded-lg p-4 space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-[10px] font-semibold tracking-widest uppercase text-[#7D7D7D]">
            This Week&apos;s Matchup{opponent ? ` vs ${opponent}` : ''}
          </p>
          <p className="text-xs text-[#494949] mt-0.5">
            {netBehind > 0 ? (
              <span className="text-rose-400 font-semibold">Trailing in {behind.length}</span>
            ) : (
              <span className="text-emerald-400 font-semibold">Leading in {leading.length}</span>
            )}
            {' · '}
            <span className={leading.length > behind.length ? 'text-emerald-400' : 'text-[#494949]'}>
              {leading.length} W
            </span>
            {' · '}
            <span className={behind.length > 0 ? 'text-rose-400' : 'text-[#494949]'}>
              {behind.length} L
            </span>
            {tied.length > 0 && <span className="text-[#7D7D7D]"> · {tied.length} T</span>}
          </p>
        </div>
        {hasMore && (
          <button onClick={() => setExpanded(!expanded)} className="text-[10px] text-[#7D7D7D] hover:text-white flex items-center gap-0.5">
            {expanded ? <><ChevronUp className="h-3 w-3" /> Less</> : <><ChevronDown className="h-3 w-3" /> More</>}
          </button>
        )}
      </div>

      {/* Behind categories */}
      {behind.length > 0 && (
        <div className="space-y-1">
          <p className="text-[9px] font-bold uppercase tracking-widest text-rose-400">▼ Behind</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
            {behindVisible.map((d) => {
              const label = waiverCatLabel(d.category) ?? d.category
              const gap = Math.abs(d.deficit)
              return (
                <div key={d.category} className="flex items-center justify-between bg-rose-950/20 border border-rose-900/30 rounded px-2.5 py-1.5">
                  <div className="flex items-center gap-1.5">
                    <TrendingDown className="h-3 w-3 text-rose-400 flex-shrink-0" />
                    <span className="text-[11px] font-bold text-rose-300 w-10">{label}</span>
                  </div>
                  <div className="flex items-center gap-1.5 text-[10px] tabular-nums">
                    <span className="text-white">{fmtStat(d.category, d.my_total)}</span>
                    <span className="text-[#494949]">vs</span>
                    <span className="text-[#7D7D7D]">{fmtStat(d.category, d.opponent_total)}</span>
                    <span className="text-rose-400 font-bold ml-1">−{fmtStat(d.category, gap)}</span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Leading categories */}
      {leading.length > 0 && (
        <div className="space-y-1">
          <p className="text-[9px] font-bold uppercase tracking-widest text-emerald-400">▲ Leading</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
            {leadingVisible.map((d) => {
              const label = waiverCatLabel(d.category) ?? d.category
              const lead = Math.abs(d.deficit)
              return (
                <div key={d.category} className="flex items-center justify-between bg-emerald-950/20 border border-emerald-900/30 rounded px-2.5 py-1.5">
                  <div className="flex items-center gap-1.5">
                    <TrendingUp className="h-3 w-3 text-emerald-400 flex-shrink-0" />
                    <span className="text-[11px] font-bold text-emerald-300 w-10">{label}</span>
                  </div>
                  <div className="flex items-center gap-1.5 text-[10px] tabular-nums">
                    <span className="text-white">{fmtStat(d.category, d.my_total)}</span>
                    <span className="text-[#494949]">vs</span>
                    <span className="text-[#7D7D7D]">{fmtStat(d.category, d.opponent_total)}</span>
                    <span className="text-emerald-400 font-bold ml-1">+{fmtStat(d.category, lead)}</span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

export default function WaiverPage() {
  const [sort, setSort] = useState<'need_score' | 'projected_points'>('need_score')
  const [posFilter, setPosFilter] = useState('All')

  const { data, isLoading, isError, error, refetch } = useQuery({
    queryKey: ['waiver', sort],
    queryFn: () => endpoints.getWaiver(sort),
    staleTime: 3 * 60_000,
  })

  const filterPlayers = (players: WaiverAvailablePlayer[]) => {
    if (posFilter === 'All') return players
    return players.filter((p) => {
      const positions = p.positions ?? (p.position ? [p.position] : [])
      return positions.some((pos) => pos === posFilter || pos.startsWith(posFilter))
    })
  }

  if (isLoading) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center">
        <div className="flex items-center gap-2 text-[#7D7D7D]">
          <Loader2 className="h-5 w-5 animate-spin text-[#FFC000]" />
          <span className="text-sm">Loading waiver wire…</span>
        </div>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center">
        <div className="bg-[#202020] rounded-lg p-6 max-w-md w-full">
          <div className="flex items-center gap-2 text-rose-400 mb-2">
            <AlertCircle className="h-5 w-5" />
            <span className="text-sm font-semibold">Failed to load waiver wire</span>
          </div>
          <p className="text-[#969696] text-sm">
            {error instanceof Error ? error.message : 'Unknown error'}
          </p>
          <button onClick={() => refetch()} className="mt-4 text-xs text-[#FFC000] hover:text-amber-300 font-semibold">
            Retry
          </button>
        </div>
      </div>
    )
  }

  const topAvailable = filterPlayers(data?.top_available ?? [])
  const twoStarters = filterPlayers(data?.two_start_pitchers ?? [])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <ListFilter className="h-3.5 w-3.5 text-[#FFC000]" />
          <span className="text-xs font-bold tracking-widest uppercase text-[#FFC000]">
            Waiver Wire
          </span>
        </div>
        <div className="flex items-center gap-2 text-[10px] text-[#494949]">
          {data?.il_slots_available != null && data.il_slots_available > 0 && (
            <span className="text-emerald-400">
              {data.il_slots_available} IL slot{data.il_slots_available > 1 ? 's' : ''} open
            </span>
          )}
        </div>
      </div>

      {/* Alerts */}
      {data?.urgent_alert && (
        <div className="bg-amber-900/20 border border-amber-700/40 rounded-lg p-3 flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-amber-400 flex-shrink-0" />
          <span className="text-sm text-amber-300">{data.urgent_alert.message}</span>
        </div>
      )}
      {data?.closer_alert === 'NO_CLOSERS' && (
        <div className="bg-rose-900/20 border border-rose-700/40 rounded-lg p-3 flex items-center gap-2">
          <AlertCircle className="h-4 w-4 text-rose-400 flex-shrink-0" />
          <span className="text-sm text-rose-300">
            No closers on your roster — consider adding saves coverage.
          </span>
        </div>
      )}

      {/* Category deficits */}
      {data?.category_deficits && (
        <CategoryDeficitsBar
          deficits={data.category_deficits}
          opponent={data.matchup_opponent}
        />
      )}

      {/* Sort + position filter controls */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="flex items-center gap-1 bg-[#202020] rounded-lg p-1">
          {(['need_score', 'projected_points'] as const).map((s) => (
            <button
              key={s}
              onClick={() => setSort(s)}
              className={cn(
                'text-[10px] px-3 py-1.5 rounded font-semibold tracking-wider uppercase transition-colors',
                sort === s ? 'bg-[#FFC000] text-black' : 'text-[#7D7D7D] hover:text-white',
              )}
            >
              {s === 'need_score' ? 'Match Score' : 'Projected'}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1 flex-wrap">
          {POSITION_FILTERS.map((pos) => (
            <button
              key={pos}
              onClick={() => setPosFilter(pos)}
              className={cn(
                'text-[10px] px-2.5 py-1 rounded font-semibold tracking-wider transition-colors',
                posFilter === pos
                  ? 'bg-[#2A2A2A] text-white border border-[#494949]'
                  : 'text-[#494949] hover:text-[#969696]',
              )}
            >
              {pos}
            </button>
          ))}
        </div>
      </div>

      {/* Two-Start Pitchers */}
      {twoStarters.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-3.5 w-3.5 text-emerald-400" />
            <p className="text-xs font-semibold tracking-widest uppercase text-emerald-400">
              Two-Start Pitchers · {twoStarters.length}
            </p>
          </div>
          {twoStarters.map((p) => <PlayerRow key={p.player_id} player={p} />)}
        </div>
      )}

      {/* Top Available */}
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <Users className="h-3.5 w-3.5 text-[#FFC000]" />
          <p className="text-xs font-semibold tracking-widest uppercase text-[#7D7D7D]">
            Top Available · {topAvailable.length}
          </p>
        </div>
        {topAvailable.length === 0 ? (
          <div className="bg-[#202020] rounded-lg p-8 text-center">
            <p className="text-[#494949] text-sm">No players match this filter.</p>
          </div>
        ) : (
          topAvailable.map((p) => <PlayerRow key={p.player_id} player={p} />)
        )}
      </div>
    </div>
  )
}
