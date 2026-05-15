'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import type { WaiverAvailablePlayer, WaiverResponse, WaiverRosterPlayer } from '@/lib/types'
import {
  ListFilter, Loader2, AlertCircle, TrendingUp,
  Flame, Snowflake, AlertTriangle, Users,
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

function NeedBar({ score }: { score: number }) {
  const pct = Math.min(100, Math.max(0, score * 10)) // scale: 0-10 → 0-100%
  const color = score >= 7.0 ? 'bg-status-safe' : score >= 4.0 ? 'bg-status-bubble' : 'bg-text-muted'
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-bg-inset rounded-full overflow-hidden">
        <div className={cn('h-full rounded-full transition-all duration-700 ease-out', color)} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-text-primary tabular-nums w-8 text-right">
        {score.toFixed(2)}
      </span>
    </div>
  )
}

function OwnershipBadge({ pct }: { pct: number | null | undefined }) {
  if (pct === null || pct === undefined || pct === 0) {
    return <span className="text-[10px] text-text-muted">— owned</span>
  }
  return (
    <span className={cn(
      'text-[10px] tabular-nums',
      pct >= 70 ? 'text-status-bubble' : pct >= 30 ? 'text-text-secondary' : 'text-text-muted',
    )}>
      {pct.toFixed(0)}% owned
    </span>
  )
}

function HotColdBadge({ hotCold, rankPercentile }: { hotCold?: string | null; rankPercentile?: number | null }) {
  // Design System v2: gate badges to top 20% to prevent inflation
  if (!hotCold || (rankPercentile ?? 0) < 80) return null
  if (hotCold === 'HOT') {
    return (
      <span className="flex items-center gap-0.5 text-[10px] text-status-behind font-semibold">
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

function positionBadgeClass(pos: string): string {
  if (pos === 'SP') return 'bg-blue-900/30 text-blue-400'
  if (pos === 'RP' || pos === 'P') return 'bg-purple-900/30 text-purple-400'
  if (pos === 'OF' || pos === 'LF' || pos === 'CF' || pos === 'RF') return 'bg-emerald-900/30 text-emerald-400'
  if (pos === 'C') return 'bg-amber-900/30 text-amber-400'
  if (pos === '1B' || pos === '3B') return 'bg-orange-900/30 text-orange-400'
  if (pos === '2B' || pos === 'SS' || pos === 'MI') return 'bg-sky-900/30 text-sky-400'
  return 'bg-bg-elevated text-text-secondary'
}

function ZScoreDisplay({ z, rosterPlayer }: {
  z: number
  rosterPlayer?: WaiverRosterPlayer | null
}) {
  const zColor = z >= 2 ? 'text-status-safe' : z >= 0 ? 'text-text-primary' : 'text-status-lost'
  const delta = rosterPlayer != null ? z - rosterPlayer.z_score : null
  const showDelta = delta != null && Math.abs(delta) >= 0.3
  return (
    <div>
      <p className="text-[9px] text-text-muted uppercase tracking-wider mb-1">Season Value</p>
      <div className="flex items-center gap-1.5">
        <span className={cn('text-sm font-bold tabular-nums', zColor)}>
          {z >= 0 ? '+' : ''}{z.toFixed(1)}z
        </span>
        {showDelta && (
          <span className={cn(
            'text-[10px] font-semibold tabular-nums',
            delta > 0 ? 'text-status-safe' : 'text-status-bubble',
          )}>
            {delta > 0 ? '↑' : '↓'}{Math.abs(delta).toFixed(1)}
          </span>
        )}
      </div>
      {rosterPlayer && showDelta && (
        <p className="text-[9px] text-text-muted mt-0.5 truncate">
          vs {rosterPlayer.name.split(' ').slice(-1)[0]}
        </p>
      )}
    </div>
  )
}

function PositionContextBanner({ rosterPlayer, position }: {
  rosterPlayer: WaiverRosterPlayer
  position: string
}) {
  const zColor = rosterPlayer.z_score >= 2 ? 'text-status-safe'
    : rosterPlayer.z_score >= 0 ? 'text-text-secondary'
    : 'text-status-lost'
  return (
    <div className="bg-bg-surface border border-border-subtle rounded-lg px-4 py-2.5 flex items-center gap-3 flex-wrap">
      <span className="text-[9px] font-semibold tracking-widest uppercase text-text-muted">
        {position} Context
      </span>
      <div className="w-px h-3 bg-border-subtle flex-shrink-0" />
      <div className="flex items-center gap-2 flex-1 min-w-0">
        <span className="text-xs font-semibold text-text-secondary truncate">{rosterPlayer.name}</span>
        <span className="text-[10px] text-text-muted">{rosterPlayer.team}</span>
      </div>
      <div className="flex items-center gap-1.5 ml-auto flex-shrink-0">
        <span className="text-[9px] text-text-muted">weakest at pos</span>
        <span className={cn('text-sm font-bold tabular-nums', zColor)}>
          {rosterPlayer.z_score >= 0 ? '+' : ''}{rosterPlayer.z_score.toFixed(1)}z
        </span>
      </div>
    </div>
  )
}

function PlayerRow({ player, rosterPlayer }: {
  player: WaiverAvailablePlayer
  rosterPlayer?: WaiverRosterPlayer | null
}) {
  const ownedPct = player.percent_owned ?? player.owned_pct ?? null
  const positions = player.positions ?? (player.position ? [player.position] : [])
  const needMatches = (player.category_need_match ?? []).map((k) => waiverCatLabel(k) ?? k).filter(Boolean)
  const z = player.z_score ?? null

  return (
    <div className="bg-bg-surface border border-border-subtle rounded-lg p-4 flex flex-col sm:flex-row sm:items-start gap-3 hover:bg-bg-elevated transition-colors duration-150">
      {/* Identity */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <p className="text-sm font-bold text-text-primary truncate">{player.name}</p>
          <HotColdBadge hotCold={player.hot_cold} rankPercentile={player.rank_percentile} />
          {player.injury_status && (
            <span className="text-[10px] px-1.5 py-0.5 bg-status-lost/10 text-status-lost border border-status-lost/30 rounded font-semibold">
              {player.injury_status}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 mt-0.5 flex-wrap">
          <span className="text-xs text-text-secondary">{player.team}</span>
          <OwnershipBadge pct={ownedPct} />
          {positions.map((pos) => (
            <span key={pos} className={cn('text-[10px] px-1.5 py-0.5 rounded font-semibold', positionBadgeClass(pos))}>
              {pos}
            </span>
          ))}
        </div>
        {player.two_start && (
          <p className="text-[10px] text-status-safe mt-1 font-semibold">
            2-START WEEK
            {player.start1_opp ? ` · vs ${player.start1_opp}` : ''}
            {player.start2_opp ? `, ${player.start2_opp}` : ''}
          </p>
        )}
        {/* Category need matches — shows which of your deficits this player addresses */}
        {needMatches.length > 0 && (
          <div className="flex gap-1 mt-1.5 flex-wrap">
            <span className="text-[9px] text-text-muted uppercase tracking-wider self-center">Addresses:</span>
            {needMatches.map((label) => (
              <span key={label} className="text-[10px] px-1.5 py-0.5 bg-status-bubble/10 text-status-bubble border border-status-bubble/30 rounded font-bold">
                {label}
              </span>
            ))}
          </div>
        )}
        {player.statcast_signals && player.statcast_signals.length > 0 && (
          <div className="flex gap-1 mt-1 flex-wrap">
            {player.statcast_signals.map((sig) => (
              <span key={sig} className="text-[10px] px-1.5 py-0.5 bg-status-safe/10 text-status-safe border border-status-safe/20 rounded">
                {sig}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Scores column */}
      <div className="w-full sm:w-40 flex-shrink-0 space-y-3">
        {/* Weekly match score */}
        <div>
          <p className="text-[9px] text-text-muted uppercase tracking-wider mb-1">Match Score</p>
          <NeedBar score={player.need_score} />
          <p className="text-[9px] text-text-muted mt-0.5">fit for your gaps</p>
        </div>
        {/* Season value (z_score) with roster comparison */}
        {z != null && (
          <ZScoreDisplay z={z} rosterPlayer={rosterPlayer} />
        )}
      </div>
    </div>
  )
}

function CategoryDeficitsBar({ deficits, opponent }: {
  deficits: WaiverResponse['category_deficits']
  opponent?: string
}) {
  // Filter out display-only stats (H_AB, IP, GS)
  const scored = (deficits ?? []).filter((d) => waiverCatLabel(d.category) !== null)
  if (scored.length === 0) return null

  const wCount = scored.filter((d) => d.winning).length
  const lCount = scored.filter((d) => !d.winning && d.deficit !== 0).length
  const tCount = scored.filter((d) => d.deficit === 0).length

  function fmtVal(catKey: string, val: number): string {
    const label = waiverCatLabel(catKey) ?? catKey
    if (['ERA', 'WHIP', 'AVG', 'OPS', 'K/9', 'K_9'].includes(label) || ['ERA', 'WHIP', 'AVG', 'OPS', 'K/9', 'K_9'].includes(catKey)) {
      return val.toFixed(2)
    }
    return Number.isInteger(val) ? val.toString() : val.toFixed(1)
  }

  return (
    <div className="bg-bg-surface border border-border-subtle rounded-lg p-4">
      {/* Header — same layout as roster MatchupStrip for visual consistency */}
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <p className="text-[10px] font-semibold tracking-widest uppercase text-text-secondary">
          This Week{opponent ? ` · vs ${opponent}` : ''}
        </p>
        <div className="flex items-center gap-1.5">
          <span className="text-xs font-bold text-status-safe">{wCount}W</span>
          <span className="text-[10px] text-text-muted">·</span>
          <span className="text-xs font-bold text-status-lost">{lCount}L</span>
          {tCount > 0 && (
            <>
              <span className="text-[10px] text-text-muted">·</span>
              <span className="text-xs font-bold text-status-bubble">{tCount}T</span>
            </>
          )}
        </div>
      </div>
      {/* Pill grid — mirrors MatchupStrip on roster page for a unified visual language */}
      <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-9 gap-2">
        {scored.map((d) => {
          const label = waiverCatLabel(d.category) ?? d.category
          const isLowerBetter = ['ERA', 'WHIP', 'L', 'HRA'].includes(label)
          const outcome: 'W' | 'L' | 'T' = d.deficit === 0 ? 'T' : d.winning ? 'W' : 'L'
          const outcomeBg = outcome === 'W'
            ? 'bg-status-safe/10 border-status-safe/30'
            : outcome === 'L'
              ? 'bg-status-lost/10 border-status-lost/30'
              : 'bg-status-bubble/10 border-status-bubble/30'
          const outcomeText = outcome === 'W' ? 'text-status-safe' : outcome === 'L' ? 'text-status-lost' : 'text-status-bubble'
          const myAhead = isLowerBetter ? d.my_total < d.opponent_total : d.my_total > d.opponent_total
          return (
            <div key={d.category} className={cn('rounded p-1.5 border text-center', outcomeBg)}>
              <p className="text-[9px] text-text-secondary uppercase tracking-wider leading-none mb-1">{label}</p>
              <div className={cn('text-[10px] font-bold leading-none', outcomeText)}>{outcome}</div>
              <p className="text-[9px] text-text-muted leading-none mt-1">
                <span className={myAhead ? 'text-text-primary' : ''}>{fmtVal(d.category, d.my_total)}</span>
                <span className="mx-0.5 text-border-subtle">·</span>
                {fmtVal(d.category, d.opponent_total)}
              </p>
            </div>
          )
        })}
      </div>
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
        <div className="flex items-center gap-2 text-text-secondary">
          <Loader2 className="h-5 w-5 animate-spin text-accent-gold" />
          <span className="text-sm">Loading waiver wire…</span>
        </div>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center">
        <div className="bg-bg-surface border border-border-subtle rounded-lg p-6 max-w-md w-full">
          <div className="flex items-center gap-2 text-status-lost mb-2">
            <AlertCircle className="h-5 w-5" />
            <span className="text-sm font-semibold">Failed to load waiver wire</span>
          </div>
          <p className="text-text-secondary text-sm">
            {error instanceof Error ? error.message : 'Unknown error'}
          </p>
          <button onClick={() => refetch()} className="mt-4 text-xs text-accent-gold hover:text-amber-300 font-semibold">
            Retry
          </button>
        </div>
      </div>
    )
  }

  const topAvailable = filterPlayers(data?.top_available ?? [])
  const twoStarters = filterPlayers(data?.two_start_pitchers ?? [])
  const rosterCtx = data?.roster_context ?? {}
  const activeRosterPlayer = posFilter !== 'All' ? (rosterCtx[posFilter] ?? null) : null

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <ListFilter className="h-3.5 w-3.5 text-accent-gold" />
          <span className="text-xs font-bold tracking-widest uppercase text-accent-gold">
            Waiver Wire
          </span>
        </div>
        <div className="flex items-center gap-2 text-[10px] text-text-muted">
          {data?.il_slots_available != null && data.il_slots_available > 0 && (
            <span className="text-status-safe">
              {data.il_slots_available} IL slot{data.il_slots_available > 1 ? 's' : ''} open
            </span>
          )}
        </div>
      </div>

      {/* Alerts */}
      {data?.urgent_alert && (
        <div className="bg-status-bubble/10 border border-status-bubble/30 rounded-lg p-3 flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-status-bubble flex-shrink-0" />
          <span className="text-sm text-status-bubble">{data.urgent_alert.message}</span>
        </div>
      )}
      {data?.closer_alert === 'NO_CLOSERS' && (
        <div className="bg-status-lost/10 border border-status-lost/30 rounded-lg p-3 flex items-center gap-2">
          <AlertCircle className="h-4 w-4 text-status-lost flex-shrink-0" />
          <span className="text-sm text-status-lost">
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
        <div className="flex items-center gap-1 bg-bg-surface border border-border-subtle rounded-lg p-1">
          {(['need_score', 'projected_points'] as const).map((s) => (
            <button
              key={s}
              onClick={() => setSort(s)}
              className={cn(
                'text-[10px] px-3 py-1.5 rounded font-semibold tracking-wider uppercase transition-colors',
                sort === s ? 'bg-accent-gold text-black' : 'text-text-secondary hover:text-text-primary',
              )}
            >
              {s === 'need_score' ? 'Match Score' : 'Overall Value'}
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
                  ? 'bg-bg-elevated text-text-primary border border-border-default'
                  : 'text-text-muted hover:text-text-secondary',
              )}
            >
              {pos}
            </button>
          ))}
        </div>
      </div>

      {/* Position context — shows weakest roster player at active position for upgrade comparison */}
      {activeRosterPlayer && (
        <PositionContextBanner rosterPlayer={activeRosterPlayer} position={posFilter} />
      )}

      {/* Two-Start Pitchers */}
      {twoStarters.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-3.5 w-3.5 text-status-safe" />
            <p className="text-xs font-semibold tracking-widest uppercase text-status-safe">
              Two-Start Pitchers · {twoStarters.length}
            </p>
          </div>
          {twoStarters.map((p) => (
            <PlayerRow key={p.player_id} player={p} rosterPlayer={activeRosterPlayer} />
          ))}
        </div>
      )}

      {/* Top Available */}
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <Users className="h-3.5 w-3.5 text-accent-gold" />
          <p className="text-xs font-semibold tracking-widest uppercase text-text-secondary">
            Top Available · {topAvailable.length}
          </p>
        </div>
        {topAvailable.length === 0 ? (
          <div className="bg-bg-surface border border-border-subtle rounded-lg p-8 text-center">
            <p className="text-text-muted text-sm">No players match this filter.</p>
          </div>
        ) : (
          topAvailable.map((p) => (
            <PlayerRow key={p.player_id} player={p} rosterPlayer={activeRosterPlayer} />
          ))
        )}
      </div>
    </div>
  )
}
