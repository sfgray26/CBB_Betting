'use client'

import { useState, useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import type { RosterPlayer, RosterMoveResponse } from '@/lib/types'
import {
  Users,
  Loader2,
  AlertCircle,
  ChevronRight,
  ShieldAlert,
  Activity,
  TrendingUp,
  Clock,
  BarChart3,
} from 'lucide-react'
import { cn } from '@/lib/utils'

// ───────────────────────────────────────────────────────────────────────────
// Constants
// ───────────────────────────────────────────────────────────────────────────

const VALID_POSITIONS = [
  'C', '1B', '2B', '3B', 'SS', 'OF', 'Util',
  'SP', 'RP', 'P', 'BN', 'IL', 'IL60',
]

const BATTER_DISPLAY = ['HR_B', 'RBI', 'AVG', 'OPS', 'NSB', 'R', 'TB', 'K_B']
const PITCHER_DISPLAY = ['ERA', 'WHIP', 'K_P', 'W', 'QS', 'K_9', 'L', 'HR_P']

const STATUS_COLORS: Record<string, string> = {
  playing: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  probable: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  not_playing: 'bg-rose-500/20 text-rose-400 border-rose-500/30',
  IL: 'bg-rose-600/20 text-rose-500 border-rose-600/30',
  minors: 'bg-zinc-600/20 text-zinc-400 border-zinc-600/30',
}

const STATUS_LABELS: Record<string, string> = {
  playing: 'Active',
  probable: 'DTD',
  not_playing: 'Out',
  IL: 'IL',
  minors: 'Minors',
}

// ───────────────────────────────────────────────────────────────────────────
// Helpers
// ───────────────────────────────────────────────────────────────────────────

function getStat(stats: Record<string, number | null> | undefined, key: string): string {
  if (!stats) return '-'
  const v = stats[key]
  if (v === null || v === undefined) return '-'
  if (key === 'AVG' || key === 'OPS') return v.toFixed(3).replace(/^0\./, '.')
  if (key === 'ERA' || key === 'WHIP') return v.toFixed(2)
  return String(Math.round(v))
}

function formatCategoryLabel(code: string): string {
  return code
    .replace('_B', '')
    .replace('_P', '')
    .replace('_', '/')
}

// ───────────────────────────────────────────────────────────────────────────
// Player Card
// ───────────────────────────────────────────────────────────────────────────

function PlayerCard({
  player,
  onMove,
  isMoving,
}: {
  player: RosterPlayer
  onMove: (playerId: string, toSlot: string) => void
  isMoving: boolean
}) {
  const [selectedSlot, setSelectedSlot] = useState('')

  const eligible = player.eligible_positions ?? []
  const isPitcher = eligible.some((p) => ['SP', 'RP', 'P'].includes(p))
  const displayCats = isPitcher ? PITCHER_DISPLAY : BATTER_DISPLAY

  const handleMoveClick = () => {
    if (!selectedSlot || !player.yahoo_player_key) return
    onMove(player.yahoo_player_key, selectedSlot)
    setSelectedSlot('')
  }

  const statusClass = STATUS_COLORS[player.status] ?? STATUS_COLORS.playing
  const statusLabel = STATUS_LABELS[player.status] ?? player.status

  return (
    <div className="bg-[#202020] rounded-lg p-4 flex flex-col sm:flex-row sm:items-start gap-4">
      {/* Identity */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <p className="text-sm font-bold text-white truncate">{player.player_name}</p>
          <span
            className={cn(
              'text-[10px] px-1.5 py-0.5 rounded border font-semibold uppercase tracking-wider',
              statusClass,
            )}
          >
            {statusLabel}
          </span>
        </div>
        <p className="text-xs text-[#7D7D7D] mt-0.5">
          {player.team}
          {player.ownership_pct !== null && player.ownership_pct !== undefined
            ? ` · ${player.ownership_pct.toFixed(0)}% owned`
            : ''}
        </p>
        <div className="flex flex-wrap gap-1 mt-2">
          {eligible.map((pos) => (
            <span
              key={pos}
              className="text-[10px] px-1.5 py-0.5 bg-[#2A2A2A] text-[#969696] rounded"
            >
              {pos}
            </span>
          ))}
        </div>
        {player.injury_status && (
          <p className="text-xs text-rose-400 mt-2 flex items-center gap-1">
            <ShieldAlert className="h-3 w-3" />
            {player.injury_status}
            {player.injury_return_timeline ? ` (${player.injury_return_timeline})` : ''}
          </p>
        )}
        {player.game_context && (
          <p className="text-[10px] text-[#494949] mt-1">
            vs {player.game_context.opponent} ({player.game_context.home_away})
            {player.game_context.game_time ? ` · ${new Date(player.game_context.game_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}` : ''}
          </p>
        )}
      </div>

      {/* Stats */}
      <div className="flex-shrink-0">
        <div className="grid grid-cols-4 sm:grid-cols-8 gap-x-3 gap-y-1">
          {displayCats.map((cat) => (
            <div key={cat} className="text-center">
              <p className="text-[9px] text-[#494949] uppercase tracking-wider">
                {formatCategoryLabel(cat)}
              </p>
              <p className="text-xs font-semibold text-white tabular-nums">
                {getStat(player.season_stats?.values, cat)}
              </p>
            </div>
          ))}
        </div>
        {player.rolling_14d && (
          <div className="mt-1.5 pt-1.5 border-t border-[#2A2A2A]">
            <div className="grid grid-cols-4 sm:grid-cols-8 gap-x-3 gap-y-1">
              {displayCats.map((cat) => (
                <div key={`${cat}-14d`} className="text-center">
                  <p className="text-[9px] text-[#494949] uppercase tracking-wider">
                    {formatCategoryLabel(cat)}14
                  </p>
                  <p className="text-xs font-semibold text-[#969696] tabular-nums">
                    {getStat(player.rolling_14d?.values, cat)}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Move controls */}
      <div className="flex-shrink-0 flex items-center gap-2">
        <select
          value={selectedSlot}
          onChange={(e) => setSelectedSlot(e.target.value)}
          className="bg-[#2A2A2A] text-xs text-white border border-[#494949] rounded px-2 py-1.5 focus:outline-none focus:border-[#FFC000] min-w-[90px]"
        >
          <option value="">Move to…</option>
          {VALID_POSITIONS.map((pos) => (
            <option key={pos} value={pos}>
              {pos}
            </option>
          ))}
        </select>
        <button
          onClick={handleMoveClick}
          disabled={!selectedSlot || isMoving}
          className={cn(
            'p-1.5 rounded transition-colors',
            selectedSlot && !isMoving
              ? 'bg-[#FFC000] text-black hover:bg-amber-300'
              : 'bg-[#2A2A2A] text-[#494949] cursor-not-allowed',
          )}
          title="Move player"
        >
          <ChevronRight className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  )
}

// ───────────────────────────────────────────────────────────────────────────
// Category Summary
// ───────────────────────────────────────────────────────────────────────────

function CategorySummary({ players }: { players: RosterPlayer[] }) {
  const allSeasonStats = players
    .filter((p) => p.season_stats?.values)
    .map((p) => p.season_stats!.values)

  const totals: Record<string, number> = {}
  for (const stats of allSeasonStats) {
    for (const [key, val] of Object.entries(stats)) {
      if (val !== null && val !== undefined) {
        if (!totals[key]) totals[key] = 0
        // Skip rate stats for simple summation
        if (!['AVG', 'ERA', 'WHIP', 'OPS', 'K_9'].includes(key)) {
          totals[key] += val
        }
      }
    }
  }

  const allCats = [...BATTER_DISPLAY, ...PITCHER_DISPLAY]

  return (
    <div className="bg-[#202020] rounded-lg p-4">
      <div className="flex items-center gap-2 mb-3">
        <BarChart3 className="h-3.5 w-3.5 text-[#FFC000]" />
        <p className="text-xs font-semibold tracking-widest uppercase text-[#7D7D7D]">
          Category Contributions
        </p>
      </div>
      <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-9 gap-3">
        {allCats.map((cat) => {
          const isRate = ['AVG', 'OPS', 'ERA', 'WHIP', 'K_9'].includes(cat)
          return (
            <div key={cat} className="text-center">
              <p className="text-[10px] text-[#494949] uppercase tracking-wider">
                {formatCategoryLabel(cat)}
              </p>
              <p className="text-sm font-bold text-white tabular-nums">
                {isRate ? '-' : (totals[cat] ?? 0).toLocaleString()}
              </p>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ───────────────────────────────────────────────────────────────────────────
// Main Page
// ───────────────────────────────────────────────────────────────────────────

export default function RosterPage() {
  const queryClient = useQueryClient()
  const [moveError, setMoveError] = useState<string | null>(null)
  const [moveSuccess, setMoveSuccess] = useState<string | null>(null)

  const roster = useQuery({
    queryKey: ['roster'],
    queryFn: endpoints.getRoster,
    staleTime: 5 * 60_000,
  })

  const moveMutation = useMutation({
    mutationFn: ({ playerId, toSlot }: { playerId: string; toSlot: string }) =>
      endpoints.movePlayer(playerId, '', toSlot),
    onSuccess: (data: RosterMoveResponse) => {
      setMoveError(null)
      setMoveSuccess(data.message)
      queryClient.invalidateQueries({ queryKey: ['roster'] })
      setTimeout(() => setMoveSuccess(null), 4000)
    },
    onError: (err: Error) => {
      setMoveSuccess(null)
      setMoveError(err.message)
    },
  })

  const handleMove = useCallback(
    (playerId: string, toSlot: string) => {
      setMoveError(null)
      setMoveSuccess(null)
      moveMutation.mutate({ playerId, toSlot })
    },
    [moveMutation],
  )

  // Loading state
  if (roster.isLoading) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center">
        <div className="flex items-center gap-2 text-[#7D7D7D]">
          <Loader2 className="h-5 w-5 animate-spin text-[#FFC000]" />
          <span className="text-sm">Loading roster…</span>
        </div>
      </div>
    )
  }

  // Error state
  if (roster.isError) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center">
        <div className="bg-[#202020] rounded-lg p-6 max-w-md w-full">
          <div className="flex items-center gap-2 text-rose-400 mb-2">
            <AlertCircle className="h-5 w-5" />
            <span className="text-sm font-semibold">Failed to load roster</span>
          </div>
          <p className="text-[#969696] text-sm">
            {roster.error instanceof Error ? roster.error.message : 'Unknown error'}
          </p>
          <button
            onClick={() => roster.refetch()}
            className="mt-4 text-xs text-[#FFC000] hover:text-amber-300 font-semibold"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  const data = roster.data

  // Empty state
  if (!data || data.players.length === 0) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-2">
          <Users className="h-3.5 w-3.5 text-[#FFC000]" />
          <span className="text-xs font-bold tracking-widest uppercase text-[#FFC000]">
            My Roster
          </span>
        </div>
        <div className="bg-[#202020] rounded-lg p-8 text-center">
          <p className="text-xs font-semibold tracking-widest uppercase text-[#494949]">
            Empty Roster
          </p>
          <p className="text-[#7D7D7D] text-sm mt-2">
            No players found on your roster.
          </p>
        </div>
      </div>
    )
  }

  // Group players by status
  const ilPlayers = data.players.filter((p) => p.status === 'IL')
  const activePlayers = data.players.filter(
    (p) => p.status === 'playing' || p.status === 'probable',
  )
  const otherPlayers = data.players.filter(
    (p) => p.status !== 'IL' && p.status !== 'playing' && p.status !== 'probable',
  )

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <Users className="h-3.5 w-3.5 text-[#FFC000]" />
          <span className="text-xs font-bold tracking-widest uppercase text-[#FFC000]">
            My Roster
          </span>
        </div>
        <div className="flex items-center gap-3 text-[10px] text-[#494949] tracking-wider uppercase">
          <span>{data.count} players</span>
          <span>·</span>
          <span className="font-mono">{data.team_key}</span>
          {data.freshness?.computed_at && (
            <>
              <span>·</span>
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {new Date(data.freshness.computed_at).toLocaleTimeString([], {
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </span>
            </>
          )}
        </div>
      </div>

      {/* Move feedback */}
      {moveError && (
        <div className="bg-rose-900/20 border border-rose-800/50 rounded-lg p-3 flex items-center gap-2">
          <AlertCircle className="h-4 w-4 text-rose-400 flex-shrink-0" />
          <span className="text-sm text-rose-300">{moveError}</span>
        </div>
      )}
      {moveSuccess && (
        <div className="bg-emerald-900/20 border border-emerald-800/50 rounded-lg p-3 flex items-center gap-2">
          <Activity className="h-4 w-4 text-emerald-400 flex-shrink-0" />
          <span className="text-sm text-emerald-300">{moveSuccess}</span>
        </div>
      )}

      {/* Category summary */}
      <CategorySummary players={data.players} />

      {/* Active Players */}
      {activePlayers.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-3.5 w-3.5 text-emerald-400" />
            <p className="text-xs font-semibold tracking-widest uppercase text-[#7D7D7D]">
              Active · {activePlayers.length}
            </p>
          </div>
          {activePlayers.map((player) => (
            <PlayerCard
              key={player.yahoo_player_key ?? player.player_name}
              player={player}
              onMove={handleMove}
              isMoving={moveMutation.isPending}
            />
          ))}
        </div>
      )}

      {/* IL Players */}
      {ilPlayers.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <ShieldAlert className="h-3.5 w-3.5 text-rose-400" />
            <p className="text-xs font-semibold tracking-widest uppercase text-rose-400">
              Injured List · {ilPlayers.length}
            </p>
          </div>
          {ilPlayers.map((player) => (
            <PlayerCard
              key={player.yahoo_player_key ?? player.player_name}
              player={player}
              onMove={handleMove}
              isMoving={moveMutation.isPending}
            />
          ))}
        </div>
      )}

      {/* Other / Not Active */}
      {otherPlayers.length > 0 && (
        <div className="space-y-3">
          <p className="text-xs font-semibold tracking-widest uppercase text-[#494949]">
            Not Active · {otherPlayers.length}
          </p>
          {otherPlayers.map((player) => (
            <PlayerCard
              key={player.yahoo_player_key ?? player.player_name}
              player={player}
              onMove={handleMove}
              isMoving={moveMutation.isPending}
            />
          ))}
        </div>
      )}
    </div>
  )
}
