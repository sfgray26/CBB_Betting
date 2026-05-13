'use client'

import { useState, useCallback, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import type { RosterPlayer, RosterMoveResponse, RosterOptimizeResponse, BudgetData, ScoreboardResponse } from '@/lib/types'
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
  Sparkles,
  Calendar,
  CalendarOff,
  AlertTriangle,
  X,
  Gauge,
  Swords,
} from 'lucide-react'
import { cn } from '@/lib/utils'

// ───────────────────────────────────────────────────────────────────────────
// Constants
// ───────────────────────────────────────────────────────────────────────────

// Positions that every player can be moved to regardless of eligibility
const UNIVERSAL_SLOTS = ['BN', 'IL', 'IL60']

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

const SLOT_COLORS: Record<string, string> = {
  BN: 'bg-[#2A2A2A] text-[#969696]',
  IL: 'bg-rose-900/30 text-rose-400',
  IL60: 'bg-rose-900/30 text-rose-400',
  SP: 'bg-blue-900/30 text-blue-400',
  RP: 'bg-purple-900/30 text-purple-400',
  P: 'bg-purple-900/30 text-purple-400',
}

type ViewMode = 'season' | '7d' | '14d' | '30d' | 'ros'
type SortMode = 'default' | 'name' | 'ros_value'
type PosFilter = 'All' | 'SP' | 'RP' | 'OF' | '1B' | '2B' | '3B' | 'SS' | 'C'

// ───────────────────────────────────────────────────────────────────────────
// Helpers
// ───────────────────────────────────────────────────────────────────────────

function getStatWindow(player: RosterPlayer, mode: ViewMode) {
  switch (mode) {
    case 'season': return player.season_stats?.values
    case '7d': return player.rolling_7d?.values
    case '14d': return player.rolling_14d?.values
    case '30d': return player.rolling_30d?.values
    case 'ros': return player.ros_projection?.values
    default: return player.season_stats?.values
  }
}

function getStat(values: Record<string, number | null> | undefined, key: string): string {
  if (!values) return '-'
  const v = values[key]
  if (v === null || v === undefined) return '-'
  if (key === 'AVG' || key === 'OPS') return v.toFixed(3).replace(/^0\./, '.')
  if (key === 'ERA' || key === 'WHIP') return v.toFixed(2)
  if (key === 'K_9') return v.toFixed(1)
  return String(Math.round(v))
}

function formatCat(code: string): string {
  const labels: Record<string, string> = {
    HR_B: 'HR', K_B: 'K', K_P: 'K', HR_P: 'HR', K_9: 'K/9', NSB: 'NSB',
  }
  return labels[code] ?? code.replace(/_[BP]$/, '')
}

function rosValueScore(player: RosterPlayer): number {
  const vals = player.ros_projection?.values
  if (!vals) return -999
  const RATE_CATS = ['AVG', 'OPS', 'ERA', 'WHIP', 'K_9']
  return Object.entries(vals)
    .filter(([k]) => !RATE_CATS.includes(k))
    .reduce((sum, [, v]) => sum + (v ?? 0), 0)
}

// ───────────────────────────────────────────────────────────────────────────
// Sub-components
// ───────────────────────────────────────────────────────────────────────────

function ViewToggle({ mode, onChange }: { mode: ViewMode; onChange: (m: ViewMode) => void }) {
  const options: { value: ViewMode; label: string }[] = [
    { value: 'season', label: 'Season' },
    { value: '7d', label: '7D' },
    { value: '14d', label: '14D' },
    { value: '30d', label: '30D' },
    { value: 'ros', label: 'RoS Proj' },
  ]
  return (
    <div className="flex items-center gap-1 bg-[#181818] border border-[#2A2A2A] rounded-lg p-1">
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          className={cn(
            'text-[10px] px-3 py-1.5 rounded font-semibold tracking-wider uppercase transition-colors',
            mode === opt.value
              ? opt.value === 'ros'
                ? 'bg-[#FFC000] text-black'
                : 'bg-[#2A2A2A] text-white border border-[#494949]'
              : 'text-[#494949] hover:text-[#969696]',
          )}
        >
          {opt.label}
        </button>
      ))}
    </div>
  )
}

function GamePill({ player }: { player: RosterPlayer }) {
  if (player.game_context) {
    const timeStr = player.game_context.game_time
      ? new Date(player.game_context.game_time).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
      : null
    return (
      <span className="flex items-center gap-1 text-[10px] bg-emerald-900/20 text-emerald-400 border border-emerald-800/30 rounded px-1.5 py-0.5 font-medium">
        <Calendar className="h-2.5 w-2.5" />
        vs {player.game_context.opponent}
        {timeStr ? ` · ${timeStr}` : ''}
      </span>
    )
  }
  return (
    <span className="flex items-center gap-1 text-[10px] text-[#494949] border border-[#2A2A2A] rounded px-1.5 py-0.5">
      <CalendarOff className="h-2.5 w-2.5" />
      No Game
    </span>
  )
}

function SlotPill({ slot }: { slot?: string | null }) {
  if (!slot) return null
  const colorClass = SLOT_COLORS[slot] ?? 'bg-[#2A2A2A] text-[#969696]'
  return (
    <span className={cn('text-[10px] px-1.5 py-0.5 rounded font-mono font-semibold', colorClass)}>
      {slot}
    </span>
  )
}

// ───────────────────────────────────────────────────────────────────────────
// Budget Panel
// ───────────────────────────────────────────────────────────────────────────

function BudgetPanel({ budget }: { budget: BudgetData }) {
  const acqPct = budget.acquisition_limit > 0
    ? Math.min(100, (budget.acquisitions_used / budget.acquisition_limit) * 100)
    : 0
  const ipPct = budget.ip_minimum > 0
    ? Math.min(100, (budget.ip_accumulated / budget.ip_minimum) * 100)
    : 0
  const paceColor = budget.ip_pace === 'BEHIND'
    ? 'text-rose-400'
    : budget.ip_pace === 'AHEAD'
      ? 'text-emerald-400'
      : 'text-amber-400'
  const movesLeft = budget.acquisition_limit - budget.acquisitions_used
  const movesWarning = budget.acquisition_warning || movesLeft <= 1

  return (
    <div className="bg-[#202020] rounded-lg p-4">
      <div className="flex items-center gap-2 mb-3">
        <Gauge className="h-3.5 w-3.5 text-[#FFC000]" />
        <p className="text-xs font-semibold tracking-widest uppercase text-[#7D7D7D]">
          Constraints
        </p>
      </div>
      <div className="grid grid-cols-3 gap-4">
        {/* Weekly Moves */}
        <div>
          <p className="text-[9px] text-[#494949] uppercase tracking-wider mb-1">Weekly Moves</p>
          <div className="flex items-center gap-1.5 mb-1">
            <div className="flex-1 h-1.5 bg-[#2A2A2A] rounded-full overflow-hidden">
              <div
                className={cn('h-full rounded-full', acqPct >= 80 ? 'bg-rose-500' : acqPct >= 60 ? 'bg-amber-400' : 'bg-emerald-500')}
                style={{ width: `${acqPct}%` }}
              />
            </div>
            <span className={cn('text-xs font-bold tabular-nums', movesWarning ? 'text-amber-400' : 'text-white')}>
              {budget.acquisitions_used}/{budget.acquisition_limit}
            </span>
          </div>
          <p className={cn('text-[10px]', movesWarning ? 'text-amber-400 font-semibold' : 'text-[#494949]')}>
            {movesLeft} remaining
          </p>
        </div>

        {/* IP Pace */}
        <div>
          <p className="text-[9px] text-[#494949] uppercase tracking-wider mb-1">IP Pace</p>
          <div className="flex items-center gap-1.5 mb-1">
            <div className="flex-1 h-1.5 bg-[#2A2A2A] rounded-full overflow-hidden">
              <div
                className="h-full rounded-full bg-blue-500"
                style={{ width: `${ipPct}%` }}
              />
            </div>
            <span className={cn('text-xs font-bold tabular-nums', paceColor)}>
              {budget.ip_pace}
            </span>
          </div>
          <p className="text-[10px] text-[#494949]">
            {budget.ip_accumulated.toFixed(1)} / {budget.ip_minimum} IP
          </p>
        </div>

        {/* IL */}
        <div>
          <p className="text-[9px] text-[#494949] uppercase tracking-wider mb-1">IL Slots</p>
          <p className="text-sm font-bold text-white tabular-nums">
            {budget.il_used}<span className="text-[#494949] font-normal">/{budget.il_total}</span>
          </p>
          {budget.il_used < budget.il_total && (
            <p className="text-[10px] text-emerald-400 font-semibold mt-0.5">
              {budget.il_total - budget.il_used} open
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

// ───────────────────────────────────────────────────────────────────────────
// Category Summary — raw stats for actual windows; z-score edge for RoS
// ───────────────────────────────────────────────────────────────────────────

function CategorySummary({ players, viewMode }: { players: RosterPlayer[]; viewMode: ViewMode }) {
  const activePlayers = players.filter((p) => {
    const slot = p.current_slot?.toUpperCase()
    if (slot === 'BN' || slot === 'IL' || slot === 'IL60') return false
    return p.status !== 'IL'
  })

  // ── RoS mode: cat_scores are z-scores (DB: "Dict of category -> z-score")
  // Sum counting-cat z-scores across starters; average rate-cat z-scores.
  // Positive = above league average (strength), negative = weakness.
  if (viewMode === 'ros') {
    const zSums: Record<string, number[]> = {}
    const RATE_CATS = new Set(['AVG', 'OPS', 'ERA', 'WHIP', 'K_9'])
    for (const player of activePlayers) {
      const vals = player.ros_projection?.values
      if (!vals) continue
      for (const [key, val] of Object.entries(vals)) {
        if (val === null || val === undefined) continue
        if (!zSums[key]) zSums[key] = []
        zSums[key].push(val)
      }
    }
    const allCats = [...BATTER_DISPLAY, ...PITCHER_DISPLAY]
    const hasAnyZ = allCats.some((cat) => (zSums[cat]?.length ?? 0) > 0)
    if (!hasAnyZ) {
      return (
        <div className="bg-[#202020] rounded-lg p-4 text-center">
          <p className="text-xs text-[#494949]">No RoS projections available for active starters.</p>
        </div>
      )
    }
    return (
      <div className="bg-[#202020] rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-3.5 w-3.5 text-[#FFC000]" />
            <p className="text-xs font-semibold tracking-widest uppercase text-[#FFC000]">
              Projected Team Edge
            </p>
          </div>
          <p className="text-[9px] text-[#494949]">
            Z-scores vs. league avg · green = strength · red = weakness
          </p>
        </div>
        <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-9 gap-3">
          {allCats.map((cat) => {
            const vals = zSums[cat]
            if (!vals || vals.length === 0) return (
              <div key={cat} className="text-center">
                <p className="text-[10px] text-[#494949] uppercase tracking-wider">{formatCat(cat)}</p>
                <p className="text-sm font-bold text-[#2A2A2A]">–</p>
              </div>
            )
            // Rate cats: average z; counting cats: sum z
            const z = RATE_CATS.has(cat)
              ? vals.reduce((a, b) => a + b, 0) / vals.length
              : vals.reduce((a, b) => a + b, 0)
            const color = z >= 1.5
              ? 'text-emerald-400'
              : z >= 0.3
                ? 'text-emerald-600'
                : z <= -1.5
                  ? 'text-rose-400'
                  : z <= -0.3
                    ? 'text-rose-600'
                    : 'text-[#7D7D7D]'
            return (
              <div key={cat} className="text-center">
                <p className="text-[10px] text-[#494949] uppercase tracking-wider">{formatCat(cat)}</p>
                <p className={cn('text-sm font-bold tabular-nums', color)}>
                  {z > 0 ? '+' : ''}{z.toFixed(1)}
                </p>
              </div>
            )
          })}
        </div>
        <p className="text-[9px] text-[#494949] mt-2">
          Counting cats: sum of player z-scores · Rate cats: avg z-score of starters
        </p>
      </div>
    )
  }

  // ── Actual stat windows (Season / 7D / 14D / 30D)
  const totals: Record<string, number> = {}
  for (const player of activePlayers) {
    const vals = getStatWindow(player, viewMode)
    if (!vals) continue
    for (const [key, val] of Object.entries(vals)) {
      if (val !== null && val !== undefined) {
        if (!totals[key]) totals[key] = 0
        if (!['AVG', 'ERA', 'WHIP', 'OPS', 'K_9'].includes(key)) {
          totals[key] += val
        }
      }
    }
  }

  const avgValues = activePlayers
    .map((p) => getStatWindow(p, viewMode)?.['AVG'])
    .filter((v): v is number => v != null && v > 0)
  const medianAvg = avgValues.length > 0
    ? [...avgValues].sort((a, b) => a - b)[Math.floor(avgValues.length / 2)]
    : null

  const eraValues = activePlayers
    .map((p) => {
      const vals = getStatWindow(p, viewMode)
      const isPitcher = p.eligible_positions.some((pos) => ['SP', 'RP', 'P'].includes(pos))
      return isPitcher ? vals?.['ERA'] : null
    })
    .filter((v): v is number => v != null && v > 0)
  const medianEra = eraValues.length > 0
    ? [...eraValues].sort((a, b) => a - b)[Math.floor(eraValues.length / 2)]
    : null

  const whipValues = activePlayers
    .map((p) => {
      const vals = getStatWindow(p, viewMode)
      const isPitcher = p.eligible_positions.some((pos) => ['SP', 'RP', 'P'].includes(pos))
      return isPitcher ? vals?.['WHIP'] : null
    })
    .filter((v): v is number => v != null && v > 0)
  const medianWhip = whipValues.length > 0
    ? [...whipValues].sort((a, b) => a - b)[Math.floor(whipValues.length / 2)]
    : null

  const opsValues = activePlayers
    .map((p) => {
      const vals = getStatWindow(p, viewMode)
      const isPitcher = p.eligible_positions.some((pos) => ['SP', 'RP', 'P'].includes(pos))
      return !isPitcher ? vals?.['OPS'] : null
    })
    .filter((v): v is number => v != null && v > 0)
  const medianOps = opsValues.length > 0
    ? [...opsValues].sort((a, b) => a - b)[Math.floor(opsValues.length / 2)]
    : null

  const k9Values = activePlayers
    .map((p) => {
      const vals = getStatWindow(p, viewMode)
      const isPitcher = p.eligible_positions.some((pos) => ['SP', 'RP', 'P'].includes(pos))
      return isPitcher ? vals?.['K_9'] : null
    })
    .filter((v): v is number => v != null && v > 0)
  const medianK9 = k9Values.length > 0
    ? [...k9Values].sort((a, b) => a - b)[Math.floor(k9Values.length / 2)]
    : null

  const allCats = [...BATTER_DISPLAY, ...PITCHER_DISPLAY]

  function getDisplayVal(cat: string): string {
    if (cat === 'AVG') return medianAvg != null ? medianAvg.toFixed(3).replace(/^0\./, '.') : '–'
    if (cat === 'OPS') return medianOps != null ? medianOps.toFixed(3).replace(/^0\./, '.') : '–'
    if (cat === 'ERA') return medianEra != null ? medianEra.toFixed(2) : '–'
    if (cat === 'WHIP') return medianWhip != null ? medianWhip.toFixed(2) : '–'
    if (cat === 'K_9') return medianK9 != null ? medianK9.toFixed(1) : '–'
    return (totals[cat] ?? 0).toLocaleString()
  }

  return (
    <div className="bg-[#202020] rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <BarChart3 className="h-3.5 w-3.5 text-[#FFC000]" />
          <p className="text-xs font-semibold tracking-widest uppercase text-[#7D7D7D]">
            Category Totals
          </p>
        </div>
        <p className="text-[9px] text-[#494949]">
          Active starters · {activePlayers.length} players · Rate stats = median
        </p>
      </div>
      <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-9 gap-3">
        {allCats.map((cat) => (
          <div key={cat} className="text-center">
            <p className="text-[10px] text-[#494949] uppercase tracking-wider">
              {formatCat(cat)}
            </p>
            <p className="text-sm font-bold text-white tabular-nums">
              {getDisplayVal(cat)}
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}

// ───────────────────────────────────────────────────────────────────────────
// Matchup Strip — current week category-by-category vs opponent
// ───────────────────────────────────────────────────────────────────────────

function statusToLabel(status: string | null): 'W' | 'L' | 'T' {
  if (!status) return 'T'
  if (status.includes('win')) return 'W'
  if (status.includes('loss')) return 'L'
  return 'T'
}

function MatchupStrip({ scoreboard }: { scoreboard: ScoreboardResponse }) {
  const { opponent_name, categories_won, categories_lost, categories_tied, overall_win_probability, rows } = scoreboard
  const winPct = overall_win_probability != null ? Math.round(overall_win_probability * 100) : null

  return (
    <div className="bg-[#1A1A1A] border border-[#2A2A2A] rounded-lg p-4">
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <Swords className="h-3.5 w-3.5 text-[#FFC000]" />
          <p className="text-xs font-semibold tracking-widest uppercase text-[#7D7D7D]">
            This Week · vs {opponent_name}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="text-xs font-bold text-emerald-400">{categories_won}W</span>
            <span className="text-[10px] text-[#494949]">·</span>
            <span className="text-xs font-bold text-rose-400">{categories_lost}L</span>
            <span className="text-[10px] text-[#494949]">·</span>
            <span className="text-xs font-bold text-amber-400">{categories_tied}T</span>
          </div>
          {winPct != null && (
            <span className={cn(
              'text-[10px] font-semibold px-2 py-0.5 rounded border',
              winPct >= 65 ? 'text-emerald-400 border-emerald-800/40 bg-emerald-900/20' :
              winPct <= 35 ? 'text-rose-400 border-rose-800/40 bg-rose-900/20' :
              'text-amber-400 border-amber-800/40 bg-amber-900/20',
            )}>
              {winPct}% win prob
            </span>
          )}
        </div>
      </div>

      {/* Category grid */}
      <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-9 gap-2">
        {rows.map((row) => {
          const outcome = statusToLabel(row.status)
          const my = row.my_current
          const opp = row.opp_current
          const lower = row.is_lower_better
          const fmtStat = (v: number | null) => {
            if (v === null || v === undefined) return '–'
            if (['AVG', 'OPS'].includes(row.category)) return v.toFixed(3).replace(/^0\./, '.')
            if (['ERA', 'WHIP'].includes(row.category)) return v.toFixed(2)
            return Math.round(v).toString()
          }
          const outcomeBg = outcome === 'W'
            ? 'bg-emerald-900/30 border-emerald-800/40'
            : outcome === 'L'
              ? 'bg-rose-900/30 border-rose-800/40'
              : 'bg-amber-900/20 border-amber-800/30'
          const outcomeText = outcome === 'W' ? 'text-emerald-400' : outcome === 'L' ? 'text-rose-400' : 'text-amber-400'

          // Flip logic for lower-is-better cats so red always = bad for me
          const myStr = fmtStat(my)
          const oppStr = fmtStat(opp)
          const myIsAhead = my !== null && opp !== null && (lower ? my < opp : my > opp)

          return (
            <div key={row.category} className={cn('rounded p-1.5 border text-center', outcomeBg)}>
              <p className="text-[9px] text-[#7D7D7D] uppercase tracking-wider leading-none mb-1">
                {row.category_label || formatCat(row.category)}
              </p>
              <div className={cn('text-[10px] font-bold leading-none', outcomeText)}>
                {outcome}
              </div>
              <p className="text-[9px] text-[#494949] leading-none mt-1">
                <span className={myIsAhead ? 'text-white' : ''}>{myStr}</span>
                <span className="mx-0.5 text-[#2A2A2A]">·</span>
                {oppStr}
              </p>
            </div>
          )
        })}
      </div>
      <p className="text-[9px] text-[#494949] mt-2">Me · Opponent · W/L based on current stats</p>
    </div>
  )
}

// ───────────────────────────────────────────────────────────────────────────
// Optimize Panel
// ───────────────────────────────────────────────────────────────────────────

function OptimizePanel({
  data,
  onApplyMove,
  onClose,
  isApplying,
}: {
  data: RosterOptimizeResponse
  onApplyMove: (playerKey: string, slot: string) => void
  onClose: () => void
  isApplying: boolean
}) {
  return (
    <div className="bg-[#181818] border border-[#FFC000]/30 rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-[#FFC000]" />
          <p className="text-xs font-bold tracking-widest uppercase text-[#FFC000]">
            Optimized Lineup — {data.target_date}
          </p>
        </div>
        <button onClick={onClose} className="text-[#494949] hover:text-white transition-colors">
          <X className="h-4 w-4" />
        </button>
      </div>
      <p className="text-xs text-[#7D7D7D]">{data.message}</p>

      <div className="grid sm:grid-cols-2 gap-2">
        {[...data.starters, ...data.bench].map((assignment) => (
          <div key={assignment.player_key} className="flex items-center gap-3 bg-[#202020] rounded px-3 py-2">
            <SlotPill slot={assignment.assigned_slot} />
            <div className="flex-1 min-w-0">
              <p className="text-xs font-semibold text-white truncate">{assignment.player_name}</p>
              <p className="text-[10px] text-[#494949] truncate">{assignment.reasoning}</p>
            </div>
            <button
              onClick={() => onApplyMove(assignment.player_key, assignment.assigned_slot)}
              disabled={isApplying}
              className="text-[10px] text-[#FFC000] hover:text-amber-300 font-semibold whitespace-nowrap disabled:opacity-50"
            >
              Apply
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}

// ───────────────────────────────────────────────────────────────────────────
// Player Card
// ───────────────────────────────────────────────────────────────────────────

function PlayerCard({
  player,
  viewMode,
  onMove,
  isMoving,
}: {
  player: RosterPlayer
  viewMode: ViewMode
  onMove: (playerId: string, toSlot: string) => void
  isMoving: boolean
}) {
  const [selectedSlot, setSelectedSlot] = useState('')

  const eligible = player.eligible_positions ?? []
  const isPitcher = eligible.some((p) => ['SP', 'RP', 'P'].includes(p))
  const displayCats = isPitcher ? PITCHER_DISPLAY : BATTER_DISPLAY
  const statValues = getStatWindow(player, viewMode)

  // Move dropdown: eligible positions + universal slots, deduplicated
  const moveOptions = Array.from(new Set([...eligible, ...UNIVERSAL_SLOTS]))

  const handleMoveClick = () => {
    if (!selectedSlot || !player.yahoo_player_key) return
    onMove(player.yahoo_player_key, selectedSlot)
    setSelectedSlot('')
  }

  const statusClass = STATUS_COLORS[player.status] ?? STATUS_COLORS.playing
  const statusLabel = STATUS_LABELS[player.status] ?? player.status
  const isRoS = viewMode === 'ros'
  const hasRoS = !!player.ros_projection?.values && Object.values(player.ros_projection.values).some((v) => v != null)

  return (
    <div className={cn(
      'bg-[#202020] rounded-lg p-4 flex flex-col gap-3',
      isRoS && hasRoS && 'border-l-2 border-[#FFC000]/40',
    )}>
      {/* Identity row */}
      <div className="flex items-start gap-3 flex-wrap sm:flex-nowrap">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <p className="text-sm font-bold text-white">{player.player_name}</p>
            <SlotPill slot={player.current_slot} />
            <span className={cn(
              'text-[10px] px-1.5 py-0.5 rounded border font-semibold uppercase tracking-wider',
              statusClass,
            )}>
              {statusLabel}
            </span>
            <GamePill player={player} />
          </div>
          <div className="flex items-center gap-2 mt-1 flex-wrap">
            <span className="text-xs text-[#7D7D7D]">{player.team}</span>
            {player.ownership_pct != null && player.ownership_pct > 0 && (
              <span className="text-[10px] text-[#494949]">{player.ownership_pct.toFixed(0)}% owned</span>
            )}
            {eligible.map((pos) => (
              <span key={pos} className="text-[10px] px-1.5 py-0.5 bg-[#2A2A2A] text-[#969696] rounded">
                {pos}
              </span>
            ))}
          </div>
          {player.injury_status && (
            <p className="text-xs text-rose-400 mt-1.5 flex items-center gap-1">
              <ShieldAlert className="h-3 w-3" />
              {player.injury_status}
              {player.injury_return_timeline ? ` · ${player.injury_return_timeline}` : ''}
            </p>
          )}
        </div>

        {/* Move controls */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <select
            value={selectedSlot}
            onChange={(e) => setSelectedSlot(e.target.value)}
            className="bg-[#2A2A2A] text-xs text-white border border-[#494949] rounded px-2 py-1.5 focus:outline-none focus:border-[#FFC000] min-w-[90px]"
          >
            <option value="">Move to…</option>
            {moveOptions.map((pos) => (
              <option key={pos} value={pos}>{pos}</option>
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

      {/* Stats grid */}
      <div>
        {isRoS && !hasRoS ? (
          <p className="text-[10px] text-[#494949] italic">RoS projection not available for this player</p>
        ) : (
          <div className="grid grid-cols-4 sm:grid-cols-8 gap-x-3 gap-y-1">
            {displayCats.map((cat) => (
              <div key={cat} className="text-center">
                <p className={cn(
                  'text-[9px] uppercase tracking-wider',
                  isRoS ? 'text-[#FFC000]/60' : 'text-[#494949]',
                )}>
                  {formatCat(cat)}
                </p>
                <p className={cn(
                  'text-xs font-semibold tabular-nums',
                  isRoS ? 'text-[#FFC000]' : 'text-white',
                )}>
                  {getStat(statValues, cat)}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// ───────────────────────────────────────────────────────────────────────────
// Main Page
// ───────────────────────────────────────────────────────────────────────────

const POSITION_FILTERS: PosFilter[] = ['All', 'SP', 'RP', 'OF', '1B', '2B', '3B', 'SS', 'C']

export default function RosterPage() {
  const queryClient = useQueryClient()
  const [viewMode, setViewMode] = useState<ViewMode>('season')
  const [sortMode, setSortMode] = useState<SortMode>('default')
  const [posFilter, setPosFilter] = useState<PosFilter>('All')
  const [moveError, setMoveError] = useState<string | null>(null)
  const [moveSuccess, setMoveSuccess] = useState<string | null>(null)
  const [optimizeResult, setOptimizeResult] = useState<RosterOptimizeResponse | null>(null)

  const roster = useQuery({
    queryKey: ['roster'],
    queryFn: endpoints.getRoster,
    staleTime: 5 * 60_000,
  })

  const budget = useQuery({
    queryKey: ['budget'],
    queryFn: endpoints.getBudget,
    staleTime: 10 * 60_000,
  })

  const scoreboard = useQuery({
    queryKey: ['scoreboard'],
    queryFn: endpoints.getScoreboard,
    staleTime: 5 * 60_000,
    retry: 1,
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

  const optimizeMutation = useMutation({
    mutationFn: () => endpoints.optimizeRoster(),
    onSuccess: (data: RosterOptimizeResponse) => {
      setOptimizeResult(data)
    },
    onError: (err: Error) => {
      setMoveError(`Optimize failed: ${err.message}`)
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

  // Keep hooks before early returns; loading/error/empty states render below.
  const filteredSorted = useMemo(() => {
    const players = roster.data?.players ?? []
    let filteredPlayers = [...players]

    if (posFilter !== 'All') {
      filteredPlayers = filteredPlayers.filter((p) =>
        p.eligible_positions.some((pos) => pos === posFilter || pos.startsWith(posFilter))
      )
    }

    if (sortMode === 'name') {
      filteredPlayers.sort((a, b) => a.player_name.localeCompare(b.player_name))
    } else if (sortMode === 'ros_value') {
      filteredPlayers.sort((a, b) => rosValueScore(b) - rosValueScore(a))
    }

    return filteredPlayers
  }, [roster.data?.players, posFilter, sortMode])

  // ── Loading / Error / Empty states ──────────────────────────────────────
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
          <button onClick={() => roster.refetch()} className="mt-4 text-xs text-[#FFC000] hover:text-amber-300 font-semibold">
            Retry
          </button>
        </div>
      </div>
    )
  }

  const data = roster.data

  if (!data || data.players.length === 0) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-2">
          <Users className="h-3.5 w-3.5 text-[#FFC000]" />
          <span className="text-xs font-bold tracking-widest uppercase text-[#FFC000]">My Roster</span>
        </div>
        <div className="bg-[#202020] rounded-lg p-8 text-center">
          <p className="text-xs font-semibold tracking-widest uppercase text-[#494949]">Empty Roster</p>
          <p className="text-[#7D7D7D] text-sm mt-2">No players found on your roster.</p>
        </div>
      </div>
    )
  }

  // Group players by status (only when default sort)
  const useGrouped = sortMode === 'default' && posFilter === 'All'
  const ilPlayers = filteredSorted.filter((p) => p.status === 'IL')
  const activePlayers = filteredSorted.filter((p) => p.status === 'playing' || p.status === 'probable')
  const otherPlayers = filteredSorted.filter((p) => p.status !== 'IL' && p.status !== 'playing' && p.status !== 'probable')

  // Human-readable team label
  const teamLabel = (() => {
    const match = data.team_key.match(/\.t\.(\d+)$/)
    return match ? `Team ${match[1]}` : data.team_key
  })()

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-2">
          <Users className="h-3.5 w-3.5 text-[#FFC000]" />
          <span className="text-xs font-bold tracking-widest uppercase text-[#FFC000]">My Roster</span>
          <span className="text-[10px] text-[#494949]">· {teamLabel} · {data.count} players</span>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {data.freshness?.computed_at && (
            <span className="flex items-center gap-1 text-[10px] text-[#494949]">
              <Clock className="h-3 w-3" />
              {new Date(data.freshness.computed_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
          )}
          <button
            onClick={() => optimizeMutation.mutate()}
            disabled={optimizeMutation.isPending}
            className={cn(
              'flex items-center gap-1.5 text-[10px] px-3 py-1.5 rounded font-semibold tracking-wider uppercase transition-colors',
              optimizeMutation.isPending
                ? 'bg-[#2A2A2A] text-[#494949] cursor-not-allowed'
                : 'bg-[#FFC000] text-black hover:bg-amber-300',
            )}
          >
            {optimizeMutation.isPending
              ? <><Loader2 className="h-3 w-3 animate-spin" /> Optimizing…</>
              : <><Sparkles className="h-3 w-3" /> Optimize Lineup</>
            }
          </button>
        </div>
      </div>

      {/* Staleness warning */}
      {data.freshness?.is_stale && (
        <div className="bg-amber-900/20 border border-amber-700/40 rounded-lg p-3 flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-amber-400 flex-shrink-0" />
          <span className="text-sm text-amber-300">Roster data may be stale — last fetched over an hour ago.</span>
        </div>
      )}

      {/* Feedback banners */}
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

      {/* Optimize result panel */}
      {optimizeResult && (
        <OptimizePanel
          data={optimizeResult}
          onApplyMove={handleMove}
          onClose={() => setOptimizeResult(null)}
          isApplying={moveMutation.isPending}
        />
      )}

      {/* Budget panel */}
      {budget.data && <BudgetPanel budget={budget.data.budget} />}

      {/* Matchup context strip */}
      {scoreboard.data && <MatchupStrip scoreboard={scoreboard.data} />}

      {/* Category summary */}
      <CategorySummary players={data.players} viewMode={viewMode} />

      {/* View toggle + sort + filter controls */}
      <div className="flex flex-col sm:flex-row gap-3 flex-wrap">
        <ViewToggle mode={viewMode} onChange={setViewMode} />

        <div className="flex items-center gap-2 flex-wrap">
          {/* Sort */}
          <div className="flex items-center gap-1 bg-[#181818] border border-[#2A2A2A] rounded-lg p-1">
            {([['default', 'Default'], ['name', 'A–Z'], ['ros_value', 'RoS Value']] as const).map(([val, label]) => (
              <button
                key={val}
                onClick={() => setSortMode(val)}
                className={cn(
                  'text-[10px] px-2.5 py-1.5 rounded font-semibold tracking-wider uppercase transition-colors',
                  sortMode === val ? 'bg-[#2A2A2A] text-white border border-[#494949]' : 'text-[#494949] hover:text-[#969696]',
                )}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Position filter */}
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
      </div>

      {/* Player list */}
      {useGrouped ? (
        <>
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
                  viewMode={viewMode}
                  onMove={handleMove}
                  isMoving={moveMutation.isPending}
                />
              ))}
            </div>
          )}

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
                  viewMode={viewMode}
                  onMove={handleMove}
                  isMoving={moveMutation.isPending}
                />
              ))}
            </div>
          )}

          {otherPlayers.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <CalendarOff className="h-3.5 w-3.5 text-[#494949]" />
                <p className="text-xs font-semibold tracking-widest uppercase text-[#494949]">
                  Not Active · {otherPlayers.length}
                </p>
              </div>
              {otherPlayers.map((player) => (
                <PlayerCard
                  key={player.yahoo_player_key ?? player.player_name}
                  player={player}
                  viewMode={viewMode}
                  onMove={handleMove}
                  isMoving={moveMutation.isPending}
                />
              ))}
            </div>
          )}
        </>
      ) : (
        <div className="space-y-3">
          {filteredSorted.length === 0 ? (
            <div className="bg-[#202020] rounded-lg p-8 text-center">
              <p className="text-[#494949] text-sm">No players match this filter.</p>
            </div>
          ) : (
            filteredSorted.map((player) => (
              <PlayerCard
                key={player.yahoo_player_key ?? player.player_name}
                player={player}
                viewMode={viewMode}
                onMove={handleMove}
                isMoving={moveMutation.isPending}
              />
            ))
          )}
        </div>
      )}
    </div>
  )
}
