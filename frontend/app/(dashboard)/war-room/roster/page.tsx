'use client'

import { useState, useCallback, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import type { RosterPlayer, RosterMoveResponse, RosterOptimizeResponse, BudgetData, ScoreboardResponse, RotoCategory } from '@/lib/types'
import { CATEGORY_COLOR } from '@/lib/types'
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
  playing: 'bg-status-safe/10 text-status-safe border-status-safe/30',
  probable: 'bg-status-bubble/10 text-status-bubble border-status-bubble/30',
  not_playing: 'bg-status-lost/10 text-status-lost border-status-lost/30',
  IL: 'bg-status-lost/10 text-status-lost border-status-lost/30',
  minors: 'bg-text-muted/10 text-text-muted border-text-muted/30',
}

const STATUS_LABELS: Record<string, string> = {
  playing: 'Active',
  probable: 'DTD',
  not_playing: 'Out',
  IL: 'IL',
  minors: 'Minors',
}

const SLOT_COLORS: Record<string, string> = {
  BN: 'bg-bg-elevated text-text-secondary',
  IL: 'bg-status-lost/10 text-status-lost',
  IL60: 'bg-status-lost/10 text-status-lost',
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
    <div className="flex items-center gap-1 bg-bg-surface border border-border-subtle rounded-lg p-1">
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          className={cn(
            'text-[10px] px-3 py-1.5 rounded font-semibold tracking-wider uppercase transition-colors',
            mode === opt.value
              ? opt.value === 'ros'
                ? 'bg-accent-gold text-black'
                : 'bg-bg-elevated text-text-primary border border-border-default'
              : 'text-text-muted hover:text-text-secondary',
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
      <span className="flex items-center gap-1 text-[10px] bg-status-safe/10 text-status-safe border border-status-safe/20 rounded px-1.5 py-0.5 font-medium">
        <Calendar className="h-2.5 w-2.5" />
        vs {player.game_context.opponent}
        {timeStr ? ` · ${timeStr}` : ''}
      </span>
    )
  }
  return (
    <span className="flex items-center gap-1 text-[10px] text-text-muted border border-border-subtle rounded px-1.5 py-0.5">
      <CalendarOff className="h-2.5 w-2.5" />
      No Game
    </span>
  )
}

function SlotPill({ slot }: { slot?: string | null }) {
  if (!slot) return null
  const colorClass = SLOT_COLORS[slot] ?? 'bg-bg-elevated text-text-secondary'
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
    <div className="bg-bg-surface border border-border-subtle rounded-lg p-4">
      <div className="flex items-center gap-2 mb-3">
        <Gauge className="h-3.5 w-3.5 text-accent-gold" />
        <p className="text-xs font-semibold tracking-widest uppercase text-text-secondary">
          Constraints
        </p>
      </div>
      <div className="grid grid-cols-3 gap-4">
        {/* Weekly Moves */}
        <div>
          <p className="text-[9px] text-text-muted uppercase tracking-wider mb-1">Weekly Moves</p>
          <div className="flex items-center gap-1.5 mb-1">
            <div className="flex-1 h-1.5 bg-bg-inset rounded-full overflow-hidden">
              <div
                className={cn('h-full rounded-full', acqPct >= 80 ? 'bg-status-lost' : acqPct >= 60 ? 'bg-status-bubble' : 'bg-status-safe')}
                style={{ width: `${acqPct}%` }}
              />
            </div>
            <span className={cn('text-xs font-bold tabular-nums', movesWarning ? 'text-status-bubble' : 'text-text-primary')}>
              {budget.acquisitions_used}/{budget.acquisition_limit}
            </span>
          </div>
          <p className={cn('text-[10px]', movesWarning ? 'text-status-bubble font-semibold' : 'text-text-muted')}>
            {movesLeft} remaining
          </p>
        </div>

        {/* IP Pace */}
        <div>
          <p className="text-[9px] text-text-muted uppercase tracking-wider mb-1">IP Pace</p>
          <div className="flex items-center gap-1.5 mb-1">
            <div className="flex-1 h-1.5 bg-bg-inset rounded-full overflow-hidden">
              <div
                className="h-full rounded-full bg-blue-500"
                style={{ width: `${ipPct}%` }}
              />
            </div>
            <span className={cn('text-xs font-bold tabular-nums', paceColor)}>
              {budget.ip_pace}
            </span>
          </div>
          <p className="text-[10px] text-text-muted">
            {budget.ip_accumulated.toFixed(1)} / {budget.ip_minimum} IP
          </p>
        </div>

        {/* IL */}
        <div>
          <p className="text-[9px] text-text-muted uppercase tracking-wider mb-1">IL Slots</p>
          <p className="text-sm font-bold text-text-primary tabular-nums">
            {budget.il_used}<span className="text-text-muted font-normal">/{budget.il_total}</span>
          </p>
          {budget.il_used < budget.il_total && (
            <p className="text-[10px] text-status-safe font-semibold mt-0.5">
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
        <div className="bg-bg-surface border border-border-subtle rounded-lg p-4 text-center">
          <p className="text-xs text-text-muted">No RoS projections available for active starters.</p>
        </div>
      )
    }
    return (
      <div className="bg-bg-surface border border-border-subtle rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-3.5 w-3.5 text-accent-gold" />
            <p className="text-xs font-semibold tracking-widest uppercase text-accent-gold">
              Projected Team Edge
            </p>
          </div>
          <p className="text-[9px] text-text-muted">
            Z-scores vs. league avg · green = strength · red = weakness
          </p>
        </div>
        <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-9 gap-3">
          {allCats.map((cat) => {
            const vals = zSums[cat]
            const catColor = CATEGORY_COLOR[cat as RotoCategory]
            if (!vals || vals.length === 0) return (
              <div key={cat} className="text-center">
                <div className="flex items-center justify-center gap-1 mb-0.5">
                  {catColor && <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: catColor }} />}
                  <p className="text-[9px] text-text-muted uppercase tracking-wider">{formatCat(cat)}</p>
                </div>
                <p className="text-sm font-bold text-bg-elevated">–</p>
              </div>
            )
            // Rate cats: average z; counting cats: sum z
            const z = RATE_CATS.has(cat)
              ? vals.reduce((a, b) => a + b, 0) / vals.length
              : vals.reduce((a, b) => a + b, 0)
            const color = z >= 1.5
              ? 'text-status-safe'
              : z >= 0.3
                ? 'text-status-lead'
                : z <= -1.5
                  ? 'text-status-lost'
                  : z <= -0.3
                    ? 'text-status-behind'
                    : 'text-text-secondary'
            return (
              <div key={cat} className="text-center">
                <div className="flex items-center justify-center gap-1 mb-0.5">
                  {catColor && <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: catColor }} />}
                  <p className="text-[9px] text-text-muted uppercase tracking-wider">{formatCat(cat)}</p>
                </div>
                <p className={cn('text-sm font-bold tabular-nums', color)}>
                  {z > 0 ? '+' : ''}{z.toFixed(1)}
                </p>
              </div>
            )
          })}
        </div>
        <p className="text-[9px] text-text-muted mt-2">
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
    <div className="bg-bg-surface border border-border-subtle rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <BarChart3 className="h-3.5 w-3.5 text-accent-gold" />
          <p className="text-xs font-semibold tracking-widest uppercase text-text-secondary">
            Category Totals
          </p>
        </div>
        <p className="text-[9px] text-text-muted">
          Active starters · {activePlayers.length} players · Rate stats = median
        </p>
      </div>
      <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-9 gap-3">
        {allCats.map((cat) => {
          const catColor = CATEGORY_COLOR[cat as RotoCategory]
          return (
            <div key={cat} className="text-center">
              <div className="flex items-center justify-center gap-1 mb-0.5">
                {catColor && <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: catColor }} />}
                <p className="text-[9px] text-text-muted uppercase tracking-wider">{formatCat(cat)}</p>
              </div>
              <p className="text-sm font-bold text-text-primary tabular-nums">
                {getDisplayVal(cat)}
              </p>
            </div>
          )
        })}
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
    <div className="bg-bg-surface border border-border-subtle rounded-lg p-4">
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <Swords className="h-3.5 w-3.5 text-accent-gold" />
          <p className="text-xs font-semibold tracking-widest uppercase text-text-secondary">
            This Week · vs {opponent_name}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="text-xs font-bold text-status-safe">{categories_won}W</span>
            <span className="text-[10px] text-text-muted">·</span>
            <span className="text-xs font-bold text-status-lost">{categories_lost}L</span>
            <span className="text-[10px] text-text-muted">·</span>
            <span className="text-xs font-bold text-status-bubble">{categories_tied}T</span>
          </div>
          {winPct != null && (
            <span className={cn(
              'text-[10px] font-semibold px-2 py-0.5 rounded border',
              winPct >= 65 ? 'text-status-safe border-status-safe/30 bg-status-safe/10' :
              winPct <= 35 ? 'text-status-lost border-status-lost/30 bg-status-lost/10' :
              'text-status-bubble border-status-bubble/30 bg-status-bubble/10',
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
            ? 'bg-status-safe/10 border-status-safe/30'
            : outcome === 'L'
              ? 'bg-status-lost/10 border-status-lost/30'
              : 'bg-status-bubble/10 border-status-bubble/30'
          const outcomeText = outcome === 'W' ? 'text-status-safe' : outcome === 'L' ? 'text-status-lost' : 'text-status-bubble'

          // Flip logic for lower-is-better cats so red always = bad for me
          const myStr = fmtStat(my)
          const oppStr = fmtStat(opp)
          const myIsAhead = my !== null && opp !== null && (lower ? my < opp : my > opp)

          return (
            <div key={row.category} className={cn('rounded p-1.5 border text-center', outcomeBg)}>
              <p className="text-[9px] text-text-secondary uppercase tracking-wider leading-none mb-1">
                {row.category_label || formatCat(row.category)}
              </p>
              <div className={cn('text-[10px] font-bold leading-none', outcomeText)}>
                {outcome}
              </div>
              <p className="text-[9px] text-text-muted leading-none mt-1">
                <span className={myIsAhead ? 'text-text-primary' : ''}>{myStr}</span>
                <span className="mx-0.5 text-border-subtle">·</span>
                {oppStr}
              </p>
            </div>
          )
        })}
      </div>
      <p className="text-[9px] text-text-muted mt-2">Me · Opponent · W/L based on current stats</p>
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
    <div className="bg-bg-surface border border-accent-gold/30 rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-accent-gold" />
          <p className="text-xs font-bold tracking-widest uppercase text-accent-gold">
            Optimized Lineup — {data.target_date}
          </p>
        </div>
        <button onClick={onClose} className="text-text-muted hover:text-text-primary transition-colors">
          <X className="h-4 w-4" />
        </button>
      </div>
      <p className="text-xs text-text-secondary">{data.message}</p>

      <div className="grid sm:grid-cols-2 gap-2">
        {[...data.starters, ...data.bench].map((assignment) => (
          <div key={assignment.player_key} className="flex items-center gap-3 bg-bg-elevated rounded px-3 py-2">
            <SlotPill slot={assignment.assigned_slot} />
            <div className="flex-1 min-w-0">
              <p className="text-xs font-semibold text-text-primary truncate">{assignment.player_name}</p>
              <p className="text-[10px] text-text-muted truncate">{assignment.reasoning}</p>
            </div>
            <button
              onClick={() => onApplyMove(assignment.player_key, assignment.assigned_slot)}
              disabled={isApplying}
              className="text-[10px] text-accent-gold hover:text-amber-300 font-semibold whitespace-nowrap disabled:opacity-50"
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

  const isIL = player.status === 'IL'

  return (
    <div className={cn(
      'bg-bg-surface border border-border-subtle rounded-lg p-4 flex flex-col gap-3 hover:bg-bg-elevated transition-colors duration-150',
      isRoS && hasRoS && 'border-l-2 border-accent-gold/40',
      isIL && 'bg-rose-900/10 border-rose-900/30',
    )}>
      {/* Identity row */}
      <div className="flex items-start gap-3 flex-wrap sm:flex-nowrap">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <p className="text-sm font-bold text-text-primary">{player.player_name}</p>
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
            <span className="text-xs text-text-secondary">{player.team}</span>
            {player.ownership_pct != null && player.ownership_pct > 0 && (
              <span className="text-[10px] text-text-muted">{player.ownership_pct.toFixed(0)}% owned</span>
            )}
            {eligible.map((pos) => (
              <span key={pos} className="text-[10px] px-1.5 py-0.5 bg-bg-elevated text-text-secondary rounded">
                {pos}
              </span>
            ))}
          </div>
          {player.injury_status && (
            <p className="text-xs text-status-lost mt-1.5 flex items-center gap-1">
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
            className="bg-bg-elevated text-xs text-text-primary border border-border-default rounded px-2 py-1.5 focus:outline-none focus:border-accent-gold min-w-[90px]"
          >
            <option value="" className="bg-bg-elevated text-text-primary">Move to…</option>
            {moveOptions.map((pos) => (
              <option key={pos} value={pos} className="bg-bg-elevated text-text-primary">{pos}</option>
            ))}
          </select>
          <button
            onClick={handleMoveClick}
            disabled={!selectedSlot || isMoving}
            className={cn(
              'p-1.5 rounded transition-colors',
              selectedSlot && !isMoving
                ? 'bg-accent-gold text-black hover:bg-amber-300'
                : 'bg-bg-elevated text-text-muted cursor-not-allowed',
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
          <p className="text-[10px] text-text-muted italic">RoS projection not available for this player</p>
        ) : (
          <div className="grid grid-cols-4 sm:grid-cols-8 gap-x-3 gap-y-1">
            {displayCats.map((cat) => (
              <div key={cat} className="text-center">
                <p className={cn(
                  'text-[9px] uppercase tracking-wider',
                  isRoS ? 'text-accent-gold/60' : 'text-text-muted',
                )}>
                  {formatCat(cat)}
                </p>
                <p className={cn(
                  'text-xs font-semibold tabular-nums',
                  isRoS ? 'text-accent-gold' : 'text-text-primary',
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
        <div className="flex items-center gap-2 text-text-secondary">
          <Loader2 className="h-5 w-5 animate-spin text-accent-gold" />
          <span className="text-sm">Loading roster…</span>
        </div>
      </div>
    )
  }

  if (roster.isError) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center">
        <div className="bg-bg-surface border border-border-subtle rounded-lg p-6 max-w-md w-full">
          <div className="flex items-center gap-2 text-status-lost mb-2">
            <AlertCircle className="h-5 w-5" />
            <span className="text-sm font-semibold">Failed to load roster</span>
          </div>
          <p className="text-text-secondary text-sm">
            {roster.error instanceof Error ? roster.error.message : 'Unknown error'}
          </p>
          <button onClick={() => roster.refetch()} className="mt-4 text-xs text-accent-gold hover:text-amber-300 font-semibold">
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
          <Users className="h-3.5 w-3.5 text-accent-gold" />
          <span className="text-xs font-bold tracking-widest uppercase text-accent-gold">My Roster</span>
        </div>
        <div className="bg-bg-surface border border-border-subtle rounded-lg p-8 text-center">
          <p className="text-xs font-semibold tracking-widest uppercase text-text-muted">Empty Roster</p>
          <p className="text-text-secondary text-sm mt-2">No players found on your roster.</p>
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
          <Users className="h-3.5 w-3.5 text-accent-gold" />
          <span className="text-xs font-bold tracking-widest uppercase text-accent-gold">My Roster</span>
          <span className="text-[10px] text-text-muted">· {teamLabel} · {data.count} players</span>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {data.freshness?.computed_at && (
            <span className="flex items-center gap-1 text-[10px] text-text-muted">
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
                ? 'bg-bg-elevated text-text-muted cursor-not-allowed'
                : 'bg-accent-gold text-black hover:bg-amber-300',
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
        <div className="bg-status-bubble/10 border border-status-bubble/30 rounded-lg p-3 flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-status-bubble flex-shrink-0" />
          <span className="text-sm text-status-bubble">Roster data may be stale — last fetched over an hour ago.</span>
        </div>
      )}

      {/* Feedback banners */}
      {moveError && (
        <div className="bg-status-lost/10 border border-status-lost/30 rounded-lg p-3 flex items-center gap-2">
          <AlertCircle className="h-4 w-4 text-status-lost flex-shrink-0" />
          <span className="text-sm text-status-lost">{moveError}</span>
        </div>
      )}
      {moveSuccess && (
        <div className="bg-status-safe/10 border border-status-safe/30 rounded-lg p-3 flex items-center gap-2">
          <Activity className="h-4 w-4 text-status-safe flex-shrink-0" />
          <span className="text-sm text-status-safe">{moveSuccess}</span>
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
          <div className="flex items-center gap-1 bg-bg-surface border border-border-subtle rounded-lg p-1">
            {([['default', 'Default'], ['name', 'A–Z'], ['ros_value', 'RoS Value']] as const).map(([val, label]) => (
              <button
                key={val}
                onClick={() => setSortMode(val)}
                className={cn(
                  'text-[10px] px-2.5 py-1.5 rounded font-semibold tracking-wider uppercase transition-colors',
                  sortMode === val ? 'bg-bg-elevated text-text-primary border border-border-default' : 'text-text-muted hover:text-text-secondary',
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
                    ? 'bg-bg-elevated text-text-primary border border-border-default'
                    : 'text-text-muted hover:text-text-secondary',
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
                <TrendingUp className="h-3.5 w-3.5 text-status-safe" />
                <p className="text-xs font-semibold tracking-widest uppercase text-text-secondary">
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
                <ShieldAlert className="h-3.5 w-3.5 text-status-lost" />
                <p className="text-xs font-semibold tracking-widest uppercase text-status-lost">
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
                <CalendarOff className="h-3.5 w-3.5 text-text-muted" />
                <p className="text-xs font-semibold tracking-widest uppercase text-text-muted">
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
            <div className="bg-bg-surface border border-border-subtle rounded-lg p-8 text-center">
              <p className="text-text-muted text-sm">No players match this filter.</p>
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
