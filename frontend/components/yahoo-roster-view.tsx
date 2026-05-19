'use client'

import { useState } from 'react'
import type { RosterPlayer } from '@/lib/types'
import { cn } from '@/lib/utils'
import {
  TrendingUp, TrendingDown, Minus, ShieldAlert,
  ArrowUpCircle, ArrowDownCircle,
} from 'lucide-react'

// ---------------------------------------------------------------------------
// Position slot definitions — mirrors Yahoo Fantasy Baseball roster slots
// ---------------------------------------------------------------------------
export const ROSTER_SLOTS: { id: string; label: string; type: 'bat' | 'pit' | 'flex' | 'bench' | 'il' }[] = [
  { id: 'C',   label: 'C',   type: 'bat' },
  { id: '1B',  label: '1B',  type: 'bat' },
  { id: '2B',  label: '2B',  type: 'bat' },
  { id: '3B',  label: '3B',  type: 'bat' },
  { id: 'SS',  label: 'SS',  type: 'bat' },
  { id: 'OF1', label: 'OF',  type: 'bat' },
  { id: 'OF2', label: 'OF',  type: 'bat' },
  { id: 'OF3', label: 'OF',  type: 'bat' },
  { id: 'Util', label: 'UTIL', type: 'flex' },
  { id: 'SP1', label: 'SP',  type: 'pit' },
  { id: 'SP2', label: 'SP',  type: 'pit' },
  { id: 'RP1', label: 'RP',  type: 'pit' },
  { id: 'RP2', label: 'RP',  type: 'pit' },
  { id: 'P1',  label: 'P',   type: 'pit' },
  { id: 'P2',  label: 'P',   type: 'pit' },
  { id: 'BN1', label: 'BN',  type: 'bench' },
  { id: 'BN2', label: 'BN',  type: 'bench' },
  { id: 'BN3', label: 'BN',  type: 'bench' },
  { id: 'BN4', label: 'BN',  type: 'bench' },
  { id: 'BN5', label: 'BN',  type: 'bench' },
  { id: 'IL1', label: 'IL',  type: 'il' },
  { id: 'IL2', label: 'IL',  type: 'il' },
]

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function slotIsPitcher(slotId: string): boolean {
  return ['SP1','SP2','RP1','RP2','P1','P2'].includes(slotId)
}

function getSlotCategoryLabel(slotId: string): string {
  const type = ROSTER_SLOTS.find((s) => s.id === slotId)?.type
  if (type === 'bat') return 'Batters'
  if (type === 'pit') return 'Pitchers'
  if (type === 'flex') return 'Utility'
  if (type === 'bench') return 'Bench'
  return 'Injured List'
}

// ---------------------------------------------------------------------------
// Hot/Cold streak indicator
// ---------------------------------------------------------------------------
function StreakBadge({ player }: { player: RosterPlayer }) {
  const stats = player.rolling_7d
  if (!stats) return null

  // Simple heuristic: if rolling OPS/ERA is trending well vs season average
  const rollingOps = stats.values?.['OPS'] ?? null
  const seasonOps = player.season_stats?.values?.['OPS'] ?? 0
  const isHot = rollingOps != null && seasonOps > 0 && rollingOps > seasonOps * 1.05
  const isCold = rollingOps != null && seasonOps > 0 && rollingOps < seasonOps * 0.95

  if (isHot) {
    return (
      <span className="inline-flex items-center gap-0.5 text-[10px] font-bold text-amber-400">
        <TrendingUp className="h-3 w-3" /> HOT
      </span>
    )
  }
  if (isCold) {
    return (
      <span className="inline-flex items-center gap-0.5 text-[10px] font-bold text-sky-400">
        <TrendingDown className="h-3 w-3" /> COLD
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-0.5 text-[10px] text-text-muted">
      <Minus className="h-3 w-3" /> EVEN
    </span>
  )
}

// ---------------------------------------------------------------------------
// Player card — Yahoo-style compact row
// ---------------------------------------------------------------------------
function SlotPlayerCard({
  player,
  slotId,
  onMove,
  isMoving,
}: {
  player: RosterPlayer
  slotId: string
  onMove?: (player: RosterPlayer, targetSlot: string) => void
  isMoving?: boolean
}) {
  const [isExpanded, setIsExpanded] = useState(false)
  const isPitcher = slotIsPitcher(slotId) || player.eligible_positions?.some((p) => ['SP','RP'].includes(p))
  const isBench = slotId.startsWith('BN')
  const isIL   = slotId.startsWith('IL')
  const season = player.season_stats
  const ros    = player.ros_projection

  const sv = season?.values
  const rv = ros?.values
  const keyStats = isPitcher
    ? [
        { label: 'ERA',  val: sv?.['ERA']  != null ? Number(sv['ERA']).toFixed(2)  : '—', rosVal: rv?.['ERA']  != null ? Number(rv['ERA']).toFixed(2)  : undefined },
        { label: 'WHIP', val: sv?.['WHIP'] != null ? Number(sv['WHIP']).toFixed(2) : '—', rosVal: rv?.['WHIP'] != null ? Number(rv['WHIP']).toFixed(2) : undefined },
        { label: 'K/9',  val: sv?.['K_9']  != null ? Number(sv['K_9']).toFixed(1)  : '—', rosVal: rv?.['K_9']  != null ? Number(rv['K_9']).toFixed(1)  : undefined },
        { label: 'W',    val: sv?.['W']   ?? '—', rosVal: rv?.['W'] },
        { label: 'SV',   val: sv?.['NSV'] ?? '—', rosVal: rv?.['NSV'] },
      ]
    : [
        { label: 'AVG',  val: sv?.['AVG']  != null ? Number(sv['AVG']).toFixed(3).replace(/^0/, '')  : '—', rosVal: rv?.['AVG']  != null ? Number(rv['AVG']).toFixed(3).replace(/^0/, '')  : undefined },
        { label: 'HR',   val: sv?.['HR_B'] ?? '—', rosVal: rv?.['HR_B'] },
        { label: 'RBI',  val: sv?.['RBI']  ?? '—', rosVal: rv?.['RBI'] },
        { label: 'SB',   val: sv?.['NSB']  ?? '—', rosVal: rv?.['NSB'] },
        { label: 'OPS',  val: sv?.['OPS']  != null ? Number(sv['OPS']).toFixed(3).replace(/^0/, '')  : '—', rosVal: rv?.['OPS']  != null ? Number(rv['OPS']).toFixed(3).replace(/^0/, '')  : undefined },
      ]

  return (
    <div
      className={cn(
        'group relative rounded-lg border transition-all duration-150',
        isIL
          ? 'bg-status-lost/5 border-status-lost/20 opacity-60'
          : isBench
            ? 'bg-bg-surface/60 border-border-subtle/50'
            : 'bg-bg-surface border-border-subtle hover:border-border-default hover:bg-bg-elevated',
      )}
    >
      {/* Main row */}
      <div className="flex items-center gap-2 px-3 py-2 cursor-pointer" onClick={() => setIsExpanded((v) => !v)}>
        {/* Player photo / avatar placeholder */}
        <div className="h-9 w-9 rounded-full bg-zinc-800 flex items-center justify-center flex-shrink-0 overflow-hidden">
          {player.mlbam_id ? (
            <img
              src={`https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:silo:current.png/r_max/w_180,q_auto:best,f_auto/v1/people/${player.mlbam_id}/headshot/silo/current`}
              alt={player.player_name}
              className="h-full w-full object-cover"
              onError={(e) => { (e.target as HTMLImageElement).src = '' }}
            />
          ) : (
            <span className="text-xs font-bold text-zinc-500">
              {player.player_name.split(' ').map((n) => n[0]).join('')}
            </span>
          )}
        </div>

        {/* Name + team + positions */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className={cn(
              'text-sm font-semibold truncate',
              isIL ? 'text-text-muted' : 'text-text-primary',
            )}>
              {player.player_name}
            </span>
            {player.injury_status && (
              <span className="text-[9px] px-1 py-0.5 bg-status-lost/10 text-status-lost border border-status-lost/30 rounded font-bold">
                {player.injury_status}
              </span>
            )}
            <StreakBadge player={player} />
          </div>
          <div className="flex items-center gap-1.5 mt-0.5">
            <span className="text-[10px] text-text-muted">{player.team}</span>
            <span className="text-[9px] text-border-default">|</span>
            <div className="flex gap-0.5">
              {player.eligible_positions?.map((pos) => (
                <span
                  key={pos}
                  className={cn(
                    'text-[9px] px-1 py-0.5 rounded font-semibold',
                    ['SP','RP','P'].includes(pos)
                      ? 'bg-purple-900/30 text-purple-400'
                      : ['C'].includes(pos)
                        ? 'bg-amber-900/30 text-amber-400'
                        : ['1B','3B'].includes(pos)
                          ? 'bg-orange-900/30 text-orange-400'
                          : ['2B','SS','MI'].includes(pos)
                            ? 'bg-sky-900/30 text-sky-400'
                            : ['OF','LF','CF','RF'].includes(pos)
                              ? 'bg-emerald-900/30 text-emerald-400'
                              : 'bg-bg-elevated text-text-secondary',
                  )}
                >
                  {pos}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Quick stats row */}
        <div className="hidden sm:flex items-center gap-3 flex-shrink-0">
          {keyStats.slice(0, 4).map((s) => (
            <div key={s.label} className="text-center min-w-[2.5rem]">
              <div className="text-[9px] text-text-muted uppercase tracking-wider">{s.label}</div>
              <div className="text-xs font-bold tabular-nums text-text-primary">{s.val}</div>
            </div>
          ))}
        </div>

        {/* Start/Bench toggle */}
        {onMove && !isIL && (
          <button
            onClick={(e) => {
              e.stopPropagation()
              const target = isBench ? 'Util' : 'BN1'
              onMove(player, target)
            }}
            disabled={isMoving}
            className={cn(
              'flex-shrink-0 p-1.5 rounded-md transition-colors',
              isBench
                ? 'bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 border border-emerald-500/30'
                : 'bg-bg-elevated text-text-muted hover:text-text-secondary border border-border-subtle',
              isMoving && 'opacity-50 cursor-not-allowed',
            )}
            title={isBench ? 'Start player' : 'Bench player'}
          >
            {isBench ? (
              <ArrowUpCircle className="h-4 w-4" />
            ) : (
              <ArrowDownCircle className="h-4 w-4" />
            )}
          </button>
        )}
      </div>

      {/* Expanded: full stat row */}
      {isExpanded && (
        <div className="border-t border-border-subtle/50 px-3 py-2 bg-bg-inset/30">
          <div className="grid grid-cols-5 gap-2 text-center">
            {keyStats.map((s) => (
              <div key={s.label}>
                <div className="text-[9px] text-text-muted uppercase tracking-wider">{s.label}</div>
                <div className="text-xs font-bold tabular-nums text-text-primary">{s.val}</div>
                {s.rosVal != null && s.rosVal !== s.val && (
                  <div className="text-[9px] text-text-muted tabular-nums">RoS {s.rosVal}</div>
                )}
              </div>
            ))}
          </div>
          {/* Game context */}
          {player.game_context && (
            <div className="mt-2 flex items-center gap-2 text-[10px] text-text-muted">
              <span>vs {player.game_context.opponent}</span>
              {player.game_context.opposing_sp_handedness && (
                <span className={player.game_context.opposing_sp_handedness === 'L' ? 'text-sky-400' : 'text-text-muted'}>
                  vs {player.game_context.opposing_sp_handedness}HP
                </span>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Empty slot placeholder
// ---------------------------------------------------------------------------
function EmptySlot({ slotId }: { slotId: string }) {
  const slot = ROSTER_SLOTS.find((s) => s.id === slotId)!
  return (
    <div className="flex items-center gap-2 px-3 py-2 rounded-lg border border-dashed border-border-subtle/50 bg-bg-inset/20 min-h-[3.5rem]">
      <div className="h-9 w-9 rounded-full bg-bg-elevated flex items-center justify-center flex-shrink-0">
        <span className="text-xs font-bold text-text-muted/50">+</span>
      </div>
      <div className="flex-1">
        <span className="text-sm font-medium text-text-muted/60">{slot.label}</span>
        <span className="text-[10px] text-text-muted/40 ml-2">Empty</span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Section header for position groups
// ---------------------------------------------------------------------------
function SectionHeader({ title, count, icon: Icon }: { title: string; count: number; icon: React.ComponentType<{ className?: string }> }) {
  return (
    <div className="flex items-center gap-2 mb-2 mt-4 first:mt-0">
      <Icon className="h-4 w-4 text-text-secondary" />
      <h3 className="text-xs font-bold tracking-widest uppercase text-text-secondary">
        {title}
      </h3>
      <span className="text-[10px] text-text-muted ml-auto tabular-nums">
        {count} / {ROSTER_SLOTS.filter((s) => getSlotCategoryLabel(s.id) === title).length}
      </span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main Yahoo-style roster view
// ---------------------------------------------------------------------------
interface YahooRosterViewProps {
  players: RosterPlayer[]
  onMove?: (player: RosterPlayer, targetSlot: string) => void
  isMoving?: boolean
}

export default function YahooRosterView({ players, onMove, isMoving }: YahooRosterViewProps) {
  // Build slot → player map
  const slotMap: Record<string, RosterPlayer | null> = {}

  // First pass: assign players to their exact current_slot
  const assigned = new Set<string>()
  for (const slot of ROSTER_SLOTS) {
    const pos = slot.id.replace(/\d+$/, '')
    const match = players.find((p) => {
      const key = p.yahoo_player_key ?? p.player_name
      if (assigned.has(key)) return false
      const current = p.current_slot?.toUpperCase()
      if (pos === 'OF' && current?.startsWith('OF')) return true
      if (pos === 'BN' && (current?.startsWith('BN') || !current)) return true
      if (pos === 'IL' && (current?.startsWith('IL') || p.status === 'DL')) return true
      return current === pos
    })
    if (match) {
      slotMap[slot.id] = match
      assigned.add(match.yahoo_player_key ?? match.player_name)
    } else {
      slotMap[slot.id] = null
    }
  }

  // Group by section
  const batters = ROSTER_SLOTS.filter((s) => s.type === 'bat' || s.type === 'flex')
  const pitchers = ROSTER_SLOTS.filter((s) => s.type === 'pit')
  const bench = ROSTER_SLOTS.filter((s) => s.type === 'bench')
  const il = ROSTER_SLOTS.filter((s) => s.type === 'il')

  return (
    <div className="space-y-4">
      {/* Batters section */}
      <div>
        <SectionHeader title="Batters" count={batters.filter((s) => slotMap[s.id]).length} icon={TrendingUp} />
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
          {batters.map((slot) => (
            <div key={slot.id}>
              <div className="text-[9px] text-text-muted uppercase tracking-wider mb-1 px-1">
                {slot.label}
              </div>
              {slotMap[slot.id] ? (
                <SlotPlayerCard
                  player={slotMap[slot.id]!}
                  slotId={slot.id}
                  onMove={onMove}
                  isMoving={isMoving}
                />
              ) : (
                <EmptySlot slotId={slot.id} />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Pitchers section */}
      <div>
        <SectionHeader title="Pitchers" count={pitchers.filter((s) => slotMap[s.id]).length} icon={ShieldAlert} />
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
          {pitchers.map((slot) => (
            <div key={slot.id}>
              <div className="text-[9px] text-text-muted uppercase tracking-wider mb-1 px-1">
                {slot.label}
              </div>
              {slotMap[slot.id] ? (
                <SlotPlayerCard
                  player={slotMap[slot.id]!}
                  slotId={slot.id}
                  onMove={onMove}
                  isMoving={isMoving}
                />
              ) : (
                <EmptySlot slotId={slot.id} />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Bench section */}
      <div>
        <SectionHeader title="Bench" count={bench.filter((s) => slotMap[s.id]).length} icon={Minus} />
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
          {bench.map((slot) => (
            <div key={slot.id}>
              <div className="text-[9px] text-text-muted uppercase tracking-wider mb-1 px-1">
                {slot.label}
              </div>
              {slotMap[slot.id] ? (
                <SlotPlayerCard
                  player={slotMap[slot.id]!}
                  slotId={slot.id}
                  onMove={onMove}
                  isMoving={isMoving}
                />
              ) : (
                <EmptySlot slotId={slot.id} />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* IL section */}
      {il.some((s) => slotMap[s.id]) && (
        <div>
          <SectionHeader title="Injured List" count={il.filter((s) => slotMap[s.id]).length} icon={ShieldAlert} />
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
            {il.map((slot) => (
              <div key={slot.id}>
                <div className="text-[9px] text-text-muted uppercase tracking-wider mb-1 px-1">
                  {slot.label}
                </div>
                {slotMap[slot.id] ? (
                  <SlotPlayerCard
                    player={slotMap[slot.id]!}
                    slotId={slot.id}
                    onMove={onMove}
                    isMoving={isMoving}
                  />
                ) : (
                  <EmptySlot slotId={slot.id} />
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
