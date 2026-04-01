'use client'

import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Trophy, Filter, RefreshCw, Target, Users, Star, CheckCircle2, Clock } from 'lucide-react'
import { endpoints } from '@/lib/api'
import { Card, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'
import type { FantasyPlayer, DraftSession, DraftPick } from '@/lib/types'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SESSION_STORAGE_KEY = 'cbb_draft_session_key'

const TIER_COLORS: Record<number, string> = {
  1: 'text-amber-400 bg-amber-400/10 border-amber-500/30',
  2: 'text-sky-400 bg-sky-400/10 border-sky-500/30',
  3: 'text-emerald-400 bg-emerald-400/10 border-emerald-500/30',
  4: 'text-violet-400 bg-violet-400/10 border-violet-500/30',
  5: 'text-zinc-300 bg-zinc-700/50 border-zinc-600/30',
}

const POSITION_OPTIONS = ['All', 'C', '1B', '2B', '3B', 'SS', 'OF', 'SP', 'RP']
const TYPE_OPTIONS = [
  { label: 'All', value: '' },
  { label: 'Batters', value: 'batter' },
  { label: 'Pitchers', value: 'pitcher' },
]

// ---------------------------------------------------------------------------
// Treemendous snake pick order
// R1+R2: linear 1→12, R3+: alternating snake
// ---------------------------------------------------------------------------

function getPickOrder(roundNum: number, numTeams: number): number[] {
  const positions = Array.from({ length: numTeams }, (_, i) => i + 1)
  if (roundNum <= 2) return positions
  const snakeRound = roundNum - 2
  return snakeRound % 2 === 1 ? [...positions].reverse() : [...positions]
}

function getCurrentDrafter(pickNumber: number, numTeams: number): number {
  const roundNum = Math.floor((pickNumber - 1) / numTeams) + 1
  const posInRound = (pickNumber - 1) % numTeams
  const order = getPickOrder(roundNum, numTeams)
  return order[posInRound]
}

function picksUntilMyTurn(currentPick: number, myPosition: number, numTeams: number): number | null {
  for (let i = 0; i < numTeams * 3; i++) {
    const drafter = getCurrentDrafter(currentPick + i, numTeams)
    if (drafter === myPosition) return i
  }
  return null
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

function tierBadge(tier: number) {
  const cls = TIER_COLORS[tier] ?? 'text-zinc-500 bg-zinc-800 border-zinc-700'
  return (
    <span className={cn('px-1.5 py-0.5 rounded border text-xs font-mono font-semibold', cls)}>
      T{tier}
    </span>
  )
}

function positionBadges(positions: string[]) {
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
// Filter bar (shared between both tabs)
// ---------------------------------------------------------------------------

interface FilterBarProps {
  posFilter: string
  setPosFilter: (v: string) => void
  typeFilter: string
  setTypeFilter: (v: string) => void
  tierMax: number | undefined
  setTierMax: (v: number | undefined) => void
  search: string
  setSearch: (v: string) => void
}

function FilterBar({ posFilter, setPosFilter, typeFilter, setTypeFilter, tierMax, setTierMax, search, setSearch }: FilterBarProps) {
  return (
    <div className="flex flex-wrap gap-3 items-center">
      <Filter className="h-4 w-4 text-zinc-500 flex-shrink-0" />
      <input
        type="text"
        placeholder="Search player or team..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="px-3 py-1.5 text-sm bg-zinc-800 border border-zinc-700 rounded-md text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-amber-500 w-48"
      />
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
      <select
        value={tierMax ?? ''}
        onChange={(e) => setTierMax(e.target.value ? Number(e.target.value) : undefined)}
        className="px-2.5 py-1.5 text-xs bg-zinc-800 border border-zinc-700 rounded-md text-zinc-400 focus:outline-none focus:border-amber-500"
      >
        <option value="">All Tiers</option>
        {[1, 2, 3, 4, 5, 6, 7].map((t) => (
          <option key={t} value={t}>Tier &le; {t}</option>
        ))}
      </select>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Draft Board tab (read-only)
// ---------------------------------------------------------------------------

function DraftBoardTab({ players, isLoading, isError }: { players: FantasyPlayer[]; isLoading: boolean; isError: boolean }) {
  if (isLoading) {
    return (
      <div className="overflow-x-auto rounded-lg border border-zinc-800 animate-pulse">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-800 bg-zinc-900/60">
              {[40, 160, 60, 70, 80, 80, 80].map((w, i) => (
                <th key={i} className="px-3 py-3">
                  <div className="h-3 bg-zinc-800 rounded" style={{ width: w }} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/60">
            {Array.from({ length: 12 }).map((_, i) => (
              <tr key={i}>
                {[40, 160, 60, 70, 80, 80, 80].map((w, j) => (
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
  if (isError) {
    return <div className="flex items-center justify-center h-48 text-rose-400">Failed to load draft board. Check API connection.</div>
  }
  return (
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
            <tr key={player.id} className={cn(
              'transition-colors',
              player.avoid ? 'bg-rose-950/20 hover:bg-rose-950/30' : 'hover:bg-zinc-800/40',
            )}>
              <td className="px-3 py-2.5 text-zinc-500 font-mono text-xs">{player.rank}</td>
              <td className="px-3 py-2.5">{tierBadge(player.tier)}</td>
              <td className="px-3 py-2.5">
                <div className="flex items-center gap-1.5 flex-wrap">
                  <span className={cn('font-medium', player.avoid ? 'text-rose-300' : 'text-zinc-100')}>{player.name}</span>
                  {player.is_keeper && (
                    <span className="px-1 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 border border-emerald-500/40 text-emerald-400">K·R{player.keeper_round}</span>
                  )}
                  {player.avoid && (
                    <span className="px-1 py-0.5 rounded text-[10px] font-bold bg-rose-500/20 border border-rose-500/40 text-rose-400">AVOID</span>
                  )}
                  {!player.avoid && player.injury_risk && player.injury_risk !== 'low' && (
                    <span className={cn('px-1 py-0.5 rounded text-[10px] font-bold border', player.injury_risk === 'high' ? 'bg-orange-500/20 border-orange-500/40 text-orange-400' : 'bg-yellow-500/20 border-yellow-500/40 text-yellow-400')} title={player.injury_note ?? ''}>
                      {player.injury_risk === 'high' ? '⚠ HIGH RISK' : '⚠ MED RISK'}
                    </span>
                  )}
                </div>
                <div className="text-xs text-zinc-500 capitalize">{player.type}</div>
              </td>
              <td className="px-3 py-2.5 text-zinc-400 font-mono text-xs">{player.team}</td>
              <td className="px-3 py-2.5">{positionBadges(player.positions)}</td>
              <td className="px-3 py-2.5 text-right text-zinc-400 font-mono text-xs">{player.adp.toFixed(0)}</td>
              <td className={cn('px-3 py-2.5 text-right font-mono text-xs font-semibold', zScoreColor(player.z_score))}>
                {player.z_score >= 0 ? '+' : ''}{player.z_score.toFixed(2)}
              </td>
              <td className="px-3 py-2.5 text-zinc-400 text-xs font-mono">{projSummary(player)}</td>
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
  )
}

// ---------------------------------------------------------------------------
// My Roster panel
// ---------------------------------------------------------------------------

function MyRosterPanel({ myPicks }: { myPicks: DraftPick[] }) {
  const grouped: Record<string, DraftPick[]> = {}
  for (const pick of myPicks) {
    const pos = (pick.player_positions ?? ['?'])[0]
    if (!grouped[pos]) grouped[pos] = []
    grouped[pos].push(pick)
  }

  if (myPicks.length === 0) {
    return (
      <div className="text-center text-zinc-600 text-sm py-6">
        No picks yet. Your drafted players will appear here.
      </div>
    )
  }

  return (
    <div className="space-y-1">
      {myPicks.map((pick) => (
        <div
          key={pick.player_id}
          className="flex items-center justify-between px-3 py-2 rounded-md bg-zinc-800/50 border border-zinc-700/50 text-sm"
        >
          <div className="flex items-center gap-2">
            {pick.pick_number === 0 ? (
              <span className="px-1 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 border border-emerald-500/40 text-emerald-400">K·R{pick.round}</span>
            ) : (
              <span className="text-xs text-zinc-600 font-mono w-6">{pick.round}.{pick.pick_number}</span>
            )}
            <span className="font-medium text-zinc-200">{pick.player_name}</span>
          </div>
          <div className="flex items-center gap-2">
            {pick.player_positions && (
              <span className="text-xs text-zinc-500">{pick.player_positions.slice(0, 2).join('/')}</span>
            )}
            {pick.player_tier != null && tierBadge(pick.player_tier)}
          </div>
        </div>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Draft Session tab
// ---------------------------------------------------------------------------

interface DraftSessionTabProps {
  session: DraftSession | null | undefined
  sessionLoading: boolean
  availablePlayers: FantasyPlayer[]
  draftedIds: Set<string>
  recommendations: FantasyPlayer[]
  isPickPending: boolean
  isCurrentlyMyTurn: boolean
  myTurnIn: number | null
  currentDrafter: number | null
  draftPosition: number
  setDraftPosition: (v: number) => void
  numTeams: number
  setNumTeams: (v: number) => void
  numRounds: number
  setNumRounds: (v: number) => void
  isCreating: boolean
  onCreateSession: () => void
  onDraftMe: (playerId: string) => void
  onMarkDrafted: (playerId: string) => void
  sessionKey: string | null
  onClearSession: () => void
  sessionError: boolean
}

function DraftSessionTab({
  session,
  sessionLoading,
  availablePlayers,
  draftedIds,
  recommendations,
  isPickPending,
  isCurrentlyMyTurn,
  myTurnIn,
  currentDrafter,
  draftPosition,
  setDraftPosition,
  numTeams,
  setNumTeams,
  numRounds,
  setNumRounds,
  isCreating,
  onCreateSession,
  onDraftMe,
  onMarkDrafted,
  sessionKey,
  onClearSession,
  sessionError,
}: DraftSessionTabProps) {
  // No session — show setup form
  if (!sessionKey) {
    return (
      <Card className="max-w-md mx-auto">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5 text-amber-400" />
            Start Draft Session
          </CardTitle>
        </CardHeader>
        <div className="px-6 pb-6 space-y-4">
          <p className="text-sm text-zinc-500">
            Configure your Treemendous league settings to track picks in real time.
          </p>
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-zinc-400 mb-1.5">My Draft Position</label>
              <input
                type="number"
                min={1}
                max={20}
                value={draftPosition}
                onChange={(e) => setDraftPosition(Number(e.target.value))}
                className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-md text-zinc-200 text-sm focus:outline-none focus:border-amber-500"
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-zinc-400 mb-1.5">Teams</label>
                <input
                  type="number"
                  min={4}
                  max={20}
                  value={numTeams}
                  onChange={(e) => setNumTeams(Number(e.target.value))}
                  className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-md text-zinc-200 text-sm focus:outline-none focus:border-amber-500"
                />
              </div>
              <div>
                <label className="block text-xs text-zinc-400 mb-1.5">Rounds</label>
                <input
                  type="number"
                  min={10}
                  max={30}
                  value={numRounds}
                  onChange={(e) => setNumRounds(Number(e.target.value))}
                  className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-md text-zinc-200 text-sm focus:outline-none focus:border-amber-500"
                />
              </div>
            </div>
          </div>
          <button
            onClick={onCreateSession}
            disabled={isCreating}
            className="w-full py-2.5 bg-amber-500 hover:bg-amber-400 disabled:opacity-50 text-zinc-950 font-semibold text-sm rounded-md transition-colors"
          >
            {isCreating ? 'Creating...' : 'Start Draft'}
          </button>
          <p className="text-xs text-zinc-600 text-center">
            Treemendous: R1-R2 linear, R3+ snake. Draft pos {draftPosition} of {numTeams}.
          </p>
        </div>
      </Card>
    )
  }

  // Session exists — show full draft assistant
  if (sessionLoading) {
    return <div className="flex items-center justify-center h-48 text-zinc-500">Loading session...</div>
  }

  if (sessionError || !session) {
    return (
      <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-4 text-rose-400 text-sm flex items-center justify-between">
        <span>Failed to load draft session.</span>
        <button onClick={onClearSession} className="text-xs underline hover:no-underline">
          Clear session
        </button>
      </div>
    )
  }

  const totalPicks = session.num_teams * session.num_rounds
  const currentRound = Math.floor((session.current_pick - 1) / session.num_teams) + 1

  return (
    <div className="space-y-4">
      {/* Status bar */}
      <div className={cn(
        'rounded-lg border p-4',
        isCurrentlyMyTurn
          ? 'border-amber-500/50 bg-amber-500/10'
          : 'border-zinc-700 bg-zinc-800/40',
      )}>
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap gap-6 text-sm">
            <div>
              <span className="text-zinc-500 text-xs block">Overall Pick</span>
              <span className="font-mono font-bold text-zinc-100 text-lg">
                {session.current_pick}
                <span className="text-zinc-500 text-sm font-normal"> / {totalPicks}</span>
              </span>
            </div>
            <div>
              <span className="text-zinc-500 text-xs block">Round</span>
              <span className="font-mono font-bold text-zinc-100 text-lg">{currentRound}</span>
            </div>
            <div>
              <span className="text-zinc-500 text-xs block">Now Drafting</span>
              <span className={cn(
                'font-mono font-bold text-lg',
                currentDrafter === session.my_draft_position ? 'text-amber-400' : 'text-zinc-200',
              )}>
                Pick #{currentDrafter}
              </span>
            </div>
            <div>
              <span className="text-zinc-500 text-xs block">My Turn In</span>
              <span className={cn(
                'font-mono font-bold text-lg',
                isCurrentlyMyTurn ? 'text-amber-400' : 'text-zinc-400',
              )}>
                {isCurrentlyMyTurn ? 'NOW' : myTurnIn != null ? `${myTurnIn} picks` : '—'}
              </span>
            </div>
            <div>
              <span className="text-zinc-500 text-xs block">My Picks</span>
              <span className="font-mono font-bold text-zinc-100 text-lg">{session.my_picks_count}</span>
            </div>
          </div>
          <button
            onClick={onClearSession}
            className="text-xs text-zinc-600 hover:text-zinc-400 transition-colors"
          >
            End session
          </button>
        </div>
      </div>

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <Card className="p-0">
          <CardHeader className="px-4 pt-4 pb-0 mb-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Star className="h-4 w-4 text-amber-400" />
              Recommended for Your Next Pick
            </CardTitle>
          </CardHeader>
          <div className="px-4 pb-4 grid grid-cols-1 sm:grid-cols-5 gap-2">
            {recommendations.slice(0, 5).map((p) => (
              <button
                key={p.id}
                onClick={() => onDraftMe(p.id)}
                disabled={isPickPending}
                className="text-left rounded-md border border-amber-500/30 bg-amber-500/5 hover:bg-amber-500/10 transition-colors p-2.5 disabled:opacity-50"
              >
                <div className="text-xs font-semibold text-zinc-200 leading-tight">{p.name}</div>
                <div className="text-xs text-zinc-500 mt-0.5 font-mono">
                  {p.positions.slice(0, 2).join('/')} · {p.z_score >= 0 ? '+' : ''}{p.z_score.toFixed(1)}z
                </div>
              </button>
            ))}
          </div>
        </Card>
      )}

      {/* Main two-column layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

        {/* Available players */}
        <div className="lg:col-span-2 space-y-2">
          <h3 className="text-sm font-semibold text-zinc-300 flex items-center gap-2">
            <Users className="h-4 w-4 text-zinc-500" />
            Available ({availablePlayers.length})
          </h3>
          <div className="overflow-x-auto rounded-lg border border-zinc-800 max-h-[600px] overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 z-10">
                <tr className="border-b border-zinc-800 bg-zinc-900">
                  <th className="px-3 py-2 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider">Player</th>
                  <th className="px-3 py-2 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider">Pos</th>
                  <th className="px-3 py-2 text-right text-xs font-semibold text-zinc-500 uppercase tracking-wider">Z</th>
                  <th className="px-3 py-2 text-right text-xs font-semibold text-zinc-500 uppercase tracking-wider w-36">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800/60">
                {availablePlayers.map((player) => (
                  <tr
                    key={player.id}
                    className={cn(
                      'transition-colors',
                      isCurrentlyMyTurn && player.tier <= 3
                        ? 'bg-amber-500/3 hover:bg-amber-500/8'
                        : 'hover:bg-zinc-800/40',
                    )}
                  >
                    <td className="px-3 py-2">
                      <div className="flex items-center gap-2">
                        {tierBadge(player.tier)}
                        <div>
                          <div className="flex items-center gap-1.5">
                            <span className="font-medium text-zinc-100 text-xs">{player.name}</span>
                            {player.is_keeper && (
                              <span className="px-1 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 border border-emerald-500/40 text-emerald-400">K</span>
                            )}
                          </div>
                          <div className="text-xs text-zinc-600">{player.team}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-3 py-2">{positionBadges(player.positions)}</td>
                    <td className={cn('px-3 py-2 text-right font-mono text-xs font-semibold', zScoreColor(player.z_score))}>
                      {player.z_score >= 0 ? '+' : ''}{player.z_score.toFixed(2)}
                    </td>
                    <td className="px-3 py-2">
                      <div className="flex gap-1.5 justify-end">
                        <button
                          onClick={() => onDraftMe(player.id)}
                          disabled={isPickPending}
                          className="px-2 py-1 text-xs bg-amber-500/20 hover:bg-amber-500/30 text-amber-400 rounded border border-amber-500/30 transition-colors disabled:opacity-40 min-h-[28px]"
                        >
                          Mine
                        </button>
                        <button
                          onClick={() => onMarkDrafted(player.id)}
                          disabled={isPickPending}
                          className="px-2 py-1 text-xs bg-zinc-700/60 hover:bg-zinc-700 text-zinc-400 rounded border border-zinc-600/50 transition-colors disabled:opacity-40 min-h-[28px]"
                        >
                          Taken
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
                {availablePlayers.length === 0 && (
                  <tr>
                    <td colSpan={4} className="px-3 py-6 text-center text-zinc-600 text-xs">
                      All filtered players have been drafted.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* My Roster */}
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-zinc-300 flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-emerald-500" />
            My Roster ({session.my_picks_count} / {session.num_rounds})
          </h3>
          <Card className="p-0">
            <div className="p-3 max-h-[600px] overflow-y-auto">
              <MyRosterPanel myPicks={session.my_picks} />
            </div>
          </Card>
          {session.my_picks_count > 0 && (
            <div className="text-xs text-zinc-600 text-center">
              {session.num_rounds - session.my_picks_count} picks remaining
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function FantasyPage() {
  const { data: featureFlags } = useQuery({
    queryKey: ['feature-flags'],
    queryFn: endpoints.featureFlags,
    staleTime: 5 * 60_000,
  })
  const draftBoardEnabled = featureFlags?.draft_board_enabled ?? true

  const [activeTab, setActiveTab] = useState<'board' | 'session'>('board')

  // Shared filter state
  const [posFilter, setPosFilter] = useState('All')
  const [typeFilter, setTypeFilter] = useState('')
  const [tierMax, setTierMax] = useState<number | undefined>(undefined)
  const [search, setSearch] = useState('')

  // Session state
  const [sessionKey, setSessionKey] = useState<string | null>(null)
  const [draftPosition, setDraftPosition] = useState(7)
  const [numTeams, setNumTeams] = useState(12)
  const [numRounds, setNumRounds] = useState(23)
  const [recommendations, setRecommendations] = useState<FantasyPlayer[]>([])

  const queryClient = useQueryClient()

  // Restore session key from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem(SESSION_STORAGE_KEY)
    if (stored) {
      setSessionKey(stored)
      setActiveTab('session')
    }
  }, [])

  // Draft board query
  const boardQueryParams = {
    position: posFilter !== 'All' ? posFilter : undefined,
    player_type: typeFilter || undefined,
    tier_max: tierMax,
    limit: 300,
  }
  const { data: boardData, isLoading: boardLoading, isError: boardError, refetch, isFetching } = useQuery({
    queryKey: ['fantasy-draft-board', boardQueryParams],
    queryFn: () => endpoints.fantasyDraftBoard(boardQueryParams),
    staleTime: 5 * 60_000,
  })

  // Session query
  const { data: session, isLoading: sessionLoading, isError: sessionError } = useQuery({
    queryKey: ['draft-session', sessionKey],
    queryFn: () => endpoints.fantasyGetSession(sessionKey!),
    enabled: !!sessionKey,
    refetchInterval: 30_000,
  })

  // Create session mutation
  const createMutation = useMutation({
    mutationFn: () =>
      endpoints.fantasyCreateSession({ my_draft_position: draftPosition, num_teams: numTeams, num_rounds: numRounds }),
    onSuccess: (data) => {
      setSessionKey(data.session_key)
      localStorage.setItem(SESSION_STORAGE_KEY, data.session_key)
      queryClient.invalidateQueries({ queryKey: ['draft-session', data.session_key] })
    },
  })

  // Record pick mutation
  const pickMutation = useMutation({
    mutationFn: ({ playerId, drafterPosition, isMyPick }: { playerId: string; drafterPosition: number; isMyPick: boolean }) =>
      endpoints.fantasyRecordPick(sessionKey!, { player_id: playerId, drafter_position: drafterPosition, is_my_pick: isMyPick }),
    onSuccess: (data) => {
      setRecommendations(data.next_recommendations)
      queryClient.invalidateQueries({ queryKey: ['draft-session', sessionKey] })
    },
  })

  // Derived session values
  const currentDrafter = session ? getCurrentDrafter(session.current_pick, session.num_teams) : null
  const myTurnIn = session ? picksUntilMyTurn(session.current_pick, session.my_draft_position, session.num_teams) : null
  const isCurrentlyMyTurn = myTurnIn === 0

  // Drafted player IDs
  const draftedIds = new Set(session?.picks.map((p) => p.player_id) ?? [])

  // All board players filtered
  const allPlayers = (boardData?.players ?? []).filter((p) =>
    (posFilter === 'All' || p.positions.some((pos) => pos === posFilter || (posFilter === 'OF' && ['LF', 'CF', 'RF'].includes(pos)))) &&
    (!typeFilter || p.type === typeFilter) &&
    (!tierMax || p.tier <= tierMax) &&
    (search === '' || p.name.toLowerCase().includes(search.toLowerCase()) || p.team.toLowerCase().includes(search.toLowerCase()))
  )

  // Available = not yet drafted
  const availablePlayers = allPlayers.filter((p) => !draftedIds.has(p.id))

  function handleDraftMe(playerId: string) {
    if (!session) return
    pickMutation.mutate({ playerId, drafterPosition: session.my_draft_position, isMyPick: true })
  }

  function handleMarkDrafted(playerId: string) {
    if (!session) return
    const drafter = getCurrentDrafter(session.current_pick, session.num_teams)
    pickMutation.mutate({ playerId, drafterPosition: drafter, isMyPick: false })
  }

  async function handleClearSession() {
    if (sessionKey) {
      endpoints.fantasyDeleteSession(sessionKey).catch(() => {})
    }
    setSessionKey(null)
    setRecommendations([])
    localStorage.removeItem(SESSION_STORAGE_KEY)
    queryClient.removeQueries({ queryKey: ['draft-session'] })
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-zinc-100 flex items-center gap-2">
            <Trophy className="h-6 w-6 text-amber-400" />
            Fantasy Baseball
          </h1>
          <p className="text-sm text-zinc-500 mt-1">
            2026 Steamer/ZiPS consensus · {boardData?.count ?? '—'} players ·{' '}
            <span className="text-zinc-600">2026 season active</span>
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* Tab switcher — only shown during draft season */}
          {draftBoardEnabled && (
            <div className="flex rounded-md overflow-hidden border border-zinc-700">
              <button
                onClick={() => setActiveTab('board')}
                className={cn(
                  'px-4 py-2 text-xs font-medium transition-colors flex items-center gap-1.5',
                  activeTab === 'board'
                    ? 'bg-zinc-700 text-zinc-100'
                    : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200',
                )}
              >
                <Filter className="h-3.5 w-3.5" />
                Draft Board
              </button>
              <button
                onClick={() => setActiveTab('session')}
                className={cn(
                  'px-4 py-2 text-xs font-medium transition-colors flex items-center gap-1.5',
                  activeTab === 'session'
                    ? 'bg-amber-500/20 text-amber-400'
                    : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200',
                )}
              >
                <Clock className="h-3.5 w-3.5" />
                Live Draft
                {sessionKey && (
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-400 ml-0.5" />
                )}
              </button>
            </div>
          )}
          <button
            onClick={() => refetch()}
            disabled={isFetching}
            className="flex items-center gap-2 px-3 py-2 text-sm text-zinc-400 hover:text-zinc-200 bg-zinc-800 hover:bg-zinc-700 rounded-md transition-colors disabled:opacity-50"
          >
            <RefreshCw className={cn('h-4 w-4', isFetching && 'animate-spin')} />
          </button>
        </div>
      </div>

      {/* Draft board UI — hidden when draft season is off */}
      {draftBoardEnabled && (
        <>
          <FilterBar
            posFilter={posFilter}
            setPosFilter={setPosFilter}
            typeFilter={typeFilter}
            setTypeFilter={setTypeFilter}
            tierMax={tierMax}
            setTierMax={setTierMax}
            search={search}
            setSearch={setSearch}
          />

          {activeTab === 'board' ? (
            <DraftBoardTab
              players={allPlayers}
              isLoading={boardLoading}
              isError={boardError}
            />
          ) : (
            <DraftSessionTab
          session={session}
          sessionLoading={sessionLoading}
          availablePlayers={availablePlayers}
          draftedIds={draftedIds}
          recommendations={recommendations}
          isPickPending={pickMutation.isPending}
          isCurrentlyMyTurn={isCurrentlyMyTurn}
          myTurnIn={myTurnIn}
          currentDrafter={currentDrafter}
          draftPosition={draftPosition}
          setDraftPosition={setDraftPosition}
          numTeams={numTeams}
          setNumTeams={setNumTeams}
          numRounds={numRounds}
          setNumRounds={setNumRounds}
          isCreating={createMutation.isPending}
          onCreateSession={() => createMutation.mutate()}
          onDraftMe={handleDraftMe}
          onMarkDrafted={handleMarkDrafted}
              sessionKey={sessionKey}
              onClearSession={handleClearSession}
              sessionError={sessionError}
            />
          )}
        </>
      )}
    </div>
  )
}
