'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import type { WaiverAvailablePlayer, WaiverResponse, WaiverRosterPlayer, WaiverRecommendation, DropPlayerOut, CategoryDelta } from '@/lib/types'
import {
  ListFilter, Loader2, AlertCircle, TrendingUp,
  AlertTriangle, Users, Zap, ChevronDown, ChevronUp, AlertTriangle as WarnIcon,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { ErrorBoundary } from '@/components/error-boundary'
import { HotColdBadge } from '@/components/hot-cold-badge'
import { Tooltip } from '@/components/shared/tooltip'

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

function formatContributionKey(key: string): string {
  const labels: Record<string, string> = {
    HR_B: 'HR', K_B: 'K', K_P: 'Ks', HR_P: 'HRA', K_9: 'K/9', NSB: 'NSB',
    R: 'R', H: 'H', HR: 'HR', RBI: 'RBI', TB: 'TB', AVG: 'AVG', OPS: 'OPS',
    W: 'W', L: 'L', ERA: 'ERA', WHIP: 'WHIP', QS: 'QS', SV: 'SV', NSV: 'SV',
  }
  return `${labels[key] ?? key} fit`
}

function NeedScoreTooltipContent({ score, contributions }: { score: number; contributions?: Record<string, number> }) {
  const tier = score >= 20 ? 'Premium target' : score >= 15 ? 'Strong target' : 'Standard target'
  const breakdown = contributions
    ? Object.entries(contributions).map(([k, v]) => `${formatContributionKey(k)}: ${v >= 0 ? '+' : ''}${v.toFixed(1)}`).join(' | ')
    : null

  return (
    <div className="space-y-1.5 max-w-[240px]">
      <p className="font-semibold text-text-primary">{score.toFixed(2)} — {tier}</p>
      {breakdown && <p className="text-text-secondary">{breakdown}</p>}
      <p className="text-text-muted text-[10px]">Scores range 0-30. {'>'}20 = premium target</p>
    </div>
  )
}

function NeedBar({ score, contributions }: { score: number; contributions?: Record<string, number> }) {
  const pct = Math.min(100, Math.max(0, score * 10)) // scale: 0-10 → 0-100%
  const color = score >= 7.0 ? 'bg-status-safe' : score >= 4.0 ? 'bg-status-bubble' : 'bg-text-muted'
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-bg-inset rounded-full overflow-hidden">
        <div className={cn('h-full rounded-full transition-all duration-700 ease-out', color)} style={{ width: `${pct}%` }} />
      </div>
      <Tooltip content={<NeedScoreTooltipContent score={score} contributions={contributions} />}>
        <span className="text-xs text-text-primary tabular-nums w-8 text-right cursor-help underline decoration-dotted">
          {score.toFixed(2)}
        </span>
      </Tooltip>
    </div>
  )
}

function OwnershipBadge({ pct }: { pct: number | null | undefined }) {
  if (pct === null || pct === undefined) {
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

function positionBadgeClass(pos: string): string {
  if (pos === 'SP') return 'bg-blue-50 text-blue-700'
  if (pos === 'RP' || pos === 'P') return 'bg-purple-50 text-purple-700'
  if (pos === 'OF' || pos === 'LF' || pos === 'CF' || pos === 'RF') return 'bg-emerald-50 text-emerald-700'
  if (pos === 'C') return 'bg-amber-50 text-amber-700'
  if (pos === '1B' || pos === '3B') return 'bg-orange-50 text-orange-700'
  if (pos === '2B' || pos === 'SS' || pos === 'MI') return 'bg-sky-50 text-sky-700'
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

  const tierBadge = (() => {
    const s = player.need_score
    if (s == null) return null
    if (s >= 20) return { label: 'PREMIUM', className: 'bg-accent-gold/10 text-accent-gold border border-accent-gold/30' }
    if (s >= 15) return { label: 'STRONG', className: 'bg-text-muted/10 text-text-secondary border border-text-muted/30' }
    return null
  })()

  return (
    <div className="bg-bg-surface border border-border-subtle rounded-lg p-4 flex flex-col sm:flex-row sm:items-start gap-3 hover:bg-bg-elevated transition-colors duration-150">
      {/* Identity */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <p className="text-sm font-bold text-text-primary truncate">{player.name}</p>
          {tierBadge && (
            <span className={cn('text-[10px] px-1.5 py-0.5 rounded font-semibold uppercase tracking-wider', tierBadge.className)}>
              {tierBadge.label}
            </span>
          )}
          {(player.starts_this_week ?? 0) >= 2 && (
            <span className="text-[10px] px-1.5 py-0.5 bg-status-safe/10 text-status-safe border border-status-safe/30 rounded font-semibold uppercase tracking-wider">
              2-Start
            </span>
          )}
          <HotColdBadge hotCold={player.hot_cold} rankPercentile={player.rank_percentile} />
          {player.momentum_signal && player.momentum_signal !== 'STABLE' && (
            <span className={
              ['SURGING','HOT'].includes(player.momentum_signal)
                ? 'text-status-safe text-xs'
                : 'text-status-behind text-xs'
            }>
              {['SURGING','HOT'].includes(player.momentum_signal) ? '▲' : '▼'}
            </span>
          )}
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
        {player.league_drop && (
          <p className={cn(
            'text-[10px] mt-1 font-semibold',
            player.league_drop.days_ago < 3
              ? 'text-accent-gold'
              : 'text-text-muted',
          )}>
            ⬇️ Dropped {player.league_drop.days_ago === 0 ? 'today' : `${player.league_drop.days_ago} day${player.league_drop.days_ago !== 1 ? 's' : ''} ago`} by {player.league_drop.team_name}
          </p>
        )}
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
          <div className="flex items-center justify-between mb-1">
            <p className="text-[9px] text-text-muted uppercase tracking-wider">Match Score</p>
            {player.small_sample && (
              <span className="text-[10px] px-1.5 py-0.5 bg-status-bubble/10 text-status-bubble border border-status-bubble/30 rounded font-semibold">
                ⚠️ Small Sample
              </span>
            )}
          </div>
          <NeedBar score={player.need_score} contributions={player.category_contributions} />
          <p className="text-[9px] text-text-muted mt-0.5">
            {needMatches.length > 0
              ? `Fits: ${needMatches.slice(0, 3).join(', ')}`
              : 'fit for your gaps'}
          </p>
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

// ─────────────────────────────────────────────────────────────────────────────
// ADD/DROP Recommendation Card
// ─────────────────────────────────────────────────────────────────────────────

const CAT_LABEL: Record<string, string> = {
  r: 'R', h: 'H', hr_b: 'HR', rbi: 'RBI', k_b: 'K', tb: 'TB', avg: 'AVG', ops: 'OPS', nsb: 'NSB',
  w: 'W', l: 'L', hr_p: 'HRA', k_p: 'Ks', era: 'ERA', whip: 'WHIP', k_9: 'K/9', qs: 'QS', nsv: 'SV',
}

function catLabel(key: string): string {
  return CAT_LABEL[key.toLowerCase()] ?? key.toUpperCase()
}

function NetArrow({ net }: { net: number }) {
  if (net > 0.1) return <span className="text-status-safe font-bold">▲</span>
  if (net < -0.1) return <span className="text-status-lost font-bold">▼</span>
  return <span className="text-text-muted">~</span>
}

function AddPanel({ rec }: { rec: WaiverRecommendation }) {
  const fa = rec.add_player
  if (!fa) return null
  return (
    <div className="flex-1 min-w-0 space-y-1">
      <p className="text-[10px] font-bold tracking-widest uppercase text-status-safe">ADD</p>
      <p className="text-sm font-semibold text-text-primary truncate">{fa.name}</p>
      <p className="text-[10px] text-text-muted">{fa.position} · {fa.team}</p>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[10px] text-text-secondary">
        {fa.z_score !== undefined && (
          <span>Z <span className="text-text-primary font-mono">{fa.z_score >= 0 ? '+' : ''}{fa.z_score.toFixed(2)}</span></span>
        )}
        {(fa.starts_this_week ?? 0) > 0 && (
          <span className="text-status-safe">{fa.starts_this_week}-start</span>
        )}
        {rec.statcast_signals.map((sig) => (
          <span key={sig} className="text-accent-gold">[{sig}]</span>
        ))}
      </div>
    </div>
  )
}

function DropPanel({ drop }: { drop: DropPlayerOut }) {
  const zClass = drop.z_score >= 0 ? 'text-text-primary' : 'text-status-lost'
  return (
    <div className="flex-1 min-w-0 space-y-1">
      <p className="text-[10px] font-bold tracking-widest uppercase text-status-lost">DROP</p>
      <p className="text-sm font-semibold text-text-primary truncate">{drop.name}</p>
      <p className="text-[10px] text-text-muted">{drop.position} · {drop.percent_owned.toFixed(0)}% owned</p>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[10px] text-text-secondary">
        <span>Z <span className={cn('font-mono', zClass)}>{drop.z_score >= 0 ? '+' : ''}{drop.z_score.toFixed(2)}</span></span>
        <span>T{drop.tier}</span>
        {drop.status && drop.status !== 'Active' && (
          <span className="text-status-bubble">{drop.status}</span>
        )}
      </div>
    </div>
  )
}

function CategoryNetRow({ deltas }: { deltas: Record<string, CategoryDelta> }) {
  const entries = Object.entries(deltas).sort(([a], [b]) => a.localeCompare(b))
  if (entries.length === 0) return null
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1 text-[10px]">
      {entries.map(([cat, d]) => (
        <span key={cat} className="flex items-center gap-0.5">
          <NetArrow net={d.net} />
          <span className="text-text-secondary">{catLabel(cat)}</span>
          {d.cat_win_prob !== null
            ? <span className="text-text-muted ml-0.5">{Math.round(d.cat_win_prob * 100)}%</span>
            : <span className="text-text-muted ml-0.5">—</span>
          }
        </span>
      ))}
    </div>
  )
}

function WinBar({ before, after }: { before: number; after: number }) {
  const beforePct = Math.round(before * 100)
  const afterPct = Math.round(after * 100)
  const gain = afterPct - beforePct
  return (
    <div className="flex items-center gap-2 text-[10px]">
      <span className="text-text-muted">Win%</span>
      <span className="text-text-secondary tabular-nums">{beforePct}%</span>
      <span className="text-text-muted">→</span>
      <span className={cn('font-semibold tabular-nums', afterPct > beforePct ? 'text-status-safe' : afterPct < beforePct ? 'text-status-lost' : 'text-text-secondary')}>
        {afterPct}%
      </span>
      {gain !== 0 && (
        <span className={cn('tabular-nums', gain > 0 ? 'text-status-safe' : 'text-status-lost')}>
          {gain > 0 ? '+' : ''}{gain}pp
        </span>
      )}
      <div className="flex-1 h-1.5 bg-bg-inset rounded-full overflow-hidden max-w-[80px]">
        <div
          className={cn('h-full rounded-full', afterPct > beforePct ? 'bg-status-safe' : 'bg-status-bubble')}
          style={{ width: `${afterPct}%` }}
        />
      </div>
    </div>
  )
}

function RecommendationCard({ rec }: { rec: WaiverRecommendation }) {
  const [showRationale, setShowRationale] = useState(false)
  const drop = rec.drop_player

  return (
    <div className="bg-bg-surface border border-border-subtle rounded-lg overflow-hidden">
      {/* Two-panel row */}
      <div className="p-3 flex flex-col sm:flex-row gap-3">
        {rec.add_player && <AddPanel rec={rec} />}
        {drop && (
          <>
            <div className="hidden sm:block w-px bg-border-subtle self-stretch" />
            <DropPanel drop={drop} />
          </>
        )}
      </div>

      {/* Category net row */}
      {Object.keys(rec.category_deltas).length > 0 && (
        <div className="px-3 pb-2">
          <CategoryNetRow deltas={rec.category_deltas} />
        </div>
      )}

      {/* Win probability */}
      <div className="px-3 pb-2">
        {rec.mcmc_enabled
          ? <WinBar before={rec.win_prob_before} after={rec.win_prob_after} />
          : <span className="text-[10px] text-text-muted">Win%: unavailable</span>
        }
      </div>

      {/* Positional impact warnings */}
      {drop && drop.positional_impact.length > 0 && (
        <div className="px-3 pb-2 flex flex-wrap gap-1">
          {drop.positional_impact.map((msg, i) => (
            <span key={i} className="flex items-center gap-1 text-[10px] text-status-bubble">
              <WarnIcon className="h-3 w-3 flex-shrink-0" />
              {msg}
            </span>
          ))}
        </div>
      )}

      {/* Alternative drops */}
      {rec.alternative_drops.length > 0 && (
        <div className="px-3 pb-2">
          <span className="text-[10px] text-text-muted">Alt drops: </span>
          {rec.alternative_drops.map((alt, i) => (
            <span key={alt.name} className="text-[10px] text-text-secondary">
              {i > 0 && <span className="mx-1 text-text-muted">·</span>}
              {alt.name} (Z {alt.z_score >= 0 ? '+' : ''}{alt.z_score.toFixed(1)}, T{alt.tier})
              {alt.positional_impact.length > 0 && (
                <span className="text-status-bubble ml-0.5">⚠</span>
              )}
            </span>
          ))}
        </div>
      )}

      {/* Roster context */}
      {(rec.roster_context.add_weekly_starts > 0 || rec.roster_context.drop_weekly_starts > 0) && (
        <div className="px-3 pb-2 text-[10px] text-text-muted">
          {rec.roster_context.add_weekly_starts > 0 && (
            <span className="text-status-safe">+{rec.roster_context.add_weekly_starts} starts</span>
          )}
          {rec.roster_context.add_weekly_starts > 0 && rec.roster_context.drop_weekly_starts > 0 && (
            <span className="mx-1">·</span>
          )}
          {rec.roster_context.drop_weekly_starts > 0 && (
            <span>{drop?.name ?? 'Drop'}: {rec.roster_context.drop_weekly_starts} starts</span>
          )}
        </div>
      )}

      {/* Rationale toggle */}
      <div className="px-3 pb-3">
        <button
          onClick={() => setShowRationale((v) => !v)}
          className="flex items-center gap-1 text-[10px] text-text-muted hover:text-text-secondary transition-colors"
        >
          {showRationale ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
          {showRationale ? 'Hide' : 'Rationale'}
        </button>
        {showRationale && (
          <p className="mt-1 text-[11px] text-text-secondary leading-relaxed">{rec.rationale}</p>
        )}
      </div>
    </div>
  )
}

function RecommendationsPanel() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['waiver-recommendations'],
    queryFn: async () => {
      try {
        return await endpoints.getWaiverRecommendations()
      } catch (e) {
        console.error('Waiver recommendations fetch failed:', e)
        throw e
      }
    },
    staleTime: 5 * 60_000,
    retry: 1,
  })

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-text-secondary text-xs py-2">
        <Loader2 className="h-3.5 w-3.5 animate-spin text-accent-gold" />
        Loading recommendations…
      </div>
    )
  }

  if (isError || !data) return null

  const recs = data.recommendations.filter((r) => r.action === 'ADD_DROP')
  if (recs.length === 0) return null

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Zap className="h-3.5 w-3.5 text-accent-gold" />
        <p className="text-xs font-bold tracking-widest uppercase text-accent-gold">
          ADD/DROP Recommendations · {recs.length}
        </p>
      </div>
      {recs.map((rec, i) => (
        <RecommendationCard key={i} rec={rec} />
      ))}
    </div>
  )
}

function WaiverPageInner() {
  const [sort, setSort] = useState<'need_score' | 'projected_points'>('need_score')
  const [posFilter, setPosFilter] = useState('All')

  const { data, isLoading, isError, error, refetch } = useQuery({
    queryKey: ['waiver', sort],
    queryFn: async () => {
      try {
        return await endpoints.getWaiver(sort)
      } catch (e) {
        console.error('Waiver fetch failed:', e)
        throw e
      }
    },
    staleTime: 3 * 60_000,
    retry: 1,
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

      {/* ADD/DROP Recommendations */}
      <RecommendationsPanel />

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

export default function WaiverPage() {
  return (
    <ErrorBoundary>
      <WaiverPageInner />
    </ErrorBoundary>
  )
}
