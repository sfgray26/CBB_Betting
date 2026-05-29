'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import { Loader2, AlertCircle, Zap, TrendingUp, TrendingDown } from 'lucide-react'
import type { WaiverAvailablePlayer, CategoryDeficit } from '@/lib/types'
import { CATEGORY_LABEL, CATEGORY_COLOR } from '@/lib/types'

export default function StreamingStationPage() {
  const [hideOwned, setHideOwned] = useState(true)
  const [minNeedScore, setMinNeedScore] = useState(0.0)
  const [showTwoStartOnly, setShowTwoStartOnly] = useState(false)

  const waiver = useQuery({
    queryKey: ['waiver'],
    queryFn: () => endpoints.getWaiver(),
    staleTime: 5 * 60_000,
    retry: 1,
    retryDelay: 2000,
  })

  if (waiver.isLoading) {
    return (
      <div className="min-h-screen bg-bg-base p-6">
        <h1 className="text-xl font-bold tracking-widest uppercase text-accent-gold mb-6">
          STREAMING STATION
        </h1>
        <div className="flex items-center gap-2 text-text-secondary">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span className="text-sm">Loading waiver data...</span>
        </div>
      </div>
    )
  }

  if (waiver.isError) {
    return (
      <div className="min-h-screen bg-bg-base p-6">
        <h1 className="text-xl font-bold tracking-widest uppercase text-accent-gold mb-6">
          STREAMING STATION
        </h1>
        <div className="flex items-center gap-2 text-status-lost">
          <AlertCircle className="h-4 w-4" />
          <span className="text-sm">{waiver.error?.message ?? 'Failed to load waiver data'}</span>
        </div>
      </div>
    )
  }

  if (!waiver.data) {
    return (
      <div className="min-h-screen bg-bg-base p-6">
        <h1 className="text-xl font-bold tracking-widest uppercase text-accent-gold mb-6">
          STREAMING STATION
        </h1>
        <p className="text-text-secondary text-sm">No waiver data available.</p>
      </div>
    )
  }

  const { top_available, two_start_pitchers, category_deficits, faab_balance } = waiver.data

  const behindCats = (category_deficits ?? [])
    .filter((d: CategoryDeficit) => !d.winning)
    .map((d: CategoryDeficit) => d.category)

  const sortedDeficits = [...(category_deficits ?? [])].sort(
    (a, b) => Math.abs(b.deficit ?? 0) - Math.abs(a.deficit ?? 0)
  )

  function passesFilters(p: WaiverAvailablePlayer): boolean {
    if (hideOwned && (p.percent_owned == null && p.owned_pct == null)) return false
    if (p.need_score != null && p.need_score <= minNeedScore) return false
    if (showTwoStartOnly && !p.two_start) return false
    return true
  }

  function deficitWeightedScore(
    player: WaiverAvailablePlayer,
    deficits: CategoryDeficit[]
  ): number {
    const contribs = player.category_contributions ?? {}
    const losingDeficits = deficits.filter(d => !d.winning)
    if (losingDeficits.length === 0) return player.need_score ?? 0

    let score = 0
    for (const d of losingDeficits) {
      const contrib = contribs[d.category] ?? 0
      // Weight by how far behind we are (larger deficit = more important)
      const weight = Math.min(3.0, 1.0 + Math.abs(d.deficit ?? 0) * 0.3)
      score += contrib * weight
    }
    // Blend 70% deficit-weighted + 30% season need_score to preserve overall quality
    return score * 0.7 + (player.need_score ?? 0) * 0.3
  }

  const filteredTwoStarters = (two_start_pitchers ?? [])
    .filter(passesFilters)
    .sort((a, b) =>
      deficitWeightedScore(b, category_deficits ?? []) -
      deficitWeightedScore(a, category_deficits ?? [])
    )
  const filteredTopAvailable = (top_available ?? [])
    .filter(passesFilters)
    .sort((a, b) =>
      deficitWeightedScore(b, category_deficits ?? []) -
      deficitWeightedScore(a, category_deficits ?? [])
    )

  return (
    <div className="min-h-screen bg-bg-base p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold tracking-widest uppercase text-accent-gold">
          STREAMING STATION
        </h1>
        {faab_balance != null && (
          <span className="text-xs font-semibold tracking-widest text-text-secondary uppercase">
            FAAB ${faab_balance.toFixed(0)} remaining
          </span>
        )}
      </div>

      {/* Filters */}
      <div className="bg-bg-surface border border-border-subtle rounded-lg p-4 space-y-4">
        <p className="text-[10px] font-semibold tracking-widest uppercase text-text-muted">
          Filters
        </p>
        <div className="flex flex-col sm:flex-row sm:items-center gap-4">
          {/* Hide Owned Players */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={hideOwned}
              onChange={(e) => setHideOwned(e.target.checked)}
              className="h-4 w-4 rounded border-border-default text-accent-primary focus:ring-accent-primary"
            />
            <span className="text-xs text-text-secondary">Hide Owned Players</span>
          </label>

          {/* 2-Start Only */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showTwoStartOnly}
              onChange={(e) => setShowTwoStartOnly(e.target.checked)}
              className="h-4 w-4 rounded border-border-default text-accent-primary focus:ring-accent-primary"
            />
            <span className="text-xs text-text-secondary">2-Start SPs Only</span>
          </label>

          {/* Minimum Need Score */}
          <div className="flex items-center gap-3 flex-1 max-w-xs">
            <span className="text-xs text-text-secondary whitespace-nowrap">Min Need Score</span>
            <input
              type="range"
              min="-5"
              max="10"
              step="0.1"
              value={minNeedScore}
              onChange={(e) => setMinNeedScore(parseFloat(e.target.value))}
              className="flex-1 h-1.5 bg-bg-inset rounded-lg appearance-none cursor-pointer accent-accent-primary"
            />
            <span className="text-xs font-mono font-bold text-accent-gold w-10 text-right">
              {minNeedScore.toFixed(1)}
            </span>
          </div>
        </div>
      </div>

      {/* Category deficits — sorted by magnitude, severity-colored */}
      {sortedDeficits.length > 0 && (
        <div>
          <p className="text-[10px] font-semibold tracking-widest uppercase text-text-muted mb-2">
            Category Deficits
            <span className="text-text-tertiary font-normal ml-2">
              (vs opponent this week)
            </span>
          </p>
          <div className="flex flex-wrap gap-2">
            {sortedDeficits.map((d: CategoryDeficit) => {
              const isAhead = d.winning
              const absVal = Math.abs(d.deficit ?? 0)
              const deficitVal = d.deficit ?? 0
              const sign = deficitVal > 0 ? '+' : ''
              const label = CATEGORY_LABEL[d.category as keyof typeof CATEGORY_LABEL] ?? d.category
              const catColor = CATEGORY_COLOR[d.category as keyof typeof CATEGORY_COLOR] ?? '#6b6b8a'
              const ArrowIcon = isAhead ? TrendingUp : TrendingDown

              // Severity tint on border only — background stays neutral
              const borderColor = isAhead
                ? 'rgba(34, 197, 94, 0.4)'
                : absVal >= 3.0
                  ? 'rgba(239, 68, 68, 0.5)'
                  : absVal >= 1.0
                    ? 'rgba(245, 158, 11, 0.4)'
                    : 'rgba(58, 58, 77, 0.6)'

              const scoreColor = isAhead
                ? 'text-status-safe'
                : absVal >= 3.0
                  ? 'text-status-lost'
                  : absVal >= 1.0
                    ? 'text-status-bubble'
                    : 'text-text-tertiary'

              return (
                <span
                  key={d.category}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md bg-bg-elevated border text-xs"
                  style={{ borderColor }}
                >
                  {/* Category identity dot */}
                  <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: catColor }} />
                  <span className="text-text-secondary font-semibold">{label}</span>
                  <span className={`font-mono font-bold ${scoreColor}`}>
                    {sign}{deficitVal.toFixed(1)}
                  </span>
                  <ArrowIcon className={`h-3 w-3 ${scoreColor} shrink-0`} />
                </span>
              )
            })}
          </div>
        </div>
      )}

      {/* Two-start pitchers */}
      {filteredTwoStarters.length > 0 && (
        <div>
          <p className="text-[10px] font-semibold tracking-widest uppercase text-text-muted mb-2 flex items-center gap-1.5">
            <Zap className="h-3 w-3 text-accent-gold" />
            Two-Start Pitchers ({filteredTwoStarters.length})
            {filteredTwoStarters.length !== (two_start_pitchers ?? []).length && (
              <span className="text-text-tertiary font-normal normal-case tracking-normal">
                · {(two_start_pitchers ?? []).length - filteredTwoStarters.length} hidden
              </span>
            )}
          </p>
          <div className="space-y-2">
            {filteredTwoStarters.map((p: WaiverAvailablePlayer) => (
              <WaiverPlayerRow key={p.player_id} player={p} highlight behindCats={behindCats} />
            ))}
          </div>
        </div>
      )}

      {/* Top available */}
      {filteredTopAvailable.length > 0 && (
        <div>
          <p className="text-[10px] font-semibold tracking-widest uppercase text-text-muted mb-2">
            Top Available ({filteredTopAvailable.length})
            {filteredTopAvailable.length !== (top_available ?? []).length && (
              <span className="text-text-tertiary font-normal normal-case tracking-normal ml-1">
                · {(top_available ?? []).length - filteredTopAvailable.length} hidden
              </span>
            )}
          </p>
          <div className="space-y-1">
            {filteredTopAvailable.map((p: WaiverAvailablePlayer) => (
              <WaiverPlayerRow key={p.player_id} player={p} behindCats={behindCats} />
            ))}
          </div>
        </div>
      )}

      {filteredTopAvailable.length === 0 && filteredTwoStarters.length === 0 && (
        <p className="text-text-secondary text-sm">No waiver targets match the current filters.</p>
      )}
    </div>
  )
}

function WaiverPlayerRow({
  player,
  highlight = false,
  behindCats = [],
}: {
  player: WaiverAvailablePlayer
  highlight?: boolean
  behindCats?: string[]
}) {
  const positions = player.positions ?? (player.position ? [player.position] : [])
  const catMatches = player.category_need_match ?? []
  const hitsCats = catMatches.filter(c => behindCats.includes(c))

  return (
    <div
      className={`px-3 py-2.5 rounded-md ${
        highlight ? 'bg-bg-elevated border border-accent-gold/30' : 'bg-bg-surface'
      } ${hitsCats.length > 0 ? 'border-l-2 border-l-accent-gold' : ''}`}
    >
      {/* Row 1: identity + need score */}
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0 flex items-center gap-2">
          <span className="text-text-primary text-sm font-medium truncate">{player.name}</span>
          <span className="text-text-secondary text-xs">{player.team}</span>
          <span className="text-text-muted text-xs">{positions.join('/')}</span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {player.percent_owned != null ? (
            <span className="text-text-muted text-xs">{player.percent_owned.toFixed(0)}%</span>
          ) : (
            <span className="text-text-muted text-xs">—</span>
          )}
          {player.momentum_signal && player.momentum_signal !== 'STABLE' && (
            <span className={
              ['SURGING','HOT'].includes(player.momentum_signal)
                ? 'text-status-safe text-xs'
                : 'text-status-behind text-xs'
            }>
              {['SURGING','HOT'].includes(player.momentum_signal) ? '▲' : '▼'}
            </span>
          )}
          <div className="text-right">
            <span className="text-[8px] text-text-muted uppercase tracking-wider block">Need</span>
            <span className="text-accent-gold text-xs font-mono font-bold">
              {player.need_score != null ? player.need_score.toFixed(1) : '—'}
            </span>
          </div>
        </div>
      </div>
      {/* Row 2: two-start matchup info */}
      {player.two_start && (
        <p className="text-[10px] text-status-safe mt-1 font-semibold">
          ⚡ 2-START
          {player.start1_opp ? ` · vs ${player.start1_opp}` : ''}
          {player.start2_opp ? `, ${player.start2_opp}` : ''}
        </p>
      )}
      {/* Row 3: category match badges with color dots */}
      {catMatches.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-1.5">
          {catMatches.map(c => {
            const label = CATEGORY_LABEL[c as keyof typeof CATEGORY_LABEL] ?? c
            const catColor = CATEGORY_COLOR[c as keyof typeof CATEGORY_COLOR] ?? '#6b6b8a'
            const isBehind = behindCats.includes(c)
            return (
              <span
                key={c}
                className={`inline-flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded font-semibold ${
                  isBehind
                    ? 'bg-bg-elevated text-text-primary border border-border-default'
                    : 'bg-bg-inset text-text-tertiary'
                }`}
              >
                <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: catColor }} />
                {label}
              </span>
            )
          })}
        </div>
      )}
    </div>
  )
}
