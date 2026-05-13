'use client'

import { useQuery } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import { Loader2, AlertCircle, Zap, TrendingUp, TrendingDown } from 'lucide-react'
import type { WaiverAvailablePlayer, CategoryDeficit } from '@/lib/types'
import { CATEGORY_LABEL, CATEGORY_COLOR, RotoCategory } from '@/lib/types'

export default function StreamingStationPage() {
  const waiver = useQuery({
    queryKey: ['waiver'],
    queryFn: () => endpoints.getWaiver(),
    staleTime: 5 * 60_000,
  })

  if (waiver.isLoading) {
    return (
      <div className="min-h-screen bg-black p-6">
        <h1 className="text-xl font-bold tracking-widest uppercase text-[#FFC000] mb-6">
          STREAMING STATION
        </h1>
        <div className="flex items-center gap-2 text-[#7D7D7D]">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span className="text-sm">Loading waiver data...</span>
        </div>
      </div>
    )
  }

  if (waiver.isError) {
    return (
      <div className="min-h-screen bg-black p-6">
        <h1 className="text-xl font-bold tracking-widest uppercase text-[#FFC000] mb-6">
          STREAMING STATION
        </h1>
        <div className="flex items-center gap-2 text-rose-400">
          <AlertCircle className="h-4 w-4" />
          <span className="text-sm">{waiver.error?.message ?? 'Failed to load waiver data'}</span>
        </div>
      </div>
    )
  }

  if (!waiver.data) {
    return (
      <div className="min-h-screen bg-black p-6">
        <h1 className="text-xl font-bold tracking-widest uppercase text-[#FFC000] mb-6">
          STREAMING STATION
        </h1>
        <p className="text-[#7D7D7D] text-sm">No waiver data available.</p>
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

  return (
    <div className="min-h-screen bg-bg-base p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold tracking-widest uppercase text-accent-gold">
          STREAMING STATION
        </h1>
        {faab_balance != null && (
          <span className="text-xs font-semibold tracking-widest text-[#7D7D7D] uppercase">
            FAAB ${faab_balance.toFixed(0)} remaining
          </span>
        )}
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
      {two_start_pitchers?.length > 0 && (
        <div>
          <p className="text-[10px] font-semibold tracking-widest uppercase text-text-muted mb-2 flex items-center gap-1.5">
            <Zap className="h-3 w-3 text-accent-gold" />
            Two-Start Pitchers ({two_start_pitchers.length})
          </p>
          <div className="space-y-2">
            {two_start_pitchers.map((p: WaiverAvailablePlayer) => (
              <WaiverPlayerRow key={p.player_id} player={p} highlight behindCats={behindCats} />
            ))}
          </div>
        </div>
      )}

      {/* Top available */}
      {top_available?.length > 0 && (
        <div>
          <p className="text-[10px] font-semibold tracking-widest uppercase text-text-muted mb-2">
            Top Available ({top_available.length})
          </p>
          <div className="space-y-1">
            {top_available.map((p: WaiverAvailablePlayer) => (
              <WaiverPlayerRow key={p.player_id} player={p} behindCats={behindCats} />
            ))}
          </div>
        </div>
      )}

      {top_available?.length === 0 && two_start_pitchers?.length === 0 && (
        <p className="text-text-secondary text-sm">No waiver targets found for the current period.</p>
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
          {player.percent_owned != null && player.percent_owned > 0 && (
            <span className="text-text-muted text-xs">{player.percent_owned.toFixed(0)}%</span>
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
