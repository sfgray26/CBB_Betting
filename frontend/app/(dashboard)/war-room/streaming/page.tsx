'use client'

import { useQuery } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import { Loader2, AlertCircle, Zap, TrendingUp, TrendingDown } from 'lucide-react'
import type { WaiverAvailablePlayer, CategoryDeficit } from '@/lib/types'
import { CATEGORY_LABEL } from '@/lib/types'

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

  return (
    <div className="min-h-screen bg-black p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold tracking-widest uppercase text-[#FFC000]">
          STREAMING STATION
        </h1>
        {faab_balance != null && (
          <span className="text-xs font-semibold tracking-widest text-[#7D7D7D] uppercase">
            FAAB ${faab_balance.toFixed(0)} remaining
          </span>
        )}
      </div>

      {/* Category deficits */}
      {category_deficits?.length > 0 && (
        <div>
          <p className="text-[10px] font-semibold tracking-widest uppercase text-[#494949] mb-2">
            Category Deficits
            <span className="text-[#7D7D7D] font-normal ml-2">
              (vs opponent this week)
            </span>
          </p>
          <div className="flex flex-wrap gap-2">
            {category_deficits.map((d: CategoryDeficit) => {
              const isAhead = d.winning
              const scoreColor = isAhead ? 'text-emerald-400' : 'text-rose-400'
              const ArrowIcon = isAhead ? TrendingUp : TrendingDown
              const deficitVal = d.deficit ?? 0
              const sign = deficitVal > 0 ? '+' : ''
              const label = CATEGORY_LABEL[d.category as keyof typeof CATEGORY_LABEL] ?? d.category

              return (
                <span
                  key={d.category}
                  className="px-2 py-1 bg-[#1A1A1A] border border-[#303030] text-xs font-mono flex items-center gap-1.5"
                >
                  <span className="text-[#7D7D7D]">{label}</span>
                  <span className={scoreColor}>
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
          <p className="text-[10px] font-semibold tracking-widest uppercase text-[#494949] mb-2 flex items-center gap-1.5">
            <Zap className="h-3 w-3 text-[#FFC000]" />
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
          <p className="text-[10px] font-semibold tracking-widest uppercase text-[#494949] mb-2">
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
        <p className="text-[#7D7D7D] text-sm">No waiver targets found for the current period.</p>
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
      className={`px-3 py-2.5 ${
        highlight ? 'bg-[#1A1A0A] border border-[#3A3A00]' : 'bg-[#181818]'
      } ${hitsCats.length > 0 ? 'border-l-2 border-l-amber-500' : ''}`}
    >
      {/* Row 1: identity + need score */}
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0 flex items-center gap-2">
          <span className="text-white text-sm font-medium truncate">{player.name}</span>
          <span className="text-[#7D7D7D] text-xs">{player.team}</span>
          <span className="text-[#494949] text-xs">{positions.join('/')}</span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {player.percent_owned != null && (
            <span className="text-[#494949] text-xs">{player.percent_owned.toFixed(0)}%</span>
          )}
          <div className="text-right">
            <span className="text-[8px] text-[#494949] uppercase tracking-wider block">Need</span>
            <span className="text-[#FFC000] text-xs font-mono font-bold">
              {player.need_score != null ? player.need_score.toFixed(1) : '—'}
            </span>
          </div>
        </div>
      </div>
      {/* Row 2: two-start matchup info */}
      {player.two_start && (
        <p className="text-[10px] text-emerald-400 mt-1 font-semibold">
          ⚡ 2-START
          {player.start1_opp ? ` · vs ${player.start1_opp}` : ''}
          {player.start2_opp ? `, ${player.start2_opp}` : ''}
        </p>
      )}
      {/* Row 3: category match badges */}
      {catMatches.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-1.5">
          {catMatches.map(c => {
            const label = CATEGORY_LABEL[c as keyof typeof CATEGORY_LABEL] ?? c
            const isBehind = behindCats.includes(c)
            return (
              <span
                key={c}
                className={`text-[9px] px-1.5 py-0.5 rounded font-semibold ${
                  isBehind
                    ? 'bg-amber-900/30 text-amber-400 border border-amber-700/40'
                    : 'bg-[#2A2A2A] text-[#7D7D7D]'
                }`}
              >
                {label}
              </span>
            )
          })}
        </div>
      )}
    </div>
  )
}
