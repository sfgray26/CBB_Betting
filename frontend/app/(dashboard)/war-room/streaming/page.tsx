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

  // Early-return for loading state — prevents concurrent-render conflicts
  // between the loading block and the data block in the same render pass.
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
              const sign = d.deficit > 0 ? '+' : ''

              return (
                <span
                  key={d.category}
                  className="px-2 py-1 bg-[#1A1A1A] border border-[#303030] text-xs font-mono flex items-center gap-1.5"
                >
                  <span className="text-[#7D7D7D]">{CATEGORY_LABEL[d.category as keyof typeof CATEGORY_LABEL] ?? d.category}</span>
                  <span className={scoreColor}>
                    {sign}{d.deficit.toFixed(1)}
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
              <WaiverPlayerRow key={p.player_id} player={p} highlight />
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
              <WaiverPlayerRow key={p.player_id} player={p} />
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
}: {
  player: WaiverAvailablePlayer
  highlight?: boolean
}) {
  return (
    <div
      className={`flex items-center justify-between px-3 py-2 ${
        highlight ? 'bg-[#1A1A0A] border border-[#3A3A00]' : 'bg-[#181818]'
      }`}
    >
      <div className="min-w-0">
        <span className="text-white text-sm font-medium truncate">{player.name}</span>
        <span className="text-[#7D7D7D] text-xs ml-2">{player.team}</span>
        <span className="text-[#494949] text-xs ml-2">{(player.positions ?? (player.position ? [player.position] : [])).join('/')}</span>
      </div>
      <div className="flex items-center gap-3 shrink-0 ml-3">
        {player.percent_owned != null && (
          <span className="text-[#7D7D7D] text-xs">{player.percent_owned.toFixed(0)}%</span>
        )}
        <span className="text-[#FFC000] text-xs font-mono">
          {player.need_score.toFixed(1)}
        </span>
      </div>
    </div>
  )
}
