'use client'

import { useQuery } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import { Loader2, AlertCircle, Swords } from 'lucide-react'
import { MatchupHeader } from '@/components/war-room/matchup-header'
import { CategoryBattlefield } from '@/components/war-room/category-battlefield'

export default function WarRoomPage() {
  const matchup = useQuery({
    queryKey: ['matchup'],
    queryFn: endpoints.getMatchup,
    staleTime: 5 * 60_000,
    refetchInterval: 5 * 60_000,
  })

  // Simulate runs Monte Carlo — loads after current values; stale 5min, refresh 15min
  const simulate = useQuery({
    queryKey: ['matchup', 'simulate'],
    queryFn: endpoints.simulateMatchup,
    staleTime: 5 * 60_000,
    refetchInterval: 15 * 60_000,
    retry: 1,
  })

  if (matchup.isLoading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="flex items-center gap-3 text-[#7D7D7D]">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span className="text-xs tracking-widest uppercase">Loading matchup...</span>
        </div>
      </div>
    )
  }

  if (matchup.isError) {
    return (
      <div className="min-h-screen bg-black p-6">
        <div className="flex items-center gap-2 text-rose-500">
          <AlertCircle className="h-4 w-4" />
          <span className="text-sm font-mono">{matchup.error?.message ?? 'Failed to load matchup'}</span>
        </div>
      </div>
    )
  }

  if (!matchup.data) return null

  return (
    <div className="min-h-screen bg-black">
      <div className="max-w-2xl mx-auto p-6 space-y-3">
        {/* Page label */}
        <div className="flex items-center gap-2 mb-1">
          <Swords className="h-3.5 w-3.5 text-[#FFC000]" />
          <span className="text-xs font-bold tracking-widest uppercase text-[#FFC000]">War Room</span>
          {simulate.isLoading && (
            <span className="ml-auto flex items-center gap-1.5 text-[#494949] text-[10px] tracking-widest uppercase">
              <Loader2 className="h-3 w-3 animate-spin" />
              Simulating...
            </span>
          )}
        </div>

        <MatchupHeader data={matchup.data} simulate={simulate.data} />
        <CategoryBattlefield data={matchup.data} simulate={simulate.data} />
      </div>
    </div>
  )
}
