'use client'

import { useQuery } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import { Loader2, AlertCircle } from 'lucide-react'

export default function WarRoomPage() {
  const today = new Date().toISOString().slice(0, 10)

  const matchup = useQuery({
    queryKey: ['matchup'],
    queryFn: endpoints.getMatchup,
    staleTime: 5 * 60_000,
  })

  const lineup = useQuery({
    queryKey: ['lineup', today],
    queryFn: () => endpoints.getLineup(today),
    staleTime: 5 * 60_000,
  })

  const isLoading = matchup.isLoading || lineup.isLoading
  const isError = matchup.isError || lineup.isError

  return (
    <div className="min-h-screen bg-black p-6 space-y-6">
      <h1 className="text-xl font-bold tracking-widest uppercase text-[#FFC000]">
        WAR ROOM
      </h1>

      {isLoading && (
        <div className="flex items-center gap-2 text-[#7D7D7D]">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span className="text-sm">Loading matchup data...</span>
        </div>
      )}

      {isError && (
        <div className="flex items-center gap-2 text-rose-400">
          <AlertCircle className="h-4 w-4" />
          <span className="text-sm">
            {matchup.error?.message ?? lineup.error?.message ?? 'API error'}
          </span>
        </div>
      )}

      {matchup.data && (
        <div className="bg-[#202020] p-4 space-y-2">
          <p className="text-xs font-semibold tracking-widest uppercase text-[#7D7D7D]">MATCHUP — API OK</p>
          <p className="text-white text-sm">Week {matchup.data.week} vs {matchup.data.opponent_name}</p>
          <pre className="text-xs text-[#969696] overflow-auto max-h-48">
            {JSON.stringify(matchup.data, null, 2)}
          </pre>
        </div>
      )}

      {lineup.data && (
        <div className="bg-[#202020] p-4 space-y-2">
          <p className="text-xs font-semibold tracking-widest uppercase text-[#7D7D7D]">LINEUP — API OK</p>
          <p className="text-white text-sm">{lineup.data.lineup.length} players · {today}</p>
          <pre className="text-xs text-[#969696] overflow-auto max-h-48">
            {JSON.stringify(lineup.data, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}
