'use client'

import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import { Loader2, AlertCircle, Swords, Play } from 'lucide-react'
import { MatchupHeader } from '@/components/war-room/matchup-header'
import { CategoryBattlefield } from '@/components/war-room/category-battlefield'
import type { MatchupSimulateResponse } from '@/lib/types'

export default function WarRoomPage() {
  const [simulateData, setSimulateData] = useState<MatchupSimulateResponse | undefined>(undefined)

  const matchup = useQuery({
    queryKey: ['matchup'],
    queryFn: endpoints.getMatchup,
    staleTime: 5 * 60_000,
    refetchInterval: 5 * 60_000,
  })

  const simulateMutation = useMutation({
    mutationFn: endpoints.simulateMatchup,
    onSuccess: (data) => {
      setSimulateData(data)
    },
  })

  const handleSimulate = () => {
    simulateMutation.mutate()
  }

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

          {/* Run Simulation button */}
          <button
            onClick={handleSimulate}
            disabled={simulateMutation.isPending}
            className="ml-auto flex items-center gap-1.5 px-3 py-1.5 bg-[#FFC000] hover:bg-[#E5AC00] disabled:bg-[#494949] disabled:cursor-not-allowed text-black text-[10px] font-bold tracking-widest uppercase transition-colors"
          >
            {simulateMutation.isPending ? (
              <>
                <Loader2 className="h-3 w-3 animate-spin" />
                Simulating...
              </>
            ) : (
              <>
                <Play className="h-3 w-3" />
                Run Simulation
              </>
            )}
          </button>
        </div>

        {/* Simulation error message */}
        {simulateMutation.isError && (
          <div className="flex items-center gap-2 text-rose-500 text-xs bg-[#202020] px-3 py-2">
            <AlertCircle className="h-3.5 w-3.5" />
            <span>{simulateMutation.error?.message ?? 'Simulation failed'}</span>
          </div>
        )}

        <MatchupHeader data={matchup.data} simulate={simulateData} />
        <CategoryBattlefield data={matchup.data} simulate={simulateData} />
      </div>
    </div>
  )
}
