'use client'

import { useState, useEffect } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import { Loader2, AlertCircle, Swords, Play } from 'lucide-react'
import { MatchupHeader } from '@/components/war-room/matchup-header'
import { CategoryBattlefield } from '@/components/war-room/category-battlefield'
import { MatchupSkeleton } from '@/components/war-room/matchup-skeleton'
import { cn } from '@/lib/utils'
import type { MatchupSimulateResponse } from '@/lib/types'

export default function WarRoomPage() {
  const [simulateData, setSimulateData] = useState<MatchupSimulateResponse | undefined>(undefined)

  const matchup = useQuery({
    queryKey: ['matchup'],
    queryFn: endpoints.getMatchup,
    staleTime: 5 * 60_000,
    refetchInterval: 5 * 60_000,
  })

  const { data: projStatus } = useQuery({
    queryKey: ['projection-status'],
    queryFn: endpoints.getProjectionStatus,
    staleTime: 30 * 60_000,
    refetchInterval: 60 * 60_000,
  })

  const simulateMutation = useMutation({
    mutationFn: endpoints.simulateMatchup,
    onSuccess: (data) => {
      setSimulateData(data)
    },
  })

  useEffect(() => {
    if (matchup.data && !simulateData && !simulateMutation.isPending) {
      simulateMutation.mutate()
    }
  }, [matchup.data]) // eslint-disable-line react-hooks/exhaustive-deps

  const handleSimulate = () => {
    simulateMutation.mutate()
  }

  if (matchup.isLoading) {
    return (
      <div className="min-h-screen bg-bg-base">
        <MatchupSkeleton />
      </div>
    )
  }

  if (matchup.isError) {
    return (
      <div className="min-h-screen bg-black p-6">
        <div className="flex items-center gap-2 text-status-lost">
          <AlertCircle className="h-6 w-6" />
          <span className="text-base font-mono">{matchup.error?.message ?? 'Failed to load matchup'}</span>
        </div>
      </div>
    )
  }

  if (!matchup.data) return null

  return (
    <div className="min-h-screen bg-black">
      <div className="max-w-6xl mx-auto p-6 lg:p-8 space-y-6">
        {/* Page header row */}
        <div className="flex items-center gap-3 mb-2">
          <Swords className="h-6 w-6 text-accent-gold" />
          <span className="text-lg font-bold tracking-widest uppercase text-accent-gold">War Room</span>

          {/* Projection freshness badge */}
          {projStatus && (
            <div className={cn(
              'flex items-center gap-1.5 text-[10px] px-2 py-1 rounded border font-semibold tracking-wider',
              projStatus.is_stale
                ? 'bg-status-bubble/10 border-status-bubble/30 text-status-bubble'
                : 'bg-bg-surface border-border-subtle text-text-muted',
            )}>
              <span className={cn(
                'w-1.5 h-1.5 rounded-full flex-shrink-0',
                projStatus.is_stale ? 'bg-status-bubble animate-pulse' : 'bg-status-safe',
              )} />
              {'PROJ '}
              {projStatus.age_hours != null
                ? projStatus.age_hours < 1 ? 'FRESH' : `${projStatus.age_hours}H AGO`
                : 'UNKNOWN'}
            </div>
          )}

          {/* Run Simulation button */}
          <button
            onClick={handleSimulate}
            disabled={simulateMutation.isPending}
            className="ml-auto flex items-center gap-2 px-5 py-2.5 bg-accent-gold hover:bg-amber-500 disabled:bg-text-muted disabled:cursor-not-allowed text-black text-sm font-bold tracking-widest uppercase transition-colors rounded-sm"
          >
            {simulateMutation.isPending ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Simulating...
              </>
            ) : (
              <>
                <Play className="h-4 w-4" />
                {simulateData ? 'Re-run' : 'Run Simulation'}
              </>
            )}
          </button>
        </div>

        {/* Simulation error message */}
        {simulateMutation.isError && (
          <div className="flex items-center gap-2 text-status-lost text-sm bg-bg-surface border border-border-subtle px-4 py-3 rounded-sm">
            <AlertCircle className="h-5 w-5" />
            <span>{simulateMutation.error?.message ?? 'Simulation failed'}</span>
          </div>
        )}

        <MatchupHeader data={matchup.data} simulate={simulateData} />
        <CategoryBattlefield data={matchup.data} simulate={simulateData} />
      </div>
    </div>
  )
}
