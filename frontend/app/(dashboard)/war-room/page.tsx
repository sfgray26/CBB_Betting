'use client'

import { useState, useEffect, useCallback } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import { Loader2, AlertCircle, Swords, Play, RefreshCw } from 'lucide-react'
import { MatchupHeader } from '@/components/war-room/matchup-header'
import { FreshnessBadge } from '@/components/freshness/freshness-badge'
import { CategoryBattlefield } from '@/components/war-room/category-battlefield'
import { MatchupSkeleton } from '@/components/war-room/matchup-skeleton'
import { cn } from '@/lib/utils'
import type { MatchupSimulateResponse } from '@/lib/types'

// ─── Time helpers ─────────────────────────────────────────────────────────────

function getETHour(): number {
  const now = new Date()
  const etStr = now.toLocaleString('en-US', {
    timeZone: 'America/New_York',
    hour: 'numeric',
    hour12: false,
  })
  return parseInt(etStr, 10)
}

function isActiveGameWindow(): boolean {
  const hour = getETHour()
  return hour >= 13 && hour < 23 // 1PM – 11PM ET
}

/** Configurable auto-run threshold in minutes */
const AUTO_RUN_STALENESS_THRESHOLD_MINUTES = 45

// ─── Staleness Banner ─────────────────────────────────────────────────────────

function StalenessBanner({
  ageMinutes,
  onReRun,
  isPending,
}: {
  ageMinutes: number
  onReRun: () => void
  isPending: boolean
}) {
  return (
    <div className="bg-status-bubble/10 border border-status-bubble/40 rounded-lg p-4 flex items-start gap-3 mb-4">
      <AlertCircle className="h-5 w-5 text-status-bubble flex-shrink-0 mt-0.5" />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-status-bubble">
          ⚠️ Projections are {ageMinutes} minutes old
        </p>
        <p className="text-xs text-text-secondary mt-0.5">
          Data may be stale during active game window. Click Re-Run to update.
        </p>
      </div>
      <button
        onClick={onReRun}
        disabled={isPending}
        className="flex items-center gap-1.5 px-4 py-2 bg-accent-gold hover:bg-amber-500 disabled:bg-text-muted disabled:cursor-not-allowed text-black text-xs font-bold tracking-wider uppercase transition-colors rounded-sm flex-shrink-0"
      >
        {isPending ? (
          <>
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Updating…
          </>
        ) : (
          <>
            <RefreshCw className="h-3.5 w-3.5" />
            Re-Run
          </>
        )}
      </button>
    </div>
  )
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function WarRoomPage() {
  const [simulateData, setSimulateData] = useState<MatchupSimulateResponse | undefined>(undefined)
  const [autoRunFired, setAutoRunFired] = useState(false)

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

  const { data: globalFreshness, refetch: refetchFreshness } = useQuery({
    queryKey: ['global-freshness'],
    queryFn: endpoints.getGlobalFreshness,
    staleTime: 2 * 60_000,
    refetchInterval: 5 * 60_000,
  })

  const simulateMutation = useMutation({
    mutationFn: endpoints.simulateMatchup,
    onSuccess: (data) => {
      setSimulateData(data)
    },
  })

  const handleSimulate = useCallback(() => {
    simulateMutation.mutate()
  }, [simulateMutation])

  // Auto-trigger simulation when projections are stale during active game window
  useEffect(() => {
    if (
      !autoRunFired &&
      matchup.data &&
      projStatus &&
      projStatus.age_hours != null &&
      !simulateMutation.isPending
    ) {
      const ageMinutes = projStatus.age_hours * 60
      const shouldAutoRun =
        ageMinutes > AUTO_RUN_STALENESS_THRESHOLD_MINUTES && isActiveGameWindow()
      if (shouldAutoRun) {
        setAutoRunFired(true)
        handleSimulate()
      }
    }
  }, [matchup.data, projStatus, autoRunFired, handleSimulate, simulateMutation.isPending])

  // Initial auto-simulate when matchup loads
  useEffect(() => {
    if (matchup.data && !simulateData && !simulateMutation.isPending) {
      simulateMutation.mutate()
    }
  }, [matchup.data]) // eslint-disable-line react-hooks/exhaustive-deps

  const stalenessMinutes =
    projStatus?.age_hours != null ? Math.round(projStatus.age_hours * 60) : null

  const showStalenessBanner =
    stalenessMinutes != null &&
    stalenessMinutes > 60 &&
    isActiveGameWindow()

  if (matchup.isLoading) {
    return (
      <div className="min-h-screen bg-bg-base">
        <MatchupSkeleton />
      </div>
    )
  }

  if (matchup.isError) {
    return (
      <div className="min-h-screen bg-bg-base p-6">
        <div className="flex items-center gap-2 text-status-lost">
          <AlertCircle className="h-6 w-6" />
          <span className="text-base font-mono">{matchup.error?.message ?? 'Failed to load matchup'}</span>
        </div>
      </div>
    )
  }

  if (!matchup.data) return null

  return (
    <div className="min-h-screen bg-bg-base">
      <div className="max-w-6xl mx-auto p-6 lg:p-8 space-y-6">
        {/* Prominent staleness banner */}
        {showStalenessBanner && (
          <StalenessBanner
            ageMinutes={stalenessMinutes}
            onReRun={handleSimulate}
            isPending={simulateMutation.isPending}
          />
        )}

        {/* Page header row */}
        <div className="flex items-center gap-3 mb-2">
          <Swords className="h-6 w-6 text-accent-gold" />
          <span className="text-lg font-bold tracking-widest uppercase text-accent-gold">War Room</span>
          {matchup.data && matchup.data.week != null && matchup.data.week > 0 && (
            <span className="text-[10px] px-2 py-1 bg-accent-gold/10 text-accent-gold border border-accent-gold/30 rounded font-bold tracking-wider uppercase">
              Week {matchup.data.week} · IN-FLIGHT
            </span>
          )}

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

          {/* Global freshness badge */}
          {globalFreshness && (
            <FreshnessBadge
              severity={globalFreshness.severity}
              minutesAgo={globalFreshness.minutes_ago}
              warningText={globalFreshness.warning_text}
              isClickable={true}
              onRefresh={() => refetchFreshness()}
            />
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
                Simulating…
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
