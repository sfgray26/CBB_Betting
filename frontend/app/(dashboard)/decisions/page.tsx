'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import type { DecisionWithExplanation, DecisionPipelineStatus } from '@/lib/types'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import {
  User,
  TrendingUp,
  Shield,
  ChevronDown,
  ChevronUp,
  Loader2,
  AlertCircle,
  CheckCircle2,
  Info,
} from 'lucide-react'

type DecisionTypeFilter = 'lineup' | 'waiver'

const LINEUP_SLOT_ORDER = ['C', '1B', '2B', '3B', 'SS', 'OF', 'Util', 'SP', 'RP', 'P'] as const
const LINEUP_SLOT_CAPACITY: Record<(typeof LINEUP_SLOT_ORDER)[number], number> = {
  C: 1,
  '1B': 1,
  '2B': 1,
  '3B': 1,
  SS: 1,
  OF: 3,
  Util: 1,
  SP: 2,
  RP: 2,
  P: 1,
}

function getLineupCoverage(decisions: DecisionWithExplanation[]) {
  const filledCounts = Object.fromEntries(
    LINEUP_SLOT_ORDER.map((slot) => [slot, 0]),
  ) as Record<(typeof LINEUP_SLOT_ORDER)[number], number>

  for (const item of decisions) {
    const slot = item.decision.target_slot as (typeof LINEUP_SLOT_ORDER)[number] | null
    if (slot && slot in filledCounts) {
      filledCounts[slot] += 1
    }
  }

  const missingSlots = LINEUP_SLOT_ORDER.flatMap((slot) => {
    const missingCount = Math.max(0, LINEUP_SLOT_CAPACITY[slot] - filledCounts[slot])
    if (missingCount === 0) {
      return []
    }
    return missingCount === 1 ? [slot] : [`${slot} x${missingCount}`]
  })

  const expectedStarters = Object.values(LINEUP_SLOT_CAPACITY).reduce((sum, count) => sum + count, 0)
  const actualStarters = Object.values(filledCounts).reduce((sum, count) => sum + count, 0)

  return {
    actualStarters,
    expectedStarters,
    missingSlots,
  }
}

function sortLineupDecisions(decisions: DecisionWithExplanation[]) {
  const slotRank = Object.fromEntries(LINEUP_SLOT_ORDER.map((slot, index) => [slot, index]))

  return [...decisions].sort((left, right) => {
    const leftSlot = left.decision.target_slot ?? 'ZZZ'
    const rightSlot = right.decision.target_slot ?? 'ZZZ'
    const leftRank = slotRank[leftSlot] ?? Number.MAX_SAFE_INTEGER
    const rightRank = slotRank[rightSlot] ?? Number.MAX_SAFE_INTEGER

    if (leftRank !== rightRank) {
      return leftRank - rightRank
    }

    return right.decision.confidence - left.decision.confidence
  })
}

function StatusBadge({ status }: { status: DecisionPipelineStatus }) {
  const { verdict, message, decision_results } = status

  const verdictConfig = {
    healthy: { icon: CheckCircle2, color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30' },
    stale: { icon: AlertCircle, color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/30' },
    partial: { icon: Info, color: 'text-blue-400', bg: 'bg-blue-500/10', border: 'border-blue-500/30' },
    missing: { icon: AlertCircle, color: 'text-zinc-400', bg: 'bg-zinc-500/10', border: 'border-zinc-500/30' },
  }[verdict]

  const StatusIcon = verdictConfig.icon

  return (
    <div className={cn(
      'flex items-center gap-3 px-4 py-3 rounded-lg border',
      verdictConfig.bg, verdictConfig.border
    )}>
      <StatusIcon className={cn('h-5 w-5', verdictConfig.color)} />
      <div className="flex-1 min-w-0">
        <p className={cn('text-sm font-medium', verdictConfig.color)}>
          {message}
        </p>
        {decision_results.latest_as_of_date && (
          <p className="text-xs text-zinc-500 mt-0.5">
            Latest data: {decision_results.latest_as_of_date}
            {decision_results.breakdown_by_type && (
              <span> ({decision_results.breakdown_by_type.lineup ?? 0} lineup, {decision_results.breakdown_by_type.waiver ?? 0} waiver)</span>
            )}
          </p>
        )}
      </div>
    </div>
  )
}

function DecisionCard({ item }: { item: DecisionWithExplanation }) {
  const [showExplanation, setShowExplanation] = useState(false)
  const { decision, explanation } = item
  const isLineup = decision.decision_type === 'lineup'

  const confidencePct = Math.round(decision.confidence * 100)

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 overflow-hidden">
      {/* Main decision row */}
      <div className="p-4 space-y-3">
        {/* Header: player and type */}
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-full bg-zinc-800 flex items-center justify-center">
              <User className="h-5 w-5 text-zinc-400" />
            </div>
            <div>
              <div className="font-medium text-zinc-100">
                {decision.player_name || `Player #${decision.bdl_player_id}`}
              </div>
              <div className="flex items-center gap-2 mt-0.5">
                <span
                  className={cn(
                    'text-xs px-2 py-0.5 rounded-full font-medium',
                    isLineup
                      ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                      : 'bg-blue-500/10 text-blue-400 border border-blue-500/30',
                  )}
                >
                  {isLineup ? 'Lineup' : 'Waiver'}
                </span>
                {decision.target_slot && (
                  <span className="text-xs text-zinc-500">
                    Slot: <span className="text-zinc-300">{decision.target_slot}</span>
                  </span>
                )}
                {decision.drop_player_id && (
                  <span className="text-xs text-zinc-500">
                    Drop: <span className="text-zinc-300">
                      {decision.drop_player_name || `#${decision.drop_player_id}`}
                    </span>
                  </span>
                )}
              </div>
            </div>
          </div>
          <div className="text-right">
            <div className="text-xs text-zinc-500">Confidence</div>
            <div
              className={cn(
                'text-lg font-semibold tabular-nums',
                confidencePct >= 80
                  ? 'text-emerald-400'
                  : confidencePct >= 60
                    ? 'text-amber-400'
                    : 'text-zinc-300',
              )}
            >
              {confidencePct}%
            </div>
          </div>
        </div>

        {/* Value gain */}
        {decision.value_gain !== null && (
          <div className="flex items-center gap-2 text-sm">
            <TrendingUp className="h-4 w-4 text-amber-400" />
            <span className="text-zinc-400">Value gain:</span>
            <span className="font-mono text-amber-400 tabular-ns">
              {decision.value_gain > 0 ? '+' : ''}
              {decision.value_gain.toFixed(1)}
            </span>
          </div>
        )}

        {/* Reasoning */}
        {decision.reasoning && (
          <p className="text-sm text-zinc-300">{decision.reasoning}</p>
        )}

        {/* Explanation toggle */}
        {explanation && (
          <button
            onClick={() => setShowExplanation((v) => !v)}
            className="flex items-center gap-2 text-sm text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            {showExplanation ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
            {showExplanation ? 'Hide' : 'Show'} explanation
          </button>
        )}
      </div>

      {/* Explanation panel */}
      {explanation && showExplanation && (
        <div className="border-t border-zinc-800 bg-zinc-800/30 p-4 space-y-3">
          <div className="text-sm text-zinc-200">{explanation.summary}</div>

          {explanation.factors.length > 0 && (
            <div className="space-y-2">
              <div className="text-xs font-medium text-zinc-500 uppercase tracking-wider">
                Key Factors
              </div>
              <div className="grid gap-2 sm:grid-cols-2">
                {explanation.factors.map((factor, idx) => (
                  <div
                    key={idx}
                    className="bg-zinc-800/50 rounded-md p-3 border border-zinc-700/50"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-sm font-medium text-zinc-300">
                        {factor.label || factor.name}
                      </span>
                      {factor.value && (
                        <span className="font-mono text-xs text-amber-400 tabular-nums">
                          {factor.value}
                        </span>
                      )}
                    </div>
                    {/* Narrative is the high-signal part - hide technical weights */}
                    {factor.narrative && (
                      <p className="text-xs text-zinc-300 mt-2">{factor.narrative}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Narratives - simplified to just confidence and risk */}
          {(explanation.confidence_narrative || explanation.risk_narrative) && (
            <div className="flex flex-wrap gap-3 text-xs">
              {explanation.confidence_narrative && (
                <div className="flex items-center gap-1.5 text-zinc-300">
                  <Shield className="h-3.5 w-3.5 text-emerald-400" />
                  <span>{explanation.confidence_narrative}</span>
                </div>
              )}
              {explanation.risk_narrative && (
                <div className="flex items-center gap-1.5 text-zinc-300">
                  <AlertCircle className="h-3.5 w-3.5 text-amber-400" />
                  <span>{explanation.risk_narrative}</span>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function EmptyState({
  filter,
  selectedDate,
  resolvedAsOfDate,
}: {
  filter: DecisionTypeFilter
  selectedDate: string
  resolvedAsOfDate?: string
}) {
  const filterText =
    filter === 'lineup'
      ? 'No lineup decisions available.'
      : 'No waiver recommendations available.'

  const detailText = selectedDate
    ? `No recommendations for ${selectedDate}.`
      : resolvedAsOfDate
        ? `No recommendations available for the latest data (${resolvedAsOfDate}).`
        : 'No recommendations available yet.'

  return (
    <Card>
      <div className="flex items-center gap-3 text-zinc-400 py-8">
        <CheckCircle2 className="h-8 w-8 text-zinc-600" />
        <div>
          <p className="text-sm">{filterText}</p>
          <p className="text-xs text-zinc-600 mt-1">{detailText}</p>
          <p className="text-xs text-zinc-600 mt-1">
            {filter === 'lineup'
              ? 'Lineup recommendations will appear once your roster is synced and players are analyzed.'
              : 'Waiver recommendations are generated daily based on free agent availability and projected value.'
            }
          </p>
        </div>
      </div>
    </Card>
  )
}

export default function DecisionsPage() {
  const [typeFilter, setTypeFilter] = useState<DecisionTypeFilter>('lineup')
  const [dateFilter, setDateFilter] = useState<string>('')

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['decisions', typeFilter, dateFilter],
    queryFn: () =>
      endpoints.getDecisions({
        decision_type: typeFilter,
        as_of_date: dateFilter || undefined,
        limit: 100,
      }),
    })

  const filterButtons: { label: string; value: DecisionTypeFilter }[] = [
    { label: 'Lineup', value: 'lineup' },
    { label: 'Waiver', value: 'waiver' },
  ]

  const displayedDecisions =
    typeFilter === 'lineup' && data ? sortLineupDecisions(data.decisions) : data?.decisions ?? []
  const lineupCoverage =
    typeFilter === 'lineup' && data ? getLineupCoverage(data.decisions) : null

  const { data: statusData } = useQuery({
    queryKey: ['decisions-status'],
    queryFn: () => endpoints.getDecisionsStatus(),
    refetchInterval: 5 * 60 * 1000, // Refresh every 5 minutes
  })

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-semibold text-zinc-100">Daily Decisions</h1>
        <p className="text-sm text-zinc-500 mt-1">
          Trusted decision engine outputs for lineup and waiver optimization.
        </p>
      </div>

      {/* Status block - show when decisions are empty or status indicates issues */}
      {statusData && (data?.decisions.length === 0 || statusData.verdict !== 'healthy') && (
        <StatusBadge status={statusData} />
      )}

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4">
        {/* Type filter */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-500 uppercase tracking-wider">Type:</span>
          <div className="flex bg-zinc-900 rounded-lg p-1 border border-zinc-800">
            {filterButtons.map((btn) => (
              <button
                key={btn.value}
                onClick={() => setTypeFilter(btn.value)}
                className={cn(
                  'px-3 py-1.5 text-xs font-medium rounded-md transition-colors',
                  typeFilter === btn.value
                    ? 'bg-amber-400 text-zinc-950'
                    : 'text-zinc-400 hover:text-zinc-200',
                )}
              >
                {btn.label}
              </button>
            ))}
          </div>
        </div>

        {/* Date filter */}
        <div className="flex items-center gap-2">
          <label htmlFor="date-filter" className="text-xs text-zinc-500 uppercase tracking-wider">
            Date:
          </label>
          <input
            id="date-filter"
            type="date"
            value={dateFilter}
            onChange={(e) => setDateFilter(e.target.value)}
            className="bg-zinc-900 border border-zinc-800 rounded-md px-3 py-1.5 text-xs text-zinc-300 focus:outline-none focus:ring-1 focus:ring-amber-400/50"
          />
          {dateFilter && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setDateFilter('')}
              className="text-xs text-zinc-500 hover:text-zinc-300 h-7 px-2"
            >
              Clear
            </Button>
          )}
        </div>
      </div>

      {/* Status info */}
      {data && (
        <div className="flex items-center gap-4 text-xs text-zinc-500">
          <span>
            Showing {data.count} decision{data.count !== 1 ? 's' : ''}
          </span>
          <span>as of {data.as_of_date}</span>
        </div>
      )}

      {typeFilter === 'lineup' && lineupCoverage && data && data.decisions.length > 0 && (
        <Card>
          <div className="flex flex-col gap-2 py-3 text-sm text-zinc-300">
            <div className="flex items-center gap-2">
              <Info className="h-4 w-4 text-amber-400" />
              <span>
                Showing starter recommendations only, not your full roster or bench.
              </span>
            </div>
            <div className="text-xs text-zinc-500">
              Starter slot coverage: {lineupCoverage.actualStarters}/{lineupCoverage.expectedStarters}
              {lineupCoverage.missingSlots.length > 0 && (
                <span> | Missing: {lineupCoverage.missingSlots.join(', ')}</span>
              )}
            </div>
          </div>
        </Card>
      )}

      {/* Loading state */}
      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-6 w-6 text-amber-400 animate-spin" />
          <span className="ml-3 text-sm text-zinc-400">Loading decisions...</span>
        </div>
      )}

      {/* Error state */}
      {isError && (
        <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-6 text-rose-400 text-sm">
          Failed to load decisions:{' '}
          {error instanceof Error ? error.message : 'Unknown error'}
        </div>
      )}

      {/* Empty state */}
      {!isLoading && !isError && data && data.decisions.length === 0 && (
        <EmptyState
          filter={typeFilter}
          selectedDate={dateFilter}
          resolvedAsOfDate={data.as_of_date}
        />
      )}

      {/* Decision cards */}
      {!isLoading && !isError && data && data.decisions.length > 0 && (
        <div className="space-y-3">
          {displayedDecisions.map((item, idx) => (
            <DecisionCard key={idx} item={item} />
          ))}
        </div>
      )}
    </div>
  )
}
