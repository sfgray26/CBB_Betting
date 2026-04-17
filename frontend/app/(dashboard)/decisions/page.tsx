'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import type { DecisionWithExplanation } from '@/lib/types'
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
} from 'lucide-react'

type DecisionTypeFilter = 'all' | 'lineup' | 'waiver'

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
                    Drop: <span className="text-zinc-300">#{decision.drop_player_id}</span>
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
                Factors
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
                    {factor.narrative && (
                      <p className="text-xs text-zinc-400 mt-1">{factor.narrative}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {(explanation.confidence_narrative ||
            explanation.risk_narrative ||
            explanation.track_record_narrative) && (
            <div className="flex flex-wrap gap-3 text-xs">
              {explanation.confidence_narrative && (
                <div className="flex items-center gap-1.5 text-zinc-400">
                  <Shield className="h-3.5 w-3.5" />
                  <span>{explanation.confidence_narrative}</span>
                </div>
              )}
              {explanation.risk_narrative && (
                <div className="flex items-center gap-1.5 text-zinc-400">
                  <AlertCircle className="h-3.5 w-3.5" />
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
    filter === 'all'
      ? 'No decisions available yet.'
      : filter === 'lineup'
        ? 'No lineup decisions available.'
        : 'No waiver decisions available.'

  const detailText = selectedDate
    ? `No decision rows were returned for ${selectedDate}.`
    : resolvedAsOfDate
      ? `The backend returned no decision rows for its latest available date (${resolvedAsOfDate}).`
      : 'The backend returned no decision rows yet.'

  return (
    <Card>
      <div className="flex items-center gap-3 text-zinc-400 py-8">
        <CheckCircle2 className="h-8 w-8 text-zinc-600" />
        <div>
          <p className="text-sm">{filterText}</p>
          <p className="text-xs text-zinc-600 mt-1">{detailText}</p>
          <p className="text-xs text-zinc-600 mt-1">
            This page is loading correctly, but the decision pipeline has not populated this environment yet.
          </p>
        </div>
      </div>
    </Card>
  )
}

export default function DecisionsPage() {
  const [typeFilter, setTypeFilter] = useState<DecisionTypeFilter>('all')
  const [dateFilter, setDateFilter] = useState<string>('')

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['decisions', typeFilter, dateFilter],
    queryFn: () =>
      endpoints.getDecisions({
        decision_type: typeFilter === 'all' ? undefined : typeFilter,
        as_of_date: dateFilter || undefined,
        limit: 100,
      }),
    })

  const filterButtons: { label: string; value: DecisionTypeFilter }[] = [
    { label: 'All', value: 'all' },
    { label: 'Lineup', value: 'lineup' },
    { label: 'Waiver', value: 'waiver' },
  ]

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-semibold text-zinc-100">Daily Decisions</h1>
        <p className="text-sm text-zinc-500 mt-1">
          Trusted decision engine outputs for lineup and waiver optimization.
        </p>
      </div>

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
          {data.decisions.map((item, idx) => (
            <DecisionCard key={idx} item={item} />
          ))}
        </div>
      )}
    </div>
  )
}
