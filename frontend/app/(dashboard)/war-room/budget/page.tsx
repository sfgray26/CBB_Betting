'use client'

import { useQuery } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import { Loader2, AlertCircle, DollarSign, Calendar, TrendingUp } from 'lucide-react'
import { BudgetPanel } from '@/components/dashboard/budget-panel'

export default function BudgetPage() {
  const { data, isLoading, isError, error, refetch } = useQuery({
    queryKey: ['budget'],
    queryFn: endpoints.getBudget,
    staleTime: 5 * 60_000,
    refetchInterval: 10 * 60_000,
  })

  if (isLoading) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center">
        <div className="flex items-center gap-2 text-text-secondary">
          <Loader2 className="h-5 w-5 animate-spin text-accent-gold" />
          <span className="text-sm">Loading budget...</span>
        </div>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center">
        <div className="bg-bg-surface border border-border-subtle rounded-lg p-6 max-w-md w-full">
          <div className="flex items-center gap-2 text-status-lost mb-2">
            <AlertCircle className="h-5 w-5" />
            <span className="text-sm font-semibold">Failed to load budget</span>
          </div>
          <p className="text-text-secondary text-sm">
            {error instanceof Error ? error.message : 'Unknown error'}
          </p>
          <button onClick={() => refetch()} className="mt-4 text-xs text-accent-gold hover:text-amber-300 font-semibold">
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (!data) return null

  const { budget, freshness } = data
  const fetchedAt = freshness?.fetched_at
    ? new Date(freshness.fetched_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : null

  return (
    <div className="space-y-6 max-w-md">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <DollarSign className="h-3.5 w-3.5 text-accent-gold" />
          {/* Page-level label; the panel below carries the "Constraint Budget"
              card title — avoid the duplicate identical heading (§B2). */}
          <span className="text-xs font-bold tracking-widest uppercase text-accent-gold">
            Weekly Budget
          </span>
        </div>
        {fetchedAt && (
          <span className="text-[10px] text-text-muted">
            {freshness.is_stale ? '⚠ stale · ' : ''}updated {fetchedAt}
          </span>
        )}
      </div>

      {/* Stale warning */}
      {freshness?.is_stale && (
        <div className="bg-status-bubble/10 border border-status-bubble/30 rounded-lg px-3 py-2 text-xs text-status-bubble">
          Budget data is stale — Yahoo stats may not reflect the latest transactions.
        </div>
      )}

      <BudgetPanel budget={budget} />

      {/* Season Pace panel */}
      <div className="bg-bg-surface border border-border-subtle rounded-lg px-4 py-3 space-y-3">
        <div className="flex items-center gap-2">
          <Calendar className="h-3.5 w-3.5 text-accent-gold" />
          <span className="text-xs font-bold tracking-widest uppercase text-accent-gold">
            Season Pace
          </span>
        </div>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          <div>
            <p className="text-[10px] text-text-muted uppercase tracking-wide">Current Week</p>
            <p className="text-sm font-semibold text-text-primary">
              {budget.week_label ?? '—'}
            </p>
          </div>
          <div>
            <p className="text-[10px] text-text-muted uppercase tracking-wide">Weeks Left</p>
            <p className="text-sm font-semibold text-text-primary">
              {budget.weeks_remaining != null ? budget.weeks_remaining : '—'}
            </p>
          </div>
          <div>
            <p className="text-[10px] text-text-muted uppercase tracking-wide">Days Left in Week</p>
            <p className="text-sm font-semibold text-text-primary">
              {budget.days_in_week_remaining != null ? budget.days_in_week_remaining : '—'}
            </p>
          </div>
          <div>
            <p className="text-[10px] text-text-muted uppercase tracking-wide">Season Adds</p>
            <p className="text-sm font-semibold text-text-primary">
              {budget.acquisitions_this_season != null ? budget.acquisitions_this_season : '—'}
            </p>
          </div>
        </div>
      </div>

      {/* Remaining acquisitions callout */}
      <div className="bg-bg-surface border border-border-subtle rounded-lg px-4 py-3">
        <div className="flex items-center gap-2 mb-1">
          <TrendingUp className="h-3 w-3 text-text-muted" />
          <p className="text-xs text-text-secondary">
            <span className="text-text-primary font-semibold">{budget.acquisitions_remaining}</span>
            {' '}weekly acquisition{budget.acquisitions_remaining !== 1 ? 's' : ''} remaining
            {budget.acquisition_warning && (
              <span className="text-status-bubble ml-2 font-semibold">— budget tight</span>
            )}
          </p>
        </div>
      </div>
    </div>
  )
}
