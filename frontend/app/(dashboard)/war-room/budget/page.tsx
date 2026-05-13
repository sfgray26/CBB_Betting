'use client'

import { useQuery } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import { Loader2, AlertCircle, DollarSign } from 'lucide-react'
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
        <div className="flex items-center gap-2 text-[#7D7D7D]">
          <Loader2 className="h-5 w-5 animate-spin text-[#FFC000]" />
          <span className="text-sm">Loading budget...</span>
        </div>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center">
        <div className="bg-[#202020] rounded-lg p-6 max-w-md w-full">
          <div className="flex items-center gap-2 text-rose-400 mb-2">
            <AlertCircle className="h-5 w-5" />
            <span className="text-sm font-semibold">Failed to load budget</span>
          </div>
          <p className="text-[#969696] text-sm">
            {error instanceof Error ? error.message : 'Unknown error'}
          </p>
          <button onClick={() => refetch()} className="mt-4 text-xs text-[#FFC000] hover:text-amber-300 font-semibold">
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
          <DollarSign className="h-3.5 w-3.5 text-[#FFC000]" />
          <span className="text-xs font-bold tracking-widest uppercase text-[#FFC000]">
            Constraint Budget
          </span>
        </div>
        {fetchedAt && (
          <span className="text-[10px] text-[#494949]">
            {freshness.is_stale ? '⚠ stale · ' : ''}updated {fetchedAt}
          </span>
        )}
      </div>

      {/* Stale warning */}
      {freshness?.is_stale && (
        <div className="bg-amber-900/20 border border-amber-700/40 rounded-lg px-3 py-2 text-xs text-amber-300">
          Budget data is stale — Yahoo stats may not reflect the latest transactions.
        </div>
      )}

      <BudgetPanel budget={budget} />

      {/* Remaining acquisitions callout */}
      <div className="bg-[#181818] rounded-lg px-4 py-3">
        <p className="text-xs text-[#7D7D7D]">
          <span className="text-white font-semibold">{budget.acquisitions_remaining}</span>
          {' '}acquisition{budget.acquisitions_remaining !== 1 ? 's' : ''} remaining this season
          {budget.acquisition_warning && (
            <span className="text-amber-400 ml-2 font-semibold">— budget tight</span>
          )}
        </p>
      </div>
    </div>
  )
}
