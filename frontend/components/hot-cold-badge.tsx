'use client'

import { Flame, Snowflake } from 'lucide-react'
import { cn } from '@/lib/utils'

interface HotColdBadgeProps {
  hotCold?: 'HOT' | 'COLD' | string | null
  rankPercentile?: number | null
  trend?: 'hot' | 'cold' | 'neutral' | string | null
  trendScore?: number | null
  className?: string
}

/**
 * Shared Hot/Cold badge used across waiver, roster, and streaming pages.
 *
 * Supports two data shapes:
 * 1. Waiver-style: hotCold string + rankPercentile (gates to top 20%)
 * 2. Dashboard-style: trend string + trendScore (gates by score magnitude)
 */
export function HotColdBadge({
  hotCold,
  rankPercentile,
  trend,
  trendScore,
  className,
}: HotColdBadgeProps) {
  // Normalize to a simple "hot" | "cold" | null
  let kind: 'hot' | 'cold' | null = null

  if (hotCold) {
    if ((rankPercentile ?? 0) < 80) return null
    if (hotCold === 'HOT') kind = 'hot'
    else if (hotCold === 'COLD') kind = 'cold'
  } else if (trend) {
    // Dashboard streaks: gate by trend score magnitude (≥1.0 or ≤-1.0)
    const score = trendScore ?? 0
    if (trend === 'hot' && score >= 1.0) kind = 'hot'
    else if (trend === 'cold' && score <= -1.0) kind = 'cold'
  }

  if (!kind) return null

  const isHot = kind === 'hot'

  const tooltip = isHot
    ? 'HOT: avg category z-score > 0.75 across recent stats (7-day window)'
    : 'COLD: avg category z-score < −0.5 across recent stats (7-day window)'

  return (
    <span
      title={tooltip}
      className={cn(
        'inline-flex items-center gap-0.5 text-[10px] font-semibold cursor-help',
        isHot ? 'text-status-behind' : 'text-signal-consider',
        className,
      )}
    >
      {isHot ? <Flame className="h-3 w-3" /> : <Snowflake className="h-3 w-3" />}
      {isHot ? 'HOT' : 'COLD'}
    </span>
  )
}
