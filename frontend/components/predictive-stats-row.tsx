'use client'

import { cn } from '@/lib/utils'

/**
 * PredictiveStatsRow — Secondary stat row for advanced/predictive metrics.
 *
 * Reads FIP/xFIP/SIERA for pitchers and xwOBA/Hard-Hit% for batters from a
 * flexible stats dictionary. Designed to sit below the primary stats row on
 * roster cards, waiver rows, and probable-pitcher widgets.
 *
 * Data hierarchy (checked in order):
 *   1. Provided `stats` prop (season, ros_projection, or statcast_stats)
 *   2. Fallback to `statcast_stats` if passed separately
 *
 * When data is missing the cell renders "—" so the UI never breaks.
 */

const PITCHER_KEYS = [
  { key: 'FIP', label: 'FIP', fmt: (v: number) => v.toFixed(2) },
  { key: 'xFIP', label: 'xFIP', fmt: (v: number) => v.toFixed(2) },
  { key: 'SIERA', label: 'SIERA', fmt: (v: number) => v.toFixed(2) },
]

const BATTER_KEYS = [
  { key: 'xwOBA', label: 'xwOBA', fmt: (v: number) => v.toFixed(3).replace(/^0\./, '.') },
  { key: 'HardHit', label: 'Hard-Hit%', fmt: (v: number) => `${v.toFixed(1)}%` },
]

interface PredictiveStatsRowProps {
  /** Primary stats dictionary (e.g. season_stats.values, ros_projection.values) */
  stats?: Record<string, number | null> | null
  /** Optional separate statcast dictionary for fallback lookups */
  statcastStats?: Record<string, number | null> | null
  isPitcher: boolean
  /** Visual density */
  size?: 'sm' | 'md'
  /** Whether to show the "Predictive" label chip */
  showLabel?: boolean
  className?: string
}

export function PredictiveStatsRow({
  stats,
  statcastStats,
  isPitcher,
  size = 'sm',
  showLabel = true,
  className,
}: PredictiveStatsRowProps) {
  const keys = isPitcher ? PITCHER_KEYS : BATTER_KEYS

  function getValue(key: string): number | null {
    if (stats && key in stats) return stats[key] ?? null
    if (statcastStats && key in statcastStats) return statcastStats[key] ?? null
    return null
  }

  const hasAny = keys.some((k) => getValue(k.key) != null)
  if (!hasAny) return null

  return (
    <div className={cn('flex items-center gap-2', className)}>
      {showLabel && (
        <span className={cn(
          'shrink-0 px-1.5 py-0.5 rounded border font-semibold tracking-wider uppercase',
          'bg-accent-primary/10 text-accent-primary border-accent-primary/20',
          size === 'sm' ? 'text-[9px]' : 'text-[10px]',
        )}>
          Predictive
        </span>
      )}
      <div className="flex items-center gap-3 flex-wrap">
        {keys.map(({ key, label, fmt }) => {
          const v = getValue(key)
          return (
            <div key={key} className="flex items-center gap-1">
              <span className={cn(
                'text-text-muted uppercase tracking-wider',
                size === 'sm' ? 'text-[9px]' : 'text-[10px]',
              )}>
                {label}
              </span>
              <span className={cn(
                'font-semibold tabular-nums text-text-primary',
                size === 'sm' ? 'text-xs' : 'text-sm',
              )}>
                {v != null ? fmt(v) : '—'}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
