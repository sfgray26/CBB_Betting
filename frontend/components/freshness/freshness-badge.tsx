'use client'

import { cn } from '@/lib/utils'
import { RefreshCw } from 'lucide-react'

export type FreshnessSeverity = 'fresh' | 'warning' | 'critical' | 'unknown'

interface FreshnessBadgeProps {
  severity: FreshnessSeverity
  minutesAgo?: number | null
  warningText?: string | null
  isClickable?: boolean
  onRefresh?: () => void
}

export function computeFreshnessSeverity(minutesAgo: number | null): FreshnessSeverity {
  if (minutesAgo == null) return 'unknown'
  if (minutesAgo > 120) return 'critical'
  if (minutesAgo > 60) return 'warning'
  return 'fresh'
}

export function FreshnessBadge({
  severity,
  minutesAgo,
  warningText,
  isClickable,
  onRefresh,
}: FreshnessBadgeProps) {
  const dotClass: Record<FreshnessSeverity, string> = {
    fresh: 'bg-status-safe',
    warning: 'bg-status-bubble',
    critical: 'bg-status-lost',
    unknown: 'bg-text-muted',
  }

  const badgeClass: Record<FreshnessSeverity, string> = {
    fresh: 'bg-status-safe/10 border-status-safe/30 text-status-safe',
    warning: 'bg-status-bubble/10 border-status-bubble/30 text-status-bubble',
    critical: 'bg-status-lost/10 border-status-lost/30 text-status-lost',
    unknown: 'bg-bg-surface border-border-subtle text-text-muted',
  }

  const label: Record<FreshnessSeverity, string> = {
    fresh: 'FRESH',
    warning: 'STALE',
    critical: 'CRITICAL',
    unknown: 'UNKNOWN',
  }

  const timeText = (() => {
    if (minutesAgo == null) return ''
    if (minutesAgo < 1) return '· just now'
    if (minutesAgo < 60) return `· ${minutesAgo}m ago`
    const hours = Math.floor(minutesAgo / 60)
    if (hours < 24) return `· ${hours}h ago`
    const days = Math.floor(hours / 24)
    return `· ${days}d ago`
  })()

  return (
    <button
      type="button"
      onClick={isClickable ? onRefresh : undefined}
      disabled={!isClickable || !onRefresh}
      className={cn(
        'inline-flex items-center gap-1.5 text-[10px] px-2 py-1 rounded border font-semibold tracking-wider uppercase transition-opacity',
        badgeClass[severity],
        isClickable && onRefresh ? 'cursor-pointer hover:opacity-80' : 'cursor-default',
      )}
      title={warningText ?? undefined}
    >
      <span
        className={cn(
          'w-1.5 h-1.5 rounded-full flex-shrink-0',
          severity === 'warning' || severity === 'critical' ? 'animate-pulse' : '',
          dotClass[severity],
        )}
      />
      {label[severity]}
      {timeText && (
        <span className="font-normal normal-case tracking-normal">{timeText}</span>
      )}
      {isClickable && onRefresh && <RefreshCw className="h-3 w-3" />}
    </button>
  )
}
