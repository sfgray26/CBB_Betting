'use client'

import { useEffect, useState } from 'react'
import { cn } from '@/lib/utils'
import { RefreshCw } from 'lucide-react'

/**
 * Sync status levels with human-readable labels.
 * - LIVE: Data updated within 5 minutes
 * - STALE: Data updated 5-60 minutes ago OR auth required
 * - OFFLINE: Data updated >60 minutes ago OR fetch failed
 */
export type SyncStatus = 'live' | 'stale' | 'offline'

interface FreshnessBadgeProps {
  severity: 'fresh' | 'warning' | 'critical'
  minutesAgo?: number | null
  warningText?: string | null
  isClickable?: boolean
  onRefresh?: () => void
  /** Enable auto-polling every 30 seconds (default: true) */
  enablePolling?: boolean
}

/**
 * Convert API severity to sync status with proper labels.
 * Maps: fresh→live, warning→stale, critical→offline.
 */
function severityToSyncStatus(severity: 'fresh' | 'warning' | 'critical'): SyncStatus {
  switch (severity) {
    case 'fresh':
      return 'live'
    case 'warning':
      return 'stale'
    case 'critical':
      return 'offline'
  }
}

/**
 * Format timestamp as "LIVE · 2 min ago" or "STALE · 45 min ago"
 */
function formatTimestamp(minutesAgo: number | null, status: SyncStatus): string {
  if (minutesAgo == null) {
    return status === 'live' ? '· just now' : '· unavailable'
  }
  if (minutesAgo < 1) return '· just now'
  if (minutesAgo < 60) return `· ${minutesAgo} min ago`
  const hours = Math.floor(minutesAgo / 60)
  if (hours < 24) return `· ${hours} hr ago`
  const days = Math.floor(hours / 24)
  return `· ${days} day ago`
}

export function computeFreshnessSeverity(minutesAgo: number | null): 'fresh' | 'warning' | 'critical' {
  if (minutesAgo == null) return 'warning'  // No timestamp → STALE (auth required or pending)
  if (minutesAgo > 60) return 'critical'  // > 60 min → OFFLINE
  if (minutesAgo > 5) return 'warning'     // 5-60 min → STALE
  return 'fresh'                           // < 5 min → LIVE
}

/**
 * FreshnessBadge component with LIVE/STALE/OFFLINE labels and auto-polling.
 *
 * Features:
 * - Auto-polls global-freshness endpoint every 30 seconds (enablePolling=true)
 * - Shows LIVE (< 5 min), STALE (5-60 min), OFFLINE (> 60 min)
 * - Never shows "UNKNOWN" as terminal state - uses OFFLINE instead
 * - Displays human-readable timestamps: "LIVE · 2 min ago"
 * - Clickable to refresh (revalidates global-freshness query)
 */
export function FreshnessBadge({
  severity,
  minutesAgo,
  warningText,
  isClickable,
  onRefresh,
  enablePolling = true,
}: FreshnessBadgeProps) {
  const syncStatus = severityToSyncStatus(severity)
  const [isRefreshing, setIsRefreshing] = useState(false)

  // Auto-poll every 30 seconds when enabled
  useEffect(() => {
    if (!enablePolling || !onRefresh) return

    const interval = setInterval(() => {
      setIsRefreshing(true)
      onRefresh()
      // Brief flash of loading state, then reset
      setTimeout(() => setIsRefreshing(false), 500)
    }, 30_000)

    return () => clearInterval(interval)
  }, [enablePolling, onRefresh])

  const handleRefresh = () => {
    if (!isClickable || !onRefresh) return
    setIsRefreshing(true)
    onRefresh()
    setTimeout(() => setIsRefreshing(false), 500)
  }

  const dotClass: Record<SyncStatus, string> = {
    live: 'bg-status-safe',
    stale: 'bg-status-bubble',
    offline: 'bg-status-lost',
  }

  const badgeClass: Record<SyncStatus, string> = {
    live: 'bg-status-safe/10 border-status-safe/30 text-status-safe',
    stale: 'bg-status-bubble/10 border-status-bubble/30 text-status-bubble',
    offline: 'bg-status-lost/10 border-status-lost/30 text-status-lost',
  }

  const label: Record<SyncStatus, string> = {
    live: 'LIVE',
    stale: 'STALE',
    offline: 'OFFLINE',
  }

  const timeText = formatTimestamp(minutesAgo ?? null, syncStatus)

  return (
    <button
      type="button"
      onClick={handleRefresh}
      disabled={!isClickable || !onRefresh}
      className={cn(
        'inline-flex items-center gap-1.5 text-[10px] px-2 py-1 rounded border font-semibold tracking-wider uppercase transition-opacity',
        badgeClass[syncStatus],
        isClickable && onRefresh ? 'cursor-pointer hover:opacity-80' : 'cursor-default',
        isRefreshing && 'opacity-70',
      )}
      title={warningText ?? `Click to refresh${timeText ? ` — ${timeText.slice(2)}` : ''}`}
    >
      <span
        className={cn(
          'w-1.5 h-1.5 rounded-full flex-shrink-0',
          syncStatus === 'offline' ? 'animate-pulse' : '',
          dotClass[syncStatus],
        )}
      />
      {label[syncStatus]}
      {timeText && (
        <span className="font-normal normal-case tracking-normal">{timeText}</span>
      )}
      {(isClickable || isRefreshing) && (
        <RefreshCw className={cn('h-3 w-3', isRefreshing && 'animate-spin')} />
      )}
    </button>
  )
}
