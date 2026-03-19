'use client'

import { useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { formatDistanceToNow, parseISO } from 'date-fns'
import {
  Info,
  AlertTriangle,
  AlertOctagon,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Zap,
} from 'lucide-react'
import { endpoints } from '@/lib/api'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import type { Alert, LiveAlert } from '@/lib/types'
import { cn } from '@/lib/utils'

function ErrorCard({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-6 text-rose-400 text-sm">
      {message}
    </div>
  )
}

// API returns uppercase severity: INFO, WARNING, CRITICAL
function severityConfig(severity: string) {
  switch (severity) {
    case 'CRITICAL':
      return {
        Icon: AlertOctagon,
        color: 'text-rose-500',
        borderColor: 'border-rose-500/30',
        bgColor: 'bg-rose-500/5',
        label: 'Critical',
      }
    case 'WARNING':
      return {
        Icon: AlertTriangle,
        color: 'text-amber-400',
        borderColor: 'border-amber-400/30',
        bgColor: 'bg-amber-400/5',
        label: 'Warning',
      }
    default:
      return {
        Icon: Info,
        color: 'text-sky-400',
        borderColor: 'border-sky-400/30',
        bgColor: 'bg-sky-400/5',
        label: 'Info',
      }
  }
}

function AlertCard({ alert, onAck }: { alert: Alert; onAck: (id: number) => void }) {
  const cfg = severityConfig(alert.severity)
  const Icon = cfg.Icon

  return (
    <div
      className={cn(
        'rounded-lg border p-4 flex items-start gap-3 transition-opacity',
        cfg.bgColor,
        cfg.borderColor,
        alert.acknowledged && 'opacity-40',
      )}
    >
      <Icon className={cn('h-5 w-5 flex-shrink-0 mt-0.5', cfg.color)} />
      <div className="flex-1 min-w-0">
        <div className="flex items-start justify-between gap-4">
          <div>
            <span className={cn('text-xs font-semibold uppercase tracking-wider', cfg.color)}>
              {alert.alert_type}
            </span>
            <p className="text-sm text-zinc-200 mt-0.5">{alert.message}</p>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <span className="text-xs text-zinc-500 whitespace-nowrap">
              {alert.created_at
                ? formatDistanceToNow(parseISO(alert.created_at), { addSuffix: true })
                : 'Unknown time'}
            </span>
            {!alert.acknowledged && (
              <Button
                variant="ghost"
                size="sm"
                className="text-xs text-zinc-400 hover:text-zinc-100 h-7 px-2"
                onClick={() => onAck(alert.id)}
              >
                Acknowledge
              </Button>
            )}
          </div>
        </div>
        {alert.acknowledged && alert.acknowledged_at && (
          <p className="text-xs text-zinc-600 mt-1">
            Acknowledged{' '}
            {formatDistanceToNow(parseISO(alert.acknowledged_at), { addSuffix: true })}
          </p>
        )}
      </div>
    </div>
  )
}

function LiveAlertCard({ alert }: { alert: LiveAlert }) {
  const cfg = severityConfig(alert.severity)
  const Icon = cfg.Icon

  return (
    <div
      className={cn(
        'rounded-lg border p-4 flex items-start gap-3',
        cfg.bgColor,
        cfg.borderColor,
      )}
    >
      <Icon className={cn('h-5 w-5 flex-shrink-0 mt-0.5', cfg.color)} />
      <div className="flex-1 min-w-0">
        <span className={cn('text-xs font-semibold uppercase tracking-wider', cfg.color)}>
          {alert.alert_type}
        </span>
        <p className="text-sm text-zinc-200 mt-0.5">{alert.message}</p>
        {alert.recommendation && (
          <p className="text-xs text-zinc-400 mt-1 italic">{alert.recommendation}</p>
        )}
      </div>
    </div>
  )
}

export default function AlertsPage() {
  const queryClient = useQueryClient()
  const [showAcknowledged, setShowAcknowledged] = useState(false)
  const [ackingId, setAckingId] = useState<number | null>(null)

  const { data, isLoading, isError } = useQuery({
    queryKey: ['alerts'],
    queryFn: endpoints.alerts,
    refetchInterval: 30_000,
  })

  const active = (data?.alerts ?? []).filter((a) => !a.acknowledged)
  const acknowledged = (data?.alerts ?? []).filter((a) => a.acknowledged)
  const liveAlerts = data?.live_alerts ?? []

  async function handleAck(id: number) {
    setAckingId(id)
    try {
      await endpoints.acknowledgeAlert(id)
      queryClient.invalidateQueries({ queryKey: ['alerts'] })
    } catch (e) {
      console.error('Failed to acknowledge alert', e)
    } finally {
      setAckingId(null)
    }
  }

  if (isError) {
    return <ErrorCard message="Failed to load alerts. Check your connection." />
  }

  return (
    <div className="space-y-6 max-w-3xl">
      {/* Live Alerts */}
      {liveAlerts.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Zap className="h-3.5 w-3.5 text-amber-400" />
            <h2 className="text-sm font-medium text-zinc-400 uppercase tracking-wider">
              Live Alerts
            </h2>
          </div>
          {liveAlerts.map((alert, i) => (
            <LiveAlertCard key={i} alert={alert} />
          ))}
        </div>
      )}

      {/* Active Alerts */}
      <div className="space-y-3">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-medium text-zinc-400 uppercase tracking-wider">
            Active Alerts
          </h2>
          {!isLoading && (
            <span className="font-mono text-xs tabular-nums text-zinc-500">
              {active.length} active
            </span>
          )}
        </div>

        {isLoading ? (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-20 bg-zinc-800 rounded-lg animate-pulse" />
            ))}
          </div>
        ) : active.length === 0 ? (
          <Card>
            <div className="flex items-center gap-3 text-emerald-400">
              <CheckCircle2 className="h-5 w-5" />
              <span className="text-sm">No active alerts. All systems nominal.</span>
            </div>
          </Card>
        ) : (
          <div className="space-y-3">
            {active.map((alert) => (
              <AlertCard
                key={alert.id}
                alert={alert}
                onAck={ackingId === alert.id ? () => {} : handleAck}
              />
            ))}
          </div>
        )}
      </div>

      {/* Acknowledged Alerts toggle */}
      {acknowledged.length > 0 && (
        <div className="space-y-3">
          <button
            onClick={() => setShowAcknowledged((v) => !v)}
            className="flex items-center gap-2 text-sm text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            {showAcknowledged ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
            {showAcknowledged ? 'Hide' : 'Show'} acknowledged ({acknowledged.length})
          </button>

          {showAcknowledged && (
            <div className="space-y-3">
              {acknowledged.map((alert) => (
                <AlertCard key={alert.id} alert={alert} onAck={() => {}} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
