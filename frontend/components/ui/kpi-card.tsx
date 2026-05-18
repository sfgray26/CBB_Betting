import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { Card, CardTitle } from './card'
import { cn } from '@/lib/utils'

interface KpiCardProps {
  title: string
  value: string | number
  unit?: string
  delta?: number
  deltaLabel?: string
  trend?: 'up' | 'down' | 'neutral'
  loading?: boolean
  valueClassName?: string
}

export function KpiCard({
  title,
  value,
  unit,
  delta,
  deltaLabel,
  trend = 'neutral',
  loading = false,
  valueClassName,
}: KpiCardProps) {
  const trendColor =
    trend === 'up' ? 'text-green-600' : trend === 'down' ? 'text-red-600' : 'text-gray-500'

  const TrendIcon = trend === 'up' ? TrendingUp : trend === 'down' ? TrendingDown : Minus

  if (loading) {
    return (
      <Card>
        <CardTitle className="mb-3">{title}</CardTitle>
        <div className="animate-pulse space-y-2">
          <div className="h-8 bg-gray-200 rounded w-3/4" />
          <div className="h-4 bg-gray-200 rounded w-1/2" />
        </div>
      </Card>
    )
  }

  return (
    <Card>
      <CardTitle className="mb-3">{title}</CardTitle>
      <div className={cn('text-2xl font-semibold font-mono tabular-nums text-gray-900', valueClassName)}>
        {value}
        {unit && <span className="text-base text-gray-500 ml-1">{unit}</span>}
      </div>
      {(delta !== undefined || deltaLabel) && (
        <div className={cn('flex items-center gap-1 mt-2 text-xs', trendColor)}>
          <TrendIcon className="h-3 w-3" />
          {delta !== undefined && (
            <span className="font-mono tabular-nums">
              {delta > 0 ? '+' : ''}
              {delta.toFixed(1)}%
            </span>
          )}
          {deltaLabel && <span className="text-gray-500">{deltaLabel}</span>}
        </div>
      )}
    </Card>
  )
}
