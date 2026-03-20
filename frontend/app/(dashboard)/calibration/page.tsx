'use client'

import { useQuery } from '@tanstack/react-query'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import { endpoints } from '@/lib/api'
import { KpiCard } from '@/components/ui/kpi-card'
import { Card, CardHeader, CardTitle } from '@/components/ui/card'
import { DataTable, type Column } from '@/components/ui/data-table'
import type { CalibrationBucket } from '@/lib/types'

function ErrorCard({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-6 text-rose-400 text-sm">
      {message}
    </div>
  )
}

const darkTooltipStyle = {
  backgroundColor: '#27272a',
  border: '1px solid #3f3f46',
  borderRadius: '8px',
  color: '#fafafa',
  fontSize: '12px',
}

export default function CalibrationPage() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['calibration'],
    queryFn: () => endpoints.calibration(90),
  })

  // Build calibration curve data: diagonal reference + actual points
  const curveData = (data?.calibration_buckets ?? []).map((b) => ({
    predicted: Math.round(b.predicted_prob * 100),
    actual: Math.round(b.actual_win_rate * 100),
    perfect: Math.round(b.predicted_prob * 100),
    count: b.count,
    label: b.bin,
  }))

  const binColumns: Column<CalibrationBucket>[] = [
    {
      key: 'bin',
      header: 'Predicted %',
      accessor: (r) => <span>{r.bin}</span>,
      sortValue: (r) => r.predicted_prob,
    },
    {
      key: 'actual',
      header: 'Actual Win Rate',
      accessor: (r) => (
        <span className="font-mono tabular-nums">
          {(r.actual_win_rate * 100).toFixed(1)}%
        </span>
      ),
      sortValue: (r) => r.actual_win_rate,
      className: 'text-right',
      headerClassName: 'text-right',
    },
    {
      key: 'count',
      header: 'Count',
      accessor: (r) => <span className="font-mono tabular-nums">{r.count}</span>,
      sortValue: (r) => r.count,
      className: 'text-right',
      headerClassName: 'text-right',
    },
    {
      key: 'delta',
      header: 'Over/Under',
      accessor: (r) => {
        const delta = r.actual_win_rate - r.predicted_prob
        const label = Math.abs(delta) < 0.02 ? 'On target' : delta > 0 ? 'Over' : 'Under'
        const color =
          Math.abs(delta) < 0.02
            ? 'text-zinc-400'
            : delta > 0
              ? 'text-emerald-400'
              : 'text-rose-400'
        return (
          <span className={`font-mono tabular-nums text-xs ${color}`}>
            {delta > 0 ? '+' : ''}
            {(delta * 100).toFixed(1)}% ({label})
          </span>
        )
      },
      sortValue: (r) => r.actual_win_rate - r.predicted_prob,
      className: 'text-right',
      headerClassName: 'text-right',
    },
  ]

  if (isError) {
    return <ErrorCard message="Failed to load calibration data. Check your connection." />
  }

  return (
    <div className="space-y-6 max-w-7xl">
      {/* KPI Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <KpiCard
          title="Brier Score"
          value={data?.brier_score != null ? data.brier_score.toFixed(4) : '--'}
          loading={isLoading}
          trend={
            data?.brier_score != null
              ? data.brier_score < 0.22
                ? 'up'
                : data.brier_score > 0.28
                  ? 'down'
                  : 'neutral'
              : 'neutral'
          }
          deltaLabel="lower is better"
        />
        <KpiCard
          title="Calibrated Bets"
          value={data ? (data.calibration_buckets?.reduce((s, b) => s + b.count, 0) ?? 0).toString() : '--'}
          loading={isLoading}
          trend="neutral"
        />
      </div>

      {/* Calibration Curve */}
      <Card className="p-0">
        <CardHeader className="px-6 pt-6 pb-0 mb-4">
          <CardTitle>Calibration Curve (90 Days)</CardTitle>
        </CardHeader>
        {isLoading ? (
          <div className="px-6 pb-6">
            <div className="h-64 bg-zinc-800 rounded animate-pulse" />
          </div>
        ) : curveData.length === 0 ? (
          <div className="px-6 pb-6 py-16 text-center text-zinc-500 text-sm">
            Not enough data to draw calibration curve.
          </div>
        ) : (
          <div className="px-2 pb-6">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={curveData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis
                  dataKey="predicted"
                  type="number"
                  domain={[40, 80]}
                  tickFormatter={(v: number) => `${v}%`}
                  tick={{ fill: '#71717a', fontSize: 11 }}
                  axisLine={{ stroke: '#3f3f46' }}
                  tickLine={false}
                  label={{
                    value: 'Predicted Probability',
                    position: 'insideBottom',
                    offset: -4,
                    fill: '#71717a',
                    fontSize: 11,
                  }}
                />
                <YAxis
                  domain={[40, 80]}
                  tickFormatter={(v: number) => `${v}%`}
                  tick={{ fill: '#71717a', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  label={{
                    value: 'Actual Win Rate',
                    angle: -90,
                    position: 'insideLeft',
                    offset: 10,
                    fill: '#71717a',
                    fontSize: 11,
                  }}
                />
                <Tooltip
                  contentStyle={darkTooltipStyle}
                  formatter={(value: number, name: string) => [
                    `${value.toFixed(1)}%`,
                    name === 'actual' ? 'Actual Win Rate' : 'Perfect Calibration',
                  ]}
                />
                <Legend
                  wrapperStyle={{ fontSize: '11px', color: '#71717a' }}
                />
                {/* Perfect calibration reference line (y=x) */}
                <Line
                  type="linear"
                  dataKey="perfect"
                  name="Perfect Calibration"
                  stroke="#3f3f46"
                  strokeDasharray="6 3"
                  strokeWidth={1}
                  dot={false}
                />
                {/* Actual calibration */}
                <Line
                  type="monotone"
                  dataKey="actual"
                  name="Actual Win Rate"
                  stroke="#fbbf24"
                  strokeWidth={2}
                  dot={{ fill: '#fbbf24', r: 4 }}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </Card>

      {/* Calibration Bins Table */}
      <Card className="p-0">
        <CardHeader className="px-6 pt-6 pb-0 mb-2">
          <CardTitle>Calibration Bins</CardTitle>
        </CardHeader>
        {isLoading ? (
          <div className="p-6 space-y-2">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="h-10 bg-zinc-800 rounded animate-pulse" />
            ))}
          </div>
        ) : (
          <DataTable
            columns={binColumns}
            data={data?.calibration_buckets ?? []}
            keyExtractor={(r) => r.bin}
            emptyMessage="No calibration bins available."
          />
        )}
      </Card>
    </div>
  )
}
