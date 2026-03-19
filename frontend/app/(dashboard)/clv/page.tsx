'use client'

import { useQuery } from '@tanstack/react-query'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import { endpoints } from '@/lib/api'
import { KpiCard } from '@/components/ui/kpi-card'
import { Card, CardHeader, CardTitle } from '@/components/ui/card'
import { DataTable, type Column } from '@/components/ui/data-table'
import type { ClvBetEntry } from '@/lib/types'

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

// mean_clv / clv_prob come back as decimals (0.012) — multiply by 100 for display
const pct = (v: number, decimals = 2) => (v * 100).toFixed(decimals)
const signed = (v: number | null | undefined, decimals = 2) => {
  if (v == null) return '—'
  return `${v >= 0 ? '+' : ''}${v.toFixed(decimals)}`
}

interface ConfidenceRow {
  tier: string
  count: number
  mean_clv: number | null
}

export default function ClvPage() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['clv-analysis'],
    queryFn: endpoints.clvAnalysis,
  })

  const hasData = data && data.bets_with_clv > 0

  // distribution object → sorted array for chart
  const distData = data?.distribution
    ? [
        { bucket: 'Strong −', count: data.distribution.strong_negative, isPositive: false },
        { bucket: 'Negative', count: data.distribution.negative, isPositive: false },
        { bucket: 'Neutral', count: data.distribution.neutral, isPositive: true },
        { bucket: 'Positive', count: data.distribution.positive, isPositive: true },
        { bucket: 'Strong +', count: data.distribution.strong_positive, isPositive: true },
      ]
    : []

  // clv_by_confidence object → array for table
  const confRows: ConfidenceRow[] = Object.entries(data?.clv_by_confidence ?? {}).map(
    ([tier, v]) => ({ tier, count: v.count, mean_clv: v.mean_clv }),
  )

  const confColumns: Column<ConfidenceRow>[] = [
    {
      key: 'tier',
      header: 'Confidence Tier',
      accessor: (r) => <span className="capitalize">{r.tier.replace('_', ' ')}</span>,
      sortValue: (r) => r.tier,
    },
    {
      key: 'count',
      header: 'Bets',
      accessor: (r) => <span className="font-mono tabular-nums">{r.count}</span>,
      sortValue: (r) => r.count,
      className: 'text-right',
      headerClassName: 'text-right',
    },
    {
      key: 'mean_clv',
      header: 'Avg CLV',
      accessor: (r) => {
        if (r.mean_clv == null) return <span className="text-zinc-500">—</span>
        return (
          <span
            className={`font-mono tabular-nums ${r.mean_clv >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}
          >
            {signed(r.mean_clv * 100)}%
          </span>
        )
      },
      sortValue: (r) => r.mean_clv ?? 0,
      className: 'text-right',
      headerClassName: 'text-right',
    },
  ]

  const clvBetColumns: Column<ClvBetEntry>[] = [
    {
      key: 'pick',
      header: 'Pick',
      accessor: (r) => <span className="text-zinc-200">{r.pick}</span>,
    },
    {
      key: 'clv_prob',
      header: 'CLV %',
      accessor: (r) => {
        if (r.clv_prob == null) return <span className="text-zinc-500">—</span>
        return (
          <span
            className={`font-mono tabular-nums ${r.clv_prob >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}
          >
            {signed(r.clv_prob * 100)}%
          </span>
        )
      },
      sortValue: (r) => r.clv_prob ?? 0,
      className: 'text-right',
      headerClassName: 'text-right',
    },
    {
      key: 'clv_points',
      header: 'CLV pts',
      accessor: (r) => {
        const pts = r.clv_points
        if (pts == null) return <span className="text-zinc-500">—</span>
        return (
          <span className={`font-mono tabular-nums ${pts >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
            {signed(pts)}
          </span>
        )
      },
      sortValue: (r) => r.clv_points ?? 0,
      className: 'text-right',
      headerClassName: 'text-right',
    },
    {
      key: 'outcome',
      header: 'Result',
      accessor: (r) => {
        if (r.outcome === 1) return <span className="text-emerald-400 font-mono text-xs">W</span>
        if (r.outcome === 0) return <span className="text-rose-400 font-mono text-xs">L</span>
        if (r.outcome === -1) return <span className="text-zinc-400 font-mono text-xs">P</span>
        return <span className="text-zinc-500 font-mono text-xs">—</span>
      },
      sortValue: (r) => r.outcome ?? -2,
      className: 'text-right',
      headerClassName: 'text-right',
    },
  ]

  if (isError) {
    return <ErrorCard message="Failed to load CLV analysis. Check your connection." />
  }

  return (
    <div className="space-y-6 max-w-7xl">
      {/* Empty state */}
      {!isLoading && data && !hasData && (
        <div className="rounded-lg border border-zinc-700 bg-zinc-800/50 p-6 text-zinc-400 text-sm text-center">
          {data.message ?? 'No CLV data yet. Requires closing lines to be captured.'}
        </div>
      )}

      {/* KPI Row */}
      <div className="grid grid-cols-2 gap-4">
        <KpiCard
          title="Avg CLV"
          value={hasData && data?.mean_clv != null ? `${signed(data.mean_clv * 100)}` : '--'}
          unit="%"
          loading={isLoading}
          trend={
            hasData && data?.mean_clv != null
              ? data.mean_clv > 0
                ? 'up'
                : data.mean_clv < 0
                  ? 'down'
                  : 'neutral'
              : 'neutral'
          }
        />
        <KpiCard
          title="Positive CLV"
          value={hasData && data?.positive_clv_rate != null ? pct(data.positive_clv_rate, 1) : '--'}
          unit="%"
          loading={isLoading}
          trend={
            hasData && data?.positive_clv_rate != null
              ? data.positive_clv_rate >= 0.55
                ? 'up'
                : data.positive_clv_rate < 0.45
                  ? 'down'
                  : 'neutral'
              : 'neutral'
          }
        />
      </div>

      {/* CLV Distribution Chart */}
      <Card className="p-0">
        <CardHeader className="px-6 pt-6 pb-0 mb-4">
          <CardTitle>CLV Distribution</CardTitle>
        </CardHeader>
        {isLoading ? (
          <div className="px-6 pb-6">
            <div className="h-56 bg-zinc-800 rounded animate-pulse" />
          </div>
        ) : distData.length === 0 || !hasData ? (
          <div className="px-6 pb-6 py-12 text-center text-zinc-500 text-sm">
            No CLV distribution data available yet.
          </div>
        ) : (
          <div className="px-2 pb-6">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={distData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis
                  dataKey="bucket"
                  tick={{ fill: '#71717a', fontSize: 11 }}
                  axisLine={{ stroke: '#3f3f46' }}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: '#71717a', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={darkTooltipStyle}
                  formatter={(value: number) => [value, 'Bets']}
                />
                <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                  {distData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.isPositive ? '#34d399' : '#f43f5e'}
                      fillOpacity={0.8}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </Card>

      {/* By Confidence Table */}
      <Card className="p-0">
        <CardHeader className="px-6 pt-6 pb-0 mb-2">
          <CardTitle>CLV by Confidence Tier</CardTitle>
        </CardHeader>
        {isLoading ? (
          <div className="p-6 space-y-2">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-10 bg-zinc-800 rounded animate-pulse" />
            ))}
          </div>
        ) : (
          <DataTable
            columns={confColumns}
            data={confRows}
            keyExtractor={(r) => r.tier}
            emptyMessage="No confidence tier data available."
          />
        )}
      </Card>

      {/* Top / Bottom CLV Bets */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <Card className="p-0">
          <CardHeader className="px-6 pt-6 pb-0 mb-2">
            <CardTitle>Top 10 CLV Bets</CardTitle>
          </CardHeader>
          {isLoading ? (
            <div className="p-6 space-y-2">
              {[1, 2, 3, 4, 5].map((i) => (
                <div key={i} className="h-10 bg-zinc-800 rounded animate-pulse" />
              ))}
            </div>
          ) : (
            <DataTable
              columns={clvBetColumns}
              data={data?.top_10_clv ?? []}
              keyExtractor={(r) => r.bet_id}
              emptyMessage="No top CLV bets yet."
            />
          )}
        </Card>

        <Card className="p-0">
          <CardHeader className="px-6 pt-6 pb-0 mb-2">
            <CardTitle>Bottom 10 CLV Bets</CardTitle>
          </CardHeader>
          {isLoading ? (
            <div className="p-6 space-y-2">
              {[1, 2, 3, 4, 5].map((i) => (
                <div key={i} className="h-10 bg-zinc-800 rounded animate-pulse" />
              ))}
            </div>
          ) : (
            <DataTable
              columns={clvBetColumns}
              data={data?.bottom_10_clv ?? []}
              keyExtractor={(r) => r.bet_id}
              emptyMessage="No bottom CLV bets yet."
            />
          )}
        </Card>
      </div>
    </div>
  )
}
