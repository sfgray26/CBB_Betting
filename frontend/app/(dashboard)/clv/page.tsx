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
import type { BetLog, CLVAnalysis } from '@/lib/types'

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

interface ConfidenceRow {
  confidence: string
  count: number
  avg_clv: number
  win_rate: number
}

export default function ClvPage() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['clv-analysis'],
    queryFn: endpoints.clvAnalysis,
  })

  const distData =
    data?.clv_distribution.map((d) => ({
      bucket: d.bucket,
      count: d.count,
      avg_clv: d.avg_clv,
      isPositive: d.avg_clv >= 0,
    })) ?? []

  const confColumns: Column<ConfidenceRow>[] = [
    {
      key: 'confidence',
      header: 'Confidence',
      accessor: (r) => <span className="capitalize">{r.confidence}</span>,
      sortValue: (r) => r.confidence,
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
      key: 'avg_clv',
      header: 'Avg CLV',
      accessor: (r) => (
        <span
          className={`font-mono tabular-nums ${r.avg_clv >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}
        >
          {r.avg_clv >= 0 ? '+' : ''}
          {r.avg_clv.toFixed(2)} pts
        </span>
      ),
      sortValue: (r) => r.avg_clv,
      className: 'text-right',
      headerClassName: 'text-right',
    },
    {
      key: 'win_rate',
      header: 'Win Rate',
      accessor: (r) => (
        <span className="font-mono tabular-nums">{(r.win_rate * 100).toFixed(1)}%</span>
      ),
      sortValue: (r) => r.win_rate,
      className: 'text-right',
      headerClassName: 'text-right',
    },
  ]

  const clvBetColumns: Column<BetLog>[] = [
    {
      key: 'pick',
      header: 'Pick',
      accessor: (r) => <span className="text-zinc-200">{r.pick}</span>,
    },
    {
      key: 'clv_points',
      header: 'CLV',
      accessor: (r) => (
        <span
          className={`font-mono tabular-nums ${(r.clv_points ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}
        >
          {(r.clv_points ?? 0) >= 0 ? '+' : ''}
          {(r.clv_points ?? 0).toFixed(2)}
        </span>
      ),
      sortValue: (r) => r.clv_points ?? 0,
      className: 'text-right',
      headerClassName: 'text-right',
    },
    {
      key: 'clv_grade',
      header: 'Grade',
      accessor: (r) => (
        <span className="font-mono tabular-nums text-zinc-300">{r.clv_grade ?? '-'}</span>
      ),
      className: 'text-right',
      headerClassName: 'text-right',
    },
    {
      key: 'outcome',
      header: 'Result',
      accessor: (r) => {
        const pl = r.profit_loss_units
        if (pl === undefined || pl === null) return <span className="text-zinc-500">-</span>
        return (
          <span
            className={`font-mono tabular-nums ${pl > 0 ? 'text-emerald-400' : pl < 0 ? 'text-rose-400' : 'text-zinc-400'}`}
          >
            {pl > 0 ? '+' : ''}
            {pl.toFixed(2)}u
          </span>
        )
      },
      sortValue: (r) => r.profit_loss_units ?? 0,
      className: 'text-right',
      headerClassName: 'text-right',
    },
  ]

  if (isError) {
    return <ErrorCard message="Failed to load CLV analysis. Check your connection." />
  }

  return (
    <div className="space-y-6 max-w-7xl">
      {/* KPI Row */}
      <div className="grid grid-cols-2 gap-4">
        <KpiCard
          title="Avg CLV"
          value={
            data
              ? `${data.avg_clv_points >= 0 ? '+' : ''}${data.avg_clv_points.toFixed(2)}`
              : '--'
          }
          unit="pts"
          loading={isLoading}
          trend={data ? (data.avg_clv_points > 0 ? 'up' : data.avg_clv_points < 0 ? 'down' : 'neutral') : 'neutral'}
        />
        <KpiCard
          title="Positive CLV %"
          value={data ? `${data.pct_positive_clv.toFixed(1)}` : '--'}
          unit="%"
          loading={isLoading}
          trend={data ? (data.pct_positive_clv >= 55 ? 'up' : data.pct_positive_clv < 45 ? 'down' : 'neutral') : 'neutral'}
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
        ) : distData.length === 0 ? (
          <div className="px-6 pb-6 py-12 text-center text-zinc-500 text-sm">
            No CLV data available yet.
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
                  formatter={(value: number) => [value, 'Count']}
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
            data={data?.by_confidence ?? []}
            keyExtractor={(r) => r.confidence}
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
              data={data?.top_clv ?? []}
              keyExtractor={(r) => r.id}
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
              data={data?.bottom_clv ?? []}
              keyExtractor={(r) => r.id}
              emptyMessage="No bottom CLV bets yet."
            />
          )}
        </Card>
      </div>
    </div>
  )
}
