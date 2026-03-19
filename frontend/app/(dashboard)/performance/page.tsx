'use client'

import { useQuery } from '@tanstack/react-query'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { format, parseISO } from 'date-fns'
import { endpoints } from '@/lib/api'
import { KpiCard } from '@/components/ui/kpi-card'
import { Card, CardHeader, CardTitle } from '@/components/ui/card'
import { DataTable, type Column } from '@/components/ui/data-table'

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

interface ByTypeRow {
  type: string
  bets: number
  wins: number
  roi: number
}

interface ByEdgeRow {
  bucket: string
  bets: number
  wins: number
  roi: number
}

export default function PerformancePage() {
  const {
    data: summary,
    isLoading: summaryLoading,
    isError: summaryError,
  } = useQuery({
    queryKey: ['performance-summary'],
    queryFn: endpoints.performanceSummary,
  })

  const {
    data: timeline,
    isLoading: timelineLoading,
    isError: timelineError,
  } = useQuery({
    queryKey: ['performance-timeline'],
    queryFn: () => endpoints.performanceTimeline(30),
  })

  const byTypeRows: ByTypeRow[] = Object.entries(summary?.by_type ?? {}).map(([type, v]) => ({
    type,
    bets: v.bets,
    wins: v.wins,
    roi: v.roi,
  }))

  const byEdgeRows: ByEdgeRow[] = Object.entries(summary?.by_edge_bucket ?? {}).map(([bucket, v]) => ({
    bucket,
    bets: v.bets,
    wins: v.wins,
    roi: v.roi,
  }))

  const typeColumns: Column<ByTypeRow>[] = [
    {
      key: 'type',
      header: 'Type',
      accessor: (r) => <span className="capitalize">{r.type}</span>,
      sortValue: (r) => r.type,
    },
    {
      key: 'bets',
      header: 'Bets',
      accessor: (r) => <span className="font-mono tabular-nums">{r.bets}</span>,
      sortValue: (r) => r.bets,
      className: 'text-right',
      headerClassName: 'text-right',
    },
    {
      key: 'win_rate',
      header: 'Win Rate',
      accessor: (r) => (
        <span className="font-mono tabular-nums">
          {r.bets > 0 ? ((r.wins / r.bets) * 100).toFixed(1) : '0.0'}%
        </span>
      ),
      sortValue: (r) => (r.bets > 0 ? r.wins / r.bets : 0),
      className: 'text-right',
      headerClassName: 'text-right',
    },
    {
      key: 'roi',
      header: 'ROI',
      accessor: (r) => (
        <span
          className={`font-mono tabular-nums ${r.roi >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}
        >
          {r.roi >= 0 ? '+' : ''}
          {r.roi.toFixed(1)}%
        </span>
      ),
      sortValue: (r) => r.roi,
      className: 'text-right',
      headerClassName: 'text-right',
    },
  ]

  const edgeColumns: Column<ByEdgeRow>[] = [
    {
      key: 'bucket',
      header: 'Edge Tier',
      accessor: (r) => <span>{r.bucket}</span>,
      sortValue: (r) => r.bucket,
    },
    {
      key: 'bets',
      header: 'Bets',
      accessor: (r) => <span className="font-mono tabular-nums">{r.bets}</span>,
      sortValue: (r) => r.bets,
      className: 'text-right',
      headerClassName: 'text-right',
    },
    {
      key: 'win_rate',
      header: 'Win Rate',
      accessor: (r) => (
        <span className="font-mono tabular-nums">
          {r.bets > 0 ? ((r.wins / r.bets) * 100).toFixed(1) : '0.0'}%
        </span>
      ),
      sortValue: (r) => (r.bets > 0 ? r.wins / r.bets : 0),
      className: 'text-right',
      headerClassName: 'text-right',
    },
    {
      key: 'roi',
      header: 'ROI',
      accessor: (r) => (
        <span
          className={`font-mono tabular-nums ${r.roi >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}
        >
          {r.roi >= 0 ? '+' : ''}
          {r.roi.toFixed(1)}%
        </span>
      ),
      sortValue: (r) => r.roi,
      className: 'text-right',
      headerClassName: 'text-right',
    },
  ]

  const chartData = timeline?.timeline.map((p) => ({
    date: format(parseISO(p.date), 'MMM d'),
    units: p.cumulative_units,
    daily: p.daily_units,
  })) ?? []

  return (
    <div className="space-y-6 max-w-7xl">
      {/* KPI Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {summaryError ? (
          <div className="col-span-4">
            <ErrorCard message="Failed to load performance data. Check your connection." />
          </div>
        ) : (
          <>
            <KpiCard
              title="Total ROI"
              value={summary ? `${summary.roi >= 0 ? '+' : ''}${summary.roi.toFixed(1)}` : '--'}
              unit="%"
              delta={
                summary
                  ? summary.rolling_7d.roi - summary.rolling_30d.roi
                  : undefined
              }
              deltaLabel="vs 30d"
              trend={
                summary
                  ? summary.roi > 0
                    ? 'up'
                    : summary.roi < 0
                      ? 'down'
                      : 'neutral'
                  : 'neutral'
              }
              loading={summaryLoading}
            />
            <KpiCard
              title="Win Rate"
              value={summary ? `${(summary.win_rate * 100).toFixed(1)}` : '--'}
              unit="%"
              loading={summaryLoading}
              trend={
                summary
                  ? summary.win_rate >= 0.53
                    ? 'up'
                    : summary.win_rate < 0.47
                      ? 'down'
                      : 'neutral'
                  : 'neutral'
              }
            />
            <KpiCard
              title="Avg CLV"
              value={summary ? `${summary.avg_clv >= 0 ? '+' : ''}${summary.avg_clv.toFixed(2)}` : '--'}
              unit="pts"
              loading={summaryLoading}
              trend={
                summary
                  ? summary.avg_clv > 0
                    ? 'up'
                    : summary.avg_clv < 0
                      ? 'down'
                      : 'neutral'
                  : 'neutral'
              }
            />
            <KpiCard
              title="Total P&L"
              value={
                summary
                  ? `${summary.total_profit_units >= 0 ? '+' : ''}${summary.total_profit_units.toFixed(2)}`
                  : '--'
              }
              unit="u"
              loading={summaryLoading}
              trend={
                summary
                  ? summary.total_profit_units > 0
                    ? 'up'
                    : summary.total_profit_units < 0
                      ? 'down'
                      : 'neutral'
                  : 'neutral'
              }
            />
          </>
        )}
      </div>

      {/* Bankroll Curve */}
      <Card className="p-0">
        <CardHeader className="px-6 pt-6 pb-0 mb-4">
          <CardTitle>Bankroll Curve (30 Days)</CardTitle>
        </CardHeader>
        {timelineError ? (
          <div className="px-6 pb-6">
            <ErrorCard message="Failed to load timeline data." />
          </div>
        ) : timelineLoading ? (
          <div className="px-6 pb-6">
            <div className="h-64 bg-zinc-800 rounded animate-pulse" />
          </div>
        ) : chartData.length === 0 ? (
          <div className="px-6 pb-6 text-zinc-500 text-sm py-16 text-center">
            No timeline data available yet.
          </div>
        ) : (
          <div className="px-2 pb-6">
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorUnitsPos" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#34d399" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="#34d399" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorUnitsNeg" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f43f5e" stopOpacity={0} />
                    <stop offset="95%" stopColor="#f43f5e" stopOpacity={0.2} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis
                  dataKey="date"
                  tick={{ fill: '#71717a', fontSize: 11 }}
                  axisLine={{ stroke: '#3f3f46' }}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: '#71717a', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}u`}
                />
                <Tooltip
                  contentStyle={darkTooltipStyle}
                  formatter={(value: number) => [
                    `${value >= 0 ? '+' : ''}${value.toFixed(2)}u`,
                    'Cumulative P&L',
                  ]}
                />
                <ReferenceLine y={0} stroke="#3f3f46" strokeDasharray="4 2" />
                <Area
                  type="monotone"
                  dataKey="units"
                  stroke="#34d399"
                  strokeWidth={2}
                  fill="url(#colorUnitsPos)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}
      </Card>

      {/* Rolling Windows */}
      {summary && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card>
            <CardHeader>
              <CardTitle>Rolling 7-Day</CardTitle>
            </CardHeader>
            <div className="flex gap-6">
              <div>
                <div className="text-xs text-zinc-500 mb-1">Bets</div>
                <div className="font-mono text-lg text-zinc-100 tabular-nums">
                  {summary.rolling_7d.bets}
                </div>
              </div>
              <div>
                <div className="text-xs text-zinc-500 mb-1">Win Rate</div>
                <div className="font-mono text-lg text-zinc-100 tabular-nums">
                  {summary.rolling_7d.bets > 0
                    ? ((summary.rolling_7d.wins / summary.rolling_7d.bets) * 100).toFixed(1)
                    : '0.0'}
                  %
                </div>
              </div>
              <div>
                <div className="text-xs text-zinc-500 mb-1">ROI</div>
                <div
                  className={`font-mono text-lg tabular-nums ${summary.rolling_7d.roi >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}
                >
                  {summary.rolling_7d.roi >= 0 ? '+' : ''}
                  {summary.rolling_7d.roi.toFixed(1)}%
                </div>
              </div>
            </div>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Rolling 30-Day</CardTitle>
            </CardHeader>
            <div className="flex gap-6">
              <div>
                <div className="text-xs text-zinc-500 mb-1">Bets</div>
                <div className="font-mono text-lg text-zinc-100 tabular-nums">
                  {summary.rolling_30d.bets}
                </div>
              </div>
              <div>
                <div className="text-xs text-zinc-500 mb-1">Win Rate</div>
                <div className="font-mono text-lg text-zinc-100 tabular-nums">
                  {summary.rolling_30d.bets > 0
                    ? ((summary.rolling_30d.wins / summary.rolling_30d.bets) * 100).toFixed(1)
                    : '0.0'}
                  %
                </div>
              </div>
              <div>
                <div className="text-xs text-zinc-500 mb-1">ROI</div>
                <div
                  className={`font-mono text-lg tabular-nums ${summary.rolling_30d.roi >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}
                >
                  {summary.rolling_30d.roi >= 0 ? '+' : ''}
                  {summary.rolling_30d.roi.toFixed(1)}%
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Tables */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <Card className="p-0">
          <CardHeader className="px-6 pt-6 pb-0 mb-2">
            <CardTitle>Performance by Type</CardTitle>
          </CardHeader>
          {summaryLoading ? (
            <div className="p-6 space-y-2">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-10 bg-zinc-800 rounded animate-pulse" />
              ))}
            </div>
          ) : (
            <DataTable
              columns={typeColumns}
              data={byTypeRows}
              keyExtractor={(r) => r.type}
              emptyMessage="No type data available."
            />
          )}
        </Card>

        <Card className="p-0">
          <CardHeader className="px-6 pt-6 pb-0 mb-2">
            <CardTitle>Performance by Edge Tier</CardTitle>
          </CardHeader>
          {summaryLoading ? (
            <div className="p-6 space-y-2">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-10 bg-zinc-800 rounded animate-pulse" />
              ))}
            </div>
          ) : (
            <DataTable
              columns={edgeColumns}
              data={byEdgeRows}
              keyExtractor={(r) => r.bucket}
              emptyMessage="No edge tier data available."
            />
          )}
        </Card>
      </div>
    </div>
  )
}
