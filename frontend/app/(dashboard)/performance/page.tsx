'use client'

import { useQuery } from '@tanstack/react-query'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
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

// roi/win_rate come back as decimals (0.043) — multiply by 100 for display
const pct = (v: number) => (v * 100).toFixed(1)
const signed = (v: number, decimals = 1) => `${v >= 0 ? '+' : ''}${v.toFixed(decimals)}`

interface ByTypeRow { type: string; bets: number; win_rate: number; roi: number }
interface ByEdgeRow { bucket: string; bets: number; win_rate: number; roi: number }

export default function PerformancePage() {
  const {
    data: summary, isLoading: summaryLoading, isError: summaryError,
  } = useQuery({
    queryKey: ['performance-summary'],
    queryFn: endpoints.performanceSummary,
  })

  const {
    data: timeline, isLoading: timelineLoading, isError: timelineError,
  } = useQuery({
    queryKey: ['performance-timeline'],
    queryFn: () => endpoints.performanceTimeline(30),
  })

  const overall = summary?.overall
  const hasData = !!overall

  const byTypeRows: ByTypeRow[] = Object.entries(summary?.by_bet_type ?? {}).map(([type, v]) => ({
    type, bets: v.bets, win_rate: v.win_rate, roi: v.roi,
  }))

  const byEdgeRows: ByEdgeRow[] = Object.entries(summary?.by_edge_bucket ?? {}).map(([bucket, v]) => ({
    bucket, bets: v.bets, win_rate: v.win_rate, roi: v.roi,
  }))

  const typeColumns: Column<ByTypeRow>[] = [
    { key: 'type', header: 'Type', accessor: (r) => <span className="capitalize">{r.type}</span>, sortValue: (r) => r.type },
    { key: 'bets', header: 'Bets', accessor: (r) => <span className="font-mono tabular-nums">{r.bets}</span>, sortValue: (r) => r.bets, className: 'text-right', headerClassName: 'text-right' },
    { key: 'win_rate', header: 'Win Rate', accessor: (r) => <span className="font-mono tabular-nums">{pct(r.win_rate)}%</span>, sortValue: (r) => r.win_rate, className: 'text-right', headerClassName: 'text-right' },
    { key: 'roi', header: 'ROI', accessor: (r) => <span className={`font-mono tabular-nums ${r.roi >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>{signed(r.roi * 100)}%</span>, sortValue: (r) => r.roi, className: 'text-right', headerClassName: 'text-right' },
  ]

  const edgeColumns: Column<ByEdgeRow>[] = [
    { key: 'bucket', header: 'Edge Tier', accessor: (r) => <span>{r.bucket}</span>, sortValue: (r) => r.bucket },
    { key: 'bets', header: 'Bets', accessor: (r) => <span className="font-mono tabular-nums">{r.bets}</span>, sortValue: (r) => r.bets, className: 'text-right', headerClassName: 'text-right' },
    { key: 'win_rate', header: 'Win Rate', accessor: (r) => <span className="font-mono tabular-nums">{pct(r.win_rate)}%</span>, sortValue: (r) => r.win_rate, className: 'text-right', headerClassName: 'text-right' },
    { key: 'roi', header: 'ROI', accessor: (r) => <span className={`font-mono tabular-nums ${r.roi >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>{signed(r.roi * 100)}%</span>, sortValue: (r) => r.roi, className: 'text-right', headerClassName: 'text-right' },
  ]

  const chartData = timeline?.timeline.map((p) => ({
    date: format(parseISO(p.date), 'MMM d'),
    units: p.cumulative_units,
  })) ?? []

  const rw = summary?.rolling_windows

  return (
    <div className="space-y-6 max-w-7xl">

      {/* KPI Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {summaryError ? (
          <div className="col-span-4"><ErrorCard message="Failed to load performance data." /></div>
        ) : (
          <>
            <KpiCard
              title="Total ROI"
              value={hasData ? `${signed(overall!.roi * 100)}` : '--'}
              unit="%"
              trend={hasData ? (overall!.roi > 0 ? 'up' : overall!.roi < 0 ? 'down' : 'neutral') : 'neutral'}
              loading={summaryLoading}
            />
            <KpiCard
              title="Win Rate"
              value={hasData ? pct(overall!.win_rate) : '--'}
              unit="%"
              trend={hasData ? (overall!.win_rate >= 0.53 ? 'up' : overall!.win_rate < 0.47 ? 'down' : 'neutral') : 'neutral'}
              loading={summaryLoading}
            />
            <KpiCard
              title="Avg CLV"
              value={hasData && overall!.mean_clv != null ? `${signed(overall!.mean_clv * 100, 2)}` : '--'}
              unit="pts"
              trend={hasData && overall!.mean_clv != null ? (overall!.mean_clv > 0 ? 'up' : overall!.mean_clv < 0 ? 'down' : 'neutral') : 'neutral'}
              loading={summaryLoading}
            />
            <KpiCard
              title="Total P&L"
              value={hasData ? `${signed(overall!.total_profit_dollars, 0)}` : '--'}
              unit="$"
              trend={hasData ? (overall!.total_profit_dollars > 0 ? 'up' : overall!.total_profit_dollars < 0 ? 'down' : 'neutral') : 'neutral'}
              loading={summaryLoading}
            />
          </>
        )}
      </div>

      {/* No data notice */}
      {!summaryLoading && !summaryError && !hasData && (
        <div className="rounded-lg border border-zinc-700 bg-zinc-800/50 p-6 text-zinc-400 text-sm text-center">
          {summary?.message ?? 'No settled bets yet. P&L will appear here once bets are settled.'}
        </div>
      )}

      {/* Bankroll Curve */}
      <Card className="p-0">
        <CardHeader className="px-6 pt-6 pb-0 mb-4">
          <CardTitle>Bankroll Curve (30 Days)</CardTitle>
        </CardHeader>
        {timelineError ? (
          <div className="px-6 pb-6"><ErrorCard message="Failed to load timeline data." /></div>
        ) : timelineLoading ? (
          <div className="px-6 pb-6"><div className="h-64 bg-zinc-800 rounded animate-pulse" /></div>
        ) : chartData.length === 0 ? (
          <div className="px-6 pb-10 text-zinc-500 text-sm text-center">No timeline data yet.</div>
        ) : (
          <div className="px-2 pb-6">
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorUnitsPos" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#34d399" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="#34d399" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis dataKey="date" tick={{ fill: '#71717a', fontSize: 11 }} axisLine={{ stroke: '#3f3f46' }} tickLine={false} />
                <YAxis tick={{ fill: '#71717a', fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}u`} />
                <Tooltip contentStyle={darkTooltipStyle} formatter={(v: number) => [`${v >= 0 ? '+' : ''}${v.toFixed(2)}u`, 'Cumulative P&L']} />
                <ReferenceLine y={0} stroke="#3f3f46" strokeDasharray="4 2" />
                <Area type="monotone" dataKey="units" stroke="#34d399" strokeWidth={2} fill="url(#colorUnitsPos)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}
      </Card>

      {/* Rolling Windows */}
      {rw && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {([['Last 10 Bets', rw.last_10], ['Last 50 Bets', rw.last_50], ['Last 100 Bets', rw.last_100]] as const).map(([label, w]) => (
            <Card key={label}>
              <CardHeader><CardTitle>{label}</CardTitle></CardHeader>
              <div className="flex gap-6">
                <div>
                  <div className="text-xs text-zinc-500 mb-1">Bets</div>
                  <div className="font-mono text-lg text-zinc-100 tabular-nums">{w.bets}</div>
                </div>
                <div>
                  <div className="text-xs text-zinc-500 mb-1">Win Rate</div>
                  <div className="font-mono text-lg text-zinc-100 tabular-nums">{pct(w.win_rate)}%</div>
                </div>
                <div>
                  <div className="text-xs text-zinc-500 mb-1">Avg CLV</div>
                  <div className={`font-mono text-lg tabular-nums ${(w.mean_clv ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {w.mean_clv != null ? `${signed(w.mean_clv * 100, 2)}pts` : '--'}
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Tables */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <Card className="p-0">
          <CardHeader className="px-6 pt-6 pb-0 mb-2"><CardTitle>Performance by Type</CardTitle></CardHeader>
          {summaryLoading ? (
            <div className="p-6 space-y-2">{[1,2,3].map(i => <div key={i} className="h-10 bg-zinc-800 rounded animate-pulse" />)}</div>
          ) : (
            <DataTable columns={typeColumns} data={byTypeRows} keyExtractor={(r) => r.type} emptyMessage="No type data yet." />
          )}
        </Card>
        <Card className="p-0">
          <CardHeader className="px-6 pt-6 pb-0 mb-2"><CardTitle>Performance by Edge Tier</CardTitle></CardHeader>
          {summaryLoading ? (
            <div className="p-6 space-y-2">{[1,2,3].map(i => <div key={i} className="h-10 bg-zinc-800 rounded animate-pulse" />)}</div>
          ) : (
            <DataTable columns={edgeColumns} data={byEdgeRows} keyExtractor={(r) => r.bucket} emptyMessage="No edge tier data yet." />
          )}
        </Card>
      </div>
    </div>
  )
}
