'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Trophy, AlertTriangle, RefreshCw, ChevronDown, ChevronUp } from 'lucide-react'
import { endpoints } from '@/lib/api'
import { KpiCard } from '@/components/ui/kpi-card'
import { Card, CardHeader, CardTitle } from '@/components/ui/card'
import { DataTable, type Column } from '@/components/ui/data-table'
import { cn } from '@/lib/utils'
import type { TeamAdvancement } from '@/lib/types'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const REGION_COLORS: Record<string, string> = {
  east: 'text-sky-400',
  south: 'text-emerald-400',
  west: 'text-amber-400',
  midwest: 'text-rose-400',
}

const REGION_BADGE: Record<string, string> = {
  east: 'text-sky-400 bg-sky-400/10 border-sky-400/30',
  south: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/30',
  west: 'text-amber-400 bg-amber-400/10 border-amber-400/30',
  midwest: 'text-rose-400 bg-rose-400/10 border-rose-400/30',
}

function pctBar(value: number, max = 100) {
  const w = Math.min((value / max) * 100, 100)
  const color =
    value >= 30 ? '#34d399' :
    value >= 15 ? '#fbbf24' :
    value >= 5  ? '#71717a' :
    '#3f3f46'
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{ width: `${w}%`, backgroundColor: color }}
        />
      </div>
      <span className="font-mono tabular-nums text-xs w-10 text-right text-zinc-300">
        {value.toFixed(1)}%
      </span>
    </div>
  )
}

function SeedBadge({ seed }: { seed: number }) {
  const bg =
    seed === 1  ? 'bg-amber-400/20 text-amber-300' :
    seed <= 4   ? 'bg-emerald-400/15 text-emerald-400' :
    seed <= 8   ? 'bg-zinc-700/80 text-zinc-300' :
    seed <= 12  ? 'bg-rose-400/10 text-rose-400' :
    'bg-rose-500/20 text-rose-300'
  return (
    <span className={cn('text-xs font-bold px-1.5 py-0.5 rounded font-mono tabular-nums', bg)}>
      #{seed}
    </span>
  )
}

// ---------------------------------------------------------------------------
// Champion hero
// ---------------------------------------------------------------------------

function ChampionHero({
  champion,
  prob,
  nSims,
}: {
  champion: string | null
  prob: number
  nSims: number
}) {
  return (
    <div className="relative rounded-xl border border-amber-500/40 bg-gradient-to-br from-amber-500/10 via-zinc-900 to-zinc-900 p-6 overflow-hidden">
      <div className="absolute top-3 right-4 text-xs text-zinc-600 font-mono">
        {nSims.toLocaleString()} sims
      </div>
      <div className="flex items-center gap-3 mb-1">
        <Trophy className="h-5 w-5 text-amber-400" />
        <span className="text-xs font-semibold uppercase tracking-wider text-amber-500">
          Projected Champion
        </span>
      </div>
      {champion ? (
        <>
          <div className="text-3xl font-bold text-amber-300 mt-1">{champion}</div>
          <div className="text-sm text-zinc-400 mt-1 font-mono tabular-nums">
            {prob.toFixed(1)}% championship probability
          </div>
        </>
      ) : (
        <div className="text-zinc-500 text-sm mt-2">No simulation data available.</div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Final Four card
// ---------------------------------------------------------------------------

function FinalFourCard({
  teams,
  advancement,
}: {
  teams: string[]
  advancement: Record<string, TeamAdvancement>
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Projected Final Four</CardTitle>
      </CardHeader>
      <div className="grid grid-cols-2 gap-3 pt-1">
        {teams.map((team, i) => {
          const a = advancement[team]
          return (
            <div
              key={team}
              className={cn(
                'rounded-lg border p-3',
                i === 0
                  ? 'border-amber-500/30 bg-amber-500/5'
                  : 'border-zinc-700/50 bg-zinc-800/30',
              )}
            >
              <div className="text-sm font-semibold text-zinc-100 mb-1">{team}</div>
              {a && (
                <div className="flex gap-3 text-xs font-mono tabular-nums text-zinc-500">
                  <span>
                    F4: <span className={a.f4_pct >= 25 ? 'text-emerald-400' : 'text-zinc-400'}>
                      {a.f4_pct.toFixed(0)}%
                    </span>
                  </span>
                  <span>
                    Champ: <span className={a.champion_pct >= 15 ? 'text-amber-400' : 'text-zinc-400'}>
                      {a.champion_pct.toFixed(1)}%
                    </span>
                  </span>
                  {a.region && (
                    <span className={cn(REGION_COLORS[a.region] ?? 'text-zinc-500')}>
                      {a.region}
                    </span>
                  )}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Upset alerts
// ---------------------------------------------------------------------------

function UpsetAlerts({
  alerts,
}: {
  alerts: Array<{ team: string; seed: number; region: string; r64_win_prob: number }>
}) {
  const [expanded, setExpanded] = useState(false)
  const shown = expanded ? alerts : alerts.slice(0, 4)

  if (alerts.length === 0) return null

  return (
    <Card className="p-0">
      <CardHeader className="px-5 pt-5 pb-0 mb-3">
        <CardTitle className="flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-amber-400" />
          Upset Alerts
          <span className="ml-1 text-sm font-normal text-zinc-500">
            (seed ≥ 10 with ≥ 35% R64 win prob)
          </span>
        </CardTitle>
      </CardHeader>
      <div className="px-5 pb-5 space-y-2">
        {shown.map((a) => (
          <div
            key={a.team}
            className="flex items-center justify-between py-2 px-3 rounded-md bg-amber-500/5 border border-amber-500/20"
          >
            <div className="flex items-center gap-2.5">
              <SeedBadge seed={a.seed} />
              <span className="text-sm font-medium text-zinc-200">{a.team}</span>
              <span
                className={cn(
                  'text-xs border px-1.5 py-0.5 rounded capitalize',
                  REGION_BADGE[a.region] ?? 'text-zinc-500 border-zinc-700',
                )}
              >
                {a.region}
              </span>
            </div>
            <div className="text-right">
              <div className="font-mono tabular-nums text-sm text-amber-400 font-semibold">
                {a.r64_win_prob.toFixed(1)}%
              </div>
              <div className="text-xs text-zinc-600">R64 win prob</div>
            </div>
          </div>
        ))}
        {alerts.length > 4 && (
          <button
            onClick={() => setExpanded((e) => !e)}
            className="flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300 transition-colors mt-1"
          >
            {expanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
            {expanded ? 'Show fewer' : `Show ${alerts.length - 4} more`}
          </button>
        )}
      </div>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Advancement table
// ---------------------------------------------------------------------------

interface TableRow {
  team: string
  seed: number
  region: string
  r32_pct: number
  s16_pct: number
  e8_pct: number
  f4_pct: number
  runner_up_pct: number
  champion_pct: number
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function BracketPage() {
  const [nSims, setNSims] = useState(10000)
  const [regionFilter, setRegionFilter] = useState<string>('ALL')

  const { data, isLoading, isError, refetch, isFetching } = useQuery({
    queryKey: ['bracket-projection', nSims],
    queryFn: () => endpoints.bracketProjection(nSims),
    staleTime: 10 * 60 * 1000, // 10 min — sims are expensive
  })

  const champion = data?.projected_champion ?? null
  const champProb = (champion ? data?.advancement_probs[champion]?.champion_pct : undefined) ?? 0
  const finalFour = data?.projected_final_four ?? []
  const upsets = data?.upset_alerts ?? []
  const advProbs = data?.advancement_probs ?? {}

  const allRegions = ['ALL', ...Array.from(new Set(Object.values(advProbs).map((t) => t.region))).sort()]

  const tableRows: TableRow[] = Object.entries(advProbs)
    .filter(([, v]) => regionFilter === 'ALL' || v.region === regionFilter)
    .map(([team, v]) => ({ team, ...v }))
    .sort((a, b) => b.champion_pct - a.champion_pct)

  const columns: Column<TableRow>[] = [
    {
      key: 'seed',
      header: '#',
      accessor: (r) => <SeedBadge seed={r.seed} />,
      sortValue: (r) => r.seed,
    },
    {
      key: 'team',
      header: 'Team',
      accessor: (r) => (
        <span className="text-sm font-medium text-zinc-200">{r.team}</span>
      ),
      sortValue: (r) => r.team,
    },
    {
      key: 'region',
      header: 'Region',
      accessor: (r) => (
        <span
          className={cn(
            'text-xs capitalize',
            REGION_COLORS[r.region] ?? 'text-zinc-500',
          )}
        >
          {r.region}
        </span>
      ),
      sortValue: (r) => r.region,
    },
    {
      key: 'r32',
      header: 'R32',
      accessor: (r) => pctBar(r.r32_pct),
      sortValue: (r) => r.r32_pct,
      className: 'w-32',
      headerClassName: 'text-right',
    },
    {
      key: 's16',
      header: 'S16',
      accessor: (r) => pctBar(r.s16_pct),
      sortValue: (r) => r.s16_pct,
      className: 'w-32',
      headerClassName: 'text-right',
    },
    {
      key: 'e8',
      header: 'E8',
      accessor: (r) => pctBar(r.e8_pct),
      sortValue: (r) => r.e8_pct,
      className: 'w-32',
      headerClassName: 'text-right',
    },
    {
      key: 'f4',
      header: 'F4',
      accessor: (r) => pctBar(r.f4_pct),
      sortValue: (r) => r.f4_pct,
      className: 'w-32',
      headerClassName: 'text-right',
    },
    {
      key: 'runner_up',
      header: 'Runner-Up',
      accessor: (r) => pctBar(r.runner_up_pct),
      sortValue: (r) => r.runner_up_pct,
      className: 'w-32',
      headerClassName: 'text-right',
    },
    {
      key: 'champion',
      header: 'Champion',
      accessor: (r) => pctBar(r.champion_pct, 25),
      sortValue: (r) => r.champion_pct,
      className: 'w-36',
      headerClassName: 'text-right',
    },
  ]

  return (
    <div className="space-y-6 max-w-7xl">

      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-zinc-100 flex items-center gap-2">
            <Trophy className="h-5 w-5 text-amber-400" />
            Bracket Simulator
          </h1>
          <p className="text-sm text-zinc-500 mt-0.5">
            Monte Carlo tournament projection · V9.1 composite ratings
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* Sim count selector */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500">Sims:</span>
            {([1000, 5000, 10000, 25000] as const).map((n) => (
              <button
                key={n}
                onClick={() => setNSims(n)}
                className={cn(
                  'text-xs px-2.5 py-1 rounded font-mono transition-colors',
                  nSims === n
                    ? 'bg-zinc-700 text-zinc-100'
                    : 'text-zinc-500 hover:text-zinc-300',
                )}
              >
                {n >= 1000 ? `${n / 1000}k` : n}
              </button>
            ))}
          </div>
          <button
            onClick={() => refetch()}
            disabled={isFetching}
            className="flex items-center gap-1.5 text-xs text-zinc-500 hover:text-zinc-300 transition-colors disabled:opacity-40"
          >
            <RefreshCw className={cn('h-3.5 w-3.5', isFetching && 'animate-spin')} />
            Re-run
          </button>
        </div>
      </div>

      {/* Error */}
      {isError && (
        <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-4 text-rose-400 text-sm">
          Failed to load bracket projection. Check that the backend is running and bracket_2026.json exists.
        </div>
      )}

      {/* Loading */}
      {isLoading && (
        <div className="space-y-4">
          <div className="h-28 bg-zinc-800 rounded-xl animate-pulse" />
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-20 bg-zinc-800 rounded-lg animate-pulse" />
            ))}
          </div>
        </div>
      )}

      {/* Content */}
      {!isLoading && data && (
        <>
          {/* Champion hero + KPI row */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="lg:col-span-2">
              <ChampionHero
                champion={champion}
                prob={champProb}
                nSims={data.n_sims}
              />
            </div>
            <div className="grid grid-cols-2 lg:grid-cols-1 gap-4">
              <KpiCard
                title="Avg Upsets / Tourney"
                value={data.avg_upsets_per_tournament.toFixed(1)}
                unit="upsets"
                trend="neutral"
                loading={false}
              />
              <KpiCard
                title="Avg Title Margin"
                value={data.avg_championship_margin.toFixed(1)}
                unit="pts"
                trend="neutral"
                loading={false}
              />
            </div>
          </div>

          {/* Final Four + Upset Alerts side by side */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {finalFour.length > 0 && (
              <FinalFourCard teams={finalFour} advancement={advProbs} />
            )}
            <UpsetAlerts alerts={upsets} />
          </div>

          {/* Advancement probability table */}
          <Card className="p-0">
            <CardHeader className="px-6 pt-5 pb-0 mb-3">
              <div className="flex items-center justify-between">
                <CardTitle>Advancement Probabilities</CardTitle>
                {/* Region filter */}
                <div className="flex gap-1 p-1 bg-zinc-800/60 rounded-lg">
                  {allRegions.map((r) => (
                    <button
                      key={r}
                      onClick={() => setRegionFilter(r)}
                      className={cn(
                        'px-2.5 py-1 rounded text-xs font-medium capitalize transition-colors',
                        regionFilter === r
                          ? 'bg-zinc-700 text-zinc-100'
                          : r !== 'ALL'
                            ? cn('hover:text-zinc-300', REGION_COLORS[r] ?? 'text-zinc-500')
                            : 'text-zinc-500 hover:text-zinc-300',
                      )}
                    >
                      {r}
                    </button>
                  ))}
                </div>
              </div>
            </CardHeader>
            <DataTable
              columns={columns}
              data={tableRows}
              keyExtractor={(r) => r.team}
              emptyMessage="No bracket data. Run the simulation above."
            />
          </Card>

          <div className="text-xs text-zinc-600 text-center pb-2">
            Source: {data.data_source} · {data.n_sims.toLocaleString()} simulations
          </div>
        </>
      )}
    </div>
  )
}
