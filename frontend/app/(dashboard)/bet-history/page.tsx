'use client'

import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { format, parseISO } from 'date-fns'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { endpoints } from '@/lib/api'
import { Card, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { DataTable, type Column } from '@/components/ui/data-table'
import type { BetLog } from '@/lib/types'
import { cn } from '@/lib/utils'

function ErrorCard({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-6 text-rose-400 text-sm">
      {message}
    </div>
  )
}

function outcomeVariant(pl?: number | null): 'win' | 'loss' | 'push' | 'pending' {
  if (pl == null) return 'pending'
  if (pl > 0) return 'win'
  if (pl < 0) return 'loss'
  return 'push'
}

function outcomeLabel(pl?: number | null): string {
  if (pl == null) return 'Pending'
  if (pl > 0) return 'Win'
  if (pl < 0) return 'Loss'
  return 'Push'
}

const PAGE_SIZE = 50

export default function BetHistoryPage() {
  const [status, setStatus] = useState('all')
  const [days, setDays] = useState(60)
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)

  const { data, isLoading, isError } = useQuery({
    queryKey: ['bets', status, days],
    queryFn: () => endpoints.bets(status, days),
  })

  const filtered = useMemo(() => {
    const bets = data?.bets ?? []
    if (!search.trim()) return bets
    const q = search.toLowerCase()
    return bets.filter((b) => (b.pick ?? '').toLowerCase().includes(q))
  }, [data, search])

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
  const paginated = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)

  const settled = (data?.bets ?? []).filter(
    (b) => b.profit_loss_units !== undefined && b.profit_loss_units !== null,
  ).length
  const pending = (data?.bets ?? []).length - settled

  const columns: Column<BetLog>[] = [
    {
      key: 'date',
      header: 'Date',
      accessor: (r) => (
        <span className="font-mono tabular-nums text-zinc-400 text-xs whitespace-nowrap">
          {r.timestamp ? format(parseISO(r.timestamp), 'MMM d, yyyy') : '—'}
        </span>
      ),
      sortValue: (r) => r.timestamp ?? '',
    },
    {
      key: 'pick',
      header: 'Pick',
      accessor: (r) => (
        <div>
          <span className="text-zinc-100 text-sm">{r.pick ?? '—'}</span>
          {r.is_paper_trade && (
            <span className="ml-2 text-xs bg-zinc-800 text-zinc-500 border border-zinc-700 rounded px-1 py-0.5">
              Paper
            </span>
          )}
        </div>
      ),
      sortValue: (r) => r.pick,
    },
    {
      key: 'type',
      header: 'Type',
      accessor: (r) => (
        <span className="text-zinc-400 text-xs uppercase tracking-wider">{r.bet_type}</span>
      ),
      sortValue: (r) => r.bet_type,
    },
    {
      key: 'odds',
      header: 'Odds',
      accessor: (r) => {
        const odds = r.odds_taken ?? 0
        return (
          <span className="font-mono tabular-nums text-zinc-300">
            {odds > 0 ? '+' : ''}{odds}
          </span>
        )
      },
      sortValue: (r) => r.odds_taken ?? 0,
      className: 'text-right',
      headerClassName: 'text-right',
    },
    {
      key: 'units',
      header: 'Units',
      accessor: (r) => (
        <span className="font-mono tabular-nums text-zinc-300">
          {(r.bet_size_units ?? 0).toFixed(2)}u
        </span>
      ),
      sortValue: (r) => r.bet_size_units ?? 0,
      className: 'text-right',
      headerClassName: 'text-right',
    },
    {
      key: 'outcome',
      header: 'Outcome',
      accessor: (r) => (
        <Badge variant={outcomeVariant(r.profit_loss_units)}>
          {outcomeLabel(r.profit_loss_units)}
        </Badge>
      ),
      sortValue: (r) => outcomeLabel(r.profit_loss_units),
      className: 'text-right',
      headerClassName: 'text-right',
    },
    {
      key: 'pl',
      header: 'P&L',
      accessor: (r) => {
        const pl = r.profit_loss_units
        if (pl === undefined || pl === null)
          return <span className="font-mono tabular-nums text-zinc-500">-</span>
        return (
          <span
            className={cn(
              'font-mono tabular-nums',
              pl > 0 ? 'text-emerald-400' : pl < 0 ? 'text-rose-400' : 'text-zinc-400',
            )}
          >
            {pl > 0 ? '+' : ''}
            {pl.toFixed(2)}u
          </span>
        )
      },
      sortValue: (r) => r.profit_loss_units ?? -9999,
      className: 'text-right',
      headerClassName: 'text-right',
    },
    {
      key: 'clv',
      header: 'CLV',
      accessor: (r) => {
        const clv = r.clv_points
        if (clv === undefined || clv === null)
          return <span className="font-mono tabular-nums text-zinc-500">-</span>
        return (
          <span
            className={cn(
              'font-mono tabular-nums',
              clv > 0 ? 'text-emerald-400' : clv < 0 ? 'text-rose-400' : 'text-zinc-400',
            )}
          >
            {clv > 0 ? '+' : ''}
            {clv.toFixed(2)}
          </span>
        )
      },
      sortValue: (r) => r.clv_points ?? 0,
      className: 'text-right',
      headerClassName: 'text-right',
    },
  ]

  return (
    <div className="space-y-6 max-w-7xl">
      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3">
        <select
          value={status}
          onChange={(e) => {
            setStatus(e.target.value)
            setPage(1)
          }}
          className="bg-zinc-800 border border-zinc-700 text-zinc-200 text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-amber-400/40"
        >
          <option value="all">All Status</option>
          <option value="placed">Placed (Real)</option>
          <option value="pending">Pending</option>
          <option value="settled">Settled</option>
        </select>

        <select
          value={days}
          onChange={(e) => {
            setDays(Number(e.target.value))
            setPage(1)
          }}
          className="bg-zinc-800 border border-zinc-700 text-zinc-200 text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-amber-400/40"
        >
          <option value={7}>7 Days</option>
          <option value={30}>30 Days</option>
          <option value={60}>60 Days</option>
          <option value={90}>90 Days</option>
        </select>

        <input
          type="text"
          placeholder="Search by pick..."
          value={search}
          onChange={(e) => {
            setSearch(e.target.value)
            setPage(1)
          }}
          className="bg-zinc-800 border border-zinc-700 text-zinc-200 text-sm rounded-lg px-3 py-2 placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-amber-400/40 w-48"
        />

        {/* Summary chips */}
        <div className="flex items-center gap-3 ml-auto text-xs text-zinc-500 font-mono">
          <span>
            <span className="text-zinc-300">{data?.total ?? 0}</span> total
          </span>
          <span>
            <span className="text-zinc-300">{settled}</span> settled
          </span>
          <span>
            <span className="text-sky-400">{pending}</span> pending
          </span>
        </div>
      </div>

      {isError && (
        <ErrorCard message="Failed to load bet history. Check your connection." />
      )}

      {/* Table */}
      <Card className="p-0">
        {isLoading ? (
          <div className="p-6 space-y-2">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="h-12 bg-zinc-800 rounded animate-pulse" />
            ))}
          </div>
        ) : (
          <DataTable
            columns={columns}
            data={paginated}
            keyExtractor={(r) => r.id}
            emptyMessage="No bets found for the selected filters."
          />
        )}

        {/* Pagination */}
        {!isLoading && filtered.length > PAGE_SIZE && (
          <div className="px-6 py-4 border-t border-zinc-800 flex items-center justify-between text-sm text-zinc-500">
            <span className="font-mono tabular-nums">
              {(page - 1) * PAGE_SIZE + 1}–{Math.min(page * PAGE_SIZE, filtered.length)} of{' '}
              {filtered.length}
            </span>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="icon"
                disabled={page <= 1}
                onClick={() => setPage((p) => p - 1)}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <span className="font-mono tabular-nums text-zinc-400">
                {page} / {totalPages}
              </span>
              <Button
                variant="ghost"
                size="icon"
                disabled={page >= totalPages}
                onClick={() => setPage((p) => p + 1)}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        )}
      </Card>
    </div>
  )
}
