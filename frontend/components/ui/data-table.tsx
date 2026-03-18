'use client'

import { useState } from 'react'
import { ChevronUp, ChevronDown, ChevronsUpDown } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface Column<T> {
  key: string
  header: string
  accessor: (row: T) => React.ReactNode
  sortValue?: (row: T) => string | number
  className?: string
  headerClassName?: string
}

interface DataTableProps<T> {
  columns: Column<T>[]
  data: T[]
  keyExtractor: (row: T) => string | number
  className?: string
  emptyMessage?: string
}

export function DataTable<T>({
  columns,
  data,
  keyExtractor,
  className,
  emptyMessage = 'No data available.',
}: DataTableProps<T>) {
  const [sortKey, setSortKey] = useState<string | null>(null)
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')

  function handleSort(key: string) {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir('desc')
    }
  }

  const sorted = [...data].sort((a, b) => {
    if (!sortKey) return 0
    const col = columns.find((c) => c.key === sortKey)
    if (!col?.sortValue) return 0
    const av = col.sortValue(a)
    const bv = col.sortValue(b)
    if (typeof av === 'number' && typeof bv === 'number') {
      return sortDir === 'asc' ? av - bv : bv - av
    }
    const as = String(av)
    const bs = String(bv)
    return sortDir === 'asc' ? as.localeCompare(bs) : bs.localeCompare(as)
  })

  return (
    <div className={cn('overflow-x-auto', className)}>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-800">
            {columns.map((col) => (
              <th
                key={col.key}
                className={cn(
                  'px-4 py-3 text-left text-xs font-medium text-zinc-400 uppercase tracking-wider whitespace-nowrap',
                  col.sortValue && 'cursor-pointer select-none hover:text-zinc-200',
                  col.headerClassName,
                )}
                onClick={() => col.sortValue && handleSort(col.key)}
              >
                <span className="flex items-center gap-1">
                  {col.header}
                  {col.sortValue && (
                    <span className="text-zinc-600">
                      {sortKey === col.key ? (
                        sortDir === 'asc' ? (
                          <ChevronUp className="h-3 w-3" />
                        ) : (
                          <ChevronDown className="h-3 w-3" />
                        )
                      ) : (
                        <ChevronsUpDown className="h-3 w-3" />
                      )}
                    </span>
                  )}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.length === 0 ? (
            <tr>
              <td
                colSpan={columns.length}
                className="px-4 py-8 text-center text-zinc-500 text-sm"
              >
                {emptyMessage}
              </td>
            </tr>
          ) : (
            sorted.map((row) => (
              <tr
                key={keyExtractor(row)}
                className="border-b border-zinc-800/50 hover:bg-zinc-800/30 transition-colors"
              >
                {columns.map((col) => (
                  <td
                    key={col.key}
                    className={cn('px-4 py-3 text-zinc-300', col.className)}
                  >
                    {col.accessor(row)}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  )
}
