'use client'

import { useQuery } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import { Loader2, AlertCircle } from 'lucide-react'

export default function RosterLabPage() {
  const projections = useQuery({
    queryKey: ['canonicalProjections'],
    queryFn: endpoints.getCanonicalProjections,
    staleTime: 15 * 60_000,
  })

  return (
    <div className="min-h-screen bg-black p-6 space-y-6">
      <h1 className="text-xl font-bold tracking-widest uppercase text-[#FFC000]">
        ROSTER LAB
      </h1>

      {projections.isLoading && (
        <div className="flex items-center gap-2 text-[#7D7D7D]">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span className="text-sm">Loading projection data...</span>
        </div>
      )}

      {projections.isError && (
        <div className="flex items-center gap-2 text-rose-400">
          <AlertCircle className="h-4 w-4" />
          <span className="text-sm">{projections.error?.message ?? 'API error'}</span>
        </div>
      )}

      {projections.data && (
        <div className="bg-[#202020] p-4 space-y-2">
          <p className="text-xs font-semibold tracking-widest uppercase text-[#7D7D7D]">PROJECTIONS — API OK</p>
          <p className="text-white text-sm">
            {projections.data.players.length} players · as of {projections.data.as_of_date}
          </p>
          <pre className="text-xs text-[#969696] overflow-auto max-h-64">
            {JSON.stringify({ players: projections.data.players.slice(0, 3), as_of_date: projections.data.as_of_date }, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}
