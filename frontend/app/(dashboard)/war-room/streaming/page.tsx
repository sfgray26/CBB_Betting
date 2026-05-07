'use client'

import { useQuery } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import { Loader2, AlertCircle } from 'lucide-react'

export default function StreamingStationPage() {
  const waiver = useQuery({
    queryKey: ['waiver'],
    queryFn: () => endpoints.getWaiver(),
    staleTime: 5 * 60_000,
  })

  return (
    <div className="min-h-screen bg-black p-6 space-y-6">
      <h1 className="text-xl font-bold tracking-widest uppercase text-[#FFC000]">
        STREAMING STATION
      </h1>

      {waiver.isLoading && (
        <div className="flex items-center gap-2 text-[#7D7D7D]">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span className="text-sm">Loading waiver data...</span>
        </div>
      )}

      {waiver.isError && (
        <div className="flex items-center gap-2 text-rose-400">
          <AlertCircle className="h-4 w-4" />
          <span className="text-sm">{waiver.error?.message ?? 'API error'}</span>
        </div>
      )}

      {waiver.data && (
        <div className="bg-[#202020] p-4 space-y-2">
          <p className="text-xs font-semibold tracking-widest uppercase text-[#7D7D7D]">WAIVER — API OK</p>
          <p className="text-white text-sm">
            {waiver.data.two_start_pitchers.length} two-start pitchers ·{' '}
            {waiver.data.category_deficits.length} category deficits ·{' '}
            FAAB ${waiver.data.faab_balance ?? '—'}
          </p>
          <pre className="text-xs text-[#969696] overflow-auto max-h-64">
            {JSON.stringify(waiver.data, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}
