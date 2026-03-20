'use client'

import { AlertTriangle } from 'lucide-react'

export default function WaiverError({
  error,
  reset,
}: {
  error: Error
  reset: () => void
}) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[400px] gap-4 text-center p-8">
      <AlertTriangle className="h-10 w-10 text-amber-400" />
      <div>
        <p className="text-zinc-200 font-medium">Waiver Wire failed to load</p>
        <p className="text-zinc-500 text-sm mt-1">{error.message}</p>
      </div>
      <button
        onClick={reset}
        className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-sm rounded-md transition-colors min-h-[44px]"
      >
        Try again
      </button>
    </div>
  )
}
