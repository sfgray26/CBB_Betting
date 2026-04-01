'use client'

import { AlertTriangle } from 'lucide-react'

export default function MatchupError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[40vh] space-y-4">
      <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-6 max-w-md w-full text-center space-y-3">
        <div className="flex justify-center">
          <AlertTriangle className="h-8 w-8 text-rose-400" />
        </div>
        <p className="text-rose-400 font-medium">Matchup data unavailable</p>
        <p className="text-zinc-500 text-sm">
          {error.message || 'The backend endpoint may not be available.'}
        </p>
        <button
          onClick={reset}
          className="px-4 py-2 text-sm bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-md transition-colors"
        >
          Try again
        </button>
      </div>
    </div>
  )
}
