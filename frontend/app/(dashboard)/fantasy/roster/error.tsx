'use client'

import { AlertTriangle } from 'lucide-react'

function SkeletonPlaceholder() {
  return (
    <div className="overflow-x-auto rounded-lg border border-zinc-800 animate-pulse opacity-30">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-800 bg-zinc-900/60">
            {[160, 70, 60, 80, 80, 90].map((w, i) => (
              <th key={i} className="px-3 py-3">
                <div className="h-3 bg-zinc-700 rounded" style={{ width: w }} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-800/60">
          {Array.from({ length: 8 }).map((_, i) => (
            <tr key={i}>
              {[160, 70, 60, 80, 80, 90].map((w, j) => (
                <td key={j} className="px-3 py-3">
                  <div className="h-3 bg-zinc-800/70 rounded" style={{ width: w * 0.8 }} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function RosterError({
  error,
  reset,
}: {
  error: Error
  reset: () => void
}) {
  return (
    <div className="space-y-4 max-w-5xl">
      <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-5 flex items-center justify-between">
        <div className="flex items-start gap-3">
          <AlertTriangle className="h-5 w-5 text-rose-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-rose-400 font-medium text-sm">Roster failed to load</p>
            <p className="text-rose-400/60 text-xs mt-0.5">{error.message}</p>
          </div>
        </div>
        <button
          onClick={reset}
          className="px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-xs rounded-md transition-colors min-h-[36px] flex-shrink-0"
        >
          Try again
        </button>
      </div>
      <SkeletonPlaceholder />
    </div>
  )
}
