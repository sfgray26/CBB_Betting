export default function MatchupLoading() {
  return (
    <div className="space-y-6 max-w-5xl animate-pulse">
      {/* Header skeleton */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="space-y-2">
          <div className="h-6 w-48 bg-zinc-800 rounded" />
          <div className="h-4 w-72 bg-zinc-800/70 rounded" />
        </div>
        <div className="h-9 w-28 bg-zinc-800 rounded" />
      </div>

      {/* Score card skeleton */}
      <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-6">
        <div className="flex items-center justify-between gap-4">
          <div className="space-y-2 flex-1">
            <div className="h-5 w-32 bg-zinc-800 rounded" />
            <div className="h-8 w-16 bg-zinc-800 rounded" />
          </div>
          <div className="h-10 w-12 bg-zinc-800 rounded" />
          <div className="space-y-2 flex-1 items-end flex flex-col">
            <div className="h-5 w-32 bg-zinc-800 rounded" />
            <div className="h-8 w-16 bg-zinc-800 rounded" />
          </div>
        </div>
      </div>

      {/* Category rows skeleton */}
      <div className="rounded-xl border border-zinc-800 overflow-hidden">
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="flex items-center px-4 py-3 border-b border-zinc-800/60 last:border-0">
            <div className="h-4 w-24 bg-zinc-800 rounded flex-1" />
            <div className="h-4 w-12 bg-zinc-800/70 rounded mx-8" />
            <div className="h-4 w-24 bg-zinc-800 rounded flex-1" />
          </div>
        ))}
      </div>
    </div>
  )
}
