export default function FantasyLoading() {
  return (
    <div className="space-y-6 animate-pulse">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <div className="h-7 w-52 bg-zinc-800 rounded" />
          <div className="h-4 w-72 bg-zinc-800 rounded" />
        </div>
        <div className="flex gap-3">
          <div className="h-9 w-48 bg-zinc-800 rounded-md" />
          <div className="h-9 w-9 bg-zinc-800 rounded-md" />
        </div>
      </div>

      {/* Filter bar */}
      <div className="flex gap-3">
        <div className="h-8 w-48 bg-zinc-800 rounded-md" />
        <div className="h-8 w-32 bg-zinc-800 rounded-md" />
        <div className="h-8 w-64 bg-zinc-800 rounded-md" />
      </div>

      {/* Table rows */}
      <div className="rounded-lg border border-zinc-800 overflow-hidden">
        <div className="h-10 bg-zinc-900" />
        {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
          <div key={i} className="h-12 border-t border-zinc-800/60 bg-zinc-900/30" />
        ))}
      </div>
    </div>
  )
}
