export default function TodayLoading() {
  return (
    <div className="space-y-6 max-w-4xl animate-pulse">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <div className="h-6 w-36 bg-zinc-800 rounded" />
          <div className="h-4 w-48 bg-zinc-800 rounded" />
        </div>
        <div className="h-6 w-24 bg-zinc-800 rounded" />
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="h-20 bg-zinc-800 rounded-lg" />
        ))}
      </div>

      {/* Bet cards */}
      {[1, 2, 3].map((i) => (
        <div key={i} className="h-36 bg-zinc-800 rounded-lg" />
      ))}
    </div>
  )
}
