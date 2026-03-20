export default function LineupLoading() {
  return (
    <div className="space-y-6 max-w-6xl animate-pulse">
      <div className="space-y-2">
        <div className="h-6 w-48 bg-zinc-800 rounded" />
        <div className="h-4 w-72 bg-zinc-800 rounded" />
      </div>
      <div className="h-8 w-40 bg-zinc-800 rounded" />
      <div className="space-y-3">
        <div className="h-5 w-24 bg-zinc-800 rounded" />
        <div className="h-64 bg-zinc-800 rounded-lg" />
      </div>
      <div className="space-y-3">
        <div className="h-5 w-36 bg-zinc-800 rounded" />
        <div className="h-48 bg-zinc-800 rounded-lg" />
      </div>
    </div>
  )
}
