export default function WaiverLoading() {
  return (
    <div className="space-y-6 max-w-6xl animate-pulse">
      <div className="space-y-2">
        <div className="h-6 w-52 bg-zinc-800 rounded" />
        <div className="h-4 w-64 bg-zinc-800 rounded" />
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        {Array.from({ length: 10 }).map((_, i) => (
          <div key={i} className="h-24 bg-zinc-800 rounded-lg" />
        ))}
      </div>
      <div className="h-64 bg-zinc-800 rounded-lg" />
      <div className="h-40 bg-zinc-800 rounded-lg" />
    </div>
  )
}
