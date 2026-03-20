export default function AdminLoading() {
  return (
    <div className="space-y-6 max-w-5xl animate-pulse">
      <div className="space-y-2">
        <div className="h-6 w-48 bg-zinc-800 rounded" />
        <div className="h-4 w-64 bg-zinc-800 rounded" />
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="h-52 bg-zinc-800 rounded-lg" />
        ))}
      </div>
    </div>
  )
}
