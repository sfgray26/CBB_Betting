export default function BracketLoading() {
  return (
    <div className="space-y-6 max-w-7xl animate-pulse">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-2">
          <div className="h-6 w-52 bg-zinc-800 rounded" />
          <div className="h-4 w-72 bg-zinc-800 rounded" />
        </div>
        <div className="h-8 w-48 bg-zinc-800 rounded" />
      </div>

      {/* Champion hero + KPIs */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 h-28 bg-zinc-800 rounded-xl" />
        <div className="grid grid-cols-2 lg:grid-cols-1 gap-4">
          <div className="h-20 bg-zinc-800 rounded-lg" />
          <div className="h-20 bg-zinc-800 rounded-lg" />
        </div>
      </div>

      {/* Final Four + Upset Alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="h-48 bg-zinc-800 rounded-lg" />
        <div className="h-48 bg-zinc-800 rounded-lg" />
      </div>

      {/* Advancement table */}
      <div className="h-96 bg-zinc-800 rounded-lg" />
    </div>
  )
}
