export function MatchupSkeleton() {
  return (
    <div className="max-w-6xl mx-auto p-6 lg:p-8 space-y-6 animate-pulse">
      {/* Header bar */}
      <div className="flex items-center gap-3 mb-2">
        <div className="h-6 w-6 rounded bg-bg-elevated" />
        <div className="h-5 w-24 rounded bg-bg-elevated" />
        <div className="ml-auto h-9 w-32 rounded-sm bg-bg-elevated" />
      </div>

      {/* Matchup header card */}
      <div className="bg-bg-surface border border-border-subtle rounded-lg p-6 lg:p-8">
        <div className="h-3 w-28 rounded bg-bg-elevated mb-6" />
        <div className="flex items-center gap-6">
          {/* My team */}
          <div className="flex-1 space-y-2">
            <div className="h-6 w-40 rounded bg-bg-elevated" />
            <div className="h-3 w-16 rounded bg-bg-inset" />
          </div>
          {/* Score */}
          <div className="flex-shrink-0 text-center px-4 space-y-2">
            <div className="flex items-center gap-3 justify-center">
              <div className="h-14 w-10 rounded bg-bg-elevated" />
              <div className="h-8 w-3 rounded bg-bg-inset" />
              <div className="h-14 w-10 rounded bg-bg-elevated" />
            </div>
            <div className="h-3 w-16 rounded bg-bg-inset mx-auto" />
          </div>
          {/* Opponent */}
          <div className="flex-1 space-y-2 flex flex-col items-end">
            <div className="h-6 w-40 rounded bg-bg-elevated" />
            <div className="h-3 w-16 rounded bg-bg-inset" />
          </div>
        </div>
        {/* Projected strip */}
        <div className="mt-6 pt-4 border-t border-border-subtle flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="h-3 w-16 rounded bg-bg-inset" />
            <div className="h-4 w-8 rounded bg-bg-elevated" />
            <div className="h-4 w-3 rounded bg-bg-inset" />
            <div className="h-4 w-8 rounded bg-bg-elevated" />
          </div>
          <div className="flex items-center gap-3">
            <div className="h-3 w-24 rounded bg-bg-inset" />
            <div className="h-6 w-10 rounded bg-bg-elevated" />
          </div>
        </div>
      </div>

      {/* Category battlefield card */}
      <div className="bg-bg-surface border border-border-subtle rounded-lg">
        {/* Controls bar */}
        <div className="flex items-center justify-between gap-4 px-6 pt-5 pb-4 border-b border-border-subtle">
          <div className="flex gap-3">
            {[80, 60, 48, 56].map((w, i) => (
              <div key={i} className="h-3 rounded bg-bg-elevated" style={{ width: w }} />
            ))}
          </div>
          <div className="flex gap-2">
            {[1, 2, 3, 4].map(i => (
              <div key={i} className="h-8 w-16 rounded-sm bg-bg-elevated" />
            ))}
          </div>
          <div className="h-8 w-36 rounded-sm bg-bg-elevated" />
        </div>

        {/* Column headers */}
        <div className="flex items-center gap-3 px-6 pt-4 pb-2">
          <div className="w-10" />
          <div className="w-14 h-3 rounded bg-bg-inset" />
          <div className="flex-1" />
          <div className="w-14 h-3 rounded bg-bg-inset" />
          <div className="w-20 h-3 rounded bg-bg-inset hidden sm:block" />
          <div className="w-16 h-3 rounded bg-bg-inset" />
          <div className="w-20 h-3 rounded bg-bg-inset hidden md:block" />
        </div>

        {/* Category rows */}
        <div className="px-6 pb-5">
          <div className="h-3 w-16 rounded bg-accent-gold/30 my-3" />
          {[1, 2, 3, 4, 5, 6, 7, 8, 9].map(i => (
            <CategoryRowSkeleton key={i} />
          ))}
          <div className="h-3 w-16 rounded bg-accent-gold/30 my-3 mt-5" />
          {[1, 2, 3, 4, 5, 6, 7, 8, 9].map(i => (
            <CategoryRowSkeleton key={i + 9} />
          ))}
        </div>
      </div>
    </div>
  )
}

function CategoryRowSkeleton() {
  return (
    <div className="flex items-center gap-3 py-3 border-b border-bg-elevated last:border-0 px-2">
      <div className="w-10 h-3 rounded bg-bg-elevated flex-shrink-0" />
      <div className="w-14 h-4 rounded bg-bg-elevated flex-shrink-0" />
      <div className="flex-1 h-2 rounded bg-bg-inset" />
      <div className="w-14 h-4 rounded bg-bg-elevated flex-shrink-0" />
      <div className="w-20 h-3 rounded bg-bg-inset flex-shrink-0 hidden sm:block" />
      <div className="w-16 h-3 rounded bg-bg-elevated flex-shrink-0" />
      <div className="w-20 h-3 rounded bg-bg-inset flex-shrink-0 hidden md:block" />
    </div>
  )
}
