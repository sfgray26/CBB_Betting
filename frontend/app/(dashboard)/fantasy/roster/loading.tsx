export default function RosterLoading() {
  return (
    <div className="space-y-6 max-w-5xl animate-pulse">
      <div className="space-y-2">
        <div className="h-6 w-36 bg-zinc-800 rounded" />
        <div className="h-4 w-56 bg-zinc-800 rounded" />
      </div>
      <div className="overflow-x-auto rounded-lg border border-zinc-800">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-800 bg-zinc-900/60">
              {[160, 60, 60, 80, 80, 90].map((w, i) => (
                <th key={i} className="px-3 py-3">
                  <div className="h-3 bg-zinc-800 rounded" style={{ width: w }} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/60">
            {Array.from({ length: 10 }).map((_, i) => (
              <tr key={i}>
                {[160, 60, 60, 80, 80, 90].map((w, j) => (
                  <td key={j} className="px-3 py-3">
                    <div className="h-3 bg-zinc-800/70 rounded" style={{ width: w * 0.8 }} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
