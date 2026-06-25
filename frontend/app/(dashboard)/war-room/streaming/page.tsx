'use client'

import { useState } from 'react'
import { StreamingRecommendations } from '@/components/streaming/streaming-recommendations'

export default function StreamingPage() {
  const [targetDate, setTargetDate] = useState(() => {
    const now = new Date()
    return now.toISOString().split('T')[0]
  })

  return (
    <div className="min-h-screen bg-bg-base p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold tracking-widest uppercase text-accent-gold">
          STREAMING RECOMMENDATIONS
        </h1>
        <div className="flex items-center gap-3">
          <label className="text-xs text-text-secondary uppercase tracking-wider">
            Target Date:
          </label>
          <input
            type="date"
            value={targetDate}
            onChange={(e) => setTargetDate(e.target.value)}
            className="bg-bg-surface border border-border-subtle rounded px-3 py-1.5 text-sm text-text-primary focus:outline-none focus:border-accent-primary"
          />
        </div>
      </div>

      {/* Streaming Recommendations Component */}
      <StreamingRecommendations targetDate={targetDate} />
    </div>
  )
}
