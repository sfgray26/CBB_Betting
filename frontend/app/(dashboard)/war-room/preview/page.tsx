'use client'

import { useQuery } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import { Loader2, AlertCircle, Eye, Calendar, Zap, ArrowRight } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { CategoryProjection, RotoCategory } from '@/lib/types'
import {
  BATTER_CATEGORIES,
  PITCHER_CATEGORIES,
  CATEGORY_LABEL,
  CATEGORY_COLOR,
  RATIO_CATEGORIES,
} from '@/lib/types'
import Link from 'next/link'

function formatStat(cat: RotoCategory, val: number | null): string {
  if (val === null || val === undefined) return '—'
  if (cat === 'AVG' || cat === 'OPS') return val.toFixed(3).replace(/^0/, '')
  if (cat === 'ERA' || cat === 'WHIP' || cat === 'K_9') return val.toFixed(2)
  return String(Math.round(val))
}

function statusFromWinProb(winProb: number): { text: string; color: string; bg: string; border: string } {
  if (winProb > 0.65) {
    return { text: 'PROJECTED WIN', color: 'text-status-safe', bg: 'bg-status-safe/10', border: 'border-status-safe/30' }
  }
  if (winProb < 0.35) {
    return { text: 'PROJECTED LOSS', color: 'text-status-lost', bg: 'bg-status-lost/10', border: 'border-status-lost/30' }
  }
  return { text: 'BUBBLE', color: 'text-status-bubble', bg: 'bg-status-bubble/10', border: 'border-status-bubble/30' }
}

function PreviewRow({ proj }: { proj: CategoryProjection }) {
  const isRatio = RATIO_CATEGORIES.includes(proj.category)
  const label = CATEGORY_LABEL[proj.category]
  const catColor = CATEGORY_COLOR[proj.category]
  const status = statusFromWinProb(proj.win_prob)

  const myDisplay = formatStat(proj.category, proj.my_proj)
  const oppDisplay = formatStat(proj.category, proj.opp_proj)

  return (
    <div className="flex items-center gap-3 py-3 px-4 border-b border-border-subtle last:border-0 hover:bg-bg-elevated transition-colors duration-150">
      {/* Category label */}
      <span className="w-14 text-sm font-semibold tracking-wide text-text-secondary uppercase flex items-center gap-1.5 flex-shrink-0">
        <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: catColor }} />
        {label}
      </span>

      {/* My projected */}
      <span className="w-16 text-right text-base tabular-nums font-mono text-text-primary flex-shrink-0">
        {myDisplay}
      </span>

      {/* Comparison bar */}
      {isRatio ? (
        <div className="flex-1" />
      ) : (
        <div className="flex-1 flex h-2 gap-px min-w-0">
          <div className="flex-1 flex justify-end bg-bg-inset rounded-l-sm overflow-hidden">
            <div
              className="h-full transition-all duration-500"
              style={{
                width: `${proj.win_prob * 100}%`,
                backgroundColor: proj.win_prob > 0.5 ? `${catColor}cc` : '#3a3a4d',
              }}
            />
          </div>
          <div className="w-px bg-border-subtle flex-shrink-0" />
          <div className="flex-1 bg-bg-inset rounded-r-sm overflow-hidden">
            <div
              className="h-full transition-all duration-500"
              style={{
                width: `${(1 - proj.win_prob) * 100}%`,
                backgroundColor: proj.win_prob <= 0.5 ? '#969696' : '#2a2a3d',
              }}
            />
          </div>
        </div>
      )}

      {/* Opp projected */}
      <span className="w-16 text-base tabular-nums font-mono text-text-secondary flex-shrink-0">
        {oppDisplay}
      </span>

      {/* Win prob */}
      <span className="w-16 text-right text-sm font-mono text-text-muted flex-shrink-0">
        {Math.round(proj.win_prob * 100)}%
      </span>

      {/* Status tag */}
      <div className="w-28 flex justify-end flex-shrink-0">
        <span className={cn('text-[10px] font-bold tracking-wider uppercase px-2 py-1 rounded border', status.bg, status.border, status.color)}>
          {status.text}
        </span>
      </div>
    </div>
  )
}

export default function WeeklyPreviewPage() {
  const preview = useQuery({
    queryKey: ['matchup-preview'],
    queryFn: async () => {
      try {
        return await endpoints.getMatchupPreview()
      } catch (e) {
        console.error('Matchup preview fetch failed:', e)
        throw e
      }
    },
    staleTime: 5 * 60_000,
    retry: 1,
  })

  if (preview.isLoading) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center">
        <div className="flex items-center gap-2 text-text-secondary">
          <Loader2 className="h-5 w-5 animate-spin text-accent-gold" />
          <span className="text-sm">Loading weekly preview…</span>
        </div>
      </div>
    )
  }

  // Graceful 404 placeholder when endpoint is not ready
  if (preview.isError) {
    const errMsg = preview.error instanceof Error ? preview.error.message : ''
    const is404 = errMsg.includes('404') || errMsg.includes('Not Found') || errMsg.includes('not found')

    if (is404) {
      return (
        <div className="min-h-[60vh] flex items-center justify-center p-6">
          <div className="bg-bg-surface border border-border-subtle rounded-lg p-8 max-w-md w-full text-center space-y-4">
            <Eye className="h-10 w-10 text-accent-gold mx-auto" />
            <h2 className="text-lg font-bold text-text-primary tracking-widest uppercase">Weekly Preview</h2>
            <p className="text-text-secondary text-sm">
              Next-week matchup projections are coming soon. Check back after the backend endpoint is deployed.
            </p>
            <div className="text-[10px] text-text-muted uppercase tracking-wider">
              Endpoint: /api/fantasy/matchup-preview
            </div>
          </div>
        </div>
      )
    }

    return (
      <div className="min-h-[60vh] flex items-center justify-center p-6">
        <div className="bg-bg-surface border border-border-subtle rounded-lg p-6 max-w-md w-full">
          <div className="flex items-center gap-2 text-status-lost mb-2">
            <AlertCircle className="h-5 w-5" />
            <span className="text-sm font-semibold">Failed to load preview</span>
          </div>
          <p className="text-text-secondary text-sm">{errMsg || 'Unknown error'}</p>
          <button
            onClick={() => preview.refetch()}
            className="mt-4 text-xs text-accent-gold hover:text-amber-300 font-semibold"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (!preview.data) return null

  const data = preview.data
  const hitters = data.category_projections.filter(p => BATTER_CATEGORIES.includes(p.category as typeof BATTER_CATEGORIES[number]))
  const pitchers = data.category_projections.filter(p => PITCHER_CATEGORIES.includes(p.category as typeof PITCHER_CATEGORIES[number]))

  return (
    <div className="min-h-screen bg-bg-base space-y-6">
      {/* Header */}
      <div className="max-w-6xl mx-auto p-6 lg:p-8">
        <div className="flex items-center gap-3 mb-6">
          <Eye className="h-6 w-6 text-accent-gold" />
          <span className="text-lg font-bold tracking-widest uppercase text-accent-gold">Weekly Preview</span>
          {data.week_number > 0 && (
            <span className="text-[10px] px-2 py-1 bg-accent-primary/10 text-accent-primary border border-accent-primary/30 rounded font-bold tracking-wider uppercase">
              Week {data.week_number} · PREVIEW
            </span>
          )}
        </div>

        {/* Opponent card */}
        <div className="bg-bg-surface border border-border-subtle rounded-lg p-6 mb-6">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-4">
              {data.opponent_logo ? (
                <img src={data.opponent_logo} alt={data.opponent_name} className="h-12 w-12 rounded-full object-cover" />
              ) : (
                <div className="h-12 w-12 rounded-full bg-bg-inset flex items-center justify-center">
                  <span className="text-lg font-bold text-text-muted">{(data.opponent_name || '?').charAt(0)}</span>
                </div>
              )}
              <div>
                <p className="text-xs text-text-muted uppercase tracking-wider">Next Opponent</p>
                <p className="text-xl font-bold text-text-primary">{data.opponent_name}</p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-xs text-text-muted uppercase tracking-wider">Projected Win%</p>
              <p className={cn(
                'text-2xl font-bold font-mono',
                data.overall_win_prob > 0.5 ? 'text-status-safe' : data.overall_win_prob < 0.5 ? 'text-status-lost' : 'text-status-bubble',
              )}>
                {Math.round(data.overall_win_prob * 100)}%
              </p>
            </div>
          </div>
        </div>

        {/* Category projections */}
        <div className="bg-bg-surface border border-border-subtle rounded-lg mb-6">
          <div className="flex items-center justify-between px-6 pt-5 pb-4 border-b border-border-subtle">
            <span className="text-xs font-semibold tracking-widest uppercase text-text-muted">Category Projections</span>
            <div className="flex items-center gap-3 text-xs">
              <span className="text-status-safe font-semibold">WIN &gt;65%</span>
              <span className="text-status-bubble font-semibold">BUBBLE 35-65%</span>
              <span className="text-status-lost font-semibold">LOSS &lt;35%</span>
            </div>
          </div>

          {/* Column headers */}
          <div className="flex items-center gap-3 px-6 pt-4 pb-2">
            <span className="w-14" />
            <span className="w-16 text-right text-xs font-semibold tracking-widest uppercase text-text-muted">ME</span>
            <div className="flex-1" />
            <span className="w-16 text-xs font-semibold tracking-widest uppercase text-text-muted">OPP</span>
            <span className="w-16 text-right text-xs font-semibold tracking-widest uppercase text-text-muted hidden sm:block">WIN%</span>
            <span className="w-28 text-right text-xs font-semibold tracking-widest uppercase text-text-muted">STATUS</span>
          </div>

          {/* Batting */}
          {hitters.length > 0 && (
            <div className="px-2">
              <p className="text-sm font-semibold tracking-widest uppercase text-accent-gold pt-4 pb-2 px-2">
                BATTING
              </p>
              {hitters.map((proj) => (
                <PreviewRow key={proj.category} proj={proj} />
              ))}
            </div>
          )}

          {/* Pitching */}
          {pitchers.length > 0 && (
            <div className="px-2 pb-4">
              <p className="text-sm font-semibold tracking-widest uppercase text-accent-gold pt-5 pb-2 px-2">
                PITCHING
              </p>
              {pitchers.map((proj) => (
                <PreviewRow key={proj.category} proj={proj} />
              ))}
            </div>
          )}
        </div>

        {/* Needs Streaming */}
        {data.weak_categories.length > 0 && (
          <div className="bg-bg-surface border border-border-subtle rounded-lg p-6 mb-6">
            <div className="flex items-center gap-2 mb-4">
              <Zap className="h-4 w-4 text-status-bubble" />
              <span className="text-xs font-bold tracking-widest uppercase text-status-bubble">Needs Streaming</span>
            </div>
            <div className="space-y-2">
              {data.weak_categories.map((wc) => (
                <div key={wc.category} className="flex items-center justify-between gap-4 bg-bg-elevated rounded-md px-4 py-3">
                  <div className="flex items-center gap-3">
                    <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: CATEGORY_COLOR[wc.category] }} />
                    <span className="text-sm font-semibold text-text-primary">{wc.label}</span>
                    <span className="text-xs text-text-muted">{wc.reason}</span>
                  </div>
                  <Link
                    href={`/war-room/waiver?category=${wc.category}`}
                    className="flex items-center gap-1 text-xs text-accent-gold hover:text-amber-300 font-semibold transition-colors"
                  >
                    Find players <ArrowRight className="h-3 w-3" />
                  </Link>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Schedule Advantage */}
        <div className="bg-bg-surface border border-border-subtle rounded-lg p-6">
          <div className="flex items-center gap-2 mb-4">
            <Calendar className="h-4 w-4 text-accent-gold" />
            <span className="text-xs font-bold tracking-widest uppercase text-accent-gold">Schedule Advantage</span>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center p-4 bg-bg-elevated rounded-md">
              <p className="text-[10px] text-text-muted uppercase tracking-wider mb-1">My Team</p>
              <p className="text-2xl font-bold font-mono text-text-primary">{data.schedule_advantage.my_games}</p>
              <p className="text-xs text-text-muted">games</p>
            </div>
            <div className="text-center p-4 bg-bg-elevated rounded-md">
              <p className="text-[10px] text-text-muted uppercase tracking-wider mb-1">{data.opponent_name}</p>
              <p className="text-2xl font-bold font-mono text-text-primary">{data.schedule_advantage.opponent_games}</p>
              <p className="text-xs text-text-muted">games</p>
            </div>
          </div>
          {data.schedule_advantage.my_games !== data.schedule_advantage.opponent_games && (
            <p className={cn(
              'text-xs text-center mt-3 font-semibold',
              data.schedule_advantage.my_games > data.schedule_advantage.opponent_games
                ? 'text-status-safe'
                : 'text-status-bubble',
            )}>
              {data.schedule_advantage.my_games > data.schedule_advantage.opponent_games
                ? `+${data.schedule_advantage.my_games - data.schedule_advantage.opponent_games} game advantage`
                : `${data.schedule_advantage.my_games - data.schedule_advantage.opponent_games} game disadvantage`}
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
