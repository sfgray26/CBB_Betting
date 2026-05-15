'use client'

import { cn } from '@/lib/utils'
import type { MatchupResponse, MatchupSimulateResponse, RotoCategory } from '@/lib/types'
import { BATTER_CATEGORIES, PITCHER_CATEGORIES, LOWER_IS_BETTER } from '@/lib/types'

const ALL_SCORING: RotoCategory[] = [...BATTER_CATEGORIES, ...PITCHER_CATEGORIES]

function toNum(v: number | string | null | undefined): number | null {
  if (v === null || v === undefined || v === '' || v === '-') return null
  const n = Number(v)
  return isFinite(n) ? n : null
}

function computeScore(data: MatchupResponse) {
  let myWins = 0
  let oppWins = 0
  for (const cat of ALL_SCORING) {
    const a = toNum(data.my_team.stats[cat])
    const b = toNum(data.opponent.stats[cat])
    if (a === null || b === null || a === b) continue
    const lowerBetter = LOWER_IS_BETTER.includes(cat)
    if (lowerBetter ? a < b : a > b) myWins++
    else oppWins++
  }
  return { myWins, oppWins }
}

function computeProjectedScore(simulate: MatchupSimulateResponse) {
  let projWins = 0
  let projLosses = 0
  for (const proj of simulate.category_projections) {
    if (proj.win_prob > 0.5) projWins++
    else if (proj.win_prob < 0.5) projLosses++
  }
  return { projWins, projLosses }
}

interface Props {
  data: MatchupResponse
  simulate?: MatchupSimulateResponse
}

export function MatchupHeader({ data, simulate }: Props) {
  const { myWins, oppWins } = computeScore(data)
  const leading = myWins > oppWins
  const projScore = simulate ? computeProjectedScore(simulate) : null
  const winPct = simulate ? Math.round(simulate.win_prob * 100) : null

  return (
    <div className="bg-bg-surface border border-border-subtle rounded-lg p-6 lg:p-8">
      {/* Season label */}
      <p className="text-sm font-semibold tracking-widest uppercase text-text-secondary mb-6">
        {data.is_playoffs ? 'PLAYOFFS' : 'REGULAR SEASON'}{data.week ? ` — WEEK ${data.week}` : ''}
      </p>

      {/* Teams and score */}
      <div className="flex items-center gap-6">
        {/* My team */}
        <div className="flex-1 min-w-0">
          <p className="text-lg lg:text-xl font-bold tracking-wide text-text-primary uppercase truncate">
            {data.my_team.team_name || 'MY TEAM'}
          </p>
          <p className="text-sm text-text-secondary mt-1 uppercase tracking-wide">My Team</p>
        </div>

        {/* Score */}
        <div className="flex-shrink-0 text-center px-4">
          <div className="flex items-baseline gap-3">
            <span className={cn('text-5xl lg:text-6xl font-bold tabular-nums', leading ? 'text-accent-gold' : 'text-text-secondary')}>
              {myWins}
            </span>
            <span className="text-text-muted text-3xl font-light">-</span>
            <span className={cn('text-5xl lg:text-6xl font-bold tabular-nums', !leading ? 'text-accent-gold' : 'text-text-muted')}>
              {oppWins}
            </span>
          </div>
          <p className={cn(
            'text-sm mt-2 uppercase tracking-wider font-semibold',
            myWins === oppWins ? 'text-text-secondary' : leading ? 'text-accent-gold' : 'text-text-secondary'
          )}>
            {myWins === oppWins ? 'TIED' : leading ? 'LEADING' : 'TRAILING'}
          </p>
        </div>

        {/* Opponent */}
        <div className="flex-1 min-w-0 text-right">
          <p className="text-lg lg:text-xl font-bold tracking-wide text-text-secondary uppercase truncate">
            {data.opponent.team_name || 'OPPONENT'}
          </p>
          <p className="text-sm text-text-secondary mt-1 uppercase tracking-wide">Opponent</p>
        </div>
      </div>

      {/* Projected record strip */}
      {projScore && winPct !== null && (
        <div className="mt-6 pt-4 border-t border-border-subtle flex items-center justify-between">
          <div className="flex items-center gap-4">
            <span className="text-xs font-semibold tracking-widest uppercase text-text-muted">Projected</span>
            <span className="text-base font-bold tabular-nums text-accent-gold">{projScore.projWins}</span>
            <span className="text-text-muted text-lg">-</span>
            <span className="text-base font-bold tabular-nums text-text-secondary">{projScore.projLosses}</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs font-semibold tracking-widest uppercase text-text-muted">Win Probability</span>
            <span className={cn(
              'text-xl font-bold tabular-nums',
              winPct >= 60 ? 'text-accent-gold' : winPct <= 40 ? 'text-text-secondary' : 'text-text-secondary',
            )}>
              {winPct}%
            </span>
          </div>
        </div>
      )}

      {data.message && (
        <p className="mt-4 text-sm text-text-muted border-t border-border-subtle pt-4">{data.message}</p>
      )}
    </div>
  )
}
