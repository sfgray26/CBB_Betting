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
    <div className="bg-[#202020] p-5">
      <p className="text-xs font-semibold tracking-widest uppercase text-[#7D7D7D] mb-4">
        {data.is_playoffs ? 'PLAYOFFS' : 'REGULAR SEASON'}{data.week ? ` — WEEK ${data.week}` : ''}
      </p>

      <div className="flex items-center gap-4">
        {/* My team */}
        <div className="flex-1 min-w-0">
          <p className="text-base font-bold tracking-wide text-white uppercase truncate">
            {data.my_team.team_name || 'MY TEAM'}
          </p>
          <p className="text-xs text-[#7D7D7D] mt-0.5 uppercase tracking-wide">My Team</p>
        </div>

        {/* Score */}
        <div className="flex-shrink-0 text-center">
          <div className="flex items-baseline gap-2">
            <span className={cn('text-4xl font-bold tabular-nums', leading ? 'text-[#FFC000]' : 'text-[#969696]')}>
              {myWins}
            </span>
            <span className="text-[#494949] text-2xl font-light">-</span>
            <span className={cn('text-4xl font-bold tabular-nums', !leading ? 'text-[#FFC000]' : 'text-[#494949]')}>
              {oppWins}
            </span>
          </div>
          <p className="text-xs text-[#494949] mt-1 uppercase tracking-wider">
            {myWins === oppWins ? 'TIED' : leading ? 'LEADING' : 'TRAILING'}
          </p>
        </div>

        {/* Opponent */}
        <div className="flex-1 min-w-0 text-right">
          <p className="text-base font-bold tracking-wide text-[#969696] uppercase truncate">
            {data.opponent.team_name || 'OPPONENT'}
          </p>
          <p className="text-xs text-[#7D7D7D] mt-0.5 uppercase tracking-wide">Opponent</p>
        </div>
      </div>

      {/* Projected record strip — only when simulate data is available */}
      {projScore && winPct !== null && (
        <div className="mt-4 pt-3 border-t border-[#2A2A2A] flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-[10px] font-semibold tracking-widest uppercase text-[#494949]">PROJ</span>
            <span className="text-sm font-bold tabular-nums text-[#FFC000]">{projScore.projWins}</span>
            <span className="text-[#494949]">-</span>
            <span className="text-sm font-bold tabular-nums text-[#969696]">{projScore.projLosses}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-semibold tracking-widest uppercase text-[#494949]">WIN%</span>
            <span className={cn(
              'text-sm font-bold tabular-nums',
              winPct >= 60 ? 'text-[#FFC000]' : winPct <= 40 ? 'text-[#969696]' : 'text-[#7D7D7D]',
            )}>
              {winPct}%
            </span>
          </div>
        </div>
      )}

      {data.message && (
        <p className="mt-3 text-xs text-[#494949] border-t border-[#2A2A2A] pt-3">{data.message}</p>
      )}
    </div>
  )
}
