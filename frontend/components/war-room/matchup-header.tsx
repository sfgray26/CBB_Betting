'use client'

import { cn } from '@/lib/utils'
import type { MatchupResponse, RotoCategory } from '@/lib/types'
import { BATTER_CATEGORIES, PITCHER_CATEGORIES, LOWER_IS_BETTER } from '@/lib/types'

const ALL_SCORING: RotoCategory[] = [...BATTER_CATEGORIES, ...PITCHER_CATEGORIES]

function computeScore(data: MatchupResponse) {
  let myWins = 0
  let oppWins = 0
  for (const cat of ALL_SCORING) {
    const myVal = data.my_team.stats[cat] ?? null
    const oppVal = data.opponent.stats[cat] ?? null
    if (myVal === null || oppVal === null || myVal === oppVal) continue
    const lowerBetter = LOWER_IS_BETTER.includes(cat)
    if (lowerBetter ? myVal < oppVal : myVal > oppVal) myWins++
    else oppWins++
  }
  return { myWins, oppWins }
}

interface Props {
  data: MatchupResponse
}

export function MatchupHeader({ data }: Props) {
  const { myWins, oppWins } = computeScore(data)
  const leading = myWins > oppWins

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

      {data.message && (
        <p className="mt-3 text-xs text-[#494949] border-t border-[#2A2A2A] pt-3">{data.message}</p>
      )}
    </div>
  )
}
