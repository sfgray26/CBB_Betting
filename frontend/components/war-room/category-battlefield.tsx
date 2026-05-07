'use client'

import { cn } from '@/lib/utils'
import type { MatchupResponse, RotoCategory } from '@/lib/types'
import {
  BATTER_CATEGORIES,
  PITCHER_CATEGORIES,
  CATEGORY_LABEL,
  LOWER_IS_BETTER,
  RATIO_CATEGORIES,
} from '@/lib/types'

function formatStat(cat: RotoCategory, val: number | null | undefined): string {
  if (val === null || val === undefined) return '—'
  if (cat === 'AVG' || cat === 'OPS') return val.toFixed(3).replace(/^0/, '')
  if (cat === 'ERA' || cat === 'WHIP' || cat === 'K_9') return val.toFixed(2)
  // NSB/NSV can be negative
  return val < 0 ? String(val) : String(Math.round(val))
}

function isWinning(myVal: number | null | undefined, oppVal: number | null | undefined, lowerBetter: boolean): boolean | null {
  if (myVal == null || oppVal == null) return null
  if (myVal === oppVal) return null
  return lowerBetter ? myVal < oppVal : myVal > oppVal
}

function barMyPct(myVal: number | null | undefined, oppVal: number | null | undefined): number {
  if (myVal == null || oppVal == null) return 50
  // Don't draw bars for negative stats (NSB/NSV edge cases)
  if (myVal < 0 || oppVal < 0) return 50
  const total = myVal + oppVal
  if (total === 0) return 50
  return (myVal / total) * 100
}

interface RowProps {
  cat: RotoCategory
  myVal: number | null | undefined
  oppVal: number | null | undefined
}

function CategoryRow({ cat, myVal, oppVal }: RowProps) {
  const lowerBetter = LOWER_IS_BETTER.includes(cat)
  const isRatio = RATIO_CATEGORIES.includes(cat)
  const winning = isWinning(myVal, oppVal, lowerBetter)
  const label = CATEGORY_LABEL[cat]
  const pct = barMyPct(myVal, oppVal)

  const myTextCls =
    winning === true ? 'text-[#FFC000]' : winning === false ? 'text-[#969696]' : 'text-[#7D7D7D]'
  const oppTextCls =
    winning === false ? 'text-[#FFC000]' : 'text-[#7D7D7D]'
  const badgeText = winning === true ? 'W' : winning === false ? 'L' : '—'
  const badgeCls =
    winning === true ? 'text-[#FFC000]' : winning === false ? 'text-[#969696]' : 'text-[#494949]'

  return (
    <div className="flex items-center gap-3 py-2.5 border-b border-[#161616] last:border-0">
      {/* Category label */}
      <span className="w-9 text-xs font-mono font-semibold tracking-wide text-[#7D7D7D] uppercase">
        {label}
      </span>

      {/* My value */}
      <span className={cn('w-14 text-right text-sm tabular-nums font-mono', myTextCls)}>
        {formatStat(cat, myVal)}
      </span>

      {/* Bar (counting stats) or spacer (ratio stats) */}
      {isRatio ? (
        <div className="flex-1" />
      ) : (
        <div className="flex-1 flex h-1.5 gap-px">
          {/* My side — fills right-to-left from center */}
          <div className="flex-1 flex justify-end bg-[#111]">
            <div
              className="h-full transition-all duration-500"
              style={{
                width: `${pct}%`,
                backgroundColor: winning === true ? '#FFC000' : '#3A3A3A',
              }}
            />
          </div>
          {/* Center divider */}
          <div className="w-px bg-[#333] flex-shrink-0" />
          {/* Opp side — fills left-to-right from center */}
          <div className="flex-1 bg-[#111]">
            <div
              className="h-full transition-all duration-500"
              style={{
                width: `${100 - pct}%`,
                backgroundColor: winning === false ? '#969696' : '#2A2A2A',
              }}
            />
          </div>
        </div>
      )}

      {/* Opponent value */}
      <span className={cn('w-14 text-sm tabular-nums font-mono', oppTextCls)}>
        {formatStat(cat, oppVal)}
      </span>

      {/* Win/Loss badge */}
      <span className={cn('w-5 text-right text-xs font-bold', badgeCls)}>
        {badgeText}
      </span>
    </div>
  )
}

interface Props {
  data: MatchupResponse
}

export function CategoryBattlefield({ data }: Props) {
  const myStats = data.my_team.stats
  const oppStats = data.opponent.stats

  return (
    <div className="bg-[#202020]">
      {/* Column headers */}
      <div className="flex items-center gap-3 px-5 pt-4 pb-2 border-b border-[#2A2A2A]">
        <span className="w-9" />
        <span className="w-14 text-right text-xs font-semibold tracking-widest uppercase text-[#494949]">ME</span>
        <div className="flex-1" />
        <span className="w-14 text-xs font-semibold tracking-widest uppercase text-[#494949]">OPP</span>
        <span className="w-5" />
      </div>

      {/* Batting section */}
      <div className="px-5">
        <p className="text-xs font-semibold tracking-widest uppercase text-[#FFC000] pt-4 pb-2">
          BATTING
        </p>
        {BATTER_CATEGORIES.map((cat) => (
          <CategoryRow
            key={cat}
            cat={cat}
            myVal={myStats[cat]}
            oppVal={oppStats[cat]}
          />
        ))}
      </div>

      {/* Pitching section */}
      <div className="px-5 pb-4">
        <p className="text-xs font-semibold tracking-widest uppercase text-[#FFC000] pt-5 pb-2">
          PITCHING
        </p>
        {PITCHER_CATEGORIES.map((cat) => (
          <CategoryRow
            key={cat}
            cat={cat}
            myVal={myStats[cat]}
            oppVal={oppStats[cat]}
          />
        ))}
      </div>
    </div>
  )
}
