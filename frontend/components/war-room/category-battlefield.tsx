'use client'

import { useState, useMemo } from 'react'
import { cn } from '@/lib/utils'
import type { MatchupResponse, MatchupSimulateResponse, RotoCategory, CategoryProjection } from '@/lib/types'
import {
  BATTER_CATEGORIES,
  PITCHER_CATEGORIES,
  CATEGORY_LABEL,
  LOWER_IS_BETTER,
  RATIO_CATEGORIES,
} from '@/lib/types'
import { CategoryStatusTag } from './category-status-tag'

function formatStat(cat: RotoCategory, val: number | string | null | undefined): string {
  if (val === null || val === undefined || val === '' || val === '-') return '—'
  const n = Number(val)
  if (!isFinite(n)) return '—'
  if (cat === 'AVG' || cat === 'OPS') return n.toFixed(3).replace(/^0/, '')
  if (cat === 'ERA' || cat === 'WHIP' || cat === 'K_9') return n.toFixed(2)
  return n < 0 ? String(n) : String(Math.round(n))
}

function toNum(v: number | string | null | undefined): number | null {
  if (v === null || v === undefined || v === '' || v === '-') return null
  const n = Number(v)
  return isFinite(n) ? n : null
}

function isWinning(myVal: number | string | null | undefined, oppVal: number | string | null | undefined, lowerBetter: boolean): boolean | null {
  const a = toNum(myVal), b = toNum(oppVal)
  if (a == null || b == null) return null
  if (a === b) return null
  return lowerBetter ? a < b : a > b
}

function barMyPct(myVal: number | string | null | undefined, oppVal: number | string | null | undefined): number {
  const a = toNum(myVal), b = toNum(oppVal)
  if (a == null || b == null) return 50
  if (a < 0 || b < 0) return 50
  const total = a + b
  if (total === 0) return 50
  return (a / total) * 100
}

function actionHint(proj: CategoryProjection | undefined, lowerBetter: boolean): string {
  if (!proj) return ''
  const { win_prob, my_proj, opp_proj } = proj
  if (win_prob > 0.95) return 'Protect'
  if (win_prob < 0.05) return '—'
  if (win_prob >= 0.35 && win_prob <= 0.65) {
    // Bubble
    if (my_proj != null && opp_proj != null) {
      const delta = lowerBetter
        ? +(my_proj - opp_proj).toFixed(2)
        : +(opp_proj - my_proj).toFixed(2)
      if (delta > 0) return `Need +${delta}`
      return lowerBetter ? 'Ratio risk' : 'Hold'
    }
    return 'Close'
  }
  if (win_prob > 0.65) return 'Hold'
  return ''
}

type FilterChip = 'all' | 'bubbles' | 'hitting' | 'pitching'
type SortMode = 'flip' | 'margin' | 'type'

interface RowProps {
  cat: RotoCategory
  myVal: number | string | null | undefined
  oppVal: number | string | null | undefined
  proj: CategoryProjection | undefined
}

function CategoryRow({ cat, myVal, oppVal, proj }: RowProps) {
  const lowerBetter = LOWER_IS_BETTER.includes(cat)
  const isRatio = RATIO_CATEGORIES.includes(cat)
  const winning = isWinning(myVal, oppVal, lowerBetter)
  const label = CATEGORY_LABEL[cat]
  const pct = barMyPct(myVal, oppVal)
  const winProb = proj?.win_prob ?? null
  const hint = actionHint(proj, lowerBetter)

  const myTextCls =
    winning === true ? 'text-[#FFC000]' : winning === false ? 'text-[#969696]' : 'text-[#7D7D7D]'
  const oppTextCls =
    winning === false ? 'text-[#FFC000]' : 'text-[#7D7D7D]'

  return (
    <div className="flex items-center gap-2 py-2.5 border-b border-[#161616] last:border-0">
      {/* Category label */}
      <span className="w-9 text-xs font-mono font-semibold tracking-wide text-[#7D7D7D] uppercase flex-shrink-0">
        {label}
      </span>

      {/* My current value */}
      <span className={cn('w-12 text-right text-sm tabular-nums font-mono flex-shrink-0', myTextCls)}>
        {formatStat(cat, myVal)}
      </span>

      {/* Bar or spacer */}
      {isRatio ? (
        <div className="flex-1" />
      ) : (
        <div className="flex-1 flex h-1.5 gap-px min-w-0">
          <div className="flex-1 flex justify-end bg-[#111]">
            <div
              className="h-full transition-all duration-500"
              style={{
                width: `${pct}%`,
                backgroundColor: winning === true ? '#FFC000' : '#3A3A3A',
              }}
            />
          </div>
          <div className="w-px bg-[#333] flex-shrink-0" />
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

      {/* Opponent current value */}
      <span className={cn('w-12 text-sm tabular-nums font-mono flex-shrink-0', oppTextCls)}>
        {formatStat(cat, oppVal)}
      </span>

      {/* Projected final (mine → opp) */}
      <span className="w-16 text-right text-xs font-mono text-[#494949] flex-shrink-0 hidden sm:block">
        {proj ? `${formatStat(cat, proj.my_proj)}→${formatStat(cat, proj.opp_proj)}` : '—'}
      </span>

      {/* Status tag */}
      <div className="w-16 flex justify-end flex-shrink-0">
        <CategoryStatusTag winProb={winProb} />
      </div>

      {/* Action hint */}
      <span className="w-14 text-right text-[10px] font-mono text-[#494949] tracking-wide flex-shrink-0 hidden md:block">
        {hint}
      </span>
    </div>
  )
}

interface Props {
  data: MatchupResponse
  simulate?: MatchupSimulateResponse
}

export function CategoryBattlefield({ data, simulate }: Props) {
  const [filter, setFilter] = useState<FilterChip>('all')
  const [sort, setSort] = useState<SortMode>('flip')

  const myStats = data.my_team.stats
  const oppStats = data.opponent.stats

  const projMap = useMemo(() => {
    const m = new Map<string, CategoryProjection>()
    simulate?.category_projections.forEach(p => m.set(p.category, p))
    return m
  }, [simulate])

  const filtered = useMemo(() => {
    const allCats: { cat: RotoCategory; type: 'hitting' | 'pitching' }[] = [
      ...BATTER_CATEGORIES.map(c => ({ cat: c as RotoCategory, type: 'hitting' as const })),
      ...PITCHER_CATEGORIES.map(c => ({ cat: c as RotoCategory, type: 'pitching' as const })),
    ]
    return allCats.filter(({ cat, type }) => {
      if (filter === 'hitting') return type === 'hitting'
      if (filter === 'pitching') return type === 'pitching'
      if (filter === 'bubbles') {
        const wp = projMap.get(cat)?.win_prob ?? null
        return wp !== null && wp >= 0.35 && wp <= 0.65
      }
      return true
    })
  }, [filter, projMap])

  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      if (sort === 'type') {
        // Group by type: hitters before pitchers
        if (a.type !== b.type) return a.type === 'hitting' ? -1 : 1
        return 0
      }
      if (sort === 'flip') {
        const wpA = projMap.get(a.cat)?.win_prob ?? 0.5
        const wpB = projMap.get(b.cat)?.win_prob ?? 0.5
        // Closest to 0.5 = most flippable
        return Math.abs(wpA - 0.5) - Math.abs(wpB - 0.5)
      }
      if (sort === 'margin') {
        const pA = projMap.get(a.cat)
        const pB = projMap.get(b.cat)
        // Use projected margin if available, otherwise fall back to current differential
        if (pA && pB && pA.my_proj != null && pA.opp_proj != null && pB.my_proj != null && pB.opp_proj != null) {
          const mA = Math.abs(pA.my_proj - pA.opp_proj)
          const mB = Math.abs(pB.my_proj - pB.opp_proj)
          return mA - mB
        }
        // Fallback to current stat differential
        const myA = toNum(myStats[a.cat])
        const oppA = toNum(oppStats[a.cat])
        const myB = toNum(myStats[b.cat])
        const oppB = toNum(oppStats[b.cat])
        const diffA = (myA ?? 0) - (oppA ?? 0)
        const diffB = (myB ?? 0) - (oppB ?? 0)
        return Math.abs(diffA) - Math.abs(diffB)
      }
      return 0
    })
  }, [filtered, sort, projMap, myStats, oppStats])

  const hitters = sorted.filter(x => x.type === 'hitting')
  const pitchers = sorted.filter(x => x.type === 'pitching')

  return (
    <div className="bg-[#202020]">
      {/* Controls */}
      <div className="flex items-center justify-between gap-3 px-5 pt-4 pb-3 border-b border-[#2A2A2A]">
        {/* Filter chips */}
        <div className="flex gap-1">
          {(['all', 'bubbles', 'hitting', 'pitching'] as FilterChip[]).map(chip => (
            <button
              key={chip}
              onClick={() => setFilter(chip)}
              className={cn(
                'px-2.5 py-1 text-[10px] font-semibold tracking-widest uppercase transition-colors',
                filter === chip
                  ? 'bg-[#FFC000] text-black'
                  : 'text-[#494949] hover:text-[#7D7D7D]',
              )}
            >
              {chip === 'all' ? 'ALL' : chip === 'bubbles' ? 'BUBBLES' : chip === 'hitting' ? 'HITTING' : 'PITCHING'}
            </button>
          ))}
        </div>

        {/* Sort */}
        <select
          value={sort}
          onChange={e => setSort(e.target.value as SortMode)}
          className="bg-transparent text-[#494949] text-[10px] tracking-widest uppercase border border-[#2A2A2A] px-2 py-1 focus:outline-none focus:border-[#494949]"
        >
          <option value="flip">Sort: flip prob</option>
          <option value="margin">Sort: margin</option>
          <option value="type">Sort: type</option>
        </select>
      </div>

      {/* Column headers */}
      <div className="flex items-center gap-2 px-5 pt-3 pb-2">
        <span className="w-9" />
        <span className="w-12 text-right text-[10px] font-semibold tracking-widest uppercase text-[#494949]">ME</span>
        <div className="flex-1" />
        <span className="w-12 text-[10px] font-semibold tracking-widest uppercase text-[#494949]">OPP</span>
        <span className="w-16 text-right text-[10px] font-semibold tracking-widest uppercase text-[#494949] hidden sm:block">PROJ</span>
        <span className="w-16 text-right text-[10px] font-semibold tracking-widest uppercase text-[#494949]">STATUS</span>
        <span className="w-14 text-right text-[10px] font-semibold tracking-widest uppercase text-[#494949] hidden md:block">ACTION</span>
      </div>

      {filter === 'bubbles' ? (
        // Flat list for bubbles (no section headers)
        <div className="px-5 pb-4">
          {sorted.length === 0 ? (
            <p className="text-xs text-[#494949] py-4 text-center">
              {simulate ? 'No bubble categories — projections are decisive.' : 'Run simulate to see bubble categories.'}
            </p>
          ) : (
            sorted.map(({ cat }) => (
              <CategoryRow
                key={cat}
                cat={cat}
                myVal={myStats[cat]}
                oppVal={oppStats[cat]}
                proj={projMap.get(cat)}
              />
            ))
          )}
        </div>
      ) : (
        // Grouped by hitting / pitching
        <>
          {(filter === 'all' || filter === 'hitting') && hitters.length > 0 && (
            <div className="px-5">
              <p className="text-xs font-semibold tracking-widest uppercase text-[#FFC000] pt-3 pb-2">
                BATTING
              </p>
              {hitters.map(({ cat }) => (
                <CategoryRow
                  key={cat}
                  cat={cat}
                  myVal={myStats[cat]}
                  oppVal={oppStats[cat]}
                  proj={projMap.get(cat)}
                />
              ))}
            </div>
          )}
          {(filter === 'all' || filter === 'pitching') && pitchers.length > 0 && (
            <div className="px-5 pb-4">
              <p className="text-xs font-semibold tracking-widest uppercase text-[#FFC000] pt-5 pb-2">
                PITCHING
              </p>
              {pitchers.map(({ cat }) => (
                <CategoryRow
                  key={cat}
                  cat={cat}
                  myVal={myStats[cat]}
                  oppVal={oppStats[cat]}
                  proj={projMap.get(cat)}
                />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}
