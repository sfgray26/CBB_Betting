'use client'

import { useState, useMemo } from 'react'
import { cn } from '@/lib/utils'
import type { MatchupResponse, MatchupSimulateResponse, RotoCategory, CategoryProjection } from '@/lib/types'
import {
  BATTER_CATEGORIES,
  PITCHER_CATEGORIES,
  CATEGORY_LABEL,
  CATEGORY_COLOR,
  LOWER_IS_BETTER,
  RATIO_CATEGORIES,
} from '@/lib/types'

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

function statusLabel(winProb: number | null): { text: string; color: string; description: string } {
  if (winProb === null) return { text: '—', color: 'text-text-muted', description: 'No simulation data' }
  if (winProb > 0.85) return { text: 'SAFE', color: 'text-status-safe', description: `${Math.round(winProb * 100)}% win prob - Safe lead` }
  if (winProb > 0.65) return { text: 'LEAD', color: 'text-status-lead', description: `${Math.round(winProb * 100)}% win prob - Leaning ahead` }
  if (winProb >= 0.35 && winProb <= 0.65) return { text: 'BUBBLE', color: 'text-status-bubble', description: `${Math.round(winProb * 100)}% win prob - Could go either way` }
  if (winProb >= 0.15) return { text: 'BEHIND', color: 'text-status-behind', description: `${Math.round(winProb * 100)}% win prob - Leaning behind` }
  return { text: 'LOST', color: 'text-status-lost', description: `${Math.round(winProb * 100)}% win prob - Unlikely to win` }
}

function actionHint(proj: CategoryProjection | undefined, lowerBetter: boolean): string {
  if (!proj) return ''
  const { win_prob, my_proj, opp_proj } = proj
  if (win_prob > 0.95) return 'Protect'
  if (win_prob < 0.05) return '—'
  if (win_prob >= 0.35 && win_prob <= 0.65) {
    if (my_proj != null && opp_proj != null) {
      const delta = lowerBetter
        ? +(my_proj - opp_proj).toFixed(2)
        : +(opp_proj - my_proj).toFixed(2)
      if (delta > 0) return lowerBetter ? `Need -${delta}` : `Need +${delta}`
      return lowerBetter ? 'Ratio risk' : 'Hold'
    }
    return 'Close'
  }
  if (win_prob > 0.65) return 'Hold'
  return 'Punt?'
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
  const catColor = CATEGORY_COLOR[cat]
  const pct = barMyPct(myVal, oppVal)
  const winProb = proj?.win_prob ?? null
  const hint = actionHint(proj, lowerBetter)
  const status = statusLabel(winProb)

  // Design System v2: my value is always primary white; gold is reserved for CTAs
  const myTextCls = 'text-text-primary'
  const oppTextCls = 'text-text-secondary'

  const projDisplay = proj
    ? isRatio
      ? `${Math.round(proj.win_prob * 100)}%`
      : `${formatStat(cat, proj.my_proj)}→${formatStat(cat, proj.opp_proj)}`
    : '—'

  // Bar colors: my side uses category color when winning, neutral gray when losing
  const myBarColor = winning === true ? `${catColor}cc` : '#3a3a4d'
  const oppBarColor = winning === false ? '#969696' : '#2a2a3d'

  return (
    <div className="border-b border-border-subtle last:border-0 hover:bg-bg-elevated transition-colors duration-150 px-2 rounded-sm">
      {/* Main row */}
      <div className="flex items-center gap-3 py-3">
        {/* Category label with identity dot */}
        <span className="w-12 text-sm font-semibold tracking-wide text-text-secondary uppercase flex-shrink-0 flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: catColor }} />
          {label}
        </span>

        {/* My current value */}
        <span className={cn('w-14 text-right text-base tabular-nums font-mono flex-shrink-0', myTextCls)}>
          {formatStat(cat, myVal)}
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
                  width: `${pct}%`,
                  backgroundColor: myBarColor,
                }}
              />
            </div>
            <div className="w-px bg-border-subtle flex-shrink-0" />
            <div className="flex-1 bg-bg-inset rounded-r-sm overflow-hidden">
              <div
                className="h-full transition-all duration-500"
                style={{
                  width: `${100 - pct}%`,
                  backgroundColor: oppBarColor,
                }}
              />
            </div>
          </div>
        )}

        {/* Opponent current value */}
        <span className={cn('w-14 text-base tabular-nums font-mono flex-shrink-0', oppTextCls)}>
          {formatStat(cat, oppVal)}
        </span>

        {/* Projected final — hidden on mobile, shown sm+ */}
        <span className="w-20 text-right text-sm font-mono text-text-muted flex-shrink-0 hidden sm:block">
          {projDisplay}
        </span>

        {/* Status tag */}
        <div className="w-16 flex justify-end flex-shrink-0">
          <span
            className={cn('text-xs font-bold tracking-wider uppercase', status.color)}
            title={status.description}
          >
            {status.text}
          </span>
        </div>

        {/* Action hint — hidden on mobile, shown md+ */}
        <span className="w-20 text-right text-xs font-mono text-text-tertiary tracking-wide flex-shrink-0 hidden md:block">
          {hint}
        </span>
      </div>

      {/* Mobile-only second row: proj + action */}
      <div className="flex items-center justify-between pb-2 sm:hidden text-[10px] font-mono text-text-muted">
        <span>{projDisplay}</span>
        {hint && hint !== '—' && <span>{hint}</span>}
      </div>
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
    return allCats.filter(({ cat }) => {
      if (filter === 'hitting') return BATTER_CATEGORIES.includes(cat as typeof BATTER_CATEGORIES[number])
      if (filter === 'pitching') return PITCHER_CATEGORIES.includes(cat as typeof PITCHER_CATEGORIES[number])
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
        if (a.type !== b.type) return a.type === 'hitting' ? -1 : 1
        return 0
      }
      if (sort === 'flip') {
        const wpA = projMap.get(a.cat)?.win_prob ?? 0.5
        const wpB = projMap.get(b.cat)?.win_prob ?? 0.5
        return Math.abs(wpA - 0.5) - Math.abs(wpB - 0.5)
      }
      if (sort === 'margin') {
        const pA = projMap.get(a.cat)
        const pB = projMap.get(b.cat)
        if (pA && pB && pA.my_proj != null && pA.opp_proj != null && pB.my_proj != null && pB.opp_proj != null) {
          const mA = Math.abs(pA.my_proj - pA.opp_proj)
          const mB = Math.abs(pB.my_proj - pB.opp_proj)
          return mA - mB
        }
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
    <div className="bg-bg-surface border border-border-subtle rounded-lg">
      {/* Controls */}
      <div className="flex items-center justify-between gap-4 px-6 pt-5 pb-4 border-b border-border-subtle">
        {/* Bubble ratings legend */}
        <div className="flex items-center gap-3 text-xs">
          <span className="text-text-muted font-semibold tracking-widest uppercase">Win Prob:</span>
          <span className="text-status-safe">SAFE &gt;85%</span>
          <span className="text-status-lead">LEAD 65-85%</span>
          <span className="text-status-bubble">BUBBLE 35-65%</span>
          <span className="text-status-behind">BEHIND 15-35%</span>
          <span className="text-status-lost">LOST &lt;15%</span>
        </div>
        <div className="flex gap-2">
          {(['all', 'bubbles', 'hitting', 'pitching'] as FilterChip[]).map(chip => (
            <button
              key={chip}
              onClick={() => setFilter(chip)}
              className={cn(
                'px-4 py-2 text-xs font-semibold tracking-widest uppercase transition-colors rounded-sm',
                filter === chip
                  ? 'bg-accent-gold text-black'
                  : 'text-text-muted hover:text-text-secondary bg-bg-elevated',
              )}
            >
              {chip === 'all' ? 'ALL' : chip === 'bubbles' ? 'BUBBLES' : chip === 'hitting' ? 'HITTING' : 'PITCHING'}
            </button>
          ))}
        </div>

        <select
          value={sort}
          onChange={e => setSort(e.target.value as SortMode)}
          className="bg-bg-elevated text-text-secondary text-xs tracking-widest uppercase border border-border-subtle px-3 py-2 focus:outline-none focus:border-border-focus rounded-sm"
        >
          <option value="flip">Sort: Flip Probability</option>
          <option value="margin">Sort: Margin</option>
          <option value="type">Sort: Type</option>
        </select>
      </div>

      {/* Column headers */}
      <div className="flex items-center gap-3 px-6 pt-4 pb-2">
        <span className="w-12" />
        <span className="w-14 text-right text-xs font-semibold tracking-widest uppercase text-text-muted">ME</span>
        <div className="flex-1" />
        <span className="w-14 text-xs font-semibold tracking-widest uppercase text-text-muted">OPP</span>
        <span className="w-20 text-right text-xs font-semibold tracking-widest uppercase text-text-muted hidden sm:block">PROJ / WIN%</span>
        <span className="w-16 text-right text-xs font-semibold tracking-widest uppercase text-text-muted">STATUS</span>
        <span className="w-20 text-right text-xs font-semibold tracking-widest uppercase text-text-muted hidden md:block">ACTION</span>
      </div>

      {filter === 'bubbles' ? (
        <div className="px-6 pb-5">
          {sorted.length === 0 ? (
            <p className="text-sm text-text-muted py-6 text-center">
              {simulate ? 'No bubble categories — projections are decisive.' : 'Run simulation to see bubble categories.'}
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
        <>
          {(filter === 'all' || filter === 'hitting') && hitters.length > 0 && (
            <div className="px-6">
              <p className="text-sm font-semibold tracking-widest uppercase text-accent-gold pt-4 pb-2">
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
            <div className="px-6 pb-5">
              <p className="text-sm font-semibold tracking-widest uppercase text-accent-gold pt-5 pb-2">
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
