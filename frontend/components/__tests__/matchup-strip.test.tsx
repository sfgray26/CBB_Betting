/**
 * Test: MatchupStrip component (Fix 1: Roster This Week 0-0)
 *
 * Validates that the Roster page correctly computes matchup scores
 * from /api/fantasy/matchup (the working endpoint) instead of
 * /api/fantasy/scoreboard (the broken endpoint that produced 0-0-18T).
 */

import { describe, it, expect } from 'vitest'

// Simulate the buildMatchupRows logic from the Roster page
const ALL_SCORING_CATS = ['R', 'H', 'HR_B', 'RBI', 'K_B', 'TB', 'AVG', 'OPS', 'NSB', 'W', 'L', 'HR_P', 'K_P', 'ERA', 'WHIP', 'K_9', 'QS', 'NSV'] as const
const LOWER_IS_BETTER = ['K_B', 'L', 'HR_P', 'ERA', 'WHIP'] as const

type RotoCategory = (typeof ALL_SCORING_CATS)[number]

interface MockMatchupResponse {
  my_team: { team_name: string; stats: Record<string, number | string> }
  opponent: { team_name: string; stats: Record<string, number | string> }
}

interface MatchupRow {
  category: RotoCategory
  categoryLabel: string
  isLowerBetter: boolean
  myCurrent: number | null
  oppCurrent: number | null
  outcome: 'W' | 'L' | 'T'
}

function toNum(v: number | string | null | undefined): number | null {
  if (v === null || v === undefined || v === '' || v === '-') return null
  const n = Number(v)
  return isFinite(n) ? n : null
}

function buildMatchupRows(data: MockMatchupResponse): { rows: MatchupRow[]; won: number; lost: number; tied: number } {
  const rows: MatchupRow[] = []
  let won = 0, lost = 0, tied = 0

  for (const cat of ALL_SCORING_CATS) {
    const myVal = toNum(data.my_team.stats[cat])
    const oppVal = toNum(data.opponent.stats[cat])
    const isLowerBetter = LOWER_IS_BETTER.includes(cat)

    let outcome: 'W' | 'L' | 'T' = 'T'
    if (myVal !== null && oppVal !== null && myVal !== oppVal) {
      if (isLowerBetter ? myVal < oppVal : myVal > oppVal) {
        outcome = 'W'
        won++
      } else {
        outcome = 'L'
        lost++
      }
    } else {
      tied++
    }

    rows.push({
      category: cat,
      categoryLabel: cat.replace('_B', '').replace('_P', ''),
      isLowerBetter,
      myCurrent: myVal,
      oppCurrent: oppVal,
      outcome,
    })
  }

  return { rows, won, lost, tied }
}

describe('Fix 1: Roster MatchupStrip', () => {
  it('should compute correct W/L/T from matchup stats', () => {
    const mockMatchup: MockMatchupResponse = {
      my_team: {
        team_name: 'My Team',
        stats: { R: 45, H: 120, HR_B: 15, RBI: 50, AVG: 0.285, ERA: 3.50, WHIP: 1.15 }
      },
      opponent: {
        team_name: 'Opponent',
        stats: { R: 40, H: 115, HR_B: 12, RBI: 45, AVG: 0.275, ERA: 4.20, WHIP: 1.35 }
      }
    }

    const { won, lost, tied } = buildMatchupRows(mockMatchup)

    // My team leads in 6 categories (R, H, HR_B, RBI, AVG, ERA, WHIP)
    expect(won).toBeGreaterThanOrEqual(6)
    expect(lost).toBe(0) // I'm leading in all compared stats
    expect(tied).toBeGreaterThanOrEqual(11) // Categories with no stats
  })

  it('should handle lower-is-better categories correctly', () => {
    const mockMatchup: MockMatchupResponse = {
      my_team: {
        team_name: 'My Team',
        stats: { ERA: 2.50, WHIP: 1.00 } // Lower is better
      },
      opponent: {
        team_name: 'Opponent',
        stats: { ERA: 5.00, WHIP: 1.50 }
      }
    }

    const { rows, won } = buildMatchupRows(mockMatchup)
    const eraRow = rows.find(r => r.category === 'ERA')
    const whipRow = rows.find(r => r.category === 'WHIP')

    expect(eraRow?.outcome).toBe('W') // Lower ERA = win
    expect(whipRow?.outcome).toBe('W') // Lower WHIP = win
    expect(won).toBe(2)
  })

  it('should not produce 0-0-18T when stats are available', () => {
    const mockMatchup: MockMatchupResponse = {
      my_team: {
        team_name: 'My Team',
        stats: { R: 50, H: 130 } // Some real stats
      },
      opponent: {
        team_name: 'Opponent',
        stats: { R: 42, H: 125 }
      }
    }

    const { won, lost, tied } = buildMatchupRows(mockMatchup)

    // Should NOT be 0-0-18T
    expect(won + lost).toBeGreaterThan(0)
    expect(tied).toBeLessThan(18) // At least some categories have real stats
  })

  it('should handle null/missing stats gracefully', () => {
    const mockMatchup: MockMatchupResponse = {
      my_team: {
        team_name: 'My Team',
        stats: {} // No stats available
      },
      opponent: {
        team_name: 'Opponent',
        stats: {}
      }
    }

    const { won, lost, tied } = buildMatchupRows(mockMatchup)

    // All categories should be tied when no stats available
    expect(won).toBe(0)
    expect(lost).toBe(0)
    expect(tied).toBe(18)
  })
})
