/**
 * Shared constants and helpers used across fantasy baseball pages.
 * Centralised here to prevent drift between matchup, lineup, waiver, and API client.
 */

import fantasyStatContract from './fantasy-stat-contract.json'

/**
 * Return today's date as YYYY-MM-DD anchored to US Eastern Time.
 * Uses en-CA locale which produces the ISO-style format natively.
 * Always use this instead of new Date().toISOString().slice(0,10) which is UTC.
 */
export function etTodayStr(): string {
  return new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' })
}

export const STAT_LABELS: Record<string, string> = fantasyStatContract.statLabels

// Ratio stats get 3 decimal places; counting stats get integer display.
export const RATIO_STATS = new Set([
  'AVG', 'OBP', 'OPS', 'ERA', 'WHIP', 'K9', 'K/9', '3', '26', '27', '55',
])

export const LOWER_IS_BETTER = new Set(fantasyStatContract.lowerIsBetter)

export const MATCHUP_DISPLAY_ONLY = new Set(fantasyStatContract.matchupDisplayOnly)

export const MATCHUP_STAT_ORDER: string[] = fantasyStatContract.matchupStatOrder
